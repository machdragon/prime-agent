"""Goal graph: one decomposable graph of work, executed from the kernel.

A goal and a single deterministic step are the same node type, so a goal can be
split without limit and the split is an ordinary return value rather than a
planning phase. The graph is plain Python objects backed by a JSON file, and the
default node body runs inline in this kernel, so adding a node costs
approximately nothing and only nodes that genuinely need judgement cost a model.

    from goal_graph import Graph, Node, Expand, Done, Reject, register

    @register
    def split(ctx):
        return Expand([Node(intent=f"check {name}", fn="check", args={"name": name})
                       for name in ctx.args["names"]])

    @register
    def check(ctx):
        return Done({"name": ctx.args["name"], "ok": True})

    g = Graph.open("release")
    g.add(Node(intent="check everything", fn="split", args={"names": ["a", "b"]}))
    report = await g.run()

A run ends when the frontier is empty, not when a continuation counter expires.
`max_supersteps` and `max_nodes` are safety limits, not completion conditions.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic as _monotonic
from time import time as _now
from typing import Any, Iterable, Mapping

from .dispatch import (
    CHILD_COMPLETED,
    CHILD_ERROR,
    CHILD_RUNNING,
    DispatchHandle,
    Dispatcher,
    RlmDispatcher,
    build_child_prompt,
    parse_result,
    read_result,
    result_path,
)
from .model import (
    NODE_STATES,
    Context,
    Done,
    Expand,
    InlineBody,
    Node,
    Outcome,
    Reject,
    assert_jsonable,
    assert_storable,
    register,
    registered,
    resolve,
)
from .store import STORE_VERSION, GraphStore, default_store_dir, graph_path, results_dir

__all__ = [
    "Context",
    "DispatchHandle",
    "Dispatcher",
    "Done",
    "Expand",
    "Graph",
    "GraphStore",
    "InFlight",
    "InlineBody",
    "Node",
    "Outcome",
    "Reject",
    "RlmDispatcher",
    "RunReport",
    "default_store_dir",
    "graph_path",
    "register",
    "registered",
    "resolve",
]

DEFAULT_MAX_SUPERSTEPS = 10_000
DEFAULT_MAX_NODES = 10_000
DEFAULT_MAX_IN_FLIGHT = 4
DEFAULT_POLL_SECONDS = 2.0

#: Why `run()` returned. Only "complete" means the graph has nothing left to do.
STOP_REASONS: tuple[str, ...] = (
    "complete",
    "pending_dispatch",
    "in_flight",
    "detached",
    "blocked",
    "max_supersteps",
    "max_nodes",
)


@dataclass(frozen=True)
class InFlight:
    """One dispatched child, with enough to tell working from wedged.

    A bare list of node ids cannot distinguish a child that is making progress
    from one that has stopped responding, which is the thing you most need to
    know while waiting.
    """

    node_id: str
    intent: str
    model: str | None
    child_id: str | None
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "intent": self.intent,
            "model": self.model,
            "child_id": self.child_id,
            "seconds": round(self.seconds, 1),
        }


@dataclass(frozen=True)
class RunReport:
    stopped: str
    supersteps: int
    executed: int
    counts: Mapping[str, int]
    pending_dispatch: tuple[str, ...] = ()
    in_flight: tuple[InFlight, ...] = ()
    blocked: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.stopped == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped": self.stopped,
            "supersteps": self.supersteps,
            "executed": self.executed,
            "counts": dict(self.counts),
            "pending_dispatch": list(self.pending_dispatch),
            "in_flight": [entry.to_dict() for entry in self.in_flight],
            "blocked": list(self.blocked),
        }


class Graph:
    """A goal graph and the loop that runs it."""

    def __init__(
        self,
        name: str,
        store: GraphStore,
        nodes: Iterable[Node] = (),
        *,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self.name = name
        self._store = store
        self._dispatcher = dispatcher
        self._nodes: dict[str, Node] = {}
        for node in nodes:
            self._nodes[node.id] = node

    @classmethod
    def open(cls, name: str, *, store_dir: str | None = None, dispatcher: Dispatcher | None = None) -> Graph:
        """Load the named graph, or start an empty one under the same name.

        Without a dispatcher, model nodes are reported as pending rather than
        run, which is the whole behaviour when no child runtime is available.
        """
        store = GraphStore(graph_path(name, store_dir))
        payload = store.read()
        if payload is None:
            return cls(name, store, dispatcher=dispatcher)
        nodes = [Node.from_dict(record) for record in payload.get("nodes", ())]
        graph = cls(payload.get("name", name), store, nodes, dispatcher=dispatcher)
        graph._assert_consistent()
        return graph

    @property
    def path(self) -> str:
        return str(self._store.path)

    @property
    def results_dir(self) -> Path:
        return results_dir(self._store.path)

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        counts = self.counts()
        rendered = ", ".join(f"{state}={counts[state]}" for state in NODE_STATES if counts[state])
        return f"<Graph {self.name!r} nodes={len(self._nodes)} {rendered or 'empty'}>"

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"graph {self.name!r} has no node {node_id!r}") from None

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def children_of(self, node_id: str) -> list[Node]:
        """Children are derived from `parents`, never stored, so they cannot drift."""
        return [node for node in self._nodes.values() if node_id in node.parents]

    def counts(self) -> dict[str, int]:
        counts = {state: 0 for state in NODE_STATES}
        for node in self._nodes.values():
            counts[node.state] += 1
        return counts

    def add(self, node: Node) -> Node:
        """Add one node. Its `needs` must already exist and must not form a cycle."""
        assert_storable(node)
        self._check_additions([node])
        self._nodes[node.id] = node
        return node

    def frontier(self) -> list[Node]:
        """Open nodes whose entire `needs` barrier has been released."""
        return [node for node in self._nodes.values() if node.state == "open" and self._deps_done(node)]

    def blocked(self) -> list[Node]:
        """Open nodes that can never run because something upstream failed.

        Transitive, not direct: a node two hops downstream of a failure is just
        as unrunnable, and a report that stops with "blocked" while listing
        nothing is worse than no report.
        """
        doomed = {
            node_id for node_id, node in self._nodes.items() if node.state in ("failed", "rejected")
        }
        while True:
            newly = {
                node_id
                for node_id, node in self._nodes.items()
                if node.state == "open"
                and node_id not in doomed
                and any(need in doomed for need in node.needs)
            }
            if not newly:
                break
            doomed |= newly
        return [node for node_id, node in self._nodes.items() if node_id in doomed and node.state == "open"]

    async def run(
        self,
        *,
        max_supersteps: int = DEFAULT_MAX_SUPERSTEPS,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        wait_seconds: float = 0.0,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        save: bool = True,
    ) -> RunReport:
        """Run the frontier until nothing is runnable.

        Each superstep joins whatever children have finished, executes every
        runnable inline and collector node, dispatches model nodes up to
        `max_in_flight`, then checkpoints.

        With `wait_seconds=0` the run returns `"in_flight"` as soon as it is
        waiting only on children, so the turn ends and the children keep going.
        Call `run()` again when one messages back, or pass `wait_seconds` to
        block here instead. Limits are safety rails: reaching one is not
        completion.
        """
        if max_supersteps < 1:
            raise ValueError("max_supersteps must be at least 1")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

        supersteps = 0
        executed = 0
        stopped = "complete"
        deadline = _monotonic() + max(0.0, wait_seconds)

        while True:
            # Checked before any work, including a join, so max_supersteps bounds
            # every path through the loop and a defect reports instead of hanging.
            if supersteps >= max_supersteps:
                stopped = "max_supersteps"
                break
            joined = await self._ingest()
            frontier = self.frontier()
            inline = [node for node in frontier if node.kind != "model"]
            dispatchable = self._dispatchable(frontier, max_in_flight)

            if not inline and not dispatchable:
                if joined:
                    supersteps += 1
                    continue
                # Waiting is not work, so it does not spend the superstep budget.
                if self._in_flight() and _monotonic() < deadline:
                    await asyncio.sleep(min(poll_seconds, max(0.0, deadline - _monotonic())))
                    continue
                stopped = self._terminal_reason(frontier)
                break

            supersteps += 1
            for node in inline:
                await self._execute(node)
                executed += 1
            for node in dispatchable:
                await self._dispatch(node)
            if save:
                self.save()
            if len(self._nodes) > max_nodes:
                stopped = "max_nodes"
                break

        if save:
            self.save()
        return RunReport(
            stopped=stopped,
            supersteps=supersteps,
            executed=executed,
            counts=self.counts(),
            pending_dispatch=tuple(node.id for node in self.frontier() if node.kind == "model"),
            in_flight=tuple(self._in_flight_report()),
            blocked=tuple(node.id for node in self.blocked()),
        )

    async def _ingest(self) -> int:
        """Join finished children. Returns how many claimed nodes were resolved.

        A result file is the outcome. A child the registry reports as finished
        without one did not follow the protocol, and its node fails saying so
        rather than holding the graph open forever.
        """
        claimed = self._in_flight()
        if not claimed:
            return 0
        statuses: dict[str, str] | None = None

        async def status_of(node: Node) -> str | None:
            nonlocal statuses
            if statuses is None and self._dispatcher is not None:
                statuses = await self._dispatcher.statuses()
            return (statuses or {}).get(node.child_id or "")

        resolved = 0
        for node in claimed:
            path = result_path(self.results_dir, node.id, max(0, len(node.tried_models) - 1))
            if path.exists():
                try:
                    outcome = parse_result(read_result(path), node)
                except ValueError as exc:
                    # A child still working may be mid-write. Failing it here
                    # would discard finished work over timing, so only a child
                    # that has stopped can be blamed for what its file contains.
                    if await status_of(node) == CHILD_RUNNING:
                        continue
                    node.state = "failed"
                    node.error = str(exc)
                else:
                    self._apply(node, outcome)
                resolved += 1
                continue
            status = await status_of(node)
            if status in (CHILD_COMPLETED, CHILD_ERROR):
                # A child that stopped without a result usually means the
                # provider gave out rather than the work being wrong, which is
                # what a quota-exhausted provider looks like from here: the
                # spawn is accepted because the credentials are still valid, and
                # the child dies partway. So the node reopens for the next
                # candidate model, and only an exhausted list fails it.
                node.error = (
                    f"child {node.child_id} on {node.tried_models[-1] if node.tried_models else 'the parent model'} "
                    f"finished as {status} without writing {path}"
                )
                node.state = "open"
                node.child_id = None
                node.claimed_at = None
                resolved += 1
        return resolved

    async def _dispatch(self, node: Node) -> None:
        assert self._dispatcher is not None
        remaining = [model for model in self._dispatcher.candidates(node) if _untried(model, node)]
        if not remaining:
            node.state = "failed"
            # The last attempt's reason is the diagnosis; exhaustion is only the
            # reason there will not be another one.
            exhausted = f"no candidate model left to try (tried {', '.join(node.tried_models) or 'the parent model'})"
            node.error = f"{node.error}; {exhausted}" if node.error else exhausted
            return

        model = remaining[0]
        attempt = len(node.tried_models)
        path = result_path(self.results_dir, node.id, attempt)
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            handle = await self._dispatcher.spawn(node, build_child_prompt(node, path), model)
        except Exception as exc:
            node.error = f"dispatch on {model or 'the parent model'} failed: {type(exc).__name__}: {exc}"
            if model is None:
                # Nothing was requested, so there is no other model to fall back to.
                node.state = "failed"
                return
            # Left open: the next superstep tries the next candidate, and an
            # exhausted list is what finally fails the node.
            node.tried_models = (*node.tried_models, model)
            return

        node.state = "claimed"
        node.child_id = handle.child_id
        node.claimed_at = _now()
        node.tried_models = (*node.tried_models, model or handle.model)

    def _in_flight_report(self) -> list[InFlight]:
        now = _now()
        return [
            InFlight(
                node_id=node.id,
                intent=node.intent,
                model=node.tried_models[-1] if node.tried_models else None,
                child_id=node.child_id,
                seconds=max(0.0, now - node.claimed_at) if node.claimed_at else 0.0,
            )
            for node in self._in_flight()
        ]

    def _in_flight(self) -> list[Node]:
        return [node for node in self._nodes.values() if node.state == "claimed"]

    def _dispatchable(self, frontier: list[Node], max_in_flight: int) -> list[Node]:
        if self._dispatcher is None:
            return []
        budget = max_in_flight - len(self._in_flight())
        if budget <= 0:
            return []
        return [node for node in frontier if node.kind == "model"][:budget]

    def save(self) -> None:
        """Checkpoint the graph.

        This is a whole-file write under the store lock, which is correct while
        one process owns the graph. A second writer needs a read-merge-write
        here, and lands with the dispatcher that introduces one.
        """
        with self._store.locked():
            self._store.write(
                {
                    "version": STORE_VERSION,
                    "name": self.name,
                    "nodes": [node.to_dict() for node in self._nodes.values()],
                }
            )

    def _deps_done(self, node: Node) -> bool:
        return all(self._nodes[need].state == "done" for need in node.needs)

    def _terminal_reason(self, frontier: list[Node]) -> str:
        if self._in_flight():
            # Without a dispatcher nothing can adjudicate a child that died, so
            # these nodes can only ever resolve if a result file appears.
            # Calling that "in_flight" would imply someone is still working.
            return "in_flight" if self._dispatcher is not None else "detached"
        if any(node.kind == "model" for node in frontier):
            return "pending_dispatch"
        if any(node.state == "open" for node in self._nodes.values()):
            return "blocked"
        return "complete"

    async def _execute(self, node: Node) -> None:
        results = {need: self._nodes[need].result for need in node.needs}
        try:
            if node.fn is None:
                outcome: Any = Done(self._collect(node, results))
            else:
                body = resolve(node.fn)
                outcome = body(Context(node=node, results=results, args=dict(node.args)))
                if inspect.isawaitable(outcome):
                    outcome = await outcome
        except Exception as exc:
            node.state = "failed"
            node.error = f"{type(exc).__name__}: {exc}"
            return
        self._apply(node, outcome)

    @staticmethod
    def _collect(node: Node, results: Mapping[str, Any]) -> Any:
        if not node.needs:
            return None
        return [results[need] for need in node.needs]

    def _apply(self, node: Node, outcome: Any) -> None:
        if not isinstance(outcome, (Done, Reject, Expand)):
            # A body that just returns a value has finished with that value.
            outcome = Done(outcome)

        if isinstance(outcome, Reject):
            node.state = "rejected"
            node.reason = outcome.reason
            return

        if isinstance(outcome, Done):
            try:
                assert_jsonable(outcome.result, node.id)
            except ValueError as exc:
                node.state = "failed"
                node.error = str(exc)
                return
            node.state = "done"
            node.result = outcome.result
            return

        # `replace` rather than listing fields: parentage is the only thing being
        # changed, and a hand-written copy silently drops every field added later.
        children = tuple(
            replace(child, args=dict(child.args), parents=tuple(dict.fromkeys((*child.parents, node.id))))
            for child in outcome.children
        )
        try:
            for child in children:
                assert_storable(child)
            self._check_additions(children, extra_needs={node.id: {child.id for child in children}})
        except ValueError as exc:
            node.state = "failed"
            node.error = str(exc)
            return
        for child in children:
            self._nodes[child.id] = child
        node.needs = tuple(dict.fromkeys((*node.needs, *(child.id for child in children))))
        node.fn = outcome.then
        # The whole body is replaced, so everything that described the old body
        # goes with it. A `model` left behind outlives the `prompt` it selected
        # for, and `Node` rejects that pairing on load, which turns one expansion
        # into a graph that can never be reopened.
        node.prompt = None
        node.model = None
        node.child_id = None
        # Set explicitly rather than left alone: an expansion arriving from a
        # dispatched child finds the node claimed, and a node that stays claimed
        # is never runnable and has its result file re-read forever.
        node.state = "open"
        # It runs again once its new barrier releases, and `then` may expand
        # again, which is what makes decomposition unbounded.

    def _check_additions(self, additions: Iterable[Node], extra_needs: Mapping[str, set[str]] | None = None) -> None:
        additions = list(additions)
        new_ids = [node.id for node in additions]
        duplicates = {node_id for node_id in new_ids if new_ids.count(node_id) > 1}
        if duplicates:
            raise ValueError(f"duplicate node ids in one addition: {sorted(duplicates)}")
        clashes = set(new_ids) & set(self._nodes)
        if clashes:
            raise ValueError(f"node ids already exist in graph {self.name!r}: {sorted(clashes)}")

        adjacency: dict[str, set[str]] = {node_id: set(node.needs) for node_id, node in self._nodes.items()}
        for node in additions:
            adjacency[node.id] = set(node.needs)
        for node_id, extra in (extra_needs or {}).items():
            adjacency.setdefault(node_id, set()).update(extra)

        unknown = sorted({need for needs in adjacency.values() for need in needs if need not in adjacency})
        if unknown:
            raise ValueError(f"needs reference unknown nodes: {unknown}")
        _assert_acyclic(adjacency)

    def _assert_consistent(self) -> None:
        adjacency = {node_id: set(node.needs) for node_id, node in self._nodes.items()}
        unknown = sorted({need for needs in adjacency.values() for need in needs if need not in adjacency})
        if unknown:
            raise ValueError(f"graph {self.name!r} references unknown nodes: {unknown}")
        _assert_acyclic(adjacency)


def _untried(model: str | None, node: Node) -> bool:
    """A named model is untried until it appears in `tried_models`.

    `None` means "whatever the parent is using", which is one attempt and has no
    alternative to fall back to, so it is untried only before anything ran.
    """
    return not node.tried_models if model is None else model not in node.tried_models


def _assert_acyclic(adjacency: Mapping[str, set[str]]) -> None:
    """Depth-first cycle check.

    A cycle in `needs` would otherwise present as a permanently blocked frontier
    with no failed dependency, which is a confusing way to learn about a typo.
    """
    unvisited, visiting, visited = 0, 1, 2
    marks: dict[str, int] = {node_id: unvisited for node_id in adjacency}
    for root in adjacency:
        if marks[root] != unvisited:
            continue
        stack: list[tuple[str, Any]] = [(root, iter(sorted(adjacency[root])))]
        marks[root] = visiting
        path = [root]
        while stack:
            node_id, pending = stack[-1]
            advanced = False
            for need in pending:
                if marks[need] == visiting:
                    cycle = path[path.index(need):] + [need]
                    raise ValueError(f"needs form a cycle: {' -> '.join(cycle)}")
                if marks[need] == unvisited:
                    marks[need] = visiting
                    path.append(need)
                    stack.append((need, iter(sorted(adjacency[need]))))
                    advanced = True
                    break
            if not advanced:
                marks[node_id] = visited
                stack.pop()
                path.pop()
