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

import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
from .store import STORE_VERSION, GraphStore, default_store_dir, graph_path

__all__ = [
    "Context",
    "Done",
    "Expand",
    "Graph",
    "GraphStore",
    "InlineBody",
    "Node",
    "Outcome",
    "Reject",
    "RunReport",
    "default_store_dir",
    "graph_path",
    "register",
    "registered",
    "resolve",
]

DEFAULT_MAX_SUPERSTEPS = 10_000
DEFAULT_MAX_NODES = 10_000

#: Why `run()` returned. Only "complete" means the graph has nothing left to do.
STOP_REASONS: tuple[str, ...] = (
    "complete",
    "pending_dispatch",
    "blocked",
    "max_supersteps",
    "max_nodes",
)


@dataclass(frozen=True)
class RunReport:
    stopped: str
    supersteps: int
    executed: int
    counts: Mapping[str, int]
    pending_dispatch: tuple[str, ...] = ()
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
            "blocked": list(self.blocked),
        }


class Graph:
    """A goal graph and the loop that runs it."""

    def __init__(self, name: str, store: GraphStore, nodes: Iterable[Node] = ()) -> None:
        self.name = name
        self._store = store
        self._nodes: dict[str, Node] = {}
        for node in nodes:
            self._nodes[node.id] = node

    @classmethod
    def open(cls, name: str, *, store_dir: str | None = None) -> Graph:
        """Load the named graph, or start an empty one under the same name."""
        store = GraphStore(graph_path(name, store_dir))
        payload = store.read()
        if payload is None:
            return cls(name, store)
        nodes = [Node.from_dict(record) for record in payload.get("nodes", ())]
        graph = cls(payload.get("name", name), store, nodes)
        graph._assert_consistent()
        return graph

    @property
    def path(self) -> str:
        return str(self._store.path)

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
        """Open nodes that can never run because a dependency failed or was rejected."""
        return [
            node
            for node in self._nodes.values()
            if node.state == "open"
            and any(self._nodes[need].state in ("failed", "rejected") for need in node.needs)
        ]

    async def run(
        self,
        *,
        max_supersteps: int = DEFAULT_MAX_SUPERSTEPS,
        max_nodes: int = DEFAULT_MAX_NODES,
        save: bool = True,
    ) -> RunReport:
        """Run the frontier until nothing is runnable.

        Each superstep executes every runnable inline and collector node, then
        checkpoints. Model nodes are left for the dispatcher and reported as
        pending. Limits are safety rails: reaching one is not completion.
        """
        if max_supersteps < 1:
            raise ValueError("max_supersteps must be at least 1")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")

        supersteps = 0
        executed = 0
        stopped = "complete"

        while True:
            frontier = self.frontier()
            batch = [node for node in frontier if node.kind != "model"]
            if not batch:
                stopped = self._terminal_reason(frontier)
                break
            if supersteps >= max_supersteps:
                stopped = "max_supersteps"
                break
            supersteps += 1
            for node in batch:
                await self._execute(node)
                executed += 1
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
            blocked=tuple(node.id for node in self.blocked()),
        )

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
        if any(node.kind == "model" for node in frontier):
            return "pending_dispatch"
        if any(node.state in ("open", "claimed") for node in self._nodes.values()):
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

        children = tuple(
            Node(
                intent=child.intent,
                id=child.id,
                fn=child.fn,
                prompt=child.prompt,
                args=dict(child.args),
                needs=child.needs,
                parents=tuple(dict.fromkeys((*child.parents, node.id))),
                state=child.state,
                result=child.result,
                error=child.error,
                reason=child.reason,
            )
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
        node.prompt = None
        # The node stays open: it runs again once its new barrier releases, and
        # `then` may expand again, which is what makes decomposition unbounded.

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
