"""Node model for the goal graph.

There is exactly one node type at every scale. A goal, a task, and a single
deterministic step are the same record; only the body differs. Decomposition is
an ordinary return value (`Expand`), not a privileged operation of a planner, so
a node can always be split further without changing its type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

NODE_STATES: tuple[str, ...] = ("open", "claimed", "done", "failed", "rejected")


def new_node_id() -> str:
    return f"n_{uuid4().hex[:12]}"


@dataclass
class Node:
    """One unit of work.

    The body is one of three forms:

    - inline: `fn` names a callable registered with `register()`. It runs in the
      orchestrator's own kernel, so an extra node costs approximately nothing.
    - model: `prompt` is set. Executing it needs a model, which the dispatcher
      owns; this module only reports such nodes as pending.
    - collector: neither is set. The node completes with the results of
      everything in `needs`, which is what a milestone or a join is.
    """

    intent: str
    id: str = field(default_factory=new_node_id)
    fn: str | None = None
    prompt: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    needs: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    state: str = "open"
    result: Any = None
    error: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("node intent must be a non-empty string")
        self.intent = self.intent.strip()
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("node id must be a non-empty string")
        if self.fn is not None and self.prompt is not None:
            raise ValueError(f"node {self.id} sets both fn and prompt; a node has one body")
        if self.state not in NODE_STATES:
            raise ValueError(f"node {self.id} has unknown state {self.state!r}")
        if not isinstance(self.args, dict):
            raise TypeError(f"node {self.id} args must be a dict, got {type(self.args).__name__}")
        self.needs = _id_tuple(self.needs, f"node {self.id} needs")
        self.parents = _id_tuple(self.parents, f"node {self.id} parents")
        if self.id in self.needs:
            raise ValueError(f"node {self.id} cannot need itself")

    @property
    def kind(self) -> str:
        if self.fn is not None:
            return "inline"
        if self.prompt is not None:
            return "model"
        return "collector"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent,
            "state": self.state,
            "fn": self.fn,
            "prompt": self.prompt,
            "args": self.args,
            "needs": list(self.needs),
            "parents": list(self.parents),
            "result": self.result,
            "error": self.error,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Node:
        if not isinstance(raw, dict):
            raise ValueError(f"node record must be an object, got {type(raw).__name__}")
        try:
            return cls(
                intent=raw["intent"],
                id=raw["id"],
                fn=raw.get("fn"),
                prompt=raw.get("prompt"),
                args=dict(raw.get("args") or {}),
                needs=tuple(raw.get("needs") or ()),
                parents=tuple(raw.get("parents") or ()),
                state=raw.get("state", "open"),
                result=raw.get("result"),
                error=raw.get("error"),
                reason=raw.get("reason"),
            )
        except KeyError as exc:
            raise ValueError(f"node record is missing {exc.args[0]!r}") from None


def _id_tuple(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{label} must be a sequence of ids, not a bare string")
    try:
        items = list(value)
    except TypeError:
        raise TypeError(f"{label} must be a sequence of ids") from None
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} must contain non-empty string ids")
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class Done:
    """The node is finished. `result` must be JSON-serializable to persist."""

    result: Any = None


@dataclass(frozen=True)
class Reject:
    """The approach was tried and is wrong.

    Rejection keeps the falsified branch and its reason in the graph instead of
    deleting it, so a later run can see what was already ruled out.
    """

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Reject requires a non-empty reason")


@dataclass(frozen=True)
class Expand:
    """Decompose into children and wait for them.

    The expanding node keeps its identity: the children are added to `needs`,
    the body is replaced by `then`, and the node stays open so it runs again
    once the children finish. `then` may itself return `Expand`, which is what
    makes decomposition unbounded.
    """

    children: Sequence[Node]
    then: str | None = None

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if not children:
            raise ValueError("Expand requires at least one child node")
        for child in children:
            if not isinstance(child, Node):
                raise TypeError(f"Expand children must be Node, got {type(child).__name__}")
        if self.then is not None and (not isinstance(self.then, str) or not self.then.strip()):
            raise ValueError("Expand then must be a registered body name or None")
        object.__setattr__(self, "children", children)


Outcome = Done | Reject | Expand


@dataclass(frozen=True)
class Context:
    """What an inline body is given.

    The graph is deliberately absent: a node influences the graph through its
    return value, never by mutating it.
    """

    node: Node
    results: Mapping[str, Any]
    args: Mapping[str, Any]


InlineBody = Callable[[Context], Any]

_REGISTRY: dict[str, InlineBody] = {}


def register(fn: InlineBody | None = None, *, name: str | None = None) -> Any:
    """Register an inline node body under a stable name.

    Nodes reference a body by name so a graph stays plain JSON and survives a
    kernel restart. Re-registering a name replaces it, because re-running a cell
    is ordinary in a kernel.
    """

    def decorate(target: InlineBody) -> InlineBody:
        key = name or getattr(target, "__name__", "")
        if not key:
            raise ValueError("inline body needs a name")
        _REGISTRY[key] = target
        return target

    return decorate if fn is None else decorate(fn)


def resolve(name: str) -> InlineBody:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no inline body registered as {name!r}; import the module that registers it") from None


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def assert_jsonable(value: Any, node_id: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"node {node_id} produced a result that cannot be stored as JSON: {exc}") from None
