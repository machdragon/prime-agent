"""Dispatching model nodes to RLM children, and joining their results.

`rlm()` returns an admission handle, never the child's answer, and a child runs
in its own session with its own kernel. So the join is built here rather than
borrowed: every dispatch names a result file, the child writes that file, and
the parent ingests it. An agent message from the child is only a wake-up; the
file is the result. That keeps a large result out of the 16KB message cap and
survives the parent compacting or restarting before the child finishes.

A child that the subagent registry reports as finished without leaving a result
file did not follow the protocol, and its node fails with that stated reason
rather than waiting forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .model import Done, Expand, Node, Outcome, Reject

#: Statuses the RLM subagent registry reports.
CHILD_RUNNING = "running"
CHILD_COMPLETED = "completed"
CHILD_ERROR = "error"


@dataclass(frozen=True)
class DispatchHandle:
    child_id: str
    name: str
    model: str


class Dispatcher(Protocol):
    """What the graph needs from a child runtime.

    Narrow on purpose: it is the seam the capacity-aware router plugs into, and
    the seam a test replaces without booting a session.
    """

    async def spawn(self, node: Node, prompt: str) -> DispatchHandle: ...

    async def statuses(self) -> dict[str, str]: ...


class RlmDispatcher:
    """The default dispatcher: one RLM child per model node."""

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model

    async def spawn(self, node: Node, prompt: str) -> DispatchHandle:
        # Imported per call, not at module scope: `rlm` exists only inside the
        # Prime Agent kernel, and the rest of this package must stay importable
        # (and testable) outside one.
        import rlm

        kwargs: dict[str, Any] = {"name": _child_name(node)}
        model = node.model or self.default_model
        if model:
            kwargs["model"] = model
        handle = await rlm.run(prompt, **kwargs)
        return DispatchHandle(child_id=handle.rlm_child_id, name=handle.name, model=handle.model)

    async def statuses(self) -> dict[str, str]:
        import rlm

        return {child.rlm_child_id: child.status for child in await rlm.list_subagents()}


def _child_name(node: Node) -> str:
    # The registry rejects duplicate names, and a node is dispatched once, so
    # the node id is both stable and unique.
    return f"node-{node.id}"


def result_path(results_dir: Path, node_id: str) -> Path:
    return Path(results_dir) / f"{node_id}.json"


def build_child_prompt(node: Node, path: Path) -> str:
    """Wrap a node's prompt with the result protocol.

    The node id is the correlation token. It is stated in the prompt and is the
    result file's name, so a reply that loses it can still be matched.
    """
    return f"""{node.prompt}

---
You are working on one node of a goal graph. Its id is {node.id}.

When you are finished, write your outcome to this exact path as JSON:

    {path}

The file must be an object with an "outcome" key set to one of:

- {{"outcome": "done", "result": <json>}} when the work is complete. `result` is
  what the rest of the graph receives, so make it the answer, not a narration.
- {{"outcome": "reject", "reason": "<why>"}} when the approach is wrong. Prefer
  this over forcing a result; a rejected node keeps its reason in the graph.
- {{"outcome": "expand", "children": [{{"intent": "...", "prompt": "..."}}, ...]}}
  when the work should be split. Each child needs an "intent" and either a
  "prompt" (a model does it), an "fn" naming a body registered in the parent, or
  neither (it collects its dependencies). You may also set "needs" to a list of
  ids of other children in the same list, and "model" to a provider/model
  selector. Your node completes when its children do.

Write the file before you reply. Then send one short line to your parent with
`agent_message.send(..., receiver_role="parent")` starting with {node.id}. The
message is only a wake-up; the file carries the result.
"""


def parse_result(payload: Any, node: Node) -> Outcome:
    """Turn a child's result file into an outcome.

    Every failure here is the child's protocol error, so each raises ValueError
    with what was wrong; the caller fails that node and leaves the graph intact.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"result file for {node.id} must contain an object, got {type(payload).__name__}")
    outcome = payload.get("outcome")
    if outcome == "done":
        return Done(payload.get("result"))
    if outcome == "reject":
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"result file for {node.id} rejected without a reason")
        return Reject(reason.strip())
    if outcome == "expand":
        raw_children = payload.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ValueError(f"result file for {node.id} expanded without children")
        return Expand([_child_node(raw, node) for raw in raw_children])
    raise ValueError(f"result file for {node.id} has unknown outcome {outcome!r}")


def _child_node(raw: Any, parent: Node) -> Node:
    if not isinstance(raw, dict):
        raise ValueError(f"child of {parent.id} must be an object, got {type(raw).__name__}")
    intent = raw.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError(f"child of {parent.id} is missing an intent")
    node = Node(
        intent=intent,
        fn=_optional_str(raw.get("fn"), "fn", parent),
        prompt=_optional_str(raw.get("prompt"), "prompt", parent),
        args=dict(raw.get("args") or {}),
        needs=tuple(raw.get("needs") or ()),
        model=_optional_str(raw.get("model"), "model", parent),
    )
    if "id" in raw:
        # Ids are assigned here, not by the child: a child cannot know what is
        # already in the graph, and a collision would fail the whole expansion.
        raise ValueError(f"child of {parent.id} must not choose its own id")
    return node


def _optional_str(value: Any, field: str, parent: Node) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"child of {parent.id} has an invalid {field}")
    return value.strip()


def read_result(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"result file {path} is not valid JSON: {exc}") from None
