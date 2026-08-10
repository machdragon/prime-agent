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
from typing import Any, Protocol, Sequence

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

    Narrow on purpose: it is the seam a capacity-aware router plugs into, and
    the seam a test replaces without booting a session. `candidates` is that
    seam: a router that reads remaining quota answers it differently without
    the graph changing.
    """

    def candidates(self, node: Node) -> tuple[str | None, ...]: ...

    async def spawn(self, node: Node, prompt: str, model: str | None) -> DispatchHandle: ...

    async def statuses(self) -> dict[str, str]: ...


class RlmDispatcher:
    """The default dispatcher: one RLM child per model node.

    `models` is an ordered fallback list. A provider that has run out of quota
    does not fail at spawn, because its credentials are still valid; it fails
    once the child stops without producing a result. So the list is what the
    graph walks when an attempt dies, not just when one is refused.
    """

    def __init__(self, models: Sequence[str] | None = None, default_model: str | None = None) -> None:
        configured: list[str] = list(models or ())
        if default_model and default_model not in configured:
            configured.insert(0, default_model)
        self.models: tuple[str, ...] = tuple(configured)

    def candidates(self, node: Node) -> tuple[str | None, ...]:
        if node.model:
            # The node's own choice leads; the rest stay available as fallback.
            return (node.model, *(model for model in self.models if model != node.model))
        return self.models or (None,)

    async def spawn(self, node: Node, prompt: str, model: str | None) -> DispatchHandle:
        # Imported per call, not at module scope: `rlm` exists only inside the
        # Prime Agent kernel, and the rest of this package must stay importable
        # (and testable) outside one.
        import rlm

        kwargs: dict[str, Any] = {"name": child_name(node, len(node.tried_models))}
        if model:
            kwargs["model"] = model
        handle = await rlm.run(prompt, **kwargs)
        return DispatchHandle(child_id=handle.rlm_child_id, name=handle.name, model=handle.model)

    async def statuses(self) -> dict[str, str]:
        import rlm

        return {child.rlm_child_id: child.status for child in await rlm.list_subagents()}


def child_name(node: Node, attempt: int) -> str:
    # The registry rejects duplicate names, so a retry cannot reuse the first
    # attempt's name.
    return f"node-{node.id}-{attempt}"


def result_path(results_dir: Path, node_id: str, attempt: int) -> Path:
    """One file per attempt, so a retry can never read the last attempt's result."""
    return Path(results_dir) / f"{node_id}.{attempt}.json"


def build_child_prompt(node: Node, path: Path) -> str:
    """Wrap a node's prompt with the result protocol.

    The node id is the correlation token. It is stated in the prompt and is the
    result file's name, so a reply that loses it can still be matched.
    """
    return f"""{node.prompt}

---
You are working on one node of a goal graph. Its id is {node.id}.

When you are finished, write your outcome as JSON to this exact path:

    {path}

Write it to a temporary file in the same directory and rename it into place, so
your parent never reads a half-written file. In Python:

    import json, os
    tmp = "{path}.partial"
    with open(tmp, "w") as handle:
        json.dump(outcome, handle)
    os.replace(tmp, "{path}")

The file must be an object with an "outcome" key set to one of:

- {{"outcome": "done", "result": <json>}} when the work is complete. `result` is
  what the rest of the graph receives, so make it the answer, not a narration.
- {{"outcome": "reject", "reason": "<why>"}} when the approach is wrong. Prefer
  this over forcing a result; a rejected node keeps its reason in the graph.
- {{"outcome": "expand", "children": [{{"intent": "...", "prompt": "..."}}, ...]}}
  when the work should be split. Each child needs an "intent" and either a
  "prompt" (a model does it), an "fn" naming a body registered in the parent, or
  neither (it collects its dependencies). "model" may name a provider/model
  selector for that child. Your node completes when its children do.

  To order children, give a child a "key" of your choosing and list those keys
  in another child's "needs". Keys are local to this one list and are resolved
  to real ids here; you cannot see or set ids. For example:

      {{"outcome": "expand", "children": [
        {{"key": "build", "intent": "build it", "prompt": "..."}},
        {{"intent": "test it", "prompt": "...", "needs": ["build"]}}
      ]}}

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
        return Expand(_expand_children(raw_children, node))
    raise ValueError(f"result file for {node.id} has unknown outcome {outcome!r}")


def _expand_children(raw_children: list[Any], parent: Node) -> list[Node]:
    """Build the children, resolving sibling ordering by local key.

    A child cannot know the ids in the parent's graph and is not allowed to
    invent them, so ordering is expressed with keys it chooses itself and this
    maps them to the ids assigned here.
    """
    built = [(_child_node(raw, parent), raw) for raw in raw_children]

    keys: dict[str, str] = {}
    for child, raw in built:
        key = raw.get("key")
        if key is None:
            continue
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"child of {parent.id} has an invalid key")
        if key in keys:
            raise ValueError(f"children of {parent.id} reuse the key {key!r}")
        keys[key] = child.id

    for child, raw in built:
        requested = raw.get("needs") or []
        if isinstance(requested, str) or not isinstance(requested, list):
            raise ValueError(f"child of {parent.id} must give needs as a list of sibling keys")
        resolved = []
        for entry in requested:
            if not isinstance(entry, str) or entry not in keys:
                raise ValueError(
                    f"child of {parent.id} needs {entry!r}, which is not the key of any sibling in this expansion"
                )
            resolved.append(keys[entry])
        child.needs = tuple(dict.fromkeys(resolved))
    return [child for child, _ in built]


def _child_node(raw: Any, parent: Node) -> Node:
    if not isinstance(raw, dict):
        raise ValueError(f"child of {parent.id} must be an object, got {type(raw).__name__}")
    if "id" in raw:
        # Ids are assigned here, not by the child: a child cannot know what is
        # already in the graph, and a collision would fail the whole expansion.
        # Ordering between siblings goes through "key" instead.
        raise ValueError(f"child of {parent.id} must not choose its own id")
    intent = raw.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError(f"child of {parent.id} is missing an intent")
    # `needs` is deliberately absent here: it names sibling keys, which only mean
    # something once every sibling has an id. `_expand_children` fills it in.
    return Node(
        intent=intent,
        fn=_optional_str(raw.get("fn"), "fn", parent),
        prompt=_optional_str(raw.get("prompt"), "prompt", parent),
        args=dict(raw.get("args") or {}),
        model=_optional_str(raw.get("model"), "model", parent),
    )


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
