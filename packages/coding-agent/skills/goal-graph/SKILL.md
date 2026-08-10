---
name: goal-graph
description: Decompose a goal into a graph of work and run it from IPython. Use when a goal is large enough to need tracked sub-work, when work must resume across turns or sessions, or when you need to record that an approach was tried and rejected.
---

# Goal Graph

A goal graph holds one kind of node at every scale. A goal, a task, and a single
deterministic step are the same record, so any node can be split further without
changing its type and without a separate planning pass. Decomposition is a
return value.

The graph is Python objects backed by a JSON file under
`~/.prime/agent/goal-graphs/`. It outlives the session, and another session or a
person can read it directly.

```python
from goal_graph import Graph, Node, Expand, Done, Reject, register

@register
def split_files(ctx):
    return Expand([
        Node(intent=f"lint {path}", fn="lint_one", args={"path": path})
        for path in ctx.args["paths"]
    ])

@register
def lint_one(ctx):
    ok = check(ctx.args["path"])
    return Done({"path": ctx.args["path"], "ok": ok})

g = Graph.open("lint-pass")
g.add(Node(intent="lint the package", fn="split_files", args={"paths": ["a.py", "b.py"]}))
report = await g.run()
report.stopped   # "complete"
```

## Node bodies

A node has exactly one body:

- **inline**: `fn` names a callable registered with `@register`. It runs in this
  kernel. An inline node costs approximately nothing, which is what makes deep
  decomposition affordable.
- **model**: `prompt` is set. Running it needs a model, so it is dispatched to an
  RLM child. Without a dispatcher it is reported in `report.pending_dispatch`
  instead. `model` may name a `provider/model` selector for that child.
- **collector**: neither is set. The node completes with the list of its `needs`
  results, in `needs` order. Use it for a milestone or a join.

An inline body is called with a `Context` carrying `node`, `results` (a dict of
dependency id to result), and `args`. It may be sync or async. It returns:

- `Done(result)`, or any plain value, which is treated as `Done(value)`. The
  result must be JSON-serializable.
- `Reject(reason)` when the approach is wrong. The branch stays in the graph
  with its reason instead of being deleted, so a later run can see what was
  ruled out.
- `Expand([...], then=None)` to decompose. The children are added, the node
  gains them as `needs`, its body becomes `then`, and it stays open so it runs
  again once they finish. `then` may expand again.

A body that raises marks its node `failed` with the exception text. Other
branches keep running; nodes downstream of the failure become blocked.

## Dispatching model nodes

`rlm()` returns an admission handle, never the child's answer, so the join is
built here. Each dispatched node names a result file; the child writes it and
sends one short line to its parent. The message is only a wake-up, the file is
the result. That keeps a large result out of the 16KB agent-message cap and
survives the parent compacting or restarting first.

```python
from goal_graph import Graph, Node, RlmDispatcher

dispatcher = RlmDispatcher(models=["opencode-go/glm-5.2", "devin-2/glm-5-2", "devin-1/glm-5-2"])
g = Graph.open("review", dispatcher=dispatcher)
g.add(Node(intent="review the diff", prompt="Review the staged diff and list defects."))

report = await g.run()          # dispatches, then returns "in_flight"
# ... the turn ends; when a child messages back:
report = await g.run()          # joins whatever finished and keeps going
```

`models` is an ordered fallback list. A provider that has run out of quota does
not refuse the spawn, because its credentials are still valid: the child is
admitted and then dies partway. So a child that stops without leaving a result
reopens its node for the next model rather than failing it, and only an
exhausted list fails the node. The last attempt's failure stays as the
diagnosis. A node's own `model` leads and the rest remain as fallback. Each
attempt writes its own result file, so a retry can never read the previous
attempt's answer.

`report.in_flight` carries the node id, intent, model, child id, and how long
the attempt has been running, because a list of ids cannot tell a child that is
working from one that has stopped responding.

Pass `wait_seconds` to block in the cell instead of ending the turn. A child
that the subagent registry reports as finished without leaving a result file
fails its node saying so, rather than holding the graph open forever.

A dispatched child may itself return `expand`, so decomposition is available at
any depth, not only to whatever created the graph. Its children may name an `fn`
registered in this kernel, and may name a `model`.

A child cannot see or choose node ids. To order its children it gives one a
`key` of its own choosing and lists that key in another child's `needs`; the
keys are resolved to real ids here. Children are told to write their result to a
temporary file and rename it into place, and a result that will not parse while
the child is still running is retried rather than treated as a failure, so
finished work is not discarded over timing.

## API

- `Graph.open(name, store_dir=None, dispatcher=None)` — load the named graph or
  start an empty one. Without a dispatcher, model nodes are reported rather than
  run.
- `g.add(node)` — add one node. Its `needs` must already exist and must not form
  a cycle.
- `await g.run(max_supersteps=10000, max_nodes=10000, max_in_flight=4, wait_seconds=0, poll_seconds=2, save=True)`
  — join finished children, run the frontier, dispatch model nodes, checkpoint,
  repeat until nothing is runnable. Returns a `RunReport` with `stopped`,
  `supersteps`, `executed`, `counts`, `pending_dispatch`, `in_flight`, and
  `blocked`.
- `g.results_dir` — where dispatched children write results, beside the graph
  file.
- `g.frontier()` — open nodes whose `needs` are all done.
- `g.blocked()` — open nodes with a failed or rejected dependency.
- `g.children_of(node_id)` — derived from `parents`, never stored.
- `g.get(node_id)`, `g.nodes()`, `g.counts()`, `g.save()`.

## Rules

- A run ends when the frontier is empty. `max_supersteps`, `max_nodes`, and
  `in_flight` are reported as their own stop reasons and are not completion.
  Only `stopped == "complete"` means there is nothing left to do.
- Register bodies by name and keep them importable. A node stores the name, not
  the function, so a graph reloads after a kernel restart. A missing body fails
  that node with a clear message rather than corrupting the graph.
- Do not mutate the graph from inside a node body. Return `Expand` instead; a
  body that reaches around the return value makes the checkpoint wrong.
- Use `Reject` rather than deleting a node when an approach is falsified.
- One process should own a graph while it runs. `save()` writes the whole file
  under a lock, which is correct for a single writer.
