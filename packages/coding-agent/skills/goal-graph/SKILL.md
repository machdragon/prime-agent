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
- **model**: `prompt` is set. Running it needs a model. `run()` does not execute
  these; it reports them in `report.pending_dispatch` and stops with
  `"pending_dispatch"`.
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

## API

- `Graph.open(name, store_dir=None)` — load the named graph or start an empty
  one.
- `g.add(node)` — add one node. Its `needs` must already exist and must not form
  a cycle.
- `await g.run(max_supersteps=10000, max_nodes=10000, save=True)` — run the
  frontier until nothing is runnable, checkpointing after each superstep.
  Returns a `RunReport` with `stopped`, `supersteps`, `executed`, `counts`,
  `pending_dispatch`, and `blocked`.
- `g.frontier()` — open nodes whose `needs` are all done.
- `g.blocked()` — open nodes with a failed or rejected dependency.
- `g.children_of(node_id)` — derived from `parents`, never stored.
- `g.get(node_id)`, `g.nodes()`, `g.counts()`, `g.save()`.

## Rules

- A run ends when the frontier is empty. `max_supersteps` and `max_nodes` are
  safety limits; hitting one is reported as its own stop reason and is not
  completion.
- Register bodies by name and keep them importable. A node stores the name, not
  the function, so a graph reloads after a kernel restart. A missing body fails
  that node with a clear message rather than corrupting the graph.
- Do not mutate the graph from inside a node body. Return `Expand` instead; a
  body that reaches around the return value makes the checkpoint wrong.
- Use `Reject` rather than deleting a node when an approach is falsified.
- One process should own a graph while it runs. `save()` writes the whole file
  under a lock, which is correct for a single writer.
