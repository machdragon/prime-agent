from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from goal_graph import Done, Expand, Graph, Node, Reject, register
from goal_graph.store import GraphStore, graph_path, slug


def run(coro):
    return asyncio.run(coro)


class GraphTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.store_dir = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def graph(self, name: str = "test") -> Graph:
        return Graph.open(name, store_dir=self.store_dir)


class NodeModelTest(GraphTestCase):
    def test_a_node_has_exactly_one_body(self) -> None:
        with self.assertRaisesRegex(ValueError, "one body"):
            Node(intent="both", fn="a", prompt="b")

    def test_node_kind_is_derived_from_the_body(self) -> None:
        self.assertEqual(Node(intent="x", fn="a").kind, "inline")
        self.assertEqual(Node(intent="x", prompt="a").kind, "model")
        self.assertEqual(Node(intent="x").kind, "collector")

    def test_intent_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "intent"):
            Node(intent="   ")

    def test_needs_reject_a_bare_string(self) -> None:
        with self.assertRaisesRegex(TypeError, "not a bare string"):
            Node(intent="x", needs="other")

    def test_a_node_cannot_need_itself(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot need itself"):
            Node(intent="x", id="n1", needs=("n1",))

    def test_reject_requires_a_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty reason"):
            Reject("  ")

    def test_expand_requires_children(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one child"):
            Expand([])

    def test_node_round_trips_through_a_dict(self) -> None:
        node = Node(intent="x", fn="body", args={"k": 1}, needs=("a",), parents=("p",))
        restored = Node.from_dict(json.loads(json.dumps(node.to_dict())))
        self.assertEqual(restored, node)


class FrontierTest(GraphTestCase):
    def test_needs_act_as_a_barrier(self) -> None:
        register(lambda ctx: Done("first"), name="first")
        register(lambda ctx: Done("second"), name="second")
        g = self.graph()
        a = g.add(Node(intent="a", fn="first"))
        b = g.add(Node(intent="b", fn="second", needs=(a.id,)))

        self.assertEqual([node.id for node in g.frontier()], [a.id])
        report = run(g.run())
        self.assertTrue(report.complete)
        self.assertEqual(g.get(b.id).result, "second")

    def test_a_plain_return_value_completes_the_node(self) -> None:
        register(lambda ctx: {"ok": True}, name="plain")
        g = self.graph()
        node = g.add(Node(intent="plain", fn="plain"))
        run(g.run())
        self.assertEqual(g.get(node.id).state, "done")
        self.assertEqual(g.get(node.id).result, {"ok": True})

    def test_a_collector_gathers_its_needs_in_order(self) -> None:
        register(lambda ctx: Done(ctx.args["value"]), name="value")
        g = self.graph()
        a = g.add(Node(intent="a", fn="value", args={"value": 1}))
        b = g.add(Node(intent="b", fn="value", args={"value": 2}))
        joined = g.add(Node(intent="join", needs=(a.id, b.id)))
        run(g.run())
        self.assertEqual(g.get(joined.id).result, [1, 2])

    def test_an_async_body_is_awaited(self) -> None:
        async def slow(ctx):
            await asyncio.sleep(0)
            return Done("slow")

        register(slow, name="slow")
        g = self.graph()
        node = g.add(Node(intent="slow", fn="slow"))
        run(g.run())
        self.assertEqual(g.get(node.id).result, "slow")

    def test_a_body_receives_its_dependency_results(self) -> None:
        register(lambda ctx: Done(2), name="two")
        register(lambda ctx: Done(sum(ctx.results.values()) + ctx.args["add"]), name="total")
        g = self.graph()
        a = g.add(Node(intent="a", fn="two"))
        b = g.add(Node(intent="b", fn="total", needs=(a.id,), args={"add": 5}))
        run(g.run())
        self.assertEqual(g.get(b.id).result, 7)


class ExpandTest(GraphTestCase):
    def test_expand_adds_children_and_the_parent_waits_for_them(self) -> None:
        register(
            lambda ctx: Expand([Node(intent=f"child {i}", fn="leaf", args={"i": i}) for i in range(3)]),
            name="fan",
        )
        register(lambda ctx: Done(ctx.args["i"]), name="leaf")

        g = self.graph()
        root = g.add(Node(intent="fan out", fn="fan"))
        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(len(g), 4)
        children = g.children_of(root.id)
        self.assertEqual(len(children), 3)
        self.assertEqual(sorted(g.get(root.id).result), [0, 1, 2])
        for child in children:
            self.assertEqual(child.parents, (root.id,))

    def test_expand_then_runs_after_the_children(self) -> None:
        register(lambda ctx: Expand([Node(intent="leaf", fn="leaf")], then="finish"), name="start")
        register(lambda ctx: Done(4), name="leaf")
        register(lambda ctx: Done(sum(ctx.results.values()) * 10), name="finish")

        g = self.graph()
        root = g.add(Node(intent="start", fn="start"))
        run(g.run())
        self.assertEqual(g.get(root.id).result, 40)

    def test_decomposition_is_unbounded(self) -> None:
        def descend(ctx):
            depth = ctx.args["depth"]
            if depth == 0:
                return Done(0)
            return Expand([Node(intent=f"depth {depth - 1}", fn="descend", args={"depth": depth - 1})])

        register(descend, name="descend")
        g = self.graph()
        root = g.add(Node(intent="descend", fn="descend", args={"depth": 6}))
        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(len(g), 7)
        self.assertEqual(g.get(root.id).result, [[[[[[0]]]]]])

    def test_expand_rejects_a_child_that_reuses_an_existing_id(self) -> None:
        g = self.graph()
        existing = g.add(Node(intent="existing", fn="noop"))
        register(lambda ctx: Expand([Node(intent="clash", id=existing.id)]), name="clash")
        register(lambda ctx: Done(None), name="noop")
        root = g.add(Node(intent="root", fn="clash"))

        run(g.run())
        self.assertEqual(g.get(root.id).state, "failed")
        self.assertIn("already exist", g.get(root.id).error or "")

    def test_expand_rejects_a_child_that_would_create_a_cycle(self) -> None:
        register(lambda ctx: Expand([Node(intent="child", needs=(ctx.node.id,))]), name="loop")
        g = self.graph()
        root = g.add(Node(intent="root", fn="loop"))
        run(g.run())
        self.assertEqual(g.get(root.id).state, "failed")
        self.assertIn("cycle", g.get(root.id).error or "")


class FailureTest(GraphTestCase):
    def test_a_raising_body_fails_only_its_own_branch(self) -> None:
        def boom(ctx):
            raise RuntimeError("no")

        register(boom, name="boom")
        register(lambda ctx: Done("fine"), name="fine")

        g = self.graph()
        bad = g.add(Node(intent="bad", fn="boom"))
        downstream = g.add(Node(intent="downstream", fn="fine", needs=(bad.id,)))
        sibling = g.add(Node(intent="sibling", fn="fine"))

        report = run(g.run())

        self.assertEqual(g.get(bad.id).state, "failed")
        self.assertIn("RuntimeError: no", g.get(bad.id).error or "")
        self.assertEqual(g.get(sibling.id).state, "done")
        self.assertEqual(g.get(downstream.id).state, "open")
        self.assertEqual(report.stopped, "blocked")
        self.assertEqual(report.blocked, (downstream.id,))

    def test_reject_keeps_the_branch_and_its_reason(self) -> None:
        register(lambda ctx: Reject("the API does not support it"), name="no")
        g = self.graph()
        node = g.add(Node(intent="try it", fn="no"))
        run(g.run())
        self.assertEqual(g.get(node.id).state, "rejected")
        self.assertEqual(g.get(node.id).reason, "the API does not support it")

    def test_an_unstorable_result_fails_the_node_not_the_graph(self) -> None:
        register(lambda ctx: Done(object()), name="unstorable")
        register(lambda ctx: Done("ok"), name="ok")
        g = self.graph()
        bad = g.add(Node(intent="bad", fn="unstorable"))
        good = g.add(Node(intent="good", fn="ok"))

        run(g.run())

        self.assertEqual(g.get(bad.id).state, "failed")
        self.assertIn("cannot be stored as JSON", g.get(bad.id).error or "")
        self.assertEqual(g.get(good.id).state, "done")
        self.assertEqual(Graph.open("test", store_dir=self.store_dir).get(good.id).result, "ok")

    def test_a_missing_body_fails_that_node_with_a_clear_message(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="ghost", fn="never_registered_body"))
        run(g.run())
        self.assertEqual(g.get(node.id).state, "failed")
        self.assertIn("never_registered_body", g.get(node.id).error or "")


class ValidationTest(GraphTestCase):
    def test_add_rejects_an_unknown_need(self) -> None:
        g = self.graph()
        with self.assertRaisesRegex(ValueError, "unknown nodes"):
            g.add(Node(intent="orphan", needs=("missing",)))

    def test_add_rejects_a_duplicate_id(self) -> None:
        g = self.graph()
        first = g.add(Node(intent="first"))
        with self.assertRaisesRegex(ValueError, "already exist"):
            g.add(Node(intent="second", id=first.id))

    def test_get_names_the_missing_node(self) -> None:
        g = self.graph()
        with self.assertRaisesRegex(KeyError, "nope"):
            g.get("nope")


class TerminationTest(GraphTestCase):
    def test_a_model_node_stops_the_run_as_pending_dispatch(self) -> None:
        register(lambda ctx: Done("done"), name="inline")
        g = self.graph()
        inline = g.add(Node(intent="inline", fn="inline"))
        model = g.add(Node(intent="think", prompt="decide something"))

        report = run(g.run())

        self.assertEqual(report.stopped, "pending_dispatch")
        self.assertEqual(report.pending_dispatch, (model.id,))
        self.assertEqual(g.get(inline.id).state, "done")

    def test_max_supersteps_is_a_safety_limit_not_completion(self) -> None:
        def chain(ctx):
            depth = ctx.args["depth"]
            if depth == 0:
                return Done(0)
            return Expand([Node(intent="next", fn="chain", args={"depth": depth - 1})])

        register(chain, name="chain")
        g = self.graph()
        g.add(Node(intent="chain", fn="chain", args={"depth": 50}))
        report = run(g.run(max_supersteps=3))
        self.assertEqual(report.stopped, "max_supersteps")
        self.assertEqual(report.supersteps, 3)
        self.assertFalse(report.complete)

    def test_max_nodes_stops_a_runaway_expansion(self) -> None:
        register(
            lambda ctx: Expand([Node(intent=f"child {i}", fn="wide", args={}) for i in range(4)]),
            name="wide",
        )
        g = self.graph()
        g.add(Node(intent="wide", fn="wide"))
        report = run(g.run(max_nodes=10))
        self.assertEqual(report.stopped, "max_nodes")

    def test_an_empty_graph_completes(self) -> None:
        report = run(self.graph().run())
        self.assertTrue(report.complete)
        self.assertEqual(report.executed, 0)


class PersistenceTest(GraphTestCase):
    def test_a_graph_reloads_and_continues(self) -> None:
        register(lambda ctx: Done("first"), name="first")
        register(lambda ctx: Done("second"), name="second")

        g = self.graph("resume")
        a = g.add(Node(intent="a", fn="first"))
        b = g.add(Node(intent="b", fn="second", needs=(a.id,), prompt=None))
        run(g.run())

        reloaded = Graph.open("resume", store_dir=self.store_dir)
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.get(a.id).result, "first")
        self.assertEqual(reloaded.get(b.id).result, "second")
        self.assertEqual(reloaded.name, "resume")

    def test_a_pending_model_node_survives_a_reload(self) -> None:
        g = self.graph("pending")
        model = g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        reloaded = Graph.open("pending", store_dir=self.store_dir)
        self.assertEqual(reloaded.get(model.id).prompt, "decide")
        self.assertEqual(reloaded.get(model.id).kind, "model")

    def test_the_store_file_is_readable_json(self) -> None:
        register(lambda ctx: Done(1), name="one")
        g = self.graph("readable")
        g.add(Node(intent="one", fn="one"))
        run(g.run())

        payload = json.loads(Path(graph_path("readable", self.store_dir)).read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], "readable")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(len(payload["nodes"]), 1)

    def test_an_unsupported_store_version_is_refused(self) -> None:
        path = Path(graph_path("versioned", self.store_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 999, "name": "versioned", "nodes": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported version"):
            Graph.open("versioned", store_dir=self.store_dir)

    def test_a_corrupt_store_is_refused(self) -> None:
        path = Path(graph_path("corrupt", self.store_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            Graph.open("corrupt", store_dir=self.store_dir)

    def test_a_stored_cycle_is_refused_on_load(self) -> None:
        path = Path(graph_path("cyclic", self.store_dir))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "name": "cyclic",
                    "nodes": [
                        {"id": "a", "intent": "a", "needs": ["b"]},
                        {"id": "b", "intent": "b", "needs": ["a"]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            Graph.open("cyclic", store_dir=self.store_dir)

    def test_the_lock_is_a_sidecar_file(self) -> None:
        store = GraphStore(Path(self.store_dir) / "x.json")
        with store.locked():
            store.write({"version": 1, "name": "x", "nodes": []})
        self.assertTrue(store.path.exists())
        self.assertTrue(store.lock_path.exists())
        self.assertNotEqual(store.path, store.lock_path)

    def test_slug_normalizes_a_name(self) -> None:
        self.assertEqual(slug("Ship The Release!"), "ship-the-release")
        with self.assertRaisesRegex(ValueError, "alphanumeric"):
            slug("///")


if __name__ == "__main__":
    unittest.main()
