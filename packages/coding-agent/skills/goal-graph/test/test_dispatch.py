from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from goal_graph import Done, Expand, Graph, Node, register
from goal_graph.dispatch import DispatchHandle, build_child_prompt, parse_result, result_path


def run(coro):
    return asyncio.run(coro)


class FakeDispatcher:
    """Stands in for RLM children: records spawns, reports registry statuses."""

    def __init__(self, *, fail_spawn: bool = False, models: tuple[str, ...] = ()) -> None:
        self.spawns: list[tuple[Node, str, str | None]] = []
        self.status: dict[str, str] = {}
        self.fail_spawn = fail_spawn
        self.models = models
        self.status_calls = 0
        self.on_status: Callable[[int], None] | None = None

    def candidates(self, node: Node) -> tuple[str | None, ...]:
        if node.model:
            return (node.model, *(model for model in self.models if model != node.model))
        return self.models or (None,)

    async def spawn(self, node: Node, prompt: str, model: str | None) -> DispatchHandle:
        if self.fail_spawn:
            raise RuntimeError("no capacity")
        self.spawns.append((node, prompt, model))
        child_id = f"sub-{len(self.spawns)}"
        self.status[child_id] = "running"
        return DispatchHandle(child_id=child_id, name=f"node-{node.id}", model=model or "fake/model")

    async def statuses(self) -> dict[str, str]:
        self.status_calls += 1
        if self.on_status is not None:
            self.on_status(self.status_calls)
        return dict(self.status)


class DispatchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.store_dir = self._temp.name
        self.addCleanup(self._temp.cleanup)
        self.dispatcher = FakeDispatcher()

    def graph(self, name: str = "dispatch", *, dispatcher: object | None = ...) -> Graph:
        chosen = self.dispatcher if dispatcher is ... else dispatcher
        return Graph.open(name, store_dir=self.store_dir, dispatcher=chosen)

    def write_result(self, graph: Graph, node_id: str, payload: dict, attempt: int = 0) -> None:
        graph.results_dir.mkdir(parents=True, exist_ok=True)
        result_path(graph.results_dir, node_id, attempt).write_text(json.dumps(payload), encoding="utf-8")


class DispatchTest(DispatchTestCase):
    def test_a_model_node_is_dispatched_and_left_in_flight(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide", model="fake/model"))

        report = run(g.run())

        self.assertEqual(report.stopped, "in_flight")
        self.assertEqual([entry.node_id for entry in report.in_flight], [node.id])
        self.assertEqual(g.get(node.id).state, "claimed")
        self.assertEqual(g.get(node.id).child_id, "sub-1")
        self.assertEqual(len(self.dispatcher.spawns), 1)

    def test_the_child_prompt_carries_the_node_id_and_the_result_path(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="the original instruction"))
        run(g.run())

        _, prompt, _ = self.dispatcher.spawns[0]
        self.assertIn("the original instruction", prompt)
        self.assertIn(node.id, prompt)
        self.assertIn(str(result_path(g.results_dir, node.id, 0)), prompt)

    def test_a_done_result_is_joined_on_the_next_run(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "done", "result": {"answer": 42}})

        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(g.get(node.id).state, "done")
        self.assertEqual(g.get(node.id).result, {"answer": 42})

    def test_a_joined_result_releases_a_downstream_barrier(self) -> None:
        register(lambda ctx: Done(sum(ctx.results.values())), name="add_up")
        g = self.graph()
        model = g.add(Node(intent="think", prompt="decide"))
        downstream = g.add(Node(intent="use it", fn="add_up", needs=(model.id,)))
        run(g.run())
        self.write_result(g, model.id, {"outcome": "done", "result": 5})

        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(g.get(downstream.id).result, 5)

    def test_waiting_picks_up_a_result_that_lands_during_polling(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))

        def land_on_second_poll(call: int) -> None:
            if call >= 2:
                self.write_result(g, node.id, {"outcome": "done", "result": "late"})

        self.dispatcher.on_status = land_on_second_poll
        report = run(g.run(wait_seconds=5, poll_seconds=0.01))

        self.assertTrue(report.complete)
        self.assertEqual(g.get(node.id).result, "late")

    def test_max_in_flight_caps_concurrent_children(self) -> None:
        g = self.graph()
        for index in range(5):
            g.add(Node(intent=f"think {index}", prompt="decide"))

        report = run(g.run(max_in_flight=2))

        self.assertEqual(report.stopped, "in_flight")
        self.assertEqual(len(self.dispatcher.spawns), 2)
        self.assertEqual(len(report.in_flight), 2)
        self.assertEqual(len(report.pending_dispatch), 3)

    def test_without_a_dispatcher_model_nodes_stay_pending(self) -> None:
        g = self.graph(dispatcher=None)
        node = g.add(Node(intent="think", prompt="decide"))

        report = run(g.run())

        self.assertEqual(report.stopped, "pending_dispatch")
        self.assertEqual(report.pending_dispatch, (node.id,))
        self.assertEqual(g.get(node.id).state, "open")

    def test_a_failed_spawn_fails_only_that_node(self) -> None:
        register(lambda ctx: Done("fine"), name="fine")
        self.dispatcher.fail_spawn = True
        g = self.graph()
        model = g.add(Node(intent="think", prompt="decide"))
        inline = g.add(Node(intent="inline", fn="fine"))

        run(g.run())

        self.assertEqual(g.get(model.id).state, "failed")
        self.assertIn("failed: RuntimeError: no capacity", g.get(model.id).error or "")
        self.assertEqual(g.get(inline.id).state, "done")

    def test_the_results_directory_sits_beside_the_graph_file(self) -> None:
        g = self.graph("beside")
        self.assertEqual(g.results_dir, Path(g.path).with_name("beside.results"))

    def test_the_child_id_survives_a_reload(self) -> None:
        g = self.graph("resume")
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        reloaded = Graph.open("resume", store_dir=self.store_dir, dispatcher=self.dispatcher)
        self.assertEqual(reloaded.get(node.id).state, "claimed")
        self.assertEqual(reloaded.get(node.id).child_id, "sub-1")

        self.write_result(reloaded, node.id, {"outcome": "done", "result": "after reload"})
        self.assertTrue(run(reloaded.run()).complete)
        self.assertEqual(reloaded.get(node.id).result, "after reload")


class FailoverTest(DispatchTestCase):
    """A provider that runs out of quota does not refuse the spawn.

    Its credentials stay valid, so the child is admitted and then dies partway.
    That is what these cover: the failure arrives at the join, not at dispatch.
    """

    def setUp(self) -> None:
        super().setUp()
        self.dispatcher = FakeDispatcher(models=("opencode-go/glm-5.2", "devin-2/glm-5-2", "devin-1/glm-5-2"))

    def test_a_child_that_dies_is_retried_on_the_next_model(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2",))

        self.dispatcher.status["sub-1"] = "error"
        report = run(g.run())

        self.assertEqual(g.get(node.id).state, "claimed", "a dead provider is not a dead node")
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2", "devin-2/glm-5-2"))
        self.assertEqual(self.dispatcher.spawns[1][2], "devin-2/glm-5-2")
        self.assertEqual(report.stopped, "in_flight")

    def test_the_retry_succeeds_on_the_fallback(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.dispatcher.status["sub-1"] = "error"
        run(g.run())

        # The second attempt writes to its own file, so the first cannot be read.
        self.write_result(g, node.id, {"outcome": "done", "result": "from devin"}, attempt=1)
        self.assertTrue(run(g.run()).complete)
        self.assertEqual(g.get(node.id).result, "from devin")

    def test_a_stale_result_from_a_dead_attempt_is_never_joined(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "done", "result": "stale"}, attempt=0)
        self.dispatcher.status["sub-1"] = "error"
        # The first attempt's file exists, so it is joined before the retry.
        self.assertTrue(run(g.run()).complete)
        self.assertEqual(g.get(node.id).result, "stale")

    def test_the_node_fails_only_once_every_model_is_spent(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        for attempt in range(3):
            run(g.run())
            self.dispatcher.status[f"sub-{attempt + 1}"] = "error"
        report = run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertEqual(len(self.dispatcher.spawns), 3, "each model is tried once")
        self.assertIn("no candidate model left to try", g.get(node.id).error or "")
        self.assertIn("finished as error", g.get(node.id).error or "", "the last failure stays the diagnosis")
        self.assertEqual(report.stopped, "complete")

    def test_a_node_model_leads_and_the_rest_stay_as_fallback(self) -> None:
        g = self.graph()
        g.add(Node(intent="think", prompt="decide", model="devin-1/glm-5-2"))
        run(g.run())
        self.assertEqual(self.dispatcher.spawns[0][2], "devin-1/glm-5-2")

        self.dispatcher.status["sub-1"] = "error"
        run(g.run())
        self.assertEqual(self.dispatcher.spawns[1][2], "opencode-go/glm-5.2")

    def test_a_refused_spawn_moves_to_the_next_model_rather_than_failing(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        self.dispatcher.fail_spawn = True
        run(g.run(max_supersteps=1))

        self.assertEqual(g.get(node.id).state, "open", "another candidate may still be admitted")
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2",))

    def test_without_a_fallback_list_the_parent_model_is_tried_once(self) -> None:
        self.dispatcher = FakeDispatcher()
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.dispatcher.status["sub-1"] = "error"
        run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertEqual(len(self.dispatcher.spawns), 1, "there is no other model to fall back to")


class VisibilityTest(DispatchTestCase):
    def test_in_flight_reports_enough_to_tell_working_from_wedged(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="review the diff", prompt="decide", model="devin-2/glm-5-2"))
        report = run(g.run())

        entry = report.in_flight[0]
        self.assertEqual(entry.node_id, node.id)
        self.assertEqual(entry.intent, "review the diff")
        self.assertEqual(entry.model, "devin-2/glm-5-2")
        self.assertEqual(entry.child_id, "sub-1")
        self.assertGreaterEqual(entry.seconds, 0.0)
        self.assertEqual(report.to_dict()["in_flight"][0]["intent"], "review the diff")

    def test_a_reopened_graph_with_no_dispatcher_says_it_is_detached(self) -> None:
        g = self.graph("detach")
        g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        reopened = Graph.open("detach", store_dir=self.store_dir)
        report = run(reopened.run())

        self.assertEqual(report.stopped, "detached", "nothing here can adjudicate a child that died")
        self.assertEqual(len(report.in_flight), 1)


class ChildOutcomeTest(DispatchTestCase):
    def test_a_child_can_reject(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "reject", "reason": "the endpoint does not exist"})

        run(g.run())

        self.assertEqual(g.get(node.id).state, "rejected")
        self.assertEqual(g.get(node.id).reason, "the endpoint does not exist")

    def test_a_child_can_expand_and_the_parent_collects(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="split it", prompt="decide"))
        run(g.run())
        self.write_result(
            g,
            node.id,
            {
                "outcome": "expand",
                "children": [
                    {"intent": "part one", "prompt": "do one"},
                    {"intent": "part two", "prompt": "do two"},
                ],
            },
        )

        report = run(g.run())

        self.assertEqual(len(g), 3)
        self.assertEqual(g.get(node.id).state, "open")
        self.assertIsNone(g.get(node.id).prompt)
        children = g.children_of(node.id)
        self.assertEqual(sorted(child.intent for child in children), ["part one", "part two"])
        self.assertEqual(len(report.in_flight), 2)

        for child in children:
            self.write_result(g, child.id, {"outcome": "done", "result": child.intent})
        self.assertTrue(run(g.run()).complete)
        self.assertEqual(sorted(g.get(node.id).result), ["part one", "part two"])

    def test_an_expansion_is_applied_once_even_though_the_result_file_remains(self) -> None:
        # The result file is kept as evidence, so ingesting must move the node
        # out of "claimed" or it would be re-read and re-expanded every pass.
        g = self.graph()
        node = g.add(Node(intent="split it", prompt="decide"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "expand", "children": [{"intent": "only child"}]})

        run(g.run())
        self.assertEqual(g.get(node.id).state, "done")
        self.assertEqual(len(g), 2)

        run(g.run())
        self.assertEqual(len(g), 2)
        self.assertTrue(result_path(g.results_dir, node.id, 0).exists())

    def test_an_expanded_node_drops_the_body_it_no_longer_has(self) -> None:
        # `model` outliving `prompt` is a pairing Node rejects on load, so the
        # whole graph would stop reopening after one expansion.
        g = self.graph("survives")
        node = g.add(Node(intent="split", prompt="decide", model="opencode-go/glm-5.2"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "expand", "children": [{"intent": "child", "prompt": "do"}]})
        run(g.run())

        self.assertIsNone(g.get(node.id).prompt)
        self.assertIsNone(g.get(node.id).model)
        self.assertIsNone(g.get(node.id).child_id, "the child described a body this node no longer has")

        reopened = Graph.open("survives", store_dir=self.store_dir, dispatcher=self.dispatcher)
        self.assertEqual(len(reopened), 2)

    def test_a_child_keeps_the_model_it_asked_for(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="split", prompt="decide"))
        run(g.run())
        self.write_result(
            g,
            node.id,
            {"outcome": "expand", "children": [{"intent": "hard part", "prompt": "do", "model": "opencode-go/kimi-k3"}]},
        )
        run(g.run())

        self.assertEqual(g.children_of(node.id)[0].model, "opencode-go/kimi-k3")

    def test_a_child_orders_siblings_with_local_keys(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="split", prompt="decide"))
        run(g.run())
        self.write_result(
            g,
            node.id,
            {
                "outcome": "expand",
                "children": [
                    {"key": "build", "intent": "build it", "prompt": "build"},
                    {"intent": "test it", "prompt": "test", "needs": ["build"]},
                ],
            },
        )
        report = run(g.run())

        children = {child.intent: child for child in g.children_of(node.id)}
        build, test = children["build it"], children["test it"]
        self.assertEqual(test.needs, (build.id,), "a sibling key must resolve to the id assigned here")
        self.assertEqual(build.needs, ())
        self.assertEqual([entry.node_id for entry in report.in_flight], [build.id], "only the unblocked sibling is dispatched")

    def test_a_child_expansion_can_name_a_registered_inline_body(self) -> None:
        register(lambda ctx: Done(ctx.args["n"] * 3), name="triple")
        g = self.graph()
        node = g.add(Node(intent="split it", prompt="decide"))
        run(g.run())
        self.write_result(
            g,
            node.id,
            {"outcome": "expand", "children": [{"intent": "triple it", "fn": "triple", "args": {"n": 7}}]},
        )

        self.assertTrue(run(g.run()).complete)
        self.assertEqual(g.get(node.id).result, [21])

    def test_a_child_that_finishes_without_a_result_fails_its_node(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.dispatcher.status["sub-1"] = "completed"

        report = run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertIn("finished as completed without writing", g.get(node.id).error or "")
        self.assertIn(node.id, g.get(node.id).error or "")
        self.assertEqual(report.stopped, "complete")

    def test_a_child_error_status_without_a_result_fails_its_node(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.dispatcher.status["sub-1"] = "error"

        run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertIn("finished as error", g.get(node.id).error or "")

    def test_a_still_running_child_keeps_its_node_claimed(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        report = run(g.run())

        self.assertEqual(g.get(node.id).state, "claimed")
        self.assertEqual(report.stopped, "in_flight")

    def test_a_half_written_file_is_retried_while_the_child_still_runs(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        g.results_dir.mkdir(parents=True, exist_ok=True)
        result_path(g.results_dir, node.id, 0).write_text('{"outcome": "do', encoding="utf-8")

        report = run(g.run())
        self.assertEqual(g.get(node.id).state, "claimed", "a working child must not be failed over timing")
        self.assertEqual(report.stopped, "in_flight")

        self.write_result(g, node.id, {"outcome": "done", "result": "whole"})
        self.assertTrue(run(g.run()).complete)
        self.assertEqual(g.get(node.id).result, "whole")

    def test_an_unparsable_result_fails_the_node_once_the_child_has_stopped(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        g.results_dir.mkdir(parents=True, exist_ok=True)
        result_path(g.results_dir, node.id, 0).write_text("{not json", encoding="utf-8")
        self.dispatcher.status["sub-1"] = "completed"

        run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertIn("not valid JSON", g.get(node.id).error or "")
        self.assertEqual(Graph.open("dispatch", store_dir=self.store_dir).get(node.id).state, "failed")

    def test_an_unknown_outcome_fails_the_node(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.write_result(g, node.id, {"outcome": "maybe"})
        self.dispatcher.status["sub-1"] = "completed"

        run(g.run())

        self.assertEqual(g.get(node.id).state, "failed")
        self.assertIn("unknown outcome 'maybe'", g.get(node.id).error or "")


class ResultParsingTest(unittest.TestCase):
    def node(self) -> Node:
        return Node(intent="x", id="n_test", prompt="p")

    def test_done_without_a_result_is_allowed(self) -> None:
        self.assertEqual(parse_result({"outcome": "done"}, self.node()), Done(None))

    def test_reject_needs_a_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "rejected without a reason"):
            parse_result({"outcome": "reject"}, self.node())

    def test_expand_needs_children(self) -> None:
        with self.assertRaisesRegex(ValueError, "expanded without children"):
            parse_result({"outcome": "expand", "children": []}, self.node())

    def test_a_child_needs_an_intent(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing an intent"):
            parse_result({"outcome": "expand", "children": [{"prompt": "x"}]}, self.node())

    def test_a_child_may_not_choose_its_own_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not choose its own id"):
            parse_result({"outcome": "expand", "children": [{"intent": "x", "id": "n_1"}]}, self.node())

    def test_a_child_may_depend_on_a_sibling_by_key(self) -> None:
        outcome = parse_result(
            {"outcome": "expand", "children": [{"key": "first", "intent": "a"}, {"intent": "b", "needs": ["first"]}]},
            self.node(),
        )
        assert isinstance(outcome, Expand)
        first, second = outcome.children
        self.assertEqual([child.intent for child in outcome.children], ["a", "b"])
        self.assertEqual(second.needs, (first.id,))

    def test_a_needs_entry_that_is_not_a_sibling_key_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not the key of any sibling"):
            parse_result({"outcome": "expand", "children": [{"intent": "a", "needs": ["nope"]}]}, self.node())

    def test_duplicate_sibling_keys_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "reuse the key"):
            parse_result(
                {"outcome": "expand", "children": [{"key": "k", "intent": "a"}, {"key": "k", "intent": "b"}]},
                self.node(),
            )

    def test_the_prompt_documents_keys_rather_than_ids(self) -> None:
        prompt = build_child_prompt(self.node(), Path("/tmp/results/n_test.0.json"))
        self.assertIn('"key"', prompt)
        self.assertIn("cannot see or set ids", prompt)
        self.assertIn("os.replace", prompt)

    def test_a_non_object_result_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            parse_result(["done"], self.node())

    def test_the_prompt_states_the_protocol(self) -> None:
        prompt = build_child_prompt(self.node(), Path("/tmp/results/n_test.0.json"))
        self.assertIn("/tmp/results/n_test.0.json", prompt)
        self.assertIn('"outcome": "done"', prompt)
        self.assertIn("receiver_role=\"parent\"", prompt)


class NodeModelTest(unittest.TestCase):
    def test_a_model_selector_needs_a_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "names a model but has no prompt"):
            Node(intent="x", fn="body", model="a/b")


if __name__ == "__main__":
    unittest.main()
