from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from goal_graph import Done, Expand, Graph, Node, register
from goal_graph.dispatch import (
    DispatchHandle,
    build_child_prompt,
    parse_result,
    read_result,
    result_path,
)


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


class JoinRobustnessTest(DispatchTestCase):
    """Defects that escaped the join's ValueError-only handling and either
    crashed the whole run or discarded finished work."""

    def test_a_non_dict_args_in_an_expansion_fails_only_that_node(self) -> None:
        # `dict(5)` raises TypeError, not ValueError, so before the fix one
        # badly formed child answer aborted the entire run instead of failing
        # just that node.
        register(lambda ctx: Done("fine"), name="survives")
        g = self.graph()
        model = g.add(Node(intent="split", prompt="decide"))
        inline = g.add(Node(intent="inline", fn="survives"))
        run(g.run())
        self.write_result(
            g,
            model.id,
            {"outcome": "expand", "children": [{"intent": "bad", "prompt": "x", "args": 5}]},
        )
        self.dispatcher.status["sub-1"] = "completed"

        run(g.run())

        self.assertEqual(g.get(model.id).state, "failed")
        self.assertIn("args must be an object", g.get(model.id).error or "")
        self.assertEqual(g.get(inline.id).state, "done", "the run survived the malformed child")

    def test_read_result_converts_a_vanished_file_to_a_value_error(self) -> None:
        # A result file that existed at the probe and vanished before the read
        # raises OSError, which would escape the join. It is surfaced as a
        # ValueError so the node fails or reopens rather than the run crashing.
        with self.assertRaisesRegex(ValueError, "could not be read"):
            read_result(Path(self.store_dir) / "does-not-exist.json")

    def test_an_answer_landing_between_the_probe_and_the_status_check_is_joined(self) -> None:
        # The result file is probed before the registry snapshot, so an answer
        # that lands in that gap used to be treated as missing and the completed
        # work discarded. The join now re-tests the file after a terminal status.
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        def land_during_status(call: int) -> None:
            self.write_result(g, node.id, {"outcome": "done", "result": "landed"})
            self.dispatcher.status["sub-1"] = "completed"

        self.dispatcher.on_status = land_during_status
        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(g.get(node.id).state, "done")
        self.assertEqual(g.get(node.id).result, "landed")


class StaleClaimTest(DispatchTestCase):
    """A parent that compacts or restarts finds its old child absent from the
    session-scoped registry. Without a timeout the node stays claimed forever."""

    def setUp(self) -> None:
        super().setUp()
        self.dispatcher = FakeDispatcher(models=("opencode-go/glm-5.2", "devin-2/glm-5-2"))

    def test_an_absent_child_past_the_timeout_reopens_for_the_next_model(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2",))

        # Simulate a restart: the registry no longer reports the old child, and
        # the claim is older than the timeout.
        self.dispatcher.status.clear()
        g.get(node.id).claimed_at = 0.0

        report = run(g.run(claim_timeout_seconds=1))

        self.assertEqual(
            g.get(node.id).state,
            "claimed",
            "an absent child past the timeout is treated as a dead attempt and retried",
        )
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2", "devin-2/glm-5-2"))
        self.assertEqual(self.dispatcher.spawns[1][2], "devin-2/glm-5-2")
        self.assertEqual(report.stopped, "in_flight")

    def test_an_absent_child_within_the_timeout_stays_claimed(self) -> None:
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())

        self.dispatcher.status.clear()
        # claimed_at is fresh, so the timeout has not elapsed.

        report = run(g.run(claim_timeout_seconds=3600))

        self.assertEqual(g.get(node.id).state, "claimed")
        self.assertEqual(report.stopped, "in_flight")


class StaleMetadataTest(DispatchTestCase):
    """A node that failed over before succeeding carried the dead attempt's
    metadata. It is cleared once the work succeeds so a checkpoint does not
    read as though the node failed or is still mid-dispatch."""

    def test_a_done_node_drops_a_prior_attempt_s_error(self) -> None:
        self.dispatcher = FakeDispatcher(models=("opencode-go/glm-5.2", "devin-2/glm-5-2"))
        g = self.graph()
        node = g.add(Node(intent="think", prompt="decide"))
        run(g.run())
        # First attempt dies, setting node.error and reopening the node.
        self.dispatcher.status["sub-1"] = "error"
        run(g.run())
        self.assertIsNotNone(g.get(node.id).error)

        # Second attempt succeeds.
        self.write_result(g, node.id, {"outcome": "done", "result": "ok"}, attempt=1)
        report = run(g.run())

        self.assertTrue(report.complete)
        self.assertEqual(g.get(node.id).state, "done")
        self.assertIsNone(g.get(node.id).error, "a succeeded node carries no failure message")

    def test_an_expansion_clears_the_old_body_s_claim_metadata(self) -> None:
        self.dispatcher = FakeDispatcher(models=("opencode-go/glm-5.2",))
        g = self.graph()
        node = g.add(Node(intent="split", prompt="decide", model="opencode-go/glm-5.2"))
        run(g.run())
        self.assertEqual(g.get(node.id).tried_models, ("opencode-go/glm-5.2",))
        self.assertIsNotNone(g.get(node.id).claimed_at)

        self.write_result(g, node.id, {"outcome": "expand", "children": [{"intent": "child", "prompt": "do"}]})
        run(g.run())

        self.assertIsNone(g.get(node.id).claimed_at, "the old attempt's start is stale on a collector node")
        self.assertEqual(g.get(node.id).tried_models, (), "the old body's attempts do not describe the new body")
        self.assertIsNone(g.get(node.id).error)
        self.assertIsNone(g.get(node.id).child_id)


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


class RoutingDispatcher(FakeDispatcher):
    """A dispatcher that knows a child moved model mid-session."""

    def __init__(self, moved: dict[str, str] | None = None, *, raises: bool = False, answer=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.moved = moved or {}
        self.raises = raises
        self.answer = answer
        self.asked: list[tuple[str, str | None]] = []

    def ran_on(self, session_name: str, dispatched: str | None) -> str | None:
        self.asked.append((session_name, dispatched))
        if self.raises:
            raise RuntimeError("failover log unreadable")
        if self.answer is not None:
            return self.answer
        return self.moved.get(session_name, dispatched)


class RanOnTest(DispatchTestCase):
    """A node must report the model that actually did the work.

    The parent spawns `devin-2` and gets a result back, so it records `devin-2`
    even when the child was moved onto another provider inside its own session.
    That record reads as evidence while being false.
    """

    def dispatch(self, dispatcher: FakeDispatcher) -> tuple[Graph, Node]:
        self.dispatcher = dispatcher
        graph = self.graph()
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        return graph, node

    def test_a_plain_dispatcher_reports_the_dispatched_model(self) -> None:
        # RlmDispatcher knows nothing about routing, and the dispatched model is
        # then the truth as far as anything here knows.
        graph, node = self.dispatch(FakeDispatcher(models=("devin-2/glm-5-2",)))
        self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
        self.dispatcher.status["sub-1"] = "completed"
        run(graph.run())
        self.assertEqual(graph.get(node.id).ran_on, "devin-2/glm-5-2")

    def test_a_switch_inside_the_child_is_recorded(self) -> None:
        dispatcher = RoutingDispatcher({"node-": "cursor/auto"}, models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        dispatcher.moved[f"node-{node.id}-0"] = "cursor/auto"
        self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
        dispatcher.status["sub-1"] = "completed"
        run(graph.run())

        resolved = graph.get(node.id)
        self.assertEqual(resolved.ran_on, "cursor/auto")
        # The dispatch record is untouched: both facts matter, and one is not a
        # correction of the other.
        self.assertEqual(resolved.tried_models, ("devin-2/glm-5-2",))

    def test_correlation_uses_the_child_name_the_graph_assigned(self) -> None:
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
        dispatcher.status["sub-1"] = "completed"
        run(graph.run())
        self.assertIn((f"node-{node.id}-0", "devin-2/glm-5-2"), dispatcher.asked)

    def test_a_router_that_cannot_read_its_log_does_not_fail_the_node(self) -> None:
        # Attribution is a reporting detail. Losing it must not discard work
        # that already succeeded.
        dispatcher = RoutingDispatcher(raises=True, models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
        dispatcher.status["sub-1"] = "completed"
        run(graph.run())

        resolved = graph.get(node.id)
        self.assertEqual(resolved.state, "done")
        self.assertEqual(resolved.ran_on, "devin-2/glm-5-2")

    def test_a_nonsense_answer_falls_back_to_the_dispatched_model(self) -> None:
        for answer in ("", "   ", 7):
            with self.subTest(answer=answer):
                dispatcher = RoutingDispatcher(answer=answer, models=("devin-2/glm-5-2",))
                graph = self.graph(f"nonsense-{answer!r}", dispatcher=dispatcher)
                node = graph.add(Node(intent="patch", prompt="fix it"))
                run(graph.run())
                self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
                dispatcher.status["sub-1"] = "completed"
                run(graph.run())
                self.assertEqual(graph.get(node.id).ran_on, "devin-2/glm-5-2")

    def test_a_reopened_node_drops_the_old_attribution(self) -> None:
        # The next attempt runs somewhere else, so keeping it would name a model
        # that did not produce the result the node ends up with.
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2", "devin-1/glm-5-2"))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        dispatcher.status["sub-1"] = "error"
        run(graph.run())

        reopened = graph.get(node.id)
        self.assertIsNone(reopened.ran_on)
        self.assertEqual(reopened.tried_models, ("devin-2/glm-5-2", "devin-1/glm-5-2"))

    def test_a_failed_child_is_still_attributed(self) -> None:
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        dispatcher.moved[f"node-{node.id}-0"] = "cursor/auto"
        self.write_result(graph, node.id, {"outcome": "nonsense"})
        dispatcher.status["sub-1"] = "completed"
        run(graph.run())

        resolved = graph.get(node.id)
        self.assertEqual(resolved.state, "failed")
        self.assertEqual(resolved.ran_on, "cursor/auto")

    def test_ran_on_survives_a_checkpoint(self) -> None:
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        dispatcher.moved[f"node-{node.id}-0"] = "cursor/auto"
        self.write_result(graph, node.id, {"outcome": "done", "result": "ok"})
        dispatcher.status["sub-1"] = "completed"
        run(graph.run())

        self.assertEqual(self.graph(dispatcher=dispatcher).get(node.id).ran_on, "cursor/auto")


class InFlightReportTest(DispatchTestCase):
    def test_a_live_switch_is_visible_before_the_child_finishes(self) -> None:
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        node = graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())
        dispatcher.moved[f"node-{node.id}-0"] = "cursor/auto"

        entry = run(graph.run()).in_flight[0]
        self.assertEqual(entry.model, "devin-2/glm-5-2")
        self.assertEqual(entry.ran_on, "cursor/auto")

    def test_a_child_that_has_not_moved_reports_no_switch(self) -> None:
        # A value present in the report always means something moved, so a
        # reader never has to compare two fields to find out.
        dispatcher = RoutingDispatcher(models=("devin-2/glm-5-2",))
        graph = self.graph(dispatcher=dispatcher)
        graph.add(Node(intent="patch", prompt="fix it"))
        run(graph.run())

        entry = run(graph.run()).in_flight[0]
        self.assertEqual(entry.model, "devin-2/glm-5-2")
        self.assertIsNone(entry.ran_on)
        self.assertIsNone(entry.to_dict()["ran_on"])


if __name__ == "__main__":
    unittest.main()
