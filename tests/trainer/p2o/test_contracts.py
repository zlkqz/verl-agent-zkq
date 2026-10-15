"""Contract acceptance only: no real adapters, GEPA, model generation or training.

The fake evaluator below never examines an environment name. The fixture native
fields are for reviewer context; adapters in TODO-02 must extract them before use.
"""

import copy
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from agent_system.p2o.base import P2OEnvAdapter, Stateful
from verl.trainer.p2o.contracts import (
    ActionRequest,
    AdapterCapabilities,
    ArtifactRef,
    ChatMessage,
    ComponentState,
    ContextPair,
    ContractError,
    EpisodeIdentity,
    EpisodeOutcome,
    EpisodeTrace,
    EvalInstance,
    EvalRequest,
    EvalResult,
    EvaluationPanel,
    FeedbackRecord,
    GeneratedResponse,
    GroupEvidence,
    HardSampleRecord,
    IdentityError,
    IncompleteEpisodeError,
    LatestSolved,
    P2OState,
    PolicyCapabilities,
    PromptSnapshot,
    ResetBatch,
    ResetEntry,
    ResetRequest,
    SampleKey,
    SchemaVersionError,
    SolvingEvidence,
    TaskRecipe,
    TemplateBinding,
    TemplateCycle,
    TemplateRuleContext,
    TokenAlignmentError,
    TokenScores,
    Transition,
    TransitionBatch,
    TurnIdentity,
    UnsupportedCapabilityError,
    outcome_from_trace,
    stable_hash,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures/traces.json").read_text())


def recipe(fixture=0):
    data = FIXTURES[fixture]
    return TaskRecipe(SampleKey(**data["key"]), data["selector"], "fixture-data-v1", "fixture-code-v1", {}, (), data["initial_observation"], {})


def instance(task=None, seed=0):
    task = task or recipe()
    return EvalInstance(task.sample_key, task.recipe_id, f"{task.sample_key.env}-instance-{seed}", {"fixture_seed": seed}, task.initial_observation)


def episode(task=None, template=None, phase="base", traj="t0", group="g0"):
    task = task or recipe()
    return EpisodeIdentity(task.sample_key, instance(task).instance_id, group, traj, "episode-" + traj, "policy-v1", phase, template,
                           parent_group_uid="parent-g0" if phase == "rescue" else None)


def transition(identity=None, **kwargs):
    data = copy.deepcopy(FIXTURES[0]["transitions"][0])
    data["executed_action"] = data.pop("action")
    data.update(kwargs)
    return Transition(identity or TurnIdentity(episode(), 0, 0, "request-0", "attempt-0"), **data)


def outcome(identity=None, success=True):
    return EpisodeOutcome(identity or episode(), success, None, 10.0 if success else 0.0, 0.0,
                          "success" if success else "environment_done", True, 1)


class FakeAdapter:
    """Mechanical protocol fixture, not an ALFWorld/WebShop implementation."""

    capabilities = AdapterCapabilities(subset_reset=True, subset_step=True, terminal_snapshot=True, replay=True)

    def __init__(self, fixture):
        self.fixture = copy.deepcopy(fixture)
        self.slots = {}
        self.closed = False

    def capture_task(self, slot, reset_observation, reset_info):
        return TaskRecipe(SampleKey(**self.fixture["key"]), self.fixture["selector"], "fixture-data-v1", "fixture-code-v1", {}, (), reset_observation, {})

    def capture_instance(self, slot):
        if self.slots[slot]["cursor"] != 0:
            raise IdentityError("capture_instance must precede the first action")
        return self.slots[slot]["request"].instance

    def make_instances(self, recipes, seed_schedule, mode, replay_instances=()):
        self.capabilities.require(mode)
        if len(recipes) != len(seed_schedule) or len(recipes) != len(replay_instances):
            raise IdentityError("replay input lengths disagree")
        for task, saved in zip(recipes, replay_instances):
            if (task.sample_key, task.recipe_id) != (saved.sample_key, saved.recipe_id):
                raise IdentityError("replay task/recipe mismatch")
        return copy.deepcopy(replay_instances)

    def reset_instances(self, requests):
        self.capabilities.require("subset_reset")
        if len({row.slot for row in requests}) != len(requests):
            raise IdentityError("duplicate reset slots")
        entries = []
        for row in requests:
            self.slots[row.slot] = {"cursor": 0, "request": row, "closed": False}
            entries.append(ResetEntry(row, self.fixture["initial_observation"], self.fixture["available_actions"], {}))
        result = ResetBatch(tuple(reversed(entries)))  # Exercise completion reordering.
        result.assert_requests(requests)
        return result

    def step(self, requests):
        self.capabilities.require("subset_step")
        rows = []
        for request in requests:
            state = self.slots[request.identity.slot]
            if state["closed"]:
                raise IdentityError("step after terminal")
            saved = state["request"]
            identity = request.identity.episode
            if (identity.episode_id, identity.instance_id, identity.sample_key) != (saved.episode_id, saved.instance.instance_id, saved.instance.sample_key):
                raise IdentityError("step for a different initialized instance")
            raw = copy.deepcopy(self.fixture["transitions"][state["cursor"]])
            if request.action != raw.pop("action"):
                raise IdentityError("fixture action mismatch")
            row = Transition(request.identity, executed_action=request.action, **raw)
            state["cursor"] += 1
            state["closed"] = row.done or row.truncated or not row.valid
            rows.append(row)
        batch = TransitionBatch(tuple(reversed(rows)))
        batch.assert_requests(requests)
        return batch

    def episode_outcome(self, trace):
        return outcome_from_trace(trace)

    def feedback(self, trace, token_budget):
        return FeedbackRecord(trace.identity, self.episode_outcome(trace), trace.initial_observation, (), {}, "fixture", 1, token_budget, "fixture-v1")

    def template_constraints(self, recipes, traces):
        return TemplateRuleContext(tuple(task.sample_key for task in recipes), "fixture-v1", (), {})

    def state_dict(self):
        return ComponentState("fake-adapter", 1, {"closed": self.closed})

    def load_state_dict(self, state):
        if state.component != "fake-adapter" or state.version != 1:
            raise SchemaVersionError("fake adapter state version mismatch")
        self.closed = state.payload["closed"]

    def close(self):
        self.closed = True


def run_fake_episode(adapter, task):
    """Shared evaluator-shaped harness: only calls the public adapter protocol."""
    identity = episode(task)
    request = ResetRequest(3, "reset-0", identity.episode_id, instance(task))
    reset = adapter.reset_instances((request,))
    reset.assert_requests((request,))
    saved = adapter.capture_instance(3)
    rows = []
    actions = reset.entries[0].available_actions
    for turn in range(10):
        request = ActionRequest(TurnIdentity(identity, turn, 3, f"step-{turn}", "attempt-0"), actions[0])
        batch = adapter.step((request,))
        batch.assert_requests((request,))
        row = batch.transitions[0]
        rows.append(row)
        if row.done or row.truncated or not row.valid:
            trace = EpisodeTrace(identity, saved.initial_observation, tuple(rows))
            return trace, adapter.episode_outcome(trace)
        actions = row.available_actions
    raise IncompleteEpisodeError("fixture exceeded evaluation budget")


class ContractTests(unittest.TestCase):
    def roundtrip(self, value):
        state = json.loads(json.dumps(value.state_dict(), allow_nan=False))
        restored = type(value).from_state_dict(state)
        self.assertEqual(restored, value)
        return restored

    def test_two_environment_examples_use_one_evaluator_and_outcome_schema(self):
        outputs = []
        for fixture in FIXTURES:
            with self.subTest(env=fixture["key"]["env"]):
                task = TaskRecipe(SampleKey(**fixture["key"]), fixture["selector"], "fixture-data-v1", "fixture-code-v1", {}, (), fixture["initial_observation"], {})
                adapter = FakeAdapter(fixture)
                self.assertIsInstance(adapter, P2OEnvAdapter)
                self.assertIsInstance(adapter, Stateful)
                trace, result = run_fake_episode(adapter, task)
                self.roundtrip(trace)
                self.roundtrip(result)
                self.roundtrip(adapter.feedback(trace, 10))
                self.roundtrip(adapter.template_constraints((task,), (trace,)))
                self.assertEqual(adapter.capture_task(3, fixture["initial_observation"], {}).sample_key, task.sample_key)
                with self.assertRaises(IdentityError):
                    adapter.capture_instance(3)
                outputs.append((result.success, result.task_return, result.task_score, result.steps))
                adapter.close()
        self.assertEqual(outputs, [(True, 10.0, None, 1), (False, 0.0, 0.4, 2)])

    def test_required_schema_version_and_fields_fail_closed(self):
        original = outcome().state_dict()
        for mutation in (lambda s: s.update(schema_version=2), lambda s: s.update(schema_version=True),
                         lambda s: s.update(record_type="Other"), lambda s: s["payload"].pop("success"),
                         lambda s: s["payload"].update(success="false"), lambda s: s["payload"].update(extra=1),
                         lambda s: s["payload"]["identity"]["payload"].pop("instance_id")):
            with self.subTest(mutation=mutation):
                state = copy.deepcopy(original)
                mutation(state)
                with self.assertRaises(ContractError):
                    EpisodeOutcome.from_state_dict(state)

    def test_nested_inputs_and_exported_state_are_owned_snapshots(self):
        config = {"world": {"prices": [1, 2]}}
        task = replace(recipe(), env_config=config)
        before = task.recipe_id
        config["world"]["prices"][0] = 99
        exported = task.state_dict()
        exported["payload"]["env_config"]["world"]["prices"][0] = 88
        self.assertEqual(task.env_config["world"]["prices"], [1, 2])
        self.assertEqual(task.recipe_id, before)
        self.roundtrip(task)

    def test_identity_ignores_instance_randomness_but_not_task_or_catalog(self):
        for i in range(2):
            key = recipe(i).sample_key
            a = instance(recipe(i), 1)
            b = replace(instance(recipe(i), 2), initialization={"fixture_seed": 2, "price": 42, "worker": 7})
            self.assertEqual(a.sample_key.canonical, b.sample_key.canonical)
            self.assertNotEqual(a.instance_id, b.instance_id)
            self.assertNotEqual(key.canonical, replace(key, task_id=key.task_id + "-other").canonical)
            self.assertNotEqual(key.canonical, replace(key, task_catalog_version="other").canonical)
            self.roundtrip(key)
        # Delimiters in different fields cannot collide.
        self.assertNotEqual(SampleKey("a/b", "c", "v", "id").canonical, SampleKey("a", "b/c", "v", "id").canonical)
        with self.assertRaises(IdentityError):
            replace(recipe().sample_key, task_id="")
        with self.assertRaises(ContractError):
            replace(recipe().sample_key, task_id=500)

    def test_train_dev_split_rejects_same_task_across_random_instances(self):
        with self.assertRaises(IdentityError):
            EvaluationPanel("panel", "policy-v1", (instance(seed=1),), (instance(seed=2),))
        self.roundtrip(EvaluationPanel("panel", "policy-v1", (instance(recipe(0)),), (instance(recipe(1)),)))
        with self.assertRaises(IdentityError):
            EvaluationPanel("panel", "policy-v1", (), (instance(seed=1), instance(seed=2)))

    def test_capabilities_are_explicit_and_no_implicit_downgrade(self):
        capabilities = AdapterCapabilities(subset_step=True)
        capabilities.require("subset_step")
        for name in ("subset_reset", "resample", "unknown"):
            with self.assertRaises(UnsupportedCapabilityError):
                capabilities.require(name)
        with self.assertRaises(UnsupportedCapabilityError):
            PolicyCapabilities(raw_response_tokens=True).require("aligned_rollout_log_probs")
        self.roundtrip(capabilities)

    def test_subset_reset_step_reordering_and_inactive_state(self):
        adapter = FakeAdapter(FIXTURES[0])
        task = recipe()
        requests = tuple(ResetRequest(slot, f"reset-{slot}", f"episode-t{slot}", instance(task)) for slot in (1, 5))
        batch = adapter.reset_instances(requests)
        self.assertEqual(tuple(row.request.slot for row in batch.entries), (5, 1))
        untouched = copy.deepcopy(adapter.slots[5])
        identity = replace(episode(traj="t1"), episode_id="episode-t1")
        step = ActionRequest(TurnIdentity(identity, 0, 1, "step-1", "attempt-0"), FIXTURES[0]["transitions"][0]["action"])
        result = adapter.step((step,))
        self.assertEqual(adapter.slots[5], untouched)
        self.roundtrip(result)
        with self.assertRaises(IdentityError):
            result.assert_requests((replace(step, identity=replace(step.identity, slot=5)),))
        with self.assertRaises(IdentityError):
            replace(result, transitions=result.transitions * 2)
        with self.assertRaises(IdentityError):
            ResetBatch(batch.entries[:1]).assert_requests(requests)
        with self.assertRaises(IdentityError):
            replace(step, identity=replace(step.identity, is_padding=True))
        replay = adapter.make_instances((task,), (0,), "replay", (instance(task),))
        self.assertEqual(replay, (instance(task),))
        with self.assertRaises(UnsupportedCapabilityError):
            adapter.make_instances((task,), (1,), "resample")

    def test_task_failure_truncation_infrastructure_and_penalty_stay_distinct(self):
        identity = episode()
        success = transition(invalid_action_penalty=-0.1)
        result = outcome_from_trace(EpisodeTrace(identity, "start", (success,)))
        self.assertTrue(result.success)
        self.assertEqual((result.task_return, result.invalid_action_penalty), (10, -0.1))
        truncated = transition(raw_reward=0, done=False, truncated=True, success=False, termination_reason="max_steps")
        self.assertFalse(outcome_from_trace(EpisodeTrace(identity, "start", (truncated,))).success)
        infra = transition(done=False, success=None, valid=False, termination_reason="infra_error", terminal_observation=None)
        result = outcome_from_trace(EpisodeTrace(identity, "start", (infra,)))
        self.assertIsNone(result.success)
        self.assertFalse(result.valid)
        self.roundtrip(result)
        for value in (lambda: replace(infra, success=False), lambda: replace(success, success=None),
                      lambda: replace(success, terminal_observation=None), lambda: replace(outcome(), valid=False)):
            with self.assertRaises(ContractError):
                value()
        live = transition(done=False, termination_reason=None, terminal_observation=None, success=False)
        for rows in ((), (live,)):
            with self.assertRaises(IncompleteEpisodeError):
                outcome_from_trace(EpisodeTrace(identity, "start", rows))

    def test_trace_rejects_late_auto_reset_rows_and_wrong_identity(self):
        first = transition()
        with self.assertRaises(ContractError):
            EpisodeTrace(episode(), "start", (first, replace(first, identity=replace(first.identity, turn_index=1, request_id="late"))))
        with self.assertRaises(IdentityError):
            EpisodeTrace(episode(), "start", (replace(first, identity=replace(first.identity, episode=replace(episode(), instance_id="other"))),))

    def response(self):
        template = TemplateBinding("handwritten", "Check current actions.")
        messages = (ChatMessage("user", "task"),)
        base = PromptSnapshot(messages, (0, 1, 2), (0, 1, 1), (0, 0, 1), "tokenizer-v1")
        aug = replace(base, messages=messages + (ChatMessage("user", template.text),), input_ids=(1, 2, 3), attention_mask=(1, 1, 1), position_ids=(0, 1, 2))
        identity = TurnIdentity(episode(template=template), 0, 0, "request-0", "attempt-0")
        # EOS==pad is intentional: only the saved mask distinguishes the last 2 ids.
        return GeneratedResponse("row-0", identity, ContextPair(base, aug, template), (10, 11, 0, 0), (1, 1, 1, 0),
                                 (-0.1, -0.2, -0.3, 0.0), "eos", "policy-v1", "decoding-v1")

    def test_raw_response_and_scoring_alignment_with_eos_padding(self):
        response = self.response()
        self.roundtrip(response)
        scores = TokenScores(response.row_id, "policy-v1", "base", response.contexts.base.prompt_hash,
                             response.response_hash, response.response_mask, (-1, -2, -3, 0))
        scores.assert_response(response)
        self.roundtrip(scores)
        for changes in ({"row_id": "other"}, {"policy_version": "initial-model"}, {"prompt_hash": response.contexts.augmented.prompt_hash},
                        {"response_hash": stable_hash("parsed-action-retokenized")}, {"response_mask": (1, 1, 0, 0)}):
            with self.subTest(changes=changes), self.assertRaises(TokenAlignmentError):
                replace(scores, **changes).assert_response(response)
        for changes in ({"augmented_log_probs": ()}, {"response_mask": (1, 1)}, {"response_mask": (1, 1, 2, 0)}):
            with self.assertRaises(ContractError):
                replace(response, **changes)
        with self.assertRaises(IdentityError):
            replace(response, generation_policy_version="initial-model")
        with self.assertRaises(IdentityError):
            replace(response, contexts=replace(response.contexts, template=TemplateBinding("other", "changed")))
        self.roundtrip(replace(response, stop_reason="length"))

    def test_empty_response_requires_explicit_nontrainable_status(self):
        response = self.response()
        empty = replace(response, response_ids=(), response_mask=(), augmented_log_probs=(), stop_reason="empty")
        self.roundtrip(empty)
        with self.assertRaises(TokenAlignmentError):
            replace(empty, stop_reason="eos")
        with self.assertRaises(TokenAlignmentError):
            replace(response, stop_reason="empty")
        with self.assertRaises(TokenAlignmentError):
            replace(response.contexts, template=None)
        self.roundtrip(ContextPair(response.contexts.base, response.contexts.base, None))

    def cycle(self, cycle_id="cycle-1", solve=True):
        template = TemplateBinding("selected-" + cycle_id, "Check available actions.", cycle_id)
        evidence = SolvingEvidence(cycle_id, template.template_id, outcome(episode(template=template, phase="search_train")))
        return TemplateCycle(cycle_id, "policy-v1", "panel-1", (template,), (recipe(1).sample_key,), {template.template_id: (1.0,)}, (evidence,) if solve else ())

    def state(self):
        results = tuple(outcome(episode(traj=f"t{i}"), success=False) for i in range(8))
        evidence = GroupEvidence(recipe().sample_key, instance().instance_id, "g0", "policy-v1", "base", 8, results, 22, 22)
        hard = HardSampleRecord(recipe(), (evidence,), "last-observation")
        return P2OState(22, 22, 0, "TRAIN_SAVED", (hard,), (recipe().sample_key,), (self.cycle(),),
                        (LatestSolved(recipe().sample_key, "cycle-1"),), components=(ComponentState("budget", 1, {"used": 4, "reserved": 1}),))

    def test_complete_history_snapshot_roundtrip_and_reference_integrity(self):
        state = self.state()
        self.roundtrip(state)
        for changes in ({"ever_hard": ()}, {"latest_solved_cycle": (LatestSolved(recipe().sample_key, "missing"),)},
                        {"last_gepa_completed_epoch": 23}, {"cycle_template_registry": ()}, {"latest_solved_cycle": ()}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(state, **changes)
        # A newer global cycle with no success must retain the older successful cycle.
        self.roundtrip(replace(state, cycle_template_registry=state.cycle_template_registry + (self.cycle("cycle-2", solve=False),)))
        # A newer successful cycle must become the latest successful reference.
        with self.assertRaises(IdentityError):
            replace(state, cycle_template_registry=state.cycle_template_registry + (self.cycle("cycle-2"),))
        self.roundtrip(replace(state, cycle_template_registry=state.cycle_template_registry + (self.cycle("cycle-2"),),
                               latest_solved_cycle=(LatestSolved(recipe().sample_key, "cycle-2"),)))

    def test_rejected_candidate_success_cannot_be_historical_solution(self):
        cycle = self.cycle()
        rejected = TemplateBinding("rejected", "A candidate which succeeded.", cycle.cycle_id)
        evidence = SolvingEvidence(cycle.cycle_id, rejected.template_id, outcome(episode(template=rejected)))
        with self.assertRaises(IdentityError):
            replace(cycle, solving_evidence=(evidence,))
        with self.assertRaises(ContractError):
            replace(evidence, outcome=outcome(episode(template=rejected), success=False))

    def test_duplicate_padding_cannot_add_group_episodes(self):
        evidence = self.state().interval_hard[0].evidence[0]
        with self.assertRaises(IdentityError):
            replace(evidence, outcomes=evidence.outcomes[:7] + evidence.outcomes[:1])
        with self.assertRaises(IdentityError):
            replace(evidence, outcomes=(outcome(episode(group="other")),))
        # Partial/invalid groups can be recorded, but classification is TODO-04.
        self.roundtrip(replace(evidence, outcomes=evidence.outcomes[:7]))

    def test_eval_request_result_keep_task_instance_template_policy_phase(self):
        task = recipe()
        identity = episode(task, phase="search_train")
        request = EvalRequest("eval-0", "attempt-0", "panel-1", instance(task), None, "policy-v1", "train", {"temperature": 0.8})
        result = EvalResult(request, outcome(identity), None)
        self.roundtrip(result)
        for changes in ({"instance": replace(instance(task), instance_id="other")}, {"policy_version": "wrong-policy"}, {"split": "dev"}):
            with self.assertRaises(IdentityError):
                replace(result, request=replace(request, **changes))

    def test_json_and_numeric_schemas_reject_hidden_python_objects_and_nan(self):
        for payload in ({"tensor": object()}, {"rng": {1, 2}}, {1: "not-json-key"}, {"value": math.nan}):
            with self.assertRaises(ContractError):
                ComponentState("component", 1, payload)
        with self.assertRaises(ContractError):
            replace(outcome(), task_return=math.inf)
        with self.assertRaises(ContractError):
            replace(outcome(), success=1)
        with self.assertRaises(ContractError):
            replace(instance(), captured_turn=1)
        self.roundtrip(ArtifactRef("world.json", "a" * 64))
        with self.assertRaises(ContractError):
            ArtifactRef("world.json", "bad-hash")


if __name__ == "__main__":
    unittest.main()
