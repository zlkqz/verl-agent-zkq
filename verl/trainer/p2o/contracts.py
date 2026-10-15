"""Versioned, JSON-safe boundaries for P2O (no environment or training imports).

Records own deep copies of their inputs. Frozen dataclasses prevent field replacement;
callers must treat nested dictionaries as snapshots too. ``state_dict`` validates again
and returns detached data. Tensor/world/RNG blobs belong in checksummed ArtifactRefs,
not JSON payloads. Schema changes require an explicit version/migration, never coercion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1
Phase = Literal["base", "rescue", "search_train", "search_dev", "validation"]
TerminationReason = Literal["success", "environment_done", "max_steps", "infra_error"]
ResourcePhase = Literal["TRAIN", "TRAIN_SAVED", "GEPA_RUNNING", "GEPA_COMMITTED", "TRAIN_RESTORED"]


class ContractError(ValueError):
    """Malformed data; never reinterpret missing results as task failure."""


class SchemaVersionError(ContractError):
    """Unknown schema/type/field; a migration is required."""


class IdentityError(ContractError):
    """Task, instance, request, or policy identities disagree."""


class UnsupportedCapabilityError(ContractError):
    """The caller must discuss a missing capability, rather than silently downgrade."""


class IncompleteEpisodeError(ContractError):
    """An episode has no committed final result."""


class TokenAlignmentError(ContractError):
    """Scores are not for exactly the saved prompt/response/policy."""


class InfrastructureError(RuntimeError):
    """Transport/worker failure. Retry policy is a caller decision, not success=False."""


def _ensure(condition, message, error=ContractError):
    if not condition:
        raise error(message)


def _json_copy(value, path="payload"):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        _ensure(math.isfinite(value), f"{path}: non-finite number")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_copy(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        _ensure(all(type(key) is str for key in value), f"{path}: JSON keys must be strings")
        return {key: _json_copy(item, f"{path}.{key}") for key, item in value.items()}
    raise ContractError(f"{path}: expected JSON data, got {type(value).__name__}")


def stable_hash(value) -> str:
    """Content fingerprint; no Python hash(), pickle, or implicit repr()."""
    data = json.dumps(_json_copy(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _hints(cls):
    return get_type_hints(cls)


def _typed(value, annotation, path):
    """Validate constructors and decode JSON using the same strict field contract."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return _json_copy(value, path)
    if origin is Union:
        if len(args) == 2 and type(None) in args:
            if value is None:
                return None
            return _typed(value, next(item for item in args if item is not type(None)), path)
        for candidate in args:
            try:
                return _typed(value, candidate, path)
            except ContractError:
                pass
        raise ContractError(f"{path}: does not match {annotation}")
    if origin is Literal:
        _ensure(any(type(value) is type(item) and value == item for item in args), f"{path}: expected one of {args}")
        return value
    if origin is tuple:
        _ensure(isinstance(value, (tuple, list)), f"{path}: expected sequence")
        _ensure(len(args) == 2 and args[1] is Ellipsis, f"{path}: unsupported tuple schema")
        return tuple(_typed(item, args[0], f"{path}[{i}]") for i, item in enumerate(value))
    if origin is dict:
        _ensure(isinstance(value, dict), f"{path}: expected object")
        return {_typed(key, args[0], path): _typed(item, args[1], f"{path}.{key}") for key, item in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, Record):
        if type(value) is annotation:
            result = copy.deepcopy(value)
            result.validate()
            return result
        _ensure(isinstance(value, dict), f"{path}: expected {annotation.__name__}")
        return annotation.from_state_dict(value)
    if annotation is float:
        _ensure(type(value) in (int, float) and math.isfinite(value), f"{path}: expected finite number")
        return float(value)
    _ensure(type(value) is annotation, f"{path}: expected {annotation}, got {type(value).__name__}")
    return value


def _encode(value):
    if isinstance(value, Record):
        return value.state_dict()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return _json_copy(value)


@dataclass(frozen=True)
class Record:
    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, _typed(getattr(self, item.name), _hints(type(self))[item.name], f"{type(self).__name__}.{item.name}"))
        self.validate()

    def validate(self):
        """Subclasses add semantic invariants after field validation."""

    def state_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = {item.name: _encode(_typed(getattr(self, item.name), _hints(type(self))[item.name], item.name)) for item in fields(self)}
        return {"schema_version": self.schema_version, "record_type": type(self).__name__, "payload": payload}

    @classmethod
    def from_state_dict(cls, state):
        _ensure(isinstance(state, dict), f"{cls.__name__}: expected state object", SchemaVersionError)
        _ensure(set(state) == {"schema_version", "record_type", "payload"}, f"{cls.__name__}: unknown/missing envelope fields", SchemaVersionError)
        _ensure(type(state["schema_version"]) is int and state["schema_version"] == cls.schema_version, f"{cls.__name__}: unsupported schema_version", SchemaVersionError)
        _ensure(state["record_type"] == cls.__name__, f"expected record_type={cls.__name__}", SchemaVersionError)
        payload = state["payload"]
        _ensure(isinstance(payload, dict), f"{cls.__name__}: expected payload object", SchemaVersionError)
        _ensure(set(payload) == {item.name for item in fields(cls)}, f"{cls.__name__}: unknown/missing payload fields", SchemaVersionError)
        return cls(**copy.deepcopy(payload))


def _ids(**values):
    for name, value in values.items():
        _ensure(isinstance(value, str) and bool(value.strip()) and value == value.strip(), f"{name}: missing/noncanonical identity", IdentityError)


def _nonnegative(**values):
    for name, value in values.items():
        _ensure(value >= 0, f"{name}: must be nonnegative")


def _unique(values, label):
    _ensure(len(values) == len(set(values)), f"duplicate {label}", IdentityError)


@dataclass(frozen=True)
class SampleKey(Record):
    env: str
    split: str
    task_catalog_version: str
    task_id: str

    def validate(self):
        _ids(env=self.env, split=self.split, task_catalog_version=self.task_catalog_version, task_id=self.task_id)

    @property
    def canonical(self) -> str:
        # Length/escaping are unambiguous; seed, price, slot and policy cannot enter.
        return "p2o:v1:" + json.dumps([self.env, self.split, self.task_catalog_version, self.task_id], ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ArtifactRef(Record):
    uri: str
    sha256: str

    def validate(self):
        _ids(uri=self.uri)
        _ensure(len(self.sha256) == 64 and all(c in "0123456789abcdef" for c in self.sha256), "artifact sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class TaskRecipe(Record):
    sample_key: SampleKey
    task_selector: Dict[str, Any]
    data_version: str
    source_version: str
    env_config: Dict[str, Any]
    allowed_variations: Tuple[str, ...]
    initial_observation: str
    original_instance_summary: Dict[str, Any]

    def validate(self):
        _ids(data_version=self.data_version, source_version=self.source_version)
        _ensure(bool(self.task_selector), "task selector is required", IdentityError)
        _unique(self.allowed_variations, "variation fields")
        for name in self.allowed_variations:
            _ids(variation=name)

    @property
    def recipe_id(self) -> str:
        return "recipe:" + stable_hash({"sample_key": self.sample_key.canonical, "selector": self.task_selector, "data": self.data_version,
                                       "source": self.source_version, "config": self.env_config, "variations": self.allowed_variations})


@dataclass(frozen=True)
class EvalInstance(Record):
    sample_key: SampleKey
    recipe_id: str
    instance_id: str
    initialization: Dict[str, Any]
    initial_observation: str
    world_artifact: Optional[ArtifactRef] = None
    captured_turn: Literal[0] = 0

    def validate(self):
        _ids(recipe_id=self.recipe_id, instance_id=self.instance_id)
        _ensure(bool(self.initialization) or self.world_artifact is not None, "instance needs reconstruction material")


@dataclass(frozen=True)
class AdapterCapabilities(Record):
    subset_reset: bool = False
    subset_step: bool = False
    terminal_snapshot: bool = False
    replay: bool = False
    resample: bool = False
    private_rng: bool = False
    sampler_state: bool = False
    resample_fields: Tuple[str, ...] = ()

    def require(self, *names: str):
        for name in names:
            _ensure(name in {item.name for item in fields(self)} and getattr(self, name) is True, f"unsupported adapter capability: {name}", UnsupportedCapabilityError)


@dataclass(frozen=True)
class PolicyCapabilities(Record):
    raw_response_tokens: bool = False
    aligned_rollout_log_probs: bool = False
    request_decoding_seed: bool = False

    def require(self, *names: str):
        for name in names:
            _ensure(name in {item.name for item in fields(self)} and getattr(self, name) is True, f"unsupported policy capability: {name}", UnsupportedCapabilityError)


@dataclass(frozen=True)
class TemplateBinding(Record):
    template_id: str
    text: str
    source_cycle: Optional[str] = None

    def validate(self):
        _ids(template_id=self.template_id)
        if self.source_cycle is not None:
            _ids(source_cycle=self.source_cycle)

    @property
    def template_hash(self) -> str:
        return stable_hash(self.text)


@dataclass(frozen=True)
class EpisodeIdentity(Record):
    sample_key: SampleKey
    instance_id: str
    group_uid: str
    traj_uid: str
    episode_id: str
    policy_version: str
    phase: Phase
    template: Optional[TemplateBinding] = None
    parent_group_uid: Optional[str] = None

    def validate(self):
        _ids(instance_id=self.instance_id, group_uid=self.group_uid, traj_uid=self.traj_uid, episode_id=self.episode_id, policy_version=self.policy_version)
        if self.parent_group_uid is not None:
            _ids(parent_group_uid=self.parent_group_uid)
        _ensure(self.phase != "rescue" or self.parent_group_uid is not None, "rescue requires parent_group_uid", IdentityError)


@dataclass(frozen=True)
class TurnIdentity(Record):
    episode: EpisodeIdentity
    turn_index: int
    slot: int
    request_id: str
    attempt_id: str
    is_padding: bool = False

    def validate(self):
        _nonnegative(turn_index=self.turn_index, slot=self.slot)
        _ids(request_id=self.request_id, attempt_id=self.attempt_id)


@dataclass(frozen=True)
class ResetRequest(Record):
    slot: int
    request_id: str
    episode_id: str
    instance: EvalInstance

    def validate(self):
        _nonnegative(slot=self.slot)
        _ids(request_id=self.request_id, episode_id=self.episode_id)


@dataclass(frozen=True)
class ResetEntry(Record):
    request: ResetRequest
    observation: str
    available_actions: Tuple[str, ...]
    info: Dict[str, Any]


@dataclass(frozen=True)
class ResetBatch(Record):
    entries: Tuple[ResetEntry, ...]

    def validate(self):
        _unique(tuple(row.request.slot for row in self.entries), "reset slots")
        _unique(tuple(row.request.request_id for row in self.entries), "reset request ids")

    def assert_requests(self, requests: Tuple[ResetRequest, ...]):
        _unique(tuple(row.slot for row in requests), "requested reset slots")
        expected = {row.slot: row for row in requests}
        actual = {row.request.slot: row.request for row in self.entries}
        _ensure(actual == expected, "reset results do not match requested subset/identity", IdentityError)


@dataclass(frozen=True)
class ActionRequest(Record):
    identity: TurnIdentity
    action: str

    def validate(self):
        _ensure(not self.identity.is_padding, "model padding cannot create an environment action", IdentityError)


@dataclass(frozen=True)
class Transition(Record):
    identity: TurnIdentity
    observation: str
    available_actions: Tuple[str, ...]
    executed_action: str
    raw_reward: float
    done: bool
    truncated: bool
    success: Optional[bool]
    task_score: Optional[float]
    valid: bool
    termination_reason: Optional[TerminationReason]
    terminal_observation: Optional[str]
    terminal_info: Dict[str, Any]
    action_diagnostics: Dict[str, Any]
    invalid_action_penalty: float = 0.0

    def validate(self):
        _ensure(not self.identity.is_padding, "padding is not a real transition", IdentityError)
        if not self.valid:
            _ensure(self.success is None and self.termination_reason == "infra_error", "infra failure requires success=None and infra_error")
        elif self.done or self.truncated:
            _ensure(type(self.success) is bool, "final transition requires explicit success")
            _ensure(self.termination_reason is not None and self.termination_reason != "infra_error", "missing/invalid terminal reason")
            _ensure(self.terminal_observation is not None, "terminal snapshot required before auto-reset")
            _ensure((self.termination_reason == "success") == self.success, "success flag/reason disagree")
        else:
            _ensure(self.termination_reason is None and self.terminal_observation is None, "live transition cannot have terminal snapshot/reason")


@dataclass(frozen=True)
class TransitionBatch(Record):
    transitions: Tuple[Transition, ...]

    def validate(self):
        _unique(tuple(row.identity.slot for row in self.transitions), "transition slots")
        _unique(tuple(row.identity.request_id for row in self.transitions), "step request ids")

    def assert_requests(self, requests: Tuple[ActionRequest, ...]):
        _unique(tuple(row.identity.slot for row in requests), "requested step slots")
        expected = {row.identity.slot: row for row in requests}
        actual = {row.identity.slot: row for row in self.transitions}
        _ensure(set(actual) == set(expected), "step changed requested active subset", IdentityError)
        for slot, row in actual.items():
            _ensure(row.identity == expected[slot].identity and row.executed_action == expected[slot].action, "step identity/action mismatch", IdentityError)


@dataclass(frozen=True)
class EpisodeTrace(Record):
    identity: EpisodeIdentity
    initial_observation: str
    transitions: Tuple[Transition, ...]

    def validate(self):
        for i, row in enumerate(self.transitions):
            _ensure(row.identity.episode == self.identity, "trace contains another episode/instance/policy", IdentityError)
            _ensure(row.identity.turn_index == i, "trace turns must be contiguous from zero", IdentityError)
            if i < len(self.transitions) - 1:
                _ensure(row.valid and not row.done and not row.truncated, "transition after episode closed")
        _unique(tuple(row.identity.request_id for row in self.transitions), "trace request ids")


@dataclass(frozen=True)
class EpisodeOutcome(Record):
    identity: EpisodeIdentity
    success: Optional[bool]
    task_score: Optional[float]
    task_return: float
    invalid_action_penalty: float
    termination_reason: TerminationReason
    valid: bool
    steps: int

    def validate(self):
        _nonnegative(steps=self.steps)
        if self.valid:
            _ensure(type(self.success) is bool and self.termination_reason != "infra_error", "valid outcome needs binary success")
            _ensure((self.termination_reason == "success") == self.success, "success flag/reason disagree")
        else:
            _ensure(self.success is None and self.termination_reason == "infra_error", "invalid outcome is not task failure")


def outcome_from_trace(trace: EpisodeTrace) -> EpisodeOutcome:
    """Reduce already-normalized adapter output; never infer success from reward."""
    _ensure(bool(trace.transitions), "episode has no final transition", IncompleteEpisodeError)
    last = trace.transitions[-1]
    _ensure(not last.valid or last.done or last.truncated, "episode is still active", IncompleteEpisodeError)
    return EpisodeOutcome(identity=trace.identity, success=last.success, task_score=last.task_score,
                          task_return=sum(row.raw_reward for row in trace.transitions),
                          invalid_action_penalty=sum(row.invalid_action_penalty for row in trace.transitions),
                          termination_reason=last.termination_reason, valid=last.valid, steps=len(trace.transitions))


@dataclass(frozen=True)
class ChatMessage(Record):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


def _tokens(ids, mask):
    _ensure(len(ids) == len(mask), "token ids/mask length mismatch", TokenAlignmentError)
    _ensure(all(token >= 0 for token in ids), "negative token id", TokenAlignmentError)
    _ensure(all(value in (0, 1) for value in mask), "mask must be binary", TokenAlignmentError)


@dataclass(frozen=True)
class PromptSnapshot(Record):
    messages: Tuple[ChatMessage, ...]
    input_ids: Tuple[int, ...]
    attention_mask: Tuple[int, ...]
    position_ids: Tuple[int, ...]
    tokenizer_fingerprint: str

    def validate(self):
        _ids(tokenizer_fingerprint=self.tokenizer_fingerprint)
        _tokens(self.input_ids, self.attention_mask)
        _ensure(len(self.input_ids) == len(self.position_ids) and all(i >= 0 for i in self.position_ids), "invalid prompt position ids", TokenAlignmentError)
        _ensure(bool(self.messages) and any(self.attention_mask), "empty prompt snapshot")

    @property
    def prompt_hash(self):
        return stable_hash(self.state_dict())


@dataclass(frozen=True)
class ContextPair(Record):
    base: PromptSnapshot
    augmented: PromptSnapshot
    template: Optional[TemplateBinding]

    def validate(self):
        _ensure(self.base.tokenizer_fingerprint == self.augmented.tokenizer_fingerprint, "context tokenizers differ", TokenAlignmentError)
        if self.template is None:
            _ensure(self.base == self.augmented, "no-template context must equal baseline", TokenAlignmentError)


@dataclass(frozen=True)
class GeneratedResponse(Record):
    row_id: str
    identity: TurnIdentity
    contexts: ContextPair
    response_ids: Tuple[int, ...]
    response_mask: Tuple[int, ...]
    augmented_log_probs: Tuple[float, ...]
    stop_reason: Literal["eos", "length", "empty", "error"]
    generation_policy_version: str
    decoding_fingerprint: str

    def validate(self):
        _ids(row_id=self.row_id, generation_policy_version=self.generation_policy_version, decoding_fingerprint=self.decoding_fingerprint)
        _ensure(self.generation_policy_version == self.identity.episode.policy_version, "generation policy version mismatch", IdentityError)
        _ensure(self.contexts.template == self.identity.episode.template, "episode template changed", IdentityError)
        _tokens(self.response_ids, self.response_mask)
        _ensure(len(self.response_ids) == len(self.augmented_log_probs), "rollout logprob/token count mismatch", TokenAlignmentError)
        if self.stop_reason in ("eos", "length"):
            _ensure(any(self.response_mask), "successful generation has no response tokens", TokenAlignmentError)
        else:
            _ensure(not any(self.response_mask), "empty/error output must not contain trainable tokens", TokenAlignmentError)

    @property
    def response_hash(self):
        return stable_hash({"ids": self.response_ids, "mask": self.response_mask})


@dataclass(frozen=True)
class TokenScores(Record):
    row_id: str
    policy_version: str
    context: Literal["base", "augmented"]
    prompt_hash: str
    response_hash: str
    response_mask: Tuple[int, ...]
    log_probs: Tuple[float, ...]

    def validate(self):
        _ids(row_id=self.row_id, policy_version=self.policy_version, prompt_hash=self.prompt_hash, response_hash=self.response_hash)
        _ensure(len(self.log_probs) == len(self.response_mask) and all(v in (0, 1) for v in self.response_mask), "score mask/length mismatch", TokenAlignmentError)

    def assert_response(self, response: GeneratedResponse):
        snapshot = response.contexts.base if self.context == "base" else response.contexts.augmented
        _ensure(self.row_id == response.row_id and self.policy_version == response.generation_policy_version
                and self.prompt_hash == snapshot.prompt_hash and self.response_hash == response.response_hash
                and self.response_mask == response.response_mask, "scores do not match saved response/context/policy", TokenAlignmentError)


@dataclass(frozen=True)
class FeedbackRecord(Record):
    identity: EpisodeIdentity
    outcome: EpisodeOutcome
    visible_task: str
    evidence: Tuple[Dict[str, Any], ...]
    diagnostics: Dict[str, Any]
    rendered_text: str
    token_count: int
    token_budget: int
    rule_version: str

    def validate(self):
        _ensure(self.identity == self.outcome.identity, "feedback outcome identity mismatch", IdentityError)
        _nonnegative(token_count=self.token_count, token_budget=self.token_budget)
        _ensure(self.token_count <= self.token_budget, "feedback exceeds token budget")
        _ids(rule_version=self.rule_version)


@dataclass(frozen=True)
class TemplateRuleContext(Record):
    """Verifier-only data. Never include this record in a reflection prompt."""
    sample_keys: Tuple[SampleKey, ...]
    rule_version: str
    private_identifiers: Tuple[str, ...]
    constraints: Dict[str, Any]

    def validate(self):
        _ids(rule_version=self.rule_version)


@dataclass(frozen=True)
class EvaluationPanel(Record):
    panel_id: str
    policy_version: str
    train_instances: Tuple[EvalInstance, ...]
    dev_instances: Tuple[EvalInstance, ...]

    def validate(self):
        _ids(panel_id=self.panel_id, policy_version=self.policy_version)
        train = {item.sample_key for item in self.train_instances}
        dev = {item.sample_key for item in self.dev_instances}
        _ensure(not train & dev, "same sample_key in train and dev, even across instances", IdentityError)
        _unique(tuple(item.sample_key for item in self.dev_instances), "dev sample keys (one fixed instance per task)")
        _unique(tuple(item.instance_id for item in self.train_instances + self.dev_instances), "panel instance ids")


@dataclass(frozen=True)
class EvalRequest(Record):
    request_id: str
    attempt_id: str
    panel_id: str
    instance: EvalInstance
    template: Optional[TemplateBinding]
    policy_version: str
    split: Literal["train", "dev"]
    decoding: Dict[str, Any]

    def validate(self):
        _ids(request_id=self.request_id, attempt_id=self.attempt_id, panel_id=self.panel_id, policy_version=self.policy_version)
        _ensure(bool(self.decoding), "explicit decoding configuration required")


@dataclass(frozen=True)
class EvalResult(Record):
    request: EvalRequest
    outcome: EpisodeOutcome
    trace_ref: Optional[ArtifactRef]

    def validate(self):
        request, identity = self.request, self.outcome.identity
        _ensure(request.instance.sample_key == identity.sample_key and request.instance.instance_id == identity.instance_id
                and request.policy_version == identity.policy_version and request.template == identity.template,
                "evaluation result identity mismatch", IdentityError)
        _ensure(identity.phase == ("search_train" if request.split == "train" else "search_dev"), "evaluation phase mismatch", IdentityError)


@dataclass(frozen=True)
class GroupEvidence(Record):
    sample_key: SampleKey
    instance_id: str
    group_uid: str
    policy_version: str
    phase: Phase
    expected_k: int
    outcomes: Tuple[EpisodeOutcome, ...]
    epoch: int
    global_step: int

    def validate(self):
        _ids(instance_id=self.instance_id, group_uid=self.group_uid, policy_version=self.policy_version)
        _nonnegative(epoch=self.epoch, global_step=self.global_step)
        _ensure(self.expected_k > 0 and len(self.outcomes) <= self.expected_k, "invalid group size")
        _unique(tuple(row.identity.traj_uid for row in self.outcomes), "group trajectory ids")
        _unique(tuple(row.identity.episode_id for row in self.outcomes), "group episode ids")
        for row in self.outcomes:
            identity = row.identity
            _ensure((identity.sample_key, identity.instance_id, identity.group_uid, identity.policy_version, identity.phase)
                    == (self.sample_key, self.instance_id, self.group_uid, self.policy_version, self.phase), "cross-group outcome", IdentityError)


@dataclass(frozen=True)
class HardSampleRecord(Record):
    recipe: TaskRecipe
    evidence: Tuple[GroupEvidence, ...]
    latest_observation: str
    trace_refs: Tuple[ArtifactRef, ...] = ()

    def validate(self):
        _ensure(bool(self.evidence), "hard sample requires observed evidence")
        _ensure(all(row.sample_key == self.recipe.sample_key for row in self.evidence), "hard recipe/evidence key mismatch", IdentityError)


@dataclass(frozen=True)
class SolvingEvidence(Record):
    cycle_id: str
    template_id: str
    outcome: EpisodeOutcome

    def validate(self):
        _ids(cycle_id=self.cycle_id, template_id=self.template_id)
        template = self.outcome.identity.template
        _ensure(self.outcome.valid and self.outcome.success is True and template is not None, "solving evidence needs actual valid success")
        _ensure(template.template_id == self.template_id and template.source_cycle == self.cycle_id, "solving template/cycle mismatch", IdentityError)


@dataclass(frozen=True)
class TemplateCycle(Record):
    cycle_id: str
    policy_version: str
    panel_id: str
    selected: Tuple[TemplateBinding, ...]
    dev_sample_keys: Tuple[SampleKey, ...]
    dev_scores: Dict[str, Tuple[float, ...]]
    solving_evidence: Tuple[SolvingEvidence, ...]

    def validate(self):
        _ids(cycle_id=self.cycle_id, policy_version=self.policy_version, panel_id=self.panel_id)
        ids = tuple(item.template_id for item in self.selected)
        _unique(ids, "selected template ids")
        _unique(self.dev_sample_keys, "cycle dev keys")
        _ensure(set(ids) == set(self.dev_scores), "dev scores must match final selected set")
        for template in self.selected:
            _ensure(template.source_cycle == self.cycle_id, "selected template belongs to another cycle", IdentityError)
        for scores in self.dev_scores.values():
            _ensure(len(scores) == len(self.dev_sample_keys) and all(v in (0.0, 1.0) for v in scores), "dev scores require one binary outcome per fixed task")
        selected = {item.template_id: item for item in self.selected}
        for evidence in self.solving_evidence:
            _ensure(evidence.cycle_id == self.cycle_id and evidence.template_id in selected, "unselected template cannot establish historical solving evidence", IdentityError)
            _ensure(evidence.outcome.identity.policy_version == self.policy_version and evidence.outcome.identity.template == selected[evidence.template_id], "solving evidence policy/template mismatch", IdentityError)


@dataclass(frozen=True)
class LatestSolved(Record):
    sample_key: SampleKey
    cycle_id: str

    def validate(self):
        _ids(cycle_id=self.cycle_id)


@dataclass(frozen=True)
class ComponentState(Record):
    """Versioned module-owned payload, e.g. budget/search/RNG/sampler/resource state.

    The owning module validates its payload schema. Large or non-JSON state is saved
    as an artifact; this boundary does not claim crash-safe checkpoint I/O.
    """
    component: str
    version: int
    payload: Dict[str, Any]
    artifacts: Tuple[ArtifactRef, ...] = ()

    def validate(self):
        _ids(component=self.component)
        _ensure(self.version >= 1, "component state version must be positive")


@dataclass(frozen=True)
class P2OState(Record):
    completed_epochs: int
    global_step: int
    last_gepa_completed_epoch: int
    phase: ResourcePhase
    interval_hard: Tuple[HardSampleRecord, ...]
    ever_hard: Tuple[SampleKey, ...]
    cycle_template_registry: Tuple[TemplateCycle, ...]
    latest_solved_cycle: Tuple[LatestSolved, ...]
    active_cycle_id: Optional[str] = None
    components: Tuple[ComponentState, ...] = ()

    def validate(self):
        _nonnegative(completed_epochs=self.completed_epochs, global_step=self.global_step, last_gepa_completed_epoch=self.last_gepa_completed_epoch)
        _ensure(self.last_gepa_completed_epoch <= self.completed_epochs, "GEPA anchor exceeds completed epoch")
        _unique(self.ever_hard, "ever_hard keys")
        _unique(tuple(row.recipe.sample_key for row in self.interval_hard), "interval hard keys")
        _unique(tuple(row.cycle_id for row in self.cycle_template_registry), "cycle ids")
        _unique(tuple(row.sample_key for row in self.latest_solved_cycle), "latest solved keys")
        _unique(tuple(row.component for row in self.components), "component names")
        _ensure(all(row.recipe.sample_key in self.ever_hard for row in self.interval_hard), "interval hard missing from ever_hard", IdentityError)
        cycles = {row.cycle_id: row for row in self.cycle_template_registry}
        for assignment in self.latest_solved_cycle:
            _ensure(assignment.sample_key in self.ever_hard and assignment.cycle_id in cycles, "dangling historical solving reference", IdentityError)
            _ensure(any(row.outcome.identity.sample_key == assignment.sample_key for row in cycles[assignment.cycle_id].solving_evidence), "latest_solved requires final selected success evidence", IdentityError)
        expected_latest = {}
        for cycle in self.cycle_template_registry:  # immutable cycles in commit order
            for evidence in cycle.solving_evidence:
                key = evidence.outcome.identity.sample_key
                _ensure(key in self.ever_hard, "solving evidence is not for a historical hard task", IdentityError)
                expected_latest[key] = cycle.cycle_id
        _ensure({row.sample_key: row.cycle_id for row in self.latest_solved_cycle} == expected_latest,
                "latest_solved must retain the latest successful selected cycle for every solved task", IdentityError)
        if self.active_cycle_id is not None:
            _ids(active_cycle_id=self.active_cycle_id)
