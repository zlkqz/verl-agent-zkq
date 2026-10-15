"""Structural interfaces for task-independent P2O orchestration.

Only explicit reset/action requests may touch environment slots. Inactive slots and
the normal training sampler must not advance. Model padding never produces requests.
Reset and terminal snapshots refer to the requested episode, before any auto-reset.
Failures raise typed exceptions or return explicit invalid outcomes, never implicit 0.
"""

from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

from verl.trainer.p2o.contracts import (
    ActionRequest,
    AdapterCapabilities,
    ChatMessage,
    ComponentState,
    ContextPair,
    EpisodeOutcome,
    EpisodeTrace,
    EvalInstance,
    EvalRequest,
    EvalResult,
    FeedbackRecord,
    GeneratedResponse,
    PolicyCapabilities,
    ResetBatch,
    ResetRequest,
    TaskRecipe,
    TemplateBinding,
    TemplateRuleContext,
    TokenScores,
    TransitionBatch,
    TurnIdentity,
)


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> ComponentState: ...

    def load_state_dict(self, state: ComponentState) -> None: ...


@runtime_checkable
class P2OEnvAdapter(Stateful, Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def capture_task(self, slot: int, reset_observation: str, reset_info: Dict[str, Any]) -> TaskRecipe: ...

    def capture_instance(self, slot: int) -> EvalInstance:
        """Only at t=0, before the first action; reject later captures."""
        ...

    def make_instances(self, recipes: Tuple[TaskRecipe, ...], seed_schedule: Tuple[int, ...],
                       mode: Literal["resample", "replay"], replay_instances: Tuple[EvalInstance, ...] = ()) -> Tuple[EvalInstance, ...]:
        """Replay requires saved instances; resample changes only allowed fields.

        Recipe/seed counts must match. Replay input must match recipe/sample/version
        identities; no re-drawing an already committed panel. Seeds are not task ids.
        """
        ...

    def reset_instances(self, requests: Tuple[ResetRequest, ...]) -> ResetBatch: ...

    def step(self, requests: Tuple[ActionRequest, ...]) -> TransitionBatch: ...

    def episode_outcome(self, trace: EpisodeTrace) -> EpisodeOutcome: ...

    def feedback(self, trace: EpisodeTrace, token_budget: int) -> FeedbackRecord: ...

    def template_constraints(self, recipes: Tuple[TaskRecipe, ...], traces: Tuple[EpisodeTrace, ...]) -> TemplateRuleContext:
        """Verifier-only output; never pass to the reflection model."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class PromptContextBuilder(Protocol):
    def build_pair(self, baseline_messages: Tuple[ChatMessage, ...], template: Optional[TemplateBinding], tokenizer: Any) -> ContextPair: ...


@runtime_checkable
class PolicyGenerator(Protocol):
    @property
    def capabilities(self) -> PolicyCapabilities: ...

    def generate(self, identities: Tuple[TurnIdentity, ...], contexts: Tuple[ContextPair, ...],
                 decoding: Dict[str, Any], policy_version: str) -> Tuple[GeneratedResponse, ...]: ...


@runtime_checkable
class PolicyScorer(Protocol):
    def score(self, responses: Tuple[GeneratedResponse, ...], context: Literal["base", "augmented"], policy_version: str) -> Tuple[TokenScores, ...]:
        """Score saved response ids; augmented scoring is for offline validation only.

        Production CRM takes AUG from GeneratedResponse and scores BASE once. Returned
        rows carry identities/hashes/masks and must be aligned by row_id, not position.
        """
        ...


@runtime_checkable
class EpisodeEvaluator(Stateful, Protocol):
    def evaluate(self, requests: Tuple[EvalRequest, ...]) -> Tuple[EvalResult, ...]:
        """Complete t=0 episodes via the adapter; no env_name branches or updates."""
        ...
