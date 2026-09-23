"""Save and restore synchronized bundles of all segment learner states.

Phase 4 checkpoints are written only after a PPO update, when rollout buffers are
empty. They intentionally exclude live MuJoCo state, targets, and unfinished
episodes.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from rl_lib.data import ObservationNormalizer

from centipede.environment import AgentID
from centipede.training.learners import LearnerConfig, SegmentLearner
from centipede.training.rollout import RolloutConfig

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    """Training position and configuration restored from one learner bundle."""

    total_environment_transitions: int
    completed_updates: int
    learner_config: LearnerConfig
    rollout_config: RolloutConfig


def build_checkpoint_state(
    learners: Mapping[AgentID, SegmentLearner],
    learner_config: LearnerConfig,
    rollout_config: RolloutConfig,
    *,
    total_environment_transitions: int,
    completed_updates: int,
) -> dict[str, Any]:
    """Build one in-memory bundle containing every synchronized learner state."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "total_environment_transitions": total_environment_transitions,
        "completed_updates": completed_updates,
        "learner_config": asdict(learner_config),
        "rollout_config": asdict(rollout_config),
        "learners": {
            agent: _learner_state(learner) for agent, learner in learners.items()
        },
        "torch_rng_state": torch.get_rng_state().clone(),
    }


def restore_checkpoint_state(
    state: Mapping[str, Any],
    learners: Mapping[AgentID, SegmentLearner],
) -> CheckpointMetadata:
    """Validate and restore one checkpoint into fresh learner objects."""

    expected_fields = {
        "format_version",
        "total_environment_transitions",
        "completed_updates",
        "learner_config",
        "rollout_config",
        "learners",
        "torch_rng_state",
    }

    if set(state) != expected_fields:
        raise ValueError("Checkpoint contains missing or unexpected fields")

    if state["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {state['format_version']}"
        )

    total_transitions = state["total_environment_transitions"]
    completed_updates = state["completed_updates"]

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (total_transitions, completed_updates)
    ):
        raise ValueError("Checkpoint counters must be nonnegative integers")

    # The dataclass constructors own their field compatibility.
    try:
        learner_config = LearnerConfig(**state["learner_config"])
        rollout_config = RolloutConfig(**state["rollout_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint contains invalid configuration") from error

    saved_learners = state["learners"]
    if not isinstance(saved_learners, Mapping):
        raise TypeError("Checkpoint learners must be a mapping")

    if set(saved_learners) != set(learners):
        raise ValueError("Checkpoint agent IDs do not match current learners")

    torch_rng_state = state["torch_rng_state"]
    if not isinstance(torch_rng_state, torch.Tensor):
        raise TypeError("Checkpoint PyTorch random state must be a tensor")

    expected_learner_fields = {
        "observation_size",
        "action_size",
        "actor_model_state",
        "critic_model_state",
        "actor_optimizer_state",
        "critic_optimizer_state",
        "normalizer_state",
        "ppo_rng_state",
    }
    validated_states: dict[AgentID, Mapping[str, Any]] = {}

    # Validate every project-owned entry before modifying any learner.
    for agent, learner in learners.items():
        saved_state = saved_learners[agent]

        if not isinstance(saved_state, Mapping):
            raise TypeError(f"Checkpoint state for agent {agent} must be a mapping")

        typed_state = cast(Mapping[str, Any], saved_state)

        if set(typed_state) != expected_learner_fields:
            raise ValueError(f"Checkpoint state for agent {agent} has invalid fields")

        saved_dimensions = (
            typed_state["observation_size"],
            typed_state["action_size"],
        )
        current_dimensions = (
            learner.observation_size,
            learner.action_size,
        )

        if saved_dimensions != current_dimensions:
            raise ValueError(f"Checkpoint dimensions do not match agent {agent}")

        validated_states[agent] = typed_state

    for agent, learner in learners.items():
        _restore_learner_state(learner, validated_states[agent])

    torch.set_rng_state(torch_rng_state)

    return CheckpointMetadata(
        total_environment_transitions=total_transitions,
        completed_updates=completed_updates,
        learner_config=learner_config,
        rollout_config=rollout_config,
    )


def save_checkpoint(
    path: str | Path,
    learners: Mapping[AgentID, SegmentLearner],
    learner_config: LearnerConfig,
    rollout_config: RolloutConfig,
    *,
    total_environment_transitions: int,
    completed_updates: int,
) -> None:
    """Write one checkpoint file containing all segment learners."""

    state = build_checkpoint_state(
        learners=learners,
        learner_config=learner_config,
        rollout_config=rollout_config,
        total_environment_transitions=total_environment_transitions,
        completed_updates=completed_updates,
    )

    torch.save(state, Path(path))


def load_checkpoint(
    path: str | Path,
    learners: Mapping[AgentID, SegmentLearner],
) -> CheckpointMetadata:
    """Read one checkpoint file and restore all segment learners together."""

    loaded_state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(loaded_state, Mapping):
        raise TypeError("Checkpoint file must contain a mapping")

    return restore_checkpoint_state(
        cast(Mapping[str, Any], loaded_state),
        learners,
    )


def _learner_state(learner: SegmentLearner) -> dict[str, Any]:
    """Extract models, optimizers, normalizer, dimensions, and PPO random state."""

    return {
        "observation_size": learner.observation_size,
        "action_size": learner.action_size,
        "actor_model_state": deepcopy(learner.ppo.actor_model.state_dict()),
        "critic_model_state": deepcopy(learner.ppo.critic_model.state_dict()),
        "actor_optimizer_state": deepcopy(learner.ppo.actor_optimizer.state_dict()),
        "critic_optimizer_state": deepcopy(learner.ppo.critic_optimizer.state_dict()),
        "normalizer_state": deepcopy(learner.normalizer.state_dict()),
        "ppo_rng_state": deepcopy(learner.ppo.rng.bit_generator.state),
    }


def _restore_learner_state(
    learner: SegmentLearner,
    state: Mapping[str, Any],
) -> None:
    """Restore one previously validated segment entry into its learner."""

    learner.ppo.actor_model.load_state_dict(state["actor_model_state"])
    learner.ppo.critic_model.load_state_dict(state["critic_model_state"])

    learner.ppo.actor_optimizer.load_state_dict(state["actor_optimizer_state"])
    learner.ppo.critic_optimizer.load_state_dict(state["critic_optimizer_state"])

    learner.normalizer = ObservationNormalizer.from_state_dict(
        state["normalizer_state"]
    )

    learner.ppo.rng.bit_generator.state = deepcopy(state["ppo_rng_state"])
