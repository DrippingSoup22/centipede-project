"""Focused checks for synchronized learner checkpoint bundles."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from centipede.environment import CentipedeParallelEnv
from centipede.training.checkpoints import (
    build_checkpoint_state,
    load_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
)
from centipede.training.learners import LearnerConfig, SegmentLearner, build_learners
from centipede.training.rollout import RolloutConfig

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "assembly.xml"
LEARNER_CONFIG = LearnerConfig(hidden_sizes=(8,))
ROLLOUT_CONFIG = RolloutConfig(
    rollout_window_steps=4,
    update_epochs=2,
    minibatch_size=2,
)


@pytest.fixture
def environment() -> Iterator[CentipedeParallelEnv]:
    """Construct the public environment and release its MuJoCo resources."""
    instance = CentipedeParallelEnv(model_path=MODEL_PATH)
    try:
        yield instance
    finally:
        instance.close()


def make_learners(
    environment: CentipedeParallelEnv,
    *,
    seed_base: int,
) -> dict[int, SegmentLearner]:
    """Construct one small independent learner set for checkpoint tests."""
    seeds = {agent: seed_base + agent for agent in environment.possible_agents}
    return build_learners(environment, LEARNER_CONFIG, seeds)


def update_learner(learner: SegmentLearner) -> None:
    """Populate one learner's model, optimizer, normalizer, and PPO RNG state."""
    base_observation = np.linspace(
        -0.25,
        0.25,
        learner.observation_size,
        dtype=np.float32,
    )
    raw_observations = (
        base_observation,
        -base_observation,
        base_observation * np.float32(0.5),
        -base_observation * np.float32(0.5),
    )
    observations = np.stack(
        [
            learner.normalizer.normalize(observation, update=True)
            for observation in raw_observations
        ]
    )
    action = np.linspace(
        -0.2,
        0.2,
        learner.action_size,
        dtype=np.float32,
    )
    actions = np.stack((action, -action, action * 0.5, -action * 0.5))

    learner.ppo.update(
        observations=observations,
        actions=actions,
        old_log_probabilities=np.array([-1.0, -0.8, -0.6, -0.4], dtype=np.float32),
        advantages=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float32),
        return_targets=np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        update_epochs=ROLLOUT_CONFIG.update_epochs,
        minibatch_size=ROLLOUT_CONFIG.minibatch_size,
    )


def assert_models_equal(source: SegmentLearner, restored: SegmentLearner) -> None:
    """Compare actor and critic tensors without requiring shared objects."""
    model_pairs = (
        (source.ppo.actor_model, restored.ppo.actor_model),
        (source.ppo.critic_model, restored.ppo.critic_model),
    )

    for source_model, restored_model in model_pairs:
        source_state = source_model.state_dict()
        restored_state = restored_model.state_dict()
        assert source_state.keys() == restored_state.keys()
        for name, source_tensor in source_state.items():
            torch.testing.assert_close(source_tensor, restored_state[name])


def assert_outputs_equal(source: SegmentLearner, restored: SegmentLearner) -> None:
    """Compare deterministic policy and critic outputs after restoration."""
    observation = np.linspace(
        -0.1,
        0.1,
        source.observation_size,
        dtype=np.float32,
    )
    source_observation = source.normalizer.normalize(observation, update=False)
    restored_observation = restored.normalizer.normalize(observation, update=False)

    np.testing.assert_allclose(source_observation, restored_observation)
    np.testing.assert_allclose(
        source.ppo.select_action(source_observation, deterministic=True),
        restored.ppo.select_action(restored_observation, deterministic=True),
    )
    assert source.ppo.state_value(source_observation) == pytest.approx(
        restored.ppo.state_value(restored_observation)
    )


def test_checkpoint_state_restores_every_independent_learner(
    environment: CentipedeParallelEnv,
) -> None:
    """Restore models, optimizers, normalizers, RNGs, and training metadata."""
    with torch.random.fork_rng(devices=[]):
        source_learners = make_learners(environment, seed_base=100)
        restored_learners = make_learners(environment, seed_base=200)
        for learner in source_learners.values():
            update_learner(learner)

        state = build_checkpoint_state(
            source_learners,
            LEARNER_CONFIG,
            ROLLOUT_CONFIG,
            total_environment_transitions=512,
            completed_updates=2,
        )
        expected_random_values = torch.rand(4)
        torch.manual_seed(999)

        metadata = restore_checkpoint_state(state, restored_learners)

        assert metadata.total_environment_transitions == 512
        assert metadata.completed_updates == 2
        assert metadata.learner_config == LEARNER_CONFIG
        assert metadata.rollout_config == ROLLOUT_CONFIG
        torch.testing.assert_close(torch.rand(4), expected_random_values)

        for agent in environment.possible_agents:
            source = source_learners[agent]
            restored = restored_learners[agent]

            assert source is not restored
            assert source.normalizer is not restored.normalizer
            assert source.normalizer.state_dict() == restored.normalizer.state_dict()
            assert source.ppo.rng.bit_generator.state == (
                restored.ppo.rng.bit_generator.state
            )
            assert_models_equal(source, restored)
            assert_outputs_equal(source, restored)

            # Identical next updates exercise the restored Adam moments and PPO
            # shuffling state, not only the already-restored model tensors.
            torch_rng_state = torch.get_rng_state().clone()
            update_learner(source)
            torch.set_rng_state(torch_rng_state)
            update_learner(restored)
            assert_models_equal(source, restored)


def test_save_and_load_checkpoint_round_trip(
    environment: CentipedeParallelEnv,
    tmp_path: Path,
) -> None:
    """Serialize the supported state and reload it through the safe loader."""
    with torch.random.fork_rng(devices=[]):
        source_learners = make_learners(environment, seed_base=300)
        restored_learners = make_learners(environment, seed_base=400)
        source_learners[0].normalizer.normalize(
            np.ones(source_learners[0].observation_size, dtype=np.float32),
            update=True,
        )
        checkpoint_path = tmp_path / "checkpoint.pt"

        save_checkpoint(
            checkpoint_path,
            source_learners,
            LEARNER_CONFIG,
            ROLLOUT_CONFIG,
            total_environment_transitions=256,
            completed_updates=1,
        )
        metadata = load_checkpoint(checkpoint_path, restored_learners)

        assert checkpoint_path.is_file()
        assert metadata.total_environment_transitions == 256
        assert metadata.completed_updates == 1
        for agent in environment.possible_agents:
            assert_outputs_equal(source_learners[agent], restored_learners[agent])


def test_version_one_checkpoint_restores_the_earlier_rollout_name(
    environment: CentipedeParallelEnv,
) -> None:
    """Keep evaluation compatible with checkpoints saved before the rename."""
    with torch.random.fork_rng(devices=[]):
        state = build_checkpoint_state(
            make_learners(environment, seed_base=700),
            LEARNER_CONFIG,
            ROLLOUT_CONFIG,
            total_environment_transitions=0,
            completed_updates=0,
        )
        state["format_version"] = 1
        state["rollout_config"]["steps_per_environment"] = state["rollout_config"].pop(
            "rollout_window_steps"
        )

        metadata = restore_checkpoint_state(
            state, make_learners(environment, seed_base=800)
        )

    assert metadata.rollout_config == ROLLOUT_CONFIG


def test_restore_rejects_missing_agent_before_modifying_learners(
    environment: CentipedeParallelEnv,
) -> None:
    """Reject an incomplete synchronized bundle before loading any model state."""
    source_learners = make_learners(environment, seed_base=500)
    restored_learners = make_learners(environment, seed_base=600)
    state = build_checkpoint_state(
        source_learners,
        LEARNER_CONFIG,
        ROLLOUT_CONFIG,
        total_environment_transitions=0,
        completed_updates=0,
    )
    first_parameter_before = (
        next(restored_learners[0].ppo.actor_model.parameters()).detach().clone()
    )
    del state["learners"][environment.possible_agents[-1]]

    with pytest.raises(ValueError, match="agent IDs"):
        restore_checkpoint_state(state, restored_learners)

    first_parameter_after = next(
        restored_learners[0].ppo.actor_model.parameters()
    ).detach()
    torch.testing.assert_close(first_parameter_after, first_parameter_before)
