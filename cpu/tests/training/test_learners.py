"""Focused checks for independent PPO learner construction."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from centipede.environment import CentipedeParallelEnv
from centipede.training.learners import LearnerConfig, SegmentLearner, build_learners

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "assembly.xml"


@pytest.fixture
def environment() -> Iterator[CentipedeParallelEnv]:
    """Construct the public environment and release its MuJoCo resources."""
    instance = CentipedeParallelEnv(model_path=MODEL_PATH)
    try:
        yield instance
    finally:
        instance.close()


def learner_seeds(environment: CentipedeParallelEnv) -> dict[int, int]:
    """Give every segment a stable, distinct construction seed."""
    return {agent: 100 + agent for agent in environment.possible_agents}


def test_build_learners_uses_each_agent_space_and_action_bounds(
    environment: CentipedeParallelEnv,
) -> None:
    """Build the expected dimensions and bounded actions for every segment."""
    learners = build_learners(
        environment,
        LearnerConfig(),
        learner_seeds(environment),
    )

    assert set(learners) == set(environment.possible_agents)
    for agent, learner in learners.items():
        observation_space = environment.observation_space(agent)
        action_space = environment.action_space(agent)
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Box)
        assert observation_space.shape is not None
        assert action_space.shape is not None

        assert isinstance(learner, SegmentLearner)
        assert learner.agent_id == agent
        assert learner.observation_size == observation_space.shape[0]
        assert learner.action_size == action_space.shape[0]

        observation = np.zeros(learner.observation_size, dtype=np.float32)
        sample = learner.ppo.sample_action(observation)
        assert sample.action.shape == (learner.action_size,)
        assert np.isfinite(sample.action).all()
        assert (sample.action >= -1.0).all()
        assert (sample.action <= 1.0).all()


def test_build_learners_does_not_share_mutable_learning_state(
    environment: CentipedeParallelEnv,
) -> None:
    """Give every segment separate models, optimizers, and normalizers."""
    learners = build_learners(
        environment,
        LearnerConfig(),
        learner_seeds(environment),
    )

    assert len({id(learner.ppo.actor_model) for learner in learners.values()}) == 8
    assert len({id(learner.ppo.critic_model) for learner in learners.values()}) == 8
    assert len({id(learner.ppo.actor_optimizer) for learner in learners.values()}) == 8
    assert len({id(learner.ppo.critic_optimizer) for learner in learners.values()}) == 8
    assert len({id(learner.normalizer) for learner in learners.values()}) == 8


def test_build_learners_requires_one_seed_per_agent(
    environment: CentipedeParallelEnv,
) -> None:
    """Reject missing or unrelated seed entries before constructing models."""
    with pytest.raises(ValueError, match="exactly one entry"):
        build_learners(environment, LearnerConfig(), {0: 100})
