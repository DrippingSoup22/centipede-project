"""Construct and own the eight independent segment PPO learners.

This module composes reusable RL_lib models, algorithms, optimizers, and
normalizers. It does not collect environment transitions, calculate GAE, manage
checkpoints, or interpret MuJoCo state.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from rl_lib.algorithms.policy_gradient import PPO
from rl_lib.data import ObservationNormalizer
from rl_lib.models import GaussianPolicyNetwork, StateValueNetwork

from centipede.environment import AgentID, CentipedeParallelEnv


@dataclass(frozen=True)
class LearnerConfig:
    """Accepted first-version network, optimizer, PPO, and normalizer settings."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    initial_standard_deviation: float = 0.5
    standard_deviation_mode: str = "global"
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.001
    maximum_gradient_norm: float = 0.5
    normalization_mode: str = "running"
    normalizer_epsilon: float = 1e-8
    normalizer_clip: float = 10.0


@dataclass
class SegmentLearner:
    """Keep one segment's independent RL_lib objects and dimensions together."""

    agent_id: AgentID
    observation_size: int
    action_size: int
    ppo: PPO
    normalizer: ObservationNormalizer


def build_learners(
    environment: CentipedeParallelEnv,
    config: LearnerConfig,
    seeds: Mapping[AgentID, int],
) -> dict[AgentID, SegmentLearner]:
    """Construct one independent learner from every environment agent space."""

    expected_agents = set(environment.possible_agents)
    if set(seeds) != expected_agents:
        raise ValueError("Seeds must contain exactly one entry per possible agent")

    learners: dict[AgentID, SegmentLearner] = {}

    for agent in environment.possible_agents:
        observation_space = environment.observation_space(agent)
        action_space = environment.action_space(agent)
        if not isinstance(observation_space, spaces.Box) or not isinstance(
            action_space, spaces.Box
        ):
            raise TypeError("Centipede learners require Box observation/action spaces")
        if (
            observation_space.shape is None
            or len(observation_space.shape) != 1
            or action_space.shape is None
            or len(action_space.shape) != 1
        ):
            raise ValueError("Centipede learners require flat one-dimensional spaces")

        # Box spaces expose one lower and upper bound per action component.
        action_low = np.asarray(action_space.low, dtype=np.float32).copy()
        action_high = np.asarray(action_space.high, dtype=np.float32).copy()
        learners[agent] = _build_segment_learner(
            agent_id=agent,
            observation_size=observation_space.shape[0],
            action_low=action_low,
            action_high=action_high,
            config=config,
            seed=seeds[agent],
        )

    return learners


def _build_segment_learner(
    agent_id: AgentID,
    observation_size: int,
    action_low: NDArray[np.float32],
    action_high: NDArray[np.float32],
    config: LearnerConfig,
    seed: int,
) -> SegmentLearner:
    """Construct one actor, critic, two optimizers, PPO, and normalizer."""

    action_size = int(action_low.size)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)

        actor_model = GaussianPolicyNetwork(
            observation_size=observation_size,
            action_size=action_size,
            hidden_sizes=config.hidden_sizes,
            initial_std=config.initial_standard_deviation,
            std_mode=config.standard_deviation_mode,
        )
        critic_model = StateValueNetwork(
            observation_size=observation_size,
            hidden_sizes=config.hidden_sizes,
        )

    actor_optimizer = torch.optim.Adam(
        actor_model.parameters(), lr=config.actor_learning_rate
    )
    critic_optimizer = torch.optim.Adam(
        critic_model.parameters(), lr=config.critic_learning_rate
    )

    ppo = PPO(
        actor_model=actor_model,
        actor_optimizer=actor_optimizer,
        critic_model=critic_model,
        critic_optimizer=critic_optimizer,
        clip_ratio=config.clip_ratio,
        entropy_coefficient=config.entropy_coefficient,
        seed=seed,
        max_gradient_norm=config.maximum_gradient_norm,
        action_low=action_low,
        action_high=action_high,
    )

    normalizer = ObservationNormalizer(
        observation_size=observation_size,
        mode=config.normalization_mode,
        epsilon=config.normalizer_epsilon,
        clip=config.normalizer_clip,
    )

    return SegmentLearner(
        agent_id=agent_id,
        observation_size=observation_size,
        action_size=action_size,
        ppo=ppo,
        normalizer=normalizer,
    )
