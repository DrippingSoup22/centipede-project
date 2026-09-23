"""Coordinate synchronized rollout collection and independent PPO updates.

Environment replicas use the same segment learners. Trajectories remain
separate by environment and agent until GAE is calculated; finalized per-agent
batches may then concatenate data from several replicas or episode fragments.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from rl_lib.algorithms.policy_gradient import generalized_advantage_estimates
from rl_lib.data import (
    ContinuousPPOActionSample,
    EpisodeStep,
    PPOUpdateResult,
    rollout_arrays,
)

from centipede.environment import AgentID, Observation
from centipede.training.environment_pool import (
    AgentActions,
    AgentObservations,
    EnvironmentPool,
    EnvironmentStep,
)
from centipede.training.learners import SegmentLearner

_PolicyOutput: TypeAlias = tuple[
    AgentObservations,
    dict[AgentID, ContinuousPPOActionSample],
]


@dataclass(frozen=True)
class RolloutConfig:
    """Accepted first-version collection, GAE, and PPO update settings."""

    steps_per_environment: int = 256
    discount: float = 0.999
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 64


@dataclass(frozen=True)
class PPOBatch:
    """One agent's finalized arrays ready for an RL_lib PPO update."""

    observations: NDArray[np.float32]
    latent_actions: NDArray[np.float32]
    old_log_probabilities: NDArray[np.float32]
    advantages: NDArray[np.float32]
    return_targets: NDArray[np.float32]


@dataclass
class TrajectoryFragment:
    """Add PPO-only measurements to RL_lib's generic consecutive rollout steps.

    ``EpisodeStep`` owns observations, environment actions, latent policy actions,
    and rewards. This wrapper retains only the frozen log probabilities and values
    that PPO additionally needs while one episode fragment is being collected.
    """

    observation_size: int
    action_size: int
    steps: list[EpisodeStep[NDArray[np.float32]]] = field(default_factory=list)
    old_log_probabilities: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def append(
        self,
        normalized_observation: NDArray[np.float32],
        sample: ContinuousPPOActionSample,
        reward: float,
    ) -> None:
        """Store one trusted transition measurement for this agent."""

        step = EpisodeStep(
            state=normalized_observation.copy(),
            action=sample.action.copy(),
            reward=float(reward),
            policy_action=sample.latent_action.copy(),
        )

        self.steps.append(step)
        self.old_log_probabilities.append(float(sample.log_probability))
        self.values.append(float(sample.value))

    def finalize(
        self,
        final_observation: NDArray[np.float32],
        final_value: float,
        *,
        terminated: bool,
        config: RolloutConfig,
    ) -> PPOBatch:
        """Use RL_lib rollout conversion and GAE without crossing a reset."""

        arrays = rollout_arrays(
            self.steps,
            final_observation,
            observation_size=self.observation_size,
            action_size=self.action_size,
        )

        advantages, return_targets = generalized_advantage_estimates(
            rewards=arrays.rewards,
            values=self.values,
            final_value=final_value,
            terminated=terminated,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
        )

        return PPOBatch(
            observations=arrays.observations,
            # Providing ``action_size`` above selects RL_lib's continuous branch.
            latent_actions=cast(NDArray[np.float32], arrays.actions),
            old_log_probabilities=np.asarray(
                self.old_log_probabilities, dtype=np.float32
            ),
            advantages=advantages,
            return_targets=return_targets,
        )


class RolloutCoordinator:
    """Collect one fixed window and update all segment learners synchronously."""

    def __init__(
        self,
        pool: EnvironmentPool,
        learners: Mapping[AgentID, SegmentLearner],
        config: RolloutConfig,
    ) -> None:
        """Bind an environment pool to one shared set of segment learners."""

        # The pool owns environment lifecycle and stepping. The same learner
        # objects are intentionally reused across all of its physical replicas.
        self.pool = pool
        self.learners = dict(learners)
        self.config = config

        # reset() creates one observation dictionary and one independent fragment
        # dictionary per environment replica.
        self._observations: list[AgentObservations] = []
        self._fragments: list[dict[AgentID, TrajectoryFragment]] = []

    def reset(self, seeds: Sequence[int | None]) -> None:
        """Reset every environment and retain its initial observation dictionary."""

        self._observations = self.pool.reset_all(seeds)
        self._fragments = [
            self._new_fragments() for _ in range(self.pool.environment_count)
        ]

    def collect_window(
        self, on_transition: Callable[[int], None] | None = None
    ) -> dict[AgentID, PPOBatch]:
        """Collect one window and optionally report completed environment steps."""

        finalized_batches: dict[AgentID, list[PPOBatch]] = {
            agent: [] for agent in self.learners
        }
        collected_transitions = 0

        # Policies stay fixed until every replica contributes the configured
        # number of transitions to this optimization window.
        for _ in range(self.config.steps_per_environment):
            # Sample every replica before physics starts. A process pool can then
            # dispatch the complete action batch without changing policy order.
            policy_outputs = self._sample_all_actions()
            actions: list[AgentActions] = [
                {agent: sample.action for agent, sample in samples.items()}
                for _, samples in policy_outputs
            ]
            step_results = self.pool.step_all(actions)
            ended_indices = self._record_transitions(
                policy_outputs,
                step_results,
                finalized_batches,
            )
            collected_transitions += self.pool.environment_count

            # Final observations were consumed above; only now may ended replicas
            # continue their random streams from fresh task episodes.
            if ended_indices:
                for environment_index, observations in self.pool.reset_indices(
                    ended_indices
                ).items():
                    self._observations[environment_index] = observations

            # Progress reflects completed environment transitions and is reported
            # after all ending replicas are ready for the next synchronous step.
            if on_transition is not None:
                first_transition = collected_transitions - self.pool.environment_count
                for transition in range(
                    first_transition + 1, collected_transitions + 1
                ):
                    on_transition(transition)

        self._finalize_window_cutoffs(finalized_batches)

        return {
            agent: self._combine_batches(agent_batches)
            for agent, agent_batches in finalized_batches.items()
        }

    def update_learners(
        self,
        batches: Mapping[AgentID, PPOBatch],
    ) -> dict[AgentID, tuple[PPOUpdateResult, ...]]:
        """Update each learner only from its own finalized rollout batch."""

        results: dict[AgentID, tuple[PPOUpdateResult, ...]] = {}

        for agent, batch in batches.items():
            learner = self.learners[agent]

            results[agent] = learner.ppo.update(
                observations=batch.observations,
                actions=batch.latent_actions,
                old_log_probabilities=batch.old_log_probabilities,
                advantages=batch.advantages,
                return_targets=batch.return_targets,
                update_epochs=self.config.update_epochs,
                minibatch_size=self.config.minibatch_size,
            )

        return results

    def _new_fragments(self) -> dict[AgentID, TrajectoryFragment]:
        """Create one empty trajectory fragment for every independent learner."""

        return {
            agent: TrajectoryFragment(
                observation_size=learner.observation_size,
                action_size=learner.action_size,
            )
            for agent, learner in self.learners.items()
        }

    def _sample_all_actions(self) -> list[_PolicyOutput]:
        """Sample every replica before dispatching any environment transition."""

        return [
            self._sample_actions(observations) for observations in self._observations
        ]

    def _record_transitions(
        self,
        policy_outputs: Sequence[_PolicyOutput],
        step_results: Sequence[EnvironmentStep],
        finalized_batches: Mapping[AgentID, list[PPOBatch]],
    ) -> list[int]:
        """Store ordered pool results and return replicas whose episodes ended."""

        ended_indices: list[int] = []

        # Pool results follow stable replica order, so fragments and running
        # normalizers keep the same deterministic routing as serial collection.
        for environment_index, (
            (normalized_observations, samples),
            step_result,
        ) in enumerate(zip(policy_outputs, step_results, strict=True)):
            next_observations, rewards, terminations, truncations, _ = step_result
            fragments = self._fragments[environment_index]

            for agent in self.learners:
                fragments[agent].append(
                    normalized_observation=normalized_observations[agent],
                    sample=samples[agent],
                    reward=rewards[agent],
                )

            if any(terminations.values()) or any(truncations.values()):
                self._finalize_environment(
                    environment_index,
                    next_observations,
                    terminations,
                    finalized_batches,
                )
                ended_indices.append(environment_index)
            else:
                self._observations[environment_index] = next_observations

        return ended_indices

    def _finalize_environment(
        self,
        environment_index: int,
        final_observations: AgentObservations,
        terminations: Mapping[AgentID, bool],
        finalized_batches: Mapping[AgentID, list[PPOBatch]],
    ) -> None:
        """Finalize all agent fragments at one shared episode boundary."""

        for agent in self.learners:
            finalized_batches[agent].append(
                self._finalize_fragment(
                    environment_index,
                    agent,
                    final_observations[agent],
                    terminated=terminations[agent],
                )
            )
        self._fragments[environment_index] = self._new_fragments()

    def _finalize_window_cutoffs(
        self,
        finalized_batches: Mapping[AgentID, list[PPOBatch]],
    ) -> None:
        """Bootstrap active fragments without resetting their physical episodes."""

        for environment_index, final_observations in enumerate(self._observations):
            for agent in self.learners:
                fragment = self._fragments[environment_index][agent]
                if fragment.steps:
                    finalized_batches[agent].append(
                        self._finalize_fragment(
                            environment_index,
                            agent,
                            final_observations[agent],
                            terminated=False,
                        )
                    )
            self._fragments[environment_index] = self._new_fragments()

    def _sample_actions(
        self,
        observations: AgentObservations,
    ) -> _PolicyOutput:
        """Return normalized inputs and samples; samples already contain actions."""

        normalized_observations: AgentObservations = {}
        samples: dict[AgentID, ContinuousPPOActionSample] = {}

        for agent in observations:
            learner = self.learners[agent]
            normalized = learner.normalizer.normalize(observations[agent], update=True)
            # Learner construction always pairs PPO with a Gaussian actor, so its
            # sample follows the continuous branch of RL_lib's union return type.
            sample = cast(
                ContinuousPPOActionSample,
                learner.ppo.sample_action(normalized),
            )

            normalized_observations[agent] = normalized
            samples[agent] = sample

        return normalized_observations, samples

    def _finalize_fragment(
        self,
        environment_index: int,
        agent_id: AgentID,
        final_observation: Observation,
        *,
        terminated: bool,
    ) -> PPOBatch:
        """Close one environment-agent fragment with zero or critic bootstrap."""

        learner = self.learners[agent_id]
        fragment = self._fragments[environment_index][agent_id]

        normalized_final_observation = learner.normalizer.normalize(
            final_observation, update=False
        )

        final_value = (
            0.0 if terminated else learner.ppo.state_value(normalized_final_observation)
        )
        return fragment.finalize(
            final_observation=normalized_final_observation,
            final_value=final_value,
            terminated=terminated,
            config=self.config,
        )

    @staticmethod
    def _combine_batches(batches: Sequence[PPOBatch]) -> PPOBatch:
        """Concatenate finalized fragments belonging to one learner only."""

        return PPOBatch(
            observations=np.concatenate([batch.observations for batch in batches]),
            latent_actions=np.concatenate([batch.latent_actions for batch in batches]),
            old_log_probabilities=np.concatenate(
                [batch.old_log_probabilities for batch in batches]
            ),
            advantages=np.concatenate([batch.advantages for batch in batches]),
            return_targets=np.concatenate([batch.return_targets for batch in batches]),
        )
