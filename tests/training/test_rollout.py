"""Focused checks for rollout storage and coordinator composition."""

import multiprocessing
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rl_lib.data import ContinuousPPOActionSample, PPOUpdateResult

from centipede.environment import CentipedeParallelEnv
from centipede.training.environment_pool import (
    ProcessEnvironmentPool,
    SerialEnvironmentPool,
)
from centipede.training.learners import LearnerConfig, SegmentLearner, build_learners
from centipede.training.rollout import (
    PPOBatch,
    RolloutConfig,
    RolloutCoordinator,
    TrajectoryFragment,
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "assembly.xml"


@pytest.fixture
def environment() -> Iterator[CentipedeParallelEnv]:
    """Construct the public environment and release its MuJoCo resources."""
    instance = CentipedeParallelEnv(model_path=MODEL_PATH)
    try:
        yield instance
    finally:
        instance.close()


def build_environment_learners(
    environment: CentipedeParallelEnv,
) -> dict[int, SegmentLearner]:
    """Construct the independent learners used by coordinator checks."""
    seeds = {agent: 100 + agent for agent in environment.possible_agents}
    return build_learners(environment, LearnerConfig(), seeds)


def action_sample(
    action: list[float],
    *,
    log_probability: float,
    value: float,
) -> ContinuousPPOActionSample:
    """Build a small deterministic continuous-policy sample for storage tests."""
    action_array = np.asarray(action, dtype=np.float32)
    return ContinuousPPOActionSample(
        action=action_array,
        latent_action=action_array + np.float32(0.25),
        log_probability=log_probability,
        value=value,
    )


def test_constructor_binds_shared_learners_before_reset(
    environment: CentipedeParallelEnv,
) -> None:
    """Retain one learner set and leave episode storage empty until reset."""
    learners = build_environment_learners(environment)

    pool = SerialEnvironmentPool([environment])
    coordinator = RolloutCoordinator(pool, learners, RolloutConfig())

    assert coordinator.pool is pool
    assert coordinator.learners is not learners
    assert set(coordinator.learners) == set(learners)
    assert all(coordinator.learners[agent] is learners[agent] for agent in learners)
    assert coordinator._observations == []
    assert coordinator._fragments == []


def test_sample_actions_returns_normalized_inputs_and_continuous_samples(
    environment: CentipedeParallelEnv,
) -> None:
    """Keep the exact policy inputs beside each sampled bounded action."""
    learners = build_environment_learners(environment)
    coordinator = RolloutCoordinator(
        SerialEnvironmentPool([environment]), learners, RolloutConfig()
    )
    observations, _ = environment.reset(seed=23)

    normalized_observations, samples = coordinator._sample_actions(observations)

    assert set(normalized_observations) == set(environment.possible_agents)
    assert set(samples) == set(environment.possible_agents)
    for agent in environment.possible_agents:
        normalized = normalized_observations[agent]
        sample = samples[agent]

        assert normalized.shape == (learners[agent].observation_size,)
        assert normalized.dtype == np.float32
        assert learners[agent].normalizer.count == 1
        assert isinstance(sample, ContinuousPPOActionSample)
        assert sample.action.shape == (learners[agent].action_size,)
        assert sample.latent_action.shape == (learners[agent].action_size,)
        assert np.isfinite(sample.action).all()
        assert np.isfinite(sample.latent_action).all()
        assert np.isfinite(sample.log_probability)
        assert np.isfinite(sample.value)
        assert (sample.action >= -1.0).all()
        assert (sample.action <= 1.0).all()


def test_fragment_owns_appended_policy_measurements() -> None:
    """Protect a collected transition from later mutation of policy outputs."""
    observation = np.array([1.0, 2.0], dtype=np.float32)
    sample = action_sample([0.5], log_probability=-0.25, value=0.75)
    fragment = TrajectoryFragment(observation_size=2, action_size=1)

    fragment.append(observation, sample, reward=1.5)
    observation[:] = -10.0
    sample.action[:] = -10.0
    sample.latent_action[:] = -10.0

    assert np.array_equal(
        fragment.steps[0].state,
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert np.array_equal(
        fragment.steps[0].action,
        np.array([0.5], dtype=np.float32),
    )
    stored_policy_action = fragment.steps[0].policy_action
    assert stored_policy_action is not None
    assert np.array_equal(
        stored_policy_action,
        np.array([0.75], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("terminated", "expected_advantages", "expected_targets"),
    [
        (False, [2.5, 3.0], [3.0, 4.0]),
        (True, [1.5, 1.0], [2.0, 2.0]),
    ],
)
def test_fragment_finalization_distinguishes_bootstrap_from_termination(
    terminated: bool,
    expected_advantages: list[float],
    expected_targets: list[float],
) -> None:
    """Bootstrap a cutoff, but exclude state value after a true termination."""
    fragment = TrajectoryFragment(observation_size=2, action_size=1)
    fragment.append(
        np.array([1.0, 2.0], dtype=np.float32),
        action_sample([0.1], log_probability=-0.1, value=0.5),
        reward=1.0,
    )
    fragment.append(
        np.array([3.0, 4.0], dtype=np.float32),
        action_sample([0.2], log_probability=-0.2, value=1.0),
        reward=2.0,
    )

    batch = fragment.finalize(
        final_observation=np.array([5.0, 6.0], dtype=np.float32),
        final_value=4.0,
        terminated=terminated,
        config=RolloutConfig(discount=0.5, gae_lambda=1.0),
    )

    assert np.array_equal(
        batch.observations,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert np.array_equal(
        batch.latent_actions,
        np.array([[0.35], [0.45]], dtype=np.float32),
    )
    assert np.allclose(batch.old_log_probabilities, [-0.1, -0.2])
    assert np.allclose(batch.advantages, expected_advantages)
    assert np.allclose(batch.return_targets, expected_targets)


def test_collect_window_returns_one_isolated_batch_per_agent(
    environment: CentipedeParallelEnv,
) -> None:
    """Collect fixed-size batches without ending the live environment episode."""
    learners = build_environment_learners(environment)
    coordinator = RolloutCoordinator(
        SerialEnvironmentPool([environment]),
        learners,
        RolloutConfig(steps_per_environment=2),
    )
    coordinator.reset([23])

    batches = coordinator.collect_window()

    assert set(batches) == set(environment.possible_agents)
    for agent, batch in batches.items():
        assert batch.observations.shape == (2, learners[agent].observation_size)
        assert batch.latent_actions.shape == (2, learners[agent].action_size)
        assert batch.old_log_probabilities.shape == (2,)
        assert batch.advantages.shape == (2,)
        assert batch.return_targets.shape == (2,)
        assert np.isfinite(batch.observations).all()
        assert np.isfinite(batch.latent_actions).all()
        assert np.isfinite(batch.old_log_probabilities).all()
        assert np.isfinite(batch.advantages).all()
        assert np.isfinite(batch.return_targets).all()
        assert coordinator._fragments[0][agent].steps == []


def test_consecutive_windows_continue_the_same_physical_episode(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat a rollout cutoff as a training boundary rather than an episode end."""
    monkeypatch.setattr(
        environment,
        "_sample_target",
        lambda snapshot, random_generator: np.array([100.0, 100.0], dtype=np.float64),
    )
    learners = build_environment_learners(environment)
    coordinator = RolloutCoordinator(
        SerialEnvironmentPool([environment]),
        learners,
        RolloutConfig(steps_per_environment=1),
    )
    coordinator.reset([23])

    first_batches = coordinator.collect_window()
    assert environment.episode_steps == 1

    second_batches = coordinator.collect_window()
    assert environment.episode_steps == 2

    for agent in environment.possible_agents:
        assert first_batches[agent].observations.shape[0] == 1
        assert second_batches[agent].observations.shape[0] == 1
        assert learners[agent].normalizer.count == 2


def test_time_limit_splits_fragments_and_restarts_the_environment(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize each truncated episode before collecting from its successor."""
    monkeypatch.setattr(environment, "max_episode_steps", 1)
    monkeypatch.setattr(
        environment,
        "_sample_target",
        lambda snapshot, random_generator: np.array([100.0, 100.0], dtype=np.float64),
    )
    learners = build_environment_learners(environment)
    coordinator = RolloutCoordinator(
        SerialEnvironmentPool([environment]),
        learners,
        RolloutConfig(steps_per_environment=2),
    )
    coordinator.reset([23])

    batches = coordinator.collect_window()

    # Both one-step episodes were finalized separately and the environment is
    # ready at the initial state of the next episode.
    assert environment.episode_steps == 0
    assert environment.agents == environment.possible_agents
    for agent in environment.possible_agents:
        assert batches[agent].observations.shape[0] == 2
        assert coordinator._fragments[0][agent].steps == []


def test_two_environments_contribute_to_the_same_agent_batches(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combine replicas by agent without creating replica-specific learners."""
    second_environment = CentipedeParallelEnv(model_path=MODEL_PATH)
    try:
        learners = build_environment_learners(environment)
        normalizer_counts_at_step: list[tuple[int, ...]] = []
        original_step = environment.step

        def record_counts_before_step(actions: dict[int, np.ndarray]) -> object:
            normalizer_counts_at_step.append(
                tuple(learner.normalizer.count for learner in learners.values())
            )
            return original_step(actions)

        monkeypatch.setattr(environment, "step", record_counts_before_step)
        coordinator = RolloutCoordinator(
            SerialEnvironmentPool([environment, second_environment]),
            learners,
            RolloutConfig(steps_per_environment=2),
        )
        coordinator.reset([23, 24])

        progress: list[int] = []
        batches = coordinator.collect_window(on_transition=progress.append)

        assert progress == [1, 2, 3, 4]
        assert normalizer_counts_at_step == [(2,) * 8, (4,) * 8]
        for agent in environment.possible_agents:
            assert batches[agent].observations.shape == (
                4,
                learners[agent].observation_size,
            )
            assert batches[agent].latent_actions.shape == (
                4,
                learners[agent].action_size,
            )
            assert learners[agent].normalizer.count == 4
    finally:
        second_environment.close()


def test_update_learners_routes_each_batch_to_its_own_ppo(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call every independent PPO instance only with its matching agent data."""
    learners = build_environment_learners(environment)
    config = RolloutConfig(update_epochs=3, minibatch_size=7)
    coordinator = RolloutCoordinator(
        SerialEnvironmentPool([environment]), learners, config
    )
    batches: dict[int, PPOBatch] = {}
    update_mocks: dict[int, Mock] = {}
    expected_results: dict[int, tuple[PPOUpdateResult, ...]] = {}

    for agent, learner in learners.items():
        batch = PPOBatch(
            observations=np.full(
                (1, learner.observation_size), agent, dtype=np.float32
            ),
            latent_actions=np.full((1, learner.action_size), agent, dtype=np.float32),
            old_log_probabilities=np.array([-float(agent)], dtype=np.float32),
            advantages=np.array([float(agent)], dtype=np.float32),
            return_targets=np.array([float(agent + 1)], dtype=np.float32),
        )
        result = (
            PPOUpdateResult(
                actor_loss=float(agent),
                critic_loss=float(agent + 1),
                entropy=0.5,
            ),
        )
        update_mock = Mock(return_value=result)
        monkeypatch.setattr(learner.ppo, "update", update_mock)

        batches[agent] = batch
        update_mocks[agent] = update_mock
        expected_results[agent] = result

    results = coordinator.update_learners(batches)

    assert results == expected_results
    for agent, update_mock in update_mocks.items():
        assert update_mock.call_count == 1
        call_arguments = update_mock.call_args
        assert call_arguments is not None
        keyword_arguments = call_arguments.kwargs
        assert keyword_arguments["observations"] is batches[agent].observations
        assert keyword_arguments["actions"] is batches[agent].latent_actions
        assert (
            keyword_arguments["old_log_probabilities"]
            is batches[agent].old_log_probabilities
        )
        assert keyword_arguments["advantages"] is batches[agent].advantages
        assert keyword_arguments["return_targets"] is batches[agent].return_targets
        assert keyword_arguments["update_epochs"] == config.update_epochs
        assert keyword_arguments["minibatch_size"] == config.minibatch_size


def test_real_rollout_updates_every_segment_learner(
    environment: CentipedeParallelEnv,
) -> None:
    """Exercise MuJoCo collection, GAE, and one real PPO update per segment."""
    with torch.random.fork_rng(devices=[]):
        learners = build_environment_learners(environment)
        coordinator = RolloutCoordinator(
            SerialEnvironmentPool([environment]),
            learners,
            RolloutConfig(
                steps_per_environment=2,
                update_epochs=1,
                minibatch_size=2,
            ),
        )
        coordinator.reset([23])
        batches = coordinator.collect_window()
        parameters_before = {
            agent: [
                parameter.detach().clone()
                for model in (learner.ppo.actor_model, learner.ppo.critic_model)
                for parameter in model.parameters()
            ]
            for agent, learner in learners.items()
        }

        results = coordinator.update_learners(batches)

        assert set(results) == set(environment.possible_agents)
        for agent, learner in learners.items():
            assert batches[agent].observations.shape[0] == 2
            assert len(results[agent]) == 1
            update = results[agent][0]
            assert np.isfinite(
                [
                    update.actor_loss,
                    update.critic_loss,
                    update.entropy,
                    update.approximate_kl,
                    update.clip_fraction,
                ]
            ).all()

            parameters_after = [
                parameter.detach()
                for model in (learner.ppo.actor_model, learner.ppo.critic_model)
                for parameter in model.parameters()
            ]
            assert any(
                not torch.equal(before, after)
                for before, after in zip(
                    parameters_before[agent], parameters_after, strict=True
                )
            )


@pytest.mark.parametrize("worker_count", [2, 4])
def test_process_rollout_collects_repeated_windows_across_episode_resets(
    worker_count: int,
) -> None:
    """Exercise the complete spawned collection path without crossing episodes."""
    reference = CentipedeParallelEnv(model_path=MODEL_PATH, max_episode_steps=1)
    with torch.random.fork_rng(devices=[]):
        learners = build_environment_learners(reference)
    reference.close()

    pool = ProcessEnvironmentPool(
        model_path=MODEL_PATH,
        max_episode_steps=1,
        environment_count=worker_count,
        worker_count=worker_count,
    )
    worker_pids = {process.pid for process in pool._processes}

    try:
        steps_per_environment = 2
        coordinator = RolloutCoordinator(
            pool,
            learners,
            RolloutConfig(steps_per_environment=steps_per_environment),
        )
        coordinator.reset([700 + index for index in range(worker_count)])

        first_window = coordinator.collect_window()
        second_window = coordinator.collect_window()

        expected_samples = worker_count * steps_per_environment
        for agent, learner in learners.items():
            for batch in (first_window[agent], second_window[agent]):
                assert batch.observations.shape == (
                    expected_samples,
                    learner.observation_size,
                )
                assert batch.latent_actions.shape == (
                    expected_samples,
                    learner.action_size,
                )
                assert np.isfinite(batch.observations).all()
                assert np.isfinite(batch.return_targets).all()

        # A one-step episode ends on every transition. Empty live fragments here
        # show that each terminal observation was finalized before its reset and
        # no fragment leaked into the next episode or rollout window.
        assert all(
            not fragment.steps
            for environment_fragments in coordinator._fragments
            for fragment in environment_fragments.values()
        )
    finally:
        pool.close()

    active_pids = {process.pid for process in multiprocessing.active_children()}
    assert worker_pids.isdisjoint(active_pids)
