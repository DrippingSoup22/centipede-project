"""Checks for the public PettingZoo environment."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from centipede.environment import (
    TARGET_BEARING_LIMIT_RAD,
    TARGET_DISTANCE_RANGE_M,
    CentipedeParallelEnv,
)
from centipede.observations import (
    HEAD_OBSERVATION_SIZE,
    INTERIOR_OBSERVATION_SIZE,
    REAR_OBSERVATION_SIZE,
    head_target_displacement,
)
from centipede.rewards import RewardTerms

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "assembly.xml"


@pytest.fixture
def environment() -> CentipedeParallelEnv:
    """Construct the public environment and always release MuJoCo resources."""
    instance = CentipedeParallelEnv(model_path=MODEL_PATH)
    try:
        yield instance
    finally:
        instance.simulation.close()


def zero_actions(environment: CentipedeParallelEnv) -> dict[int, np.ndarray]:
    """Build one neutral action for every currently active segment."""
    return {
        agent: np.zeros(environment.action_space(agent).shape, dtype=np.float32)
        for agent in environment.agents
    }


def test_constructor_derives_agents_and_initial_task_state(
    environment: CentipedeParallelEnv,
) -> None:
    """Use model-owned segment IDs and leave episode state inactive until reset."""
    assert environment.possible_agents == list(range(8))
    assert environment.agents == []
    assert environment.target_position is None
    assert environment.previous_snapshot is None
    assert environment.episode_steps == 0
    assert environment.max_episode_steps == 1_000


def test_constructor_caches_action_spaces_from_simulation(
    environment: CentipedeParallelEnv,
) -> None:
    """Give every discovered segment one six-value normalized action space."""
    assert set(environment.action_spaces) == set(environment.possible_agents)

    for action_space in environment.action_spaces.values():
        assert action_space.shape == (6,)
        assert action_space.dtype == np.float32
        np.testing.assert_array_equal(action_space.low, -1.0)
        np.testing.assert_array_equal(action_space.high, 1.0)


def test_constructor_caches_observation_spaces_by_segment_position(
    environment: CentipedeParallelEnv,
) -> None:
    """Match head, interior, and rear shapes without padding absent neighbors."""
    expected_shapes = {
        0: (HEAD_OBSERVATION_SIZE,),
        **{
            agent: (INTERIOR_OBSERVATION_SIZE,)
            for agent in environment.possible_agents[1:-1]
        },
        7: (REAR_OBSERVATION_SIZE,),
    }

    assert set(environment.observation_spaces) == set(environment.possible_agents)
    for agent, observation_space in environment.observation_spaces.items():
        assert observation_space.shape == expected_shapes[agent]
        assert observation_space.dtype == np.float32
        assert np.isneginf(observation_space.low).all()
        assert np.isposinf(observation_space.high).all()


def test_observation_space_returns_cached_agent_space(
    environment: CentipedeParallelEnv,
) -> None:
    """Return the stable object created for each agent during construction."""
    for agent in environment.possible_agents:
        assert environment.observation_space(agent) is environment.observation_spaces[
            agent
        ]


def test_action_space_returns_cached_agent_space(
    environment: CentipedeParallelEnv,
) -> None:
    """Return the stable action-space object created during construction."""
    for agent in environment.possible_agents:
        assert environment.action_space(agent) is environment.action_spaces[agent]


def test_reset_reproduces_physics_target_and_observations(
    environment: CentipedeParallelEnv,
) -> None:
    """Make one public seed reproduce both independent reset streams."""
    first_observations, first_infos = environment.reset(seed=23)
    first_snapshot = environment.previous_snapshot
    first_target = environment.target_position.copy()
    second_observations, second_infos = environment.reset(seed=23)
    second_snapshot = environment.previous_snapshot

    assert first_snapshot is not None
    assert second_snapshot is not None
    np.testing.assert_array_equal(
        first_snapshot.leg_joint_position,
        second_snapshot.leg_joint_position,
    )
    np.testing.assert_array_equal(
        first_snapshot.leg_joint_velocity,
        second_snapshot.leg_joint_velocity,
    )
    np.testing.assert_array_equal(first_target, environment.target_position)
    for agent in environment.possible_agents:
        np.testing.assert_array_equal(
            first_observations[agent],
            second_observations[agent],
        )
    assert first_infos == second_infos == {
        agent: {} for agent in environment.possible_agents
    }
    assert environment.agents == environment.possible_agents
    assert environment.episode_steps == 0


def test_reset_without_seed_advances_public_random_sequence(
    environment: CentipedeParallelEnv,
) -> None:
    """Continue rather than restart random state on an unseeded reset."""
    environment.reset(seed=23)
    first_target = environment.target_position.copy()
    environment.reset()

    assert not np.array_equal(first_target, environment.target_position)


def test_target_sampling_is_reproducible(
    environment: CentipedeParallelEnv,
) -> None:
    """Generate the same target from the same snapshot and random seed."""
    environment.simulation.reset(seed=5)
    snapshot = environment.simulation.snapshot()

    first_target = environment._sample_target(snapshot, np.random.default_rng(17))
    second_target = environment._sample_target(snapshot, np.random.default_rng(17))

    np.testing.assert_array_equal(first_target, second_target)


def test_target_sampling_respects_head_relative_wedge(
    environment: CentipedeParallelEnv,
) -> None:
    """Keep sampled targets within the configured distance and forward bearing."""
    environment.simulation.reset(seed=5)
    snapshot = environment.simulation.snapshot()
    random_generator = np.random.default_rng(17)

    for _ in range(200):
        target = environment._sample_target(snapshot, random_generator)
        forward, lateral = head_target_displacement(snapshot, target)
        distance = np.hypot(forward, lateral)
        bearing = np.arctan2(lateral, forward)

        assert target.shape == (2,)
        assert target.dtype == np.float64
        assert np.isfinite(target).all()
        assert TARGET_DISTANCE_RANGE_M[0] <= distance <= TARGET_DISTANCE_RANGE_M[1]
        assert -TARGET_BEARING_LIMIT_RAD <= bearing <= TARGET_BEARING_LIMIT_RAD
        assert forward > 0.0


def test_step_advances_one_shared_transition_and_returns_all_outputs(
    environment: CentipedeParallelEnv,
) -> None:
    """Advance MuJoCo once and publish finite results for every active segment."""
    environment.reset(seed=23)
    active_agents = environment.agents.copy()
    initial_time = environment.simulation.data.time

    observations, rewards, terminations, truncations, infos = environment.step(
        zero_actions(environment)
    )

    expected_agents = set(active_agents)
    assert set(observations) == expected_agents
    assert set(rewards) == expected_agents
    assert set(terminations) == expected_agents
    assert set(truncations) == expected_agents
    assert set(infos) == expected_agents
    assert environment.simulation.data.time == pytest.approx(
        initial_time + environment.simulation.dt
    )
    assert environment.episode_steps == 1
    assert environment.agents == active_agents
    assert all(np.isfinite(observation).all() for observation in observations.values())
    assert all(np.isfinite(reward) for reward in rewards.values())
    assert not any(terminations.values())
    assert not any(truncations.values())


def test_step_requires_an_active_reset_episode(
    environment: CentipedeParallelEnv,
) -> None:
    """Reject stepping before reset and after an episode has ended."""
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        environment.step({})


def test_step_validation_failure_does_not_advance_episode(
    environment: CentipedeParallelEnv,
) -> None:
    """Leave physical and task time unchanged when the action dictionary is invalid."""
    environment.reset(seed=23)
    actions = zero_actions(environment)
    del actions[4]
    initial_time = environment.simulation.data.time
    initial_snapshot = environment.previous_snapshot

    with pytest.raises(ValueError, match="Action keys must match active agents"):
        environment.step(actions)

    assert environment.simulation.data.time == initial_time
    assert environment.episode_steps == 0
    assert environment.previous_snapshot is initial_snapshot


def test_step_arrival_terminates_every_agent_with_final_outputs(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Award arrival and retain final outputs before removing all active agents."""
    environment.reset(seed=23)
    final_snapshot = environment.previous_snapshot
    assert final_snapshot is not None
    environment.target_position = final_snapshot.head_tip_position[:2].copy()
    active_agents = environment.agents.copy()
    applied_actions: list[np.ndarray] = []

    monkeypatch.setattr(
        environment.simulation,
        "step",
        lambda action: applied_actions.append(action.copy()),
    )
    monkeypatch.setattr(environment.simulation, "snapshot", lambda: final_snapshot)

    observations, rewards, terminations, truncations, infos = environment.step(
        zero_actions(environment)
    )

    assert len(applied_actions) == 1
    assert applied_actions[0].shape == (48,)
    assert set(observations) == set(active_agents)
    assert set(rewards) == set(active_agents)
    assert all(terminations.values())
    assert not any(truncations.values())
    assert all(info["reward_arrival"] == 1.0 for info in infos.values())
    assert infos[0]["target_reached"] is True
    assert infos[0]["episode_end"] == "arrival"
    assert environment.previous_snapshot is final_snapshot
    assert environment.agents == []


def test_step_time_limit_truncates_every_agent(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncate without arrival when the completed transition reaches the limit."""
    environment.reset(seed=23)
    final_snapshot = environment.previous_snapshot
    assert final_snapshot is not None
    environment.max_episode_steps = 1
    monkeypatch.setattr(environment.simulation, "step", lambda action: None)
    monkeypatch.setattr(environment.simulation, "snapshot", lambda: final_snapshot)

    observations, rewards, terminations, truncations, infos = environment.step(
        zero_actions(environment)
    )

    assert set(observations) == set(environment.possible_agents)
    assert not any(terminations.values())
    assert all(truncations.values())
    assert infos[0]["target_reached"] is False
    assert infos[0]["episode_end"] == "time_limit"
    assert infos[0]["episode_steps"] == 1
    for agent in environment.possible_agents:
        documented_terms = (
            infos[agent]["reward_arrival"]
            + infos[agent]["reward_efficiency"]
            + infos[agent]["reward_body_contact"]
            + infos[agent]["reward_leg_contact"]
        )
        assert rewards[agent] == pytest.approx(documented_terms)
    assert environment.agents == []


def test_step_arrival_precedes_time_limit_on_final_step(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report termination rather than truncation when arrival occurs at the limit."""
    environment.reset(seed=23)
    final_snapshot = environment.previous_snapshot
    assert final_snapshot is not None
    environment.max_episode_steps = 1
    environment.target_position = final_snapshot.head_tip_position[:2].copy()
    monkeypatch.setattr(environment.simulation, "step", lambda action: None)
    monkeypatch.setattr(environment.simulation, "snapshot", lambda: final_snapshot)

    _, _, terminations, truncations, infos = environment.step(
        zero_actions(environment)
    )

    assert all(terminations.values())
    assert not any(truncations.values())
    assert infos[0]["episode_end"] == "arrival"

    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        environment.step({})


def test_completed_environment_passes_parallel_api_contract() -> None:
    """Satisfy PettingZoo's standard simultaneous-agent lifecycle checks."""
    environment = CentipedeParallelEnv(model_path=MODEL_PATH, max_episode_steps=3)
    try:
        parallel_api_test(environment, num_cycles=10)
    finally:
        environment.close()


def test_validate_actions_returns_flat_float32_in_segment_order(
    environment: CentipedeParallelEnv,
) -> None:
    """Convert accepted actions once and concatenate them by active segment ID."""
    environment.reset(seed=23)
    actions = {
        agent: np.full(6, agent / 10.0, dtype=np.float64)
        for agent in reversed(environment.agents)
    }

    accepted = environment._validate_actions(actions)

    assert accepted.shape == (48,)
    assert accepted.dtype == np.float32
    np.testing.assert_array_equal(
        accepted,
        np.concatenate(
            [np.full(6, agent / 10.0, dtype=np.float32) for agent in range(8)]
        ),
    )


@pytest.mark.parametrize(
    ("invalid_action", "error_type", "message"),
    [
        (np.zeros(5), ValueError, "shape"),
        (np.full(6, np.nan), ValueError, "finite"),
        (np.full(6, np.inf), ValueError, "finite"),
        (np.full(6, 1.01), ValueError, "within"),
        (np.full(6, -1.01), ValueError, "within"),
        (np.full(6, "invalid"), TypeError, "real numbers"),
        (np.full(6, 0.5 + 0.1j), TypeError, "real numbers"),
    ],
)
def test_validate_actions_rejects_invalid_values_before_physics(
    environment: CentipedeParallelEnv,
    invalid_action: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject one malformed agent action without advancing simulation time."""
    environment.reset(seed=23)
    actions = {
        agent: np.zeros(6, dtype=np.float32) for agent in environment.agents
    }
    actions[3] = invalid_action
    initial_time = environment.simulation.data.time

    with pytest.raises(error_type, match=message):
        environment._validate_actions(actions)

    assert environment.simulation.data.time == initial_time


@pytest.mark.parametrize("unknown_agent", [None, 8])
def test_validate_actions_requires_exact_active_agent_keys(
    environment: CentipedeParallelEnv,
    unknown_agent: int | None,
) -> None:
    """Reject missing and extra agent keys before inspecting action values."""
    environment.reset(seed=23)
    actions = {
        agent: np.zeros(6, dtype=np.float32) for agent in environment.agents
    }
    if unknown_agent is None:
        del actions[4]
    else:
        actions[unknown_agent] = np.zeros(6, dtype=np.float32)
    initial_time = environment.simulation.data.time

    with pytest.raises(ValueError, match="Action keys must match active agents"):
        environment._validate_actions(actions)

    assert environment.simulation.data.time == initial_time


def test_build_infos_exposes_local_diagnostics_and_head_summary(
    environment: CentipedeParallelEnv,
) -> None:
    """Report local contacts and rewards while reserving task facts for the head."""
    environment.reset(seed=23)
    snapshot = environment.previous_snapshot
    assert snapshot is not None

    body_contacts = np.zeros(8, dtype=np.bool_)
    leg_contacts = np.zeros(8, dtype=np.bool_)
    body_contacts[2] = True
    leg_contacts[3] = True
    snapshot = replace(
        snapshot,
        body_ground_contact=body_contacts,
        leg_leg_contact=leg_contacts,
    )
    environment.target_position = snapshot.head_tip_position[:2] + np.array(
        [0.003, 0.004]
    )
    environment.episode_steps = 7
    reward_terms = {
        agent: RewardTerms(1.0, -0.003, -0.010, -0.005)
        for agent in environment.possible_agents
    }

    infos = environment._build_infos(
        snapshot,
        reward_terms=reward_terms,
        target_reached=True,
        episode_end="arrival",
    )

    assert set(infos) == set(environment.possible_agents)
    assert infos[2]["body_ground_contact"] is True
    assert infos[3]["leg_leg_contact"] is True
    assert infos[1]["reward_arrival"] == 1.0
    assert infos[1]["reward_efficiency"] == -0.003
    assert infos[1]["reward_body_contact"] == -0.010
    assert infos[1]["reward_leg_contact"] == -0.005
    assert infos[0]["target_distance_m"] == pytest.approx(0.005)
    assert infos[0]["target_reached"] is True
    assert infos[0]["episode_end"] == "arrival"
    assert infos[0]["episode_steps"] == 7
    assert infos[0]["episode_time_s"] == pytest.approx(
        7 * environment.simulation.dt
    )

    for follower in environment.possible_agents[1:]:
        assert "target_distance_m" not in infos[follower]
        assert "target_reached" not in infos[follower]
        assert "episode_end" not in infos[follower]


def test_close_delegates_to_simulation(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release resources through the Gymnasium MuJoCo owner."""
    close_calls = 0

    def record_close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(environment.simulation, "close", record_close)

    environment.close()

    assert close_calls == 1


def test_render_delegates_to_simulation(
    environment: CentipedeParallelEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the image produced by the Gymnasium MuJoCo rendering owner."""
    expected_image = np.zeros((3, 4, 3), dtype=np.uint8)
    render_calls = 0

    def render_image() -> np.ndarray:
        nonlocal render_calls
        render_calls += 1
        return expected_image

    monkeypatch.setattr(environment.simulation, "render", render_image)

    rendered = environment.render()

    assert rendered is expected_image
    assert render_calls == 1


def test_reset_accepts_options_without_changing_seeded_result(
    environment: CentipedeParallelEnv,
) -> None:
    """Keep unused PettingZoo reset options neutral in the first version."""
    first_observations, _ = environment.reset(seed=23)
    first_target = environment.target_position.copy()

    second_observations, _ = environment.reset(
        seed=23,
        options={"unused": True},
    )

    np.testing.assert_array_equal(first_target, environment.target_position)
    for agent in environment.possible_agents:
        np.testing.assert_array_equal(
            first_observations[agent],
            second_observations[agent],
        )


@pytest.mark.parametrize("max_episode_steps", [0, -1])
def test_constructor_rejects_non_positive_episode_length(
    max_episode_steps: int,
) -> None:
    """Reject invalid task configuration before constructing MuJoCo resources."""
    with pytest.raises(ValueError, match="Max episode steps must be > 0"):
        CentipedeParallelEnv(
            model_path=MODEL_PATH,
            max_episode_steps=max_episode_steps,
        )
