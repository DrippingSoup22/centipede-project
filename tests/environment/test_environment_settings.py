"""Tests for the [environment] configuration section and its nested sections.

The generic value checks are tested with ``SettingsSection`` and the simulation
section with ``SimulationSettings``; these tests cover only the environment's
keys, defaults, and how the nested sections are handed on.
"""

import pytest

from centipede.environment.settings import (
    ClockSettings,
    EnvironmentSettings,
    RewardSettings,
    TargetSettings,
)
from centipede.settings_section import SettingsError

SIMULATION = {"model_path": "models/assembly_v3.xml", "backend": "cpu"}


def read(**values) -> EnvironmentSettings:
    return EnvironmentSettings.from_section({"simulation": SIMULATION} | values)


def test_defaults_fill_everything_but_the_simulation_requirements():
    settings = read()

    assert settings.max_episode_steps == 8192
    assert settings.observation_radius == 1
    assert not settings.spine_control  # as in every run saved before it existed
    assert settings.simulation.world_count == 1
    assert settings.target == TargetSettings(
        distance_range_m=(0.030, 0.060),
        bearing_range_deg=(-30.0, 30.0),
        arrival="head",
        arrival_radius_m=None,
        range_circle_ratio=2.5,
        range_circle_margin_m=0.0,
        after_arrival="restart",
    )
    # A file with the tip's radius, as every run saved before the head arrival,
    # keeps the tip, with no range circle.
    tip = read(target={"arrival_radius_m": 0.001}).target
    assert (tip.arrival, tip.arrival_radius_m, tip.range_circle_ratio) == (
        "tip",
        0.001,
        None,
    )
    assert not settings.passive_follower_spine and not settings.clocks
    assert settings.clock == ClockSettings(middle_tempo_hz=2.0, tempo_range_octaves=1.0)
    assert settings.rewards == RewardSettings(
        arrival_reward=1.0,
        arrival_payout=1.0,
        follower_arrival_share=1.0,
        step_cost_parts=2.0,
        body_contact_cost_parts=3.0,
        leg_contact_cost_parts=1.0,
        foot_slip_cost_parts=0.0,
        legs_off_tempo_cost_parts=0.0,
        out_of_tempo_cost_parts=0.0,
        movement_cost_parts=0.0,
        random_command_movement_deg=25.0,
        foot_slip_unit_m_per_s=0.010,
        legs_off_tempo_unit_deg=20.0,
        head_tempo_share=0.25,
        command_cost_ratio=0.0,
        cost_budget_parts=6.0,
        cost_horizon_steps=None,
        head_progress_ratio=1.0,
        progress_parts=None,
        follower_progress_share=1.0,
        follower_progress_ratio=None,
        efficiency_cost=None,
        body_contact_cost=None,
        leg_contact_cost=None,
        distance_ratio_epsilon_m=1e-6,
    )


def test_a_file_with_per_step_weights_keeps_the_reward_used_before_2026_10_08():
    rewards = read(rewards={"efficiency_cost": 0.005}).rewards

    assert rewards.uses_per_step_weights
    assert (rewards.efficiency_cost, rewards.body_contact_cost) == (0.005, 0.010)
    assert rewards.step_cost_parts is None and rewards.head_progress_ratio is None


def test_nested_sections_are_read_by_their_own_classes():
    settings = read(
        observation_radius=0,
        simulation=SIMULATION | {"world_count": 4},
        target={"bearing_range_deg": [-30, 30]},
        rewards={"arrival_reward": 0, "leg_contact_cost_parts": 0},
    )

    assert settings.observation_radius == 0
    assert settings.simulation.world_count == 4
    assert settings.target.bearing_range_deg == (-30.0, 30.0)
    assert settings.rewards.arrival_reward == 0.0
    assert settings.rewards.leg_contact_cost_parts == 0.0
    with pytest.raises(SettingsError, match="model_path is required"):
        EnvironmentSettings.from_section({})


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"observation_radius": -1}, r"\[environment\] observation_radius"),
        ({"target": {"distance_range_m": [-0.01, 0.02]}}, "distance_range_m"),
        ({"rewards": {"arrival_reward": -1}}, "arrival_reward"),
        ({"rewards": {"leg_contact_cost_parts": -1}}, "leg_contact_cost_parts"),
        (
            {"rewards": {"efficiency_cost": 0.003, "step_cost_parts": 2}},
            "mixes the per-step weights",
        ),
        ({"rewards": {"cost_budget_parts": 0}}, "must be above zero"),
        (
            {"rewards": {"progress_parts": 3, "head_progress_ratio": 1}},
            "gives the progress twice",
        ),
        ({"passive_follower_spine": True}, "needs spine_control"),
        ({"rewards": {"out_of_tempo_cost_parts": 1}}, "need the segments' clocks"),
        ({"clock": {"middle_tempo_hz": 8}}, "below 12.5 turns per second"),
        ({"target": {"range_circle_ratio": 0.8}}, "must be 0 .no circle. or above 1"),
        (
            {"target": {"arrival": "head", "arrival_radius_m": 0.001}},
            'arrival_radius_m belongs to arrival = "tip"',
        ),
        ({"target": {"arrival_radius": 0.001}}, r"\[environment.target\] has unknown"),
        ({"reward": {}}, r"\[environment\] has unknown settings: reward"),
    ],
)
def test_invalid_values_and_unknown_keys_name_their_section(values, problem):
    with pytest.raises(SettingsError, match=problem):
        read(**values)
