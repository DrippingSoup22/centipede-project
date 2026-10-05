"""Tests for the [environment] configuration section and its nested sections.

The generic value checks are tested with ``SettingsSection`` and the simulation
section with ``SimulationSettings``; these tests cover only the environment's
keys, defaults, and how the nested sections are handed on.
"""

import pytest

from centipede.environment.settings import (
    EnvironmentSettings,
    RewardSettings,
    TargetSettings,
)
from centipede.settings_section import SettingsError

SIMULATION = {"model_path": "models/assembly_v2.xml", "backend": "cpu"}


def read(**values) -> EnvironmentSettings:
    return EnvironmentSettings.from_section({"simulation": SIMULATION} | values)


def test_defaults_fill_everything_but_the_simulation_requirements():
    settings = read()

    assert settings.max_episode_steps == 8192
    assert settings.observation_radius == 1
    assert settings.simulation.world_count == 1
    assert settings.target == TargetSettings(
        distance_range_m=(0.010, 0.020),
        bearing_range_deg=(-15.0, 15.0),
        arrival_radius_m=0.001,
    )
    assert settings.rewards == RewardSettings(
        arrival_reward=1.0,
        efficiency_cost=0.003,
        body_contact_cost=0.010,
        leg_contact_cost=0.005,
        distance_ratio_epsilon_m=1e-6,
    )


def test_nested_sections_are_read_by_their_own_classes():
    settings = read(
        observation_radius=0,
        simulation=SIMULATION | {"world_count": 4},
        target={"bearing_range_deg": [-30, 30]},
        rewards={"arrival_reward": 0, "leg_contact_cost": 0},
    )

    assert settings.observation_radius == 0
    assert settings.simulation.world_count == 4
    assert settings.target.bearing_range_deg == (-30.0, 30.0)
    assert settings.rewards.arrival_reward == 0.0
    assert settings.rewards.leg_contact_cost == 0.0
    with pytest.raises(SettingsError, match="model_path is required"):
        EnvironmentSettings.from_section({})


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"observation_radius": -1}, r"\[environment\] observation_radius"),
        ({"target": {"distance_range_m": [-0.01, 0.02]}}, "distance_range_m"),
        ({"rewards": {"arrival_reward": -1}}, "arrival_reward"),
        ({"rewards": {"leg_contact_cost": -0.1}}, "leg_contact_cost"),
        ({"target": {"arrival_radius": 0.001}}, r"\[environment.target\] has unknown"),
        ({"reward": {}}, r"\[environment\] has unknown settings: reward"),
    ],
)
def test_invalid_values_and_unknown_keys_name_their_section(values, problem):
    with pytest.raises(SettingsError, match=problem):
        read(**values)
