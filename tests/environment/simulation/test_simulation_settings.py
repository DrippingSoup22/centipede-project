"""Tests for the [environment.simulation] configuration section.

The generic value checks are tested with ``SettingsSection``; these tests cover
only this section's keys, defaults, and options.
"""

from pathlib import Path

import pytest

from centipede.environment.simulation.settings import SimulationSettings
from centipede.settings_section import SettingsError

REQUIRED_ONLY = {"model_path": "models/assembly_v3.xml", "backend": "cpu"}


def test_defaults_fill_everything_but_the_required_keys():
    assert SimulationSettings.from_section(REQUIRED_ONLY) == SimulationSettings(
        model_path=Path("models/assembly_v3.xml"),
        backend="cpu",
        world_count=1,
        gpu_solver="newton",
        contacts_per_world=128,
        constraints_per_world=512,
        start_heading_range_deg=0.0,
    )


@pytest.mark.parametrize("missing_key", ["model_path", "backend"])
def test_required_keys_must_be_present(missing_key):
    values = {key: value for key, value in REQUIRED_ONLY.items() if key != missing_key}

    with pytest.raises(SettingsError, match=f"{missing_key} is required"):
        SimulationSettings.from_section(values)


def test_section_rules_are_enforced():
    with pytest.raises(SettingsError, match="backend must be one of"):
        SimulationSettings.from_section({**REQUIRED_ONLY, "backend": "tpu"})
    with pytest.raises(SettingsError, match="gpu_solver must be one of"):
        SimulationSettings.from_section({**REQUIRED_ONLY, "gpu_solver": "pgs"})
    with pytest.raises(SettingsError, match="start_heading_range_deg must be at most"):
        SimulationSettings.from_section(
            {**REQUIRED_ONLY, "start_heading_range_deg": 270}
        )
    with pytest.raises(SettingsError, match="world_count must be at least 1"):
        SimulationSettings.from_section({**REQUIRED_ONLY, "world_count": 0})
    with pytest.raises(SettingsError, match="unknown settings: world_cuont"):
        SimulationSettings.from_section({**REQUIRED_ONLY, "world_cuont": 4})
