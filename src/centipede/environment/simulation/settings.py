from dataclasses import dataclass
from pathlib import Path

from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class SimulationSettings:
    """The [environment.simulation] section of the configuration file."""

    model_path: Path
    backend: str
    world_count: int
    contacts_per_world: int
    constraints_per_world: int

    @classmethod
    def from_section(cls, values: dict) -> "SimulationSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.simulation")
        settings = cls(
            model_path=section.path("model_path"),
            backend=section.choice("backend", ("cpu", "gpu")),
            world_count=section.positive_integer("world_count", default=1),
            contacts_per_world=section.positive_integer(
                "contacts_per_world", default=128
            ),
            constraints_per_world=section.positive_integer(
                "constraints_per_world", default=512
            ),
        )
        section.reject_unknown_keys()
        return settings
