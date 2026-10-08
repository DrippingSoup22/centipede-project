from dataclasses import dataclass
from pathlib import Path

from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class SimulationSettings:
    """The [environment.simulation] section of the configuration file.

    ``start_heading_range_deg`` turns the whole body about the vertical at every
    reset, by a random angle within plus or minus this many degrees: 0 keeps the
    model's heading, 180 allows any heading.
    """

    model_path: Path
    backend: str
    world_count: int
    gpu_solver: str
    contacts_per_world: int
    constraints_per_world: int
    start_heading_range_deg: float

    @classmethod
    def from_section(cls, values: dict) -> "SimulationSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.simulation")
        settings = cls(
            model_path=section.path("model_path"),
            backend=section.choice("backend", ("cpu", "gpu")),
            world_count=section.positive_integer("world_count", default=1),
            gpu_solver=section.choice("gpu_solver", ("newton", "cg"), default="newton"),
            contacts_per_world=section.positive_integer(
                "contacts_per_world", default=128
            ),
            constraints_per_world=section.positive_integer(
                "constraints_per_world", default=512
            ),
            start_heading_range_deg=section.number(
                "start_heading_range_deg", default=0.0, minimum=0, maximum=180
            ),
        )
        section.reject_unknown_keys()
        return settings
