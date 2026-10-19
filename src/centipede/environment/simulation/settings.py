from dataclasses import dataclass
from pathlib import Path

from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class SimulationSettings:
    """The [environment.simulation] section of the configuration file.

    ``start_heading_range_deg`` turns the whole body about the vertical at every
    reset, by a random angle within plus or minus this many degrees: 0 keeps the
    model's heading, 180 allows any heading.

    ``max_restarted_worlds_per_action`` (2026-10-19, the user's decision) lets
    the GPU backend restart worlds that MuJoCo Warp made non-finite, up to this
    many in one action: each goes back to its state at the action's start, and
    its episode ends there. MuJoCo Warp guards nothing against non-finite
    values by design, and mjlab, the framework its guide names, restarts such
    worlds the same way. With 0, the default, any such world stops the run.

    ``repair_factorizations`` (2026-10-19) has the GPU backend repair the
    Newton solver's factorizations that come out non-finite (MuJoCo Warp
    issue #1415), as ``gpu_backend.REPAIR_DIAGONAL_SHARE`` explains. Off by
    default.
    """

    model_path: Path
    backend: str
    world_count: int
    gpu_solver: str
    contacts_per_world: int
    constraints_per_world: int
    start_heading_range_deg: float
    max_restarted_worlds_per_action: int
    repair_factorizations: bool

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
            max_restarted_worlds_per_action=section.integer(
                "max_restarted_worlds_per_action", default=0, minimum=0
            ),
            repair_factorizations=section.boolean(
                "repair_factorizations", default=False
            ),
        )
        section.reject_unknown_keys()
        return settings
