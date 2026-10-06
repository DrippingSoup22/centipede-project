"""The [interaction_loop] configuration section. See docs/architecture.md."""

from dataclasses import dataclass

from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class InteractionLoopSettings:
    """How long training lasts, counted in windows.

    Each of the ``update_cycles`` cycles collects ``rollout_window_steps`` steps
    in every world, then updates the agents once.
    """

    rollout_window_steps: int
    update_cycles: int

    @classmethod
    def from_section(cls, values: dict) -> "InteractionLoopSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "interaction_loop")
        settings = cls(
            rollout_window_steps=section.positive_integer(
                "rollout_window_steps", default=256
            ),
            update_cycles=section.positive_integer("update_cycles", default=128),
        )
        section.reject_unknown_keys()
        return settings
