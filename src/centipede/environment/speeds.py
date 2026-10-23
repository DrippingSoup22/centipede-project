"""The commanded speed's schedule (stage 3, docs/follow-line/task.md, the
speeds): each world's command holds a speed drawn at its level, and draws
again when the hold ends. The environment reads ``speeds`` as the command
before the bends lower it, and judges the level by what the speeds' worlds
did."""

import torch

from centipede.environment.settings import SpeedSettings
from centipede.environment.simulation.constants import ACTION_DURATION_S

# A hold of 0 keeps its speed for the whole episode: no hold ends before the
# world restarts.
WHOLE_EPISODE = 2**31 - 1
# How long after a change the speed is still being reached, for the record:
# the error then and after is reported apart.
CHANGE_WINDOW_S = 2.0


class Speeds:
    """Each world's commanded speed, ``speeds`` ``(W,)`` mm/s; the level it
    was drawn at, ``world_levels`` ``(W,)``, -1 before the levels begin; and
    the steps since it last changed, ``steps_since_change`` ``(W,)``, a
    restart counting as a change.

    ``level`` is the level reached, -1 before the levels begin, when every
    world holds ``base_speed``. From the next restart after ``set_level``, a
    world draws as ``SpeedSettings`` says.
    """

    def __init__(
        self,
        settings: SpeedSettings,
        base_speed: float,
        world_count: int,
        device: torch.device | str,
    ) -> None:
        self.settings = settings
        self.base_speed = base_speed
        self.level = -1
        self.hold_steps = torch.tensor(
            [
                round(hold / ACTION_DURATION_S) or WHOLE_EPISODE
                for hold in settings.level_hold_s
            ],
            device=device,
        )
        self.low = torch.tensor(settings.level_low_mm_per_s, device=device)
        self.span = torch.tensor(settings.level_high_mm_per_s, device=device) - self.low
        self.stop_share = torch.tensor(settings.level_stop_share, device=device)
        self.speeds = torch.full((world_count,), base_speed, device=device)
        self.world_levels = torch.full(
            (world_count,), -1, dtype=torch.long, device=device
        )
        self.steps_left = torch.full_like(self.world_levels, WHOLE_EPISODE)
        self.steps_since_change = torch.zeros_like(self.world_levels)

    @property
    def level_count(self) -> int:
        return len(self.settings.level_hold_s)

    def set_level(self, level: int) -> None:
        """Draw the speeds from the next restarts on at ``level`` (or, with
        ``mixed_levels``, up to it); -1 holds the base speed."""
        self.level = level

    def restart(self, world_mask: torch.Tensor, generator: torch.Generator) -> None:
        """Draw the masked worlds' levels and their first speeds."""
        if self.level >= 0 and self.settings.mixed_levels:
            levels = torch.randint(
                0,
                self.level + 1,
                self.world_levels.shape,
                generator=generator,
                device=self.world_levels.device,
            )
        else:
            levels = torch.full_like(self.world_levels, self.level)
        self.world_levels = torch.where(world_mask, levels, self.world_levels)
        self._draw(world_mask, generator)

    def step(self, generator: torch.Generator) -> None:
        """Count one step, and draw a new speed where a hold has ended."""
        self.steps_since_change += 1
        self.steps_left -= 1
        self._draw(self.steps_left <= 0, generator)

    def _draw(self, world_mask: torch.Tensor, generator: torch.Generator) -> None:
        """New speeds and holds for the masked worlds, each at its level."""
        levels = self.world_levels.clamp(min=0)
        draws = torch.rand(
            (2, len(levels)), generator=generator, device=self.speeds.device
        )
        speeds = self.low[levels] + draws[0] * self.span[levels]
        speeds = torch.where(draws[1] < self.stop_share[levels], 0.0, speeds)
        holds = self.hold_steps[levels]
        waiting = self.world_levels < 0
        speeds = torch.where(waiting, self.base_speed, speeds)
        holds = torch.where(waiting, WHOLE_EPISODE, holds)
        self.speeds = torch.where(world_mask, speeds, self.speeds)
        self.steps_left = torch.where(world_mask, holds, self.steps_left)
        self.steps_since_change = torch.where(world_mask, 0, self.steps_since_change)
