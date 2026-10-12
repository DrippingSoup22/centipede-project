"""The curriculum: targets that get harder as the agents arrive more often.

The experiment keeps one difficulty level between 0 and 1. Level 0 draws the
targets from [environment.target]'s ranges, level 1 from [curriculum]'s final
ranges, and a level between from ranges in between, each end moved in
proportion to the level. After each update cycle, ``update`` moves the level
by ``level_rate`` times the gap between the arrival share (arrivals.py) and
the share the curriculum holds: it rises while the agents arrive more often
than that and falls while they arrive less often, so that it settles at the
difficulty where they arrive about that often. Automatic domain randomization
(Akkaya et al., 2019, "Solving Rubik's Cube with a Robot Hand") widens its
ranges on the same principle.

With a ``level_bonus``, the level also jumps by that much whenever the agents
show they cope: at each check the arrival share (already the share of the
last episode length) must be at least the share the curriculum holds and no
more than BONUS_TOLERANCE below the last check's. The first check comes four
episode lengths into the run, so that a lucky run of easy targets cannot earn
it; after a bonus the next comes two episode lengths later, one for the
episodes started at the old level to end and one to measure the new; after a
failed check, four again. Like the "bold driver" rule for learning rates
(Vogl et al., Biological Cybernetics 1988; Battiti, Complex Systems 1989): a
bigger step while things keep going well, back to the small one as soon as
they do not.
"""

from centipede.environment.settings import TargetSettings
from centipede.experiment.configuration import CurriculumSettings

# How far the arrival share may fall between two checks and still count as
# held: with about 1,000 episodes per episode length its chance noise is about
# 1.5 points, so the difference of two readings about 2.
BONUS_TOLERANCE = 0.02


class Curriculum:
    """The target's difficulty level during training, and its target ranges."""

    def __init__(
        self,
        settings: CurriculumSettings,
        target: TargetSettings,
        level: float,
        episode_cycles: int = 1,
    ) -> None:
        """Start at ``level``: 0 for a new run, the saved level when continuing.

        ``episode_cycles`` is how many update cycles one episode length takes,
        which times the bonus's checks; a continued run starts them afresh.
        """
        self.settings = settings
        self.level = level
        self._first_wait = 4 * episode_cycles
        self._wait_after_bonus = 2 * episode_cycles
        self._cycles_to_check = self._first_wait
        self._checked_share: float | None = None
        self._start_ranges = (target.distance_range_m, target.bearing_range_deg)
        self._final_ranges = (
            settings.final_distance_range_m,
            settings.final_bearing_range_deg,
        )

    def target_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The distance (m) and bearing (degrees) ranges of the current level."""
        distance, bearing = (
            (
                low + self.level * (final_low - low),
                high + self.level * (final_high - high),
            )
            for (low, high), (final_low, final_high) in zip(
                self._start_ranges, self._final_ranges, strict=True
            )
        )
        return distance, bearing

    def update(self, arrival_share: float) -> None:
        """Move the level after a cycle, from the current arrival share."""
        gap = arrival_share - self.settings.arrival_share
        level = self.level + self.settings.level_rate * gap
        if self.settings.level_bonus:
            if self._checked_share is None:
                self._checked_share = arrival_share
            self._cycles_to_check -= 1
            if self._cycles_to_check == 0:
                held = arrival_share >= self._checked_share - BONUS_TOLERANCE
                if held and gap >= 0:
                    level += self.settings.level_bonus
                    self._cycles_to_check = self._wait_after_bonus
                else:
                    self._cycles_to_check = self._first_wait
                self._checked_share = arrival_share
        self.level = min(1.0, max(0.0, level))
