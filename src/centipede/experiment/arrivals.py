"""The training's arrival share, which the curriculum and the plateau stop follow.

It is the share of arrivals among the episodes that ended over the last
episode length of windows, as the training console shows it: over that span
every world ends at least one episode. A shorter span would mislead at the
start of a session, when all worlds start together and only arrivals can end
an episode before the time limit, so there is no share until the span is full.
"""

import math
from collections import deque


class ArrivalShare:
    """The arrival share of the last episode length of windows, or None."""

    def __init__(self, window_steps: int, episode_steps: int) -> None:
        self.value: float | None = None
        # How the episodes of the last episode length of windows ended.
        self._recent_endings: deque[list[float]] = deque(
            maxlen=math.ceil(episode_steps / window_steps)
        )

    def add(self, ending_counts: list[float]) -> None:
        """Count a window's episodes by ending, arrivals first, as the
        environment's ending histogram does."""
        self._recent_endings.append(ending_counts)
        if len(self._recent_endings) == self._recent_endings.maxlen:
            totals = [sum(column) for column in zip(*self._recent_endings, strict=True)]
            self.value = totals[0] / sum(totals)
