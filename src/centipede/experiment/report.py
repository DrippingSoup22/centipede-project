"""How the diagnostics categories reach a run's files: each as plain values.

Every training cycle's log line and every evaluation's results hold the
categories' values as numbers and lists. The follow-line branch writes no
HTML report: the training project's monitor shows the log, and its review
tools read the results.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from centipede.diagnostics_category import Description

# The simulated time of one environment step (docs/environment.md), which turns
# step counts into seconds and distances per step into speeds.
STEP_SECONDS = 0.02


@dataclass(frozen=True)
class LoggedCategory:
    """One diagnostics category as the log and the reports see it.

    ``name`` is its key in the log, ``title`` its heading, and ``note`` says
    how its values were summarised. ``read`` returns its current
    values by name, as tensors. A category that ``holds_histograms`` has the
    bin counts of values described with ``histogram_edges`` instead.
    """

    name: str
    title: str
    note: str
    descriptions: dict[str, Description]
    read: Callable[[], dict[str, torch.Tensor]]
    holds_histograms: bool = False

    def plain_values(self) -> dict[str, Any]:
        """The current values as numbers and lists; missing values are None."""
        return {name: _plain(value) for name, value in self.read().items()}


def _plain(value: torch.Tensor) -> Any:
    """A tensor as a number or nested lists; NaN and infinities become None.

    A window with no ended episode has no episode means; JSON has no NaN.
    """

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(element) for element in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return clean(value.detach().cpu().tolist())
