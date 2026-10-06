"""A run's summary pages: one for training, one per evaluation.

Each page is one self-contained HTML file that opens in any browser, offline.
This module only gathers the data as plain values; ``report_page.html`` decides
what is shown and how. Its script starts with the page's layout and a short
explanation of every value, so changing the report means editing that list.
Values the layout does not place are still shown, among the details, so a value
added to a diagnostics category always appears.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from centipede.diagnostics_category import Description

PAGE_TEMPLATE = Path(__file__).with_name("report_page.html")
DATA_PLACEHOLDER = "/*REPORT_DATA*/"

# The simulated time of one environment step (docs/environment.md), which turns
# step counts into seconds and distances per step into speeds.
STEP_SECONDS = 0.02


@dataclass(frozen=True)
class LoggedCategory:
    """One diagnostics category as the log and the reports see it.

    ``name`` is its key in the log, ``title`` its heading in a report, and
    ``note`` says how its values were summarised. ``read`` returns its current
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


def write_training_report(
    path: Path,
    title: str,
    run_facts: dict[str, Any],
    categories: Sequence[LoggedCategory],
    log: list[dict[str, Any]],
    settings: dict[str, Any],
    sessions: list[dict[str, Any]],
) -> None:
    """Write a training run's page from its log and its complete settings.

    ``run_facts`` are plain facts about the run, such as when it started, and
    ``sessions`` are the training sessions recorded in ``run_info.json``.
    """
    data = {
        "kind": "training",
        "title": title,
        "run_facts": run_facts,
        "sessions": sessions,
        "step_seconds": STEP_SECONDS,
        "categories": [_outline(category) for category in categories],
        "log": log,
        "settings": settings,
    }
    _write_page(path, data)


def write_evaluation_report(
    path: Path,
    title: str,
    run_facts: dict[str, Any],
    categories: Sequence[LoggedCategory],
    results: dict[str, list[dict[str, Any]]],
    settings: dict[str, Any],
) -> None:
    """Write an evaluation's page; ``results`` holds one record per seed and actor.

    ``run_facts`` describe the evaluation, such as the checkpoint and seeds.
    """
    data = {
        "kind": "evaluation",
        "title": title,
        "run_facts": run_facts,
        "step_seconds": STEP_SECONDS,
        "categories": [_outline(category) for category in categories],
        "results": results,
        "settings": settings,
    }
    _write_page(path, data)


def _write_page(path: Path, data: dict[str, Any]) -> None:
    """Put the data into the page template and save it."""
    # "</" inside the data would end the page's script element early.
    data_text = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    page = PAGE_TEMPLATE.read_text(encoding="utf-8")
    path.write_text(page.replace(DATA_PLACEHOLDER, data_text), encoding="utf-8")


def _outline(category: LoggedCategory) -> dict[str, Any]:
    """A category's name, title, note, and value descriptions, for the page."""
    return {
        "name": category.name,
        "title": category.title,
        "note": category.note,
        "holds_histograms": category.holds_histograms,
        "values": [
            {"name": name, **asdict(description)}
            for name, description in category.descriptions.items()
        ],
    }


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
