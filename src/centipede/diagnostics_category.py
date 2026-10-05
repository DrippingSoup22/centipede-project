"""Describing the values of a diagnostics category.

A component's diagnostics category is a frozen dataclass of tensors, allocated
once and refreshed in place, like the physical state. Every field is declared
with ``measure``, which records what the value means, its unit, and how it is
summarised over worlds and steps. Logs and reports are built from these
descriptions, so every value appears in them. See docs/diagnostics.md.

Example, for a made-up category::

    @dataclass(frozen=True)
    class ExampleFacts:
        height: torch.Tensor = measure("Height of the centre", "m")
        touching: torch.Tensor = measure("Body on the ground", summary="share")
"""

from dataclasses import dataclass, field, fields
from typing import Any

import torch

# How a value is summarised over worlds and steps.
SUMMARIES = (
    "mean",  # the average
    "share",  # the fraction of true flags
    "maximum",  # the largest value
)


@dataclass(frozen=True)
class Description:
    """What one diagnostics value means, its unit, and how it is summarised."""

    meaning: str
    unit: str
    summary: str


def measure(meaning: str, unit: str = "", summary: str = "mean") -> Any:
    """Declare one field of a diagnostics category, with its description."""
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {SUMMARIES}, not {summary!r}")
    return field(metadata={"description": Description(meaning, unit, summary)})


def descriptions(category: Any) -> dict[str, Description]:
    """Every field's description, by field name, in declaration order."""
    return {item.name: item.metadata["description"] for item in fields(category)}


def values(category: Any) -> dict[str, torch.Tensor]:
    """Every field's tensor, by field name, in declaration order."""
    return {item.name: getattr(category, item.name) for item in fields(category)}
