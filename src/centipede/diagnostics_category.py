"""Describing the values of a diagnostics category.

A component's diagnostics category is a frozen dataclass of tensors, allocated
once and refreshed in place, like the physical state. Every field is declared
with ``measure``, which records what the value means, its unit, and how it is
summarised over worlds and steps. Logs and reports are built from these
descriptions, so every value appears in them. ``WindowSummary`` summarises a
category over the steps of a window and, for values declared with
``histogram_edges``, also counts how the rows spread over those bins. See
docs/diagnostics.md.

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
    "count",  # the number of true flags
    "maximum",  # the largest value
)


@dataclass(frozen=True)
class Description:
    """What one diagnostics value means, its unit, and how it is summarised.

    ``parts`` names the entries of a value's last dimension, such as the four
    contact flags, so that logs and reports can label them. ``histogram_edges``,
    in the value's unit, asks a window summary to also count the rows in each
    bin between consecutive edges; rows outside the edges count in the first or
    the last bin.
    """

    meaning: str
    unit: str
    summary: str
    parts: tuple[str, ...] = ()
    histogram_edges: tuple[float, ...] = ()


def measure(
    meaning: str,
    unit: str = "",
    summary: str = "mean",
    parts: tuple[str, ...] = (),
    histogram_edges: tuple[float, ...] = (),
) -> Any:
    """Declare one field of a diagnostics category, with its description."""
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {SUMMARIES}, not {summary!r}")
    edges = tuple(float(edge) for edge in histogram_edges)
    description = Description(meaning, unit, summary, tuple(parts), edges)
    return field(metadata={"description": description})


def descriptions(category: Any) -> dict[str, Description]:
    """Every field's description, by field name, in declaration order."""
    return {item.name: item.metadata["description"] for item in fields(category)}


def values(category: Any) -> dict[str, torch.Tensor]:
    """Every field's tensor, by field name, in declaration order."""
    return {item.name: getattr(category, item.name) for item in fields(category)}


class WindowSummary:
    """One category summarised over the worlds and steps of a window, on its device.

    The category's tensors have the worlds as their first dimension and are
    refreshed in place each step; ``add`` folds the current values into running
    totals, and ``result`` gives each field summarised as its description says,
    with the worlds removed. Nothing here waits for the GPU.

    ``mask_field`` names a ``(W,)`` flag of the category that selects the rows
    to summarise, such as the worlds whose episode just ended. With no counted
    rows in a window, means and shares are NaN and maxima are minus infinity.
    ``histograms`` gives the bin counts of the values that declare edges.
    """

    def __init__(self, category: Any, mask_field: str | None = None) -> None:
        """Allocate the running totals, on the category's device."""
        self.category = category
        self.mask_field = mask_field
        self.descriptions = descriptions(category)
        first_value = next(iter(values(category).values()))
        self._row_count = torch.zeros((), device=first_value.device)
        self._totals = {
            name: torch.zeros(value.shape[1:], device=value.device)
            for name, value in values(category).items()
        }
        # Per histogram: the inner edges, the counts with the bins as the last
        # dimension, and each entry's first position in the flattened counts.
        self._histograms = {}
        for name, value in values(category).items():
            edges = self.descriptions[name].histogram_edges
            if not edges:
                continue
            bin_count = len(edges) - 1
            entry_count = value[0].numel()
            self._histograms[name] = (
                torch.tensor(edges[1:-1], device=value.device),
                torch.zeros((*value.shape[1:], bin_count), device=value.device),
                torch.arange(entry_count, device=value.device).reshape(value.shape[1:])
                * bin_count,
            )
        self.clear()

    def clear(self) -> None:
        """Start a new window."""
        self._row_count.zero_()
        for name, total in self._totals.items():
            if self.descriptions[name].summary == "maximum":
                total.fill_(-torch.inf)
            else:
                total.zero_()
        for _, counts, _ in self._histograms.values():
            counts.zero_()

    def add(self, counted_worlds: torch.Tensor | None = None) -> None:
        """Fold in the category's current values of the counted ``(W,)`` worlds.

        Without ``counted_worlds`` every world counts, unless ``mask_field``
        leaves some out.
        """
        category_values = values(self.category)
        mask = counted_worlds
        if self.mask_field is not None:
            row_flags = category_values[self.mask_field]
            mask = row_flags if mask is None else mask & row_flags
        if mask is None:
            world_count = next(iter(category_values.values())).shape[0]
            self._row_count += world_count
        else:
            self._row_count += mask.sum()

        for name, value in category_values.items():
            value = value.float()
            total = self._totals[name]
            if self.descriptions[name].summary == "maximum":
                if mask is not None:
                    value = torch.where(_rows(mask, value), value, -torch.inf)
                torch.maximum(total, value.amax(dim=0), out=total)
            else:
                if mask is not None:
                    value = torch.where(_rows(mask, value), value, 0.0)
                total += value.sum(dim=0)

        for name, (inner_edges, counts, entry_offsets) in self._histograms.items():
            value = category_values[name].float()
            # Bins include their lower edge; values beyond the outer edges land
            # in the first or the last bin.
            bins = torch.bucketize(value, inner_edges, right=True)
            if mask is None:
                weights = torch.ones_like(value)
            else:
                weights = _rows(mask, value).float().expand_as(value)
            counts.view(-1).index_add_(
                0, (entry_offsets + bins).reshape(-1), weights.reshape(-1)
            )

    def result(self) -> dict[str, torch.Tensor]:
        """Every field's summary so far, by field name, in declaration order."""
        summaries = {}
        for name, total in self._totals.items():
            summary = self.descriptions[name].summary
            if summary in ("mean", "share"):
                summaries[name] = total / self._row_count
            else:
                summaries[name] = total.clone()
        return summaries

    def histograms(self) -> dict[str, torch.Tensor]:
        """Bin counts so far of the values that declare histogram edges."""
        return {
            name: counts.clone() for name, (_, counts, _) in self._histograms.items()
        }


def _rows(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A ``(W,)`` mask shaped to broadcast over a tensor whose rows are worlds."""
    return mask.reshape(mask.shape + (1,) * (like.dim() - 1))
