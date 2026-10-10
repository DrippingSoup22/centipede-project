"""Describing the values of a diagnostics category.

A component's diagnostics category is a frozen dataclass of tensors, allocated
once and refreshed in place, like the physical state. Every field is declared
with ``measure``, which records what the value means, its unit, and how it is
summarised over worlds and steps. Logs and reports are built from these
descriptions, so every value appears in them. ``WindowSummary`` summarises a
category over the steps of a window and, for values declared with
``histogram_edges``, also counts how the rows spread over those bins. An angle
is summarised by its circular mean and by how steady it is, across the worlds
and within each world. See docs/diagnostics.md.

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
    "recorded",  # kept as it is for recordings; never summarised
    "angle",  # the circular mean of an angle, in rad, with its consistency and lock
)


@dataclass(frozen=True)
class Description:
    """What one diagnostics value means, its unit, and how it is summarised.

    ``parts`` names the entries of a value's last dimension, such as the four
    contact flags, so that logs and reports can label them. ``histogram_edges``,
    in the value's unit, asks a window summary to also count the rows in each
    bin between consecutive edges; rows outside the edges count in the first or
    the last bin. ``counted_where`` names a true/false value of the same
    category and shape that picks the entries a mean counts: each entry is
    averaged only over the worlds and steps where its flag is true.
    """

    meaning: str
    unit: str
    summary: str
    parts: tuple[str, ...] = ()
    histogram_edges: tuple[float, ...] = ()
    counted_where: str = ""


def measure(
    meaning: str,
    unit: str = "",
    summary: str = "mean",
    parts: tuple[str, ...] = (),
    histogram_edges: tuple[float, ...] = (),
    counted_where: str = "",
) -> Any:
    """Declare one field of a diagnostics category, with its description."""
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {SUMMARIES}, not {summary!r}")
    edges = tuple(float(edge) for edge in histogram_edges)
    description = Description(
        meaning, unit, summary, tuple(parts), edges, counted_where
    )
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
    Values declared ``recorded`` are not summarised and do not appear in
    ``descriptions`` or ``result``. A mean declared with ``counted_where``
    counts each entry only where that flag is true, and is NaN for an entry
    whose flag never was.

    An ``angle``, in rad, is summarised by the direction of the mean of its
    unit vectors (cos, sin), from -pi to pi, and by two lengths of such means,
    from 0 to 1, which ``descriptions`` and ``result`` add after it under the
    angle's name with ``_consistency`` and ``_lock``: the consistency, over
    every counted world and step, is 1 when the angle is the same everywhere
    and near 0 when it points every way; the lock, over the counted steps of
    each world then averaged over the worlds, is 1 when the angle stays the
    same within each world, whatever it is in the others.
    """

    def __init__(self, category: Any, mask_field: str | None = None) -> None:
        """Allocate the running totals, on the category's device."""
        self.category = category
        self.mask_field = mask_field
        # Recorded values are left out of the summary entirely; an angle adds
        # its consistency and lock after it.
        self.descriptions = {}
        for name, description in descriptions(category).items():
            if description.summary == "recorded":
                continue
            self.descriptions[name] = description
            if description.summary == "angle":
                self.descriptions |= _steadiness_descriptions(name, description)
        first_value = next(iter(values(category).values()))
        self._row_count = torch.zeros((), device=first_value.device)
        self._totals = {
            name: torch.zeros(value.shape[1:], device=value.device)
            for name, value in values(category).items()
            if name in self.descriptions and self.descriptions[name].summary != "angle"
        }
        # Per mean counted where a flag is true, how often each entry counted.
        self._entry_counts = {
            name: torch.zeros(value.shape[1:], device=value.device)
            for name, value in values(category).items()
            if name in self.descriptions and self.descriptions[name].counted_where
        }
        # Per angle, each world's sums of its cosine and sine over the window,
        # with the angle's entries before them; and each world's counted rows.
        self._angle_sums = {
            name: torch.zeros((*value.shape, 2), device=value.device)
            for name, value in values(category).items()
            if name in self.descriptions and self.descriptions[name].summary == "angle"
        }
        self._world_rows = torch.zeros(first_value.shape[0], device=first_value.device)
        # Per histogram: the inner edges, the counts with the bins as the last
        # dimension, and each entry's first position in the flattened counts.
        self._histograms = {}
        for name, value in values(category).items():
            if name not in self.descriptions:
                continue
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
        for counts in self._entry_counts.values():
            counts.zero_()
        for sums in self._angle_sums.values():
            sums.zero_()
        self._world_rows.zero_()

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
            self._world_rows += 1
        else:
            self._row_count += mask.sum()
            self._world_rows += mask

        for name, value in category_values.items():
            if name not in self._totals:
                continue
            value = value.float()
            total = self._totals[name]
            if self.descriptions[name].summary == "maximum":
                if mask is not None:
                    value = torch.where(_rows(mask, value), value, -torch.inf)
                torch.maximum(total, value.amax(dim=0), out=total)
            elif name in self._entry_counts:
                counted = category_values[self.descriptions[name].counted_where]
                if mask is not None:
                    counted = counted & _rows(mask, counted)
                total += torch.where(counted, value, 0.0).sum(dim=0)
                self._entry_counts[name] += counted.sum(dim=0)
            else:
                if mask is not None:
                    value = torch.where(_rows(mask, value), value, 0.0)
                total += value.sum(dim=0)

        for name, sums in self._angle_sums.items():
            angle = category_values[name].float()
            unit_vectors = torch.stack((angle.cos(), angle.sin()), dim=-1)
            if mask is not None:
                unit_vectors = torch.where(_rows(mask, unit_vectors), unit_vectors, 0.0)
            sums += unit_vectors

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
        """Every field's summary so far, by field name, in declaration order;
        an angle's consistency and lock follow it."""
        summaries = {}
        for name, description in self.descriptions.items():
            if name in self._angle_sums:
                summaries |= self._angle_summary(name)
            elif name in self._totals:
                total = self._totals[name]
                if name in self._entry_counts:
                    summaries[name] = total / self._entry_counts[name]
                elif description.summary in ("mean", "share"):
                    summaries[name] = total / self._row_count
                else:
                    summaries[name] = total.clone()
        return summaries

    def _angle_summary(self, name: str) -> dict[str, torch.Tensor]:
        """An angle's circular mean, consistency, and lock; NaN without rows."""
        sums = self._angle_sums[name]
        world_rows = self._world_rows.reshape(-1, *([1] * (sums.dim() - 1)))
        mean_vector = sums.sum(dim=0) / world_rows.sum()
        counted_worlds = (self._world_rows > 0).reshape(world_rows.shape[:-1])
        world_lengths = (sums / world_rows.clamp(min=1)).norm(dim=-1)
        lock = (world_lengths * counted_worlds).sum(dim=0) / counted_worlds.sum()
        return {
            name: torch.atan2(mean_vector[..., 1], mean_vector[..., 0]),
            f"{name}_consistency": mean_vector.norm(dim=-1),
            f"{name}_lock": lock,
        }

    def histograms(self) -> dict[str, torch.Tensor]:
        """Bin counts so far of the values that declare histogram edges."""
        return {
            name: counts.clone() for name, (_, counts, _) in self._histograms.items()
        }


def _steadiness_descriptions(
    name: str, description: Description
) -> dict[str, Description]:
    """The descriptions of an angle's consistency and lock."""
    label = name.replace("_", " ")
    return {
        f"{name}_consistency": Description(
            f"How alike the {label} is across every world and step: the length of"
            " the mean of its unit vectors, from 0 (no common direction) to 1 (the"
            " same everywhere)",
            "",
            "mean",
            description.parts,
        ),
        f"{name}_lock": Description(
            f"How steady the {label} stays within each world: the length of the"
            " mean of its unit vectors over each world's steps, averaged over the"
            " worlds, from 0 to 1 (steady)",
            "",
            "mean",
            description.parts,
        ),
    }


def _rows(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A ``(W,)`` mask shaped to broadcast over a tensor whose rows are worlds."""
    return mask.reshape(mask.shape + (1,) * (like.dim() - 1))
