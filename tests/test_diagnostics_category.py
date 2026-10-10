"""Tests for describing the values of a diagnostics category."""

import math
from dataclasses import dataclass

import pytest
import torch

from centipede.diagnostics_category import (
    Description,
    WindowSummary,
    descriptions,
    measure,
    values,
)


@dataclass(frozen=True)
class ExampleFacts:
    height: torch.Tensor = measure("Height of the centre", "m")
    touching: torch.Tensor = measure("Body on the ground", summary="share")


def test_fields_carry_their_descriptions_in_declaration_order():
    facts = ExampleFacts(
        height=torch.zeros(2), touching=torch.zeros(2, dtype=torch.bool)
    )

    assert descriptions(facts) == {
        "height": Description("Height of the centre", "m", "mean"),
        "touching": Description("Body on the ground", "", "share"),
    }
    assert list(values(facts)) == ["height", "touching"]
    assert values(facts)["height"] is facts.height


def test_an_unknown_summary_is_rejected():
    with pytest.raises(ValueError, match="summary must be one of"):
        measure("Anything", summary="median")


@dataclass(frozen=True)
class ExamplePoses:
    pose: torch.Tensor = measure("Kept for recordings", summary="recorded")
    rows: torch.Tensor = measure("Rows used", summary="maximum")


def test_a_recorded_value_is_left_out_of_the_window_summary():
    poses = ExamplePoses(pose=torch.ones(2, 3), rows=torch.tensor([4, 7]))
    window = WindowSummary(poses)
    window.add()

    assert list(window.descriptions) == ["rows"]
    assert window.result() == {"rows": torch.tensor(7.0)}


@dataclass(frozen=True)
class ExampleEnds:
    ended: torch.Tensor = measure("Episode ended", summary="count")
    length: torch.Tensor = measure("Episode length", summary="mean")
    longest_leg: torch.Tensor = measure("Longest leg", summary="maximum")


def test_a_window_summary_follows_each_values_summary_and_mask():
    facts = ExampleFacts(
        height=torch.zeros(3), touching=torch.zeros(3, dtype=torch.bool)
    )
    ends = ExampleEnds(
        ended=torch.zeros(3, dtype=torch.bool),
        length=torch.zeros(3),
        longest_leg=torch.zeros(3, 2),
    )
    fact_window = WindowSummary(facts)
    end_window = WindowSummary(ends, mask_field="ended")

    # Two steps. Only worlds 0 and 1 count; world 2 is ignored even where its
    # episode ends with the largest values.
    for height, touching, ended, length, leg in (
        ([1.0, 2.0, 9.0], [True, False, True], [True, False, True], 10.0, 5.0),
        ([3.0, 4.0, 9.0], [True, True, True], [False, True, True], 20.0, 3.0),
    ):
        facts.height.copy_(torch.tensor(height))
        facts.touching.copy_(torch.tensor(touching))
        ends.ended.copy_(torch.tensor(ended))
        ends.length.copy_(torch.tensor([length, length, 99.0]))
        ends.longest_leg.copy_(torch.tensor([[leg, 0.0], [leg, 1.0], [99.0, 99.0]]))
        counted_worlds = torch.tensor([True, True, False])
        fact_window.add(counted_worlds)
        end_window.add(counted_worlds)

    facts_summary = fact_window.result()
    assert facts_summary["height"] == 2.5
    assert facts_summary["touching"] == 0.75
    ends_summary = end_window.result()
    assert ends_summary["ended"] == 2  # world 0 on step 0, world 1 on step 1
    assert ends_summary["length"] == 15.0
    assert ends_summary["longest_leg"].tolist() == [5.0, 1.0]

    end_window.clear()
    assert end_window.result()["length"].isnan()


@dataclass(frozen=True)
class ExampleComparisons:
    compared: torch.Tensor = measure("Compared", summary="share")
    closeness: torch.Tensor = measure(
        "Closeness where compared", counted_where="compared"
    )


def test_a_mean_counted_where_a_flag_is_true_leaves_out_the_other_entries():
    facts = ExampleComparisons(
        compared=torch.zeros(2, 3, dtype=torch.bool), closeness=torch.zeros(2, 3)
    )
    window = WindowSummary(facts)
    # Two steps; world 1 does not count on the second, and the last entry is
    # never compared. No 9 may count.
    for compared, closeness, counted_worlds in (
        ([[1, 0, 0], [1, 1, 0]], [[0.2, 9, 9], [0.4, 0.6, 9]], [True, True]),
        ([[1, 1, 0], [1, 1, 0]], [[0.6, 0.8, 9], [9, 9, 9]], [True, False]),
    ):
        facts.compared.copy_(torch.tensor(compared, dtype=torch.bool))
        facts.closeness.copy_(torch.tensor(closeness))
        window.add(torch.tensor(counted_worlds))

    closeness = window.result()["closeness"]
    assert torch.allclose(closeness[:2], torch.tensor([0.4, 0.7]))
    assert closeness[2].isnan()


@dataclass(frozen=True)
class ExampleSpread:
    ended: torch.Tensor = measure("Episode ended", summary="count")
    length: torch.Tensor = measure("Episode length", histogram_edges=(0, 10, 20))
    per_segment: torch.Tensor = measure("Per segment", histogram_edges=(0, 1, 2, 3))


def test_a_window_summary_counts_the_masked_rows_in_each_bin():
    spread = ExampleSpread(
        ended=torch.tensor([True, True, True, False]),
        length=torch.tensor([-5.0, 10.0, 25.0, 15.0]),
        per_segment=torch.tensor([[0.5, 2.5], [1.0, 9.0], [2.0, 0.0], [1.5, 1.5]]),
    )
    window = WindowSummary(spread, mask_field="ended")

    window.add()
    window.add()

    histograms = window.histograms()
    assert list(histograms) == ["length", "per_segment"]
    # A bin includes its lower edge; values outside the edges land in the end
    # bins; world 3 is not counted.
    assert histograms["length"].tolist() == [2.0, 4.0]
    assert histograms["per_segment"].tolist() == [[2.0, 2.0, 2.0], [2.0, 0.0, 4.0]]
    assert window.result()["length"] == 10.0  # the mean is kept as well

    window.clear()
    assert window.histograms()["length"].tolist() == [0.0, 0.0]


@dataclass(frozen=True)
class ExampleClocks:
    offset: torch.Tensor = measure("Offset between clocks", "rad", summary="angle")


def test_an_angle_is_summarised_by_its_direction_and_how_steady_it_is():
    clocks = ExampleClocks(offset=torch.zeros(3, 2))
    window = WindowSummary(clocks)

    # Entry 0: each counted world keeps its angle, but the two are opposite.
    # Entry 1: 30 degrees everywhere. World 2 is not counted.
    for degrees in ([[90, 30], [-90, 30], [0, 0]], [[90, 30], [-90, 30], [180, 180]]):
        clocks.offset.copy_(torch.deg2rad(torch.tensor(degrees, dtype=torch.float32)))
        window.add(torch.tensor([True, True, False]))

    summary = window.result()
    assert list(summary) == ["offset", "offset_consistency", "offset_lock"]
    assert list(window.descriptions) == list(summary)
    assert summary["offset"][1].item() == pytest.approx(math.radians(30))
    assert summary["offset_consistency"].tolist() == pytest.approx([0.0, 1.0], abs=1e-6)
    assert summary["offset_lock"].tolist() == pytest.approx([1.0, 1.0])

    window.clear()
    assert window.result()["offset_lock"].isnan().all()
