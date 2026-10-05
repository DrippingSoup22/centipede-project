"""Tests for describing the values of a diagnostics category."""

from dataclasses import dataclass

import pytest
import torch

from centipede.diagnostics_category import (
    Description,
    descriptions,
    measure,
    values,
)


@dataclass(frozen=True)
class ExampleFacts:
    height: torch.Tensor = measure("Height of the centre", "m")
    touching: torch.Tensor = measure("Body on the ground", summary="share")


def test_fields_carry_their_descriptions_in_declaration_order():
    facts = ExampleFacts(height=torch.zeros(2), touching=torch.zeros(2, dtype=bool))

    assert descriptions(facts) == {
        "height": Description("Height of the centre", "m", "mean"),
        "touching": Description("Body on the ground", "", "share"),
    }
    assert list(values(facts)) == ["height", "touching"]
    assert values(facts)["height"] is facts.height


def test_an_unknown_summary_is_rejected():
    with pytest.raises(ValueError, match="summary must be one of"):
        measure("Anything", summary="median")
