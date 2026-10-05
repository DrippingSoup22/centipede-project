"""Tests for reading and checking one configuration section."""

from pathlib import Path

import pytest

from centipede.settings_section import SettingsError, SettingsSection


def read(values: dict) -> SettingsSection:
    return SettingsSection(values, "example")


def test_values_are_read_converted_and_defaulted():
    section = read(
        {
            "name": "probe",
            "backend": "gpu",
            "model_path": "models/assembly_v2.xml",
            "worlds": 4,
            "reward": 1,
            "rate": 3e-4,
            "range": [0.01, 0.02],
            "layers": [64, 64],
            "baselines": ["zero"],
            "target": {"arrival_radius_m": 0.001},
        }
    )

    assert section.text("name") == "probe"
    assert section.choice("backend", ("cpu", "gpu")) == "gpu"
    assert section.path("model_path") == Path("models/assembly_v2.xml")
    assert section.positive_integer("worlds") == 4
    assert section.number("reward") == 1.0
    assert isinstance(section.number("reward"), float)
    assert section.positive_number("rate") == 3e-4
    assert section.number_range("range") == (0.01, 0.02)
    assert section.integer_list("layers", minimum=1) == (64, 64)
    assert section.choice_list("baselines", ("zero", "random")) == ("zero",)
    assert section.table("target") == {"arrival_radius_m": 0.001}
    assert section.table("rewards") == {}
    assert section.positive_integer("missing", default=7) == 7
    section.reject_unknown_keys()


def test_missing_required_and_unknown_keys_are_reported():
    with pytest.raises(SettingsError, match=r"\[example\] backend is required"):
        read({}).choice("backend", ("cpu", "gpu"))

    section = read({"worlds": 2, "world_cuont": 3})
    section.positive_integer("worlds", default=1)
    with pytest.raises(SettingsError, match="unknown settings: world_cuont"):
        section.reject_unknown_keys()


@pytest.mark.parametrize(
    ("method", "arguments", "value", "problem"),
    [
        ("text", (), "", "must be non-empty text"),
        ("choice", (("cpu", "gpu"),), "gpuu", "must be one of"),
        ("path", (), 5, "must be a path"),
        ("integer", (), True, "must be a whole number"),
        ("positive_integer", (), 0, "must be at least 1"),
        ("number", (), "1", "must be a number"),
        ("positive_number", (), 0, "must be greater than zero"),
        ("number_range", (), [0.02, 0.01], "must have low <= high"),
        ("integer_list", (), [64, 6.4], "must be a whole number"),
        ("choice_list", (("zero",),), ["zero", "zero"], "must not repeat"),
        ("table", (), 5, "must be a table"),
    ],
)
def test_invalid_values_name_the_section_and_key(method, arguments, value, problem):
    with pytest.raises(SettingsError, match=r"\[example\] setting ") as error:
        getattr(read({"setting": value}), method)("setting", *arguments)

    assert problem in str(error.value)
