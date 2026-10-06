"""Tests for a run's folder: creating it, its log, and its checkpoints."""

import pytest
import torch

from centipede.experiment.run_folder import RunFolder
from centipede.settings_section import SettingsError


def test_folders_are_created_once_and_runs_never_overwritten(tmp_path):
    runs_folder = tmp_path / "missing" / "runs"

    first = RunFolder.create(runs_folder, "probe")
    second = RunFolder.create(runs_folder, "probe")

    assert first.path.is_dir() and second.path.is_dir()
    assert second.path.name == first.path.name + "_2"
    (first.path / "configuration.toml").touch()
    assert (
        RunFolder.open(first.path / "checkpoints" / "cycle_0001.pt").path == first.path
    )


def test_the_log_and_checkpoints_can_be_resumed(tmp_path):
    folder = RunFolder.create(tmp_path, "run")
    for cycle in (1, 2, 3):
        folder.append_log({"cycle": cycle, "value": [cycle, None]})
    folder.save_checkpoint(9999, {"weights": torch.ones(2)})
    folder.save_checkpoint(10000, {"weights": torch.zeros(2)})

    folder.cut_log_after(2)

    assert folder.read_log() == [
        {"cycle": 1, "value": [1, None]},
        {"cycle": 2, "value": [2, None]},
    ]
    latest = folder.latest_checkpoint()
    assert latest.name == "cycle_10000.pt"  # by number, not by text order
    checkpoint = RunFolder.load_checkpoint(latest, "cpu")
    assert checkpoint["completed_cycles"] == 10000
    assert torch.equal(checkpoint["agents"]["weights"], torch.zeros(2))

    torch.save({"something": "else"}, tmp_path / "foreign.pt")
    with pytest.raises(SettingsError, match="not a checkpoint"):
        RunFolder.load_checkpoint(tmp_path / "foreign.pt", "cpu")
