"""Tests for the experiment's front file, with tiny runs of the real model v2.

How each file is read and written is tested with the configuration and the
run folder; these tests cover what the front adds: the order of a run's
files, continuing a run, starting from another run's agents, and evaluating
without changing anything.
"""

import json

import pytest
import torch

from centipede.experiment.experiment import run
from centipede.settings_section import SettingsError

TRAINING_FILE = """
[run]
mode = "train"
name = "NAME"
seed = 3
checkpoint_every_cycles = 2
runs_folder = "RUNS"

[environment]
max_episode_steps = 3

[environment.simulation]
model_path = "models/assembly_v2.xml"
backend = "cpu"
world_count = 2

[agents]
hidden_layers = [8]

[agents.ppo]
minibatch_size = 4

[interaction_loop]
rollout_window_steps = 2
update_cycles = 3
"""


def run_file(tmp_path, text, **replacements):
    """Write a configuration file and run it; returns the run folder."""
    for old, new in (("RUNS", (tmp_path / "runs").as_posix()), *replacements.items()):
        text = text.replace(old, new)
    path = tmp_path / "file.toml"
    path.write_text(text, encoding="utf-8")
    return run(path)


def logged_cycles(folder):
    lines = (folder / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["cycle"] for line in lines]


def saved_update_count(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    return checkpoint["agents"]["segment_agents"][0]["update_count"]


def test_a_run_is_trained_continued_and_evaluated_without_changing_it(tmp_path):
    folder = run_file(tmp_path, TRAINING_FILE, NAME="probe")

    checkpoints = folder / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "cycle_0002.pt",
        "cycle_0003.pt",  # the last cycle is always saved
    ]
    assert logged_cycles(folder) == [1, 2, 3]
    assert "/*REPORT_DATA*/" not in (folder / "report.html").read_text(encoding="utf-8")

    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "[interaction_loop]\nupdate_cycles = 5\n"
    )
    assert run_file(tmp_path, continuing) == folder
    assert logged_cycles(folder) == [1, 2, 3, 4, 5]
    assert saved_update_count(checkpoints / "cycle_0005.pt") == 5
    sessions = json.loads((folder / "run_info.json").read_text())["sessions"]
    assert [session["first_cycle"] for session in sessions] == [0, 3]
    assert "interaction_loop.update_cycles" in sessions[1]["settings_written"]

    latest = checkpoints / "cycle_0005.pt"
    saved_bytes = latest.read_bytes()
    evaluation = (
        f'[run]\nmode = "evaluate"\nsource = "{folder.as_posix()}"\n'
        '[evaluation]\nseeds = [7, 8]\nepisodes_per_seed = 2\nbaselines = ["zero"]\n'
    )
    run_file(tmp_path, evaluation)

    assert latest.read_bytes() == saved_bytes
    (results_path,) = (folder / "evaluations").glob("*.json")
    assert results_path.with_suffix(".html").exists()
    results = json.loads(results_path.read_text())["results"]
    assert list(results) == ["agents", "zero action"]
    for records in results.values():
        assert [record["seed"] for record in records] == [7, 8]
        # Each of the two worlds counts its first episode only, in every histogram.
        for record in records:
            assert record["episodes"]["episode_ended"] == 2
            for counts in record["episode_distributions"].values():
                assert sum(counts) == 2


def test_a_new_run_can_start_from_another_runs_agents(tmp_path):
    parent = run_file(tmp_path, TRAINING_FILE, NAME="easy")
    harder_file = TRAINING_FILE.replace(
        "[interaction_loop]",
        "[environment.target]\ndistance_range_m = [0.03, 0.04]\n\n[interaction_loop]",
    ).replace(
        'runs_folder = "RUNS"',
        f'runs_folder = "RUNS"\nstart_from = "{parent.as_posix()}"',
    )

    child = run_file(tmp_path, harder_file, NAME="harder")

    # The child's agents continued from the parent's three updates.
    assert saved_update_count(child / "checkpoints" / "cycle_0003.pt") == 6
    assert logged_cycles(child) == [1, 2, 3]

    other_rate = harder_file.replace(
        "hidden_layers = [8]", "hidden_layers = [8]\nlearning_rate = 1e-3"
    )
    with pytest.raises(SettingsError, match="learning_rate must match"):
        run_file(tmp_path, other_rate, NAME="faster")
    assert len(list((tmp_path / "runs").iterdir())) == 2  # nothing left behind
