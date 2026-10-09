"""Tests for the experiment's front file, with tiny runs of the real model v3.

How each file is read and written is tested with the configuration and the
run folder; these tests cover what the front adds: the order of a run's
files, continuing a run, starting from another run's agents with its own
learning rate and, with the spine, more to see and command, and evaluating
without changing anything.
"""

import json

import numpy as np
import pytest
import torch
from mujoco_replay.recording import read_recording

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
model_path = "models/assembly_v3.xml"
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

    # Every episode length (3 steps, so two windows of 2) is one recording of
    # both worlds; the last is cut short where the run ends.
    recordings = sorted(path.name for path in (folder / "recordings").iterdir())
    assert recordings == ["cycles_0001-0002.npz", "cycles_0003-0003.npz"]
    whole = read_recording(folder / "recordings" / "cycles_0001-0002.npz")
    assert whole.qpos.shape == (4, 2, 69)
    assert whole.frame_info[:, 0].tolist() == [0, 0, 1, 1]  # updates done
    assert (whole.event_frames.tolist(), whole.event_labels) == (
        [2, 4],
        ("update 1", "update 2"),
    )
    recording = read_recording(folder / "recordings" / "cycles_0003-0003.npz")
    assert recording.qpos.shape == (2, 2, 69)
    assert recording.level.tolist() == [1, 3]  # 2 worlds over 4 levels
    assert (recording.rank.tolist(), recording.ranked_worlds) == ([1, 2], 2)
    assert recording.level_count == 4
    # The target, and the range circle as a ring of 2.5 x the start distance.
    assert recording.marker_names == ("target", "range")
    assert recording.marker_shapes == ("sphere", "ring")
    assert recording.marker_positions.shape == (2, 2, 2, 3)
    ring = recording.marker_radius[..., 1]
    assert recording.marker_radius.shape == (2, 2, 2)
    assert np.all((ring >= 2.5 * 0.030 - 1e-6) & (ring <= 2.5 * 0.060 + 1e-6))
    assert recording.frame_info[:, 0].tolist() == [2, 2]  # updates done
    assert recording.frame_info[:, 1].tolist() == [5, 6]  # steps per world
    assert (recording.event_frames.tolist(), recording.event_labels) == (
        [2],
        ("update 3",),
    )
    assert recording.setup["configuration"]["run"]["record_every_episodes"] == 1
    assert "<include" not in recording.model_xml

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
    evaluation_recordings = sorted(
        path.name.removeprefix(results_path.stem)
        for path in (folder / "evaluations").glob("*.npz")
    )
    assert evaluation_recordings == [
        "_agents_seed7.npz",
        "_agents_seed8.npz",
        "_zero_action_seed7.npz",
        "_zero_action_seed8.npz",
    ]
    first_episodes = read_recording(
        folder / "evaluations" / (results_path.stem + "_agents_seed7.npz")
    )
    assert first_episodes.qpos.shape[1:] == (2, 69)
    assert first_episodes.frame_info is None and first_episodes.event_frames is None
    assert first_episodes.level is None and first_episodes.rank is None
    assert first_episodes.setup["actor"] == "agents"
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
    harder_file = (
        TRAINING_FILE.replace(
            "[interaction_loop]",
            "[environment.target]\ndistance_range_m = [0.03, 0.04]\n\n"
            "[environment.rewards]\nmovement_cost_parts = 1\n\n"
            "[interaction_loop]",
        )
        .replace("max_episode_steps = 3", "max_episode_steps = 3\nspine_control = true")
        .replace(
            'runs_folder = "RUNS"',
            f'runs_folder = "RUNS"\nstart_from = "{parent.as_posix()}"\n'
            "record_every_episodes = 0",
        )
        .replace(
            "hidden_layers = [8]",
            "hidden_layers = [16]\nlearning_rate = 1e-3\n"
            'learning_rate_schedule = "linear"\nfinal_learning_rate = 1e-4',
        )
    )

    child = run_file(tmp_path, harder_file, NAME="harder")

    # The child's agents continued from the parent's three updates, widened to
    # wider layers and to command the spine joint behind every segment but
    # the rear.
    checkpoint = torch.load(child / "checkpoints" / "cycle_0003.pt", weights_only=True)
    assert checkpoint["agents"]["settings"]["hidden_layers"] == (16,)
    assert checkpoint["agents"]["segment_action_sizes"] == [7] * 7 + [6]
    assert saved_update_count(child / "checkpoints" / "cycle_0003.pt") == 6
    assert logged_cycles(child) == [1, 2, 3]
    assert not (child / "recordings").exists()  # recording was turned off
    # Its own learning rate replaced the parent's, along its own cycles.
    lines = (child / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    rates = [json.loads(line)["learning"]["learning_rate"] for line in lines]
    assert rates == pytest.approx([1e-3, 5.5e-4, 1e-4])

    other_optimizer = harder_file.replace(
        "hidden_layers = [16]", 'hidden_layers = [16]\noptimizer = "sgd"'
    )
    with pytest.raises(SettingsError, match="optimizer must match"):
        run_file(tmp_path, other_optimizer, NAME="other")
    deeper = harder_file.replace("hidden_layers = [16]", "hidden_layers = [16, 16]")
    with pytest.raises(SettingsError, match="as many layers"):
        run_file(tmp_path, deeper, NAME="deeper")
    assert len(list((tmp_path / "runs").iterdir())) == 2  # nothing left behind


def test_a_time_limit_stops_the_session_with_a_checkpoint_to_continue_from(tmp_path):
    """A limit too short for even one cycle stops after the first, with a
    checkpoint although checkpoints are every two cycles; a continuing file
    with a new limit finishes the run."""
    limited = TRAINING_FILE.replace(
        'runs_folder = "RUNS"', 'runs_folder = "RUNS"\ntime_limit_hours = 1e-9'
    )

    folder = run_file(tmp_path, limited, NAME="session")

    assert logged_cycles(folder) == [1]
    assert [path.name for path in (folder / "checkpoints").iterdir()] == [
        "cycle_0001.pt"
    ]

    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "time_limit_hours = 1\n"
    )
    run_file(tmp_path, continuing)

    assert logged_cycles(folder) == [1, 2, 3]
