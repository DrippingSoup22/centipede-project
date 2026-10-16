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
    recorded = TRAINING_FILE.replace(
        'runs_folder = "RUNS"', 'runs_folder = "RUNS"\nrecord_worlds = 1'
    )
    folder = run_file(tmp_path, recorded, NAME="probe")

    checkpoints = folder / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "cycle_0002.pt",
        "cycle_0003.pt",  # the last cycle is always saved
    ]
    assert logged_cycles(folder) == [1, 2, 3]
    assert "/*REPORT_DATA*/" not in (folder / "report.html").read_text(encoding="utf-8")

    # In so short a run every recording would start at the first cycle: the
    # last, at 100%, records the first world's next episode, from its start.
    (recording_path,) = (folder / "recordings").iterdir()
    assert recording_path.name == "100pct_cycles_0001-0003.npz"
    recording = read_recording(recording_path)
    assert recording.qpos.shape == (3, 1, 69)
    assert recording.episode_start[0, 0]
    assert recording.score_name == "summed reward"
    # The target, and the range circle as a ring of 2.5 x the start distance.
    assert recording.marker_shapes == ("sphere", "ring")
    ring = recording.marker_radius[..., 1]
    assert np.all((ring >= 2.5 * 0.030 - 1e-6) & (ring <= 2.5 * 0.060 + 1e-6))
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


def test_a_cooperators_run_keeps_its_level_and_tow_when_continued_and_evaluated(
    tmp_path,
):
    """The cooperators' design on model v4: two critics with baselines, the
    level and the tow logged after every cycle, saved, restored, and the
    evaluation at the level without tow. Of four worlds, the last two act
    with the mean action; the tow moves after every window of two steps."""
    cooperators = (
        TRAINING_FILE.replace("assembly_v3", "assembly_v4")
        .replace("world_count = 2", "world_count = 4")
        .replace(
            'runs_folder = "RUNS"',
            'runs_folder = "RUNS"\nmean_action_worlds_per_group = 1',
        )
        .replace(
            "max_episode_steps = 3",
            'max_episode_steps = 3\nobservation_radius = 2\ndesign = "cooperators"\n'
            "[environment.cooperators]\ntow_start = 0.4\ntow_window_steps = 2",
        )
        .replace(
            "hidden_layers = [8]",
            "hidden_layers = [8]\ncritic_baselines = true\nvalue_normalization = true",
        )
        .replace("update_cycles = 3", "update_cycles = 2")
    )
    folder = run_file(tmp_path, cooperators, NAME="cooperators")
    log = [
        json.loads(line)
        for line in (folder / "training_log.jsonl").read_text().splitlines()
    ]
    assert list(log[-1]["task"]) == ["level", "in_time_share"]
    assert len(log[-1]["tow"]["line_erased"]) == 2  # at the tow and without
    assert len(log[-1]["learning"]["baseline_explained_variances"][0]) == 2
    checkpoint = torch.load(folder / "checkpoints" / "cycle_0002.pt", weights_only=True)
    assert checkpoint["task_level"] == log[-1]["task"]["level"]
    assert checkpoint["tow_top"] == pytest.approx(log[-1]["tow"]["tow"])

    # The continued run tows from the saved top, moved by one window.
    checkpoint["tow_top"] = 0.1
    torch.save(checkpoint, folder / "checkpoints" / "cycle_0002.pt")
    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "[interaction_loop]\nupdate_cycles = 3\n"
    )
    run_file(tmp_path, continuing)
    assert logged_cycles(folder) == [1, 2, 3]
    checkpoint = torch.load(folder / "checkpoints" / "cycle_0003.pt", weights_only=True)
    assert checkpoint["tow_top"] == pytest.approx(0.1, abs=0.021)
    evaluation = (
        f'[run]\nmode = "evaluate"\nsource = "{folder.as_posix()}"\n'
        "[evaluation]\nseeds = [7]\nepisodes_per_seed = 2\nrecord = false\n"
        "[environment.cooperators]\nstagger_first_episodes = false\n"
    )
    run_file(tmp_path, evaluation)
    (results_path,) = (folder / "evaluations").glob("*.json")
    assert results_path.with_suffix(".html").exists()  # its own layout
    (record,) = json.loads(results_path.read_text())["results"]["agents"]
    assert record["episodes"]["episode_ended"] == 2  # one 3-step episode per world


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
            f'runs_folder = "RUNS"\nstart_from = "{parent.as_posix()}"',
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
    assert not (child / "recordings").exists()  # nothing recorded by default
    # Its own learning rate replaced the parent's, along its own cycles.
    lines = (child / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    rates = [json.loads(line)["learning"]["learning_rate"][0] for line in lines]
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
