"""Tests for the experiment's front file, with tiny runs of the headless tester.

How each file is read and written is tested with the configuration and the
run folder; these tests cover what the front adds: the order of a run's
files, the tow and the mean-action worlds, the ramp and the curves' level
that wait for them, continuing a run, starting from
another run's agents with its own learning rate and wider layers, and
evaluating without changing anything.
"""

import json

import pytest
import torch
from mujoco_replay.recording import read_recording

from centipede.environment.settings import CurveSettings
from centipede.experiment.experiment import CurveStart, run
from centipede.settings_section import SettingsError

# Four worlds, the last two acting with the mean action, one at the tow's top
# and one untowed; the tow's top moves after every window of two steps.
TRAINING_FILE = """
[run]
mode = "train"
name = "NAME"
seed = 3
checkpoint_every_cycles = 2
runs_folder = "RUNS"
mean_action_worlds_per_group = 1

[environment]
max_episode_steps = 3

[environment.simulation]
model_path = "models/assembly_v5_headless.xml"
backend = "cpu"
world_count = 4

[environment.tow]
start = 0.2
window_steps = 2

[agents]
hidden_layers = [8]
advantage_weights = [1.0, 0.8]
value_normalization = true

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


def training_log(folder):
    lines = (folder / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def saved_update_count(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    return checkpoint["agents"]["segment_agents"][0]["update_count"]


def test_an_adaptive_ramp_follows_the_untowed_walk_and_is_saved(tmp_path):
    """An untrained body walks slower than a band of 1000 to 2000 mm/s: the
    weight stays at its start; a continued run takes up a saved weight and
    lowers it after each window (one a cycle here), 1.1 squared, a band width
    below the band."""
    adaptive = TRAINING_FILE.replace(
        "max_episode_steps = 3\n",
        'max_episode_steps = 3\n[environment.reward]\nramp_shape = "adaptive"\n'
        "ramp_start = 0.01\nramp_band_mm_per_s = [1000, 2000]\n",
    ).replace("update_cycles = 3", "update_cycles = 2")
    folder = run_file(tmp_path, adaptive, NAME="adaptive")
    assert [record["ramp_weight"] for record in training_log(folder)] == [0.01] * 2
    checkpoint_path = folder / "checkpoints" / "cycle_0002.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["ramp_weight"] == 0.01
    checkpoint["ramp_weight"] = 0.5
    torch.save(checkpoint, checkpoint_path)
    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "[interaction_loop]\nupdate_cycles = 4\n"
    )
    run_file(tmp_path, continuing)
    weights = [record["ramp_weight"] for record in training_log(folder)[2:]]
    assert weights == pytest.approx([0.5, 0.5 / 1.1**2], rel=0.01)


def test_the_curves_begin_once_the_walk_has_formed_and_a_pair_starts_ordering():
    """Over the last 8 cycles the untowed worlds walk 60 mm/s against a
    start of 50, and the second pair's consistency, both sides' mean, is
    0.26 against 0.25, the others at chance or unmeasured: the curves may
    begin at the eighth cycle, not before, nor while the walk is slower,
    unmeasured (before the tow's first window) or no pair orders."""
    curves = CurveSettings.from_section(
        {
            "level_tightest_radius_body_lengths": [3.0],
            "level_straight_shares": [0.5],
            "start_speed_mm_per_s": 50.0,
            "start_pair_consistency": 0.25,
        }
    )

    def record(speed, pair):
        sides = [[0.2, 0.2], [pair - 0.01, pair + 0.01], [None, None]]
        return {
            "tow": {"speed": [0.0, speed]},
            "step_facts": {"neighbour_lag_consistency": sides},
        }

    start = CurveStart(curves)
    assert not any(start.ready(record(60.0, 0.26)) for _ in range(7))
    assert start.ready(record(60.0, 0.26))
    slow, unordered = CurveStart(curves), CurveStart(curves)
    assert not any(slow.ready(record(45.0, 0.3)) for _ in range(8))
    assert not any(unordered.ready(record(60.0, 0.22)) for _ in range(8))
    unmeasured = CurveStart(curves)
    unmeasured.ready(record(None, 0.3))
    assert not any(unmeasured.ready(record(60.0, 0.3)) for _ in range(7))


def test_a_run_is_trained_continued_and_evaluated_without_changing_it(tmp_path):
    # The ramp waits for a line no untrained body walks. The curves' level
    # is judged every two cycles: in 3-step episodes no line is lost, so it
    # rises at cycle 2 to the last. The spreads never fall below their floor.
    recorded = (
        TRAINING_FILE.replace(
            "hidden_layers = [8]\n", "hidden_layers = [8]\naction_std_floor = 0.5\n"
        )
        .replace('runs_folder = "RUNS"', 'runs_folder = "RUNS"\nrecord_worlds = 1')
        .replace(
            "max_episode_steps = 3\n",
            'max_episode_steps = 3\nline_shape = "curved"\n'
            "[environment.reward]\nramp_start = 0.3\n"
            "ramp_cycles = 2\nramp_after_line_share = 1.0\n"
            "[environment.curves]\nlevel_tightest_radius_body_lengths = [3.0, 1.0]\n"
            "level_straight_shares = [0.5, 0.2]\nlevel_judged_cycles = 2\n",
        )
    )
    folder = run_file(tmp_path, recorded, NAME="probe")

    checkpoints = folder / "checkpoints"
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "cycle_0002.pt",
        "cycle_0003.pt",  # the last cycle is always saved
    ]
    log = training_log(folder)
    assert [record["cycle"] for record in log] == [1, 2, 3]
    assert len(log[-1]["tow"]["line"]) == 2  # at the tow and without
    assert len(log[-1]["learning"]["critic_explained_variances"][0]) == 2
    checkpoint = torch.load(checkpoints / "cycle_0003.pt", weights_only=True)
    assert checkpoint["tow_top"] == pytest.approx(log[-1]["tow"]["tow"])
    assert [record["ramp_weight"] for record in log] == [0.3] * 3
    assert checkpoint["ramp_from"] is None
    assert [record["curve_level"] for record in log] == [0, 1, 1]
    assert [record["kept_line_share"] for record in log[:2]] == [None, 1.0]
    assert checkpoint["curve_level"] == 1
    spreads = [
        spread
        for record in log
        for agent in record["learning"]["action_std"]
        for spread in agent
    ]
    assert min(spreads) > 0.49

    # In so short a run every recording would start at the first cycle: the
    # last, at 100%, records the first world's next episode, from its start,
    # with its line.
    (recording_path,) = (folder / "recordings").iterdir()
    assert recording_path.name.startswith("100pct_cycles_0001-")
    recording = read_recording(recording_path)
    assert recording.qpos.shape[:2] == (3, 1)
    assert recording.episode_start[0, 0]
    assert recording.score_name == "line paid"
    assert recording.path_names == ("line",)
    assert recording.path_points.shape == (3, 1, 1, 21, 3)
    assert "<include" not in recording.model_xml

    # The continued run tows from the saved top, moved by a window a cycle,
    # and ramps from the saved cycle.
    checkpoint["tow_top"] = 0.1
    checkpoint["ramp_from"] = 4
    torch.save(checkpoint, checkpoints / "cycle_0003.pt")
    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "[interaction_loop]\nupdate_cycles = 5\n"
    )
    assert run_file(tmp_path, continuing) == folder
    log = training_log(folder)
    assert [record["cycle"] for record in log] == [1, 2, 3, 4, 5]
    assert [record["ramp_weight"] for record in log[3:]] == pytest.approx([0.3, 0.65])
    assert [record["curve_level"] for record in log[3:]] == [1, 1]
    assert saved_update_count(checkpoints / "cycle_0005.pt") == 5
    checkpoint = torch.load(checkpoints / "cycle_0005.pt", weights_only=True)
    assert checkpoint["tow_top"] == pytest.approx(0.1, abs=0.041)
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
    assert first_episodes.qpos.shape[1] == 2  # episodes_per_seed's worlds
    assert first_episodes.setup["actor"] == "agents"
    results = json.loads(results_path.read_text())["results"]
    assert list(results) == ["agents", "zero action"]
    for records in results.values():
        assert [record["seed"] for record in records] == [7, 8]
        # Each of the two worlds counts its whole first episode only.
        for record in records:
            assert record["episodes"]["episode_ended"] == 2
            assert record["episodes"]["duration"] == pytest.approx(0.06)


def test_a_new_run_can_start_from_another_runs_agents(tmp_path):
    parent = run_file(tmp_path, TRAINING_FILE, NAME="easy")
    wider_file = TRAINING_FILE.replace(
        'runs_folder = "RUNS"',
        f'runs_folder = "RUNS"\nstart_from = "{parent.as_posix()}"',
    ).replace(
        "hidden_layers = [8]",
        "hidden_layers = [16]\nlearning_rate = 1e-3\n"
        'learning_rate_schedule = "linear"\nfinal_learning_rate = 1e-4',
    )

    child = run_file(tmp_path, wider_file, NAME="wider")

    # The child's agents continued from the parent's three updates, widened.
    checkpoint = torch.load(child / "checkpoints" / "cycle_0003.pt", weights_only=True)
    assert checkpoint["agents"]["settings"]["hidden_layers"] == (16,)
    assert saved_update_count(child / "checkpoints" / "cycle_0003.pt") == 6
    assert not (child / "recordings").exists()  # nothing recorded by default
    # Its own learning rate replaced the parent's, along its own cycles.
    rates = [record["learning"]["learning_rate"][0] for record in training_log(child)]
    assert rates == pytest.approx([1e-3, 5.5e-4, 1e-4])

    other_optimizer = wider_file.replace(
        "hidden_layers = [16]", 'hidden_layers = [16]\noptimizer = "sgd"'
    )
    with pytest.raises(SettingsError, match="optimizer must match"):
        run_file(tmp_path, other_optimizer, NAME="other")
    deeper = wider_file.replace("hidden_layers = [16]", "hidden_layers = [16, 16]")
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

    assert [record["cycle"] for record in training_log(folder)] == [1]
    assert [path.name for path in (folder / "checkpoints").iterdir()] == [
        "cycle_0001.pt"
    ]

    continuing = (
        f'[run]\nmode = "train"\ncontinue_from = "{folder.as_posix()}"\n'
        "time_limit_hours = 1\n"
    )
    run_file(tmp_path, continuing)

    assert [record["cycle"] for record in training_log(folder)] == [1, 2, 3]
