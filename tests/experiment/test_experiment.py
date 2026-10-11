"""Tests for the experiment's front file, with tiny runs on the CPU.

The main test trains and evaluates a run shaped like the oscillators'
baseline (model v4, a clock per leg, load feedback from the foot's force,
the clean reward, the curriculum), so that the path every training takes
is run before any launch. The others cover the plateau stop.
"""

import json
import math
from types import SimpleNamespace

from centipede.experiment.experiment import _plateau_readings, recorded_blocks, run
from centipede.experiment.plateau import Plateau

# Run 26's settings, made tiny: 2 worlds, 4-step episodes, 4 cycles.
BASELINE_FILE = """
[run]
mode = "train"
name = "baseline"
seed = 1
checkpoint_every_cycles = 2
runs_folder = "RUNS"
recordings = 1

[environment]
max_episode_steps = 4
observation_radius = 2
spine_control = true
passive_follower_spine = true
leg_clocks = true
leg_accelerations = true

[environment.simulation]
model_path = "models/assembly_v4.xml"
backend = "cpu"
world_count = 2

[environment.target]
distance_range_m = [0.030, 0.060]
bearing_range_deg = [-90, 90]
arrival = "head"
range_circle_ratio = 2.5
range_circle_margin_m = 0.035
after_arrival = "new_target"

[environment.clock]
load_feedback_rad_per_s = 6.0
load_feedback_source = "force"

[environment.rewards]
arrival_payout = 0.5
follower_arrival_share = 0
follower_progress_share = 0
progress_parts = 3
cost_budget_parts = 7
step_cost_parts = 0
body_contact_cost_parts = 2.5
leg_contact_cost_parts = 2
clock_tracking_cost_parts = 5

[agents]
hidden_layers = [8]
actor_last_layer_scale = 0.01
exploration_noise_beta = 0.5

[agents.ppo]
minibatch_size = 4

[interaction_loop]
rollout_window_steps = 2
update_cycles = 4

[curriculum]
final_distance_range_m = [0.030, 0.200]
final_bearing_range_deg = [-180, 180]
"""

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


def logged_records(folder):
    lines = (folder / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_recordings_spread_from_the_first_episode_length_to_the_last():
    assert recorded_blocks(64, 5) == {0, 16, 32, 47, 63}
    assert recorded_blocks(64, 1) == {63}
    assert recorded_blocks(3, 5) == recorded_blocks(3, "all") == {0, 1, 2}
    assert recorded_blocks(64, 0) == set()


def test_a_run_shaped_like_the_baseline_trains_and_is_evaluated(tmp_path):
    folder = run_file(tmp_path, BASELINE_FILE)

    records = logged_records(folder)
    assert [record["cycle"] for record in records] == [1, 2, 3, 4]
    for record in records:
        for segment_parts in record["step_facts"]["reward_parts"]:
            # Arrival, progress, body down, legs touching, following.
            assert len(segment_parts) == 5
            assert all(math.isfinite(part) for part in segment_parts)
    checkpoints = sorted(path.name for path in (folder / "checkpoints").iterdir())
    assert checkpoints == ["cycle_0002.pt", "cycle_0004.pt"]
    # One recording, of the last episode length: two windows of 2 steps.
    recordings = sorted(path.name for path in (folder / "recordings").iterdir())
    assert recordings == ["cycles_0003-0004.npz"]

    evaluation = (
        f'[run]\nmode = "evaluate"\nsource = "{folder.as_posix()}"\n'
        "[evaluation]\nseeds = [7]\nepisodes_per_seed = 2\nwalk_steps = 5\n"
        "record = false\n"
        "[environment.target]\ndistance_range_m = [0.030, 0.200]\n"
        "bearing_range_deg = [-180, 180]\n"
    )
    run_file(tmp_path, evaluation)

    (results_path,) = (folder / "evaluations").glob("*.json")
    assert results_path.with_suffix(".html").exists()
    assert not list((folder / "evaluations").glob("*.npz"))
    results = json.loads(results_path.read_text())
    assert list(results["results"]) == ["agents"]
    assert [record["seed"] for record in results["walk"]] == [7]


def test_a_run_at_a_plateau_records_one_more_episode_length_and_stops(tmp_path):
    # Three steps never reach a target 30 mm away: the arrival share stays 0.
    # An episode length is two windows, so it is first measured at cycle 2,
    # and its first average over three episode lengths (six cycles) comes at
    # cycle 7. The return is read only in the windows in which episodes end,
    # two of every three, so its first average comes at cycle 9, and it never
    # rises after; cycle 11 is then a plateau of two cycles for both. Cycles
    # 12 and 13 are recorded, the run's only recording: its one scheduled
    # recording, the last episode length (cycles 19 and 20), never comes.
    plateau_file = TRAINING_FILE.replace(
        'runs_folder = "RUNS"',
        'runs_folder = "RUNS"\nrecordings = 1\nplateau_cycles = 2',
    ).replace("update_cycles = 3", "update_cycles = 20")
    folder = run_file(tmp_path, plateau_file, NAME="plateau")

    assert [record["cycle"] for record in logged_records(folder)] == list(range(1, 14))
    assert (folder / "checkpoints" / "cycle_0013.pt").exists()
    recordings = sorted(path.name for path in (folder / "recordings").iterdir())
    assert recordings == ["cycles_0012-0013.npz"]
    # The report says why the run stopped early.
    report = (folder / "report.html").read_text(encoding="utf-8")
    assert "Stopped after cycle 13 of 20: a plateau." in report


def test_a_run_still_climbing_the_curriculum_does_not_plateau():
    """As in the oscillators' run 16, harder targets lower the arrival share
    as fast as the level rises, so the task, their sum, stays flat; the level
    alone shows the run still climbing."""
    no_episode_ended = {"episodes": {"episode_ended": False}}

    def plateau(watched):
        result = Plateau(cycles=48, min_progress=0.02, episode_windows=8)
        for cycle in range(1, 201):
            curriculum = SimpleNamespace(level=0.002 * cycle)
            share = SimpleNamespace(value=0.9 - 0.002 * cycle, targets_per_minute=0)
            readings = _plateau_readings(no_episode_ended, share, curriculum)
            result.update(cycle, {name: readings[name] for name in watched})
        return result

    assert plateau(("task",)).reached(200)
    assert not plateau(("task", "level")).reached(200)
