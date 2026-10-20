"""Tests for the terminal's view of a run: one line per window or pass."""

import math

from centipede.experiment.progress import (
    EvaluationProgress,
    TrainingProgress,
    training_settings,
)

# Step facts of two units, as the log holds them: the line pays 0.4 and 0.2 mm
# a 20 ms step, they go 0.4 and 0.2 mm along it, 1 and 3 mm from it; one foot
# in ten lifts off each step; the feet lag those ahead by 0.5 rad, and the
# two sides alternate.
STEP_FACTS = {
    "reward_parts": [[0.4, -0.1], [0.2, -0.1]],
    "segment_progress": [0.0004, 0.0002],
    "off_line": [0.001, 0.003],
    "lift_off": [[0.1, 0.0], [0.0, 0.1]],
    "contact_flags": [[1, 1, 0.5, 0], [1, 1, 0.0, 0]],
    "neighbour_lag": [[0.5, 0.5]],
    "neighbour_lag_consistency": [[1.0, 1.0]],
    "left_right_lag": [math.pi, math.pi],
    "left_right_lag_consistency": [0.5, 0.5],
}
RESULTS = [
    *("15.0", "mm/s", "15.0", "mm/s", "2.0", "25%"),
    *("+29", "1.00", "+180", "0.50", "2.5", "25%"),
]


def test_a_window_redraws_its_line_as_it_fills_then_ends_with_its_results(capsys):
    progress = TrainingProgress(first_cycle=3, total_cycles=12, window_steps=4)
    for steps_taken in range(1, 5):
        progress.step(steps_taken)
    progress.learning()
    progress.finish(
        {
            "timing": {"collecting_seconds": 20.5, "learning_seconds": 2.6},
            "step_facts": STEP_FACTS,
            "episodes": {"strayed": 0.25},
        }
    )

    output = capsys.readouterr().out
    drawings = output.split("\r")[1:]
    assert drawings[0].startswith(" 3/12  [#####...............]  1/4 steps")
    assert drawings[4].startswith(" 3/12  [####################]  learning")
    assert drawings[5].endswith("\n")  # the results end the window's line
    assert "\n" not in "".join(drawings[:5])  # until then it is redrawn in place
    # The time left: the 9 windows after window 3 at 23.1 s each.
    assert drawings[5].split() == ["3/12", "23.1", "s", "|", *RESULTS, "|", "3m28s"]


def test_an_evaluation_pass_fills_toward_the_time_limit_then_shows_its_episodes(
    capsys,
):
    progress = EvaluationProgress(total_passes=2, max_episode_steps=10)
    progress.start("agents", seed=7)
    for steps_taken in range(1, 6):
        progress.step(steps_taken)
    progress.finish({"step_facts": STEP_FACTS, "episodes": {"strayed": 0.25}})

    drawings = capsys.readouterr().out.split("\r")[1:]
    assert drawings[0].startswith(" 1/2  agents, seed 7  ")
    assert "[##..................]  1/10 steps" in drawings[0]
    row = drawings[-1].split()
    assert row[:4] == ["1/2", "agents,", "seed", "7"]
    assert row[5:] == ["|", *RESULTS]


def test_the_settings_header_names_the_six_training_settings_and_what_they_make():
    values = {
        "run": {"seed": 1},
        "environment": {"max_episode_steps": 1024, "simulation": {"world_count": 64}},
        "agents": {"ppo": {"update_epochs": 4, "minibatch_size": 500}},
        "interaction_loop": {"rollout_window_steps": 64, "update_cycles": 32},
    }

    lines = training_settings(values, {"run.seed"}).splitlines()

    assert lines[1].split() == ["worlds", "64", "world_count"]
    assert lines[6].split() == ["epochs", "4", "update_epochs"]
    # 4,096 samples in minibatches of 500: eight full ones and a smaller last one.
    assert "= 4,096 samples" in lines[8] and "4 epochs x 9 minibatches = 36" in lines[9]
    assert "= 2,048 steps per world (2 episode lengths)" in lines[10]
    assert "131,072 samples" in lines[11]
    assert "  [run]  seed* = 1" in lines
    assert "  [environment]  max_episode_steps = 1024" in lines
