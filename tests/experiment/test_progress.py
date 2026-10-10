"""Tests for the terminal's view of a training run: one line per window."""

import math

from centipede.experiment.progress import (
    EvaluationProgress,
    TrainingProgress,
    reward_weights,
    training_settings,
)

# Step facts of two segments, as the log holds them: the head moves 0.4 mm per
# 20 ms step, 0.3 mm of it toward its target; legs touch in 30% of the steps.
STEP_FACTS = {
    "reward_parts": [[0.0, -0.003, -0.001, 0.0], [0.0, -0.002, 0.0, 0.0]],
    "contact_flags": [[1, 1, 0.5, 0.2], [1, 1, 0.0, 0.4]],
    "segment_moved": [0.0004, 0.0002],
    "head_progress": 0.0003,
}


def window(endings):
    return {
        "timing": {"collecting_seconds": 20.5, "learning_seconds": 2.6},
        "step_facts": STEP_FACTS,
        "episode_distributions": {"ending": endings},
    }


def test_a_window_redraws_its_line_as_it_fills_then_ends_with_its_results(capsys):
    # Episodes of two windows: the endings count the last two windows, and so
    # do the targets per world and minute (two worlds, 20 ms steps).
    progress = TrainingProgress(
        first_cycle=3, total_cycles=12, window_steps=4, episode_steps=8, world_count=2
    )
    for steps_taken in range(1, 5):
        progress.step(steps_taken)
    progress.learning()
    progress.finish(window([1, 0, 0, 1, 2, 0]))
    progress.finish(window([0, 0, 0, 0, 0, 0]))
    progress.finish(window([0, 1, 0, 0, 0, 1]))

    output = capsys.readouterr().out
    drawings = output.split("\r")[1:]
    assert drawings[0].startswith(" 3/12  [#####...............]  1/4 steps")
    assert drawings[4].startswith(" 3/12  [####################]  learning")
    assert drawings[5].endswith("\n")  # the results end the window's line
    assert "\n" not in "".join(drawings[:5])  # until then it is redrawn in place
    # The time left: the 9 windows after window 3 at 23.1 s each. One arrival
    # in 4 steps of 2 worlds is 375 targets per world and minute.
    assert drawings[5].split() == [
        *("3/12", "23.1", "s", "|", "-0.00300", "20.0", "mm/s", "15.0", "mm/s"),
        *("25%", "30%", "|", "4", "375.0", "25%", "0%", "0%", "25%", "50%", "0%"),
        *("|", "3m28s"),
    ]
    assert drawings[6].split()[-10:] == [
        *("4", "187.5", "25%", "0%", "0%", "25%", "50%", "0%", "|", "3m05s")
    ]
    assert drawings[7].split()[-10:-2] == [
        *("2", "0.0", "0%", "50%", "0%", "0%", "0%", "50%")
    ]


def test_an_evaluation_pass_fills_toward_the_time_limit_then_shows_its_episodes(
    capsys,
):
    progress = EvaluationProgress(total_passes=2, max_episode_steps=10)
    progress.start("agents", seed=7)
    for steps_taken in range(1, 6):  # every world arrived after 5 steps
        progress.step(steps_taken)
    episodes = {"ending": [1, 1, 0, 0, 0, 0]}
    progress.finish({"step_facts": STEP_FACTS, "episode_distributions": episodes})
    # A walk fills toward its own length and shows the targets per minute.
    progress.start("agents walking", seed=7, steps=20)
    progress.step(1)
    progress.finish(
        {
            "step_facts": STEP_FACTS,
            "episode_distributions": episodes,
            "targets_per_minute": 12.5,
        }
    )

    drawings = capsys.readouterr().out.split("\r")[1:]
    assert drawings[0].startswith(" 1/2  agents, seed 7  ")
    assert "[##..................]  1/10 steps" in drawings[0]
    # Finished early, and the row holds the first episodes.
    rows = [drawing.split() for drawing in drawings if drawing.endswith("\n")]
    assert rows[0][:4] == ["1/2", "agents,", "seed", "7"]
    assert rows[0][-15:] == [
        *("|", "20.0", "mm/s", "15.0", "mm/s", "25%", "30%", "|", "-"),
        *("50%", "50%", "0%", "0%", "0%", "0%"),
    ]
    assert "[#...................]  1/20 steps" in drawings[-2]
    assert rows[1][-7:-6] == ["12.5"]


def test_a_run_with_clocks_shows_its_rhythm_under_each_window(capsys):
    progress = TrainingProgress(
        first_cycle=1,
        total_cycles=2,
        window_steps=4,
        episode_steps=8,
        world_count=2,
        clocks=True,
    )
    progress.header()
    rhythm = {
        "tempo": [2.0, 3.0],
        "neighbour_offset": [math.pi / 4],
        "neighbour_offset_consistency": [0.5],
        "neighbour_offset_lock": [0.9],
        "legs_on_tempo": [0.8, 0.6],
        "foot_slip": [0.1, 0.3],
    }
    progress.finish(window([1, 0, 0, 0, 0, 0]) | {"rhythm": rhythm})

    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("clocks: the segments' mean tempo") for line in lines)
    # Under the cycle's label. A 45 degree lag at 2.5 Hz is an eighth of a
    # 20-step turn: 2.5 steps. The head pays 0.004 a step, the follower 0.002.
    assert lines[-1].startswith(" " * 7 + "clocks")
    assert lines[-1].split() == [
        *("clocks", "2.50", "Hz", "(2.00-3.00)", "offset", "+45", "deg", "="),
        *("+2.5", "steps", "lock", "0.90", "consistency", "0.50", "legs", "on"),
        *("tempo", "70%", "slip", "0.20", "|", "head", "-0.00400", "followers"),
        "-0.00200",
    ]


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


def test_the_reward_weights_follow_the_settings_header():
    text = reward_weights(
        {"arrival": 1.0, "progress": 0.2314, "step_cost": 0.000904},
        {"step_cost": 0.2314, "body_contact": 0.3471},
        discount=0.99729,
    )

    lines = text.splitlines()
    assert lines[1].split() == [
        *("per", "step:", "arrival", "1", "progress", "0.231"),
        *("step_cost", "0.000904"),
    ]
    assert lines[4].endswith("step_cost 0.231  body_contact 0.347  (budget 0.579)")
    assert lines[5] == "  discount 0.99729"
