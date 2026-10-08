"""Tests for the terminal's view of a training run: one line per window."""

from centipede.experiment.progress import (
    EvaluationProgress,
    TrainingProgress,
    reward_weights,
    training_settings,
)


def test_a_window_redraws_its_line_as_it_fills_then_ends_with_its_results(capsys):
    progress = TrainingProgress(first_cycle=3, total_cycles=12, window_steps=4)
    for steps_taken in range(1, 5):
        progress.step(steps_taken)
    progress.learning()
    progress.finish(
        {
            "timing": {"collecting_seconds": 20.5, "learning_seconds": 2.6},
            # Two segments; reward parts and contact flags as the log holds them.
            "step_facts": {
                "reward_parts": [[0.0, -0.003, -0.001, 0.0], [0.0, -0.002, 0.0, 0.0]],
                "contact_flags": [[1, 1, 0.5, 0], [1, 1, 0.0, 0]],
                "head_distance": 0.0567,
            },
            "episodes": {"episode_ended": 4, "arrived": 0.25, "left_range": 0.25},
        },
        remaining_s=402,
    )

    output = capsys.readouterr().out
    drawings = output.split("\r")[1:]
    assert drawings[0].startswith(" 3/12  [#####...............]  1/4 steps")
    assert drawings[4].startswith(" 3/12  [####################]  learning")
    assert drawings[5].endswith("\n")  # the results end the window's line
    assert "\n" not in "".join(drawings[:5])  # until then it is redrawn in place
    assert drawings[5].split() == [
        *("3/12", "[####################]", "20.5", "s", "2.6", "s", "6m42s", "|"),
        *("-0.00300", "57", "mm", "25%", "1", "2", "1"),  # arrived, timed out, too far
    ]


def test_an_evaluation_pass_fills_toward_the_time_limit_then_shows_its_episodes(
    capsys,
):
    progress = EvaluationProgress(total_passes=2, max_episode_steps=10)
    progress.start("agents", seed=7)
    for steps_taken in range(1, 6):  # every world arrived after 5 steps
        progress.step(steps_taken)
    progress.finish(
        {
            "episodes": {
                "arrived": 0.5,
                "segment_return": [1, -3],
                "final_distance": 0.0123,
            }
        }
    )

    drawings = capsys.readouterr().out.split("\r")[1:]
    assert drawings[0].startswith(" 1/2  agents, seed 7  ")
    assert "[##..................]  1/10 steps" in drawings[0]
    # Finished early: the bar is full, and the row holds the first episodes.
    row = drawings[-1].split()
    assert row[:5] == ["1/2", "agents,", "seed", "7", "[" + "#" * 20 + "]"]
    assert row[-5:] == ["|", "50%", "-1", "12.3", "mm"]


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
