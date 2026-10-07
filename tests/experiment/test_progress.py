"""Tests for the terminal's view of a training run: one line per window."""

from centipede.experiment.progress import TrainingProgress


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
            "episodes": {"episode_ended": 4, "arrived": 0.25},
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
        *("-0.00300", "57", "mm", "25%", "1", "3"),
    ]
