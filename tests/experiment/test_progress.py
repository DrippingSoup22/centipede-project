"""Tests for the terminal's view of a training run: one line per window."""

from centipede.experiment.progress import TrainingProgress


def test_a_window_redraws_its_line_as_it_fills_then_ends_with_its_results(capsys):
    progress = TrainingProgress(first_cycle=3, total_cycles=12, window_steps=4)
    for steps_taken in range(1, 5):
        progress.step(steps_taken)
    progress.learning()
    progress.finish(
        {
            "timing": {
                "collecting_seconds": 20.5,
                "learning_seconds": 2.6,
                "transitions_per_second": 199.6,
            },
            "episodes": {"episode_ended": 4, "arrived": 0.5, "segment_return": [1, -3]},
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
        *("3/12", "[####################]", "20.5", "s", "2.6", "s", "200"),
        *("6m42s", "|", "4", "50%", "-1"),
    ]
