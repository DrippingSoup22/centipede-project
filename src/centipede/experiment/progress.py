"""The terminal's view of a training run: one line per window, drawn as it fills.

The experiment prints a header, then hands a ``TrainingProgress`` to the
interaction loop's diagnostics, which call ``step`` after every step of a
window and ``learning`` when its update starts; the line shows a bar of ``#``
and ``.`` filling as the window is collected, then ``learning``. After the
cycle, the experiment calls ``finish``, which replaces the line with the
window's results. They are values of every step of every world, so they move
smoothly from window to window: the reward per step, the head's distance to
its target, and the share of steps with a body on the ground; episodes are
only counted, as arrivals and time-outs, because all worlds start together and
their time-outs come in waves. Lines are redrawn in place with a carriage
return, only when the bar grows, so a terminal, or the GPU desktop's launcher
passing the output through, shows one line per window.
"""

from typing import Any

BAR_WIDTH = 20
COLUMNS = (
    "{cycle}  {bar}  {collect:>7}  {learn:>6}  {left:>7}"
    "  |  {reward:>11}  {distance:>9}  {body_down:>9}  {arrived:>7}  {timed_out:>9}"
)
# Where the body-on-ground flag sits among a segment's contact flags.
BODY_ON_GROUND = 2


def duration(seconds: float) -> str:
    """A duration as ``2h05m``, ``7m30s``, or ``45s``."""
    seconds = round(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class TrainingProgress:
    """One line per window: a filling bar while it runs, then its results."""

    def __init__(self, first_cycle: int, total_cycles: int, window_steps: int) -> None:
        self._cycle = first_cycle
        self._total_cycles = total_cycles
        self._window_steps = window_steps
        self._cycle_width = max(len("cycle"), 2 * len(str(total_cycles)) + 1)
        self._filled = -1
        self._drawn_length = 0

    def header(self) -> None:
        """The column names, once before the first window."""
        print(
            COLUMNS.format(
                cycle="cycle".rjust(self._cycle_width),
                bar="window".ljust(BAR_WIDTH + 2),
                collect="collect",
                learn="learn",
                left="left",
                reward="reward/step",
                distance="to target",
                body_down="body down",
                arrived="arrived",
                timed_out="timed out",
            ),
            flush=True,
        )

    def step(self, steps_taken: int) -> None:
        """Redraw the line if the bar has grown since the last step."""
        filled = BAR_WIDTH * steps_taken // self._window_steps
        if filled != self._filled:
            self._filled = filled
            self._draw(f"{steps_taken}/{self._window_steps} steps", end="")

    def learning(self) -> None:
        self._draw("learning", end="")

    def finish(self, record: dict[str, Any], remaining_s: float) -> None:
        """Replace the line with the window's results and start a new one."""
        timing, episodes, step = (
            record["timing"],
            record["episodes"],
            record["step_facts"],
        )
        segments = step["reward_parts"]
        reward_per_step = sum(sum(parts) for parts in segments) / len(segments)
        flags = step["contact_flags"]
        body_down = sum(segment[BODY_ON_GROUND] for segment in flags) / len(flags)
        ended = episodes["episode_ended"]
        arrived = round(ended * episodes["arrived"]) if ended else 0
        row = COLUMNS.format(
            cycle=self._cycle_label(),
            bar=self._bar(BAR_WIDTH),
            collect=f"{timing['collecting_seconds']:.1f} s",
            learn=f"{timing['learning_seconds']:.1f} s",
            left=duration(remaining_s),
            reward=f"{reward_per_step:+.5f}",
            distance=f"{step['head_distance'] * 1000:.0f} mm",
            body_down=f"{body_down:.0%}",
            arrived=f"{arrived}",
            timed_out=f"{round(ended) - arrived}",
        )
        self._write(row, end="\n")
        self._cycle += 1
        self._filled = -1
        self._drawn_length = 0

    def _draw(self, state: str, end: str) -> None:
        self._write(
            f"{self._cycle_label()}  {self._bar(max(self._filled, 0))}  {state}", end
        )

    def _write(self, line: str, end: str) -> None:
        # Spaces cover the end of a longer line drawn before at this place.
        print("\r" + line.ljust(self._drawn_length), end=end, flush=True)
        self._drawn_length = len(line)

    def _cycle_label(self) -> str:
        return f"{self._cycle}/{self._total_cycles}".rjust(self._cycle_width)

    @staticmethod
    def _bar(filled: int) -> str:
        return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"
