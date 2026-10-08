"""The terminal's view of training and evaluation: one line per pass, drawn as it fills.

Before training, ``training_settings`` lists the run's settings: the six that
shape training, under the names used everywhere (worlds, episode length,
rollout window, update cycles, minibatch size, epochs), what they add up to,
and then every other setting, marked where the file sets it itself.

The experiment prints a header, then hands a ``TrainingProgress`` or an
``EvaluationProgress`` to the interaction loop's diagnostics, which call
``step`` after every step and ``learning`` when an update starts; the line
shows a bar of ``#`` and ``.`` filling as the steps are taken. When the pass
ends, the experiment calls ``finish``, which replaces the line with its
results. Lines are redrawn in place with a carriage return, only when the bar
grows, so a terminal, or the GPU desktop's launcher passing the output
through, shows one line per pass.

In training a pass is one window. Its results are values of every step of
every world, so they move smoothly from window to window: the reward per step,
the head's distance to its target, and the share of steps with a body on the
ground; episodes are only counted, as arrivals and time-outs, because all
worlds start together and their time-outs come in waves.

In evaluation a pass is one actor and seed: every world runs its first
episode, which ends at the target or at the time limit, so the bar fills
toward the time limit and jumps to full when every world has finished sooner.
Its results are those first episodes: the share that arrived, their mean
return, and the head's distance to its target at the end.
"""

import math
import time
from typing import Any

BAR_WIDTH = 20
SETTINGS_LINE_WIDTH = 100
TRAINING_COLUMNS = (
    "{cycle}  {bar}  {collect:>7}  {learn:>6}  {left:>7}"
    "  |  {reward:>11}  {distance:>9}  {body_down:>9}  {arrived:>7}  {timed_out:>9}"
)
EVALUATION_COLUMNS = (
    "{number}  {actor:<26}  {bar}  {took:>6}  {left:>7}"
    "  |  {arrived:>7}  {mean_return:>11}  {distance:>12}"
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


def training_settings(values: dict[str, Any], written_keys: set[str]) -> str:
    """The run's settings, as printed before training.

    ``values`` are the complete training settings as nested tables, every
    default filled in; ``written_keys`` are the dotted names the file sets.
    """
    loop, ppo = values["interaction_loop"], values["agents"]["ppo"]
    worlds = values["environment"]["simulation"]["world_count"]
    episode_length = values["environment"]["max_episode_steps"]
    window, cycles = loop["rollout_window_steps"], loop["update_cycles"]
    minibatch_size, epochs = ppo["minibatch_size"], ppo["update_epochs"]
    samples = worlds * window
    minibatches = math.ceil(samples / minibatch_size)
    steps_per_world = cycles * window
    lines = [
        "Training                          (set in the file as)",
        *(
            f"  {name:<14}  {amount:<16}  {key}"
            for name, amount, key in (
                ("worlds", f"{worlds:,}", "world_count"),
                ("episode length", f"{episode_length:,} steps", "max_episode_steps"),
                ("rollout window", f"{window:,} steps", "rollout_window_steps"),
                ("update cycles", f"{cycles:,}", "update_cycles"),
                ("minibatch size", f"{minibatch_size:,} samples", "minibatch_size"),
                ("epochs", f"{epochs:,}", "update_epochs"),
            )
        ),
        "",
        f"  Each update cycle: {worlds:,} worlds x {window:,} steps ="
        f" {samples:,} samples for each segment agent,",
        f"    used in {epochs:,} epochs x {minibatches:,} minibatches ="
        f" {epochs * minibatches:,} gradient steps.",
        f"  The whole run: {cycles:,} update cycles x {window:,} steps ="
        f" {steps_per_world:,} steps per world"
        f" ({steps_per_world / episode_length:.3g} episode lengths),",
        f"    {steps_per_world * worlds:,} samples for each segment agent.",
        "",
        "All settings (* set by the file, the others are defaults)",
    ]
    for section, settings in _settings_tables(values):
        items = [
            f"{key}{'*' if f'{section}.{key}' in written_keys else ''}"
            f" = {_setting_text(value)}"
            for key, value in settings.items()
        ]
        line = f"  [{section}]"
        indent = " " * 4
        for item in items:
            if len(line) + 2 + len(item) > SETTINGS_LINE_WIDTH:
                lines.append(line)
                line = indent + item
            else:
                line += ("  " if line.strip() else "") + item
        lines.append(line)
    return "\n".join(lines)


def reward_weights(
    per_step: dict[str, float], episode_shares: dict[str, float], discount: float
) -> str:
    """The weights the reward's rules give each term, printed under the settings.

    ``per_step`` and ``episode_shares`` are those of ``RewardWeights``; the
    agents' ``discount`` is shown with them, since the rules rely on it.
    """
    lines = [
        "Reward weights (from [environment.rewards]; rules in docs/environment.md)",
        "  per step:  "
        + "  ".join(f"{name} {weight:.3g}" for name, weight in per_step.items()),
    ]
    if "progress" in per_step:
        lines += [
            "    progress is per halving of the head's distance to its target;",
            "    a follower's halvings count at follower_progress_ratio",
        ]
    if episode_shares:
        lines.append(
            "  a whole episode of each cost, in arrival rewards:  "
            + "  ".join(f"{name} {share:.3g}" for name, share in episode_shares.items())
            + f"  (budget {sum(episode_shares.values()):.3g})"
        )
    lines.append(f"  discount {discount:.6g}")
    return "\n".join(lines)


def _settings_tables(values: dict[str, Any], prefix: str = ""):
    """Each table's own settings, as (dotted table name, settings), in file order."""
    own = {key: value for key, value in values.items() if not isinstance(value, dict)}
    if own:
        yield prefix.rstrip("."), own
    for key, value in values.items():
        if isinstance(value, dict):
            yield from _settings_tables(value, f"{prefix}{key}.")


def _setting_text(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_setting_text(item) for item in value) + "]"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    return f"{value:g}" if isinstance(value, float) else str(value)


def _bar(filled: int) -> str:
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


class _RedrawnLine:
    """A terminal line drawn again in place, as a pass's bar grows."""

    def __init__(self, steps_per_pass: int) -> None:
        self._steps_per_pass = steps_per_pass
        self._filled = -1
        self._drawn_length = 0

    def step(self, steps_taken: int) -> None:
        """Redraw the line if the bar has grown since the last step."""
        filled = min(BAR_WIDTH, BAR_WIDTH * steps_taken // self._steps_per_pass)
        if filled != self._filled:
            self._filled = filled
            self._write(
                f"{self._label()}  {_bar(filled)}"
                f"  {steps_taken}/{self._steps_per_pass} steps{self._estimate()}",
                end="",
            )

    def learning(self) -> None:
        self._write(f"{self._label()}  {_bar(BAR_WIDTH)}  learning", end="")

    def _label(self) -> str:
        raise NotImplementedError

    def _estimate(self) -> str:
        return ""

    def _end_pass(self, row: str) -> None:
        """Replace the line with the pass's results and start a new one."""
        self._write(row, end="\n")
        self._filled = -1
        self._drawn_length = 0

    def _write(self, line: str, end: str) -> None:
        # Spaces cover the end of a longer line drawn before at this place.
        print("\r" + line.ljust(self._drawn_length), end=end, flush=True)
        self._drawn_length = len(line)


class TrainingProgress(_RedrawnLine):
    """One line per window: a filling bar while it runs, then its results."""

    def __init__(self, first_cycle: int, total_cycles: int, window_steps: int) -> None:
        super().__init__(window_steps)
        self._cycle = first_cycle
        self._total_cycles = total_cycles
        self._cycle_width = max(len("cycle"), 2 * len(str(total_cycles)) + 1)

    def header(self) -> None:
        """The column names, once before the first window."""
        print(
            TRAINING_COLUMNS.format(
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

    def finish(self, record: dict[str, Any], remaining_s: float) -> None:
        """Replace the line with the window's results."""
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
        self._end_pass(
            TRAINING_COLUMNS.format(
                cycle=self._label(),
                bar=_bar(BAR_WIDTH),
                collect=f"{timing['collecting_seconds']:.1f} s",
                learn=f"{timing['learning_seconds']:.1f} s",
                left=duration(remaining_s),
                reward=f"{reward_per_step:+.5f}",
                distance=f"{step['head_distance'] * 1000:.0f} mm",
                body_down=f"{body_down:.0%}",
                arrived=f"{arrived}",
                timed_out=f"{round(ended) - arrived}",
            )
        )
        self._cycle += 1

    def _label(self) -> str:
        return f"{self._cycle}/{self._total_cycles}".rjust(self._cycle_width)


class EvaluationProgress(_RedrawnLine):
    """One line per actor and seed: a bar toward the time limit, then results."""

    def __init__(self, total_passes: int, max_episode_steps: int) -> None:
        super().__init__(max_episode_steps)
        self._total_passes = total_passes
        self._number_width = max(len("pass"), 2 * len(str(total_passes)) + 1)
        self._done = 0
        self._seconds_done = 0.0
        self._actor = ""
        self._started = time.monotonic()

    def header(self) -> None:
        """The column names, once before the first pass."""
        print(
            EVALUATION_COLUMNS.format(
                number="pass".rjust(self._number_width),
                actor="actor and seed",
                bar="first episodes".ljust(BAR_WIDTH + 2),
                took="took",
                left="left",
                arrived="arrived",
                mean_return="mean return",
                distance="end distance",
            ),
            flush=True,
        )

    def start(self, actor_name: str, seed: int) -> None:
        self._actor = f"{actor_name}, seed {seed}"
        self._started = time.monotonic()

    def finish(self, record: dict[str, Any]) -> None:
        """Replace the line with the pass's first episodes."""
        took = time.monotonic() - self._started
        number = self._number()
        self._done += 1
        self._seconds_done += took
        episodes = record["episodes"]
        returns = episodes["segment_return"]
        remaining = self._total_passes - self._done
        self._end_pass(
            EVALUATION_COLUMNS.format(
                number=number,
                actor=self._actor,
                bar=_bar(BAR_WIDTH),
                took=duration(took),
                left=duration(remaining * self._seconds_done / self._done),
                arrived=f"{episodes['arrived']:.0%}",
                mean_return=f"{sum(returns) / len(returns):+.3g}",
                distance=f"{episodes['final_distance'] * 1000:.1f} mm",
            )
        )

    def _number(self) -> str:
        return f"{self._done + 1}/{self._total_passes}".rjust(self._number_width)

    def _label(self) -> str:
        return f"{self._number()}  {self._actor:<26}"

    def _estimate(self) -> str:
        """The evaluation's time left, at most: this pass to the time limit, and
        the passes after it at the average so far (or at this pass's pace)."""
        if self._filled <= 0:
            return ""
        elapsed = time.monotonic() - self._started
        this_pass = elapsed * BAR_WIDTH / self._filled
        later = self._total_passes - self._done - 1
        per_pass = self._seconds_done / self._done if self._done else this_pass
        return f"  at most {duration(this_pass - elapsed + later * per_pass)} left"
