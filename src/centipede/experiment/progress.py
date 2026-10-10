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
grows, so a terminal, or a log that captures the output, shows one line per
pass.

In training a pass is one window. Its results are the time it took, values of
every step of every world, which do not depend on where each world is in its
episode and so move smoothly from window to window (the reward per step, the
head's speed along the ground and toward its target, and the shares of steps
with a body on the ground and with legs touching), and how episodes ended. All
worlds start together, so their episodes end in waves; the endings are
therefore shares of the episodes that ended over the last episode length of
windows, in which every world ends at least one, and the targets reached per
world and minute are counted over the same windows. A run with a curriculum
also shows the level its window's new targets were drawn at.

In evaluation a pass is one actor and seed: every world runs its first
episode, which ends at the target, out of the range circle, or at the time
limit, so the bar fills toward the time limit and jumps to full when every
world has finished sooner. Its results are the same values over the steps of
those first episodes, and how they ended. A walk pass instead runs every world
for a fixed number of steps and counts every episode, with the targets reached
per world and minute.

In a run whose segments have clocks, a second line under each pass shows its
rhythm: the clocks' tempo, the phase offset between neighbours with how steady
it is, how well the legs keep time, how much the feet slip, and the reward per
step of the head and of the followers, whose rewards differ in kind.
"""

import math
import time
from collections import deque
from typing import Any

from centipede.experiment.arrivals import targets_per_minute
from centipede.experiment.report import STEP_SECONDS

BAR_WIDTH = 20
SETTINGS_LINE_WIDTH = 100
# How episodes end, in the order of the environment's ending histogram: at the
# target; out of time with less than a quarter of the start distance left,
# less than half, less than all of it, or no closer; out of the range circle.
ENDINGS = ("arrived", "<1/4", "<1/2", "closer", "not closer", "left circle")
TRAINING_COLUMNS = (
    "{cycle}  {took:>6}  |  {reward:>11}  {speed:>9}  {toward:>9}  {body_down:>9}"
    "  {legs:>5}  |  {episodes:>8}  {per_minute:>11}  {endings}  |  {level}{left:>6}"
)
EVALUATION_COLUMNS = (
    "{number}  {actor:<26}  {took:>6}  |  {speed:>9}  {toward:>9}  {body_down:>9}"
    "  {legs:>5}  |  {per_minute:>11}  {endings}"
)
# Where the body-on-ground and legs-touching flags sit among a segment's
# contact flags.
BODY_ON_GROUND = 2
LEGS_TOUCHING = 3
CLOCK_LEGEND = (
    "clocks: the segments' mean tempo (lowest-highest segment); offset: the mean"
    " phase offset\nbetween neighbours, positive when the rear lags (a wave from"
    " head to tail), in degrees and\nin steps; lock: how steady each pair's offset"
    " stays within a world; consistency: how alike\nthe offsets are across every"
    " world and step, both 0 to 1; legs on tempo: 1 - the legs-off-tempo\ncost,"
    " on the steps the clock compared the legs; slip: the foot-slip cost, 0 to"
    " 1; head,\nfollowers: the reward per step."
)


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
            "    progress is per halving of the head's distance to its target,",
            "    received by every follower too, at follower_progress_share",
        ]
    if "movement" in per_step:
        lines.append(
            "    movement is per step of moving the joints as much as random"
            " commands do; calm movements pay a small fraction of it"
        )
    if episode_shares:
        lines.append(
            "  each cost over the cost horizon, in arrival rewards:  "
            + "  ".join(f"{name} {share:.3g}" for name, share in episode_shares.items())
            + f"  (budget {sum(episode_shares.values()):.3g})"
        )
    lines.append(f"  discount {discount:.6g}")
    return "\n".join(lines)


def recording_sizes(
    file_count: int, world_count: int, frames: int, position_count: int
) -> str:
    """The recordings a training run will write, with their expected sizes.

    A world's frame holds its ``position_count`` positions as 32-bit numbers,
    the target's position, and whether an episode started there; the rest of
    a file is small.
    """
    file_bytes = frames * world_count * (4 * position_count + 13)
    files = f"{file_count:,} file{'' if file_count == 1 else 's'}"
    return (
        f"Recordings: {files} of {world_count:,} worlds x {frames:,} steps,"
        f" about {file_bytes / 1e6:.3g} MB each, {file_count * file_bytes / 1e6:.3g}"
        " MB in all"
    )


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


def _legend(counted_episodes: str) -> str:
    """What the columns mean, printed above their names."""
    return (
        "speed: the head's, along the ground; toward: how fast it closes on its"
        " target;\nbody down, legs: shares of steps with a body on the ground and"
        " with legs touching,\nover the segments. targets/min: targets reached"
        f" per world and minute.\nEndings: shares of {counted_episodes}. Those"
        " out of time are split by how much of\nthe start distance was left:"
        " <1/4, <1/2, closer (less than all of it), or not closer."
    )


def _speed(metres_per_step: float) -> str:
    return f"{metres_per_step * 1000 / STEP_SECONDS:.1f} mm/s"


def _behaviour(step: dict[str, Any]) -> dict[str, str]:
    """The head's speeds and the shares of steps with a body down and with legs
    touching, from step facts."""
    flags = step["contact_flags"]
    body_down = sum(segment[BODY_ON_GROUND] for segment in flags) / len(flags)
    legs = sum(segment[LEGS_TOUCHING] for segment in flags) / len(flags)
    return {
        "speed": _speed(step["segment_moved"][0]),
        "toward": _speed(step["head_progress"]),
        "body_down": f"{body_down:.0%}",
        "legs": f"{legs:.0%}",
    }


def _endings(counts: list[float]) -> str:
    """Each ending's share of the counted episodes, under its column name."""
    total = sum(counts)
    shares = [f"{count / total:.0%}" if total else "-" for count in counts]
    return "  ".join(
        share.rjust(len(name)) for share, name in zip(shares, ENDINGS, strict=True)
    )


def _clock_line(record: dict[str, Any], indent: int) -> str:
    """A pass's rhythm, from its log record, printed under its row."""
    rhythm = record["rhythm"]
    tempo = rhythm["tempo"]
    mean_tempo = sum(tempo) / len(tempo)
    # The mean offset of all pairs: the direction of the mean of the pairs'
    # mean unit vectors, whose lengths are their consistencies.
    pairs = list(
        zip(
            rhythm["neighbour_offset"],
            rhythm["neighbour_offset_consistency"],
            strict=True,
        )
    )
    offset = math.degrees(
        math.atan2(
            sum(length * math.sin(angle) for angle, length in pairs),
            sum(length * math.cos(angle) for angle, length in pairs),
        )
    )
    # The offset as a delay: its share of a turn, times the steps of a turn.
    delay_steps = offset / 360 / (mean_tempo * STEP_SECONDS)
    consistency = sum(length for _, length in pairs) / len(pairs)
    lock = sum(rhythm["neighbour_offset_lock"]) / len(pairs)
    # None for a segment whose clock compared no legs in the window, as in a
    # window shorter than a turn after a restart.
    compared = [share for share in rhythm["legs_on_tempo"] if share is not None]
    on_tempo = f"{sum(compared) / len(compared):.0%}" if compared else "-"
    slip = sum(rhythm["foot_slip"]) / len(tempo)
    rewards = [sum(parts) for parts in record["step_facts"]["reward_parts"]]
    followers = sum(rewards[1:]) / len(rewards[1:])
    return (
        f"{' ' * indent}clocks {mean_tempo:.2f} Hz ({min(tempo):.2f}-{max(tempo):.2f})"
        f"  offset {offset:+.0f} deg = {delay_steps:+.1f} steps  lock {lock:.2f}"
        f"  consistency {consistency:.2f}  legs on tempo {on_tempo}"
        f"  slip {slip:.2f}  |  head {rewards[0]:+.5f}  followers {followers:+.5f}"
    )


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
                f"  {steps_taken}/{self._steps_per_pass} steps",
                end="",
            )

    def learning(self) -> None:
        self._write(f"{self._label()}  {_bar(BAR_WIDTH)}  learning", end="")

    def _label(self) -> str:
        raise NotImplementedError

    def _end_pass(self, row: str, line_under: str | None = None) -> None:
        """Replace the line with the pass's results, and any line to print
        under them, and start a new one."""
        self._write(row, end="\n")
        if line_under is not None:
            print(line_under, flush=True)
        self._filled = -1
        self._drawn_length = 0

    def _write(self, line: str, end: str) -> None:
        # Spaces cover the end of a longer line drawn before at this place.
        print("\r" + line.ljust(self._drawn_length), end=end, flush=True)
        self._drawn_length = len(line)


class TrainingProgress(_RedrawnLine):
    """One line per window: a filling bar while it runs, then its results."""

    def __init__(
        self,
        first_cycle: int,
        total_cycles: int,
        window_steps: int,
        episode_steps: int,
        world_count: int,
        curriculum: bool = False,
        clocks: bool = False,
    ) -> None:
        super().__init__(window_steps)
        self._world_count = world_count
        self._curriculum = curriculum
        self._clocks = clocks
        self._cycle = first_cycle
        self._total_cycles = total_cycles
        self._cycle_width = max(len("cycle"), 2 * len(str(total_cycles)) + 1)
        # The time the windows so far took, for the training time left.
        self._seconds_taken = 0.0
        self._windows_taken = 0
        # The ending counts of the windows that make the last episode length.
        self._recent_endings: deque[list[float]] = deque(
            maxlen=math.ceil(episode_steps / window_steps)
        )

    def header(self) -> None:
        """The legend and the column names, once before the first window."""
        span = self._recent_endings.maxlen
        print(
            _legend(
                f"the episodes that ended in the last {span} windows, one episode"
                " length,\nin which every world ends at least one"
            ),
            flush=True,
        )
        if self._curriculum:
            print(
                "level: the curriculum's target difficulty, from 0 (the targets of"
                " [environment.target])\nto 1 (the final ranges of [curriculum]).",
                flush=True,
            )
        if self._clocks:
            print(CLOCK_LEGEND, flush=True)
        print(
            TRAINING_COLUMNS.format(
                cycle="cycle".rjust(self._cycle_width),
                took="took",
                reward="reward/step",
                speed="speed",
                toward="toward",
                body_down="body down",
                legs="legs",
                episodes="episodes",
                per_minute="targets/min",
                endings="  ".join(ENDINGS),
                level="level  |  " if self._curriculum else "",
                left="left",
            ),
            flush=True,
        )

    def finish(self, record: dict[str, Any]) -> None:
        """Replace the line with the window's results."""
        timing, step = record["timing"], record["step_facts"]
        segments = step["reward_parts"]
        reward_per_step = sum(sum(parts) for parts in segments) / len(segments)
        self._recent_endings.append(record["episode_distributions"]["ending"])
        counts = [sum(column) for column in zip(*self._recent_endings, strict=True)]
        took = timing["collecting_seconds"] + timing["learning_seconds"]
        # The training time left, at the mean time of the windows so far.
        self._seconds_taken += took
        self._windows_taken += 1
        windows_left = self._total_cycles - self._cycle
        left = windows_left * self._seconds_taken / self._windows_taken
        self._end_pass(
            TRAINING_COLUMNS.format(
                cycle=self._label(),
                took=f"{took:.1f} s",
                reward=f"{reward_per_step:+.5f}",
                **_behaviour(step),
                episodes=f"{round(sum(counts))}",
                per_minute=f"{self._per_minute(counts[0]):.1f}",
                endings=_endings(counts),
                level=(
                    f"{record['curriculum']['level']:>5.3f}  |  "
                    if self._curriculum
                    else ""
                ),
                left=duration(left),
            ),
            _clock_line(record, self._cycle_width + 2) if self._clocks else None,
        )
        self._cycle += 1

    def _per_minute(self, arrivals: float) -> float:
        """Targets reached per world and minute over the counted windows."""
        steps = len(self._recent_endings) * self._steps_per_pass
        return targets_per_minute(arrivals, self._world_count, steps)

    def _label(self) -> str:
        return f"{self._cycle}/{self._total_cycles}".rjust(self._cycle_width)


class EvaluationProgress(_RedrawnLine):
    """One line per actor and seed: a bar toward the time limit, then results."""

    def __init__(
        self, total_passes: int, max_episode_steps: int, clocks: bool = False
    ) -> None:
        super().__init__(max_episode_steps)
        self._max_episode_steps = max_episode_steps
        self._clocks = clocks
        self._total_passes = total_passes
        self._number_width = max(len("pass"), 2 * len(str(total_passes)) + 1)
        self._done = 0
        self._actor = ""
        self._started = time.monotonic()

    def header(self) -> None:
        """The legend and the column names, once before the first pass."""
        print(
            _legend("every world's first episode, or of every episode of a walk"),
            flush=True,
        )
        if self._clocks:
            print(CLOCK_LEGEND, flush=True)
        print(
            EVALUATION_COLUMNS.format(
                number="pass".rjust(self._number_width),
                actor="actor and seed",
                took="took",
                speed="speed",
                toward="toward",
                body_down="body down",
                legs="legs",
                per_minute="targets/min",
                endings="  ".join(ENDINGS),
            ),
            flush=True,
        )

    def start(self, actor_name: str, seed: int, steps: int | None = None) -> None:
        """Begin a pass; ``steps`` is a walk's length, None up to the time limit."""
        self._actor = f"{actor_name}, seed {seed}"
        self._steps_per_pass = steps or self._max_episode_steps
        self._started = time.monotonic()

    def finish(self, record: dict[str, Any]) -> None:
        """Replace the line with the pass's results; a walk's record holds
        ``targets_per_minute``."""
        took = time.monotonic() - self._started
        number = self._number()
        self._done += 1
        per_minute = record.get("targets_per_minute")
        self._end_pass(
            EVALUATION_COLUMNS.format(
                number=number,
                actor=self._actor,
                took=duration(took),
                **_behaviour(record["step_facts"]),
                per_minute="-" if per_minute is None else f"{per_minute:.1f}",
                endings=_endings(record["episode_distributions"]["ending"]),
            ),
            _clock_line(record, self._number_width + 2) if self._clocks else None,
        )

    def _number(self) -> str:
        return f"{self._done + 1}/{self._total_passes}".rjust(self._number_width)

    def _label(self) -> str:
        return f"{self._number()}  {self._actor:<26}"
