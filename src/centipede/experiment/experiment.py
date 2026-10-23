"""The experiment's front file: one configuration file in, one run's results out.

``run`` reads the file and does what its mode says:

- **train** creates a new run folder, or reopens one with ``continue_from``;
  builds the environment, the agents, and the interaction loop; and after each
  cycle writes one line to the training log. Every ``checkpoint_every_cycles``
  cycles, and after the last, it saves a checkpoint and refreshes the report.
  With ``record_worlds`` it records that many worlds, one episode each, for
  replay, ending by 25, 50, 75 and 100% of the update cycles.
  With ``time_limit_hours`` it stops cleanly, after a checkpoint, before a
  cycle that would end after the limit, so that a run on a machine with a
  session limit never ends in the middle of one; the run is then continued in
  a new session. A run may keep worlds that act with the mean action, and
  tow its body; the tow's top moves after each cycle and is saved with the
  checkpoints. With
  ``start_from`` the new run's agents begin from another run's
  checkpoint, for example to continue learning on a harder task.
- **evaluate** loads a checkpoint, runs every seed with the agents and each
  listed baseline, and writes the results and the recordings into the run.

A run writes no HTML report: the training project's monitor shows its log
while it trains, and its review tools read the evaluations' results.

This is the only component that deals with files. See docs/architecture.md and
docs/configuration.md.
"""

import math
import platform
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from centipede.agents.agents import Agents, RandomActionBaseline, ZeroActionBaseline
from centipede.agents.settings import AgentSettings
from centipede.diagnostics_category import descriptions, values
from centipede.environment.environment import Environment
from centipede.environment.settings import CurveSettings
from centipede.experiment.configuration import (
    Configuration,
    EvaluationSettings,
    dotted_keys,
    read_configuration,
)
from centipede.experiment.progress import (
    EvaluationProgress,
    TrainingProgress,
    duration,
    recording_sizes,
    training_settings,
)
from centipede.experiment.recordings import (
    RecordingScene,
    evaluation_recording,
    training_recording,
)
from centipede.experiment.report import LoggedCategory
from centipede.experiment.run_folder import RunFolder
from centipede.interaction_loop.interaction_loop import InteractionLoop
from centipede.settings_section import SettingsError

# What evaluation can run: the trained agents or a baseline, all with ``act``.
Actor = Agents | ZeroActionBaseline | RandomActionBaseline

# The most device memory a training recording may hold while it runs: the
# poses of the recorded worlds for one episode.
RECORDING_MEMORY_LIMIT_BYTES = 1 << 30
# When training recordings end, in percent of the update cycles.
RECORDING_PERCENTS = (25, 50, 75, 100)

# Recorded with every training session, so a result can be traced to its tools.
RECORDED_PACKAGES = (
    "centipede",
    "rl_lib",
    "torch",
    "mujoco",
    "mujoco-warp",
    "warp-lang",
)
# How many update cycles the curves' start averages over.
WAVE_WINDOW_CYCLES = 8


def run(configuration_path: Path) -> Path:
    """Train or evaluate as the file says; returns the run folder written to."""
    configuration = read_configuration(configuration_path)
    # Only evaluation files have evaluation settings.
    if configuration.evaluation is not None:
        return evaluate(configuration, configuration.evaluation)
    return train(configuration, configuration_path)


# -- Training -------------------------------------------------------------------------


def train(configuration: Configuration, configuration_path: Path) -> Path:
    """Run the training cycles the configuration asks for; returns the run folder."""
    session_start = time.monotonic()
    run_settings = configuration.run
    loop_settings = configuration.interaction_loop
    continue_from = configuration.continue_from
    checkpoint_path = None
    if continue_from is not None:
        checkpoint_path = RunFolder.open(continue_from).latest_checkpoint()
    elif run_settings.start_from is not None:
        parent = RunFolder.open(run_settings.start_from)
        checkpoint_path = parent.checkpoint_named_by(run_settings.start_from)

    # Everything that can fail comes before the run folder is written, so a
    # rejected run leaves nothing behind.
    environment = Environment(configuration.environment)
    worlds_per_group = run_settings.mean_action_worlds_per_group
    agents = _agents_for(
        environment,
        configuration,
        loop_settings.rollout_window_steps,
        mean_action_world_count=2 * worlds_per_group,
    )
    # The tow, if any, fades by what the mean-action worlds do.
    towed = configuration.environment.tow.start > 0
    saved_tow_top = None
    # The ramped costs rise from cycle 1, or, with ramp_after_line_share, from
    # the cycle after the untowed mean-action worlds' line reaches that share
    # of the cap (None until then); a continued run keeps the cycle it found.
    reward_settings = configuration.environment.reward
    ramp_line_mm_per_s = (
        reward_settings.ramp_after_line_share * reward_settings.line_speed_cap_mm_per_s
    )
    ramp_from = None if ramp_line_mm_per_s else 1
    # An adaptive ramp's weight starts at ramp_start and moves after each of
    # the tow's windows by what the untowed mean-action worlds did.
    adaptive_ramp = reward_settings.ramp_shape == "adaptive"
    ramp_weight = reward_settings.ramp_start
    completed_cycles = 0
    if checkpoint_path is not None:
        checkpoint = RunFolder.load_checkpoint(
            checkpoint_path, configuration.agents.device
        )
        if continue_from is not None:
            completed_cycles = checkpoint["completed_cycles"]
            saved_tow_top = checkpoint.get("tow_top")
            ramp_from = checkpoint.get("ramp_from", ramp_from)
            _set_levels(environment, checkpoint, checkpoint_path)
            ramp_weight = checkpoint.get("ramp_weight", ramp_weight)
        else:
            _check_agents_can_start_from(
                checkpoint, configuration.agents, checkpoint_path
            )
        # A new run may give its agents more to see or command than the run
        # it starts from, such as the spine, or wider layers: its agents are
        # then widened.
        agents.load_state_dict(checkpoint["agents"], widen=continue_from is None)
    if towed:
        environment.start_tow(saved_tow_top)
    if worlds_per_group:
        environment.reserve_measuring_worlds(worlds_per_group)
    episode_steps = configuration.environment.max_episode_steps
    record_worlds = min(run_settings.record_worlds, environment.world_count)
    if record_worlds:
        _check_recording_fits(environment, episode_steps, record_worlds)

    if continue_from is not None:
        folder = RunFolder.open(continue_from)
        folder.cut_log_after(completed_cycles)
    else:
        folder = RunFolder.create(run_settings.runs_folder, run_settings.name)
    # A continuing file may have changed update_cycles, so save it again.
    folder.write_configuration(configuration)
    folder.add_session(
        _session_facts(
            configuration_path,
            configuration.agents.device,
            completed_cycles,
            checkpoint_path,
        )
    )

    total_cycles = loop_settings.update_cycles
    print(f"Run folder: {folder.path}")
    if continue_from is not None:
        print(f"Continues at update cycle {completed_cycles + 1} of {total_cycles}")
    elif checkpoint_path is not None:
        print(f"Agents start from {checkpoint_path}")
    print(
        f"{configuration.environment.simulation.backend.upper()} physics,"
        f" agents on {configuration.agents.device}\n"
    )
    print(
        training_settings(
            configuration.training_values(), set(dotted_keys(configuration_path))
        ),
        end="\n\n",
    )
    recording_starts = (
        _recording_starts(
            total_cycles, episode_steps, loop_settings.rollout_window_steps
        )
        if record_worlds
        else {}
    )
    if record_worlds:
        file_count = sum(start > completed_cycles for start in recording_starts)
        position_count = environment.diagnostics.simulation.qpos.shape[1]
        print(
            recording_sizes(file_count, record_worlds, episode_steps, position_count),
            end="\n\n",
        )
    if completed_cycles >= total_cycles:
        print(f"Already {completed_cycles} of {total_cycles} cycles; nothing to do.")
        return folder.path

    # The loop counts the cycles of this session only; the log and checkpoints
    # count from the start of the run. A resumed session draws new episodes.
    learning_worlds = (
        torch.arange(environment.world_count, device=environment.device)
        < agents.learning_world_count
        if worlds_per_group
        else None
    )
    loop = InteractionLoop(
        environment,
        agents,
        replace(loop_settings, update_cycles=total_cycles - completed_cycles),
        counted_worlds=learning_worlds,
    )
    categories = _training_categories(
        loop, agents, environment, measuring=bool(worlds_per_group)
    )
    transitions_per_cycle = environment.world_count * loop_settings.rollout_window_steps

    # Recording: the recorder is armed before a recording's first cycle and
    # taken once complete, both between cycles. One that a session's end
    # leaves incomplete is not written.
    recorder = loop.diagnostics.recorder
    scene = (
        RecordingScene.from_run(environment) if record_worlds else None
    )
    recording_first_cycle = completed_cycles + 1
    if recording_first_cycle in recording_starts:
        recorder.arm(episode_steps, record_worlds)

    # Wall-clock time per cycle in this session, everything included, for the
    # time limit.
    time_limit_s = (
        None
        if run_settings.time_limit_hours is None
        else run_settings.time_limit_hours * 3600
    )
    progress = TrainingProgress(
        completed_cycles + 1, total_cycles, loop_settings.rollout_window_steps
    )
    loop.diagnostics.progress = progress
    progress.header()
    # The learning rate of each update follows the schedule over the run's
    # cycles (an adaptive one is each agent's own), and so do scheduled action
    # spreads, the noise front, which starts with the ramped costs, and their
    # weight, unless it adapts; the loop pauses after every cycle, before the
    # next window.
    agent_settings = configuration.agents

    def follow_schedules(cycle: int) -> None:
        learning_rate = agent_settings.learning_rate_at(cycle, total_cycles)
        if learning_rate is not None:
            agents.set_learning_rate(learning_rate)
        action_std = agent_settings.action_std_at(cycle, total_cycles)
        if action_std is not None:
            agents.set_action_std(action_std)
        if agent_settings.noise_front_cycles_per_unit:
            agents.hold_action_std(
                agent_settings.units_released_at(cycle, ramp_from),
                agent_settings.initial_action_std,
            )
        if agent_settings.action_std_floor:
            agents.hold_action_std(0, agent_settings.action_std_floor)
        environment.set_ramp_weight(
            ramp_weight
            if adaptive_ramp
            else reward_settings.ramp_weight_at(cycle, ramp_from)
        )

    follow_schedules(completed_cycles + 1)
    curve_start = CurveStart(configuration.environment.curves)
    cycles_start = time.monotonic()
    cycles = loop.train(seed=run_settings.seed + completed_cycles)
    for session_cycle in cycles:
        cycle = completed_cycles + session_cycle + 1
        if worlds_per_group:
            window_closed = environment.adjust_tow()
            environment.adjust_curve_level()
            environment.adjust_speed_level()
            if ramp_from is None and environment.untowed_line() >= ramp_line_mm_per_s:
                ramp_from = cycle + 1
            if adaptive_ramp and window_closed:
                ramp_weight = reward_settings.adapted_ramp_weight(
                    ramp_weight, *environment.untowed_walk()
                )
        record = {"cycle": cycle, "transitions": cycle * transitions_per_cycle}
        record |= {category.name: category.plain_values() for category in categories}
        record["ramp_weight"] = environment.reward.ramp_weight
        if environment.line.level_count:
            record["curve_level"] = environment.line.level
            kept = environment.kept_line_share
            record["kept_line_share"] = None if math.isnan(kept) else kept
            if environment.line.level < 0 and curve_start.ready(record):
                environment.start_curves()
        if environment.speeds is not None:
            record["speed_level"] = environment.speeds.level
            record["speed_judgement"] = environment.speed_judgement
        folder.append_log(record)
        if cycle < total_cycles:
            follow_schedules(cycle + 1)
        seconds_per_cycle = (time.monotonic() - cycles_start) / (session_cycle + 1)
        progress.finish(record)
        out_of_time = (
            time_limit_s is not None
            and cycle < total_cycles
            and time.monotonic() - session_start + seconds_per_cycle > time_limit_s
        )
        if recorder.complete():
            percent = recording_starts[recording_first_cycle]
            folder.write_recording(
                f"{percent:03d}pct_cycles_{recording_first_cycle:04d}-{cycle:04d}",
                training_recording(
                    scene,
                    recorder.take(),
                    percent,
                    (recording_first_cycle, cycle),
                    configuration,
                    folder,
                ),
            )
        if (cycle + 1) in recording_starts and not recorder.recording:
            recorder.arm(episode_steps, record_worlds)
            recording_first_cycle = cycle + 1
        if (
            cycle % run_settings.checkpoint_every_cycles == 0
            or cycle == total_cycles
            or out_of_time
        ):
            folder.save_checkpoint(
                cycle,
                agents.state_dict(),
                tow_top=environment.tow.top if towed else None,
                ramp_from=ramp_from,
                ramp_weight=ramp_weight if adaptive_ramp else None,
                curve_level=environment.line.level
                if environment.line.level_count
                else None,
                speed_level=None
                if environment.speeds is None
                else environment.speeds.level,
            )
        if out_of_time:
            print(
                f"Stopped after cycle {cycle} of {total_cycles}: another cycle"
                f" (about {duration(seconds_per_cycle)}) would pass the time"
                f" limit of {run_settings.time_limit_hours:g} h. Continue it with"
                f' continue_from = "{folder.path.as_posix()}".'
            )
            return folder.path
    session_cycles = total_cycles - completed_cycles
    session_seconds = time.monotonic() - session_start
    print(
        f"\nFinished {session_cycles} cycles in {duration(session_seconds)},"
        f" {session_cycles * transitions_per_cycle / session_seconds:,.0f} steps/s"
        f" overall: {folder.path}"
    )
    return folder.path


def _agents_for(
    environment: Environment,
    configuration: Configuration,
    rollout_window_steps: int,
    mean_action_world_count: int = 0,
) -> Agents:
    """Agents that fit the environment: its body and its reward channels; in
    training, the last ``mean_action_world_count`` worlds act with the mean
    action and are not learned from. Without ``actor_sees_behind`` the
    actors do not take the units behind."""
    actor_input_mask = (
        None
        if configuration.agents.actor_sees_behind
        else ~environment.behind_observation_entries
    )
    return Agents(
        environment.segment_count,
        environment.observation_size,
        environment.world_count,
        rollout_window_steps,
        configuration.agents,
        configuration.run.seed,
        environment.segment_action_sizes,
        environment.leg_motor_count,
        environment.reward_channel_count,
        mean_action_world_count,
        actor_input_mask,
    )


def _recording_starts(
    total_cycles: int, episode_steps: int, window_steps: int
) -> dict[int, int]:
    """The cycle each training recording starts at, with its percent.

    A recorded world waits up to an episode for its next start and then
    records one, so a recording spans up to two episodes of windows; each
    starts that early, or at the first cycle, to end by its share of the
    update cycles.
    """
    span = math.ceil(2 * episode_steps / window_steps)
    return {
        max(1, round(percent / 100 * total_cycles) - span + 1): percent
        for percent in RECORDING_PERCENTS
    }


def _check_recording_fits(environment: Environment, frames: int, worlds: int) -> None:
    """Fail clearly where recording one episode would take too much device
    memory: the poses and episode starts of the recorded worlds."""
    position_count = environment.diagnostics.simulation.qpos.shape[1]
    needed = frames * worlds * (4 * position_count + 1)
    if needed > RECORDING_MEMORY_LIMIT_BYTES:
        raise SettingsError(
            f"Recording one episode ({frames:,} steps of {worlds:,} worlds) needs"
            f" {needed / 2**30:.1f} GiB of device memory, more than"
            f" {RECORDING_MEMORY_LIMIT_BYTES / 2**30:.0f} GiB: record fewer worlds"
            " with [run] record_worlds"
        )


class CurveStart:
    """Whether stage 2's curves may begin (``CurveSettings``' start): over
    the last WAVE_WINDOW_CYCLES update cycles, the untowed mean-action
    worlds' speed along the line, their mean, reaches
    ``start_speed_mm_per_s``, and some neighbour pair's wave consistency,
    its mean over both sides and those cycles, reaches
    ``start_pair_consistency``. The walk has formed and the wave is starting
    to form: in the study's runs 12 to 19 a pair passed 0.25 10 to 60 cycles
    before it ordered, and the pairs froze one by one over 100 to 200 more."""

    def __init__(self, curves: CurveSettings) -> None:
        self.curves = curves
        self.speeds: deque[float] = deque(maxlen=WAVE_WINDOW_CYCLES)
        self.consistencies: deque[torch.Tensor] = deque(maxlen=WAVE_WINDOW_CYCLES)

    def ready(self, record: dict[str, Any]) -> bool:
        """Take an update cycle's log ``record``; whether the curves may
        begin."""
        # The log's speeds: the worlds at the tow's top, then the untowed;
        # None before the tow's first window, which no cycle passes.
        speed = record["tow"]["speed"][1]
        self.speeds.append(math.nan if speed is None else speed)
        pairs = [
            [math.nan if value is None else value for value in sides]
            for sides in record["step_facts"]["neighbour_lag_consistency"]
        ]
        self.consistencies.append(torch.tensor(pairs).nanmean(dim=1))
        if len(self.speeds) < WAVE_WINDOW_CYCLES:
            return False
        order = torch.stack(tuple(self.consistencies)).nanmean(dim=0).nan_to_num()
        walking = sum(self.speeds) / WAVE_WINDOW_CYCLES
        return (
            walking >= self.curves.start_speed_mm_per_s
            and order.max().item() >= self.curves.start_pair_consistency
        )


def _set_levels(
    environment: Environment, checkpoint: dict, checkpoint_path: Path
) -> None:
    """Lay the lines at the curves' level a checkpoint reached (none: the
    first), and draw the speeds at its speeds' level (none: as the
    configuration starts them), which the configuration must have."""
    level = checkpoint.get("curve_level", 0)
    if level and level >= environment.line.level_count:
        raise SettingsError(
            f"{checkpoint_path} reached curve level {level}; the configuration's"
            f" [environment.curves] has {environment.line.level_count} levels"
        )
    environment.line.set_level(level)
    speed_level = checkpoint.get("speed_level")
    if speed_level is None:
        return
    speeds = environment.speeds
    if speeds is None or speed_level >= speeds.level_count:
        raise SettingsError(
            f"{checkpoint_path} reached speed level {speed_level}; the"
            " configuration's [environment.speeds] has"
            f" {0 if speeds is None else speeds.level_count} levels"
        )
    speeds.set_level(speed_level)


def _check_agents_can_start_from(
    checkpoint: dict[str, Any], settings: AgentSettings, path: Path
) -> None:
    """Fail clearly where a checkpoint's agents differ from the new run's.

    The networks must have as many hidden layers, each at least as wide (a
    wider one is widened), and the optimizers must be of the same kind, since
    each kind keeps its own state. Checkpoints saved before the optimizer
    could be chosen used Adam.
    """
    saved = checkpoint["agents"]["settings"]
    saved_layers = tuple(saved["hidden_layers"])
    if len(saved_layers) != len(settings.hidden_layers) or any(
        size < saved_size
        for saved_size, size in zip(saved_layers, settings.hidden_layers, strict=True)
    ):
        raise SettingsError(
            f"[agents] hidden_layers must have as many layers as the checkpoint "
            f"{path}, each at least as wide: {list(saved_layers)}, got "
            f"{list(settings.hidden_layers)}"
        )
    saved_optimizer = saved.get("optimizer", "adam")
    if saved_optimizer != settings.optimizer:
        raise SettingsError(
            f"[agents] optimizer must match the checkpoint {path}: "
            f"{saved_optimizer!r}, got {settings.optimizer!r}"
        )


def _training_categories(
    loop: InteractionLoop,
    agents: Agents,
    environment: Environment,
    measuring: bool = False,
) -> list[LoggedCategory]:
    """Every diagnostics category logged once per window, in report order;
    ``measuring`` adds the tow and what the mean-action worlds measured."""
    loop_diagnostics = loop.diagnostics
    return [
        LoggedCategory(
            "episodes",
            "Episodes",
            "Means over the episodes that ended in each window; 'episode ended' "
            "counts them. Windows in which no episode ended have no values.",
            loop.diagnostics.episode_window.descriptions,
            loop_diagnostics.episode_window.result,
        ),
        _distribution_category(
            loop,
            "How many of the episodes that ended in each window fell in each bin.",
        ),
        _step_category(
            loop,
            environment,
            "Means over every world and step of each window; shares are the "
            "fraction of those steps.",
        ),
        LoggedCategory(
            "learning",
            "Learning",
            "From the update at the end of each window, one value per segment agent.",
            descriptions(agents.diagnostics.learning),
            lambda: values(agents.diagnostics.learning),
        ),
        _timing_category(loop),
        _physics_category(loop),
        *(_tow_category(environment) if measuring else []),
    ]


def _tow_category(environment: Environment) -> list[LoggedCategory]:
    """The tow's top after each cycle and the mean-action worlds' last window."""
    facts = environment.tow.facts
    return [
        LoggedCategory(
            "tow",
            "Tow",
            "After each update cycle. The mean-action worlds act with the agents'"
            " mean action, half at the tow's top, half without tow; their values"
            " are over the window that last moved the top, NaN before one.",
            descriptions(facts),
            lambda: values(facts),
        )
    ]


def _distribution_category(loop: InteractionLoop, note: str) -> LoggedCategory:
    """The bin counts of the episode values that declare histogram edges."""
    episode_window = loop.diagnostics.episode_window
    return LoggedCategory(
        "episode_distributions",
        "Episodes, spread",
        note,
        {
            name: description
            for name, description in episode_window.descriptions.items()
            if description.histogram_edges
        },
        episode_window.histograms,
        holds_histograms=True,
    )


def _step_category(
    loop: InteractionLoop, environment: Environment, note: str
) -> LoggedCategory:
    """The environment's step facts over a window, with the reward terms named."""
    step_window = loop.diagnostics.step_window
    descriptions = dict(step_window.descriptions)
    descriptions["reward_parts"] = replace(
        descriptions["reward_parts"],
        parts=tuple(environment.diagnostics.reward_part_names),
    )
    return LoggedCategory(
        "step_facts", "Body, step by step", note, descriptions, step_window.result
    )


def _timing_category(loop: InteractionLoop) -> LoggedCategory:
    """The loop's timing of the last window."""
    timing = loop.diagnostics.timing
    return LoggedCategory(
        "timing",
        "Speed",
        "Measured for each window; the clock waits for the GPU only when "
        "collecting or learning starts and ends.",
        descriptions(timing),
        lambda: values(timing),
    )


def _physics_category(loop: InteractionLoop) -> LoggedCategory:
    """The simulation's facts over a window: how hard the physics worked."""
    simulation_window = loop.diagnostics.simulation_window
    return LoggedCategory(
        "physics",
        "Physics",
        "Maxima over every world and step of each window, read after each step's "
        "last physics call. The GPU backend reserves contacts_per_world times the "
        "number of worlds contacts and constraints_per_world rows per world.",
        simulation_window.descriptions,
        simulation_window.result,
    )


def _session_facts(
    configuration_path: Path,
    device: str,
    first_cycle: int,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """How and with what this training session runs, for ``run_info.json``.

    ``device`` is the agents' device, which matches the physics backend.
    ``settings_written`` lists the settings the file sets itself, rather than
    leaving to their defaults, so that the report can mark them.
    """
    if device == "cuda":
        device_name = torch.cuda.get_device_name()
    else:
        device_name = platform.processor() or platform.machine()
    return {
        "started": f"{datetime.now():%Y-%m-%d %H:%M}",
        "configuration_file": str(configuration_path),
        "settings_written": dotted_keys(configuration_path),
        "first_cycle": first_cycle,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in RECORDED_PACKAGES},
        "device": device_name,
    }


def _git_commit() -> str | None:
    """The code's Git commit, marked when there are uncommitted changes."""
    source = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit} with uncommitted changes" if changes else commit


def _package_version(name: str) -> str | None:
    """An installed package's version, or None when it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


# -- Evaluation -----------------------------------------------------------------------


def evaluate(configuration: Configuration, evaluation: EvaluationSettings) -> Path:
    """Evaluate a checkpoint and the listed baselines; returns the run folder."""
    folder = RunFolder.open(evaluation.source)
    checkpoint_path = folder.checkpoint_named_by(evaluation.source)

    # Each world's first episode is judged, so every one starts at the start.
    environment = Environment(
        replace(configuration.environment, stagger_first_episodes=False)
    )
    # Evaluation never stores or learns, so the agents' storage is one step.
    agents = _agents_for(environment, configuration, 1)
    checkpoint = RunFolder.load_checkpoint(checkpoint_path, configuration.agents.device)
    agents.load_state_dict(checkpoint["agents"])
    _set_levels(environment, checkpoint, checkpoint_path)
    loop = InteractionLoop(environment, agents, configuration.interaction_loop)
    categories = _evaluation_categories(loop, environment)

    # Each actor is made for a seed: only the random baseline uses it.
    actors: dict[str, Callable[[int], Actor]] = {"agents": lambda seed: agents}
    if "zero" in evaluation.baselines:
        actors["zero action"] = lambda seed: ZeroActionBaseline(environment.action_size)
    if "random" in evaluation.baselines:
        actors["random action"] = lambda seed: RandomActionBaseline(
            environment.device, seed, environment.action_size
        )
    print(f"Evaluating {checkpoint_path}")
    print(
        f"{environment.world_count} worlds, {len(evaluation.seeds)} seeds,"
        f" {', '.join(actors)}; each world's first episode, up to"
        f" {configuration.environment.max_episode_steps:,} steps\n"
    )
    progress = EvaluationProgress(
        len(actors) * len(evaluation.seeds),
        configuration.environment.max_episode_steps,
    )
    loop.diagnostics.progress = progress
    progress.header()
    stem = f"{datetime.now():%Y-%m-%d_%H%M%S}_{checkpoint_path.stem}"
    recorder = loop.diagnostics.recorder
    scene = (
        RecordingScene.from_run(environment)
        if evaluation.record
        else None
    )
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in actors}
    for actor_name, make_actor in actors.items():
        for seed in evaluation.seeds:
            if evaluation.record:
                recorder.arm(
                    configuration.environment.max_episode_steps,
                    environment.world_count,
                    at_next_episode=False,
                )
            progress.start(actor_name, seed)
            loop.evaluate(make_actor(seed), seed)
            record = {"seed": seed}
            record |= {
                category.name: category.plain_values() for category in categories
            }
            results[actor_name].append(record)
            progress.finish(record)
            if evaluation.record:
                folder.write_evaluation_recording(
                    f"{stem}_{actor_name.replace(' ', '_')}_seed{seed}",
                    evaluation_recording(
                        scene,
                        recorder.take(),
                        actor_name,
                        seed,
                        checkpoint_path,
                        checkpoint["completed_cycles"],
                        configuration,
                        folder,
                    ),
                )
    run_facts = {
        "checkpoint": str(checkpoint_path),
        "cycles_trained": checkpoint["completed_cycles"],
        "seeds": list(evaluation.seeds),
        "episodes_per_seed": evaluation.episodes_per_seed,
        "agents_act_with": "their policy's mean action, without exploration",
        "changed_from_training": evaluation.environment_changes,
    }
    settings = configuration.training_values()
    folder.write_evaluation(
        stem, {"run_facts": run_facts, "settings": settings, "results": results}
    )
    print(f"\nResults: {folder.evaluations / stem}.json")
    return folder.path


def _evaluation_categories(
    loop: InteractionLoop, environment: Environment
) -> list[LoggedCategory]:
    """The categories read after each seed's evaluation, in report order."""
    return [
        LoggedCategory(
            "episodes",
            "Episodes",
            "Each world's first episode only. Tables show the mean over seeds and, "
            "below it, the lowest to highest seed; charts show the mean over seeds.",
            loop.diagnostics.episode_window.descriptions,
            loop.diagnostics.episode_window.result,
        ),
        _distribution_category(
            loop, "How many of each world's first episodes fell in each bin."
        ),
        _step_category(
            loop, environment, "Means over the steps of each world's first episode."
        ),
        _timing_category(loop),
        _physics_category(loop),
    ]
