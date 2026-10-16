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
  a new session. A cooperators' run may keep worlds that act with the mean
  action, and tow its body; the tow's top moves after each cycle and is
  saved with the checkpoints. With
  ``start_from`` the new run's agents begin from another run's
  checkpoint, for example to continue learning on a harder task.
- **evaluate** loads a checkpoint, runs every seed with the agents and each
  listed baseline, and writes the results, their report, and the recordings
  into the run.

A run of the cooperators' design writes no training report, since the
training project's monitor shows its log while it trains; its evaluation
page has the design's own layout.

This is the only component that deals with files. See docs/architecture.md and
docs/configuration.md.
"""

import math
import platform
import subprocess
import time
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
from centipede.environment.environment import (
    CooperativeEnvironment,
    Environment,
    create_environment,
)
from centipede.experiment.configuration import (
    Configuration,
    EvaluationSettings,
    dotted_keys,
    read_configuration,
)
from centipede.experiment.progress import (
    CooperativeEvaluationProgress,
    CooperativeTrainingProgress,
    EvaluationProgress,
    TrainingProgress,
    duration,
    recording_sizes,
    reward_weights,
    training_settings,
)
from centipede.experiment.recordings import (
    RecordingScene,
    evaluation_recording,
    training_recording,
)
from centipede.experiment.report import (
    LoggedCategory,
    write_evaluation_report,
    write_training_report,
)
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
    environment = create_environment(configuration.environment)
    worlds_per_group = run_settings.mean_action_worlds_per_group
    agents = _agents_for(
        environment,
        configuration,
        loop_settings.rollout_window_steps,
        mean_action_world_count=2 * worlds_per_group,
    )
    # The cooperators' task has a level, which follows the arrivals, and may
    # have a tow, which follows the mean action.
    levelled = isinstance(environment, CooperativeEnvironment)
    towed = levelled and configuration.environment.cooperators.tow_start > 0
    saved_tow_top = None
    completed_cycles = 0
    if checkpoint_path is not None:
        checkpoint = RunFolder.load_checkpoint(
            checkpoint_path, configuration.agents.device
        )
        if continue_from is not None:
            completed_cycles = checkpoint["completed_cycles"]
            if levelled and "task_level" in checkpoint:
                environment.set_level(checkpoint["task_level"])
            saved_tow_top = checkpoint.get("tow_top")
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
    if not levelled:
        weights = configuration.environment.rewards.weights(
            configuration.environment.max_episode_steps
        )
        print(
            reward_weights(
                weights.per_step,
                weights.episode_shares,
                configuration.agents.ppo.discount,
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
        RecordingScene.from_run(configuration, environment) if record_worlds else None
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
    progress_type = CooperativeTrainingProgress if levelled else TrainingProgress
    progress = progress_type(
        completed_cycles + 1,
        total_cycles,
        loop_settings.rollout_window_steps,
        configuration.environment.max_episode_steps,
    )
    loop.diagnostics.progress = progress
    progress.header()
    # The learning rate of each update follows the schedule over the run's
    # cycles (an adaptive one is each agent's own), and so do scheduled action
    # spreads; the loop pauses after every cycle, before the next window.
    agent_settings = configuration.agents

    def follow_schedules(cycle: int) -> None:
        learning_rate = agent_settings.learning_rate_at(cycle, total_cycles)
        if learning_rate is not None:
            agents.set_learning_rate(learning_rate)
        action_std = agent_settings.action_std_at(cycle, total_cycles)
        if action_std is not None:
            agents.set_action_std(action_std)

    follow_schedules(completed_cycles + 1)
    cycles_start = time.monotonic()
    for session_cycle in loop.train(seed=run_settings.seed + completed_cycles):
        cycle = completed_cycles + session_cycle + 1
        if levelled:
            environment.adjust_level()
        if worlds_per_group:
            environment.adjust_tow()
        record = {"cycle": cycle, "transitions": cycle * transitions_per_cycle}
        record |= {category.name: category.plain_values() for category in categories}
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
                task_level=environment.level if levelled else None,
                tow_top=environment.tow.top if towed else None,
            )
            if run_settings.report and not levelled:
                _write_training_report(folder, configuration, categories)
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
    environment: Environment | CooperativeEnvironment,
    configuration: Configuration,
    rollout_window_steps: int,
    mean_action_world_count: int = 0,
) -> Agents:
    """Agents that fit the environment: its body, its reward channels and
    those each segment learns from, and the segments each segment sees; in
    training, the last ``mean_action_world_count`` worlds act with the mean
    action and are not learned from."""
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
        environment.neighbour_table,
        environment.segment_reward_channels,
        mean_action_world_count,
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


def _check_recording_fits(
    environment: Environment | CooperativeEnvironment, frames: int, worlds: int
) -> None:
    """Fail clearly where recording one episode would take too much device
    memory: the poses, targets, and episode starts of the recorded worlds."""
    position_count = environment.diagnostics.simulation.qpos.shape[1]
    needed = frames * worlds * (4 * (position_count + 2) + 1)
    if needed > RECORDING_MEMORY_LIMIT_BYTES:
        raise SettingsError(
            f"Recording one episode ({frames:,} steps of {worlds:,} worlds) needs"
            f" {needed / 2**30:.1f} GiB of device memory, more than"
            f" {RECORDING_MEMORY_LIMIT_BYTES / 2**30:.0f} GiB: record fewer worlds"
            " with [run] record_worlds"
        )


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
            _episode_descriptions(loop, environment),
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
        *_task_category(environment),
        *(_tow_category(environment) if measuring else []),
    ]


def _tow_category(environment: CooperativeEnvironment) -> list[LoggedCategory]:
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


def _task_category(
    environment: Environment | CooperativeEnvironment,
) -> list[LoggedCategory]:
    """The task's level after each cycle, for an environment that has one."""
    if not isinstance(environment, CooperativeEnvironment):
        return []
    return [
        LoggedCategory(
            "task",
            "Task",
            "After each update cycle; the level moves toward the difficulty"
            " where one target in ten is reached within its stopwatch.",
            descriptions(environment.task_facts),
            lambda: values(environment.task_facts),
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


def _episode_descriptions(
    loop: InteractionLoop, environment: Environment | CooperativeEnvironment
) -> dict[str, Any]:
    """The episode values' descriptions, with the cooperators' reward
    channels named."""
    descriptions = dict(loop.diagnostics.episode_window.descriptions)
    if isinstance(environment, CooperativeEnvironment):
        descriptions["segment_return"] = replace(
            descriptions["segment_return"],
            parts=tuple(environment.diagnostics.channel_names),
        )
    return descriptions


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


def _write_training_report(
    folder: RunFolder, configuration: Configuration, categories: list[LoggedCategory]
) -> None:
    """Refresh the run's report from its whole log."""
    sessions = folder.read_run_info()["sessions"]
    run_facts = {
        "run_folder": str(folder.path),
        "started": sessions[0]["started"],
        "training_sessions": len(sessions),
        "code_version": sessions[-1]["git_commit"] or "unknown",
        "device": sessions[-1]["device"],
    }
    if configuration.run.start_from is not None:
        run_facts["agents_started_from"] = sessions[0]["checkpoint"]
    write_training_report(
        folder.report_path,
        folder.path.name,
        run_facts,
        categories,
        folder.read_log(),
        configuration.training_values(),
        sessions,
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

    environment = create_environment(configuration.environment)
    # Evaluation never stores or learns, so the agents' storage is one step.
    agents = _agents_for(environment, configuration, 1)
    checkpoint = RunFolder.load_checkpoint(checkpoint_path, configuration.agents.device)
    agents.load_state_dict(checkpoint["agents"])
    # The task's targets are those of the level the run reached.
    levelled = isinstance(environment, CooperativeEnvironment)
    if levelled and "task_level" in checkpoint:
        environment.set_level(checkpoint["task_level"])
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
    progress_type = CooperativeEvaluationProgress if levelled else EvaluationProgress
    progress = progress_type(
        len(actors) * len(evaluation.seeds),
        configuration.environment.max_episode_steps,
    )
    loop.diagnostics.progress = progress
    progress.header()
    stem = f"{datetime.now():%Y-%m-%d_%H%M%S}_{checkpoint_path.stem}"
    recorder = loop.diagnostics.recorder
    scene = (
        RecordingScene.from_run(configuration, environment)
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
    if levelled:
        run_facts["task_level"] = environment.level
    settings = configuration.training_values()
    folder.write_evaluation(
        stem, {"run_facts": run_facts, "settings": settings, "results": results}
    )
    report_path = folder.evaluations / f"{stem}.html"
    write_evaluation_report(
        report_path, folder.path.name, run_facts, categories, results, settings
    )
    print(f"\nResults: {report_path}")
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
            _episode_descriptions(loop, environment),
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
