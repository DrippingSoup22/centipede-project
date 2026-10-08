"""The experiment's front file: one configuration file in, one run's results out.

``run`` reads the file and does what its mode says:

- **train** creates a new run folder, or reopens one with ``continue_from``;
  builds the environment, the agents, and the interaction loop; and after each
  cycle writes one line to the training log. Every ``checkpoint_every_cycles``
  cycles, and after the last, it saves a checkpoint and refreshes the report;
  every ``record_every_episodes`` episode lengths it writes one episode length
  of windows as a replay recording.
  With ``time_limit_hours`` it stops cleanly, after a checkpoint, before a
  cycle that would end after the limit, so that a run on a machine with a
  session limit never ends in the middle of one; the run is then continued in
  a new session. With
  ``start_from`` the new run's agents begin from another run's
  checkpoint, for example to continue learning on a harder task.
- **evaluate** loads a checkpoint, runs every seed with the agents and each
  listed baseline, and writes the results, their report, and the recordings
  into the run.

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
from centipede.environment.environment import Environment
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
# poses of every world for one episode length.
RECORDING_MEMORY_LIMIT_BYTES = 1 << 30

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
    environment = Environment(configuration.environment)
    agents = Agents(
        environment.segment_count,
        environment.observation_size,
        environment.world_count,
        loop_settings.rollout_window_steps,
        configuration.agents,
        run_settings.seed,
    )
    completed_cycles = 0
    if checkpoint_path is not None:
        checkpoint = RunFolder.load_checkpoint(
            checkpoint_path, configuration.agents.device
        )
        if continue_from is not None:
            completed_cycles = checkpoint["completed_cycles"]
        else:
            _check_agents_can_start_from(
                checkpoint, configuration.agents, checkpoint_path
            )
        agents.load_state_dict(checkpoint["agents"])
    # A recording covers whole windows: the episode length, rounded up.
    recording_windows = math.ceil(
        configuration.environment.max_episode_steps / loop_settings.rollout_window_steps
    )
    recording_frames = recording_windows * loop_settings.rollout_window_steps
    if run_settings.record_every_episodes:
        _check_recording_fits(environment, recording_frames)

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
    if run_settings.record_every_episodes:
        blocks = recording_windows * run_settings.record_every_episodes
        file_count = sum(
            (cycle - 1) % blocks == 0
            for cycle in range(completed_cycles + 1, total_cycles + 1)
        )
        world_count, position_count = environment.diagnostics.simulation.qpos.shape
        kept = run_settings.record_worlds
        kept = world_count if kept == "all" else min(kept, world_count)
        print(
            recording_sizes(file_count, kept, recording_frames, position_count),
            end="\n\n",
        )
    if completed_cycles >= total_cycles:
        print(f"Already {completed_cycles} of {total_cycles} cycles; nothing to do.")
        return folder.path

    # The loop counts the cycles of this session only; the log and checkpoints
    # count from the start of the run. A resumed session draws new episodes.
    loop = InteractionLoop(
        environment,
        agents,
        replace(loop_settings, update_cycles=total_cycles - completed_cycles),
    )
    categories = _training_categories(loop, agents, environment)
    transitions_per_cycle = environment.world_count * loop_settings.rollout_window_steps

    # Recording: one episode length of windows per file. All worlds start
    # together, so a recording that starts at a multiple of the episode length
    # holds every world's whole episode, from its first step to its last,
    # unless the world arrived earlier and restarted. The recorder is armed
    # before the recording's first window and taken once it is complete, or
    # when the session ends; both happen between cycles.
    recorder = loop.diagnostics.recorder
    record_every = run_settings.record_every_episodes
    recording_first_cycle = 0

    def recording_starts(cycle: int) -> bool:
        blocks = recording_windows * record_every
        return record_every > 0 and (cycle - 1) % blocks == 0

    scene = (
        RecordingScene.from_run(configuration, environment) if record_every else None
    )
    if recording_starts(completed_cycles + 1):
        recorder.arm(recording_frames)
        recording_first_cycle = completed_cycles + 1

    # Wall-clock time per cycle in this session, everything included, for the
    # time remaining and the time limit.
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
    # cycles; the loop pauses after every cycle, before the next update.
    agents.set_learning_rate(
        configuration.agents.learning_rate_at(completed_cycles + 1, total_cycles)
    )
    cycles_start = time.monotonic()
    for session_cycle in loop.train(seed=run_settings.seed + completed_cycles):
        cycle = completed_cycles + session_cycle + 1
        record = {"cycle": cycle, "transitions": cycle * transitions_per_cycle}
        record |= {category.name: category.plain_values() for category in categories}
        folder.append_log(record)
        if cycle < total_cycles:
            agents.set_learning_rate(
                configuration.agents.learning_rate_at(cycle + 1, total_cycles)
            )
        seconds_per_cycle = (time.monotonic() - cycles_start) / (session_cycle + 1)
        remaining_s = (total_cycles - cycle) * seconds_per_cycle
        progress.finish(record, remaining_s)
        out_of_time = (
            time_limit_s is not None
            and cycle < total_cycles
            and time.monotonic() - session_start + seconds_per_cycle > time_limit_s
        )
        if recorder.recording and (
            recorder.complete or cycle == total_cycles or out_of_time
        ):
            window = recorder.take(
                run_settings.record_levels,
                run_settings.record_worlds,
                run_settings.record_selection,
            )
            folder.write_recording(
                f"cycles_{recording_first_cycle:04d}-{cycle:04d}",
                training_recording(
                    scene,
                    window,
                    recording_first_cycle,
                    total_cycles,
                    configuration,
                    folder,
                ),
            )
        if cycle < total_cycles and recording_starts(cycle + 1):
            recorder.arm(recording_frames)
            recording_first_cycle = cycle + 1
        if (
            cycle % run_settings.checkpoint_every_cycles == 0
            or cycle == total_cycles
            or out_of_time
        ):
            folder.save_checkpoint(cycle, agents.state_dict())
            if run_settings.report:
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


def _check_recording_fits(environment: Environment, frames: int) -> None:
    """Fail clearly where recording one episode length would take too much
    device memory: the poses, targets, and episode starts of every world."""
    world_count, position_count = environment.diagnostics.simulation.qpos.shape
    needed = frames * world_count * (4 * (position_count + 2) + 1)
    if needed > RECORDING_MEMORY_LIMIT_BYTES:
        raise SettingsError(
            f"Recording one episode length ({frames:,} steps of {world_count:,}"
            f" worlds) needs {needed / 2**30:.1f} GiB of device memory, more than"
            f" {RECORDING_MEMORY_LIMIT_BYTES / 2**30:.0f} GiB: shorten the episodes,"
            " use fewer worlds, or set [run] record_every_episodes = 0"
        )


def _check_agents_can_start_from(
    checkpoint: dict[str, Any], settings: AgentSettings, path: Path
) -> None:
    """Fail clearly where a checkpoint's agents differ from the new run's.

    The networks must have the same layers, and the optimizers must be of the
    same kind, since each kind keeps its own state. Checkpoints saved before
    the optimizer could be chosen used Adam.
    """
    saved = checkpoint["agents"]["settings"]
    if tuple(saved["hidden_layers"]) != settings.hidden_layers:
        raise SettingsError(
            f"[agents] hidden_layers must match the checkpoint {path}: "
            f"{list(saved['hidden_layers'])}, got {list(settings.hidden_layers)}"
        )
    saved_optimizer = saved.get("optimizer", "adam")
    if saved_optimizer != settings.optimizer:
        raise SettingsError(
            f"[agents] optimizer must match the checkpoint {path}: "
            f"{saved_optimizer!r}, got {settings.optimizer!r}"
        )


def _training_categories(
    loop: InteractionLoop, agents: Agents, environment: Environment
) -> list[LoggedCategory]:
    """Every diagnostics category logged once per window, in report order."""
    loop_diagnostics = loop.diagnostics
    return [
        LoggedCategory(
            "episodes",
            "Episodes",
            "Means over the episodes that ended in each window; 'episode ended' "
            "counts them. Windows in which no episode ended have no values.",
            loop_diagnostics.episode_window.descriptions,
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

    environment = Environment(configuration.environment)
    # Evaluation never stores or learns, so the agents' storage is one step.
    agents = Agents(
        environment.segment_count,
        environment.observation_size,
        environment.world_count,
        1,
        configuration.agents,
        configuration.run.seed,
    )
    checkpoint = RunFolder.load_checkpoint(checkpoint_path, configuration.agents.device)
    agents.load_state_dict(checkpoint["agents"])
    loop = InteractionLoop(environment, agents, configuration.interaction_loop)
    categories = _evaluation_categories(loop, environment)

    # Each actor is made for a seed: only the random baseline uses it.
    actors: dict[str, Callable[[int], Actor]] = {"agents": lambda seed: agents}
    if "zero" in evaluation.baselines:
        actors["zero action"] = lambda seed: ZeroActionBaseline()
    if "random" in evaluation.baselines:
        actors["random action"] = lambda seed: RandomActionBaseline(
            environment.device, seed
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
        RecordingScene.from_run(configuration, environment)
        if evaluation.record
        else None
    )
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in actors}
    for actor_name, make_actor in actors.items():
        for seed in evaluation.seeds:
            if evaluation.record:
                recorder.arm(configuration.environment.max_episode_steps)
            progress.start(actor_name, seed)
            loop.evaluate(make_actor(seed), seed)
            record = {"seed": seed}
            record |= {
                category.name: category.plain_values() for category in categories
            }
            results[actor_name].append(record)
            progress.finish(record)
            if evaluation.record:
                window = recorder.take(1, "all", "ranked")
                folder.write_evaluation_recording(
                    f"{stem}_{actor_name.replace(' ', '_')}_seed{seed}",
                    evaluation_recording(
                        scene,
                        window,
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
