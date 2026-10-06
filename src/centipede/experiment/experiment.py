"""The experiment's front file: one configuration file in, one run's results out.

``run`` reads the file and does what its mode says:

- **train** creates a new run folder, or reopens one with ``continue_from``;
  builds the environment, the agents, and the interaction loop; and after each
  cycle writes one line to the training log. Every ``checkpoint_every_cycles``
  cycles, and after the last, it saves a checkpoint and refreshes the report.
  With ``start_from`` the new run's agents begin from another run's
  checkpoint, for example to continue learning on a harder task.
- **evaluate** loads a checkpoint, runs every seed with the agents and each
  listed baseline, and writes the results and their report into the run.

This is the only component that deals with files. See docs/architecture.md and
docs/configuration.md.
"""

import platform
import subprocess
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
    for session_cycle in loop.train(seed=run_settings.seed + completed_cycles):
        cycle = completed_cycles + session_cycle + 1
        record = {"cycle": cycle, "transitions": cycle * transitions_per_cycle}
        record |= {category.name: category.plain_values() for category in categories}
        folder.append_log(record)
        print(_progress_line(record, total_cycles))
        if cycle % run_settings.checkpoint_every_cycles == 0 or cycle == total_cycles:
            folder.save_checkpoint(cycle, agents.state_dict())
            if run_settings.report:
                _write_training_report(folder, configuration, categories)
    print(f"Finished: {folder.path}")
    return folder.path


def _check_agents_can_start_from(
    checkpoint: dict[str, Any], settings: AgentSettings, path: Path
) -> None:
    """Fail clearly where a checkpoint's agents differ from the new run's.

    The networks must have the same layers. The optimizers' saved state also
    brings back their learning rate, so a different one would be ignored.
    """
    saved = checkpoint["agents"]["settings"]
    if tuple(saved["hidden_layers"]) != settings.hidden_layers:
        raise SettingsError(
            f"[agents] hidden_layers must match the checkpoint {path}: "
            f"{list(saved['hidden_layers'])}, got {list(settings.hidden_layers)}"
        )
    if saved["learning_rate"] != settings.learning_rate:
        raise SettingsError(
            f"[agents] learning_rate must match the checkpoint {path}: its "
            f"optimizers keep {saved['learning_rate']}, got {settings.learning_rate}"
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


def _progress_line(record: dict[str, Any], total_cycles: int) -> str:
    """One short terminal line per window."""
    timing, episodes = record["timing"], record["episodes"]
    line = (
        f"cycle {record['cycle']:>4}/{total_cycles}"
        f"  {timing['transitions_per_second']:>8,.0f} steps/s"
        f"  collect {timing['collecting_seconds']:6.1f} s"
        f"  learn {timing['learning_seconds']:5.1f} s"
        f"  episodes {episodes['episode_ended']:>4.0f}"
    )
    if episodes["episode_ended"]:
        returns = episodes["segment_return"]
        line += (
            f"  arrived {episodes['arrived']:4.0%}"
            f"  mean return {sum(returns) / len(returns):+.3g}"
        )
    return line


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
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in actors}
    for actor_name, make_actor in actors.items():
        for seed in evaluation.seeds:
            loop.evaluate(make_actor(seed), seed)
            record = {"seed": seed}
            record |= {
                category.name: category.plain_values() for category in categories
            }
            results[actor_name].append(record)
            print(f"  {actor_name:<14} seed {seed:<6} {_episode_line(record)}")

    stem = f"{datetime.now():%Y-%m-%d_%H%M%S}_{checkpoint_path.stem}"
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
    print(f"Results: {report_path}")
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
    ]


def _episode_line(record: dict[str, Any]) -> str:
    """One seed's episodes, in a few numbers."""
    episodes = record["episodes"]
    returns = episodes["segment_return"]
    return (
        f"arrived {episodes['arrived']:4.0%}"
        f"  mean return {sum(returns) / len(returns):+.3g}"
        f"  final distance {episodes['final_distance']:.4f} m"
    )
