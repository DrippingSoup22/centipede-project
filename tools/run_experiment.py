"""Run training, evaluation, or both from reusable TOML presets."""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from centipede.experiment.presets import EvaluationPreset, TrainingPreset


def main(argv: Sequence[str] | None = None) -> int:
    """Select the requested workflow; all experiment choices remain in TOML."""
    parser = argparse.ArgumentParser(prog="run.cmd", description=__doc__)
    parser.add_argument("-t", "--train", type=Path, help="Training TOML preset")
    parser.add_argument("-e", "--evaluate", type=Path, help="Evaluation TOML preset")
    parser.add_argument("-b", "--benchmark", type=Path, help="Benchmark TOML preset")
    parser.add_argument(
        "-s", "--source", type=Path, help="Existing result directory to evaluate"
    )
    args = parser.parse_args(argv)

    if args.benchmark is not None:
        if any(value is not None for value in (args.train, args.evaluate, args.source)):
            parser.error("--benchmark cannot be combined with other operations")
        _run_benchmark(args.benchmark)
        return 0

    if args.train is None and args.evaluate is None:
        parser.error("provide --train, --evaluate, or --benchmark")
    if args.source is not None and args.train is not None:
        parser.error("--source is only used for evaluation without training")
    if args.evaluate is None and args.source is not None:
        parser.error("--source requires --evaluate")
    if args.train is None and args.source is None:
        parser.error("evaluation without training requires --source")

    # Project imports remain below argument parsing and inside main(). Spawned
    # Windows workers re-import this script as ``__mp_main__`` but do not call
    # main(), so they avoid loading Torch, learners, or experiment application
    # modules before entering the environment-worker target.
    from centipede.experiment.presets import (
        load_evaluation_preset,
        load_training_preset,
        load_training_snapshot,
        save_training_snapshot,
    )

    # Parse both inputs before any training work, including output creation.
    evaluation = (
        load_evaluation_preset(args.evaluate) if args.evaluate is not None else None
    )

    if args.train is not None:
        from centipede.experiment.runner import run_training

        preset = load_training_preset(args.train)
        run_dir = _new_directory(preset.output_root)
        save_training_snapshot(run_dir, preset)
        shutil.copyfile(args.train, run_dir / "training_preset.toml")
        config = preset.for_run(run_dir)
        print(f"Training directory: {run_dir}", flush=True)
        run_training(config)
        print(f"Training complete: {run_dir}", flush=True)
    else:
        run_dir = args.source.resolve()
        preset = load_training_snapshot(run_dir)

    if evaluation is not None:
        _evaluate(
            preset,
            run_dir,
            evaluation,
            args.evaluate,
        )
    return 0


def _evaluate(
    preset: TrainingPreset,
    run_dir: Path,
    evaluation: EvaluationPreset,
    evaluation_path: Path,
) -> None:
    """Restore the run configuration and keep each evaluation report distinct."""
    from centipede.experiment.evaluation import evaluate_run
    from centipede.experiment.report import write_html_report

    config = preset.for_run(run_dir, evaluation.seeds)
    output_dir = _new_directory(run_dir / "evaluations")
    shutil.copyfile(evaluation_path, output_dir / "evaluation_preset.toml")
    report_path = output_dir / "evaluation.html"
    print(f"Evaluating checkpoints in: {run_dir}", flush=True)
    result = evaluate_run(config)
    write_html_report(report_path, config, result)
    print(f"Evaluation report: {report_path}", flush=True)


def _new_directory(parent: Path) -> Path:
    """Create one unique result directory without replacing prior artifacts."""
    parent.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path = parent / name
    path.mkdir(exist_ok=False)
    return path


def _run_benchmark(path: Path) -> None:
    """Run a progressive timing matrix through the existing training runner."""

    from centipede.experiment.presets import (
        load_training_preset,
        save_training_snapshot,
    )
    from centipede.experiment.runner import run_training

    benchmark_path = path.resolve()
    values = _load_benchmark_values(benchmark_path)
    base_preset = load_training_preset(values["training_preset"])
    output_dir = _new_directory(values["output_root"])
    shutil.copyfile(benchmark_path, output_dir / "benchmark_preset.toml")

    results: list[dict[str, Any]] = []
    selected_workers: dict[int, int] = {}
    case_number = 0
    total_cases = _benchmark_case_count(values)

    def run_case(
        phase: str,
        environments: int,
        workers: int,
        steps: int,
        episode_steps: int,
        repetition: int,
    ) -> None:
        nonlocal case_number
        case_number += 1
        label = (
            f"{phase}-e{environments}-w{workers}-s{steps}-h{episode_steps}"
            f"-r{repetition}"
        )
        run_dir = output_dir / label
        rollout_config = replace(
            base_preset.rollout_config,
            steps_per_environment=steps,
        )
        preset = replace(
            base_preset,
            output_root=output_dir,
            environment_count=environments,
            worker_count=workers,
            rollout_windows=values["rollout_windows"],
            max_episode_steps=episode_steps,
            rollout_config=rollout_config,
        )
        config = preset.for_run(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        save_training_snapshot(run_dir, preset)
        print(
            f"\nBenchmark {case_number}/{total_cases}: {label}",
            flush=True,
        )

        started_at = perf_counter()
        error: str | None = None
        training_result = None
        try:
            training_result = run_training(config)
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
            print(f"Benchmark case failed: {error}", flush=True)
        wall_seconds = perf_counter() - started_at

        transitions = (
            training_result.total_environment_transitions
            if training_result is not None
            else 0
        )
        result = {
            "phase": phase,
            "environment_count": environments,
            "worker_count": workers,
            "steps_per_environment": steps,
            "max_episode_steps": episode_steps,
            "rollout_windows": values["rollout_windows"],
            "repetition": repetition,
            "wall_seconds": wall_seconds,
            "environment_transitions": transitions,
            "seconds_per_window": wall_seconds / values["rollout_windows"],
            "transitions_per_second": transitions / wall_seconds if transitions else 0,
            "error": error,
        }
        results.append(result)
        _write_benchmark_outputs(output_dir, values, results, selected_workers)

    baseline_steps = values["baseline_steps_per_environment"]
    baseline_episode_steps = values["baseline_max_episode_steps"]

    # First determine how many workers scale best for every environment count.
    for environments in values["environment_counts"]:
        for workers in values["worker_counts"]:
            if workers > environments:
                continue
            for repetition in range(1, values["repetitions"] + 1):
                run_case(
                    "workers",
                    environments,
                    workers,
                    baseline_steps,
                    baseline_episode_steps,
                    repetition,
                )

        candidates = [
            result
            for result in results
            if result["phase"] == "workers"
            and result["environment_count"] == environments
            and result["error"] is None
        ]
        if not candidates:
            raise RuntimeError(
                f"No successful worker benchmark for {environments} environments"
            )
        selected_workers[environments] = min(
            {result["worker_count"] for result in candidates},
            key=lambda worker: fmean(
                result["wall_seconds"]
                for result in candidates
                if result["worker_count"] == worker
            ),
        )
        _write_benchmark_outputs(output_dir, values, results, selected_workers)

    # Use each environment count's winner when comparing rollout window sizes.
    for environments in values["environment_counts"]:
        workers = selected_workers[environments]
        for steps in values["steps_per_environment"]:
            if steps == baseline_steps:
                continue
            for repetition in range(1, values["repetitions"] + 1):
                run_case(
                    "rollout",
                    environments,
                    workers,
                    steps,
                    baseline_episode_steps,
                    repetition,
                )

    # Episode limits are orthogonal to scaling, so test them only on the largest
    # environment group using its selected worker count and baseline window.
    largest_environment_count = max(values["environment_counts"])
    workers = selected_workers[largest_environment_count]
    for episode_steps in values["max_episode_steps"]:
        if episode_steps == baseline_episode_steps:
            continue
        for repetition in range(1, values["repetitions"] + 1):
            run_case(
                "episode",
                largest_environment_count,
                workers,
                baseline_steps,
                episode_steps,
                repetition,
            )

    _write_benchmark_outputs(output_dir, values, results, selected_workers)
    print(f"\nBenchmark report: {output_dir / 'benchmark_report.md'}", flush=True)


def _load_benchmark_values(path: Path) -> dict[str, Any]:
    """Validate the small CLI-owned benchmark matrix at its input boundary."""

    with path.open("rb") as file:
        contents = tomllib.load(file)
    if set(contents) != {"benchmark"} or not isinstance(contents["benchmark"], dict):
        raise ValueError("benchmark configuration must contain one [benchmark] table")

    values = contents["benchmark"]
    expected = {
        "training_preset",
        "output_root",
        "environment_counts",
        "worker_counts",
        "steps_per_environment",
        "max_episode_steps",
        "baseline_steps_per_environment",
        "baseline_max_episode_steps",
        "rollout_windows",
        "repetitions",
    }
    if set(values) != expected:
        raise ValueError("benchmark has missing or unexpected fields")

    base = path.parent
    parsed = {
        **values,
        "training_preset": _benchmark_path(
            values["training_preset"], "training_preset", base
        ),
        "output_root": _benchmark_path(values["output_root"], "output_root", base),
    }
    for name in (
        "environment_counts",
        "worker_counts",
        "steps_per_environment",
        "max_episode_steps",
    ):
        parsed[name] = _positive_integer_list(values[name], name)
    for name in (
        "baseline_steps_per_environment",
        "baseline_max_episode_steps",
        "rollout_windows",
        "repetitions",
    ):
        parsed[name] = _positive_integer(values[name], name)

    if parsed["baseline_steps_per_environment"] not in parsed["steps_per_environment"]:
        raise ValueError("baseline rollout steps must appear in steps_per_environment")
    if parsed["baseline_max_episode_steps"] not in parsed["max_episode_steps"]:
        raise ValueError("baseline episode steps must appear in max_episode_steps")
    return parsed


def _benchmark_case_count(values: dict[str, Any]) -> int:
    """Return the number of progressive cases before the benchmark starts."""

    worker_cases = sum(
        worker <= environments
        for environments in values["environment_counts"]
        for worker in values["worker_counts"]
    )
    rollout_cases = len(values["environment_counts"]) * (
        len(values["steps_per_environment"]) - 1
    )
    episode_cases = len(values["max_episode_steps"]) - 1
    return (worker_cases + rollout_cases + episode_cases) * values["repetitions"]


def _write_benchmark_outputs(
    output_dir: Path,
    values: dict[str, Any],
    results: list[dict[str, Any]],
    selected_workers: dict[int, int],
) -> None:
    """Refresh machine-readable and human-readable progress after each case."""

    serializable_values = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in values.items()
    }
    payload = {
        "benchmark": serializable_values,
        "selected_workers": selected_workers,
        "results": results,
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_report.md").write_text(
        _benchmark_markdown(payload),
        encoding="utf-8",
    )


def _benchmark_markdown(payload: dict[str, Any]) -> str:
    """Render a compact report that remains readable without another tool."""

    selected = (
        ", ".join(
            f"{environments} envs = {workers} worker(s)"
            for environments, workers in sorted(payload["selected_workers"].items())
        )
        or "Pending worker measurements"
    )
    lines = [
        "# Centipede CPU benchmark",
        "",
        f"**Selected workers:** {selected}",
        "",
        "| Phase | Envs | Workers | Steps | Episode | Rep | Wall s | s/window | "
        "Transitions/s | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in payload["results"]:
        status = result["error"] or "complete"
        lines.append(
            f"| {result['phase']} | {result['environment_count']} | "
            f"{result['worker_count']} | {result['steps_per_environment']} | "
            f"{result['max_episode_steps']} | {result['repetition']} | "
            f"{result['wall_seconds']:.3f} | "
            f"{result['seconds_per_window']:.3f} | "
            f"{result['transitions_per_second']:.2f} | {status} |"
        )
    lines.extend(
        (
            "",
            "Wall time covers the unchanged training call, including environment "
            "startup, collection, PPO updates, checkpoints, and shutdown.",
            "",
        )
    )
    return "\n".join(lines)


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_integer_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    parsed = [_positive_integer(item, name) for item in value]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must contain distinct values")
    return parsed


def _benchmark_path(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return (base / value).resolve()


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
