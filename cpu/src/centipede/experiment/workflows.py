"""Carry out training, evaluation, and comparison plans read from plan files.

Each workflow creates its own result directory, keeps a copy of the plan file
that produced it, and delegates the actual work to the runner, evaluation, and
report modules.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from centipede.experiment import evaluation, report, runner
from centipede.experiment.presets import (
    ComparisonPlan,
    EvaluationPlan,
    TrainingPlan,
    TrainingPreset,
    load_training_snapshot,
    save_training_snapshot,
)


def train(plan: TrainingPlan) -> Path:
    """Train one new run and evaluate it when the plan lists held-out seeds."""
    run_dir = _new_directory(plan.preset.output_root)
    save_training_snapshot(run_dir, plan.preset)
    shutil.copyfile(plan.plan_file, run_dir / "training_preset.toml")
    print(f"Training directory: {run_dir}", flush=True)
    runner.run_training(plan.preset.for_run(run_dir))
    print(f"Training complete: {run_dir}", flush=True)
    if plan.evaluation_seeds:
        _evaluate(plan.preset, run_dir, plan.evaluation_seeds)
    return run_dir


def evaluate(plan: EvaluationPlan) -> Path:
    """Evaluate an existing run with the settings saved when it was trained."""
    preset = load_training_snapshot(plan.source_run)
    output_dir, _ = _evaluate(preset, plan.source_run, plan.evaluation_seeds)
    shutil.copyfile(plan.plan_file, output_dir / "evaluation_preset.toml")
    return output_dir


def compare(plan: ComparisonPlan) -> Path:
    """Train every variant in turn and refresh one comparison report after each.

    A failed variant is recorded and the comparison continues, so a long
    comparison keeps every completed measurement.
    """
    output_dir = _new_directory(plan.output_root)
    shutil.copyfile(plan.plan_file, output_dir / "comparison_preset.toml")
    results: list[dict[str, Any]] = []

    for number, variant in enumerate(plan.variants, start=1):
        print(f"\nVariant {number}/{len(plan.variants)}: {variant.name}", flush=True)
        run_dir = output_dir / variant.name
        run_dir.mkdir()
        save_training_snapshot(run_dir, variant.preset)
        result: dict[str, Any] = {
            "variant": variant.name,
            "changes": variant.changes,
            "training_seconds": None,
            "environment_transitions": 0,
            "transitions_per_second": None,
            "evaluation": None,
            "error": None,
        }
        started_at = perf_counter()
        try:
            training = runner.run_training(variant.preset.for_run(run_dir))
            seconds = perf_counter() - started_at
            transitions = training.total_environment_transitions
            result["training_seconds"] = seconds
            result["environment_transitions"] = transitions
            result["transitions_per_second"] = transitions / seconds
            if plan.evaluation_seeds:
                report_dir, run_evaluation = _evaluate(
                    variant.preset, run_dir, plan.evaluation_seeds
                )
                summary = run_evaluation.checkpoints[-1].summary
                result["evaluation"] = {
                    "report": str(report_dir.relative_to(output_dir)),
                    "success_rate": summary.success_rate,
                    "mean_final_target_distance_m": (
                        summary.mean_final_target_distance_m
                    ),
                    "mean_head_distance_traveled_m": (
                        summary.mean_head_distance_traveled_m
                    ),
                }
        except Exception as caught:
            result["error"] = f"{type(caught).__name__}: {caught}"
            print(f"Variant failed: {result['error']}", flush=True)
        results.append(result)
        _write_comparison_outputs(output_dir, results)

    print(f"\nComparison report: {output_dir / 'comparison_report.md'}", flush=True)
    return output_dir


def _evaluate(
    preset: TrainingPreset, run_dir: Path, seeds: tuple[int, ...]
) -> tuple[Path, evaluation.RunEvaluation]:
    """Evaluate a run into a new report directory inside it."""
    config = preset.for_run(run_dir, seeds)
    output_dir = _new_directory(run_dir / "evaluations")
    print(f"Evaluating checkpoints in: {run_dir}", flush=True)
    result = evaluation.evaluate_run(config)
    report_path = report.write_html_report(
        output_dir / "evaluation.html", config, result
    )
    print(f"Evaluation report: {report_path}", flush=True)
    return output_dir, result


def _new_directory(parent: Path) -> Path:
    """Create one unique result directory without replacing prior artifacts."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def _write_comparison_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    """Refresh machine-readable and human-readable results after each variant."""
    (output_dir / "comparison_results.json").write_text(
        json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Centipede comparison",
        "",
        "| Variant | Changes | Training s | Transitions/s | Success | "
        "Final distance m | Head path m | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        changes = ", ".join(
            f"{key}={value}" for key, value in result["changes"].items()
        )
        scores = result["evaluation"] or {}
        lines.append(
            f"| {result['variant']} | {changes} | "
            f"{_cell(result['training_seconds'], '.1f')} | "
            f"{_cell(result['transitions_per_second'], '.2f')} | "
            f"{_cell(scores.get('success_rate'), '.2f')} | "
            f"{_cell(scores.get('mean_final_target_distance_m'), '.4f')} | "
            f"{_cell(scores.get('mean_head_distance_traveled_m'), '.4f')} | "
            f"{result['error'] or 'complete'} |"
        )
    lines.extend(
        (
            "",
            "Training time covers the unchanged training call, including environment "
            "startup, collection, PPO updates, checkpoints, and shutdown. Evaluation "
            "scores describe the final checkpoint.",
            "",
        )
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def _cell(value: float | None, number_format: str) -> str:
    return "—" if value is None else format(value, number_format)
