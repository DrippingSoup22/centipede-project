"""Focused checks for the experiment plan-file boundary."""

import json
from pathlib import Path

import pytest

from centipede.experiment.presets import (
    ComparisonPlan,
    EvaluationPlan,
    TrainingPlan,
    load_plan,
    load_training_snapshot,
    save_training_snapshot,
)
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig

CPU_ROOT = Path(__file__).resolve().parents[2]


def _load_training(path: Path) -> TrainingPlan:
    plan = load_plan(path)
    assert isinstance(plan, TrainingPlan)
    return plan


def test_repository_plans_load_with_their_declared_kind() -> None:
    """Every reusable plan in configs/ is valid and routes to its operation."""
    smoke = _load_training(CPU_ROOT / "configs" / "smoke.toml")
    train = _load_training(CPU_ROOT / "configs" / "train.toml")
    evaluation = load_plan(CPU_ROOT / "configs" / "evaluation.toml")
    comparison = load_plan(CPU_ROOT / "configs" / "compare_execution.toml")

    assert smoke.preset.model_path == CPU_ROOT.parent / "models" / "assembly.xml"
    assert smoke.preset.output_root == CPU_ROOT.parent / "runs" / "smoke"
    assert smoke.preset.worker_count == 2
    assert smoke.preset.learner_config == LearnerConfig()
    assert smoke.preset.rollout_config == RolloutConfig()
    assert smoke.evaluation_seeds == ()
    assert train.evaluation_seeds == (30001, 30002)
    assert isinstance(evaluation, EvaluationPlan)
    assert evaluation.evaluation_seeds == (30001, 30002)
    assert isinstance(comparison, ComparisonPlan)
    assert len(comparison.variants) == 6
    assert comparison.evaluation_seeds == ()


def test_saved_training_settings_survive_edits_to_reusable_plan(
    tmp_path: Path,
) -> None:
    """Evaluation reads the run's resolved values, not today's edited TOML."""
    source = tmp_path / "smoke.toml"
    source.write_text(
        (CPU_ROOT / "configs" / "smoke.toml")
        .read_text(encoding="utf-8")
        .replace("../../models/assembly.xml", "model.xml")
        .replace("../../runs/smoke", "runs/smoke"),
        encoding="utf-8",
    )
    preset = _load_training(source).preset
    run_dir = tmp_path / "result"
    run_dir.mkdir()
    save_training_snapshot(run_dir, preset)
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "training_seed = 2026", "training_seed = 17"
        ),
        encoding="utf-8",
    )

    restored = load_training_snapshot(run_dir)
    assert restored == preset
    assert restored.training_seed == 2026


def test_version_one_snapshot_restores_the_original_serial_execution(
    tmp_path: Path,
) -> None:
    """Keep evaluation compatible with runs saved before worker_count existed."""
    preset = _load_training(CPU_ROOT / "configs" / "smoke.toml").preset
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    snapshot_path = save_training_snapshot(run_dir, preset)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["format_version"] = 1
    del snapshot["training"]["worker_count"]
    # Version 1 also predates the explicit rollout and update-cycle names.
    snapshot["training"]["rollout_windows"] = snapshot["training"].pop("update_cycles")
    snapshot["rollout"]["steps_per_environment"] = snapshot["rollout"].pop(
        "rollout_window_steps"
    )
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    restored = load_training_snapshot(run_dir)

    assert restored.worker_count == 1
    assert restored.environment_count == preset.environment_count
    assert restored.update_cycles == preset.update_cycles
    assert restored.rollout_config == preset.rollout_config


def test_plan_files_reject_bad_external_settings(tmp_path: Path) -> None:
    """Catch mistyped values, unknown kinds, and duplicate held-out seeds once."""
    source = (CPU_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    plan_path = tmp_path / "invalid.toml"
    for original, replacement, message in (
        ("clip_ratio = 0.2", "clip_ratio = 1.0", "clip_ratio"),
        ("worker_count = 2", "worker_count = 3", "worker_count cannot exceed"),
        ('kind = "training"', 'kind = "train"', "kind must be"),
    ):
        plan_path.write_text(source.replace(original, replacement), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_plan(plan_path)

    plan_path.write_text(source + "\n[evaluation]\nseeds = [1, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation seeds"):
        load_plan(plan_path)


def test_comparison_variants_are_checked_before_any_run(tmp_path: Path) -> None:
    """Explicit variants reuse the training schema, so typos fail up front."""
    (tmp_path / "base.toml").write_text(
        (CPU_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plan_path = tmp_path / "compare.toml"
    header = 'kind = "comparison"\nbase = "base.toml"\noutput_root = "runs"\n'

    plan_path.write_text(
        header
        + '[[variant]]\nname = "short"\nrollout.rollout_window_steps = 64\n'
        + "training.update_cycles = 4\n"
        + '[[variant]]\nname = "long"\nrollout.rollout_window_steps = 256\n',
        encoding="utf-8",
    )
    plan = load_plan(plan_path)
    assert isinstance(plan, ComparisonPlan)
    short, long = plan.variants
    assert short.changes == {
        "rollout.rollout_window_steps": 64,
        "training.update_cycles": 4,
    }
    assert short.preset.update_cycles == 4
    assert long.preset.rollout_config.rollout_window_steps == 256
    assert long.preset.output_root == (tmp_path / "runs").resolve()

    for body, message in (
        ('[[variant]]\nname = "typo"\nrollout.window_steps = 64\n', "rollout"),
        ('[[variant]]\nname = "a b"\nrollout.update_epochs = 2\n', "variant names"),
        ("[grid.rollout]\nupdate_epochs = [2, 2]\n", "distinct values"),
        ("[grid.evaluation]\nseeds = [[1]]\n", "may only change"),
    ):
        plan_path.write_text(header + body, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_plan(plan_path)
