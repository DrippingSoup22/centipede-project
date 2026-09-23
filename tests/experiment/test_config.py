"""Focused checks for the experiment configuration boundary."""

import json
from pathlib import Path

import pytest

from centipede.experiment.presets import (
    load_evaluation_preset,
    load_training_preset,
    load_training_snapshot,
    save_training_snapshot,
)
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_reusable_presets_separate_training_and_evaluation() -> None:
    """Expose every current learner and rollout choice without a CLI flag."""
    training = load_training_preset(PROJECT_ROOT / "configs" / "smoke.toml")
    evaluation = load_evaluation_preset(PROJECT_ROOT / "configs" / "evaluation.toml")

    assert training.model_path == PROJECT_ROOT / "models" / "assembly.xml"
    assert training.output_root == PROJECT_ROOT / "runs" / "smoke"
    assert training.environment_count == 2
    assert training.worker_count == 2
    assert training.learner_config == LearnerConfig()
    assert training.rollout_config == RolloutConfig()
    assert training.for_run(PROJECT_ROOT / "runs" / "example").evaluation_seeds == ()
    assert evaluation.seeds == (30001, 30002)


def test_saved_training_settings_survive_edits_to_reusable_preset(
    tmp_path: Path,
) -> None:
    """Evaluation reads the run's resolved values, not today's edited TOML."""
    source = tmp_path / "smoke.toml"
    source.write_text(
        (PROJECT_ROOT / "configs" / "smoke.toml")
        .read_text(encoding="utf-8")
        .replace("../models/assembly.xml", "model.xml")
        .replace("../runs/smoke", "runs/smoke"),
        encoding="utf-8",
    )
    preset = load_training_preset(source)
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
    preset = load_training_preset(PROJECT_ROOT / "configs" / "smoke.toml")
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    snapshot_path = save_training_snapshot(run_dir, preset)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["format_version"] = 1
    del snapshot["training"]["worker_count"]
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    restored = load_training_snapshot(run_dir)

    assert restored.worker_count == 1
    assert restored.environment_count == preset.environment_count


def test_new_preset_rejects_bad_external_settings(tmp_path: Path) -> None:
    """Catch mistyped learner values and duplicate held-out seeds once."""
    source = (PROJECT_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    training_path = tmp_path / "invalid.toml"
    training_path.write_text(
        source.replace("clip_ratio = 0.2", "clip_ratio = 1.0"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="clip_ratio"):
        load_training_preset(training_path)

    training_path.write_text(
        source.replace("worker_count = 2", "worker_count = 3"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="worker_count cannot exceed"):
        load_training_preset(training_path)

    evaluation_path = tmp_path / "evaluation.toml"
    evaluation_path.write_text("[evaluation]\nseeds = [1, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation seeds"):
        load_evaluation_preset(evaluation_path)
