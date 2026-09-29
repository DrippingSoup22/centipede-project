"""Focused routing checks for the single-file command interface."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from centipede import cli
from centipede.experiment import evaluation, report, runner
from centipede.experiment.config import ExperimentConfig
from centipede.experiment.presets import load_training_snapshot

CPU_ROOT = Path(__file__).resolve().parents[2]


def _training_file(tmp_path: Path, *, evaluation_seeds: str = "") -> Path:
    """Keep generated test results outside the repository's runs directory."""
    contents = (CPU_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    contents = contents.replace("../../models/assembly.xml", "model.xml").replace(
        "../../runs/smoke", "runs/smoke"
    )
    if evaluation_seeds:
        contents += f"\n[evaluation]\nseeds = {evaluation_seeds}\n"
    path = tmp_path / "smoke.toml"
    path.write_text(contents, encoding="utf-8")
    return path


def _mock_evaluation(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Replace frozen evaluation with a result holding one checkpoint summary."""
    summary = SimpleNamespace(
        success_rate=0.5,
        mean_final_target_distance_m=0.01,
        mean_head_distance_traveled_m=0.002,
    )
    evaluator = Mock(
        return_value=SimpleNamespace(checkpoints=(SimpleNamespace(summary=summary),))
    )
    writer = Mock(side_effect=lambda path, config, result: path)
    monkeypatch.setattr(evaluation, "evaluate_run", evaluator)
    monkeypatch.setattr(report, "write_html_report", writer)
    return evaluator, writer


def test_training_plan_creates_a_distinct_run_with_saved_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reusable plan never sends results into an older run directory."""
    plan_path = _training_file(tmp_path)
    trainer = Mock()
    evaluator, _ = _mock_evaluation(monkeypatch)
    monkeypatch.setattr(runner, "run_training", trainer)

    assert cli.main([str(plan_path)]) == 0
    assert cli.main([str(plan_path)]) == 0

    first, second = (call.args[0] for call in trainer.call_args_list)
    assert first.run_dir != second.run_dir
    for config in (first, second):
        assert (config.run_dir / "training_preset.toml").is_file()
        assert load_training_snapshot(config.run_dir).training_seed == 2026
    evaluator.assert_not_called()


def test_training_plan_with_seeds_evaluates_the_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One plan trains and then passes its new run to frozen evaluation."""
    trainer = Mock()
    evaluator, writer = _mock_evaluation(monkeypatch)
    monkeypatch.setattr(runner, "run_training", trainer)

    assert cli.main([str(_training_file(tmp_path, evaluation_seeds="[7, 8]"))]) == 0

    training_config = trainer.call_args.args[0]
    evaluation_config = evaluator.call_args.args[0]
    assert training_config.run_dir == evaluation_config.run_dir
    assert training_config.evaluation_seeds == ()
    assert evaluation_config.evaluation_seeds == (7, 8)
    assert writer.call_args.args[0].parent.parent == (
        training_config.run_dir / "evaluations"
    )


def test_evaluation_plan_reads_saved_settings_from_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edited training plan cannot change the architecture of a saved run."""
    trainer = Mock()
    evaluator, writer = _mock_evaluation(monkeypatch)
    monkeypatch.setattr(runner, "run_training", trainer)
    training_path = _training_file(tmp_path)
    assert cli.main([str(training_path)]) == 0
    source = trainer.call_args.args[0].run_dir
    training_path.write_text(
        training_path.read_text(encoding="utf-8").replace(
            "training_seed = 2026", "training_seed = 17"
        ),
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluate.toml"
    evaluation_path.write_text(
        f'kind = "evaluation"\nsource = "{source.relative_to(tmp_path).as_posix()}"\n'
        "[evaluation]\nseeds = [30001]\n",
        encoding="utf-8",
    )

    assert cli.main([str(evaluation_path)]) == 0
    assert cli.main([str(evaluation_path)]) == 0

    assert trainer.call_count == 1
    assert evaluator.call_args.args[0].training_seed == 2026
    assert evaluator.call_args.args[0].run_dir == source
    first_report, second_report = (call.args[0] for call in writer.call_args_list)
    assert first_report != second_report
    assert (first_report.parent / "evaluation_preset.toml").is_file()


def test_comparison_trains_every_grid_combination_and_keeps_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A comparison reuses the runner and reports each variant, even failures."""

    def train(config: ExperimentConfig) -> SimpleNamespace:
        if config.environment_count == 4 and config.worker_count == 1:
            raise RuntimeError("simulated failure")
        return SimpleNamespace(
            total_environment_transitions=config.environment_count
            * config.rollout_config.rollout_window_steps
            * config.update_cycles
        )

    trainer = Mock(side_effect=train)
    evaluator, _ = _mock_evaluation(monkeypatch)
    monkeypatch.setattr(runner, "run_training", trainer)
    _training_file(tmp_path)
    plan_path = tmp_path / "compare.toml"
    plan_path.write_text(
        'kind = "comparison"\nbase = "smoke.toml"\noutput_root = "runs/comparisons"\n'
        "[grid.training]\nenvironment_count = [2, 4]\nworker_count = [1, 2]\n"
        "update_cycles = [1]\n[evaluation]\nseeds = [5]\n",
        encoding="utf-8",
    )

    assert cli.main([str(plan_path)]) == 0

    result_dir = next((tmp_path / "runs" / "comparisons").iterdir())
    results = json.loads(
        (result_dir / "comparison_results.json").read_text(encoding="utf-8")
    )["results"]
    assert trainer.call_count == 4
    assert evaluator.call_count == 3
    assert [result["variant"] for result in results] == [
        "environment_count-2_worker_count-1",
        "environment_count-2_worker_count-2",
        "environment_count-4_worker_count-1",
        "environment_count-4_worker_count-2",
    ]
    assert results[2]["error"] == "RuntimeError: simulated failure"
    assert results[0]["evaluation"]["success_rate"] == 0.5
    assert (result_dir / "comparison_report.md").is_file()
    assert (result_dir / "comparison_preset.toml").is_file()
    assert len(list(result_dir.glob("*/training_resolved.json"))) == 4


def test_package_entry_point_does_not_load_training_modules() -> None:
    """Keep ``python -m centipede`` free of main-only imports at module level."""
    command = (
        "import runpy, sys; "
        "runpy.run_module('centipede.__main__', run_name='__mp_main__'); "
        "print(','.join(sorted(name for name in sys.modules "
        "if name == 'torch' or name.startswith(('centipede.', 'rl_lib')))))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=CPU_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == ""
