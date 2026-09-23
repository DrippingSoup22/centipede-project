"""Focused routing checks for the reusable command interface."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from centipede.experiment import evaluation, report, runner
from centipede.experiment.config import ExperimentConfig
from centipede.experiment.presets import load_training_snapshot
from tools import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _training_file(tmp_path: Path) -> Path:
    """Keep generated test results outside the repository's runs directory."""
    contents = (PROJECT_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "smoke.toml"
    path.write_text(
        contents.replace("../models/assembly.xml", "model.xml").replace(
            "../runs/smoke", "runs/smoke"
        ),
        encoding="utf-8",
    )
    return path


def _benchmark_file(tmp_path: Path) -> Path:
    """Create a tiny progressive matrix around the temporary training preset."""
    _training_file(tmp_path)
    path = tmp_path / "benchmark.toml"
    path.write_text(
        """[benchmark]
training_preset = "smoke.toml"
output_root = "runs/benchmarks"
environment_counts = [2, 4]
worker_counts = [2]
steps_per_environment = [1, 2]
max_episode_steps = [1, 2]
baseline_steps_per_environment = 1
baseline_max_episode_steps = 1
rollout_windows = 1
repetitions = 1
""",
        encoding="utf-8",
    )
    return path


def test_train_only_creates_a_distinct_run_with_saved_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reusable preset never sends results into an older run directory."""
    preset_path = _training_file(tmp_path)
    trainer = Mock()
    monkeypatch.setattr(runner, "run_training", trainer)

    assert run_experiment.main(["--train", str(preset_path)]) == 0
    assert run_experiment.main(["-t", str(preset_path)]) == 0

    first, second = (call.args[0] for call in trainer.call_args_list)
    assert first.run_dir != second.run_dir
    for config in (first, second):
        assert config.run_dir.is_dir()
        assert (config.run_dir / "training_preset.toml").is_file()
        assert load_training_snapshot(config.run_dir).training_seed == 2026


def test_train_then_evaluate_uses_the_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One invocation passes its new run to frozen evaluation automatically."""
    trainer = Mock()
    evaluator = Mock(return_value=object())
    writer = Mock()
    monkeypatch.setattr(runner, "run_training", trainer)
    monkeypatch.setattr(evaluation, "evaluate_run", evaluator)
    monkeypatch.setattr(report, "write_html_report", writer)
    evaluation_path = PROJECT_ROOT / "configs" / "evaluation.toml"

    assert (
        run_experiment.main(
            ["-t", str(_training_file(tmp_path)), "-e", str(evaluation_path)]
        )
        == 0
    )

    training_config = trainer.call_args.args[0]
    evaluation_config = evaluator.call_args.args[0]
    assert training_config.run_dir == evaluation_config.run_dir
    assert training_config.evaluation_seeds == ()
    assert evaluation_config.evaluation_seeds == (30001, 30002)
    assert (
        writer.call_args.args[0].parent.parent
        == training_config.run_dir / "evaluations"
    )


def test_evaluation_only_reads_saved_settings_from_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edited preset cannot change the architecture of a saved checkpoint."""
    trainer = Mock()
    evaluator = Mock(return_value=object())
    writer = Mock()
    monkeypatch.setattr(runner, "run_training", trainer)
    monkeypatch.setattr(evaluation, "evaluate_run", evaluator)
    monkeypatch.setattr(report, "write_html_report", writer)
    preset_path = _training_file(tmp_path)
    assert run_experiment.main(["-t", str(preset_path)]) == 0
    source = trainer.call_args.args[0].run_dir
    trainer.reset_mock()
    preset_path.write_text(
        preset_path.read_text(encoding="utf-8").replace(
            "training_seed = 2026", "training_seed = 17"
        ),
        encoding="utf-8",
    )

    assert (
        run_experiment.main(
            [
                "--evaluate",
                str(PROJECT_ROOT / "configs" / "evaluation.toml"),
                "--source",
                str(source),
            ]
        )
        == 0
    )

    trainer.assert_not_called()
    assert evaluator.call_args.args[0].training_seed == 2026
    assert evaluator.call_args.args[0].run_dir == source

    first_report = writer.call_args.args[0]
    assert (first_report.parent / "evaluation_preset.toml").is_file()
    assert (
        run_experiment.main(
            ["-e", str(PROJECT_ROOT / "configs" / "evaluation.toml"), "-s", str(source)]
        )
        == 0
    )
    assert writer.call_args.args[0] != first_report


def test_benchmark_reuses_training_runner_and_saves_incremental_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI varies presets without adding another training implementation."""

    def complete_training(config: ExperimentConfig) -> SimpleNamespace:
        transitions = (
            config.environment_count
            * config.rollout_config.steps_per_environment
            * config.rollout_windows
        )
        return SimpleNamespace(total_environment_transitions=transitions)

    trainer = Mock(side_effect=complete_training)
    monkeypatch.setattr(runner, "run_training", trainer)

    assert run_experiment.main(["-b", str(_benchmark_file(tmp_path))]) == 0

    benchmark_root = tmp_path / "runs" / "benchmarks"
    result_dir = next(benchmark_root.iterdir())
    payload = json.loads(
        (result_dir / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert trainer.call_count == 5
    assert len(payload["results"]) == 5
    assert {result["phase"] for result in payload["results"]} == {
        "workers",
        "rollout",
        "episode",
    }
    assert all(result["error"] is None for result in payload["results"])
    assert (result_dir / "benchmark_report.md").is_file()
    assert len(list(result_dir.glob("*/training_resolved.json"))) == 5


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["-e", "evaluation.toml"],
        ["-s", "runs/example"],
        ["-t", "train.toml", "-s", "runs/example"],
        ["-b", "benchmark.toml", "-t", "train.toml"],
    ],
)
def test_invalid_argument_combinations_stop_before_any_work(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give clear usage errors instead of guessing which task was intended."""
    trainer = Mock()
    monkeypatch.setattr(runner, "run_training", trainer)
    with pytest.raises(SystemExit, match="2"):
        run_experiment.main(arguments)
    trainer.assert_not_called()


def test_spawn_import_does_not_load_training_or_torch_modules() -> None:
    """Keep the Windows child-process import path free of main-only modules."""
    command = (
        "import runpy, sys; "
        "runpy.run_path('tools/run_experiment.py', run_name='__mp_main__'); "
        "print(','.join(sorted(name for name in sys.modules "
        "if name == 'torch' or name.startswith(('centipede', 'rl_lib')))))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == ""
