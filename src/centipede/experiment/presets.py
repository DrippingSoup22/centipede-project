"""Read reusable experiment presets and preserve each run's resolved settings."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from centipede.experiment.config import ExperimentConfig
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig

SNAPSHOT_NAME = "training_resolved.json"
CURRENT_SNAPSHOT_VERSION = 2


@dataclass(frozen=True)
class TrainingPreset:
    """Reusable settings; the output root receives a new directory per run."""

    model_path: Path
    output_root: Path
    training_seed: int
    environment_count: int
    worker_count: int
    rollout_windows: int
    max_episode_steps: int
    learner_config: LearnerConfig
    rollout_config: RolloutConfig

    def for_run(
        self, run_dir: Path, evaluation_seeds: tuple[int, ...] = ()
    ) -> ExperimentConfig:
        """Supply one run location to the existing training/evaluation components."""
        return ExperimentConfig(
            model_path=self.model_path,
            run_dir=run_dir,
            training_seed=self.training_seed,
            environment_count=self.environment_count,
            worker_count=self.worker_count,
            rollout_windows=self.rollout_windows,
            max_episode_steps=self.max_episode_steps,
            evaluation_seeds=evaluation_seeds,
            learner_config=self.learner_config,
            rollout_config=self.rollout_config,
        )


@dataclass(frozen=True)
class EvaluationPreset:
    """Held-out episodes shared by all checkpoints and reference policies."""

    seeds: tuple[int, ...]


def load_training_preset(path: str | Path) -> TrainingPreset:
    """Validate a human-edited TOML training preset at the input boundary."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as file:
        contents = tomllib.load(file)
    return _parse_training(contents, config_path.parent)


def load_evaluation_preset(path: str | Path) -> EvaluationPreset:
    """Validate the small, independent evaluation TOML."""
    with Path(path).open("rb") as file:
        contents = tomllib.load(file)
    _exact_keys(contents, {"evaluation"}, "configuration")
    values = _table(contents, "evaluation")
    _exact_keys(values, {"seeds"}, "evaluation")
    seeds = values["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("evaluation seeds must be distinct nonnegative integers")
    return EvaluationPreset(seeds=tuple(seeds))


def save_training_snapshot(run_dir: Path, preset: TrainingPreset) -> Path:
    """Freeze resolved paths and hyperparameters before training starts."""
    payload = {
        "format_version": CURRENT_SNAPSHOT_VERSION,
        "training": {
            "model_path": str(preset.model_path),
            "output_root": str(preset.output_root),
            "training_seed": preset.training_seed,
            "environment_count": preset.environment_count,
            "worker_count": preset.worker_count,
            "rollout_windows": preset.rollout_windows,
            "max_episode_steps": preset.max_episode_steps,
        },
        "learner": asdict(preset.learner_config),
        "rollout": asdict(preset.rollout_config),
    }
    path = run_dir / SNAPSHOT_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_training_snapshot(run_dir: str | Path) -> TrainingPreset:
    """Restore the original training settings when evaluating an older run."""
    source = Path(run_dir).resolve()
    snapshot = source / SNAPSHOT_NAME
    if not snapshot.is_file():
        raise FileNotFoundError(f"No training snapshot in {source}")
    with snapshot.open(encoding="utf-8") as file:
        contents = json.load(file)
    _exact_keys(
        contents, {"format_version", "training", "learner", "rollout"}, "snapshot"
    )
    format_version = contents["format_version"]
    if format_version not in (1, CURRENT_SNAPSHOT_VERSION):
        raise ValueError("Unsupported training snapshot version")
    return _parse_training(
        {name: contents[name] for name in ("training", "learner", "rollout")},
        source,
        legacy_snapshot=format_version == 1,
    )


def _parse_training(
    contents: dict[str, Any],
    base: Path,
    *,
    legacy_snapshot: bool = False,
) -> TrainingPreset:
    """Apply the same schema to a preset and its saved resolved snapshot."""
    _exact_keys(contents, {"training", "learner", "rollout"}, "configuration")
    training = _table(contents, "training")
    learner = _table(contents, "learner")
    rollout = _table(contents, "rollout")
    training_fields = {
        "model_path",
        "output_root",
        "training_seed",
        "environment_count",
        "rollout_windows",
        "max_episode_steps",
    }
    if not legacy_snapshot:
        training_fields.add("worker_count")
    _exact_keys(training, training_fields, "training")
    _exact_keys(learner, {field.name for field in fields(LearnerConfig)}, "learner")
    _exact_keys(rollout, {field.name for field in fields(RolloutConfig)}, "rollout")

    sizes = learner["hidden_sizes"]
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("hidden_sizes must be a nonempty list")
    learner["hidden_sizes"] = tuple(
        _integer(size, "hidden_sizes", minimum=1) for size in sizes
    )
    for name in (
        "initial_standard_deviation",
        "actor_learning_rate",
        "critic_learning_rate",
        "maximum_gradient_norm",
        "normalizer_epsilon",
        "normalizer_clip",
    ):
        learner[name] = _number(learner[name], name, minimum=0, strict=True)
    learner["entropy_coefficient"] = _number(
        learner["entropy_coefficient"], "entropy_coefficient", minimum=0
    )
    learner["clip_ratio"] = _number(
        learner["clip_ratio"], "clip_ratio", minimum=0, maximum=1, strict=True
    )
    _choice(
        learner["standard_deviation_mode"],
        "standard_deviation_mode",
        {"global", "state_dependent"},
    )
    _choice(learner["normalization_mode"], "normalization_mode", {"none", "running"})

    for name in ("steps_per_environment", "update_epochs", "minibatch_size"):
        rollout[name] = _integer(rollout[name], name, minimum=1)
    rollout["discount"] = _number(rollout["discount"], "discount", minimum=0, maximum=1)
    rollout["gae_lambda"] = _number(
        rollout["gae_lambda"], "gae_lambda", minimum=0, maximum=1
    )
    environment_count = _integer(training["environment_count"], "environment_count", 1)
    # Version-1 snapshots predate multiprocessing and describe the serial path,
    # represented by one execution worker in the current configuration.
    worker_count = (
        1 if legacy_snapshot else _integer(training["worker_count"], "worker_count", 1)
    )
    if worker_count > environment_count:
        raise ValueError("worker_count cannot exceed environment_count")

    return TrainingPreset(
        model_path=_path(training["model_path"], "model_path", base),
        output_root=_path(training["output_root"], "output_root", base),
        training_seed=_integer(training["training_seed"], "training_seed", 0),
        environment_count=environment_count,
        worker_count=worker_count,
        rollout_windows=_integer(training["rollout_windows"], "rollout_windows", 1),
        max_episode_steps=_integer(
            training["max_episode_steps"], "max_episode_steps", 1
        ),
        learner_config=LearnerConfig(**learner),
        rollout_config=RolloutConfig(**rollout),
    )


def _table(contents: dict[str, Any], name: str) -> dict[str, Any]:
    value = contents[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    return value


def _exact_keys(values: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(
    value: Any,
    name: str,
    minimum: float,
    maximum: float | None = None,
    *,
    strict: bool = False,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (value <= minimum if strict else value < minimum) or (
        maximum is not None and (value >= maximum if strict else value > maximum)
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def _choice(value: Any, name: str, options: set[str]) -> None:
    if type(value) is not str or value not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}")


def _path(value: Any, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return (base / value).resolve()
