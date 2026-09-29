"""Read experiment plan files and preserve each run's resolved settings.

Every run starts from one TOML plan file whose top-level ``kind`` selects the
operation: ``training``, ``evaluation``, or ``comparison``. This module is the
only place that checks those files; later components trust the returned plans.
"""

import copy
import itertools
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from centipede.experiment.config import ExperimentConfig
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig

SNAPSHOT_NAME = "training_resolved.json"
CURRENT_SNAPSHOT_VERSION = 3

# Sections of a training plan; a comparison may change fields inside them.
_TRAINING_SECTIONS = ("training", "learner", "rollout")
_VARIANT_NAME = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class TrainingPreset:
    """Reusable settings; the output root receives a new directory per run."""

    model_path: Path
    output_root: Path
    training_seed: int
    environment_count: int
    worker_count: int
    update_cycles: int
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
            update_cycles=self.update_cycles,
            max_episode_steps=self.max_episode_steps,
            evaluation_seeds=evaluation_seeds,
            learner_config=self.learner_config,
            rollout_config=self.rollout_config,
        )


@dataclass(frozen=True)
class TrainingPlan:
    """Train one new run, then evaluate it when held-out seeds are given."""

    plan_file: Path
    preset: TrainingPreset
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    """Evaluate the checkpoints of an existing run on held-out seeds."""

    plan_file: Path
    source_run: Path
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ComparisonVariant:
    """One training preset derived from the comparison base."""

    name: str
    changes: dict[str, Any]
    preset: TrainingPreset


@dataclass(frozen=True)
class ComparisonPlan:
    """Train, and optionally evaluate, every variant of one base training plan."""

    plan_file: Path
    output_root: Path
    variants: tuple[ComparisonVariant, ...]
    evaluation_seeds: tuple[int, ...]


ExperimentPlan = TrainingPlan | EvaluationPlan | ComparisonPlan


def load_plan(path: str | Path) -> ExperimentPlan:
    """Read one plan file and check every setting before any work starts."""
    plan_file = Path(path).resolve()
    contents = _read_toml(plan_file)
    kind = contents.get("kind")
    if kind == "training":
        return _training_plan(contents, plan_file)
    if kind == "evaluation":
        return _evaluation_plan(contents, plan_file)
    if kind == "comparison":
        return _comparison_plan(contents, plan_file)
    raise ValueError(
        f"{plan_file.name}: kind must be 'training', 'evaluation', or 'comparison'"
    )


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
            "update_cycles": preset.update_cycles,
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
    _exact_keys(contents, {"format_version", *_TRAINING_SECTIONS}, "snapshot")
    format_version = contents["format_version"]
    if format_version not in (1, 2, CURRENT_SNAPSHOT_VERSION):
        raise ValueError("Unsupported training snapshot version")
    if format_version < 3:
        # Versions 1 and 2 stored the same two settings under earlier names.
        training = _table(contents, "training")
        rollout = _table(contents, "rollout")
        if "rollout_windows" in training:
            training["update_cycles"] = training.pop("rollout_windows")
        if "steps_per_environment" in rollout:
            rollout["rollout_window_steps"] = rollout.pop("steps_per_environment")
    return _parse_training(
        {name: contents[name] for name in _TRAINING_SECTIONS},
        source,
        legacy_snapshot=format_version == 1,
    )


def _training_plan(contents: dict[str, Any], plan_file: Path) -> TrainingPlan:
    _allowed_keys(
        contents, {"kind", *_TRAINING_SECTIONS}, {"evaluation"}, "training plan"
    )
    return TrainingPlan(
        plan_file=plan_file,
        preset=_parse_training(
            {name: contents[name] for name in _TRAINING_SECTIONS}, plan_file.parent
        ),
        evaluation_seeds=_optional_seeds(contents),
    )


def _evaluation_plan(contents: dict[str, Any], plan_file: Path) -> EvaluationPlan:
    _exact_keys(contents, {"kind", "source", "evaluation"}, "evaluation plan")
    return EvaluationPlan(
        plan_file=plan_file,
        source_run=_path(contents["source"], "source", plan_file.parent),
        evaluation_seeds=_optional_seeds(contents),
    )


def _comparison_plan(contents: dict[str, Any], plan_file: Path) -> ComparisonPlan:
    _allowed_keys(
        contents,
        {"kind", "base", "output_root"},
        {"grid", "variant", "evaluation"},
        "comparison plan",
    )
    if ("grid" in contents) == ("variant" in contents):
        raise ValueError("a comparison needs exactly one of [grid] or [[variant]]")

    base_file = _path(contents["base"], "base", plan_file.parent)
    base = _read_toml(base_file)
    if base.get("kind") != "training":
        raise ValueError("the comparison base must be a training plan")
    # Check the base on its own first; parsing modifies tables, so use a copy.
    _training_plan(copy.deepcopy(base), base_file)

    if "grid" in contents:
        named_changes = _grid_changes(contents["grid"])
    else:
        named_changes = _listed_changes(contents["variant"])

    output_root = _path(contents["output_root"], "output_root", plan_file.parent)
    variants = []
    for name, changes in named_changes:
        merged = {
            section: copy.deepcopy(base[section]) for section in _TRAINING_SECTIONS
        }
        for section, values in changes.items():
            merged[section].update(values)
        merged["training"]["output_root"] = str(output_root)
        variants.append(
            ComparisonVariant(
                name=name,
                changes={
                    f"{section}.{field}": value
                    for section, values in changes.items()
                    for field, value in values.items()
                },
                preset=_parse_training(merged, base_file.parent),
            )
        )
    return ComparisonPlan(
        plan_file=plan_file,
        output_root=output_root,
        variants=tuple(variants),
        evaluation_seeds=_optional_seeds(contents),
    )


def _grid_changes(grid: Any) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    """Expand ``[grid.<section>]`` value lists into every combination."""
    axes = []
    for section, values in _changed_sections(grid, "grid").items():
        for field, options in values.items():
            if (
                not isinstance(options, list)
                or not options
                or len({json.dumps(option) for option in options}) != len(options)
            ):
                raise ValueError(f"grid.{section}.{field} must list distinct values")
            axes.append((section, field, options))

    combinations = []
    for choice in itertools.product(*(options for _, _, options in axes)):
        changes: dict[str, dict[str, Any]] = {}
        name_parts = []
        for (section, field, options), value in zip(axes, choice, strict=True):
            changes.setdefault(section, {})[field] = value
            # Only settings that actually vary distinguish the variant names.
            if len(options) > 1:
                name_parts.append(f"{field}-{value}")
        combinations.append(("_".join(name_parts) or "base", changes))
    return combinations


def _listed_changes(variants: Any) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    """Read explicit ``[[variant]]`` tables, each with a unique name."""
    if not isinstance(variants, list) or not variants:
        raise ValueError("[[variant]] must define at least one variant")
    listed = []
    for variant in variants:
        if not isinstance(variant, dict) or "name" not in variant:
            raise ValueError("every [[variant]] needs a name")
        name = variant["name"]
        if not isinstance(name, str) or not _VARIANT_NAME.fullmatch(name):
            raise ValueError("variant names may use letters, digits, '.', '_', '-'")
        changes = {key: value for key, value in variant.items() if key != "name"}
        listed.append((name, _changed_sections(changes, f"variant {name}")))
    names = [name for name, _ in listed]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    return listed


def _changed_sections(changes: Any, name: str) -> dict[str, dict[str, Any]]:
    """Accept changes only inside the training, learner, and rollout sections."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError(f"{name} must change at least one setting")
    for section, values in changes.items():
        if section not in _TRAINING_SECTIONS or not isinstance(values, dict):
            raise ValueError(
                f"{name} may only change fields of [training], [learner], or [rollout]"
            )
    return changes


def _optional_seeds(contents: dict[str, Any]) -> tuple[int, ...]:
    """Read ``[evaluation] seeds`` when the plan asks for evaluation."""
    if "evaluation" not in contents:
        return ()
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
    return tuple(seeds)


def _parse_training(
    contents: dict[str, Any],
    base: Path,
    *,
    legacy_snapshot: bool = False,
) -> TrainingPreset:
    """Apply the same schema to a preset and its saved resolved snapshot."""
    _exact_keys(contents, set(_TRAINING_SECTIONS), "configuration")
    training = _table(contents, "training")
    learner = _table(contents, "learner")
    rollout = _table(contents, "rollout")
    training_fields = {
        "model_path",
        "output_root",
        "training_seed",
        "environment_count",
        "update_cycles",
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

    for name in ("rollout_window_steps", "update_epochs", "minibatch_size"):
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
        update_cycles=_integer(training["update_cycles"], "update_cycles", 1),
        max_episode_steps=_integer(
            training["max_episode_steps"], "max_episode_steps", 1
        ),
        learner_config=LearnerConfig(**learner),
        rollout_config=RolloutConfig(**rollout),
    )


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _table(contents: dict[str, Any], name: str) -> dict[str, Any]:
    value = contents[name]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    return value


def _allowed_keys(
    values: dict[str, Any], required: set[str], optional: set[str], name: str
) -> None:
    if not required <= set(values) <= required | optional:
        raise ValueError(f"{name} has missing or unexpected fields")


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
