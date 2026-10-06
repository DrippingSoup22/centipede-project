"""Reading a run's TOML file into the checked settings of every component.

There are three kinds of file, told apart by their ``[run]`` section:

- a **training file** (``mode = "train"``) holds every section of a new run;
  with ``start_from`` its agents begin from another run's checkpoint;
- a **continuing file** (``mode = "train"`` with ``continue_from``) names an
  existing run and may only change ``[interaction_loop]``, such as a larger
  ``update_cycles``;
- an **evaluation file** (``mode = "evaluate"``) names a run or one of its
  checkpoints, holds ``[evaluation]``, and may change ``[environment]``
  settings, such as farther targets or the CPU backend.

Continuing and evaluation files start from the named run's saved
configuration, and their own sections replace the matching values. Every
result goes through the components' own ``from_section`` checks, so a value is
checked the same way whichever file it came from. See docs/configuration.md.
"""

import re
import tomllib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from centipede.agents.settings import AgentSettings
from centipede.environment.settings import EnvironmentSettings
from centipede.interaction_loop.settings import InteractionLoopSettings
from centipede.settings_section import SettingsError, SettingsSection

# The complete configuration saved in every run folder.
SAVED_CONFIGURATION_NAME = "configuration.toml"

# The agents' device that matches each physics backend; see "Checks across
# sections" in docs/configuration.md.
DEVICE_FOR_BACKEND = {"cpu": "cpu", "gpu": "cuda"}

TRAINING_SECTIONS = ("run", "environment", "agents", "interaction_loop")


@dataclass(frozen=True)
class RunSettings:
    """The [run] section of a training run.

    ``start_from`` is a run folder (its latest checkpoint) or a checkpoint
    file whose agents the new run begins with.
    """

    name: str
    seed: int
    checkpoint_every_cycles: int
    report: bool
    runs_folder: Path
    start_from: Path | None

    @classmethod
    def from_section(cls, values: dict) -> "RunSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "run")
        section.choice("mode", ("train",))
        settings = cls(
            name=section.text("name"),
            seed=section.integer("seed", default=0, minimum=0),
            checkpoint_every_cycles=section.positive_integer(
                "checkpoint_every_cycles", default=16
            ),
            report=section.boolean("report", default=True),
            runs_folder=section.path("runs_folder", default=Path("runs")),
            start_from=section.path("start_from", default=None),
        )
        section.reject_unknown_keys()
        # The name becomes part of a folder name.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", settings.name):
            raise SettingsError(
                "[run] name may only contain letters, digits, '-' and '_', "
                f"got {settings.name!r}"
            )
        return settings


@dataclass(frozen=True)
class EvaluationSettings:
    """An evaluation file's [evaluation] section and the checkpoint it names.

    ``source`` is a run folder (its latest checkpoint) or a checkpoint file.
    Each seed runs one episode in each of ``episodes_per_seed`` worlds, for the
    agents and for every listed baseline. ``environment_changes`` lists the
    environment settings the file changed from training, by dotted name.
    """

    source: Path
    seeds: tuple[int, ...]
    episodes_per_seed: int
    baselines: tuple[str, ...]
    environment_changes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_section(
        cls, values: dict, source: Path, environment_changes: dict[str, Any]
    ) -> "EvaluationSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "evaluation")
        settings = cls(
            source=source,
            seeds=section.integer_list("seeds", minimum=0),
            episodes_per_seed=section.positive_integer("episodes_per_seed", default=8),
            baselines=section.choice_list("baselines", ("zero", "random"), default=()),
            environment_changes=environment_changes,
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class Configuration:
    """Everything one command needs: the mode and every component's settings.

    ``run``, ``environment``, ``agents``, and ``interaction_loop`` describe the
    training run, even in evaluation, where they are the evaluated run's (with
    the evaluation's environment changes). ``continue_from`` is set only when
    continuing a run, and ``evaluation`` only when evaluating.
    """

    mode: str
    run: RunSettings
    environment: EnvironmentSettings
    agents: AgentSettings
    interaction_loop: InteractionLoopSettings
    continue_from: Path | None = None
    evaluation: EvaluationSettings | None = None

    def training_values(self) -> dict[str, Any]:
        """The training run's complete settings, as TOML-ready nested tables."""
        values = {
            "run": {"mode": "train", **asdict(self.run)},
            "environment": asdict(self.environment),
            "agents": asdict(self.agents),
            "interaction_loop": asdict(self.interaction_loop),
        }
        return _toml_ready(values)


def read_configuration(path: Path) -> Configuration:
    """Read and check one configuration file, of any of the three kinds."""
    file_values = read_toml(path)
    run_values = SettingsSection(file_values, "file").table("run")
    run_section = SettingsSection(run_values, "run")
    mode = run_section.choice("mode", ("train", "evaluate"))

    if mode == "evaluate":
        return _evaluation_configuration(file_values, run_section)
    if "continue_from" in run_values:
        return _continuing_configuration(file_values, run_section)
    _allow_only_sections(file_values, TRAINING_SECTIONS, "A training file")
    return _checked_configuration(file_values, mode="train")


def write_configuration(configuration: Configuration, path: Path) -> None:
    """Save the training run's complete settings, every default filled in."""
    with path.open("wb") as file:
        tomli_w.dump(configuration.training_values(), file)


def read_toml(path: Path) -> dict[str, Any]:
    """A TOML file's contents, with a clear error when it cannot be read."""
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError:
        raise SettingsError(f"Configuration file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise SettingsError(f"{path} is not valid TOML: {error}") from None


def dotted_keys(path: Path) -> list[str]:
    """The settings a file sets itself, as ``"environment.target.arrival_radius_m"``."""
    return sorted(_dotted(read_toml(path)))


def run_folder_of(source: Path) -> Path:
    """The run folder named by ``source``: the folder itself or a checkpoint's run."""
    folder = source.parent.parent if source.suffix == ".pt" else source
    if not (folder / SAVED_CONFIGURATION_NAME).is_file():
        raise SettingsError(
            f"{source} is not a run folder or a checkpoint inside one: "
            f"{folder / SAVED_CONFIGURATION_NAME} does not exist"
        )
    return folder


# -- The three kinds of file --------------------------------------------------


def _continuing_configuration(
    file_values: dict, run_section: SettingsSection
) -> Configuration:
    """The saved run's configuration, with the file's [interaction_loop] on top."""
    continue_from = run_section.path("continue_from")
    run_section.reject_unknown_keys()
    _allow_only_sections(
        file_values, ("run", "interaction_loop"), "A file with continue_from"
    )
    saved = read_toml(run_folder_of(continue_from) / SAVED_CONFIGURATION_NAME)
    changes = {"interaction_loop": file_values.get("interaction_loop", {})}
    values = _merged(saved, changes)
    return _checked_configuration(values, mode="train", continue_from=continue_from)


def _evaluation_configuration(
    file_values: dict, run_section: SettingsSection
) -> Configuration:
    """The evaluated run's configuration, with the file's [environment] on top.

    The number of worlds is ``episodes_per_seed``, and the agents' device
    follows the backend, so that a run trained on a GPU can be evaluated on
    the CPU by changing only ``backend``.
    """
    source = run_section.path("source")
    run_section.reject_unknown_keys()
    _allow_only_sections(
        file_values, ("run", "evaluation", "environment"), "An evaluation file"
    )
    environment_changes = SettingsSection(file_values, "file").table("environment")
    simulation_changes = environment_changes.get("simulation", {})
    if "world_count" in simulation_changes:
        raise SettingsError(
            "[environment.simulation] world_count cannot be set in an evaluation "
            "file: [evaluation] episodes_per_seed sets the number of worlds"
        )

    evaluation = EvaluationSettings.from_section(
        SettingsSection(file_values, "file").table("evaluation"),
        source,
        _dotted(environment_changes),
    )
    saved = read_toml(run_folder_of(source) / SAVED_CONFIGURATION_NAME)
    values = _merged(saved, {"environment": environment_changes})
    simulation = values["environment"]["simulation"]
    simulation["world_count"] = evaluation.episodes_per_seed
    backend = simulation.get("backend")
    values.setdefault("agents", {})["device"] = DEVICE_FOR_BACKEND.get(backend, "cpu")
    return _checked_configuration(values, mode="evaluate", evaluation=evaluation)


def _checked_configuration(values: dict, mode: str, **extra: Any) -> Configuration:
    """Hand each section to its owner, then make the checks across sections."""
    file_section = SettingsSection(values, "file")
    configuration = Configuration(
        mode=mode,
        run=RunSettings.from_section(file_section.table("run")),
        environment=EnvironmentSettings.from_section(file_section.table("environment")),
        agents=AgentSettings.from_section(file_section.table("agents")),
        interaction_loop=InteractionLoopSettings.from_section(
            file_section.table("interaction_loop")
        ),
        **extra,
    )
    backend = configuration.environment.simulation.backend
    if configuration.agents.device != DEVICE_FOR_BACKEND[backend]:
        raise SettingsError(
            f"[agents] device must be {DEVICE_FOR_BACKEND[backend]!r} when "
            f"[environment.simulation] backend is {backend!r}, got "
            f"{configuration.agents.device!r}"
        )
    return configuration


# -- Helpers --------------------------------------------------------------------


def _allow_only_sections(values: dict, allowed: tuple[str, ...], kind: str) -> None:
    """Fail if the file has a top-level section this kind of file may not have."""
    unexpected = sorted(set(values) - set(allowed))
    if unexpected:
        names = ", ".join(f"[{name}]" for name in unexpected)
        sections = ", ".join(f"[{name}]" for name in allowed)
        raise SettingsError(f"{kind} may only contain {sections}; found {names}")


def _merged(base: dict, changes: dict) -> dict:
    """A copy of ``base`` with the values in ``changes`` replacing its own."""
    merged = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merged(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _dotted(values: dict, prefix: str = "") -> dict[str, Any]:
    """Nested tables flattened to ``{"target.distance_range_m": value}``."""
    flat = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat |= _dotted(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _toml_ready(value: Any) -> Any:
    """Settings values as TOML can write them; unset (None) values are left out."""
    if isinstance(value, dict):
        return {
            key: _toml_ready(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_toml_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
