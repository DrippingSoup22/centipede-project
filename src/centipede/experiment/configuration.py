"""Reading a run's TOML file into the checked settings of every component.

There are three kinds of file, told apart by their ``[run]`` section:

- a **training file** (``mode = "train"``) holds every section of a new run;
  with ``start_from`` its agents begin from another run's checkpoint;
- a **continuing file** (``mode = "train"`` with ``continue_from``) names an
  existing run and may only change ``[interaction_loop]``, such as a larger
  ``update_cycles``, and the session's ``time_limit_hours``;
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

TRAINING_SECTIONS = ("run", "environment", "agents", "interaction_loop", "curriculum")
# The cycles a run whose segments have clocks trains, by default, before its
# plateau stop may end it.
CLOCK_PLATEAU_MINIMUM_CYCLES = 128


@dataclass(frozen=True)
class RunSettings:
    """The [run] section of a training run.

    ``start_from`` is a run folder (its latest checkpoint) or a checkpoint
    file whose agents the new run begins with. ``recordings`` is how many
    episode lengths are recorded for replay, each as one file: the first, the
    last, and the others spread evenly between, or ``"all"``. The ``record_*``
    settings say how many worlds each keeps (a number, or ``"all"``), which
    ones when not all, and how many levels their ranks are sorted into.
    ``time_limit_hours`` bounds one training session: the run stops cleanly,
    with a checkpoint, before a cycle that would end after it. With
    ``plateau_cycles`` the run also stops cleanly once its progress has not
    risen by ``plateau_progress`` for that many cycles, never before
    ``plateau_minimum_cycles`` (plateau.py). See docs/configuration.md.
    """

    name: str
    seed: int
    checkpoint_every_cycles: int
    report: bool
    runs_folder: Path
    start_from: Path | None
    recordings: int | str
    record_levels: int
    record_worlds: int | str
    record_selection: str
    time_limit_hours: float | None
    plateau_cycles: int
    plateau_progress: float
    plateau_minimum_cycles: int

    @classmethod
    def from_section(cls, values: dict) -> "RunSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "run")
        section.choice("mode", ("train",))
        if "record_per_level" in values:
            raise SettingsError(
                "[run] record_per_level was replaced on 2026-10-08 by record_worlds,"
                ' how many worlds a recording keeps in all, or "all" (the default);'
                " record_levels x record_per_level gives the number it kept"
            )
        if "record_every_episodes" in values:
            raise SettingsError(
                "[run] record_every_episodes was replaced on 2026-10-09 by"
                " recordings, how many episode lengths a run records, spread from"
                ' the first to the last (5 by default), or "all"'
            )
        checkpoint_every_cycles = section.positive_integer(
            "checkpoint_every_cycles", default=16
        )
        settings = cls(
            name=section.text("name"),
            seed=section.integer("seed", default=0, minimum=0),
            checkpoint_every_cycles=checkpoint_every_cycles,
            report=section.boolean("report", default=True),
            runs_folder=section.path("runs_folder", default=Path("runs")),
            start_from=section.path("start_from", default=None),
            recordings=section.count_or_all("recordings", default=5, minimum=0),
            record_levels=section.positive_integer("record_levels", default=4),
            record_worlds=section.count_or_all("record_worlds", default=64),
            record_selection=section.choice(
                "record_selection", ("ranked", "first"), default="ranked"
            ),
            time_limit_hours=section.positive_number("time_limit_hours", default=None),
            plateau_cycles=section.integer("plateau_cycles", default=0, minimum=0),
            plateau_progress=section.positive_number("plateau_progress", default=0.02),
            plateau_minimum_cycles=section.integer(
                "plateau_minimum_cycles", default=0, minimum=0
            ),
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
class CurriculumSettings:
    """The [curriculum] section: targets that get harder as the agents improve.

    Level 0 draws the targets from [environment.target]'s ranges, level 1 from
    the final ranges here, and a level between from ranges in between. After
    each update cycle the level moves by ``level_rate`` times the gap between
    the arrival share and ``arrival_share``, within 0 and 1, so that it settles
    where the agents arrive that often. See docs/configuration.md.
    """

    final_distance_range_m: tuple[float, float]
    final_bearing_range_deg: tuple[float, float]
    arrival_share: float
    level_rate: float

    @classmethod
    def from_section(cls, values: dict) -> "CurriculumSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "curriculum")
        settings = cls(
            final_distance_range_m=section.number_range(
                "final_distance_range_m", minimum=0.0
            ),
            final_bearing_range_deg=section.number_range("final_bearing_range_deg"),
            arrival_share=section.number(
                "arrival_share", default=0.5, minimum=0.0, maximum=1.0
            ),
            level_rate=section.positive_number("level_rate", default=0.02),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class EvaluationSettings:
    """An evaluation file's [evaluation] section and the checkpoint it names.

    ``source`` is a run folder (its latest checkpoint) or a checkpoint file.
    Each seed runs one episode in each of ``episodes_per_seed`` worlds, for the
    agents and for every listed baseline. ``environment_changes`` lists the
    environment settings the file changed from training, by dotted name.
    ``record`` writes a replay recording of every world for each actor and seed.
    With ``walk_steps``, each seed also lets the agents walk that many steps in
    every world, counting every episode, for the targets they reach in a fixed
    time; 0 leaves it out.
    """

    source: Path
    seeds: tuple[int, ...]
    episodes_per_seed: int
    baselines: tuple[str, ...]
    environment_changes: dict[str, Any] = field(default_factory=dict)
    record: bool = True
    walk_steps: int = 0

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
            record=section.boolean("record", default=True),
            walk_steps=section.integer("walk_steps", default=0, minimum=0),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class Configuration:
    """Everything one command needs: the mode and every component's settings.

    ``run``, ``environment``, ``agents``, and ``interaction_loop`` describe the
    training run, even in evaluation, where they are the evaluated run's (with
    the evaluation's environment changes). ``curriculum`` is set when the run
    has one; an evaluation never follows it. ``continue_from`` is set only when
    continuing a run, and ``evaluation`` only when evaluating.
    """

    mode: str
    run: RunSettings
    environment: EnvironmentSettings
    agents: AgentSettings
    interaction_loop: InteractionLoopSettings
    curriculum: CurriculumSettings | None = None
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
        if self.curriculum is not None:
            values["curriculum"] = asdict(self.curriculum)
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
    """The saved run's configuration, with the file's changes on top.

    The file may set [interaction_loop] and, in [run], the new session's
    ``time_limit_hours``.
    """
    continue_from = run_section.path("continue_from")
    time_limit_hours = run_section.positive_number("time_limit_hours", default=None)
    run_section.reject_unknown_keys()
    _allow_only_sections(
        file_values, ("run", "interaction_loop"), "A file with continue_from"
    )
    saved = _saved_configuration(continue_from)
    changes = {"interaction_loop": file_values.get("interaction_loop", {})}
    if time_limit_hours is not None:
        changes["run"] = {"time_limit_hours": time_limit_hours}
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
    saved = _saved_configuration(source)
    values = _merged(saved, {"environment": environment_changes})
    simulation = values["environment"]["simulation"]
    simulation["world_count"] = evaluation.episodes_per_seed
    backend = simulation.get("backend")
    values.setdefault("agents", {})["device"] = DEVICE_FOR_BACKEND.get(backend, "cpu")
    return _checked_configuration(values, mode="evaluate", evaluation=evaluation)


def _checked_configuration(values: dict, mode: str, **extra: Any) -> Configuration:
    """Hand each section to its owner, then make the checks across sections."""
    file_section = SettingsSection(values, "file")
    environment = EnvironmentSettings.from_section(file_section.table("environment"))
    configuration = Configuration(
        mode=mode,
        run=RunSettings.from_section(
            _with_plateau_minimum(file_section.table("run"), environment.clocks)
        ),
        environment=environment,
        agents=AgentSettings.from_section(
            _with_episode_discount(
                file_section.table("agents"), environment.max_episode_steps
            )
        ),
        interaction_loop=InteractionLoopSettings.from_section(
            file_section.table("interaction_loop")
        ),
        curriculum=(
            CurriculumSettings.from_section(file_section.table("curriculum"))
            if "curriculum" in values
            else None
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


def _with_episode_discount(agents_values: dict, max_episode_steps: int) -> dict:
    """The agents' values, with the discount ``2^(−1/T)`` when they set none.

    Rule R0 of docs/environment.md: an arrival on an episode's last step is
    worth half the arrival reward from its first, which the reward's cost
    budget relies on. The value is then saved with the run like any other.
    """
    ppo = agents_values.get("ppo", {})
    if not isinstance(ppo, dict) or "discount" in ppo:
        return agents_values
    return _merged(agents_values, {"ppo": {"discount": 2 ** (-1 / max_episode_steps)}})


def _with_plateau_minimum(run_values: dict, clocks: bool) -> dict:
    """The [run] values, with ``plateau_minimum_cycles`` when they set none.

    A run whose segments have clocks trains CLOCK_PLATEAU_MINIMUM_CYCLES
    before its plateau stop may end it, since its segments first need to find
    a common tempo and their offsets; any other run, none. The value is then
    saved with the run like any other.
    """
    if "plateau_minimum_cycles" in run_values:
        return run_values
    minimum = CLOCK_PLATEAU_MINIMUM_CYCLES if clocks else 0
    return {**run_values, "plateau_minimum_cycles": minimum}


# Settings that older runs saved and the current code no longer has, by
# section. record_every_cycles recorded single windows; record_every_episodes
# replaced it, and recordings replaced that, each with its own default.
# record_per_level kept that many worlds of each level; record_worlds replaced
# it.
RETIRED_SETTINGS = {
    "run": ("record_every_cycles", "record_every_episodes", "record_per_level")
}


def _saved_configuration(source: Path) -> dict:
    """A saved run's configuration, without the settings retired since."""
    values = read_toml(run_folder_of(source) / SAVED_CONFIGURATION_NAME)
    for section, keys in RETIRED_SETTINGS.items():
        for key in keys:
            values.get(section, {}).pop(key, None)
    return values


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
