"""Tests for reading configuration files into the experiment's configuration.

Each section's own values are tested with its component; these tests cover
what the experiment adds: the three kinds of file, saving the complete
configuration, and the checks across sections.
"""

from dataclasses import replace

import pytest

from centipede.experiment.configuration import (
    read_configuration,
    write_configuration,
)
from centipede.settings_section import SettingsError

TRAINING_FILE = """
[run]
mode = "train"
name = "probe"

[environment.simulation]
model_path = "models/assembly_v3.xml"
backend = "cpu"
world_count = 4

[interaction_loop]
update_cycles = 6
"""


def write(folder, text, name="file.toml"):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def saved_run(tmp_path):
    """A run folder holding the complete configuration of TRAINING_FILE."""
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    configuration = read_configuration(write(tmp_path, TRAINING_FILE))
    write_configuration(configuration, run_folder / "configuration.toml")
    return run_folder, configuration


def test_the_saved_configuration_reads_back_identically(saved_run):
    run_folder, configuration = saved_run

    assert read_configuration(run_folder / "configuration.toml") == configuration
    assert configuration.interaction_loop.update_cycles == 6
    assert configuration.run.seed == 0  # a default, now written out
    assert configuration.run.record_worlds == 64  # a default, now written out
    # Unset, the discount follows the episode length (rule R0): an arrival on
    # the last of 8,192 steps is worth half.
    assert configuration.agents.ppo.discount == 2 ** (-1 / 8192)


def test_files_that_name_a_run_start_from_its_saved_configuration(saved_run, tmp_path):
    run_folder, trained = saved_run

    continuing = read_configuration(
        write(
            tmp_path,
            f'[run]\nmode = "train"\ncontinue_from = "{run_folder.as_posix()}"\n'
            "[interaction_loop]\nupdate_cycles = 9\n",
        )
    )
    assert continuing.continue_from == run_folder
    assert continuing.interaction_loop.update_cycles == 9
    assert continuing.environment == trained.environment

    evaluating = read_configuration(
        write(
            tmp_path,
            f'[run]\nmode = "evaluate"\nsource = "{run_folder.as_posix()}"\n'
            "[evaluation]\nseeds = [5, 6]\nepisodes_per_seed = 3\n"
            "[environment.target]\ndistance_range_m = [0.03, 0.04]\n",
        )
    )
    evaluation = evaluating.evaluation
    assert evaluating.mode == "evaluate" and evaluation is not None
    assert evaluation.seeds == (5, 6)
    assert evaluation.environment_changes == {"target.distance_range_m": [0.03, 0.04]}
    expected_environment = replace(
        trained.environment,
        simulation=replace(trained.environment.simulation, world_count=3),
        target=replace(trained.environment.target, distance_range_m=(0.03, 0.04)),
    )
    assert evaluating.environment == expected_environment


def with_run_setting(line):
    return TRAINING_FILE.replace('name = "probe"', f'name = "probe"\n{line}')


def test_recordings_keep_a_number_of_worlds_and_old_runs_still_read(
    saved_run, tmp_path
):
    numbered = write(tmp_path, with_run_setting("record_worlds = 9"))
    assert read_configuration(numbered).run.record_worlds == 9

    # Older runs saved record_per_level and record_every_episodes, retired since.
    run_folder, _ = saved_run
    saved = run_folder / "configuration.toml"
    text = saved.read_text(encoding="utf-8")
    retired = "record_per_level = 8\nrecord_every_episodes = 4\n"
    saved.write_text(text.replace("[run]\n", "[run]\n" + retired, 1))
    assert "record_every_episodes" in saved.read_text(encoding="utf-8")
    continuing = read_configuration(
        write(
            tmp_path,
            f'[run]\nmode = "train"\ncontinue_from = "{run_folder.as_posix()}"\n',
        )
    )
    assert (continuing.run.recordings, continuing.run.record_worlds) == (5, 64)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (TRAINING_FILE + '[agents]\ndevice = "cuda"\n', "device must be 'cpu'"),
        (with_run_setting("record_worlds = 0"), 'must be "all" or a whole number'),
        (with_run_setting('record_worlds = "most"'), 'must be "all" or a whole'),
        (with_run_setting("record_per_level = 8"), "replaced on 2026-10-08"),
        (with_run_setting("record_every_episodes = 4"), "replaced on 2026-10-09"),
        (TRAINING_FILE.replace('"probe"', '"my probe"'), "name may only contain"),
        (TRAINING_FILE + "[evaluation]\nseeds = [1]\n", "may only contain"),
        (
            '[run]\nmode = "train"\ncontinue_from = "RUN"\n[agents]\n',
            "may only contain",
        ),
        (
            '[run]\nmode = "evaluate"\nsource = "RUN"\n[evaluation]\nseeds = [1]\n'
            "[environment.simulation]\nworld_count = 2\n",
            "episodes_per_seed sets",
        ),
        (
            '[run]\nmode = "evaluate"\nsource = "nowhere"\n[evaluation]\nseeds = [1]\n',
            "is not a run folder",
        ),
    ],
)
def test_invalid_files_are_rejected_with_a_clear_error(
    saved_run, tmp_path, text, problem
):
    run_folder, _ = saved_run
    path = write(tmp_path, text.replace("RUN", run_folder.as_posix()))

    with pytest.raises(SettingsError, match=problem):
        read_configuration(path)
