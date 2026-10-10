"""One run's folder: where everything a run produces is written and read back.

A training run's folder, under ``runs_folder``::

    2026-10-07_1432_probe/
    ├─ configuration.toml   the complete settings; can be run again as they are
    ├─ run_info.json        how and with what each training session ran
    ├─ training_log.jsonl   one line of diagnostics per window
    ├─ report.html          the run's summary, unless the run turned it off
    ├─ checkpoints/         cycle_0016.pt, cycle_0032.pt, ...
    ├─ recordings/          cycles_0001-0016.npz, ...: replays of episode lengths
    └─ evaluations/         results, report, and replays of each evaluation

Folders are created only when they do not exist yet, and an existing run is
never overwritten: a new run whose name is taken gets a numbered suffix.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from mujoco_replay.recording import Recording, write_recording

from centipede.experiment.configuration import (
    SAVED_CONFIGURATION_NAME,
    Configuration,
    run_folder_of,
    write_configuration,
)
from centipede.settings_section import SettingsError


class RunFolder:
    """Reads and writes the files of one run's folder."""

    def __init__(self, path: Path) -> None:
        """Use an existing run folder; ``create`` and ``open`` are the usual ways."""
        self.path = path
        self.checkpoints = path / "checkpoints"
        self.recordings = path / "recordings"
        self.evaluations = path / "evaluations"
        self.log_path = path / "training_log.jsonl"
        self.run_info_path = path / "run_info.json"
        self.report_path = path / "report.html"

    @classmethod
    def create(cls, runs_folder: Path, name: str) -> "RunFolder":
        """A new, empty folder named ``<date>_<time>_<name>`` inside ``runs_folder``."""
        runs_folder.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
        path = runs_folder / stem
        suffix = 2
        while path.exists():
            path = runs_folder / f"{stem}_{suffix}"
            suffix += 1
        path.mkdir()
        return cls(path)

    @classmethod
    def open(cls, source: Path) -> "RunFolder":
        """The existing run named by a run folder or by a checkpoint inside one."""
        return cls(run_folder_of(source))

    # -- Configuration and run information ----------------------------------------

    def write_configuration(self, configuration: Configuration) -> None:
        """Save the run's complete settings, replacing any earlier copy."""
        write_configuration(configuration, self.path / SAVED_CONFIGURATION_NAME)

    def add_session(self, session: dict[str, Any]) -> None:
        """Add one training session's facts to ``run_info.json``."""
        run_info = self.read_run_info()
        run_info["sessions"].append(session)
        self.run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    def read_run_info(self) -> dict[str, Any]:
        """The facts of every training session so far."""
        if not self.run_info_path.exists():
            return {"sessions": []}
        return json.loads(self.run_info_path.read_text(encoding="utf-8"))

    # -- Training log -----------------------------------------------------------------

    def append_log(self, record: dict[str, Any]) -> None:
        """Add one window's line to the training log."""
        line = json.dumps(record, allow_nan=False)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def read_log(self) -> list[dict[str, Any]]:
        """Every window's record, in order."""
        if not self.log_path.exists():
            return []
        with self.log_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def cut_log_after(self, cycle: int) -> None:
        """Drop the lines of windows after ``cycle``, which a resumed run repeats.

        A session that stops between two checkpoints has logged windows that
        no checkpoint kept; continuing restarts from the checkpoint.
        """
        kept = [record for record in self.read_log() if record["cycle"] <= cycle]
        lines = "".join(json.dumps(record) + "\n" for record in kept)
        self.log_path.write_text(lines, encoding="utf-8")

    # -- Checkpoints ------------------------------------------------------------------

    def save_checkpoint(
        self,
        completed_cycles: int,
        agents_state: dict,
        curriculum_level: float | None = None,
        leg_clocks: bool = False,
    ) -> Path:
        """Save the agents after ``completed_cycles`` cycles, whether leg clocks
        drove their legs, and the curriculum's level when the run has one;
        returns the file."""
        self.checkpoints.mkdir(exist_ok=True)
        path = self.checkpoints / f"cycle_{completed_cycles:04d}.pt"
        checkpoint = {
            "completed_cycles": completed_cycles,
            "agents": agents_state,
            "leg_clocks": leg_clocks,
        }
        if curriculum_level is not None:
            checkpoint["curriculum_level"] = curriculum_level
        torch.save(checkpoint, path)
        return path

    def latest_checkpoint(self) -> Path:
        """The checkpoint with the most completed cycles."""
        checkpoints = list(self.checkpoints.glob("cycle_*.pt"))
        if not checkpoints:
            raise SettingsError(f"{self.path} has no checkpoints yet")
        return max(checkpoints, key=lambda path: int(path.stem.removeprefix("cycle_")))

    def checkpoint_named_by(self, source: Path) -> Path:
        """``source`` itself if it is a checkpoint file, else the latest one."""
        if source.suffix != ".pt":
            return self.latest_checkpoint()
        if not source.is_file():
            raise SettingsError(f"Checkpoint not found: {source}")
        return source

    @staticmethod
    def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
        """A checkpoint's contents, with its tensors on ``device``.

        PyTorch's safe loader reads only tensors and plain values.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or set(checkpoint) - {
            "curriculum_level",
            "leg_clocks",
        } != {"completed_cycles", "agents"}:
            raise SettingsError(f"{path} is not a checkpoint of this project")
        return checkpoint

    # -- Recordings -------------------------------------------------------------------

    def write_recording(self, stem: str, recording: Recording) -> Path:
        """Save a training window's recording as ``recordings/<stem>.npz``."""
        self.recordings.mkdir(exist_ok=True)
        path = self.recordings / f"{stem}.npz"
        write_recording(path, recording)
        return path

    def write_evaluation_recording(self, stem: str, recording: Recording) -> Path:
        """Save an evaluation's recording as ``evaluations/<stem>.npz``."""
        self.evaluations.mkdir(exist_ok=True)
        path = self.evaluations / f"{stem}.npz"
        write_recording(path, recording)
        return path

    # -- Evaluations ------------------------------------------------------------------

    def write_evaluation(self, stem: str, results: dict[str, Any]) -> Path:
        """Save one evaluation's results as ``evaluations/<stem>.json``."""
        self.evaluations.mkdir(exist_ok=True)
        path = self.evaluations / f"{stem}.json"
        path.write_text(json.dumps(results, indent=1, allow_nan=False), "utf-8")
        return path
