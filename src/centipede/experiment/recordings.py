"""Turning what the recorder kept into a MujocoReplay recording.

The interaction loop's recorder gives the recorded worlds' poses; this module
adds what a replay needs to stand alone: the model as one XML document, the
frame period, the line as a path around the body, and the run's facts. The
file format is MujocoReplay's (its docs/recording-format.md); the run folder
writes the result.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco_replay.recording import Recording

from centipede.environment.environment import Environment
from centipede.experiment.configuration import Configuration
from centipede.experiment.run_folder import RunFolder
from centipede.interaction_loop.recording import RecordedEpisodes

LINE_PATH = "line"


@dataclass(frozen=True)
class RecordingScene:
    """What every recording of a run shares: the model, timing, and what each
    world's score counts."""

    model_xml: str
    frame_seconds: float
    score_name: str

    @classmethod
    def from_run(
        cls, configuration: Configuration, environment: Environment
    ) -> "RecordingScene":
        """Read the model once as a single document, includes resolved."""
        model_path = configuration.environment.simulation.model_path
        return cls(
            model_xml=mujoco.MjSpec.from_file(str(model_path)).to_xml(),
            frame_seconds=environment.simulation.step_duration_s,
            score_name=environment.diagnostics.score_name,
        )


def training_recording(
    scene: RecordingScene,
    recorded: RecordedEpisodes,
    percent: int,
    cycles: tuple[int, int],
    configuration: Configuration,
    folder: RunFolder,
) -> Recording:
    """One episode of each recorded world, around ``percent`` of training.

    ``cycles`` are the first and last update cycles the recording spans; the
    agents update after each, so the episodes show them learning.
    """
    first_cycle, last_cycle = cycles
    total_cycles = configuration.interaction_loop.update_cycles
    session = _last_session(folder)
    return Recording(
        model_xml=scene.model_xml,
        frame_seconds=scene.frame_seconds,
        qpos=recorded.qpos,
        score=recorded.score,
        score_name=scene.score_name,
        world_ids=recorded.world_ids,
        episode_start=recorded.episode_start,
        **_line_path(recorded),
        title=(
            f"{folder.path.name} · {percent}% of training · cycles"
            f" {first_cycle}–{last_cycle} of {total_cycles}"
        ),
        setup={
            "run": folder.path.name,
            "share of training": f"{percent}%",
            "cycles": f"{first_cycle}–{last_cycle}",
            "worlds": configuration.environment.simulation.world_count,
            "worlds recorded": int(recorded.qpos.shape[1]),
            "device": session.get("device"),
            "code version": session.get("git_commit"),
            "configuration": configuration.training_values(),
        },
    )


def evaluation_recording(
    scene: RecordingScene,
    recorded: RecordedEpisodes,
    actor_name: str,
    seed: int,
    checkpoint_path: Path,
    cycles_trained: int,
    configuration: Configuration,
    folder: RunFolder,
) -> Recording:
    """Every world's first episode of one actor and seed, and what followed.

    Worlds are scored by their first episode only, as in the evaluation's
    results, and come best first.
    """
    return Recording(
        model_xml=scene.model_xml,
        frame_seconds=scene.frame_seconds,
        qpos=recorded.qpos,
        score=recorded.score,
        score_name=f"first episode's {scene.score_name}",
        world_ids=recorded.world_ids,
        episode_start=recorded.episode_start,
        **_line_path(recorded),
        title=(
            f"{folder.path.name} · {checkpoint_path.stem} · {actor_name} · seed {seed}"
        ),
        setup={
            "run": folder.path.name,
            "checkpoint": str(checkpoint_path),
            "cycles trained": cycles_trained,
            "actor": actor_name,
            "seed": seed,
            "worlds": int(recorded.qpos.shape[1]),
            "configuration": configuration.training_values(),
        },
    )


def _line_path(recorded: RecordedEpisodes) -> dict[str, Any]:
    """The path keys of a recording: each world's line, a piece on the
    ground around its body at each frame."""
    points = recorded.kept["line_points"]
    on_ground = np.concatenate((points, np.zeros_like(points[..., :1])), axis=-1)
    return {
        "path_names": (LINE_PATH,),
        "path_points": on_ground[:, :, None].astype(np.float32),
    }


def _last_session(folder: RunFolder) -> dict[str, Any]:
    """The facts of the training session running now, from ``run_info.json``."""
    sessions = folder.read_run_info()["sessions"]
    return sessions[-1] if sessions else {}
