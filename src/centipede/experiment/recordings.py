"""Turning what the recorder kept into a MujocoReplay recording.

The interaction loop's recorder gives the recorded worlds' poses; this module
adds what a replay needs to stand alone: the model as one XML document, the
frame period, the target as a marker (and the range circle as a ring around
it, when the run has one), and the run's facts. The file format is
MujocoReplay's (its docs/recording-format.md); the run folder writes the
result.
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

TARGET_MARKER = "target"
RANGE_MARKER = "range"
# The target's radius in replays when arrival is under the head.
TARGET_MARKER_RADIUS_M = 0.001


@dataclass(frozen=True)
class RecordingScene:
    """What every recording of a run shares: the model, timing, the markers,
    and what each world's score counts.

    ``range_circle`` says whether the run's episodes have a range circle, drawn
    as a ring around each world's target.
    """

    model_xml: str
    frame_seconds: float
    marker_radius: float
    range_circle: bool
    score_name: str

    @classmethod
    def from_run(
        cls, configuration: Configuration, environment: Environment
    ) -> "RecordingScene":
        """Read the model once as a single document, includes resolved."""
        model_path = configuration.environment.simulation.model_path
        target = configuration.environment.target
        return cls(
            model_xml=mujoco.MjSpec.from_file(str(model_path)).to_xml(),
            frame_seconds=environment.simulation.step_duration_s,
            # With arrival under the head, the target is only a point to see.
            marker_radius=target.arrival_radius_m or TARGET_MARKER_RADIUS_M,
            range_circle=bool(target.range_circle_ratio)
            and configuration.environment.design == "main",
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
        **_markers(scene, recorded),
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
        **_markers(scene, recorded),
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


def _markers(scene: RecordingScene, recorded: RecordedEpisodes) -> dict[str, Any]:
    """The marker keys of a recording: the target, and the range ring if any.

    The target is a sphere resting on the ground. The ring lies flat around the
    same point, with each world's radius at each frame, which changes when a
    world starts a new episode; such a file is MujocoReplay's format 2.
    """
    radius = scene.marker_radius
    target = recorded.kept["target_position"]
    height = np.full(target.shape[:2] + (1,), radius, dtype=np.float32)
    on_ground = np.concatenate((target, height), axis=-1)[:, :, None, :]
    if not scene.range_circle:
        return {
            "marker_names": (TARGET_MARKER,),
            "marker_positions": on_ground,
            "marker_radius": np.array([radius]),
        }
    range_radius = recorded.kept["range_radius"]
    radii = np.stack(
        (np.full(range_radius.shape, radius), range_radius), axis=-1
    ).astype(np.float32)
    return {
        "marker_names": (TARGET_MARKER, RANGE_MARKER),
        "marker_positions": np.repeat(on_ground, 2, axis=2),
        "marker_radius": radii,
        "marker_shapes": ("sphere", "ring"),
    }


def _last_session(folder: RunFolder) -> dict[str, Any]:
    """The facts of the training session running now, from ``run_info.json``."""
    sessions = folder.read_run_info()["sessions"]
    return sessions[-1] if sessions else {}
