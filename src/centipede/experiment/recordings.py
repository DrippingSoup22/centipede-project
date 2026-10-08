"""Turning what the recorder kept into a MujocoReplay recording.

The interaction loop's recorder gives the chosen worlds' poses; this module
adds what a replay needs to stand alone: the model as one XML document, the
frame period, the target as a marker (and the range circle as a ring around
it, when the run has one), the run's facts, and the events. The
file format is MujocoReplay's (its docs/recording-format.md); the run folder
writes the result.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco_replay.recording import Recording

from centipede.environment.environment import Environment
from centipede.experiment.configuration import Configuration
from centipede.experiment.run_folder import RunFolder
from centipede.interaction_loop.recording import RecordedWindow

SCORE_NAME = "summed reward"
FIRST_EPISODE_SCORE_NAME = "first episode's summed reward"
TARGET_MARKER = "target"
RANGE_MARKER = "range"
# The target's radius in replays when arrival is under the head.
TARGET_MARKER_RADIUS_M = 0.001


@dataclass(frozen=True)
class RecordingScene:
    """What every recording of a run shares: the model, timing, and the markers.

    ``range_circle`` says whether the run's episodes have a range circle, drawn
    as a ring around each world's target.
    """

    model_xml: str
    frame_seconds: float
    marker_radius: float
    range_circle: bool

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
            range_circle=bool(target.range_circle_ratio),
        )


def training_recording(
    scene: RecordingScene,
    recorded: RecordedWindow,
    first_cycle: int,
    total_cycles: int,
    configuration: Configuration,
    folder: RunFolder,
) -> Recording:
    """The windows collected from ``first_cycle`` on, usually one episode length.

    The agents update after every window, so the frames record which update
    each step was taken after, and each update is an event.
    """
    frames = recorded.qpos.shape[0]
    window_steps = configuration.interaction_loop.rollout_window_steps
    last_cycle = first_cycle + math.ceil(frames / window_steps) - 1
    frame_indices = np.arange(frames)
    updates_done = first_cycle - 1 + frame_indices // window_steps
    steps_per_world = (first_cycle - 1) * window_steps + frame_indices + 1
    update_frames = np.arange(window_steps, frames + 1, window_steps)
    session = _last_session(folder)
    return Recording(
        model_xml=scene.model_xml,
        frame_seconds=scene.frame_seconds,
        qpos=recorded.qpos,
        score=recorded.score,
        score_name=SCORE_NAME,
        world_ids=recorded.world_ids,
        level=recorded.level,
        episode_start=recorded.episode_start,
        level_count=recorded.level_count,
        rank=recorded.rank,
        ranked_worlds=recorded.ranked_worlds,
        **_markers(scene, recorded),
        frame_info_names=("updates", "steps per world"),
        frame_info=np.stack(
            (updates_done.astype(np.float64), steps_per_world.astype(np.float64)),
            axis=1,
        ),
        event_frames=update_frames,
        event_labels=tuple(
            f"update {first_cycle + index}" for index in range(len(update_frames))
        ),
        title=(
            f"{folder.path.name} · cycles {first_cycle}–{last_cycle} of {total_cycles}"
        ),
        setup={
            "run": folder.path.name,
            "cycles": f"{first_cycle}–{last_cycle}",
            "updates done at the start": first_cycle - 1,
            "worlds": configuration.environment.simulation.world_count,
            "worlds recorded": int(recorded.qpos.shape[1]),
            "device": session.get("device"),
            "code version": session.get("git_commit"),
            "configuration": configuration.training_values(),
        },
    )


def evaluation_recording(
    scene: RecordingScene,
    window: RecordedWindow,
    actor_name: str,
    seed: int,
    checkpoint_path: Path,
    cycles_trained: int,
    configuration: Configuration,
    folder: RunFolder,
) -> Recording:
    """Every world's first episode of one actor and seed, and what followed.

    Worlds are scored and ranked by their first episode only, as in the
    evaluation's results. The file holds every world, so its order is the
    run's ranking and levels would say nothing.
    """
    return Recording(
        model_xml=scene.model_xml,
        frame_seconds=scene.frame_seconds,
        qpos=window.qpos,
        score=window.score,
        score_name=FIRST_EPISODE_SCORE_NAME,
        world_ids=window.world_ids,
        episode_start=window.episode_start,
        **_markers(scene, window),
        title=(
            f"{folder.path.name} · {checkpoint_path.stem} · {actor_name} · seed {seed}"
        ),
        setup={
            "run": folder.path.name,
            "checkpoint": str(checkpoint_path),
            "cycles trained": cycles_trained,
            "actor": actor_name,
            "seed": seed,
            "worlds": int(window.qpos.shape[1]),
            "configuration": configuration.training_values(),
        },
    )


def _markers(scene: RecordingScene, recorded: RecordedWindow) -> dict[str, Any]:
    """The marker keys of a recording: the target, and the range ring if any.

    The target is a sphere resting on the ground. The ring lies flat around the
    same point, with each world's radius at each frame, which changes when a
    world starts a new episode; such a file is MujocoReplay's format 2.
    """
    radius = scene.marker_radius
    target = recorded.target
    height = np.full(target.shape[:2] + (1,), radius, dtype=np.float32)
    on_ground = np.concatenate((target, height), axis=-1)[:, :, None, :]
    if not scene.range_circle:
        return {
            "marker_names": (TARGET_MARKER,),
            "marker_positions": on_ground,
            "marker_radius": np.array([radius]),
        }
    radii = np.stack(
        (np.full(recorded.range_radius.shape, radius), recorded.range_radius), axis=-1
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
