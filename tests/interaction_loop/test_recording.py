"""Tests for the interaction loop's window recorder, with hand-made categories.

Three worlds of a two-segment body with a position vector of two entries.
The recorder reads only the categories, so no simulation is needed.
"""

import numpy as np
import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.simulation.diagnostics import SimulationDiagnostics
from centipede.interaction_loop.recording import WindowRecorder

WORLDS, SEGMENTS, POSITIONS = 3, 2, 2


def make_recorder():
    simulation = SimulationDiagnostics.allocate(WORLDS, POSITIONS, "cpu")
    diagnostics = EnvironmentDiagnostics(
        WORLDS, SEGMENTS, ["arrival"], "cpu", simulation.facts
    )
    return WindowRecorder(diagnostics, WORLDS), diagnostics


def take_steps(recorder, diagnostics, rewards_per_world, start_frame=0):
    """Steps with each world's reward; world 1 restarts an episode at frame 1."""
    for frame, reward in enumerate(rewards_per_world, start=start_frame):
        diagnostics.simulation.qpos.copy_(
            torch.arange(WORLDS * POSITIONS).reshape(WORLDS, POSITIONS) + 10 * frame
        )
        diagnostics.step.reward_parts.copy_(
            torch.tensor(reward)[:, None, None].expand(WORLDS, SEGMENTS, 1) / SEGMENTS
        )
        diagnostics.step.target_position.fill_(float(frame))
        diagnostics.episode.episode_ended.copy_(
            torch.tensor([False, frame == 1, False])
        )
        recorder.step_taken()


def test_only_armed_windows_are_recorded_and_worlds_are_ranked():
    recorder, diagnostics = make_recorder()

    recorder.start_window()  # not armed: nothing is kept
    take_steps(recorder, diagnostics, [[1.0, 1.0, 1.0]])
    assert not recorder.recording

    recorder.arm(frames=2)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0], [0.5, 2.0, -1.0]])
    window = recorder.take(levels=2, per_level=1, selection="ranked")

    # Best world 1, then the worst band's first rank (world 2), best first.
    assert window.world_ids.tolist() == [1, 2]
    assert window.level.tolist() == [1, 2]
    np.testing.assert_allclose(window.score, [4.0, -2.0])
    assert window.qpos.shape == (2, 2, POSITIONS)
    np.testing.assert_allclose(window.qpos[1, 0], [12.0, 13.0])  # frame 1, world 1
    assert window.episode_start.tolist() == [[False, False], [True, False]]
    assert window.target.shape == (2, 2, 2) and window.target[1].max() == 1.0
    assert not recorder.recording


def test_first_and_all_keep_the_asked_worlds_with_their_levels():
    recorder, diagnostics = make_recorder()
    recorder.arm(frames=4)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])  # one of four frames

    first = recorder.take(levels=1, per_level=2, selection="first")
    assert first.world_ids.tolist() == [1, 0]  # worlds 0 and 1, best first
    assert first.level.tolist() == [1, 1]
    assert first.qpos.shape == (1, 2, POSITIONS)

    recorder.arm(frames=1)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])
    everything = recorder.take(levels=3, per_level=1, selection="all")
    assert everything.world_ids.tolist() == [1, 0, 2]
    assert everything.level.tolist() == [1, 2, 3]
