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


def take_steps(recorder, diagnostics, rewards_per_world, start_frame=0, counted=None):
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
        recorder.step_taken(counted)


def test_only_armed_windows_are_recorded_and_worlds_are_ranked():
    recorder, diagnostics = make_recorder()

    recorder.start_window()  # not armed: nothing is kept
    take_steps(recorder, diagnostics, [[1.0, 1.0, 1.0]])
    assert not recorder.recording

    recorder.arm(frames=2)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0], [0.5, 2.0, -1.0]])
    window = recorder.take(levels=2, worlds=2, selection="ranked")

    # Two of three worlds, evenly over the ranks: the best and the worst.
    assert window.world_ids.tolist() == [1, 2]
    assert window.level.tolist() == [1, 2]
    assert (window.rank.tolist(), window.ranked_worlds, window.level_count) == (
        [1, 3],
        3,
        2,
    )
    np.testing.assert_allclose(window.score, [4.0, -2.0])
    assert window.qpos.shape == (2, 2, POSITIONS)
    np.testing.assert_allclose(window.qpos[1, 0], [12.0, 13.0])  # frame 1, world 1
    assert window.episode_start.tolist() == [[False, False], [True, False]]
    assert window.target.shape == (2, 2, 2) and window.target[1].max() == 1.0
    assert not recorder.recording


def test_a_recording_carries_on_through_windows_until_complete():
    recorder, diagnostics = make_recorder()
    recorder.arm(frames=3)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert recorder.recording and not recorder.complete

    recorder.start_window()  # the next window continues the recording
    take_steps(recorder, diagnostics, [[1.0, 0.0, 0.0]] * 2, start_frame=2)
    assert recorder.complete  # the fourth step was not kept

    window = recorder.take(levels=1, worlds=1, selection="ranked")
    assert window.qpos.shape == (3, 1, POSITIONS)
    np.testing.assert_allclose(window.qpos[2, 0], [20.0, 21.0])  # frame 2, world 0
    np.testing.assert_allclose(window.score, [3.0])
    recorder.start_window()  # not armed again: nothing is recorded
    assert not recorder.recording


def test_first_and_all_keep_the_asked_worlds_with_their_levels():
    recorder, diagnostics = make_recorder()
    recorder.arm(frames=4)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])  # one of four frames

    first = recorder.take(levels=1, worlds=2, selection="first")
    assert first.world_ids.tolist() == [1, 0]  # worlds 0 and 1, best first
    assert first.level.tolist() == [1, 1]
    assert first.qpos.shape == (1, 2, POSITIONS)

    recorder.arm(frames=1)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])
    everything = recorder.take(levels=3, worlds="all", selection="ranked")
    assert everything.world_ids.tolist() == [1, 0, 2]
    assert everything.level.tolist() == [1, 2, 3]
    assert (everything.rank.tolist(), everything.ranked_worlds) == ([1, 2, 3], 3)

    # Asking for more worlds than there are keeps them all.
    recorder.arm(frames=1)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])
    more = recorder.take(levels=3, worlds=128, selection="first")
    assert more.world_ids.tolist() == [1, 0, 2]


def test_only_counted_worlds_add_to_their_score():
    recorder, diagnostics = make_recorder()
    recorder.arm(frames=2)
    recorder.start_window()
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]])
    # World 1's first episode has ended: its later rewards do not count.
    counted = torch.tensor([True, False, True])
    take_steps(recorder, diagnostics, [[0.5, 2.0, -1.0]], 1, counted)
    window = recorder.take(levels=1, worlds="all", selection="ranked")
    assert window.world_ids.tolist() == [1, 0, 2]
    np.testing.assert_allclose(window.score, [2.0, 1.0, -2.0])
