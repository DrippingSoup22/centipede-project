"""Tests for the interaction loop's episode recorder, with hand-made categories.

Three worlds of a two-segment body with a position vector of two entries.
The recorder reads only the categories, so no simulation is needed.
"""

import numpy as np
import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.simulation.diagnostics import SimulationDiagnostics
from centipede.interaction_loop.recording import EpisodeRecorder

WORLDS, SEGMENTS, POSITIONS = 3, 2, 2


def make_recorder():
    simulation = SimulationDiagnostics.allocate(WORLDS, POSITIONS, "cpu")
    diagnostics = EnvironmentDiagnostics(
        WORLDS, SEGMENTS, 1e-3, "cpu", simulation.facts
    )
    return EpisodeRecorder(diagnostics), diagnostics


def take_step(recorder, diagnostics, step, started, rewards, counted=None):
    """A step with the worlds that start an episode on it and their rewards."""
    diagnostics.simulation.qpos.copy_(
        torch.arange(WORLDS * POSITIONS).reshape(WORLDS, POSITIONS) + 10 * step
    )
    # The score: the line paid, the units' mean.
    diagnostics.step.reward_parts[..., 0] = torch.tensor(rewards)[:, None]
    diagnostics.step.line_points.fill_(float(step))
    diagnostics.episode.episode_ended.copy_(torch.tensor(started))
    recorder.step_taken(counted)


def test_each_recorded_world_keeps_one_episode_from_its_own_start():
    recorder, diagnostics = make_recorder()
    take_step(recorder, diagnostics, 0, [True] * 3, [1.0] * 3)  # not armed
    assert not recorder.recording and not recorder.complete()

    recorder.arm(frames=2, worlds=2)
    take_step(recorder, diagnostics, 0, [False, True, False], [1.0, 1.0, 1.0])
    take_step(recorder, diagnostics, 1, [True, False, False], [1.0, 3.0, 1.0])
    assert not recorder.complete()  # world 1 is done, world 0 is not
    take_step(recorder, diagnostics, 2, [False, False, False], [2.0, 5.0, 1.0])
    assert recorder.complete()
    episodes = recorder.take()

    # World 1's episode (steps 0 and 1) scored 4, world 0's (1 and 2) 3; the
    # third world is not recorded.
    assert episodes.world_ids.tolist() == [1, 0]
    np.testing.assert_allclose(episodes.score, [4.0, 3.0])
    np.testing.assert_allclose(episodes.qpos[:, :, 0], [[2.0, 10.0], [12.0, 20.0]])
    assert episodes.episode_start.tolist() == [[True, True], [False, False]]
    np.testing.assert_allclose(
        episodes.kept["line_points"][..., 0, 0], [[0.0, 1.0], [1.0, 2.0]]
    )
    assert set(episodes.kept) == {"line_points"}
    assert not recorder.recording


def test_from_a_reset_every_world_records_at_once_and_scores_while_counted():
    recorder, diagnostics = make_recorder()
    recorder.arm(frames=3, worlds=8, at_next_episode=False)  # all three worlds
    take_step(recorder, diagnostics, 0, [False] * 3, [0.5, 2.0, -1.0])
    # World 1's first episode has ended: its later rewards do not count.
    counted = torch.tensor([True, False, True])
    take_step(recorder, diagnostics, 1, [False] * 3, [0.5, 2.0, -1.0], counted)

    episodes = recorder.take()  # the two frames taken so far
    assert episodes.qpos.shape == (2, 3, POSITIONS)
    assert episodes.world_ids.tolist() == [1, 0, 2]
    np.testing.assert_allclose(episodes.score, [2.0, 1.0, -2.0])
