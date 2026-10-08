"""Tests for the environment's diagnostics, with hand-made states.

Two worlds of a two-segment body. Rewards come from a hand-made StepRewards, so
these tests cover only what the diagnostics add.
"""

import math

import pytest
import torch

from centipede.diagnostics_category import descriptions
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.reward_function import StepRewards
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.diagnostics import SimulationDiagnostics

TARGET = torch.tensor([[0.020, 0.0], [0.0, 0.010]])  # ahead of world 0, left of 1
NO = torch.tensor([False, False])


def upright_state() -> PhysicalState:
    state = PhysicalState.allocate(2, 2, "cpu")
    state.body_quaternion[..., 0] = 1.0
    state.body_planar_position[:, 1, 0] = -0.004
    state.head_tip_position[:, 0] = 0.005
    return state


def make_diagnostics(reward_part_names=("first", "second")) -> EnvironmentDiagnostics:
    simulation = SimulationDiagnostics.allocate(2, 4, "cpu")
    return EnvironmentDiagnostics(
        2, 2, list(reward_part_names), "cpu", simulation.facts
    )


def step_rewards(value: float) -> StepRewards:
    parts = torch.full((2, 2, 2), value / 2)
    return StepRewards(parts.sum(dim=-1), parts, torch.full((2, 2), 0.001))


def record(
    diagnostics,
    state,
    previous_state,
    rewards,
    terminated=NO,
    truncated=NO,
    left_range=NO,
):
    diagnostics.record_step(
        state,
        previous_state.body_planar_position.clone(),
        previous_state.head_tip_position[:, :2].clone(),
        TARGET,
        rewards,
        terminated,
        truncated,
        left_range,
    )


def test_step_facts_describe_the_last_step():
    diagnostics = make_diagnostics()
    before = upright_state()
    after = upright_state()
    after.body_planar_position[:, 0, 1] = 0.003  # head centre moves 3 mm sideways
    after.body_height[:] = 0.004
    after.left_foot_ground_contact[1, 1] = True
    after.body_quaternion[1, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # upside down

    record(diagnostics, after, before, step_rewards(-0.2))
    facts = diagnostics.step

    assert torch.equal(facts.reward_parts, torch.full((2, 2, 2), -0.1))
    assert facts.contact_flags[1, 1].tolist() == [True, False, False, False]
    assert facts.contact_flags.sum() == 1
    assert facts.segment_moved[:, 0].tolist() == pytest.approx([0.003, 0.003])
    assert torch.allclose(facts.body_height, torch.full((2, 2), 0.004))
    assert facts.uprightness.tolist() == [[1.0, 1.0], [1.0, -1.0]]
    assert facts.head_distance.tolist() == pytest.approx(
        [0.015, math.hypot(0.005, 0.01)]
    )
    assert facts.heading_error.tolist() == pytest.approx(
        [0.0, math.atan2(0.010, -0.005)]
    )
    assert set(descriptions(facts)) == set(vars(facts))


def test_an_ended_episode_publishes_its_totals_and_a_new_one_starts_clean():
    diagnostics = make_diagnostics()
    state = upright_state()
    diagnostics.start_episodes(torch.tensor([True, True]), state, TARGET)
    state.body_ground_contact[:, 0] = True
    record(diagnostics, state, state, step_rewards(-1.0))
    state.body_ground_contact[:] = False
    state.left_foot_ground_contact[:] = True
    moved = upright_state()
    moved.head_tip_position[:, 0] += 0.002

    record(
        diagnostics,
        moved,
        state,
        step_rewards(-1.0),
        terminated=torch.tensor([False, True]),
    )
    episode = diagnostics.episode

    assert episode.episode_ended.tolist() == [False, True]
    assert episode.arrived.tolist() == [False, True]
    assert episode.ending[1] == 0.5  # the middle of the "arrived" bin
    assert episode.length_steps[1] == 2
    assert episode.segment_return[1].tolist() == [-2.0, -2.0]
    assert episode.start_distance[1] == pytest.approx(math.hypot(0.005, 0.01))
    assert episode.distance_closed[1] == pytest.approx(
        1 - episode.final_distance[1] / episode.start_distance[1]
    )
    assert episode.head_path_length[1] == pytest.approx(0.002)
    assert episode.segment_total_progress[1].tolist() == pytest.approx([0.002, 0.002])
    assert episode.body_contact_share[1].tolist() == [0.5, 0.0]
    assert episode.foot_contact_share[1, :, 0].tolist() == [0.0, 0.0]
    assert episode.upside_down_share[1] == 0.0
    assert episode.length_steps[0] == 0  # world 0's episode is still running

    diagnostics.start_episodes(torch.tensor([False, True]), moved, TARGET)
    record(
        diagnostics,
        moved,
        moved,
        step_rewards(-1.0),
        truncated=torch.tensor([True, True]),
        left_range=torch.tensor([False, True]),
    )
    assert episode.length_steps.tolist() == [3.0, 1.0]
    assert episode.arrived.tolist() == [False, False]
    assert episode.left_range.tolist() == [False, True]
    # World 0 ran out of time 13 mm from a target 15 mm away at the start: closer.
    assert episode.ending.tolist() == [3.5, 5.5]


def test_the_target_follows_the_worlds_just_reset():
    diagnostics = make_diagnostics()
    state = upright_state()

    record(
        diagnostics,
        state,
        state,
        step_rewards(0.0),
        terminated=torch.tensor([True, False]),
    )
    assert torch.equal(diagnostics.step.target_position, TARGET)

    new_targets = torch.tensor([[0.050, 0.0], [0.060, 0.0]])
    diagnostics.start_episodes(torch.tensor([True, False]), state, new_targets)

    assert torch.equal(
        diagnostics.step.target_position, torch.tensor([[0.050, 0.0], [0.0, 0.010]])
    )
    assert diagnostics.simulation.qpos.shape == (2, 4)
