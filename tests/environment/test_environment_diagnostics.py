"""Tests for the environment's diagnostics, with hand-made states.

Two worlds of a two-segment body. Rewards come from a hand-made StepRewards, so
these tests cover only what the diagnostics add.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from centipede.diagnostics_category import descriptions
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.reward_function import StepRewards
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.diagnostics import SimulationDiagnostics

TARGET = torch.tensor([[0.020, 0.0], [0.0, 0.010]])  # ahead of world 0, left of 1
RANGE = torch.tensor([0.0375, 0.0275])  # 2.5 x each head's start distance
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
    return StepRewards(
        parts.sum(dim=-1), parts, torch.full((2, 2), 0.001), torch.full((2, 2), 0.04)
    )


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
    after.spine_yaw_position[:, 0] = -0.1  # bent the other way

    record(diagnostics, after, before, step_rewards(-0.2))
    facts = diagnostics.step

    assert torch.equal(facts.reward_parts, torch.full((2, 2, 2), -0.1))
    assert facts.contact_flags[1, 1].tolist() == [True, False, False, False]
    assert facts.contact_flags.sum() == 1
    assert facts.segment_moved[:, 0].tolist() == pytest.approx([0.003, 0.003])
    assert torch.allclose(facts.joint_movement, torch.full((2, 2), 0.2))  # root
    assert facts.spine_bend[:, 0].tolist() == pytest.approx([0.1, 0.1])
    assert torch.allclose(facts.body_height, torch.full((2, 2), 0.004))
    assert facts.uprightness.tolist() == [[1.0, 1.0], [1.0, -1.0]]
    assert facts.head_distance.tolist() == pytest.approx(
        [0.015, math.hypot(0.005, 0.01)]
    )
    assert facts.heading_error.tolist() == pytest.approx(
        [0.0, math.atan2(0.010, -0.005)]
    )
    assert set(descriptions(facts)) == set(vars(facts))


def test_step_facts_compare_the_legs_movements_and_see_the_support():
    diagnostics = make_diagnostics()
    before = upright_state()
    diagnostics.start_episodes(torch.tensor([True, True]), before, TARGET, RANGE)
    after = upright_state()
    # World 0: the head's legs swing forward together, the next segment's
    # alternate. World 1: both segments lift their legs, by different amounts.
    after.leg_joint_position[0] = torch.tensor(
        [[0.1, 0.0, 0.0, 0.1, 0.0, 0.0], [0.1, 0.0, 0.0, -0.1, 0.0, 0.0]]
    )
    after.leg_joint_position[1] = torch.tensor(
        [[0.0, 0.2, 0.0, 0.0, 0.2, 0.0], [0.0, 0.1, 0.0, 0.0, 0.1, 0.0]]
    )
    # World 0 stands on a left and a right foot, world 1 on left feet only.
    after.left_foot_ground_contact[:, 0] = True
    after.right_foot_ground_contact[0, 1] = True

    record(diagnostics, after, before, step_rewards(0.0))
    facts = diagnostics.step

    assert facts.support.tolist() == [True, False]
    assert torch.allclose(
        facts.left_right_similarity, torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    )
    assert torch.allclose(facts.neighbour_leg_similarity, torch.tensor([[0.0], [1.0]]))

    record(diagnostics, after, after, step_rewards(0.0))  # nothing moves
    assert facts.left_right_similarity.abs().max() < 1e-6  # still: 0, not NaN


def test_the_rhythm_follows_the_clocks_and_the_legs_costs():
    clocks = SimpleNamespace(
        phase=torch.tensor([[0.5, 0.2], [0.1, 2 * math.pi - 0.1]]),
        tempo_hz=torch.tensor([[2.0, 2.5], [1.0, 4.0]]),
    )
    simulation = SimulationDiagnostics.allocate(2, 4, "cpu")
    diagnostics = EnvironmentDiagnostics(
        2, 2, ["first", "second"], "cpu", simulation.facts, clocks=clocks
    )
    state = upright_state()
    rewards = replace(
        step_rewards(0.0),
        legs_off_tempo=torch.tensor([[0.25, 0.0], [1.0, 0.5]]),
        foot_slip=torch.full((2, 2), 0.1),
    )

    record(diagnostics, state, state, rewards)
    rhythm = diagnostics.rhythm

    assert torch.equal(rhythm.tempo, clocks.tempo_hz)
    # Positive when the rear lags, within half a turn either way.
    lags = torch.tensor([[0.0, 0.3], [0.0, 0.2]])
    assert torch.allclose(rhythm.neighbour_offset, lags[:, 1:], atol=1e-6)
    assert torch.allclose(rhythm.head_offset, lags, atol=1e-6)
    assert rhythm.legs_on_tempo.tolist() == [[0.75, 1.0], [0.0, 0.5]]
    assert torch.equal(rhythm.foot_slip, rewards.foot_slip)
    assert make_diagnostics().rhythm is None  # without clocks


def test_an_ended_episode_publishes_its_totals_and_a_new_one_starts_clean():
    diagnostics = make_diagnostics()
    state = upright_state()
    diagnostics.start_episodes(torch.tensor([True, True]), state, TARGET, RANGE)
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

    diagnostics.start_episodes(torch.tensor([False, True]), moved, TARGET, RANGE)
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
    new_ranges = torch.tensor([0.1125, 0.1375])
    diagnostics.start_episodes(
        torch.tensor([True, False]), state, new_targets, new_ranges
    )

    assert torch.equal(
        diagnostics.step.target_position, torch.tensor([[0.050, 0.0], [0.0, 0.010]])
    )
    assert diagnostics.step.range_radius.tolist() == pytest.approx([0.1125, 0.0])
    assert diagnostics.simulation.qpos.shape == (2, 4)
