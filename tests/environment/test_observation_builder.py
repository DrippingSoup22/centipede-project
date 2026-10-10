"""Tests for the observation builder, with hand-made physical states.

Expected observations are assembled here from the field order documented in
docs/environment.md, independently of the builder's slices and table.
"""

import math

import pytest
import torch

from centipede.environment.observation_builder import (
    ObservationBuilder,
    head_forward_direction,
)
from centipede.environment.simulation import PhysicalState

SEGMENT_COUNT, WORLD_COUNT = 4, 2
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def distinct_state(device) -> PhysicalState:
    """A state where every value of every block differs."""
    state = PhysicalState.allocate(WORLD_COUNT, SEGMENT_COUNT, device)
    generator = torch.Generator().manual_seed(0)
    for field in vars(state).values():
        if field.dtype == torch.bool:
            field.copy_(torch.rand(field.shape, generator=generator) > 0.5)
        else:
            field.copy_(torch.rand(field.shape, generator=generator))
    return state


def block(state, world_index, segment_index) -> torch.Tensor:
    """One segment's 27 values, in the documented order."""
    row = world_index, segment_index
    return torch.cat(
        [
            state.body_height[row].reshape(1),
            state.body_quaternion[row],
            state.leg_joint_position[row],
            state.body_linear_velocity[row],
            state.body_angular_velocity[row],
            state.leg_joint_velocity[row],
            torch.stack(
                [
                    state.left_foot_ground_contact[row],
                    state.right_foot_ground_contact[row],
                    state.body_ground_contact[row],
                    state.leg_leg_contact[row],
                ]
            ).float(),
        ]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("radius", [0, 1, 2])
def test_each_segment_sees_itself_then_ahead_then_behind(radius, device):
    builder = ObservationBuilder(SEGMENT_COUNT, radius, WORLD_COUNT, device)
    state = distinct_state(device)

    observations = builder.build(state, torch.zeros(WORLD_COUNT, 2, device=device))

    assert builder.observation_size == (2 * radius + 1) * 27 + 2
    assert observations.shape == (WORLD_COUNT, SEGMENT_COUNT, builder.observation_size)
    for world_index in range(WORLD_COUNT):
        for segment_index in range(SEGMENT_COUNT):
            ahead = [segment_index - offset for offset in range(1, radius + 1)]
            behind = [segment_index + offset for offset in range(1, radius + 1)]
            expected = [
                block(state, world_index, neighbour_index)
                if 0 <= neighbour_index < SEGMENT_COUNT
                else torch.zeros(27, device=device)
                for neighbour_index in [segment_index] + ahead + behind
            ]
            actual = observations[world_index, segment_index]
            assert torch.equal(actual[:-2], torch.cat(expected))

    kept = observations.clone()
    state.body_height.fill_(7.0)
    builder.build(state, torch.zeros(WORLD_COUNT, 2, device=device))
    assert torch.equal(observations, kept)  # a later build leaves it unchanged


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, -135.0])
def test_target_values_are_in_the_heads_frame_and_only_for_the_head(heading_deg):
    builder = ObservationBuilder(SEGMENT_COUNT, 1, WORLD_COUNT, "cpu")
    state = PhysicalState.allocate(WORLD_COUNT, SEGMENT_COUNT, "cpu")
    half_turn = math.radians(heading_deg) / 2
    state.body_quaternion[:] = torch.tensor(
        [math.cos(half_turn), 0.0, 0.0, math.sin(half_turn)]
    )
    state.head_tip_position[:] = torch.tensor([0.3, -0.2, 0.005])
    forward = torch.tensor([math.cos(2 * half_turn), math.sin(2 * half_turn)])
    left = torch.tensor([-forward[1], forward[0]])
    tip = torch.tensor([0.3, -0.2])
    # World 0: 15 mm straight ahead. World 1: 10 mm ahead and 5 mm to the right.
    targets = torch.stack([tip + 0.015 * forward, tip + 0.010 * forward - 0.005 * left])

    target_values = builder.build(state, targets)[..., -2:]

    assert torch.allclose(target_values[0, 0], torch.tensor([0.015, 0.0]), atol=1e-6)
    assert torch.allclose(target_values[1, 0], torch.tensor([0.010, -0.005]), atol=1e-6)
    assert not target_values[:, 1:].any()


def test_with_the_spine_each_segment_sees_its_own_joint_last():
    plain = ObservationBuilder(SEGMENT_COUNT, 1, WORLD_COUNT, "cpu")
    builder = ObservationBuilder(
        SEGMENT_COUNT, 1, WORLD_COUNT, "cpu", spine_observed=True
    )
    state = distinct_state("cpu")
    targets = torch.zeros(WORLD_COUNT, 2)

    observations = builder.build(state, targets)

    assert builder.observation_size == plain.observation_size + 2
    assert torch.equal(observations[..., :-2], plain.build(state, targets))
    assert torch.equal(observations[..., -2], state.spine_yaw_position)
    assert torch.equal(observations[..., -1], state.spine_yaw_velocity)


@pytest.mark.parametrize("clock_values", [3, 6])  # a clock per segment, or per leg
def test_with_neighbour_clocks_each_segment_sees_them_after_its_own_clocks(
    clock_values,
):
    plain = ObservationBuilder(SEGMENT_COUNT, 2, WORLD_COUNT, "cpu")
    builder = ObservationBuilder(
        SEGMENT_COUNT,
        2,
        WORLD_COUNT,
        "cpu",
        clock_value_count=clock_values,
        neighbour_clock_observed=True,
    )
    state = distinct_state("cpu")
    targets = torch.zeros(WORLD_COUNT, 2)
    own_clocks = torch.rand(WORLD_COUNT, SEGMENT_COUNT, clock_values)
    neighbour_clocks = torch.rand(WORLD_COUNT, SEGMENT_COUNT, 4, clock_values)

    observations = builder.build(state, targets, own_clocks, neighbour_clocks)

    neighbour_count = 4 * clock_values
    clock_count = clock_values + neighbour_count
    assert builder.observation_size == plain.observation_size + clock_count
    assert torch.equal(observations[..., :-clock_count], plain.build(state, targets))
    assert torch.equal(observations[..., -clock_count:-neighbour_count], own_clocks)
    assert torch.equal(
        observations[..., -neighbour_count:], neighbour_clocks.flatten(start_dim=2)
    )


def test_forward_direction_survives_a_head_pointing_straight_up():
    nose_up = torch.tensor([[math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0]])

    assert torch.isfinite(head_forward_direction(nose_up)).all()
