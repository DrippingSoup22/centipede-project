"""Tests for the physical state container."""

import dataclasses

import pytest
import torch

from centipede.environment.simulation.physical_state import PhysicalState

# Field name, shape after the world dimension (N = 12 segments), type.
EXPECTED_FIELDS = [
    ("body_height", (12,), torch.float32),
    ("body_quaternion", (12, 4), torch.float32),
    ("leg_joint_position", (12, 6), torch.float32),
    ("body_linear_velocity", (12, 3), torch.float32),
    ("body_angular_velocity", (12, 3), torch.float32),
    ("leg_joint_velocity", (12, 6), torch.float32),
    ("left_foot_ground_contact", (12,), torch.bool),
    ("right_foot_ground_contact", (12,), torch.bool),
    ("body_ground_contact", (12,), torch.bool),
    ("leg_leg_contact", (12,), torch.bool),
    ("body_planar_position", (12, 2), torch.float32),
    ("head_tip_position", (3,), torch.float32),
    ("spine_yaw_position", (12,), torch.float32),
    ("spine_yaw_velocity", (12,), torch.float32),
]


def test_allocate_creates_zeroed_fields_in_the_documented_order():
    """A segment count other than 8 shows nothing assumes the v1 body."""
    state = PhysicalState.allocate(world_count=5, segment_count=12, device="cpu")

    assert [field.name for field in dataclasses.fields(state)] == [
        name for name, _, _ in EXPECTED_FIELDS
    ]
    for name, shape, dtype in EXPECTED_FIELDS:
        tensor = getattr(state, name)
        assert tensor.shape == (5, *shape), name
        assert tensor.dtype == dtype, name
        assert tensor.device == torch.device("cpu"), name
        assert not tensor.any(), name


def test_fields_are_filled_in_place_but_never_replaced():
    state = PhysicalState.allocate(world_count=2, segment_count=8, device="cpu")

    state.leg_joint_position[1, 3, 4] = 0.5
    assert state.leg_joint_position[1, 3, 4] == 0.5
    assert not state.leg_joint_velocity.any()  # fields do not share memory

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.leg_joint_position = torch.ones(2, 8, 6)  # type: ignore[misc]
