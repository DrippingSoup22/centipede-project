from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicalState:
    """The physical state of every world, handed from the simulation upward.

    Shapes use W for the number of worlds and N for the number of segments; see
    the physical state table in docs/environment.md for each field's meaning.
    The first ten fields form each segment's observation block, in order; the
    next three are used only for rewards and targets. The spine fields describe,
    for each segment, the yaw joint that joins it to the segment behind; the
    rear segment has none, and its entries stay zero.

    The tensors are created once and overwritten in place by every ``step()``
    and ``reset()``. After either returns, they describe the current state of
    every world until the next call. Readers never write to them and copy any
    value they need to keep longer.
    """

    body_height: torch.Tensor  # (W, N)
    body_quaternion: torch.Tensor  # (W, N, 4), (w, x, y, z)
    leg_joint_position: torch.Tensor  # (W, N, 6), action order
    body_linear_velocity: torch.Tensor  # (W, N, 3), segment frame
    body_angular_velocity: torch.Tensor  # (W, N, 3), segment frame
    leg_joint_velocity: torch.Tensor  # (W, N, 6), action order
    left_foot_ground_contact: torch.Tensor  # (W, N), bool
    right_foot_ground_contact: torch.Tensor  # (W, N), bool
    body_ground_contact: torch.Tensor  # (W, N), bool
    leg_leg_contact: torch.Tensor  # (W, N), bool
    body_planar_position: torch.Tensor  # (W, N, 2), world x and y
    head_tip_position: torch.Tensor  # (W, 3), world position
    foot_planar_position: torch.Tensor  # (W, N, 2, 2), left and right, world x, y
    spine_yaw_position: torch.Tensor  # (W, N), joint behind the segment, rad
    spine_yaw_velocity: torch.Tensor  # (W, N), joint behind the segment, rad/s

    @classmethod
    def allocate(
        cls, world_count: int, segment_count: int, device: str
    ) -> "PhysicalState":
        """Create a physical state of zeros, to be filled in place by a backend."""

        def zeros(*trailing_shape: int, dtype: torch.dtype = torch.float32):
            return torch.zeros(
                (world_count, *trailing_shape), dtype=dtype, device=device
            )

        n = segment_count
        return cls(
            body_height=zeros(n),
            body_quaternion=zeros(n, 4),
            leg_joint_position=zeros(n, 6),
            body_linear_velocity=zeros(n, 3),
            body_angular_velocity=zeros(n, 3),
            leg_joint_velocity=zeros(n, 6),
            left_foot_ground_contact=zeros(n, dtype=torch.bool),
            right_foot_ground_contact=zeros(n, dtype=torch.bool),
            body_ground_contact=zeros(n, dtype=torch.bool),
            leg_leg_contact=zeros(n, dtype=torch.bool),
            body_planar_position=zeros(n, 2),
            head_tip_position=zeros(3),
            foot_planar_position=zeros(n, 2, 2),
            spine_yaw_position=zeros(n),
            spine_yaw_velocity=zeros(n),
        )
