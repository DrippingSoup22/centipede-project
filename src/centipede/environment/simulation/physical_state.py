from dataclasses import dataclass

import torch

# The fields measured with ``measure_effort``, which the backends fill.
EFFORT_FIELDS = (
    "foot_vertical_force",
    "foot_forward_force",
    "belly_vertical_force",
    "floor_forward_force",
    "leg_leg_force",
    "leg_leg_force_squared",
    "motor_positive_work",
    "motor_negative_work",
    "motor_force",
)


@dataclass(frozen=True)
class PhysicalState:
    """The physical state of every world, handed from the simulation upward.

    Shapes use W for the number of worlds, N for the number of segments, and M
    for each segment's leg motors (six, or eight with ankles); see the physical
    state table in docs/environment.md for each field's meaning.
    The first ten fields form each segment's observation block, in order; the
    next two are used only for rewards and targets. The spine fields describe,
    for each segment, the yaw joint that joins it to the segment behind; the
    rear segment has none, and its entries stay zero.

    The tensors are created once and overwritten in place by every ``step()``
    and ``reset()``. After either returns, they describe the current state of
    every world until the next call. Readers never write to them and copy any
    value they need to keep longer.
    """

    body_height: torch.Tensor  # (W, N)
    body_quaternion: torch.Tensor  # (W, N, 4), (w, x, y, z)
    leg_joint_position: torch.Tensor  # (W, N, M), action order
    body_linear_velocity: torch.Tensor  # (W, N, 3), segment frame
    body_angular_velocity: torch.Tensor  # (W, N, 3), segment frame
    leg_joint_velocity: torch.Tensor  # (W, N, M), action order
    left_foot_ground_contact: torch.Tensor  # (W, N), bool
    right_foot_ground_contact: torch.Tensor  # (W, N), bool
    body_ground_contact: torch.Tensor  # (W, N), bool
    leg_leg_contact: torch.Tensor  # (W, N), bool
    body_planar_position: torch.Tensor  # (W, N, 2), world x and y
    head_tip_position: torch.Tensor  # (W, 3), world position
    spine_yaw_position: torch.Tensor  # (W, N), joint behind the segment, rad
    spine_yaw_velocity: torch.Tensor  # (W, N), joint behind the segment, rad/s
    # Each segment's front end in the plane, world x and y: the head's tip,
    # then each segment's joint with the segment ahead.
    front_end_position: torch.Tensor  # (W, N, 2)
    # The effort of the last step, measured after each of its physics steps
    # when the simulation is asked to (``measure_effort``), zeros otherwise:
    # forces are means over the step, in newtons, work is added up, in joules.
    # A reset measures the forces of the starting pose, and no work.
    # The floor's force on each foot (left, right) and on the body shape, its
    # vertical part: the weight each carries. On the flat floor it is the
    # contact's normal force; on uneven ground a foot pressed against a
    # stone's side carries little of it (from 2026-10-24).
    foot_vertical_force: torch.Tensor  # (W, N, 2)
    foot_forward_force: torch.Tensor  # (W, N, 2), along the segment
    belly_vertical_force: torch.Tensor  # (W, N)
    # (W, N), the floor's force on every shape of the segment it touches (feet,
    # the legs' other parts, the body), along the segment's forward axis laid
    # flat on the floor
    floor_forward_force: torch.Tensor
    leg_leg_force: torch.Tensor  # (W, N), between the segment's and others' legs
    leg_leg_force_squared: torch.Tensor  # (W, N), its square's mean, N²
    motor_positive_work: torch.Tensor  # (W, N), leg motors and spine motor behind
    motor_negative_work: torch.Tensor  # (W, N), the same motors braking, positive
    # (W, N, M + 1), each leg motor's force and the spine motor's behind,
    # each as a share of its strength (its gear), a mean over the step; zero
    # for the rear segment's missing spine motor
    motor_force: torch.Tensor
    # Measured with ``measure_joint_forces``, zeros otherwise, as a mean over
    # the step: the force the segment ahead exerts on the segment through the
    # joint between them, along the segment's forward axis, in newtons.
    # Positive pulls it forward (it lags), negative is the segment pushing the
    # one ahead. The head has no segment ahead; its entries stay zero.
    front_joint_force: torch.Tensor  # (W, N)
    # The same measure, the net force both spine joints exert on the segment,
    # its legs included, along its forward axis and up, in newtons. Zero
    # means the segment carries itself; positive along, its neighbours pull
    # it forward (it lags); positive up, they hold it up (it hangs on them).
    net_joint_force: torch.Tensor  # (W, N, 2)

    @classmethod
    def allocate(
        cls, world_count: int, segment_count: int, leg_motor_count: int, device: str
    ) -> "PhysicalState":
        """Create a physical state of zeros, to be filled in place by a backend."""

        def zeros(*trailing_shape: int, dtype: torch.dtype = torch.float32):
            return torch.zeros(
                (world_count, *trailing_shape), dtype=dtype, device=device
            )

        n, m = segment_count, leg_motor_count
        return cls(
            body_height=zeros(n),
            body_quaternion=zeros(n, 4),
            leg_joint_position=zeros(n, m),
            body_linear_velocity=zeros(n, 3),
            body_angular_velocity=zeros(n, 3),
            leg_joint_velocity=zeros(n, m),
            left_foot_ground_contact=zeros(n, dtype=torch.bool),
            right_foot_ground_contact=zeros(n, dtype=torch.bool),
            body_ground_contact=zeros(n, dtype=torch.bool),
            leg_leg_contact=zeros(n, dtype=torch.bool),
            body_planar_position=zeros(n, 2),
            head_tip_position=zeros(3),
            spine_yaw_position=zeros(n),
            spine_yaw_velocity=zeros(n),
            front_end_position=zeros(n, 2),
            foot_vertical_force=zeros(n, 2),
            foot_forward_force=zeros(n, 2),
            belly_vertical_force=zeros(n),
            floor_forward_force=zeros(n),
            leg_leg_force=zeros(n),
            leg_leg_force_squared=zeros(n),
            motor_positive_work=zeros(n),
            motor_negative_work=zeros(n),
            motor_force=zeros(n, m + 1),
            front_joint_force=zeros(n),
            net_joint_force=zeros(n, 2),
        )
