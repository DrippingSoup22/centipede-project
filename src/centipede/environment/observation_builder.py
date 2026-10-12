import torch

from centipede.environment.simulation import PhysicalState

# A segment's block holds, for its M leg joints: its body's height (1),
# orientation (4), leg angles (M), linear and angular velocities (3 + 3), leg
# speeds (M), and four contact flags; with leg accelerations, its M leg
# joints' accelerations at its end.
BODY_VALUE_COUNT = 1 + 4 + 3 + 3 + 4
# With tilt, the orientation's 4 values become 3; with the feet's physics, the
# 4 contact flags become 5.
TILT_VALUE_COUNT = 3
FEET_VALUE_COUNT = 5
TARGET_VALUE_COUNT = 2
# With spine control: the angle and speed of the spine joint behind the segment.
SPINE_VALUE_COUNT = 2


def head_forward_direction(head_quaternion: torch.Tensor) -> torch.Tensor:
    """The head's forward direction on the ground, as unit vectors ``(W, 2)``.

    ``head_quaternion`` is ``(W, 4)`` in MuJoCo's ``(w, x, y, z)`` order. The
    head's forward axis is its body's x axis; turned into the world, its flat
    ``x, y`` components are the first column of the rotation matrix. They are
    scaled to length 1; the length is kept above a small minimum so that a head
    pointing straight up or down gives a finite direction instead of NaN.
    """
    w, x, y, z = head_quaternion.unbind(dim=-1)
    forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), dim=-1)
    length = forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return forward / length


class ObservationBuilder:
    """Builds every segment's observation from the physical state and targets.

    Each segment sees its own block of 27 values, for legs of three joints
    (31 with ankles; 33 or 39 with ``leg_acceleration_observed``, which ends
    every block with its segment's leg joints' accelerations), the blocks of
    the segments up to
    ``observation_radius`` ahead and behind (nearest first), and two target
    values that are real only for the head; with ``spine_observed``, the
    angle and speed of the spine joint behind it (zero for the rear segment);
    with clocks, the ``clock_value_count`` values of its own clocks (three for
    a clock per segment, six for one per leg); with
    ``neighbour_clock_observed``, as many values for the clocks of each
    neighbour it sees, in the blocks' order; with ``step_shape_observed``,
    its own legs' step shape in use: each leg's sweep and lift amplitudes and
    its joints' centres; and with ``neighbour_tempo_observed``, last, the
    tempo of each neighbour it sees, in the blocks' order (cycle 3,
    2026-10-13: last, so that a run can start from one saved without them).
    A missing neighbour is a block of zeros. The layout is fixed in
    docs/environment.md.

    Cycle 2 (2026-10-12): with ``tilt_observed`` a block's orientation in
    the world (4 values, whose heading means nothing for walking) becomes
    its tilt, the world's vertical in the segment's frame (x and y), and its
    bend from the segment ahead, the spine joint between them (0 for the
    head); with ``feet_physics_observed`` its four contact flags become its
    feet's physics: each foot's load, each foot's sliding and its support
    over the stride (``Environment._feet_values``).
    """

    def __init__(
        self,
        segment_count: int,
        observation_radius: int,
        world_count: int,
        device: torch.device | str,
        spine_observed: bool = False,
        clock_value_count: int = 0,
        neighbour_clock_observed: bool = False,
        step_shape_observed: bool = False,
        leg_acceleration_observed: bool = False,
        leg_joint_count: int = 3,
        feet_physics_observed: bool = False,
        tilt_observed: bool = False,
        neighbour_tempo_observed: bool = False,
    ) -> None:
        """Build the neighbour table and the reusable block tensor.

        Row ``i`` of the neighbour table lists the blocks segment ``i`` sees, in
        observation order: itself, the segments ahead (toward the head, lower
        numbers), then the segments behind. A neighbour that does not exist
        points to row ``segment_count`` of the block tensor, which stays zero.
        """
        self.world_count = world_count
        self.segment_count = segment_count
        self.spine_observed = spine_observed
        self.clock_value_count = clock_value_count
        self.neighbour_clock_observed = neighbour_clock_observed
        self.step_shape_observed = step_shape_observed
        self.leg_acceleration_observed = leg_acceleration_observed
        self.feet_physics_observed = feet_physics_observed
        self.tilt_observed = tilt_observed
        self.neighbour_tempo_observed = neighbour_tempo_observed
        leg_motor_count = 2 * leg_joint_count
        self.leg_motor_count = leg_motor_count
        self.orientation_count = TILT_VALUE_COUNT if tilt_observed else 4
        self.contact_count = FEET_VALUE_COUNT if feet_physics_observed else 4
        block_size = (
            BODY_VALUE_COUNT
            - 4
            + self.orientation_count
            - 4
            + self.contact_count
            + 2 * leg_motor_count
            + (leg_motor_count if leg_acceleration_observed else 0)
        )
        missing_neighbour = segment_count

        neighbour_rows = []
        for segment_index in range(segment_count):
            ahead = [
                segment_index - offset for offset in range(1, observation_radius + 1)
            ]
            behind = [
                segment_index + offset for offset in range(1, observation_radius + 1)
            ]
            row = [segment_index] + [
                neighbour_index
                if 0 <= neighbour_index < segment_count
                else missing_neighbour
                for neighbour_index in ahead + behind
            ]
            neighbour_rows.append(row)
        self.neighbour_table = torch.tensor(
            neighbour_rows, dtype=torch.long, device=device
        )

        # One block per segment plus the zero block; refilled in place by build.
        self.blocks = torch.zeros(
            (world_count, segment_count + 1, block_size),
            dtype=torch.float32,
            device=device,
        )
        self.observation_size = (
            (2 * observation_radius + 1) * block_size
            + TARGET_VALUE_COUNT
            + (SPINE_VALUE_COUNT if spine_observed else 0)
            + clock_value_count
            + (
                2 * observation_radius * clock_value_count
                if neighbour_clock_observed
                else 0
            )
            + (2 * (2 + leg_joint_count) if step_shape_observed else 0)
            + (2 * observation_radius if neighbour_tempo_observed else 0)
        )

    def build(
        self,
        physical_state: PhysicalState,
        target_position: torch.Tensor,
        clock_values: torch.Tensor | None = None,
        neighbour_clock_values: torch.Tensor | None = None,
        step_shape: torch.Tensor | None = None,
        leg_acceleration: torch.Tensor | None = None,
        feet_values: torch.Tensor | None = None,
        neighbour_tempos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Every segment's observation, ``(W, N, observation_size)``.

        ``target_position`` is each world's target as a flat ``(W, 2)`` point;
        ``clock_values`` ``(W, N, c)`` each segment's own clocks, needed only
        with clocks, for ``c`` = ``clock_value_count``;
        ``neighbour_clock_values`` ``(W, N, 2k, c)`` its neighbours' clocks
        relative to its own, needed only with ``neighbour_clock_observed``;
        ``step_shape`` ``(W, N, 2, 2 + J)`` its legs' step shape in use,
        needed only with ``step_shape_observed``; ``leg_acceleration``
        ``(W, N, M)`` each segment's leg joints' accelerations, needed only with
        ``leg_acceleration_observed``; ``feet_values`` ``(W, N, 5)`` its feet's
        physics, needed only with ``feet_physics_observed``;
        ``neighbour_tempos`` ``(W, N, 2k)`` its neighbours' tempos, needed only
        with ``neighbour_tempo_observed``.
        The blocks are refilled in place, then gathered in neighbour-table
        order and laid end to end; the gather creates a new tensor, so a
        returned observation never changes when ``build`` runs again.
        """
        segment_count = self.segment_count
        legs = self.leg_motor_count
        blocks = self.blocks[:, :segment_count]
        blocks[..., 0] = physical_state.body_height
        angles = 1 + self.orientation_count
        if self.tilt_observed:
            w, x, y, z = physical_state.body_quaternion.unbind(dim=-1)
            # The world's z axis in the segment's frame: the rotation
            # matrix's third row; its x and y say how the segment tilts.
            blocks[..., 1] = 2 * (x * z - w * y)
            blocks[..., 2] = 2 * (y * z + w * x)
            blocks[..., 3] = 0.0
            blocks[:, 1:, 3] = physical_state.spine_yaw_position[:, :-1]
        else:
            blocks[..., 1:5] = physical_state.body_quaternion
        blocks[..., angles : angles + legs] = physical_state.leg_joint_position
        velocities = angles + legs
        blocks[..., velocities : velocities + 3] = physical_state.body_linear_velocity
        blocks[..., velocities + 3 : velocities + 6] = (
            physical_state.body_angular_velocity
        )
        speeds = velocities + 6
        blocks[..., speeds : speeds + legs] = physical_state.leg_joint_velocity
        contacts = speeds + legs
        if self.feet_physics_observed:
            blocks[..., contacts : contacts + FEET_VALUE_COUNT] = feet_values
        else:
            blocks[..., contacts] = physical_state.left_foot_ground_contact.float()
            blocks[..., contacts + 1] = physical_state.right_foot_ground_contact.float()
            blocks[..., contacts + 2] = physical_state.body_ground_contact.float()
            blocks[..., contacts + 3] = physical_state.leg_leg_contact.float()
        if self.leg_acceleration_observed:
            blocks[..., contacts + self.contact_count :] = leg_acceleration
        blocks = self.blocks

        # (W, N + 1, B) gathered by the (N, 2k + 1) table gives
        # (W, N, 2k + 1, B), with B the block size; its blocks are then laid
        # end to end per segment.
        neighbour_blocks = blocks[:, self.neighbour_table].flatten(start_dim=2)

        # The target as seen from the head: along its forward direction, and
        # along its left direction (forward turned 90 degrees anticlockwise).
        offset = target_position - physical_state.head_tip_position[:, :2]
        forward = head_forward_direction(physical_state.body_quaternion[:, 0])
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        target_values = blocks.new_zeros(
            (blocks.shape[0], segment_count, TARGET_VALUE_COUNT)
        )
        target_values[:, 0, 0] = (offset * forward).sum(dim=-1)
        target_values[:, 0, 1] = (offset * left).sum(dim=-1)

        parts = [neighbour_blocks, target_values]
        if self.spine_observed:
            parts.append(
                torch.stack(
                    (
                        physical_state.spine_yaw_position,
                        physical_state.spine_yaw_velocity,
                    ),
                    dim=-1,
                )
            )
        if self.clock_value_count:
            parts.append(clock_values)
        if self.neighbour_clock_observed:
            parts.append(neighbour_clock_values.flatten(start_dim=2))
        if self.step_shape_observed:
            parts.append(step_shape.flatten(start_dim=2))
        if self.neighbour_tempo_observed:
            parts.append(neighbour_tempos)
        return torch.cat(parts, dim=-1)
