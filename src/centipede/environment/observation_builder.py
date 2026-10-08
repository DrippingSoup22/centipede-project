import torch

from centipede.environment.simulation import PhysicalState

BLOCK_SIZE = 27
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

    Each segment sees its own block of 27 values, the blocks of the segments up
    to ``observation_radius`` ahead and behind (nearest first), and two target
    values that are real only for the head; with ``spine_observed``, last, the
    angle and speed of the spine joint behind it (zero for the rear segment).
    A missing neighbour is a block of zeros. The layout is fixed in
    docs/environment.md.
    """

    def __init__(
        self,
        segment_count: int,
        observation_radius: int,
        world_count: int,
        device: torch.device | str,
        spine_observed: bool = False,
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
            (world_count, segment_count + 1, BLOCK_SIZE),
            dtype=torch.float32,
            device=device,
        )
        self.observation_size = (
            (2 * observation_radius + 1) * BLOCK_SIZE
            + TARGET_VALUE_COUNT
            + (SPINE_VALUE_COUNT if spine_observed else 0)
        )

    def build(
        self, physical_state: PhysicalState, target_position: torch.Tensor
    ) -> torch.Tensor:
        """Every segment's observation, ``(W, N, observation_size)``.

        ``target_position`` is each world's target as a flat ``(W, 2)`` point.
        The blocks are refilled in place, then gathered in neighbour-table
        order and laid end to end; the gather creates a new tensor, so a
        returned observation never changes when ``build`` runs again.
        """
        segment_count = self.segment_count
        blocks = self.blocks
        blocks[:, :segment_count, 0] = physical_state.body_height
        blocks[:, :segment_count, 1:5] = physical_state.body_quaternion
        blocks[:, :segment_count, 5:11] = physical_state.leg_joint_position
        blocks[:, :segment_count, 11:14] = physical_state.body_linear_velocity
        blocks[:, :segment_count, 14:17] = physical_state.body_angular_velocity
        blocks[:, :segment_count, 17:23] = physical_state.leg_joint_velocity
        blocks[:, :segment_count, 23] = physical_state.left_foot_ground_contact.float()
        blocks[:, :segment_count, 24] = physical_state.right_foot_ground_contact.float()
        blocks[:, :segment_count, 25] = physical_state.body_ground_contact.float()
        blocks[:, :segment_count, 26] = physical_state.leg_leg_contact.float()

        # (W, N + 1, 27) gathered by the (N, 2k + 1) table gives
        # (W, N, 2k + 1, 27); its blocks are then laid end to end per segment.
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
        return torch.cat(parts, dim=-1)
