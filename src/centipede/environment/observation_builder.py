import torch

from centipede.environment.head import target_seen_from_head
from centipede.environment.simulation import PhysicalState

# A segment's block, besides its leg joints' angles and speeds: its height,
# orientation (4), linear and angular velocity (3 each), and four contact flags.
BLOCK_VALUES_BESIDES_LEGS = 15
TARGET_VALUE_COUNT = 2
# With spine control: the angle and speed of the spine joint behind the segment.
SPINE_VALUE_COUNT = 2


def neighbour_table(
    segment_count: int, observation_radius: int, device: torch.device | str
) -> torch.Tensor:
    """The blocks each segment sees, ``(N, 2 r + 1)``, in observation order.

    Row ``i`` lists segment ``i`` itself, the segments ahead (toward the head,
    lower numbers), then the segments behind, nearest first. A neighbour that
    does not exist points to row ``segment_count``, a block of zeros.
    """
    rows = []
    for segment_index in range(segment_count):
        offsets = list(range(-1, -observation_radius - 1, -1)) + list(
            range(1, observation_radius + 1)
        )
        neighbours = [segment_index + offset for offset in offsets]
        rows.append(
            [segment_index]
            + [
                index if 0 <= index < segment_count else segment_count
                for index in neighbours
            ]
        )
    return torch.tensor(rows, dtype=torch.long, device=device)


class ObservationBuilder:
    """Builds every segment's observation from the physical state and targets.

    Each segment sees its own block of ``15 + 2 M`` values, with M its leg
    motors (27 for legs of three joints, 31 with ankles), the blocks of the
    segments up to ``observation_radius`` ahead and behind (nearest first),
    and two target values that are real only for the head; with
    ``spine_observed``, last, the angle and speed of the spine joint behind
    it (zero for the rear segment). A missing neighbour is a block of zeros.
    The layout is fixed in docs/environment.md.
    """

    def __init__(
        self,
        segment_count: int,
        observation_radius: int,
        world_count: int,
        device: torch.device | str,
        leg_motor_count: int,
        spine_observed: bool = False,
    ) -> None:
        """Build the neighbour table (``neighbour_table``) and the reusable
        block tensor, whose last row stays zero for missing neighbours."""
        self.world_count = world_count
        self.segment_count = segment_count
        self.leg_motor_count = leg_motor_count
        self.block_size = BLOCK_VALUES_BESIDES_LEGS + 2 * leg_motor_count
        self.spine_observed = spine_observed
        self.neighbour_table = neighbour_table(
            segment_count, observation_radius, device
        )

        # One block per segment plus the zero block; refilled in place by build.
        self.blocks = torch.zeros(
            (world_count, segment_count + 1, self.block_size),
            dtype=torch.float32,
            device=device,
        )
        self.observation_size = (
            (2 * observation_radius + 1) * self.block_size
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
        m = self.leg_motor_count
        # Each part's columns, in the documented order.
        own = self.blocks[:, :segment_count]
        own[..., 0] = physical_state.body_height
        own[..., 1:5] = physical_state.body_quaternion
        own[..., 5 : 5 + m] = physical_state.leg_joint_position
        own[..., 5 + m : 8 + m] = physical_state.body_linear_velocity
        own[..., 8 + m : 11 + m] = physical_state.body_angular_velocity
        own[..., 11 + m : 11 + 2 * m] = physical_state.leg_joint_velocity
        own[..., 11 + 2 * m] = physical_state.left_foot_ground_contact.float()
        own[..., 12 + 2 * m] = physical_state.right_foot_ground_contact.float()
        own[..., 13 + 2 * m] = physical_state.body_ground_contact.float()
        own[..., 14 + 2 * m] = physical_state.leg_leg_contact.float()

        # (W, N + 1, B) gathered by the (N, 2k + 1) table gives
        # (W, N, 2k + 1, B); its blocks are then laid end to end per segment.
        blocks = self.blocks
        neighbour_blocks = blocks[:, self.neighbour_table].flatten(start_dim=2)

        # The target as seen from the head, for the head only.
        target_values = blocks.new_zeros(
            (blocks.shape[0], segment_count, TARGET_VALUE_COUNT)
        )
        target_values[:, 0] = target_seen_from_head(physical_state, target_position)

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
