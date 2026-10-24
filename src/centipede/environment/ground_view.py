"""The front unit's view of the ground ahead (docs/follow-line/observations.md).

Without a head, nothing senses the ground ahead of the body; until a head
with antennae does (the user, 2026-10-24), the front unit sees it inside its
sight sphere: the ground's height at points in a 90-degree cone straight
ahead, level, turning with its heading. Every other unit learns the ground
from the units ahead of it. The heights are read from the terrain's map, so
the front unit also sees behind a hill (accepted as a stand-in).
"""

import math

import torch

from centipede.environment.line import heading
from centipede.environment.simulation import Ground, PhysicalState

# The cone's directions, from the right to the left of the unit's heading,
# and its distances as shares of the sight radius, nearest first.
GROUND_VIEW_ANGLES_RAD = tuple(math.radians(a) for a in (-45.0, -22.5, 0.0, 22.5, 45.0))
GROUND_VIEW_REACH = (1 / 3, 2 / 3, 1.0)
GROUND_VIEW_VALUES = len(GROUND_VIEW_ANGLES_RAD) * len(GROUND_VIEW_REACH)


def ground_view(
    state: PhysicalState, ground: Ground, sight_radius_m: float
) -> torch.Tensor:
    """The front unit's view, ``(W, GROUND_VIEW_VALUES)`` m: at each point,
    distance by distance and, within one, from right to left, the ground's
    height minus the height of the unit's centre."""
    device = state.body_height.device
    centre = state.body_planar_position[:, 0]
    yaw = heading(state.body_quaternion[:, 0])
    angles = torch.tensor(GROUND_VIEW_ANGLES_RAD, device=device)
    reach = sight_radius_m * torch.tensor(GROUND_VIEW_REACH, device=device)
    directions = yaw[:, None] + angles[None, :]
    offsets = (
        reach[None, :, None, None]
        * torch.stack((directions.cos(), directions.sin()), dim=-1)[:, None]
    )
    points = centre[:, None, None] + offsets
    heights = ground.heights(points) - state.body_height[:, 0, None, None]
    return heights.flatten(start_dim=1)
