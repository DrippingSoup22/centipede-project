"""Calculate local and shared rewards from one trusted transition.

Reward calculations operate only on the previous and current physical snapshots
plus task state supplied by the public environment. They do not inspect MuJoCo,
validate actions, decide episode boundaries, or maintain history themselves.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from centipede.environment.simulation import PhysicalSnapshot


@dataclass(frozen=True)
class RewardConfig:
    """First-version reward coefficients supplied to the reward calculation."""

    # Arrival is the only positive term; the remaining coefficients scale costs.
    c_arrival: float = 1.0
    c_efficiency: float = 0.003
    c_body: float = 0.010
    c_leg: float = 0.005
    # This small denominator offset keeps the distance ratio defined at zero.
    epsilon_ratio: float = 0.000001


# Complete per-agent reward equations:
#   head:       r_0 = A + E_0 - c_body * B_0 - c_leg * L_0
#   follower i: r_i = A + E_i - c_body * B_i - c_leg * L_i
# RewardTerms stores the four resulting signed contributions, so its total is
# their direct sum rather than another application of the coefficients.
@dataclass(frozen=True)
class RewardTerms:
    """Signed reward components calculated for one segment agent.

    ``arrival`` is shared and non-negative. The other three fields are
    non-positive contributions. Their sum is the scalar agent reward.
    """

    arrival: float
    efficiency: float
    body_contact: float
    leg_contact: float

    @property
    def total(self) -> float:
        """Return the scalar reward obtained by summing all four components."""
        return self.arrival + self.efficiency + self.body_contact + self.leg_contact


# Shared immutable baseline used unless an experiment supplies another config.
DEFAULT_REWARD_CONFIG = RewardConfig()


# Efficiency for both head progress and follower path progress:
#   E = -c_efficiency * (epsilon_ratio + distance_after)
#                       / (epsilon_ratio + distance_before)
def distance_ratio_cost(
    distance_before: float,
    distance_after: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Return the non-positive efficiency cost for one distance transition."""
    return (
        -config.c_efficiency
        * (config.epsilon_ratio + distance_after)
        / (config.epsilon_ratio + distance_before)
    )


def calculate_reward_terms(
    previous_snapshot: PhysicalSnapshot,
    current_snapshot: PhysicalSnapshot,
    target_position: NDArray[np.float64],
    target_reached: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict[int, RewardTerms]:
    """Calculate shared arrival and local costs for all segment agents.

    The head compares its target distance across the transition. Each follower
    compares its distance to the position occupied by its immediate predecessor
    at the start of the transition. Contact costs use the current snapshot.
    """
    rewards: dict[int, RewardTerms] = {}
    arrival = config.c_arrival if target_reached else 0.0

    for segment_id in range(len(previous_snapshot.body_height)):
        # The head follows the target; every other segment follows the point
        # occupied by its immediate predecessor before this transition.
        if segment_id == 0:
            destination = target_position
            position_before = previous_snapshot.head_tip_position[:2]
            position_after = current_snapshot.head_tip_position[:2]
        else:
            destination = previous_snapshot.body_planar_position[segment_id - 1]
            position_before = previous_snapshot.body_planar_position[segment_id]
            position_after = current_snapshot.body_planar_position[segment_id]

        distance_before = float(np.linalg.norm(position_before - destination))
        distance_after = float(np.linalg.norm(position_after - destination))
        efficiency = distance_ratio_cost(distance_before, distance_after, config)

        # Contact flags already identify the owner. Converting them here creates
        # signed local contributions while ordinary foot contact remains free.
        body_contact = -config.c_body * float(
            current_snapshot.body_ground_contact[segment_id]
        )
        leg_contact = -config.c_leg * float(
            current_snapshot.leg_leg_contact[segment_id]
        )

        rewards[segment_id] = RewardTerms(
            arrival=arrival,
            efficiency=efficiency,
            body_contact=body_contact,
            leg_contact=leg_contact,
        )

    return rewards
