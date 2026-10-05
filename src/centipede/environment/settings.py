from dataclasses import dataclass

from centipede.environment.simulation import SimulationSettings
from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class TargetSettings:
    """The [environment.target] section: where targets go and what counts as arrival.

    A new target is placed at a distance and bearing drawn uniformly from these
    ranges, measured from the head's tip and its forward direction; positive
    bearings are to the head's left.
    """

    distance_range_m: tuple[float, float]
    bearing_range_deg: tuple[float, float]
    arrival_radius_m: float

    @classmethod
    def from_section(cls, values: dict) -> "TargetSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.target")
        settings = cls(
            distance_range_m=section.number_range(
                "distance_range_m", default=(0.010, 0.020), minimum=0.0
            ),
            bearing_range_deg=section.number_range(
                "bearing_range_deg", default=(-15.0, 15.0)
            ),
            arrival_radius_m=section.positive_number("arrival_radius_m", default=0.001),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class RewardSettings:
    """The [environment.rewards] section: the weights of the reward terms.

    Arrival is the only positive term; the costs are subtracted. A weight of
    zero switches its term off.
    ``distance_ratio_epsilon_m`` keeps the efficiency ratios defined when a
    distance reaches zero.
    """

    arrival_reward: float
    efficiency_cost: float
    body_contact_cost: float
    leg_contact_cost: float
    distance_ratio_epsilon_m: float

    @classmethod
    def from_section(cls, values: dict) -> "RewardSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.rewards")
        settings = cls(
            arrival_reward=section.number("arrival_reward", default=1.0, minimum=0.0),
            efficiency_cost=section.number(
                "efficiency_cost", default=0.003, minimum=0.0
            ),
            body_contact_cost=section.number(
                "body_contact_cost", default=0.010, minimum=0.0
            ),
            leg_contact_cost=section.number(
                "leg_contact_cost", default=0.005, minimum=0.0
            ),
            distance_ratio_epsilon_m=section.positive_number(
                "distance_ratio_epsilon_m", default=1e-6
            ),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class EnvironmentSettings:
    """The [environment] section, with its three nested sections.

    ``observation_radius`` is how many neighbours a segment sees on each side;
    0 means it sees only itself.
    """

    max_episode_steps: int
    observation_radius: int
    simulation: SimulationSettings
    target: TargetSettings
    rewards: RewardSettings

    @classmethod
    def from_section(cls, values: dict) -> "EnvironmentSettings":
        """Check the section and hand each nested section to its own class.

        A nested section left out of the file keeps all its defaults, except
        that the simulation's model file and backend are always required.
        """
        section = SettingsSection(values, "environment")
        settings = cls(
            max_episode_steps=section.positive_integer(
                "max_episode_steps", default=8192
            ),
            observation_radius=section.integer(
                "observation_radius", default=1, minimum=0
            ),
            simulation=SimulationSettings.from_section(section.table("simulation")),
            target=TargetSettings.from_section(section.table("target")),
            rewards=RewardSettings.from_section(section.table("rewards")),
        )
        section.reject_unknown_keys()
        return settings
