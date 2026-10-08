from dataclasses import dataclass

from centipede.environment.simulation import SimulationSettings
from centipede.settings_section import SettingsError, SettingsSection


@dataclass(frozen=True)
class TargetSettings:
    """The [environment.target] section: where targets go and what counts as arrival.

    A new target is placed at a distance and bearing drawn uniformly from these
    ranges, measured from the head's tip and its forward direction; positive
    bearings are to the head's left. ``arrival`` is ``"head"`` when the target
    must lie under the head's outline, or ``"tip"`` when the head's tip must come
    within ``arrival_radius_m`` (None for ``"head"``). An episode is also cut,
    like a time limit, when the head's tip leaves the range circle: centred on
    the target, with ``range_circle_ratio`` times the distance at the start as
    its radius; 0 means no circle.

    The tip and no circle are the task of the runs before 2026-10-08's head
    arrival: a file that sets ``arrival_radius_m``, as their saved files do,
    uses the tip, and has no circle unless it sets one.
    """

    distance_range_m: tuple[float, float]
    bearing_range_deg: tuple[float, float]
    arrival: str
    arrival_radius_m: float | None
    range_circle_ratio: float | None

    @classmethod
    def from_section(cls, values: dict) -> "TargetSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.target")
        tip_task = "arrival_radius_m" in values
        arrival = section.choice(
            "arrival", ("head", "tip"), default="tip" if tip_task else "head"
        )
        if arrival == "head" and tip_task:
            raise SettingsError(
                '[environment.target] arrival_radius_m belongs to arrival = "tip";'
                ' with arrival = "head" the head\'s outline decides'
            )
        range_circle_ratio = section.number(
            "range_circle_ratio", default=None if tip_task else 2.5, minimum=0.0
        )
        if range_circle_ratio is not None and 0 < range_circle_ratio <= 1:
            raise SettingsError(
                "[environment.target] range_circle_ratio must be 0 (no circle) or"
                " above 1, or the head would start outside its circle, got"
                f" {range_circle_ratio:g}"
            )
        settings = cls(
            distance_range_m=section.number_range(
                "distance_range_m", default=(0.030, 0.060), minimum=0.0
            ),
            bearing_range_deg=section.number_range(
                "bearing_range_deg", default=(-30.0, 30.0)
            ),
            arrival=arrival,
            arrival_radius_m=(
                section.positive_number("arrival_radius_m", default=0.001)
                if arrival == "tip"
                else None
            ),
            range_circle_ratio=range_circle_ratio,
        )
        section.reject_unknown_keys()
        return settings


# The settings of the reward used before 2026-10-08: per-step weights, with
# efficiency as one term. A file that sets any of them uses that reward, so the
# runs saved before then keep their reward when continued or evaluated.
PER_STEP_WEIGHT_KEYS = ("efficiency_cost", "body_contact_cost", "leg_contact_cost")
PROPORTION_KEYS = (
    "step_cost_parts",
    "body_contact_cost_parts",
    "leg_contact_cost_parts",
    "cost_budget_parts",
    "head_progress_ratio",
    "follower_progress_share",
    "follower_progress_ratio",
)


@dataclass(frozen=True)
class RewardWeights:
    """The weight of each reward term, worked out from the settings.

    ``per_step`` holds the weights the reward function multiplies its terms
    by, in its order of terms; ``episode_shares`` the cost of each cost term
    over a whole episode, as a share of the arrival reward.
    """

    per_step: dict[str, float]
    episode_shares: dict[str, float]


@dataclass(frozen=True)
class RewardSettings:
    """The [environment.rewards] section: the reward, built from its rules.

    Everything is a proportion of ``arrival_reward``; docs/environment.md
    explains the rules. The worst-case cost budget, what a whole episode of
    every cost adds up to, is split into ``cost_budget_parts`` equal parts (by
    default the sum of the three costs' parts), and each cost takes its own
    number of parts. ``head_progress_ratio`` is what one halving of the head's
    distance to its target is worth, in whole episodes of step cost, and
    ``follower_progress_share`` the share of the head's progress that every
    follower receives as well (1: the same). A cost of zero parts is switched
    off.

    ``follower_progress_ratio`` belongs to the first runs of the rules, on
    2026-10-08, whose followers were paid for halving their distance to the spot
    where the segment ahead had been; it is None unless a file sets it, and in
    such a file ``follower_progress_share`` defaults to 0, so those runs read
    back with their reward.

    A file that sets the per-step weights of the reward used before 2026-10-08
    (``efficiency_cost``, ``body_contact_cost``, ``leg_contact_cost``) uses that
    reward instead; then the proportions are None, and otherwise those weights
    are. Unset values are left out of a run's saved configuration, so a saved
    run is read back with the same reward. ``distance_ratio_epsilon_m`` keeps the
    distance ratios defined when a distance reaches zero.
    """

    arrival_reward: float
    step_cost_parts: float | None
    body_contact_cost_parts: float | None
    leg_contact_cost_parts: float | None
    cost_budget_parts: float | None
    head_progress_ratio: float | None
    follower_progress_share: float | None
    follower_progress_ratio: float | None
    efficiency_cost: float | None
    body_contact_cost: float | None
    leg_contact_cost: float | None
    distance_ratio_epsilon_m: float

    @classmethod
    def from_section(cls, values: dict) -> "RewardSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "environment.rewards")
        arrival_reward = section.number("arrival_reward", default=1.0, minimum=0.0)
        epsilon = section.positive_number("distance_ratio_epsilon_m", default=1e-6)
        per_step = [key for key in PER_STEP_WEIGHT_KEYS if key in values]
        proportions = [key for key in PROPORTION_KEYS if key in values]
        if per_step and proportions:
            raise SettingsError(
                "[environment.rewards] mixes the per-step weights of the reward used"
                f" before 2026-10-08 ({', '.join(per_step)}) with the proportions of"
                f" the current one ({', '.join(proportions)}); use one or the other"
            )
        if per_step:
            settings = cls(
                arrival_reward=arrival_reward,
                step_cost_parts=None,
                body_contact_cost_parts=None,
                leg_contact_cost_parts=None,
                cost_budget_parts=None,
                head_progress_ratio=None,
                follower_progress_share=None,
                follower_progress_ratio=None,
                efficiency_cost=section.number(
                    "efficiency_cost", default=0.003, minimum=0.0
                ),
                body_contact_cost=section.number(
                    "body_contact_cost", default=0.010, minimum=0.0
                ),
                leg_contact_cost=section.number(
                    "leg_contact_cost", default=0.005, minimum=0.0
                ),
                distance_ratio_epsilon_m=epsilon,
            )
        else:
            parts = [
                section.number(key, default=default, minimum=0.0)
                for key, default in (
                    ("step_cost_parts", 2.0),
                    ("body_contact_cost_parts", 3.0),
                    ("leg_contact_cost_parts", 1.0),
                )
            ]
            budget_parts = section.number(
                "cost_budget_parts", default=sum(parts), minimum=0.0
            )
            if budget_parts < sum(parts) or budget_parts == 0:
                raise SettingsError(
                    "[environment.rewards] cost_budget_parts must be above zero and"
                    f" at least the costs' parts together ({sum(parts):g}), got"
                    f" {budget_parts:g}: the costs cannot take more than the budget"
                )
            own_goals = "follower_progress_ratio" in values
            settings = cls(
                arrival_reward=arrival_reward,
                step_cost_parts=parts[0],
                body_contact_cost_parts=parts[1],
                leg_contact_cost_parts=parts[2],
                cost_budget_parts=budget_parts,
                head_progress_ratio=section.number(
                    "head_progress_ratio", default=1.0, minimum=0.0
                ),
                follower_progress_share=section.number(
                    "follower_progress_share",
                    default=0.0 if own_goals else 1.0,
                    minimum=0.0,
                ),
                follower_progress_ratio=(
                    section.number("follower_progress_ratio", minimum=0.0)
                    if own_goals
                    else None
                ),
                efficiency_cost=None,
                body_contact_cost=None,
                leg_contact_cost=None,
                distance_ratio_epsilon_m=epsilon,
            )
        section.reject_unknown_keys()
        return settings

    @property
    def uses_per_step_weights(self) -> bool:
        """Whether this is the reward used before 2026-10-08."""
        return self.efficiency_cost is not None

    def weights(self, max_episode_steps: int) -> RewardWeights:
        """Each term's weight, for episodes of ``max_episode_steps`` steps.

        The cost budget is ``T × (2^(1/T) − 1)`` times the arrival reward, about
        ln 2: with the agents' discount at ``2^(−1/T)``, an episode that pays
        every cost on every step and arrives on its last step adds up to zero,
        seen from its start (rules R0 and R1 of docs/environment.md). Each cost
        takes its parts of the budget, spread evenly over the episode's steps.
        """
        arrival = self.arrival_reward
        if self.uses_per_step_weights:
            return RewardWeights(
                per_step={
                    "arrival": arrival,
                    "efficiency": self.efficiency_cost,
                    "body_contact": self.body_contact_cost,
                    "leg_contact": self.leg_contact_cost,
                },
                episode_shares={},
            )
        steps = max_episode_steps
        budget = steps * (2 ** (1 / steps) - 1)
        shares = {
            name: budget * parts / self.cost_budget_parts
            for name, parts in (
                ("step_cost", self.step_cost_parts),
                ("body_contact", self.body_contact_cost_parts),
                ("leg_contact", self.leg_contact_cost_parts),
            )
        }
        return RewardWeights(
            per_step={
                "arrival": arrival,
                # Per halving of the head's distance; the term itself gives the
                # followers their share of it.
                "progress": self.head_progress_ratio * shares["step_cost"] * arrival,
                "step_cost": shares["step_cost"] * arrival / steps,
                "body_contact": shares["body_contact"] * arrival / steps,
                "leg_contact": shares["leg_contact"] * arrival / steps,
            },
            episode_shares=shares,
        )


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
