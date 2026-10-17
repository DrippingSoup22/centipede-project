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

    A file that sets ``arrival_radius_m`` uses the tip, and has no circle
    unless it sets one, as the earlier task did.
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


# The settings of the earlier efficiency reward: per-step weights, with
# efficiency as one term. A file that sets any of them uses that reward, so
# runs configured with it keep their reward when continued or evaluated.
PER_STEP_WEIGHT_KEYS = ("efficiency_cost", "body_contact_cost", "leg_contact_cost")
PROPORTION_KEYS = (
    "step_cost_parts",
    "body_contact_cost_parts",
    "leg_contact_cost_parts",
    "movement_cost_parts",
    "random_command_movement_deg",
    "command_cost_ratio",
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
    default the sum of the costs' parts), and each cost takes its own number of
    parts. The movement cost charges each segment for how far the joints it
    commands move in a step, squared, in units of
    ``random_command_movement_deg``: how far a joint moves in a step under
    random commands. It is off (0 parts) unless a file sets it. The command
    cost, also off unless set, lies outside the budget: each step it charges
    ``command_cost_ratio`` times the step cost's weight, times the segment's
    mean squared command (from 0 to 1). Gymnasium's Ant charges 0.5 times the
    sum of its 8 squared commands against a reward of 1 for every step it stays
    healthy: a ratio of 4. Being outside the budget, it breaks rule R1 when set.
    ``head_progress_ratio`` is what one halving of the head's distance to its
    target is worth, in whole episodes of step cost, and
    ``follower_progress_share`` the share of the head's progress that every
    follower receives as well (1: the same). A cost of zero parts is switched
    off.

    ``follower_progress_ratio`` pays each follower, as an earlier form of the
    reward did, for halving its own distance to the spot where the segment
    ahead had been; it is None unless a file sets it, and in such a file
    ``follower_progress_share`` defaults to 0, so runs configured with it read
    back with their reward.

    A file that sets the per-step weights of the earlier efficiency reward
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
    movement_cost_parts: float | None
    random_command_movement_deg: float | None
    command_cost_ratio: float | None
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
                movement_cost_parts=None,
                random_command_movement_deg=None,
                command_cost_ratio=None,
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
                    ("movement_cost_parts", 0.0),
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
                movement_cost_parts=parts[3],
                random_command_movement_deg=section.positive_number(
                    "random_command_movement_deg", default=25.0
                ),
                command_cost_ratio=section.number(
                    "command_cost_ratio", default=0.0, minimum=0.0
                ),
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
        """Whether this is the earlier efficiency reward."""
        return self.efficiency_cost is not None

    def weights(self, max_episode_steps: int) -> RewardWeights:
        """Each term's weight, for episodes of ``max_episode_steps`` steps.

        The cost budget is ``T × (2^(1/T) − 1)`` times the arrival reward, about
        ln 2: with the agents' discount at ``2^(−1/T)``, an episode that pays
        every cost on every step and arrives on its last step adds up to zero,
        seen from its start (rules R0 and R1 of docs/environment.md). Each cost
        takes its parts of the budget, spread evenly over the episode's steps;
        the movement cost appears only when it has parts.
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
                ("movement", self.movement_cost_parts),
            )
        }
        if not shares["movement"]:
            del shares["movement"]
        per_step = {
            "arrival": arrival,
            # Per halving of the head's distance; the term itself gives the
            # followers their share of it.
            "progress": self.head_progress_ratio * shares["step_cost"] * arrival,
            **{name: share * arrival / steps for name, share in shares.items()},
        }
        # Outside the budget, so not among the episode shares.
        if self.command_cost_ratio:
            per_step["command"] = self.command_cost_ratio * per_step["step_cost"]
        return RewardWeights(per_step=per_step, episode_shares=shares)


@dataclass(frozen=True)
class CooperatorsSettings:
    """The [environment.cooperators] section: the cooperators' task and reward,
    used with ``design = "cooperators"`` (docs/cooperators/task.md, reward.md).

    **Targets.** At level 0 a target lies between ``start_near_distance_m``
    and ``far_to_near_ratio`` times it from the head's tip, within
    ``start_bearing_deg`` either side of straight ahead; at level 1 its near
    end is ``final_near_body_lengths`` body lengths. Both ends grow in a
    straight line with the level, and the bearing opens to the full circle at
    the level where the near end clears a body length and a turning circle.
    A target stays until the head reaches it, and the next appears at once.
    **Level.** Each target's stopwatch of ``stopwatch_s`` tells whether it was
    reached in time; after each update cycle the level, from ``start_level``,
    moves by ``level_rate`` times the share reached in time over the last
    episode length, less ``level_set_point``, and stays between 0 and 1. With
    ``stagger_first_episodes`` each world's first episode after a full reset
    has a random length, so that the worlds restart at different steps.

    **Reward** (cycle 7 of docs/cooperators/reward.md), two channels for every
    segment. The goal, in millimetres: for the head, what it gains toward its
    target and, on arrival, ``arrival_approaches`` times the middle distance
    of the level's targets; for a follower, the head's line its tail erases,
    less two squared costs, each ``reference_pay_mm_per_s`` at its scale:
    the distance of its tail from the line it has not erased, against
    ``line_distance_scale_mm`` and counted up to ``line_distance_cap_mm``,
    and its front joint's force, pulled or pushed, against
    ``joint_force_scale`` unit weights. Health, in millijoules: energy, the
    pushing work, the braking work times ``braking_price`` and the force,
    each motor's squared command
    times its duration times ``force_price_w``, shared with the neighbours a
    segment sees, itself 40% and each side 30% at radius 2 with halving by
    ``energy_sharing_decay`` (``energy_sharing_weights``); pain, the squared
    force of its legs pressed by others over a leg's strongest push, priced
    as force times ``pain_price``; and hanging, what its neighbours hold it up
    with (the head's beyond ``head_free_support`` of its weight) and the
    floor's push on its body, in unit weights, priced as force times
    ``hanging_price``. The forces are averaged over ``averaging_s`` (an
    exponential average; 0 none). ``force_price`` chooses the follower's
    force cost (cycle 10): ``"front_joint"``, its front joint's force, which
    carries every segment's error behind it too; ``"own_ground"``, its
    own net ground force, the net force of its two joints on it plus its tow,
    which is its own error alone; or ``"own_joint"`` (cycle 11), that net
    force without the tow, what it passes to its neighbours, so that each
    segment brakes its own tow for free and none passes it to the head.

    **Tow** (cycle 10, docs/cooperators/tow.md): training wheels that fade. In
    training each segment is pushed forward along the floor by its world's
    tow level times its own weight; with ``tow_help_speed_mm_per_s`` (cycle
    12; 0: a constant push) the push shrinks with the segment's own forward
    speed over the step before, to none at and above that speed. A training
    world draws its level from ``tow_draw_floor`` of the tow's top (0 by
    default) to the top at every episode start; the experiment's mean-action
    worlds keep the top or none. The top starts at ``tow_start`` (0: no tow)
    and moves by ``tow_step`` after each update cycle once
    ``tow_window_steps`` steps have passed since it last moved: down while
    the mean-action worlds ``tow_fade_group`` names (``"towed"``, at the
    top, or ``"untowed"``) walk, their followers erasing more than
    ``tow_walk_mm_per_s`` of line each and their feet lifting off more than
    ``tow_lift_off_hz`` times a second, up otherwise, from 0 to
    ``tow_start``. Evaluations never tow. ``observe_tow`` gives every
    segment its world's tow level, and ``observe_instant_joint_forces``
    adds to each block its front and net joint forces of the step itself
    beside the averaged ones.

    **Spine.** Each spine motor receives ``spine_action_scale`` times its
    command, so that the whole command range spans the joint's: on v4 a
    command above about 0.35 holds it at its limit.

    **Shared reward** (cycle 13, docs/cooperators/reward.md): a step's gain
    reaches the whole chain, so its bill is shared too. Each follower's goal
    is ``goal_team_share`` the followers' mean goal and the rest its own (0:
    its own alone); the head keeps its own, the target only it sees.
    ``health_sharing`` ``"all"`` shares pain and hanging with the neighbours
    as energy is shared (``"energy"``: energy alone). Pain's weight rises
    evenly from 0 over the first ``pain_ramp_cycles`` update cycles
    (``pain_weight_at``; 0: full from the start), so that the legs can try
    wide sweeps before pain teaches their timing; evaluations count it in
    full.
    """

    start_near_distance_m: float
    final_near_body_lengths: float
    far_to_near_ratio: float
    start_bearing_deg: float
    stopwatch_s: float
    start_level: float
    level_set_point: float
    level_rate: float
    stagger_first_episodes: bool
    arrival_approaches: float
    braking_price: float
    force_price_w: float
    pain_price: float
    energy_sharing_decay: float
    reference_pay_mm_per_s: float
    line_distance_scale_mm: float
    line_distance_cap_mm: float
    joint_force_scale: float
    hanging_price: float
    head_free_support: float
    averaging_s: float
    force_price: str
    tow_start: float
    tow_step: float
    tow_window_steps: int
    tow_walk_mm_per_s: float
    tow_lift_off_hz: float
    tow_help_speed_mm_per_s: float
    tow_fade_group: str
    tow_draw_floor: float
    observe_tow: bool
    observe_instant_joint_forces: bool
    spine_action_scale: float
    goal_team_share: float
    health_sharing: str
    pain_ramp_cycles: int

    @classmethod
    def from_section(cls, values: dict) -> "CooperatorsSettings":
        """Check the section's values and fill in the defaults: the values
        agreed in docs/cooperators/."""
        section = SettingsSection(values, "environment.cooperators")
        settings = cls(
            start_near_distance_m=section.positive_number(
                "start_near_distance_m", default=0.020
            ),
            final_near_body_lengths=section.positive_number(
                "final_near_body_lengths", default=2.7
            ),
            far_to_near_ratio=section.number(
                "far_to_near_ratio", default=1.5, minimum=1.0
            ),
            start_bearing_deg=section.number(
                "start_bearing_deg", default=90.0, minimum=0.0, maximum=180.0
            ),
            stopwatch_s=section.positive_number("stopwatch_s", default=7.0),
            start_level=section.number(
                "start_level", default=0.0, minimum=0.0, maximum=1.0
            ),
            level_set_point=section.number(
                "level_set_point", default=0.10, minimum=0.0, maximum=1.0
            ),
            level_rate=section.number("level_rate", default=0.02, minimum=0.0),
            stagger_first_episodes=section.boolean(
                "stagger_first_episodes", default=True
            ),
            arrival_approaches=section.number(
                "arrival_approaches", default=1.0, minimum=0.0
            ),
            braking_price=section.number("braking_price", default=0.25, minimum=0.0),
            force_price_w=section.number("force_price_w", default=0.00198, minimum=0.0),
            pain_price=section.number("pain_price", default=1.0, minimum=0.0),
            energy_sharing_decay=section.number(
                "energy_sharing_decay", default=0.5, minimum=0.0, maximum=1.0
            ),
            reference_pay_mm_per_s=section.number(
                "reference_pay_mm_per_s", default=40.0, minimum=0.0
            ),
            line_distance_scale_mm=section.positive_number(
                "line_distance_scale_mm", default=3.0
            ),
            line_distance_cap_mm=section.positive_number(
                "line_distance_cap_mm", default=10.0
            ),
            joint_force_scale=section.positive_number("joint_force_scale", default=0.3),
            hanging_price=section.number("hanging_price", default=1.0, minimum=0.0),
            head_free_support=section.number(
                "head_free_support", default=0.5, minimum=0.0
            ),
            averaging_s=section.number("averaging_s", default=0.2, minimum=0.0),
            force_price=section.choice(
                "force_price",
                ("front_joint", "own_ground", "own_joint"),
                default="front_joint",
            ),
            tow_start=section.number(
                "tow_start", default=0.0, minimum=0.0, maximum=1.0
            ),
            tow_step=section.number("tow_step", default=0.02, minimum=0.0),
            tow_window_steps=section.positive_integer("tow_window_steps", default=256),
            tow_walk_mm_per_s=section.number(
                "tow_walk_mm_per_s", default=5.0, minimum=0.0
            ),
            tow_lift_off_hz=section.number("tow_lift_off_hz", default=1.0, minimum=0.0),
            tow_help_speed_mm_per_s=section.number(
                "tow_help_speed_mm_per_s", default=0.0, minimum=0.0
            ),
            tow_fade_group=section.choice(
                "tow_fade_group", ("towed", "untowed"), default="towed"
            ),
            tow_draw_floor=section.number(
                "tow_draw_floor", default=0.0, minimum=0.0, maximum=1.0
            ),
            observe_tow=section.boolean("observe_tow", default=False),
            observe_instant_joint_forces=section.boolean(
                "observe_instant_joint_forces", default=False
            ),
            spine_action_scale=section.number(
                "spine_action_scale", default=0.35, minimum=0.0, maximum=1.0
            ),
            goal_team_share=section.number(
                "goal_team_share", default=0.0, minimum=0.0, maximum=1.0
            ),
            health_sharing=section.choice(
                "health_sharing", ("energy", "all"), default="energy"
            ),
            pain_ramp_cycles=section.integer("pain_ramp_cycles", default=0, minimum=0),
        )
        section.reject_unknown_keys()
        return settings

    def pain_weight_at(self, cycle: int) -> float:
        """Pain's weight in update cycle ``cycle``, counted from 1: rising
        evenly from 0 over the first ``pain_ramp_cycles`` cycles, then 1;
        always 1 without a ramp."""
        if not self.pain_ramp_cycles:
            return 1.0
        return min(1.0, (cycle - 1) / self.pain_ramp_cycles)


@dataclass(frozen=True)
class EnvironmentSettings:
    """The [environment] section, with its nested sections.

    ``design`` chooses the task, observations and reward: ``"main"``, those of
    docs/environment.md, set by ``[environment.target]`` and
    ``[environment.rewards]``; or ``"cooperators"``, those of
    docs/cooperators/, set by ``[environment.cooperators]``, in which every
    segment but the rear always commands its spine joint.
    ``observation_radius`` is how many neighbours a segment sees on each side;
    0 means it sees only itself. With ``spine_control`` each segment but the
    rear also commands the spine joint behind it and observes its angle and
    speed; without it, as in every run saved before it existed, the spine
    motors receive zero.
    """

    max_episode_steps: int
    observation_radius: int
    spine_control: bool
    design: str
    simulation: SimulationSettings
    target: TargetSettings
    rewards: RewardSettings
    cooperators: CooperatorsSettings

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
            spine_control=section.boolean("spine_control", default=False),
            design=section.choice("design", ("main", "cooperators"), default="main"),
            simulation=SimulationSettings.from_section(section.table("simulation")),
            target=TargetSettings.from_section(section.table("target")),
            rewards=RewardSettings.from_section(section.table("rewards")),
            cooperators=CooperatorsSettings.from_section(section.table("cooperators")),
        )
        section.reject_unknown_keys()
        return settings
