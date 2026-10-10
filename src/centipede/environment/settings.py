import math
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
    the target, with ``range_circle_ratio`` times the distance at the start,
    plus ``range_circle_margin_m``, as its radius; 0 means no circle. The
    margin is room to turn, which a near target behind the head needs.
    ``after_arrival`` is ``"restart"`` when an arrival restarts the world like
    any other end, or ``"new_target"`` when the body walks on from where it
    arrived, toward a new target; the other ends always restart the world.

    A file that sets ``arrival_radius_m`` uses the tip, and has no circle
    unless it sets one, as the earlier task did.
    """

    distance_range_m: tuple[float, float]
    bearing_range_deg: tuple[float, float]
    arrival: str
    arrival_radius_m: float | None
    range_circle_ratio: float | None
    range_circle_margin_m: float
    after_arrival: str

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
            range_circle_margin_m=section.number(
                "range_circle_margin_m", default=0.0, minimum=0.0
            ),
            after_arrival=section.choice(
                "after_arrival", ("restart", "new_target"), default="restart"
            ),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class ClockSettings:
    """The [environment.clock] section: the clocks, with ``clocks`` or ``leg_clocks``.

    A tempo action ``a``, from −1 to 1, sets its clock's tempo to
    ``middle_tempo_hz × 2^(a × tempo_range_octaves)`` turns per second: 1 to 4
    with the defaults. With leg clocks, ``load_feedback_rad_per_s`` (σ) holds a
    loaded foot's clock in stance: its hand turns at ``ω + σ N cos φ`` rad/s,
    ``ω`` the tempo's and ``N`` 1 while the foot touches the ground; 0 is off.
    σ must stay below the slowest tempo's ω, so that no hand can ever stop.
    With leg clocks too, each leg joint's centre follows its centre action
    with the time constant ``centre_time_constant_s`` (0.5 by default with leg
    clocks, else 0); 0 follows the actions at once. ``coupling`` says which
    neighbours' clocks a segment sees and is compared with: ``"both"``, those
    ahead and behind, or ``"ahead"``, only those ahead, so that the rhythm
    passes from the head backward.
    """

    middle_tempo_hz: float
    tempo_range_octaves: float
    load_feedback_rad_per_s: float
    centre_time_constant_s: float
    coupling: str

    @classmethod
    def from_section(cls, values: dict, leg_clocks: bool = False) -> "ClockSettings":
        """Check the section's values and fill in the defaults.

        ``leg_clocks`` is [environment]'s, which sets the centres' default.
        """
        section = SettingsSection(values, "environment.clock")
        settings = cls(
            middle_tempo_hz=section.positive_number("middle_tempo_hz", default=2.0),
            tempo_range_octaves=section.positive_number(
                "tempo_range_octaves", default=1.0
            ),
            load_feedback_rad_per_s=section.number(
                "load_feedback_rad_per_s", default=0.0, minimum=0.0
            ),
            centre_time_constant_s=section.number(
                "centre_time_constant_s",
                default=0.5 if leg_clocks else 0.0,
                minimum=0.0,
            ),
            coupling=section.choice("coupling", ("both", "ahead"), default="both"),
        )
        # A loaded hand turns at ω + σ cos φ: at least ω − σ, which must stay
        # above zero at the slowest tempo. A stopped hand would never move on,
        # since the foot leaves the ground only in swing.
        slowest = settings.middle_tempo_hz / 2**settings.tempo_range_octaves
        if settings.load_feedback_rad_per_s >= 2 * math.pi * slowest:
            raise SettingsError(
                "[environment.clock] load_feedback_rad_per_s must stay below 2π times"
                f" the slowest tempo, {2 * math.pi * slowest:.4g} rad/s: at or above"
                " it, a foot on the ground can stop its leg's clock in stance, and"
                " since the foot leaves the ground only in swing, the clock would"
                f" never move on; got {settings.load_feedback_rad_per_s:g}"
            )
        fastest = settings.middle_tempo_hz * 2**settings.tempo_range_octaves
        # Load feedback can hurry a hand by up to σ rad/s.
        fastest += settings.load_feedback_rad_per_s / (2 * math.pi)
        if fastest >= 12.5:
            raise SettingsError(
                "[environment.clock] the fastest tempo, load feedback included, must"
                " stay below 12.5 turns per second, a quarter of a turn per 20 ms"
                f" step, or a step could skip past half a turn; got {fastest:g}"
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
    "foot_slip_cost_parts",
    "legs_off_tempo_cost_parts",
    "out_of_tempo_cost_parts",
    "no_support_cost_parts",
    "movement_cost_parts",
    "random_command_movement_deg",
    "foot_slip_unit_m_per_s",
    "legs_off_tempo_unit_deg",
    "head_tempo_share",
    "command_cost_ratio",
    "cost_budget_parts",
    "head_progress_ratio",
    "progress_parts",
    "follower_progress_share",
    "follower_progress_ratio",
)
# The costs that need clocks; legs off tempo needs a clock per segment.
CLOCK_COST_KEYS = ("legs_off_tempo_cost_parts", "out_of_tempo_cost_parts")


@dataclass(frozen=True)
class RewardWeights:
    """The weight of each reward term, worked out from the settings.

    ``per_step`` holds the weights the reward function multiplies its terms
    by, in its order of terms; ``episode_shares`` the cost of each cost term
    over the cost horizon (a whole episode unless ``cost_horizon_steps`` is
    set), as a share of the arrival reward.
    """

    per_step: dict[str, float]
    episode_shares: dict[str, float]


@dataclass(frozen=True)
class RewardSettings:
    """The [environment.rewards] section: the reward, built from its rules.

    Everything is a proportion of ``arrival_reward``; docs/environment.md
    explains the rules. The worst-case cost budget, what every cost adds up to
    over the cost horizon, is split into ``cost_budget_parts`` equal parts (by
    default the sum of the costs' parts), and each cost takes its own number of
    parts. The cost horizon is ``cost_horizon_steps``, or a whole episode when
    None; a horizon shorter than the episode makes every cost and the progress
    weigh more against the arrival. The movement cost charges each segment for
    how far the joints it commands move in a step, squared, in units of
    ``random_command_movement_deg``: how far a joint moves in a step under
    random commands. It is off (0 parts) unless a file sets it. The command
    cost, also off unless set, lies outside the budget: each step it charges
    ``command_cost_ratio`` times the step cost's weight, times the segment's
    mean squared command (from 0 to 1). Gymnasium's Ant charges 0.5 times the
    sum of its 8 squared commands against a reward of 1 for every step it stays
    healthy: a ratio of 4. Being outside the budget, it breaks rule R1 when set.
    ``head_progress_ratio`` is what one halving of the head's distance to its
    target is worth, in step cost over the whole cost horizon; a file may give
    it instead as ``progress_parts``, parts of the budget, which a reward
    without a step cost needs. ``follower_progress_share`` is the share of the
    head's progress that every follower receives as well (1: the same), and
    ``follower_arrival_share`` the share of the arrival. ``arrival_payout`` is
    what an arrival pays, as a share of ``arrival_reward``, the unit the rules
    measure every cost in: below 1, the costs weigh more against the arrival.
    A cost of zero parts is switched off and left out of the terms. A budget
    split into fewer parts than the costs take makes them overspend it: the
    worst arrival then ends below zero.

    Four costs judge how a segment walks. Foot slip charges each foot that
    touches the ground at both ends of a step for how fast it slid along it,
    in units of ``foot_slip_unit_m_per_s``, and no support each step on which
    neither of the segment's feet touches the ground. Legs off tempo and out
    of tempo need clocks: the first, only with a clock per segment, charges
    how far the legs are from where they were when the clock last passed the
    same point of its turn, in units of ``legs_off_tempo_unit_deg``, and the
    second how far the segment's tempos are from those they are compared
    with, of which the head pays ``head_tempo_share``. With the clocks'
    ``coupling = "ahead"``, a segment is compared only with the clocks ahead
    of it, so the head pays nothing for its neighbours, and
    ``head_tempo_share`` is None.

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
    arrival_payout: float
    follower_arrival_share: float
    step_cost_parts: float | None
    body_contact_cost_parts: float | None
    leg_contact_cost_parts: float | None
    foot_slip_cost_parts: float | None
    legs_off_tempo_cost_parts: float | None
    out_of_tempo_cost_parts: float | None
    no_support_cost_parts: float | None
    movement_cost_parts: float | None
    random_command_movement_deg: float | None
    foot_slip_unit_m_per_s: float | None
    legs_off_tempo_unit_deg: float | None
    head_tempo_share: float | None
    command_cost_ratio: float | None
    cost_budget_parts: float | None
    cost_horizon_steps: int | None
    head_progress_ratio: float | None
    progress_parts: float | None
    follower_progress_share: float | None
    follower_progress_ratio: float | None
    efficiency_cost: float | None
    body_contact_cost: float | None
    leg_contact_cost: float | None
    distance_ratio_epsilon_m: float

    @classmethod
    def from_section(cls, values: dict, coupling: str = "both") -> "RewardSettings":
        """Check the section's values and fill in the defaults.

        ``coupling`` is [environment.clock]'s, which decides whether the head
        pays a share of its tempo mismatch.
        """
        section = SettingsSection(values, "environment.rewards")
        if coupling == "ahead" and "head_tempo_share" in values:
            raise SettingsError(
                "[environment.rewards] head_tempo_share sets the share of its tempo"
                ' mismatch the head pays; with [environment.clock] coupling = "ahead"'
                " every segment is compared only with the clocks ahead of it, so the"
                " head already pays nothing for its neighbours: leave it out"
            )
        arrival_reward = section.number("arrival_reward", default=1.0, minimum=0.0)
        arrival_payout = section.number("arrival_payout", default=1.0, minimum=0.0)
        follower_arrival_share = section.number(
            "follower_arrival_share", default=1.0, minimum=0.0
        )
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
                arrival_payout=arrival_payout,
                follower_arrival_share=follower_arrival_share,
                step_cost_parts=None,
                body_contact_cost_parts=None,
                leg_contact_cost_parts=None,
                foot_slip_cost_parts=None,
                legs_off_tempo_cost_parts=None,
                out_of_tempo_cost_parts=None,
                no_support_cost_parts=None,
                movement_cost_parts=None,
                random_command_movement_deg=None,
                foot_slip_unit_m_per_s=None,
                legs_off_tempo_unit_deg=None,
                head_tempo_share=None,
                command_cost_ratio=None,
                cost_budget_parts=None,
                cost_horizon_steps=None,
                head_progress_ratio=None,
                progress_parts=None,
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
                    ("foot_slip_cost_parts", 0.0),
                    ("legs_off_tempo_cost_parts", 0.0),
                    ("out_of_tempo_cost_parts", 0.0),
                    ("no_support_cost_parts", 0.0),
                    ("movement_cost_parts", 0.0),
                )
            ]
            budget_parts = section.number(
                "cost_budget_parts", default=sum(parts), minimum=0.0
            )
            if budget_parts == 0:
                raise SettingsError(
                    "[environment.rewards] cost_budget_parts must be above zero"
                )
            if "progress_parts" in values and "head_progress_ratio" in values:
                raise SettingsError(
                    "[environment.rewards] gives the progress twice: as"
                    " head_progress_ratio (in step cost) and as progress_parts"
                    " (in parts of the budget); use one"
                )
            own_goals = "follower_progress_ratio" in values
            settings = cls(
                arrival_reward=arrival_reward,
                arrival_payout=arrival_payout,
                follower_arrival_share=follower_arrival_share,
                step_cost_parts=parts[0],
                body_contact_cost_parts=parts[1],
                leg_contact_cost_parts=parts[2],
                foot_slip_cost_parts=parts[3],
                legs_off_tempo_cost_parts=parts[4],
                out_of_tempo_cost_parts=parts[5],
                no_support_cost_parts=parts[6],
                movement_cost_parts=parts[7],
                random_command_movement_deg=section.positive_number(
                    "random_command_movement_deg", default=25.0
                ),
                foot_slip_unit_m_per_s=section.positive_number(
                    "foot_slip_unit_m_per_s", default=0.010
                ),
                legs_off_tempo_unit_deg=section.positive_number(
                    "legs_off_tempo_unit_deg", default=20.0
                ),
                head_tempo_share=(
                    None
                    if coupling == "ahead"
                    else section.number("head_tempo_share", default=0.25, minimum=0.0)
                ),
                command_cost_ratio=section.number(
                    "command_cost_ratio", default=0.0, minimum=0.0
                ),
                cost_budget_parts=budget_parts,
                cost_horizon_steps=section.positive_integer(
                    "cost_horizon_steps", default=None
                ),
                head_progress_ratio=(
                    None
                    if "progress_parts" in values
                    else section.number("head_progress_ratio", default=1.0, minimum=0.0)
                ),
                progress_parts=section.number(
                    "progress_parts", default=None, minimum=0.0
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

        The rules are measured over ``H`` steps: ``cost_horizon_steps``, or the
        whole episode. With the agents' discount at ``2^(−1/T)``, an approach
        that pays every cost on every step and arrives on step ``H`` adds up to
        zero, seen from its start (rules R0 and R1 of docs/environment.md); for
        ``H = T`` the budget is ``T × (2^(1/T) − 1)`` times the arrival reward,
        about ln 2. Each cost takes its parts of the budget, spread evenly over
        the ``H`` steps; a cost with no parts is left out. The arrival pays
        ``arrival_payout`` of the arrival reward, and a halving of the head's
        distance ``progress_parts`` of the budget, or ``head_progress_ratio``
        times the step cost's share.
        """
        arrival = self.arrival_reward
        if self.uses_per_step_weights:
            return RewardWeights(
                per_step={
                    "arrival": arrival * self.arrival_payout,
                    "efficiency": self.efficiency_cost,
                    "body_contact": self.body_contact_cost,
                    "leg_contact": self.leg_contact_cost,
                },
                episode_shares={},
            )
        steps = self.cost_horizon_steps or max_episode_steps
        discount = 2 ** (-1 / max_episode_steps)
        every_cost_per_step = (
            discount ** (steps - 1) * (1 - discount) / (1 - discount**steps)
        )
        part = steps * every_cost_per_step / self.cost_budget_parts
        shares = {
            name: part * parts
            for name, parts in (
                ("step_cost", self.step_cost_parts),
                ("body_contact", self.body_contact_cost_parts),
                ("leg_contact", self.leg_contact_cost_parts),
                ("foot_slip", self.foot_slip_cost_parts),
                ("legs_off_tempo", self.legs_off_tempo_cost_parts),
                ("out_of_tempo", self.out_of_tempo_cost_parts),
                ("no_support", self.no_support_cost_parts),
                ("movement", self.movement_cost_parts),
            )
            if parts
        }
        progress_parts = (
            self.progress_parts
            if self.progress_parts is not None
            else self.head_progress_ratio * self.step_cost_parts
        )
        per_step = {
            "arrival": arrival * self.arrival_payout,
            # Per halving of the head's distance; the term itself gives the
            # followers their share of it.
            "progress": progress_parts * part * arrival,
            **{name: share * arrival / steps for name, share in shares.items()},
        }
        # Outside the budget, so not among the episode shares.
        if self.command_cost_ratio:
            per_step["command"] = (
                self.command_cost_ratio * self.step_cost_parts * part * arrival / steps
            )
        return RewardWeights(per_step=per_step, episode_shares=shares)


@dataclass(frozen=True)
class EnvironmentSettings:
    """The [environment] section, with its three nested sections.

    ``observation_radius`` is how many neighbours a segment sees on each side;
    0 means it sees only itself. With ``spine_control`` each segment but the
    rear also commands the spine joint behind it and observes its angle and
    speed; without it, as in every run saved before it existed, the spine
    motors receive zero. With ``passive_follower_spine`` as well, only the
    head commands a spine joint, its neck; the followers' joints bend
    passively against their springs, and every segment still observes the
    joint behind it. With ``clocks`` every segment has a clock, whose tempo it
    sets with one more action and whose hand it observes. With ``leg_clocks``
    every leg has a clock instead, whose hand drives the leg through a step
    shape; it needs a model whose legs take angles, which only the
    environment can check, once the model is loaded. With
    ``neighbour_clocks`` as well as either, a segment also observes the
    clocks of the neighbours it sees, relative to its own.
    """

    max_episode_steps: int
    observation_radius: int
    spine_control: bool
    passive_follower_spine: bool
    clocks: bool
    leg_clocks: bool
    neighbour_clocks: bool
    simulation: SimulationSettings
    target: TargetSettings
    rewards: RewardSettings
    clock: ClockSettings

    @classmethod
    def from_section(cls, values: dict) -> "EnvironmentSettings":
        """Check the section and hand each nested section to its own class.

        A nested section left out of the file keeps all its defaults, except
        that the simulation's model file and backend are always required.
        """
        section = SettingsSection(values, "environment")
        leg_clocks = section.boolean("leg_clocks", default=False)
        clock = ClockSettings.from_section(section.table("clock"), leg_clocks)
        settings = cls(
            max_episode_steps=section.positive_integer(
                "max_episode_steps", default=8192
            ),
            observation_radius=section.integer(
                "observation_radius", default=1, minimum=0
            ),
            spine_control=section.boolean("spine_control", default=False),
            passive_follower_spine=section.boolean(
                "passive_follower_spine", default=False
            ),
            clocks=section.boolean("clocks", default=False),
            leg_clocks=leg_clocks,
            neighbour_clocks=section.boolean("neighbour_clocks", default=False),
            simulation=SimulationSettings.from_section(section.table("simulation")),
            target=TargetSettings.from_section(section.table("target")),
            rewards=RewardSettings.from_section(
                section.table("rewards"), clock.coupling
            ),
            clock=clock,
        )
        section.reject_unknown_keys()
        if settings.passive_follower_spine and not settings.spine_control:
            raise SettingsError(
                "[environment] passive_follower_spine leaves the head its neck, which"
                " needs spine_control = true"
            )
        if settings.clocks and settings.leg_clocks:
            raise SettingsError(
                "[environment] clocks and leg_clocks are two kinds of clock, one per"
                " segment or one per leg: set only one"
            )
        has_clocks = settings.clocks or settings.leg_clocks
        if settings.neighbour_clocks and not has_clocks:
            raise SettingsError(
                "[environment] neighbour_clocks shows each segment its neighbours'"
                " clocks, which need the segments' clocks: set clocks = true or"
                " leg_clocks = true"
            )
        rewards = settings.rewards
        clock_costs = [key for key in CLOCK_COST_KEYS if getattr(rewards, key)]
        if clock_costs and not has_clocks:
            raise SettingsError(
                f"[environment.rewards] {', '.join(clock_costs)} need the segments'"
                " clocks: set clocks = true in [environment]"
            )
        if settings.clock.load_feedback_rad_per_s and not settings.leg_clocks:
            raise SettingsError(
                "[environment.clock] load_feedback_rad_per_s holds a loaded foot's"
                " clock in stance, which needs a clock per leg: set leg_clocks = true"
                " in [environment]"
            )
        if settings.clock.centre_time_constant_s and not settings.leg_clocks:
            raise SettingsError(
                "[environment.clock] centre_time_constant_s slows the centres of the"
                " legs' step shape, which needs a clock per leg: set leg_clocks ="
                " true in [environment]"
            )
        if settings.leg_clocks and rewards.legs_off_tempo_cost_parts:
            raise SettingsError(
                "[environment.rewards] legs_off_tempo_cost_parts compares each"
                " segment's legs with its clock's last turn; with leg_clocks the legs"
                " follow their clocks by construction: leave it out"
            )
        return settings
