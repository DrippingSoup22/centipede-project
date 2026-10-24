from dataclasses import dataclass, field

from centipede.environment.simulation import SimulationSettings
from centipede.settings_section import SettingsError, SettingsSection


@dataclass(frozen=True)
class RewardSettings:
    """The [environment.reward] section (docs/follow-line/reward.md); the
    defaults are the agreed values.

    The line pays each unit's step along it up to ``line_speed_cap_mm_per_s``:
    on each step, or with ``line_cap_averaged`` on the unit's speed averaged
    over ``averaging_s`` from the cap at the episode's start, each step's
    progress scaled down by the cap over that average when it is above (a
    stride's surges and dips cancel, so walking faster than the cap gains
    nothing). With ``line_overspeed_slope`` k above 0 the cap becomes a
    target: above it a second's pay falls by k mm/s for every mm/s too fast,
    to none at (1 + 1/k) times the target and below none beyond. With
    ``commanded_speed_mm_per_s`` v* above 0 the team is paid v* - |v - v*|
    mm a second instead, v the units' mean of their forward line (the line's
    pay for each step, uncapped) averaged over ``averaging_s`` from 0 at the
    episode's start: below v* its speed, above it 1 mm/s less for each mm/s
    too fast. With ``slow_in_bends`` the command in |v - v*| falls in a bend
    so that the outer feet keep v*: v* / (1 + r k), r the feet's reach to
    the side and k the line's curvature, the units' mean, averaged as v is;
    the team keeps v* - |v - command|, so a bend lowers no pay.
    With ``match_plain_speed`` v is instead the units' mean plain speed along
    the line, each step counted whatever the unit's distance or angle from
    it, and the line is kept by its own price: with ``line_price_mm_per_s``
    above 0 each unit's own account is charged that many mm a second times
    the mean, over itself and the units behind it that it sees (the spine
    joint behind a unit places them), of 1 - exp(-d /
    ``line_distance_scale_mm``), d each one's distance from the line:
    steepest at the line, so that a few millimetres cost (Meyer et al. 2020).
    A squared force cost, at ``joint_force_scale`` unit weights, costs
    ``reference_pay_mm_per_s``. Energy: a motor fully commanded costs
    ``force_price_w``, and a motor commanded u, as a share of its strongest,
    that times |u| to the power ``force_exponent`` (2, the heat of an
    electric motor; 1, a muscle's cost of holding force); braking work costs
    ``braking_price`` of pushing work; switching force costs
    ``switching_price`` mJ/s for a motor whose command changes at 1 a
    second, as that rate squared (a muscle's cost of turning its force on
    and off, van der Zee & Kuo 2021). Pain
    costs ``pain_price`` of a fully commanded motor at a leg's strongest push,
    hanging ``hanging_price`` of it for each unit weight held. A millijoule
    counts ``mm_per_mj`` millimetres (0: no millijoule part counts). The
    motion costs ``motion_price`` mm a second for each (rad/s)² of a unit's
    joints' squared speeds, added up, and the accelerations
    ``acceleration_price`` for each (rad/s²)² of their squared changes of
    speed. With ``acceleration_halving`` above 0 the accelerations take a
    share of the line's forward pay instead of a price: each step's forward
    pay is multiplied by 1 / (1 + J / ``acceleration_halving``), J the units'
    mean of the same squared changes, each averaged over ``averaging_s``, so
    the pay halves at J = ``acceleration_halving`` (rad/s²)²; a step back is
    still charged in full. With ``time_value_mj_per_s`` h above 0, energy
    takes such a share instead: the forward pay is multiplied by
    h / (h + E), E the units' mean of their energy, priced as above, in mJ
    a second averaged over ``averaging_s``; h is the value of a second of
    the walk's time (the task's priority), so the team keeps most for the
    walk that spends least for its distance and time. With
    ``neighbourhood_energy`` the team account charges each unit the mean
    energy of the units it sees, itself and its neighbours (the units
    present), instead of the units' mean; ``energy_weight_behind`` weighs
    the units behind it against itself and the units ahead (1: all alike;
    below 1, each unit answers more for the units ahead, which then lead).
    Each unit's feet not
    carrying its weight, beyond its free support, cost it ``carry_price`` mm
    a second for each unit weight on average; with ``carry_per_leg`` each
    foot is charged instead for its averaged load's distance from half that
    weight, in either direction, beyond ``carry_band`` of the half: over a
    second the unit's weight goes through its own two legs, not through its
    neighbours, whatever the rhythm. With
    ``own_account`` false every part counts in one account, the team's, and
    each agent has one critic. Forces are averaged over ``averaging_s``.
    Energy, the motion, the accelerations, the squeeze and pain are weighed
    from ``ramp_start`` up to full over the first ``ramp_cycles`` update
    cycles (0: full from the start), in a straight line or, with
    ``ramp_shape`` "exponential", by the same factor each cycle; with
    ``ramp_after_line_share`` above 0 the ramp waits at ``ramp_start`` until
    the untowed mean-action worlds' line reaches that share of the cap, and
    rises from the next cycle. With ``ramp_shape`` "adaptive" the weight
    follows the untowed mean-action worlds instead, from ``ramp_start``,
    after each of the tow's windows (``adapted_ramp_weight``).
    """

    line_speed_cap_mm_per_s: float
    line_cap_averaged: bool
    line_overspeed_slope: float
    commanded_speed_mm_per_s: float
    slow_in_bends: bool
    match_plain_speed: bool
    reference_pay_mm_per_s: float
    joint_force_scale: float
    force_price_w: float
    force_exponent: float
    braking_price: float
    switching_price: float
    pain_price: float
    hanging_price: float
    motion_price: float
    acceleration_price: float
    acceleration_halving: float
    time_value_mj_per_s: float
    neighbourhood_energy: bool
    energy_weight_behind: float
    carry_price: float
    carry_per_leg: bool
    carry_band: float
    line_price_mm_per_s: float
    line_distance_scale_mm: float
    own_account: bool
    mm_per_mj: float
    averaging_s: float
    ramp_cycles: int
    ramp_start: float
    ramp_shape: str
    ramp_band_mm_per_s: tuple[float, float]
    ramp_factor: float
    ramp_after_line_share: float

    @classmethod
    def from_section(cls, values: dict) -> "RewardSettings":
        section = SettingsSection(values, "environment.reward")
        settings = cls(
            line_speed_cap_mm_per_s=section.positive_number(
                "line_speed_cap_mm_per_s", default=90.0
            ),
            line_cap_averaged=section.boolean("line_cap_averaged", default=False),
            line_overspeed_slope=section.number(
                "line_overspeed_slope", default=0.0, minimum=0.0
            ),
            commanded_speed_mm_per_s=section.number(
                "commanded_speed_mm_per_s", default=0.0, minimum=0.0
            ),
            slow_in_bends=section.boolean("slow_in_bends", default=False),
            match_plain_speed=section.boolean("match_plain_speed", default=False),
            reference_pay_mm_per_s=section.number(
                "reference_pay_mm_per_s", default=53.0, minimum=0.0
            ),
            joint_force_scale=section.positive_number("joint_force_scale", default=0.3),
            force_price_w=section.number("force_price_w", default=0.0166, minimum=0.0),
            force_exponent=section.positive_number("force_exponent", default=2.0),
            braking_price=section.number("braking_price", default=0.25, minimum=0.0),
            switching_price=section.number("switching_price", default=0.0, minimum=0.0),
            pain_price=section.number("pain_price", default=1.0, minimum=0.0),
            hanging_price=section.number("hanging_price", default=2.0, minimum=0.0),
            motion_price=section.number("motion_price", default=0.0, minimum=0.0),
            acceleration_price=section.number(
                "acceleration_price", default=0.0, minimum=0.0
            ),
            acceleration_halving=section.number(
                "acceleration_halving", default=0.0, minimum=0.0
            ),
            time_value_mj_per_s=section.number(
                "time_value_mj_per_s", default=0.0, minimum=0.0
            ),
            neighbourhood_energy=section.boolean("neighbourhood_energy", default=False),
            energy_weight_behind=section.number(
                "energy_weight_behind", default=1.0, minimum=0.0, maximum=1.0
            ),
            carry_price=section.number("carry_price", default=0.0, minimum=0.0),
            carry_per_leg=section.boolean("carry_per_leg", default=False),
            carry_band=section.number(
                "carry_band", default=0.25, minimum=0.0, maximum=1.0
            ),
            line_price_mm_per_s=section.number(
                "line_price_mm_per_s", default=0.0, minimum=0.0
            ),
            line_distance_scale_mm=section.positive_number(
                "line_distance_scale_mm", default=4.0
            ),
            own_account=section.boolean("own_account", default=True),
            mm_per_mj=section.number("mm_per_mj", default=1.59, minimum=0.0),
            averaging_s=section.positive_number("averaging_s", default=1.0),
            ramp_cycles=section.integer("ramp_cycles", default=128, minimum=0),
            ramp_start=section.number(
                "ramp_start", default=0.5, minimum=0.0, maximum=1.0
            ),
            ramp_shape=section.choice(
                "ramp_shape", ("linear", "exponential", "adaptive"), default="linear"
            ),
            ramp_band_mm_per_s=section.number_range(
                "ramp_band_mm_per_s", default=(70.0, 110.0), minimum=0.0
            ),
            ramp_factor=section.number("ramp_factor", default=1.1, minimum=1.0),
            ramp_after_line_share=section.number(
                "ramp_after_line_share", default=0.0, minimum=0.0, maximum=1.0
            ),
        )
        section.reject_unknown_keys()
        if settings.line_overspeed_slope and not settings.line_cap_averaged:
            raise SettingsError(
                "[environment.reward] line_overspeed_slope works on the averaged"
                " speed: set line_cap_averaged = true"
            )
        if settings.commanded_speed_mm_per_s and (
            settings.line_cap_averaged
            or settings.acceleration_halving
            or settings.time_value_mj_per_s
        ):
            raise SettingsError(
                "[environment.reward] a commanded speed replaces the line's cap"
                " and its shares: leave line_cap_averaged, acceleration_halving"
                " and time_value_mj_per_s unset"
            )
        if settings.slow_in_bends and not settings.commanded_speed_mm_per_s:
            raise SettingsError(
                "[environment.reward] slow_in_bends lowers the commanded speed:"
                " set commanded_speed_mm_per_s"
            )
        if settings.match_plain_speed and not settings.commanded_speed_mm_per_s:
            raise SettingsError(
                "[environment.reward] match_plain_speed matches the commanded"
                " speed: set commanded_speed_mm_per_s"
            )
        if settings.acceleration_price and settings.acceleration_halving:
            raise SettingsError(
                "[environment.reward] the accelerations take either a price or a"
                " share of the line: set acceleration_price or"
                " acceleration_halving, not both"
            )
        if settings.acceleration_halving and settings.time_value_mj_per_s:
            raise SettingsError(
                "[environment.reward] the line's pay has one share: set"
                " acceleration_halving or time_value_mj_per_s, not both"
            )
        if settings.ramp_shape != "linear" and not settings.ramp_start:
            raise SettingsError(
                f"[environment.reward] an {settings.ramp_shape} ramp grows from"
                " ramp_start: set it above 0"
            )
        return settings

    def ramp_weight_at(self, cycle: int, ramp_from: int | None = 1) -> float:
        """The ramped costs' weight in update cycle ``cycle``, counted from 1,
        for a ramp that starts rising at cycle ``ramp_from``; ``None``: it
        has not started (``ramp_after_line_share``)."""
        if ramp_from is None:
            return self.ramp_start
        if not self.ramp_cycles:
            return 1.0
        start = self.ramp_start
        risen = min(1.0, (cycle - ramp_from) / self.ramp_cycles)
        if self.ramp_shape == "exponential":
            return start ** (1 - risen)
        return start + (1 - start) * risen

    def adapted_ramp_weight(
        self, weight: float, speed_mm_per_s: float, stepping: bool
    ) -> float:
        """The adaptive ramp's weight after a window in which the untowed
        mean-action worlds went ``speed_mm_per_s`` along the line, their feet
        ``stepping`` or not: raised while they walk faster than the band,
        lowered while slower, held within it. Each step is ``ramp_factor``
        times, raised to one more power for each band width beyond the edge
        (at most four): a proportional term (Stooke et al. 2020). The weight
        stays within ``ramp_start`` and 1."""
        low, high = self.ramp_band_mm_per_s
        width = max(high - low, 1.0)
        if speed_mm_per_s > high and stepping:
            weight *= self.ramp_factor ** min(4.0, 1 + (speed_mm_per_s - high) / width)
        elif speed_mm_per_s < low:
            weight /= self.ramp_factor ** min(4.0, 1 + (low - speed_mm_per_s) / width)
        return min(1.0, max(self.ramp_start, weight))


@dataclass(frozen=True)
class TowSettings:
    """The [environment.tow] section (docs/follow-line/task.md, the tow); the
    defaults are the agreed values, but the tow is off unless ``start`` is set.

    In training each unit is pushed forward by its world's tow level times
    its weight, shrinking with its forward speed to none at
    ``help_speed_mm_per_s``. A training world draws its level from
    ``draw_floor`` of the top to the top at every restart. The top starts at
    ``start`` (0: no tow) and, once ``window_steps`` steps have passed since
    it last moved, falls by ``step`` while the untowed mean-action worlds
    walk (their line above ``walk_mm_per_s`` a unit, their feet lifting off
    above ``lift_off_hz``), and rises by it otherwise, up to ``start``.
    """

    start: float
    step: float
    window_steps: int
    walk_mm_per_s: float
    lift_off_hz: float
    help_speed_mm_per_s: float
    draw_floor: float

    @classmethod
    def from_section(cls, values: dict) -> "TowSettings":
        section = SettingsSection(values, "environment.tow")
        settings = cls(
            start=section.number("start", default=0.0, minimum=0.0, maximum=1.0),
            step=section.number("step", default=0.02, minimum=0.0),
            window_steps=section.positive_integer("window_steps", default=128),
            walk_mm_per_s=section.number("walk_mm_per_s", default=2.7, minimum=0.0),
            lift_off_hz=section.number("lift_off_hz", default=1.0, minimum=0.0),
            help_speed_mm_per_s=section.positive_number(
                "help_speed_mm_per_s", default=33.0
            ),
            draw_floor=section.number(
                "draw_floor", default=0.5, minimum=0.0, maximum=1.0
            ),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class CurveSettings:
    """The [environment.curves] section (docs/follow-line/task.md, the
    curves), read when the environment's ``line_shape`` is "curved"; every
    length in body lengths.

    The line runs straight through the body and ``lead_body_lengths`` beyond
    the head, then bends piece by piece: each piece between the two
    ``piece_body_lengths`` long, straight with the level's share in
    ``level_straight_shares``, else curved to the left or right, its radius
    drawn no tighter than the level's in ``level_tightest_radius_body_lengths``
    (its curvature drawn evenly up to the tightest); the curvature moves from
    the piece before's in a straight line over the first
    ``bend_change_body_lengths`` of each piece, so the line has no corner.
    The level starts at the first and rises by one, after at least
    ``level_judged_cycles`` update cycles at it, when the untowed mean-action
    worlds kept the line through ``level_up_kept_share`` of their episodes
    over the last of those cycles. With ``start_speed_mm_per_s`` above 0 the
    lines stay straight until the wave starts forming: once, over the last 8
    update cycles, the untowed mean-action worlds' speed along the line
    reaches it and some neighbour pair's wave consistency reaches
    ``start_pair_consistency`` (chance is about 0.2), the first level begins
    with the next cycle. With ``mixed_levels`` each world's line is laid at a
    level drawn evenly from the first to the one reached, and only the worlds
    at the one reached judge it (Rudin et al. 2022 send the robots that solve
    the top level back to a random one, against forgetting).
    """

    level_tightest_radius_body_lengths: tuple[float, ...]
    level_straight_shares: tuple[float, ...]
    piece_body_lengths: tuple[float, float]
    bend_change_body_lengths: float
    lead_body_lengths: float
    level_up_kept_share: float
    level_judged_cycles: int
    start_speed_mm_per_s: float
    start_pair_consistency: float
    mixed_levels: bool

    @classmethod
    def from_section(cls, values: dict) -> "CurveSettings":
        section = SettingsSection(values, "environment.curves")
        settings = cls(
            level_tightest_radius_body_lengths=section.number_list(
                "level_tightest_radius_body_lengths", default=()
            ),
            level_straight_shares=section.number_list(
                "level_straight_shares", default=(), minimum=0.0
            ),
            piece_body_lengths=section.number_range(
                "piece_body_lengths", default=(1.0, 3.0), minimum=0.0
            ),
            bend_change_body_lengths=section.positive_number(
                "bend_change_body_lengths", default=1.0
            ),
            lead_body_lengths=section.number(
                "lead_body_lengths", default=0.5, minimum=0.0
            ),
            level_up_kept_share=section.number(
                "level_up_kept_share", default=0.9, minimum=0.0, maximum=1.0
            ),
            level_judged_cycles=section.positive_integer(
                "level_judged_cycles", default=16
            ),
            start_speed_mm_per_s=section.number(
                "start_speed_mm_per_s", default=0.0, minimum=0.0
            ),
            start_pair_consistency=section.number(
                "start_pair_consistency", default=0.0, minimum=0.0, maximum=1.0
            ),
            mixed_levels=section.boolean("mixed_levels", default=False),
        )
        section.reject_unknown_keys()
        radii = settings.level_tightest_radius_body_lengths
        if len(radii) != len(settings.level_straight_shares):
            raise SettingsError(
                "[environment.curves] give each level a tightest radius and a"
                " straight share: level_tightest_radius_body_lengths and"
                " level_straight_shares must be as long"
            )
        if any(radius <= 0 for radius in radii) or any(
            share > 1 for share in settings.level_straight_shares
        ):
            raise SettingsError(
                "[environment.curves] radii must be above 0 and straight shares"
                " at most 1"
            )
        if settings.bend_change_body_lengths > settings.piece_body_lengths[0]:
            raise SettingsError(
                "[environment.curves] a bend's change must fit in the shortest"
                " piece: bend_change_body_lengths at most piece_body_lengths' low"
            )
        return settings


@dataclass(frozen=True)
class SpeedSettings:
    """The [environment.speeds] section (docs/follow-line/task.md, the
    speeds, stage 3): the commanded speed changes, level by level. Without
    levels every world keeps the reward's ``commanded_speed_mm_per_s``.

    At each restart a world draws a speed at its level: 0 with the level's
    share in ``level_stop_share``, else evenly between its
    ``level_low_mm_per_s`` and ``level_high_mm_per_s``; it holds it for the
    level's ``level_hold_s``, then draws again (a hold of 0 keeps it the
    whole episode). Stops belong to the levels whose range starts at 0. The
    levels begin at once, or with ``start_after_curves`` once the curves
    reach their last level, and rise by one, after at least
    ``level_judged_cycles`` update cycles at a level, when the untowed
    mean-action worlds drawn at it, over those cycles, kept their speed,
    averaged over 1 s, within ``level_up_speed_error_mm_per_s`` of the
    command, averaged alike, on the mean; spent at most
    ``level_up_energy_mj_per_mm`` for each mm while the command was above 0
    (a unit's energy over its distance, so that a quick, tapping gait does
    not pass); and kept the line through ``level_up_kept_share`` of their
    episodes. With ``mixed_levels`` each world draws its level evenly up to
    the one reached, as the curves do.
    """

    level_hold_s: tuple[float, ...]
    level_low_mm_per_s: tuple[float, ...]
    level_high_mm_per_s: tuple[float, ...]
    level_stop_share: tuple[float, ...]
    start_after_curves: bool
    mixed_levels: bool
    level_judged_cycles: int
    level_up_speed_error_mm_per_s: float
    level_up_energy_mj_per_mm: float
    level_up_kept_share: float

    @classmethod
    def from_section(cls, values: dict) -> "SpeedSettings":
        section = SettingsSection(values, "environment.speeds")
        settings = cls(
            level_hold_s=section.number_list("level_hold_s", default=(), minimum=0.0),
            level_low_mm_per_s=section.number_list(
                "level_low_mm_per_s", default=(), minimum=0.0
            ),
            level_high_mm_per_s=section.number_list(
                "level_high_mm_per_s", default=(), minimum=0.0
            ),
            level_stop_share=section.number_list(
                "level_stop_share", default=(), minimum=0.0
            ),
            start_after_curves=section.boolean("start_after_curves", default=False),
            mixed_levels=section.boolean("mixed_levels", default=False),
            level_judged_cycles=section.positive_integer(
                "level_judged_cycles", default=16
            ),
            level_up_speed_error_mm_per_s=section.positive_number(
                "level_up_speed_error_mm_per_s", default=8.0
            ),
            level_up_energy_mj_per_mm=section.positive_number(
                "level_up_energy_mj_per_mm", default=0.05
            ),
            level_up_kept_share=section.number(
                "level_up_kept_share", default=0.9, minimum=0.0, maximum=1.0
            ),
        )
        section.reject_unknown_keys()
        levels = (
            settings.level_hold_s,
            settings.level_low_mm_per_s,
            settings.level_high_mm_per_s,
            settings.level_stop_share,
        )
        if len({len(values) for values in levels}) > 1:
            raise SettingsError(
                "[environment.speeds] give each level a hold, a range and a stop"
                " share: level_hold_s, level_low_mm_per_s, level_high_mm_per_s"
                " and level_stop_share must be as long"
            )
        for low, high, stops in zip(*levels[1:], strict=True):
            if low > high or stops > 1 or (stops and low):
                raise SettingsError(
                    "[environment.speeds] each level's low must be at most its"
                    " high, its stop share at most 1, and only a range from 0"
                    " stops"
                )
        return settings


@dataclass(frozen=True)
class EnvironmentSettings:
    """The [environment] section, with its nested sections (docs/follow-line/).

    A unit sees within ``sight_radius_spacings`` unit spacings of its centre:
    the neighbours within it in the straight body, and the line within it.
    The line is laid at up to ``line_angle_range_deg`` either way of the
    body's heading; ``line_shape`` "straight" keeps it straight (stage 1),
    "curved" bends it beyond the head as the curves section says (stage
    2). With ``stagger_first_episodes`` each world starts its first episode
    at a random step of it. Each spine motor receives
    ``spine_action_scale`` times its command. With ``observe_priced_forces``
    false a unit's block leaves out the forces only the old costs priced: its
    legs' press and the joint forces averaged over a second. With
    ``observe_carry`` it holds its feet's load averaged as the carry is
    (each foot's, with the reward's ``carry_per_leg``), and with
    ``observe_energy`` its energy in mJ/s averaged as the energy's share is,
    and with ``observe_line_speed`` its forward line in mm/s averaged as the
    commanded speed averages it. With ``observe_command`` every unit also
    sees the commanded speed in mm/s, averaged as its speed is, which bends
    lower with the reward's ``slow_in_bends``, and with
    ``observe_current_command`` the command as it stands on the step, so that
    a change of the speeds' schedule shows at once.

    From 2026-10-24 (the terrain park, docs/follow-line/terrain.md): with
    ``start_area_weights``, ground name to weight, every episode starts at a
    point drawn uniformly in one of the terrain's start areas, chosen with
    these weights (the areas left out are never chosen); without them the
    body starts where the model puts it. With ``observe_ground_ahead`` the
    front unit sees the ground ahead of it (``GROUND_VIEW_VALUES`` heights,
    ``environment.ground_view``); the others see zeros there.
    """

    max_episode_steps: int
    sight_radius_spacings: float
    line_angle_range_deg: float
    line_shape: str
    stagger_first_episodes: bool
    spine_action_scale: float
    observe_priced_forces: bool
    observe_carry: bool
    observe_energy: bool
    observe_line_speed: bool
    observe_command: bool
    observe_current_command: bool
    simulation: SimulationSettings
    reward: RewardSettings
    tow: TowSettings
    curves: CurveSettings
    speeds: SpeedSettings
    observe_ground_ahead: bool = False
    start_area_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_section(cls, values: dict) -> "EnvironmentSettings":
        """Check the section and hand each nested section to its own class.

        A nested section left out of the file keeps all its defaults, except
        that the simulation's model file and backend are always required.
        """
        section = SettingsSection(values, "environment")
        settings = cls(
            max_episode_steps=section.positive_integer(
                "max_episode_steps", default=1000
            ),
            sight_radius_spacings=section.positive_number(
                "sight_radius_spacings", default=2.5
            ),
            line_angle_range_deg=section.number(
                "line_angle_range_deg", default=10.0, minimum=0.0, maximum=90.0
            ),
            line_shape=section.choice(
                "line_shape", ("straight", "curved"), default="straight"
            ),
            stagger_first_episodes=section.boolean(
                "stagger_first_episodes", default=True
            ),
            spine_action_scale=section.number(
                "spine_action_scale", default=0.35, minimum=0.0, maximum=1.0
            ),
            observe_priced_forces=section.boolean(
                "observe_priced_forces", default=True
            ),
            observe_carry=section.boolean("observe_carry", default=False),
            observe_energy=section.boolean("observe_energy", default=False),
            observe_line_speed=section.boolean("observe_line_speed", default=False),
            observe_command=section.boolean("observe_command", default=False),
            observe_current_command=section.boolean(
                "observe_current_command", default=False
            ),
            simulation=SimulationSettings.from_section(section.table("simulation")),
            reward=RewardSettings.from_section(section.table("reward")),
            tow=TowSettings.from_section(section.table("tow")),
            curves=CurveSettings.from_section(section.table("curves")),
            speeds=SpeedSettings.from_section(section.table("speeds")),
            observe_ground_ahead=section.boolean("observe_ground_ahead", default=False),
            start_area_weights=_weights(section.table("start_area_weights")),
        )
        section.reject_unknown_keys()
        if settings.speeds.level_hold_s and not (
            settings.reward.commanded_speed_mm_per_s
        ):
            raise SettingsError(
                "[environment.speeds] the levels change the commanded speed:"
                " set [environment.reward] commanded_speed_mm_per_s"
            )
        if settings.speeds.start_after_curves and settings.line_shape != "curved":
            raise SettingsError(
                '[environment.speeds] start_after_curves waits for a "curved"'
                " line's last level"
            )
        if (
            settings.line_shape == "curved"
            and not settings.curves.level_tightest_radius_body_lengths
        ):
            raise SettingsError(
                '[environment] a "curved" line needs its levels:'
                " set [environment.curves] level_tightest_radius_body_lengths"
                " and level_straight_shares"
            )
        return settings

    @property
    def observation_radius(self) -> int:
        """The neighbours a unit sees on each side: those within its sight
        in the straight body."""
        return int(self.sight_radius_spacings)


def _weights(values: dict) -> dict[str, float]:
    """The [environment.start_area_weights] table: non-negative numbers, at
    least one of them positive if any are given."""
    section = SettingsSection(values, "environment.start_area_weights")
    weights = {name: section.number(name, minimum=0.0) for name in values}
    if weights and not any(weights.values()):
        raise SettingsError("environment.start_area_weights needs a positive weight")
    return weights
