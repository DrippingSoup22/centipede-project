from dataclasses import dataclass

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
    to none at (1 + 1/k) times the target and below none beyond.
    A squared force cost, at ``joint_force_scale`` unit weights, costs
    ``reference_pay_mm_per_s``. Energy: a motor fully commanded costs
    ``force_price_w``, braking work ``braking_price`` of pushing work. Pain
    costs ``pain_price`` of a fully commanded motor at a leg's strongest push,
    hanging ``hanging_price`` of it for each unit weight held. A millijoule
    counts ``mm_per_mj`` millimetres. Forces are averaged over
    ``averaging_s``. Energy, the squeeze and pain are weighed from
    ``ramp_start`` up to full over the first ``ramp_cycles`` update cycles
    (0: full from the start); with ``ramp_after_line_share`` above 0 the
    ramp waits at ``ramp_start`` until the untowed mean-action worlds' line
    reaches that share of the cap, and rises from the next cycle.
    """

    line_speed_cap_mm_per_s: float
    line_cap_averaged: bool
    line_overspeed_slope: float
    reference_pay_mm_per_s: float
    joint_force_scale: float
    force_price_w: float
    braking_price: float
    pain_price: float
    hanging_price: float
    mm_per_mj: float
    averaging_s: float
    ramp_cycles: int
    ramp_start: float
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
            reference_pay_mm_per_s=section.number(
                "reference_pay_mm_per_s", default=53.0, minimum=0.0
            ),
            joint_force_scale=section.positive_number("joint_force_scale", default=0.3),
            force_price_w=section.number("force_price_w", default=0.0166, minimum=0.0),
            braking_price=section.number("braking_price", default=0.25, minimum=0.0),
            pain_price=section.number("pain_price", default=1.0, minimum=0.0),
            hanging_price=section.number("hanging_price", default=2.0, minimum=0.0),
            mm_per_mj=section.positive_number("mm_per_mj", default=1.59),
            averaging_s=section.positive_number("averaging_s", default=1.0),
            ramp_cycles=section.integer("ramp_cycles", default=128, minimum=0),
            ramp_start=section.number(
                "ramp_start", default=0.5, minimum=0.0, maximum=1.0
            ),
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
        return start + (1 - start) * min(1.0, (cycle - ramp_from) / self.ramp_cycles)


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
class EnvironmentSettings:
    """The [environment] section, with its nested sections (docs/follow-line/).

    A unit sees within ``sight_radius_spacings`` unit spacings of its centre:
    the neighbours within it in the straight body, and the line within it.
    The line is laid at up to ``line_angle_range_deg`` either way of the
    body's heading. With ``stagger_first_episodes`` each world starts its
    first episode at a random step of it. Each spine motor receives
    ``spine_action_scale`` times its command.
    """

    max_episode_steps: int
    sight_radius_spacings: float
    line_angle_range_deg: float
    stagger_first_episodes: bool
    spine_action_scale: float
    simulation: SimulationSettings
    reward: RewardSettings
    tow: TowSettings

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
            stagger_first_episodes=section.boolean(
                "stagger_first_episodes", default=True
            ),
            spine_action_scale=section.number(
                "spine_action_scale", default=0.35, minimum=0.0, maximum=1.0
            ),
            simulation=SimulationSettings.from_section(section.table("simulation")),
            reward=RewardSettings.from_section(section.table("reward")),
            tow=TowSettings.from_section(section.table("tow")),
        )
        section.reject_unknown_keys()
        return settings

    @property
    def observation_radius(self) -> int:
        """The neighbours a unit sees on each side: those within its sight
        in the straight body."""
        return int(self.sight_radius_spacings)
