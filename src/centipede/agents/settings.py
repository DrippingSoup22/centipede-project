import math
from dataclasses import dataclass

from centipede.settings_section import SettingsSection

OPTIMIZERS = ("adam", "adamw", "sgd")
LEARNING_RATE_SCHEDULES = ("constant", "linear", "cosine")
ACTION_STD_SCHEDULES = ("learned", "log_linear")


@dataclass(frozen=True)
class PPOSettings:
    """The [agents.ppo] section: how each segment agent learns from its data.

    ``clip_ratio`` must lie strictly between 0 and 1, as PPO requires.
    ``temporal_smoothness_coefficient`` weighs CAPS's temporal term (Mysore et
    al., ICRA 2021) in the actor's loss: the distance between the mean actions
    of consecutive steps (RL_lib's PPO); 0 leaves it out.
    """

    discount: float
    gae_lambda: float
    clip_ratio: float
    update_epochs: int
    minibatch_size: int
    max_gradient_norm: float
    entropy_coefficient: float
    temporal_smoothness_coefficient: float

    @classmethod
    def from_section(cls, values: dict) -> "PPOSettings":
        """Check the section's values and fill in the defaults."""
        section = SettingsSection(values, "agents.ppo")
        settings = cls(
            discount=section.number(
                "discount", default=0.999, minimum=0.0, maximum=1.0
            ),
            gae_lambda=section.number(
                "gae_lambda", default=0.95, minimum=0.0, maximum=1.0
            ),
            clip_ratio=section.number(
                "clip_ratio", default=0.2, minimum=0.01, maximum=0.999
            ),
            update_epochs=section.positive_integer("update_epochs", default=4),
            minibatch_size=section.positive_integer("minibatch_size", default=64),
            max_gradient_norm=section.positive_number("max_gradient_norm", default=0.5),
            entropy_coefficient=section.number(
                "entropy_coefficient", default=0.001, minimum=0.0
            ),
            temporal_smoothness_coefficient=section.number(
                "temporal_smoothness_coefficient", default=0.0, minimum=0.0
            ),
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class AgentSettings:
    """The [agents] section, with its nested PPO section.

    ``device`` is where networks and stored data live. ``initial_action_std``
    is the starting value of the six learned action spreads. With
    ``action_std_schedule`` "log_linear" the spreads are not learned: every
    cycle sets them all (``action_std_at``), from ``initial_action_std`` in
    the first cycle to ``final_action_std`` in the last, in a straight line of
    their logarithms, as the PPO paper did for its humanoid tasks (Schulman et
    al., 2017). ``exploration_noise_beta`` colors the exploration noise over
    time: 0 is white noise, fresh at every step; 0.5 is the PPO default of
    Hollenstein et al. (AAAI 2024) and 1 pink noise (RL_lib's
    ``SquashedGaussianPolicy``).
    Normalised observations are clipped to plus or minus ``observation_clip``.
    Actor and critic each get their own ``optimizer``, with ``weight_decay``
    (and, for SGD, ``momentum``); their learning rate follows
    ``learning_rate_schedule`` over the run's update cycles
    (``learning_rate_at``).
    """

    device: str
    hidden_layers: tuple[int, ...]
    initial_action_std: float
    action_std_schedule: str
    final_action_std: float
    exploration_noise_beta: float
    optimizer: str
    learning_rate: float
    learning_rate_schedule: str
    final_learning_rate: float
    weight_decay: float
    momentum: float
    normaliser_epsilon: float
    observation_clip: float
    ppo: PPOSettings

    def learning_rate_at(self, cycle: int, total_cycles: int) -> float:
        """The learning rate of the update in ``cycle``, counted from 1 to
        ``total_cycles``: ``learning_rate`` in the first, moving to
        ``final_learning_rate`` in the last, in a straight line or along half
        a cosine, or ``learning_rate`` throughout when constant."""
        if self.learning_rate_schedule == "constant" or total_cycles == 1:
            return self.learning_rate
        progress = (cycle - 1) / (total_cycles - 1)
        if self.learning_rate_schedule == "cosine":
            progress = (1 - math.cos(math.pi * progress)) / 2
        return self.learning_rate + progress * (
            self.final_learning_rate - self.learning_rate
        )

    def action_std_at(self, cycle: int, total_cycles: int) -> float | None:
        """The action spread of every action in ``cycle``, counted from 1 to
        ``total_cycles``, or None when the spreads are learned."""
        if self.action_std_schedule == "learned":
            return None
        if total_cycles == 1:
            return self.initial_action_std
        progress = (cycle - 1) / (total_cycles - 1)
        return (
            self.initial_action_std
            * (self.final_action_std / self.initial_action_std) ** progress
        )

    @classmethod
    def from_section(cls, values: dict) -> "AgentSettings":
        """Check the section and hand the nested PPO section to its own class.

        A PPO section left out of the file keeps all its defaults.
        """
        section = SettingsSection(values, "agents")
        learning_rate = section.positive_number("learning_rate", default=3e-4)
        initial_action_std = section.positive_number("initial_action_std", default=0.5)
        settings = cls(
            device=section.choice("device", ("cpu", "cuda"), default="cpu"),
            hidden_layers=section.integer_list(
                "hidden_layers", default=(64, 64), minimum=1
            ),
            initial_action_std=initial_action_std,
            action_std_schedule=section.choice(
                "action_std_schedule", ACTION_STD_SCHEDULES, default="learned"
            ),
            final_action_std=section.positive_number(
                "final_action_std", default=initial_action_std
            ),
            exploration_noise_beta=section.number(
                "exploration_noise_beta", default=0.0, minimum=0.0, maximum=2.0
            ),
            optimizer=section.choice("optimizer", OPTIMIZERS, default="adam"),
            learning_rate=learning_rate,
            learning_rate_schedule=section.choice(
                "learning_rate_schedule", LEARNING_RATE_SCHEDULES, default="constant"
            ),
            final_learning_rate=section.positive_number(
                "final_learning_rate", default=learning_rate / 10
            ),
            weight_decay=section.number("weight_decay", default=0.0, minimum=0.0),
            momentum=section.number("momentum", default=0.9, minimum=0.0, maximum=1.0),
            normaliser_epsilon=section.positive_number(
                "normaliser_epsilon", default=1e-8
            ),
            observation_clip=section.positive_number("observation_clip", default=10.0),
            ppo=PPOSettings.from_section(section.table("ppo")),
        )
        section.reject_unknown_keys()
        return settings
