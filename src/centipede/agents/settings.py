import math
from dataclasses import dataclass

from centipede.settings_section import SettingsError, SettingsSection

OPTIMIZERS = ("adam", "adamw", "sgd")
LEARNING_RATE_SCHEDULES = ("constant", "linear", "cosine", "adaptive")
ACTION_STD_SCHEDULES = ("learned", "log_linear")
ACTIVATIONS = ("relu", "tanh")


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
    ``SquashedGaussianPolicy``). With ``state_dependent_noise_steps`` above 0
    the agents explore instead with generalized state-dependent exploration
    (gSDE; Raffin, Kober and Stulp, CoRL 2021): the noise is the actor's
    features times an exploration matrix each world keeps for that many
    steps, so that the same state gets the same noise and exploration tries
    behaviours the actor can learn, rather than a drift through time it
    cannot copy; the noise is then not colored. Its spreads, one per feature
    and action, start at ``initial_action_std``, which an action's spread is
    while they are equal (RL_lib's ``GaussianPolicyNetwork``).
    ``actor_activation`` is the actor's hidden activation, ReLU or tanh. gSDE
    wants tanh, as Raffin et al. and Stable-Baselines3's PPO use: the noise is
    made from the last hidden layer, and ReLU's features, never negative,
    hardly change direction from one state to the next, so a held matrix adds
    mostly a constant push (83% of the noise over 32 steps with run 13's
    actors, 52% with fresh tanh ones; Training's
    ``analysis/gsde_feature_check.py``).
    Normalised observations are clipped to plus or minus ``observation_clip``.
    Actor and critic each get their own ``optimizer``, with ``weight_decay``
    (and, for SGD, ``momentum``); their learning rate follows
    ``learning_rate_schedule`` over the run's update cycles
    (``learning_rate_at``). With "adaptive", each agent's actor steers its
    own learning rate from ``learning_rate`` by how far each update moved
    its policy, toward ``desired_kl`` (``SegmentAgent.adapt_learning_rate``),
    and the critics keep ``learning_rate``.

    With an environment that pays several reward channels, each agent has a
    critic per channel, and ``advantage_weights`` weighs their normalised
    advantages, one per channel (empty: 1 each); ``head_advantage_weights``,
    when given, are the head's own. With ``critic_baselines``
    each critic also has a baseline that sees the actions the neighbours took
    at the same step (RL_lib's PPO), and with ``value_normalization`` the
    critics learn standardised return targets.
    """

    device: str
    hidden_layers: tuple[int, ...]
    actor_activation: str
    initial_action_std: float
    action_std_schedule: str
    final_action_std: float
    exploration_noise_beta: float
    state_dependent_noise_steps: int
    optimizer: str
    learning_rate: float
    learning_rate_schedule: str
    final_learning_rate: float
    desired_kl: float
    weight_decay: float
    momentum: float
    normaliser_epsilon: float
    observation_clip: float
    advantage_weights: tuple[float, ...]
    head_advantage_weights: tuple[float, ...]
    critic_baselines: bool
    value_normalization: bool
    ppo: PPOSettings

    def learning_rate_at(self, cycle: int, total_cycles: int) -> float | None:
        """The learning rate of the update in ``cycle``, counted from 1 to
        ``total_cycles``: ``learning_rate`` in the first, moving to
        ``final_learning_rate`` in the last, in a straight line or along half
        a cosine, or ``learning_rate`` throughout when constant; None when
        each agent adapts its own."""
        if self.learning_rate_schedule == "adaptive":
            return None
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
            actor_activation=section.choice(
                "actor_activation", ACTIVATIONS, default="relu"
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
            state_dependent_noise_steps=section.integer(
                "state_dependent_noise_steps", default=0, minimum=0
            ),
            optimizer=section.choice("optimizer", OPTIMIZERS, default="adam"),
            learning_rate=learning_rate,
            learning_rate_schedule=section.choice(
                "learning_rate_schedule", LEARNING_RATE_SCHEDULES, default="constant"
            ),
            final_learning_rate=section.positive_number(
                "final_learning_rate", default=learning_rate / 10
            ),
            desired_kl=section.positive_number("desired_kl", default=0.01),
            weight_decay=section.number("weight_decay", default=0.0, minimum=0.0),
            momentum=section.number("momentum", default=0.9, minimum=0.0, maximum=1.0),
            normaliser_epsilon=section.positive_number(
                "normaliser_epsilon", default=1e-8
            ),
            observation_clip=section.positive_number("observation_clip", default=10.0),
            advantage_weights=section.number_list(
                "advantage_weights", default=(), minimum=0.0
            ),
            head_advantage_weights=section.number_list(
                "head_advantage_weights", default=(), minimum=0.0
            ),
            critic_baselines=section.boolean("critic_baselines", default=False),
            value_normalization=section.boolean("value_normalization", default=False),
            ppo=PPOSettings.from_section(section.table("ppo")),
        )
        section.reject_unknown_keys()
        if settings.state_dependent_noise_steps and settings.exploration_noise_beta:
            raise SettingsError(
                "[agents] state_dependent_noise_steps and exploration_noise_beta"
                " are two kinds of noise: set only one"
            )
        return settings
