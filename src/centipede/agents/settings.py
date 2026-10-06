from dataclasses import dataclass

from centipede.settings_section import SettingsSection


@dataclass(frozen=True)
class PPOSettings:
    """The [agents.ppo] section: how each segment agent learns from its data.

    ``clip_ratio`` must lie strictly between 0 and 1, as PPO requires.
    """

    discount: float
    gae_lambda: float
    clip_ratio: float
    update_epochs: int
    minibatch_size: int
    max_gradient_norm: float
    entropy_coefficient: float

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
        )
        section.reject_unknown_keys()
        return settings


@dataclass(frozen=True)
class AgentSettings:
    """The [agents] section, with its nested PPO section.

    ``device`` is where networks and stored data live. ``initial_action_std``
    is the starting value of the six learned action spreads, and normalised
    observations are clipped to plus or minus ``observation_clip``.
    """

    device: str
    hidden_layers: tuple[int, ...]
    initial_action_std: float
    learning_rate: float
    normaliser_epsilon: float
    observation_clip: float
    ppo: PPOSettings

    @classmethod
    def from_section(cls, values: dict) -> "AgentSettings":
        """Check the section and hand the nested PPO section to its own class.

        A PPO section left out of the file keeps all its defaults.
        """
        section = SettingsSection(values, "agents")
        settings = cls(
            device=section.choice("device", ("cpu", "cuda"), default="cpu"),
            hidden_layers=section.integer_list(
                "hidden_layers", default=(64, 64), minimum=1
            ),
            initial_action_std=section.positive_number(
                "initial_action_std", default=0.5
            ),
            learning_rate=section.positive_number("learning_rate", default=3e-4),
            normaliser_epsilon=section.positive_number(
                "normaliser_epsilon", default=1e-8
            ),
            observation_clip=section.positive_number("observation_clip", default=10.0),
            ppo=PPOSettings.from_section(section.table("ppo")),
        )
        section.reject_unknown_keys()
        return settings
