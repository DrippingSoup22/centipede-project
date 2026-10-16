import math

import mujoco
import torch

from centipede.environment.simulation.body_measures import measure_body
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.environment.simulation.cpu_backend import CPUBackend
from centipede.environment.simulation.model_mapping import (
    ModelContractError,
    ModelMapping,
)
from centipede.environment.simulation.settings import SimulationSettings


def physics_steps_per_action(model: mujoco.MjModel) -> int:
    """How many of the model's timesteps make one 20 ms action.

    The model file comes from outside, so a timestep that does not fit a whole
    number of times into the action is rejected here, once.
    """
    steps = round(ACTION_DURATION_S / model.opt.timestep)
    if steps < 1 or abs(steps * model.opt.timestep - ACTION_DURATION_S) > 1e-9:
        raise ModelContractError(
            f"The model's timestep, {model.opt.timestep} s, must fit a whole "
            f"number of times into one {ACTION_DURATION_S * 1000:g} ms action"
        )
    return steps


def check_effort_contract(model: mujoco.MjModel) -> None:
    """Fail unless every contact's force can be read as the backends read it:
    elliptic friction cones, and at least the normal and two tangential
    directions in every contact."""
    if model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC or model.geom_condim.min() < 3:
        raise ModelContractError(
            "Measuring effort needs elliptic friction cones and every shape's"
            " condim at least 3"
        )


class PhysicsSimulation:
    """The physics simulation's front: the only part the environment uses.

    It loads the model named in the settings, checks it against the segment
    contract and the action's duration, and runs every world on the chosen
    backend. ``physical_state`` is refreshed in place by every ``reset`` and
    ``step``, and so is ``diagnostics.facts``, the simulation's diagnostics
    category. ``body_measures`` are the body's sizes, read from the model.
    """

    def __init__(
        self,
        settings: SimulationSettings,
        measure_effort: bool = False,
        measure_joint_forces: bool = False,
    ) -> None:
        """Load and check the model, then create the chosen backend.

        With ``measure_effort`` the backends also measure the physical state's
        effort fields after every physics step, and with
        ``measure_joint_forces`` its front joint forces.
        """
        model = mujoco.MjModel.from_xml_path(str(settings.model_path))
        mapping = ModelMapping.from_model(model)
        steps_per_action = physics_steps_per_action(model)
        if measure_effort:
            check_effort_contract(model)

        start_heading_range_rad = math.radians(settings.start_heading_range_deg)
        if settings.backend == "cpu":
            self._backend = CPUBackend(
                model,
                mapping,
                settings.world_count,
                steps_per_action,
                start_heading_range_rad,
                measure_effort,
                measure_joint_forces,
            )
        else:
            # Imported here so that a CPU-only installation, without the
            # optional GPU packages, can still use the CPU backend.
            from centipede.environment.simulation.gpu_backend import GPUBackend

            self._backend = GPUBackend(
                model,
                mapping,
                settings.world_count,
                steps_per_action,
                settings.gpu_solver,
                settings.contacts_per_world,
                settings.constraints_per_world,
                start_heading_range_rad,
                measure_effort,
                measure_joint_forces,
            )

        self.segment_count = mapping.segment_count
        # Each segment's leg motors: both legs' joints (the mapping's roles).
        self.leg_motor_count = 2 * len(mapping.leg_roles)
        self.head_outline = mapping.head_outline
        self.body_measures = measure_body(model, mapping)
        self.world_count = settings.world_count
        self.step_duration_s = ACTION_DURATION_S
        self.physical_state = self._backend.physical_state
        self.diagnostics = self._backend.diagnostics

    def reset(
        self, world_mask: torch.Tensor | None = None, seed: int | None = None
    ) -> None:
        """Reset the selected worlds (all if no mask); see the backends' ``reset``."""
        self._backend.reset(world_mask, seed)

    def step(
        self,
        leg_actions: torch.Tensor,
        spine_actions: torch.Tensor | None = None,
        body_forces: torch.Tensor | None = None,
    ) -> None:
        """Hold the (W, N, M) leg and (W, N - 1) spine actions for one step.

        Each segment's M leg actions are its left leg's joints from the body
        outward, then its right leg's (``ModelMapping``); every command goes
        unchanged to its motor, which applies it as a share of its maximum
        torque. Spine command ``i`` drives the joint between segments ``i`` and
        ``i + 1``; without spine actions, the spine motors receive zero.
        ``body_forces``, (W, N, 3) newtons in world axes, push each segment's
        body at its centre of mass through the step, as MuJoCo's applied
        forces do; without them the applied forces are left as they are, zero
        since the world's last reset unless a step gave some.
        """
        self._backend.step(leg_actions, spine_actions, body_forces)
