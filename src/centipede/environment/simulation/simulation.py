import math

import mujoco
import numpy as np
import torch

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


class PhysicsSimulation:
    """The physics simulation's front: the only part the environment uses.

    It loads the model named in the settings, checks it against the segment
    contract and the action's duration, and runs every world on the chosen
    backend. Leg actions run from -1 to 1 whatever the model: when its legs
    take target angles (model v4, ``legs_take_angles``), each action is
    mapped onto its joint's range before the backend applies it.
    ``physical_state`` is refreshed in place by every ``reset`` and ``step``,
    and so is ``diagnostics.facts``, the simulation's diagnostics category.
    ``body_weight_n`` is the whole body's weight, the unit of the feet's loads,
    and ``own_weight_n`` ``(N,)`` each segment's own, its body and legs.
    """

    def __init__(
        self,
        settings: SimulationSettings,
        average_floor_forces: bool = False,
        measure_leg_power: bool = False,
    ) -> None:
        """Load and check the model, then create the chosen backend.

        With ``average_floor_forces`` every step also measures each foot's
        normal force from the floor, averaged over the step's physics steps
        (``foot_floor_force``): a reading at the step's end alone misses
        what happened during it, by about a twentieth of a segment's weight
        on average in run 28's walk (CentipedeTraining, foot_step.py). With
        ``measure_leg_power`` every step also measures each segment's leg
        motors' positive power, averaged over its physics steps
        (``leg_motor_power``), which the energy cost reads.
        """
        model = mujoco.MjModel.from_xml_path(str(settings.model_path))
        mapping = ModelMapping.from_model(model)
        steps_per_action = physics_steps_per_action(model)

        start_heading_range_rad = math.radians(settings.start_heading_range_deg)
        if settings.backend == "cpu":
            self._backend = CPUBackend(
                model,
                mapping,
                settings.world_count,
                steps_per_action,
                start_heading_range_rad,
                average_floor_forces,
                measure_leg_power,
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
                average_floor_forces,
                measure_leg_power,
            )

        self.segment_count = mapping.segment_count
        self.leg_roles = mapping.leg_roles
        self.leg_motor_count = mapping.leg_motor_count
        self.legs_take_angles = mapping.legs_take_angles
        self.head_outline = mapping.head_outline
        # The whole body's weight, in newtons: the mass of the tree under the
        # head, the root, times the model's gravity (9.81 m/s²).
        self.body_weight_n = float(
            model.body_subtreemass[mapping.body_ids[0]]
            * np.linalg.norm(model.opt.gravity)
        )
        # Each segment's own weight, (N,) newtons: its body and legs, the mass
        # of its tree less the next segment's, which hangs from it.
        subtree_mass = model.body_subtreemass[mapping.body_ids]
        own_mass = subtree_mass - np.append(subtree_mass[1:], 0.0)
        self.own_weight_n = torch.as_tensor(
            own_mass * np.linalg.norm(model.opt.gravity),
            dtype=torch.float32,
            device=self._backend.physical_state.body_height.device,
        )
        self.world_count = settings.world_count
        self.step_duration_s = ACTION_DURATION_S
        self.physical_state = self._backend.physical_state
        self.diagnostics = self._backend.diagnostics

        # Legs that take angles: the middle and half the width of each leg
        # joint's range, (N, M) each, which the mapping checked is the range
        # of angles its motor accepts. None when the legs take torques. The
        # environment's diagnostics read it to give the step shape in angles.
        self.leg_angle_range: tuple[torch.Tensor, torch.Tensor] | None = None
        if mapping.legs_take_angles:
            lower, upper = torch.as_tensor(
                model.actuator_ctrlrange[mapping.leg_actuator_ids],
                dtype=torch.float32,
                device=self.physical_state.body_height.device,
            ).unbind(dim=-1)
            self.leg_angle_range = ((lower + upper) / 2, (upper - lower) / 2)

    def reset(
        self, world_mask: torch.Tensor | None = None, seed: int | None = None
    ) -> None:
        """Reset the selected worlds (all if no mask); see the backends' ``reset``."""
        self._backend.reset(world_mask, seed)

    def step(
        self,
        leg_actions: torch.Tensor,
        spine_actions: torch.Tensor | None = None,
        leg_actions_end: torch.Tensor | None = None,
    ) -> None:
        """Hold the (W, N, M) leg and (W, N - 1) spine actions for one step.

        Spine command ``i`` drives the joint between segments ``i`` and ``i + 1``;
        without spine actions, the spine motors receive zero. With
        ``leg_actions_end``, (W, N, M), the leg actions move in a straight line
        from ``leg_actions`` to it over the step's physics steps, reaching it on
        the last one, so that a target that moves during the step is followed
        instead of jumping once per step. When the legs take angles, each leg
        action becomes a target angle on its joint's range: -1 its lower limit,
        0 its middle, 1 its upper limit. Nothing is clipped; MuJoCo holds each
        target within its range.
        """
        if self.leg_angle_range is not None:
            middle, half_width = self.leg_angle_range
            leg_actions = torch.addcmul(middle, leg_actions, half_width)
            if leg_actions_end is not None:
                leg_actions_end = torch.addcmul(middle, leg_actions_end, half_width)
        self._backend.step(leg_actions, spine_actions, leg_actions_end)
