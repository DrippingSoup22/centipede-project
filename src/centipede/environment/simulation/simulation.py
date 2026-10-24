import math

import mujoco
import numpy as np
import torch

from centipede.environment.simulation.body_measures import measure_body
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.environment.simulation.cpu_backend import CPUBackend
from centipede.environment.simulation.ground import Ground
from centipede.environment.simulation.model_mapping import (
    ModelContractError,
    ModelMapping,
)
from centipede.environment.simulation.settings import SimulationSettings

# The body file's flat floor, which a terrain replaces.
FLOOR_NAME = "floor"
# How far above the highest ground within its reach a start puts the body.
START_CLEARANCE_M = 0.001
# What every terrain shape takes from the floor: how it collides and its
# contacts behave, its marks (owner -1, the floor's category) and its look.
FLOOR_SETTINGS = (
    "contype",
    "conaffinity",
    "condim",
    "priority",
    "friction",
    "solref",
    "solimp",
    "solmix",
    "margin",
    "gap",
    "userdata",
    "material",
)


def model_spec(settings: SimulationSettings) -> mujoco.MjSpec:
    """The body's model, standing on the terrain if the settings name one.

    A body file brings its own flat floor, the shape named ``floor``. A
    terrain file holds only ground: shapes in its world, such as a height
    field, and the assets they use. MuJoCo's model editor (``MjSpec``) joins
    the two in memory, so that neither file changes: the terrain's world is
    attached to the body's, its names prefixed with ``terrain_``, every
    shape there takes the floor's settings (``FLOOR_SETTINGS``), and the
    floor is deleted. Only the ground's shape changes, never its contact.
    """
    spec = mujoco.MjSpec.from_file(str(settings.model_path))
    if settings.terrain_path is None:
        return spec
    floor = spec.geom(FLOOR_NAME)
    terrain = mujoco.MjSpec.from_file(str(settings.terrain_path))
    spec.attach(terrain, prefix="terrain_", frame=spec.worldbody.add_frame())
    for shape in spec.worldbody.geoms:
        if shape.name != FLOOR_NAME:
            for name in FLOOR_SETTINGS:
                setattr(shape, name, getattr(floor, name))
    spec.delete(floor)
    return spec


def body_reach_m(model: mujoco.MjModel, root_position: np.ndarray) -> float:
    """How far the body's shapes reach from its root seen from above, in
    the model's pose: the farthest shape's distance plus its bounding radius."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_shapes = model.geom_bodyid > 0
    distances = np.linalg.norm(
        data.geom_xpos[body_shapes, :2] - root_position[:2], axis=1
    )
    return float((distances + model.geom_rbound[body_shapes]).max())


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
    ``model_xml`` is the model as one XML document, the terrain joined in,
    as recordings keep it, and ``model_assets`` the files it reads, by name
    (a terrain's height field). ``ground`` is the ground's height anywhere
    and the terrain's start areas (``Ground``).
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
        spec = model_spec(settings)
        model = spec.compile()
        self.model_xml = spec.to_xml()
        self.model_assets = {
            field.file: (settings.terrain_path.parent / field.file).read_bytes()
            for field in spec.hfields
            if field.file
        }
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
                settings.max_restarted_worlds_per_action,
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
        self.ground = Ground(model, self.physical_state.body_height.device)
        root = mapping.root_qpos_address
        self._root_qpos = model.qpos0[root : root + 3].copy()
        self._body_reach_m = body_reach_m(model, self._root_qpos)
        # The front segment's heading in the model pose, which a start's turn
        # turns (the yaw of its forward axis, as the line reads it).
        pose = mujoco.MjData(model)
        mujoco.mj_kinematics(model, pose)
        w, x, y, z = pose.xquat[mapping.body_ids[0]]
        self.model_heading_rad = math.atan2(
            2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
        )
        # (W,) flags, refreshed by every step: the worlds the GPU backend put
        # back to the step's start after MuJoCo Warp made them non-finite.
        self.restarted_worlds = self.diagnostics.facts.restarted_worlds

    def reset(
        self,
        world_mask: torch.Tensor | None = None,
        seed: int | None = None,
        start_positions: torch.Tensor | None = None,
        start_turns: torch.Tensor | None = None,
    ) -> None:
        """Reset the selected worlds (all if no mask); see the backends' ``reset``.

        With ``start_turns``, ``(W,)`` rad, each selected world's body is
        turned about the vertical by its turn from the model pose (from
        ``model_heading_rad``) instead of a random one. With
        ``start_positions``, ``(W, 2)`` m, each selected world's body is
        then moved so that its root stands over its position, lifted by the
        highest ground within the body's reach of it, plus
        ``START_CLEARANCE_M``: whatever its heading, it starts above the
        ground and drops onto it.
        """
        root_offsets = None
        if start_positions is not None:
            lift = self.ground.highest_near(start_positions, self._body_reach_m)
            root = torch.as_tensor(self._root_qpos, dtype=torch.float32)
            root = root.to(start_positions.device)
            root_offsets = torch.cat(
                (start_positions - root[:2], lift[:, None] + START_CLEARANCE_M), dim=-1
            )
        self._backend.reset(world_mask, seed, root_offsets, start_turns)

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
