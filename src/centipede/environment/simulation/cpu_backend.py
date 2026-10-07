import mujoco
import numpy as np
import torch
from numpy.random import Generator

from centipede.environment.simulation.constants import (
    LEG_ANGLE_NOISE_RAD,
    LEG_SPEED_NOISE_RAD_S,
    PHYSICS_STEPS_PER_ACTION,
)
from centipede.environment.simulation.diagnostics import SimulationDiagnostics
from centipede.environment.simulation.model_mapping import (
    BODY_CATEGORY,
    FLOOR_CATEGORY,
    LEFT_FOOT_CATEGORY,
    RIGHT_FOOT_CATEGORY,
    ModelMapping,
)
from centipede.environment.simulation.physical_state import PhysicalState


class CPUBackend:
    """Runs every world with CPU MuJoCo, keeping one MjData per world.

    All worlds share one model. ``physical_state`` holds every world's current
    physical values; ``reset`` and ``step`` refresh it in place, and
    ``diagnostics`` (the simulation's category) with it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        mapping: ModelMapping,
        world_count: int,
    ) -> None:
        """Create the worlds, their reset generators, and the physical state.

        The generators start from seed 0 until ``reset`` is given a seed. Each
        physical-state tensor is also kept as a NumPy view sharing its memory,
        so writing MuJoCo values into a view fills the tensor directly.
        """
        self.model = model
        self.mapping = mapping
        self.world_count = world_count

        self.world_data: list[mujoco.MjData] = []
        self.reset_generators: list[Generator] = []
        for world_index in range(world_count):
            self.world_data.append(mujoco.MjData(self.model))
            self.reset_generators.append(np.random.default_rng((0, world_index)))

        self.physical_state = PhysicalState.allocate(
            world_count, mapping.segment_count, "cpu"
        )
        self._body_height = self.physical_state.body_height.numpy()
        self._body_quaternion = self.physical_state.body_quaternion.numpy()
        self._leg_joint_position = self.physical_state.leg_joint_position.numpy()
        self._body_linear_velocity = self.physical_state.body_linear_velocity.numpy()
        self._body_angular_velocity = self.physical_state.body_angular_velocity.numpy()
        self._leg_joint_velocity = self.physical_state.leg_joint_velocity.numpy()
        self._left_foot_ground_contact = (
            self.physical_state.left_foot_ground_contact.numpy()
        )
        self._right_foot_ground_contact = (
            self.physical_state.right_foot_ground_contact.numpy()
        )
        self._body_ground_contact = self.physical_state.body_ground_contact.numpy()
        self._leg_leg_contact = self.physical_state.leg_leg_contact.numpy()
        self._body_planar_position = self.physical_state.body_planar_position.numpy()
        self._head_tip_position = self.physical_state.head_tip_position.numpy()
        self.diagnostics = SimulationDiagnostics.allocate(world_count, model.nq, "cpu")

    def reset(
        self, world_mask: torch.Tensor | None = None, seed: int | None = None
    ) -> None:
        """Return the selected worlds to the model pose with small leg noise.

        ``world_mask`` is a boolean tensor of shape (W,) marking the worlds to
        reset; ``None`` resets every world. A ``seed`` restarts the selected
        worlds' random sequences, so the same seed repeats the same resets;
        without one, each world's sequence simply continues. Only the leg joints
        are varied: each angle by up to 2 degrees, each speed by a small normal
        amount. The selected worlds' rows of the physical state are refreshed.
        """
        if world_mask is None:
            selected_world_indices = range(self.world_count)
        else:
            selected_world_indices = world_mask.nonzero().flatten().tolist()

        noise_shape = (self.mapping.segment_count, 6)
        for world_index in selected_world_indices:
            if seed is not None:
                self.reset_generators[world_index] = np.random.default_rng(
                    (seed, world_index)
                )
            data = self.world_data[world_index]
            generator = self.reset_generators[world_index]

            mujoco.mj_resetData(self.model, data)
            # The (N, 6) address tables select all 48 leg joints of this world.
            data.qpos[self.mapping.leg_qpos_addresses] += generator.uniform(
                -LEG_ANGLE_NOISE_RAD, LEG_ANGLE_NOISE_RAD, noise_shape
            )
            data.qvel[self.mapping.leg_dof_addresses] = generator.normal(
                0.0, LEG_SPEED_NOISE_RAD_S, noise_shape
            )

            mujoco.mj_forward(self.model, data)
            self._read_world(world_index)
        self.diagnostics.fill_from_cpu(self.world_data)

    def step(self, leg_actions: torch.Tensor) -> None:
        """Hold each world's leg actions for one 20 ms step.

        ``leg_actions`` has shape (W, N, 6): every world's segment actions, in
        action order. Spine motors are never written and stay at zero. Stops the
        run if the physics fails; afterwards the physical state is refreshed.
        """
        actions = leg_actions.cpu().numpy()

        for world_index, data in enumerate(self.world_data):
            data.ctrl[self.mapping.leg_actuator_ids] = actions[world_index]
            for _ in range(PHYSICS_STEPS_PER_ACTION):
                mujoco.mj_step(self.model, data)
            self._check_world(world_index)

            mujoco.mj_forward(self.model, data)
            self._read_world(world_index)
        self.diagnostics.fill_from_cpu(self.world_data)

    def _check_world(self, world_index: int) -> None:
        """Stop the run if MuJoCo reported or produced an impossible state.

        MuJoCo does not raise on a numerical explosion: it records a warning and
        silently resets the world. The warning counts are cleared by every reset,
        so any count above zero is a failure since the world's last reset.
        """
        data = self.world_data[world_index]
        for warning, problem in (
            (mujoco.mjtWarning.mjWARN_BADQACC, "an invalid acceleration"),
            (mujoco.mjtWarning.mjWARN_BADQPOS, "an invalid position"),
            (mujoco.mjtWarning.mjWARN_BADQVEL, "an invalid velocity"),
        ):
            if data.warning[warning].number > 0:
                raise RuntimeError(
                    f"Physics failed in world {world_index}: MuJoCo reported {problem}"
                )

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError(
                f"Physics failed in world {world_index}: "
                "non-finite position or velocity"
            )

    def _read_world(self, world_index: int) -> None:
        """Copy one world's current physical values into its physical-state row."""
        data = self.world_data[world_index]
        mapping = self.mapping

        self._body_height[world_index] = data.site_xpos[mapping.center_site_ids, 2]
        self._body_planar_position[world_index] = data.site_xpos[
            mapping.center_site_ids, :2
        ]
        self._body_quaternion[world_index] = data.xquat[mapping.body_ids]
        self._leg_joint_position[world_index] = data.qpos[mapping.leg_qpos_addresses]
        self._leg_joint_velocity[world_index] = data.qvel[mapping.leg_dof_addresses]
        self._head_tip_position[world_index] = data.site_xpos[mapping.head_tip_site_id]

        site_velocity = np.zeros((6,), dtype=np.float64)

        for segment_index in range(mapping.segment_count):
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_SITE,
                mapping.center_site_ids[segment_index],
                site_velocity,
                1,
            )
            self._body_angular_velocity[world_index, segment_index] = site_velocity[:3]
            self._body_linear_velocity[world_index, segment_index] = site_velocity[3:]

        self._read_contacts(world_index)

    def _read_contacts(self, world_index: int) -> None:
        """Set one world's four contact flags from MuJoCo's current contact list.

        Each contact names the two shapes that touch (``geom1`` and ``geom2``).
        A shape touching the floor sets its owner's body, left-foot, or
        right-foot flag; a leg link touching the floor sets nothing. Two leg or
        foot shapes touching set the leg-leg flag of both owners. Model v2 allows
        no other pairs.
        """
        self._left_foot_ground_contact[world_index] = False
        self._right_foot_ground_contact[world_index] = False
        self._body_ground_contact[world_index] = False
        self._leg_leg_contact[world_index] = False

        data = self.world_data[world_index]
        owner_indices = self.mapping.geom_owner_indices
        categories = self.mapping.geom_categories
        ground_flags = {
            BODY_CATEGORY: self._body_ground_contact,
            LEFT_FOOT_CATEGORY: self._left_foot_ground_contact,
            RIGHT_FOOT_CATEGORY: self._right_foot_ground_contact,
        }

        for geom_id_a, geom_id_b in zip(
            data.contact.geom1, data.contact.geom2, strict=True
        ):
            if categories[geom_id_a] == FLOOR_CATEGORY:
                other_geom_id = geom_id_b
            elif categories[geom_id_b] == FLOOR_CATEGORY:
                other_geom_id = geom_id_a
            else:
                # Neither is the floor, so both are leg links or feet.
                self._leg_leg_contact[world_index, owner_indices[geom_id_a]] = True
                self._leg_leg_contact[world_index, owner_indices[geom_id_b]] = True
                continue

            ground_flag = ground_flags.get(int(categories[other_geom_id]))
            if ground_flag is not None:
                ground_flag[world_index, owner_indices[other_geom_id]] = True
