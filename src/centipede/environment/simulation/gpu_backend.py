# Warp kernels are checked by Warp's own compiler when first launched. The
# editor's type checker misreads their indexing, number types and vector
# maths, so the rules those false alarms come from are off in this file.
# pyright: reportIndexIssue=false, reportAssignmentType=false, reportGeneralTypeIssues=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from centipede.environment.simulation.constants import (
    LEG_ANGLE_NOISE_RAD,
    LEG_SPEED_NOISE_RAD_S,
)
from centipede.environment.simulation.diagnostics import (
    SimulationDiagnostics,
    SimulationFacts,
)
from centipede.environment.simulation.model_mapping import (
    BODY_CATEGORY,
    FLOOR_CATEGORY,
    LEFT_FOOT_CATEGORY,
    RIGHT_FOOT_CATEGORY,
    ModelMapping,
)
from centipede.environment.simulation.physical_state import PhysicalState

SOLVERS = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}
# MuJoCo Warp's per-world overflow bits that mean data was dropped. The solver's
# iteration-limit bits are only notices, so they are not checked.
CONSTRAINT_OVERFLOWS = mjw.OverflowType.NEFC | mjw.OverflowType.NJMAX_NNZ
CONTACT_OVERFLOWS = mjw.OverflowType.BROADPHASE | mjw.OverflowType.NARROWPHASE
CAPACITY_OVERFLOWS = CONSTRAINT_OVERFLOWS | CONTACT_OVERFLOWS


@wp.kernel
def _apply_leg_actions(
    # In
    leg_actuator_ids: wp.array2d[wp.int32],
    leg_actions: wp.array3d[float],
    # Out
    ctrl: wp.array2d[float],
) -> None:
    """Write each world's leg actions into its row of ``ctrl``.

    Launched with one thread per (world, segment, leg joint). Each thread copies
    one action to the motor that drives its joint; the spine motors are never
    written, so they keep the zero command set by the last reset.
    """
    world_index, segment_index, leg_joint_index = wp.tid()
    motor_id = leg_actuator_ids[segment_index, leg_joint_index]
    ctrl[world_index, motor_id] = leg_actions[
        world_index, segment_index, leg_joint_index
    ]


@wp.kernel
def _add_leg_reset_noise(
    # In
    world_mask: wp.array1d[wp.bool],
    reset_seeds: wp.array1d[wp.int32],
    reset_counts: wp.array1d[wp.int32],
    leg_qpos_addresses: wp.array2d[wp.int32],
    leg_dof_addresses: wp.array2d[wp.int32],
    angle_noise: float,
    speed_noise: float,
    # Out
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
) -> None:
    """Vary each leg joint of the masked worlds after ``reset_data``.

    Launched with one thread per (world, segment, leg joint). Threads of worlds
    outside the mask return at once. Each other thread offsets its joint angle
    uniformly within ``angle_noise`` and sets its speed from a normal spread of
    ``speed_noise``. Its random stream depends only on the world's seed and
    reset count and on its own thread number, so equal seeds repeat equal
    resets and every later reset draws new numbers.
    """
    world_index, segment_index, leg_joint_index = wp.tid()
    if not world_mask[world_index]:
        return

    # Mix the world's seed with its reset count, then with this thread's number.
    reset_seed = wp.int32(
        wp.rand_init(reset_seeds[world_index], reset_counts[world_index])
    )
    segment_count = leg_qpos_addresses.shape[0]
    thread_number = (world_index * segment_count + segment_index) * 6 + leg_joint_index
    random_state = wp.rand_init(reset_seed, thread_number)

    qpos_address = leg_qpos_addresses[segment_index, leg_joint_index]
    dof_address = leg_dof_addresses[segment_index, leg_joint_index]
    qpos[world_index, qpos_address] += wp.randf(random_state, -angle_noise, angle_noise)
    qvel[world_index, dof_address] = wp.randn(random_state) * speed_noise


@wp.kernel
def _read_segment_state(
    # In: tables
    center_site_ids: wp.array1d[int],
    body_ids: wp.array1d[int],
    leg_qpos_addresses: wp.array2d[int],
    leg_dof_addresses: wp.array2d[int],
    head_tip_site_id: int,
    # In: model
    site_bodyid: wp.array1d[int],
    body_rootid: wp.array1d[int],
    # In: data
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    xquat: wp.array2d[wp.quat],
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    cvel: wp.array2d[wp.spatial_vector],
    subtree_com: wp.array2d[wp.vec3],
    # Out
    body_height: wp.array2d[float],
    body_planar_position: wp.array2d[wp.vec2],
    body_quaternion: wp.array2d[wp.quat],
    body_linear_velocity: wp.array2d[wp.vec3],
    body_angular_velocity: wp.array2d[wp.vec3],
    leg_joint_position: wp.array3d[float],
    leg_joint_velocity: wp.array3d[float],
    head_tip_position: wp.array1d[wp.vec3],
) -> None:
    """Copy each segment's position, orientation, leg joints and velocity.

    Launched with one thread per (world, segment); each thread writes only its
    own segment's row. The velocity is what ``mj_objectVelocity`` returns for
    the segment's centre site in the site's own axes: MuJoCo stores each body's
    velocity at a shared reference point (the centre of mass of its whole
    tree), so it is first moved to the site, then turned into site axes.
    """
    world_index, segment_index = wp.tid()
    site_id = center_site_ids[segment_index]
    site_position = site_xpos[world_index, site_id]
    body_height[world_index, segment_index] = site_position[2]
    body_planar_position[world_index, segment_index] = wp.vec2(
        site_position[0], site_position[1]
    )

    body_id = body_ids[segment_index]
    body_quaternion[world_index, segment_index] = xquat[world_index, body_id]

    for leg_joint_index in range(6):
        qpos_address = leg_qpos_addresses[segment_index, leg_joint_index]
        dof_address = leg_dof_addresses[segment_index, leg_joint_index]
        leg_joint_position[world_index, segment_index, leg_joint_index] = qpos[
            world_index, qpos_address
        ]
        leg_joint_velocity[world_index, segment_index, leg_joint_index] = qvel[
            world_index, dof_address
        ]

    # The site's body velocity, stored at its tree's centre of mass in world axes.
    site_body_id = site_bodyid[site_id]
    body_velocity = cvel[world_index, site_body_id]
    angular_velocity = wp.spatial_top(body_velocity)
    reference_point = subtree_com[world_index, body_rootid[site_body_id]]
    # Move it to the site (rigid body: v_site = v_ref + w x (site - ref)) ...
    offset = site_position - reference_point
    linear_velocity = wp.spatial_bottom(body_velocity) - wp.cross(
        offset, angular_velocity
    )
    # ... and turn both parts from world axes into the site's own axes.
    world_to_site = wp.transpose(site_xmat[world_index, site_id])
    body_angular_velocity[world_index, segment_index] = world_to_site @ angular_velocity
    body_linear_velocity[world_index, segment_index] = world_to_site @ linear_velocity

    if segment_index == 0:
        head_tip_position[world_index] = site_xpos[world_index, head_tip_site_id]


@wp.kernel
def _read_contact_flags(
    # In: data
    nacon: wp.array1d[int],
    contact_geom: wp.array1d[wp.vec2i],
    contact_worldid: wp.array1d[int],
    # In: tables
    geom_owner_indices: wp.array1d[int],
    geom_categories: wp.array1d[int],
    # Out
    left_foot_ground_contact: wp.array2d[wp.bool],
    right_foot_ground_contact: wp.array2d[wp.bool],
    body_ground_contact: wp.array2d[wp.bool],
    leg_leg_contact: wp.array2d[wp.bool],
) -> None:
    """Set the contact flags from MuJoCo Warp's shared contact pool.

    Launched with one thread per pool slot; only the first ``nacon`` slots hold
    contacts, from all worlds mixed, each tagged with its world. The rules are
    those of the CPU backend: a shape touching the floor sets its owner's body,
    left-foot or right-foot flag (a leg link sets nothing), and two leg or foot
    shapes touching set the leg-leg flag of both owners. The flags must be
    cleared before the launch. Several threads may set the same flag, but they
    all write True, so the result does not depend on their order.
    """
    contact_index = wp.tid()
    if contact_index >= nacon[0]:
        return

    world_index = contact_worldid[contact_index]
    pair = contact_geom[contact_index]
    geom_id_a = pair[0]
    geom_id_b = pair[1]

    if geom_categories[geom_id_a] == FLOOR_CATEGORY:
        other_geom_id = geom_id_b
    elif geom_categories[geom_id_b] == FLOOR_CATEGORY:
        other_geom_id = geom_id_a
    else:
        # Neither is the floor, so both are leg links or feet.
        leg_leg_contact[world_index, geom_owner_indices[geom_id_a]] = True
        leg_leg_contact[world_index, geom_owner_indices[geom_id_b]] = True
        return

    other_category = geom_categories[other_geom_id]
    owner_index = geom_owner_indices[other_geom_id]
    if other_category == BODY_CATEGORY:
        body_ground_contact[world_index, owner_index] = True
    elif other_category == LEFT_FOOT_CATEGORY:
        left_foot_ground_contact[world_index, owner_index] = True
    elif other_category == RIGHT_FOOT_CATEGORY:
        right_foot_ground_contact[world_index, owner_index] = True


class GPUBackend:
    """Runs every world at once on the GPU with MuJoCo Warp.

    All worlds live in one MuJoCo Warp data object, whose arrays have the world
    as their first dimension. ``physical_state`` holds every world's current
    physical values on the GPU; ``reset`` and ``step`` refresh it in place. The
    physics steps of an action are replayed from a recorded CUDA graph.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        mapping: ModelMapping,
        world_count: int,
        physics_steps_per_action: int,
        solver: str,
        contacts_per_world: int,
        constraints_per_world: int,
    ) -> None:
        """Copy the model to the GPU and allocate everything the worlds need.

        ``physics_steps_per_action`` is how many of the model's timesteps make
        one 20 ms action. ``solver`` is "newton" or "cg"; it is set on the host
        model before the copy, because MuJoCo Warp reads it while building the
        GPU model. The
        contact and constraint capacities are reserved here once: contacts in
        one pool shared by all worlds, constraints separately for each world.
        Each physical-state tensor is also kept as a Warp view sharing its
        memory, so kernels writing a view fill the tensor directly.
        """
        model.opt.solver = SOLVERS[solver]
        self.gpu_model = mjw.put_model(model)
        self.gpu_data = mjw.make_data(
            model,
            nworld=world_count,
            nconmax=contacts_per_world,
            njmax=constraints_per_world,
        )
        self.world_count = world_count
        self.segment_count = mapping.segment_count
        self.physics_steps_per_action = physics_steps_per_action

        # The Warp device that holds the model and data, and its PyTorch name.
        self.device = wp.get_device()
        torch_device = wp.device_to_torch(self.device)

        def to_gpu(table: np.ndarray) -> wp.array:
            return wp.array(table, dtype=wp.int32, device=self.device)

        self.leg_actuator_ids = to_gpu(mapping.leg_actuator_ids)
        self.leg_qpos_addresses = to_gpu(mapping.leg_qpos_addresses)
        self.leg_dof_addresses = to_gpu(mapping.leg_dof_addresses)
        self.body_ids = to_gpu(mapping.body_ids)
        self.center_site_ids = to_gpu(mapping.center_site_ids)
        self.geom_owner_indices = to_gpu(mapping.geom_owner_indices)
        self.geom_categories = to_gpu(mapping.geom_categories)
        self.head_tip_site_id = mapping.head_tip_site_id

        self.physical_state = PhysicalState.allocate(
            world_count, mapping.segment_count, torch_device
        )
        state = self.physical_state
        self._body_height = wp.from_torch(state.body_height)
        self._body_planar_position = wp.from_torch(
            state.body_planar_position, dtype=wp.vec2
        )
        # Kept in MuJoCo's (w, x, y, z) order, as MuJoCo Warp stores it.
        self._body_quaternion = wp.from_torch(state.body_quaternion, dtype=wp.quat)
        self._body_linear_velocity = wp.from_torch(
            state.body_linear_velocity, dtype=wp.vec3
        )
        self._body_angular_velocity = wp.from_torch(
            state.body_angular_velocity, dtype=wp.vec3
        )
        self._leg_joint_position = wp.from_torch(state.leg_joint_position)
        self._leg_joint_velocity = wp.from_torch(state.leg_joint_velocity)
        self._head_tip_position = wp.from_torch(state.head_tip_position, dtype=wp.vec3)
        self._left_foot_ground_contact = wp.from_torch(state.left_foot_ground_contact)
        self._right_foot_ground_contact = wp.from_torch(state.right_foot_ground_contact)
        self._body_ground_contact = wp.from_torch(state.body_ground_contact)
        self._leg_leg_contact = wp.from_torch(state.leg_leg_contact)

        # The simulation's category, as views of MuJoCo Warp's own arrays: the
        # physics refreshes them, so nothing is copied per step.
        self.diagnostics = SimulationDiagnostics(
            SimulationFacts(
                qpos=wp.to_torch(self.gpu_data.qpos),
                contact_count=wp.to_torch(self.gpu_data.nacon),
                constraint_rows=wp.to_torch(self.gpu_data.nefc),
                solver_iterations=wp.to_torch(self.gpu_data.solver_niter),
            )
        )

        # The physics steps of one action, recorded as a CUDA graph by the
        # first step; see ``step``.
        self._physics_graph: wp.Graph | None = None

        # Each world's random sequence: the seed it was last given and how many
        # resets it has had since. Together they replace the CPU generators.
        self.reset_seeds = torch.zeros(
            world_count, dtype=torch.int32, device=torch_device
        )
        self.reset_counts = torch.zeros(
            world_count, dtype=torch.int32, device=torch_device
        )

    def step(self, leg_actions: torch.Tensor) -> None:
        """Hold the (W, N, 6) leg actions for 20 ms in every world at once.

        The actions are written into ``ctrl`` once, then the physics runs its
        steps for the action (134 for model v3). Launched from Python, each
        step's solver would make the CPU wait for the GPU after every
        iteration, to decide whether to iterate again.
        So the first step runs them from Python, which also compiles every
        kernel, and then records them as a CUDA graph; later steps replay the
        recording, in which the GPU decides itself when the solver is done.
        The recording holds the addresses of MuJoCo Warp's arrays, which never
        move; the actions, a new tensor every step, are written before it. The
        check after the steps is the step's only wait. A final ``forward``
        updates the values derived from the new positions, such as site
        positions and contacts, before the physical state is refreshed.
        """
        leg_action_view = wp.from_torch(leg_actions)
        wp.launch(
            kernel=_apply_leg_actions,
            dim=(self.world_count, self.segment_count, 6),
            inputs=[self.leg_actuator_ids, leg_action_view],
            outputs=[self.gpu_data.ctrl],
            device=self.device,
        )
        if self._physics_graph is None:
            self._run_physics_steps()
            # Recording only writes the launches down; nothing runs twice.
            with wp.ScopedCapture(device=self.device) as capture:
                self._run_physics_steps()
            self._physics_graph = capture.graph
        else:
            wp.capture_launch(self._physics_graph)
        self._check_worlds()
        mjw.forward(self.gpu_model, self.gpu_data)
        self._read_state()

    def reset(
        self,
        world_mask: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        """Return the selected worlds to the model pose with small leg noise.

        ``world_mask`` is a boolean tensor of shape (W,) marking the worlds to
        reset; ``None`` resets every world. A ``seed`` restarts the selected
        worlds' random sequences, so the same seed repeats the same resets;
        without one, each world's sequence continues with its next reset count.
        """
        if world_mask is None:
            mask = torch.ones(
                self.world_count, dtype=torch.bool, device=self.reset_seeds.device
            )
        else:
            mask = world_mask
        if seed is not None:
            self.reset_seeds.masked_fill_(mask, seed)
            self.reset_counts.masked_fill_(mask, 0)

        mask_view = wp.from_torch(mask)
        mjw.reset_data(self.gpu_model, self.gpu_data, mask_view)
        wp.launch(
            _add_leg_reset_noise,
            dim=(self.world_count, self.segment_count, 6),
            inputs=[
                mask_view,
                self.reset_seeds,
                self.reset_counts,
                self.leg_qpos_addresses,
                self.leg_dof_addresses,
                LEG_ANGLE_NOISE_RAD,
                LEG_SPEED_NOISE_RAD_S,
            ],
            outputs=[
                self.gpu_data.qpos,
                self.gpu_data.qvel,
            ],
            device=self.device,
        )
        self.reset_counts += mask
        mjw.forward(self.gpu_model, self.gpu_data)
        self._read_state()

    def _run_physics_steps(self) -> None:
        """The physics steps of one action, launched from Python."""
        for _ in range(self.physics_steps_per_action):
            mjw.step(self.gpu_model, self.gpu_data)

    def _check_worlds(self) -> None:
        """Stop the run if any world's physics went wrong during the last step.

        MuJoCo Warp records, per world, every capacity overflow since the last
        reset, but it does not detect non-finite values, so both are checked
        here. The ``if`` is the step's only wait for the GPU; the failing world
        is looked up only after something went wrong.
        """
        overflow_bits = wp.to_torch(self.gpu_data.overflow)
        overflowed_worlds = (overflow_bits & CAPACITY_OVERFLOWS) != 0
        finite_qpos = torch.isfinite(wp.to_torch(self.gpu_data.qpos)).all(dim=1)
        finite_qvel = torch.isfinite(wp.to_torch(self.gpu_data.qvel)).all(dim=1)
        broken_worlds = overflowed_worlds | ~(finite_qpos & finite_qvel)

        if broken_worlds.any():
            world_index = int(torch.nonzero(broken_worlds)[0].item())
            world_overflows = int(overflow_bits[world_index].item())
            if world_overflows & CONTACT_OVERFLOWS:
                raise RuntimeError(
                    "Physics failed: the worlds together needed more than the "
                    f"{self.gpu_data.naconmax} contacts reserved for them, so "
                    "some contacts were dropped. Raise contacts_per_world in "
                    "[environment.simulation] (now "
                    f"{self.gpu_data.naconmax // self.world_count})."
                )
            if world_overflows & CONSTRAINT_OVERFLOWS:
                raise RuntimeError(
                    f"Physics failed in world {world_index}: it needed more than "
                    f"the {self.gpu_data.njmax} constraints reserved per world, so "
                    "some were dropped. Raise constraints_per_world in "
                    "[environment.simulation]."
                )
            raise RuntimeError(
                f"Physics failed in world {world_index}: MuJoCo Warp produced a "
                "non-finite position or velocity (the simulation became unstable)."
            )

    def _read_state(self) -> None:
        """Refresh every world's physical state from MuJoCo Warp's data.

        Run after ``forward``, so that site positions, orientations, velocities
        and contacts all belong to the current positions and speeds. The
        contact kernel only sets flags, so they are cleared before it runs.
        """
        wp.launch(
            kernel=_read_segment_state,
            dim=(self.world_count, self.segment_count),
            inputs=[
                self.center_site_ids,
                self.body_ids,
                self.leg_qpos_addresses,
                self.leg_dof_addresses,
                self.head_tip_site_id,
                self.gpu_model.site_bodyid,
                self.gpu_model.body_rootid,
                self.gpu_data.site_xpos,
                self.gpu_data.site_xmat,
                self.gpu_data.xquat,
                self.gpu_data.qpos,
                self.gpu_data.qvel,
                self.gpu_data.cvel,
                self.gpu_data.subtree_com,
            ],
            outputs=[
                self._body_height,
                self._body_planar_position,
                self._body_quaternion,
                self._body_linear_velocity,
                self._body_angular_velocity,
                self._leg_joint_position,
                self._leg_joint_velocity,
                self._head_tip_position,
            ],
            device=self.device,
        )

        self._left_foot_ground_contact.zero_()
        self._right_foot_ground_contact.zero_()
        self._body_ground_contact.zero_()
        self._leg_leg_contact.zero_()

        wp.launch(
            _read_contact_flags,
            dim=self.gpu_data.naconmax,
            inputs=[
                self.gpu_data.nacon,
                self.gpu_data.contact.geom,
                self.gpu_data.contact.worldid,
                self.geom_owner_indices,
                self.geom_categories,
            ],
            outputs=[
                self._left_foot_ground_contact,
                self._right_foot_ground_contact,
                self._body_ground_contact,
                self._leg_leg_contact,
            ],
            device=self.device,
        )
