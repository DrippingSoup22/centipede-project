"""Load, advance, reset, and read a batch of centipede worlds with MuJoCo Warp.

The simulation validates the shared XML, advances device state under batched leg
commands, selectively resets worlds, and ends every step and reset by filling
``physical_state`` with the fields the task needs. Targets, rewards, episode
rules, policies, and rendering belong elsewhere.

Mapping helpers preserve the CPU implementation's named XML contract without
importing the CPU package. Host arrays are initialization-time metadata; the
mappings used during stepping and extraction are copied onto the device once.

Constructor arguments and runtime inputs come from trusted application layers;
configuration is validated where it is loaded. Only the XML contract and the
physical results are checked here.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjModel  # pyright: ignore[reportAttributeAccessIssue]
from numpy.typing import NDArray

# Geometry user fields store the owning segment and its contact category.
# wp.constant lets the contact kernel read the same values at compile time.
_FLOOR_CATEGORY = wp.constant(0)
_BODY_CATEGORY = wp.constant(1)
_LEG_CATEGORY = wp.constant(2)
_FOOT_CATEGORY = wp.constant(3)
_DECORATIVE_CATEGORY = wp.constant(4)

# Each segment's six policy commands retain the accepted left-then-right order.
LEG_ACTUATOR_ROLES = (
    ("left", "sweep"),
    ("left", "lift"),
    ("left", "knee"),
    ("right", "sweep"),
    ("right", "lift"),
    ("right", "knee"),
)

# The frozen model uses 0.1 ms physics steps: 200 steps form a 20 ms action.
FRAME_SKIP = 200

# Reset variation affects leg coordinates only, in radians and radians/second.
LEG_POSITION_NOISE_LIMIT = float(np.deg2rad(2.0))
LEG_VELOCITY_NOISE_STD = 0.05

# Capacity overflows invalidate a transition. Solver iteration limits only mark
# incomplete convergence, which the CPU MuJoCo reference also accepts.
_INVALID_OVERFLOWS = mjw.OverflowType.ALL & ~(
    mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
)


# These kernels prepare simulation data; PPO does not differentiate through them.
@wp.kernel(enable_backward=False)
def _write_leg_controls(
    # In:
    leg_actions_in: wp.array3d[float],
    actuator_ids_in: wp.array2d[int],
    # Data out:
    ctrl_out: wp.array2d[float],
):
    """Place each world's segment actions into the XML's actuator slots."""
    world_id, segment_id, action_id = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    actuator_id = actuator_ids_in[segment_id, action_id]  # pyright: ignore[reportIndexIssue]
    action = leg_actions_in[world_id, segment_id, action_id]  # pyright: ignore[reportIndexIssue]
    ctrl_out[world_id, actuator_id] = action  # pyright: ignore[reportIndexIssue]


@wp.kernel(enable_backward=False)
def _initialize_reset_random_states(
    # In:
    seed: wp.uint32,
    # Out:
    random_states_out: wp.array[wp.uint32],
):
    """Initialize a distinct Warp random sequence for each world's first reset."""
    world_id = wp.tid()
    random_states_out[world_id] = wp.rand_init(wp.int32(seed), world_id)  # pyright: ignore[reportIndexIssue, reportArgumentType]


@wp.kernel(enable_backward=False)
def _randomize_reset_legs(
    # In:
    reset_mask_in: wp.array[bool],
    qpos_ids_in: wp.array[int],
    qvel_ids_in: wp.array[int],
    seed: wp.uint32,
    reseed: bool,
    position_limit: float,
    velocity_std: float,
    # Data out:
    qpos_out: wp.array2d[float],
    qvel_out: wp.array2d[float],
    # Out:
    random_states_out: wp.array[wp.uint32],
):
    """Perturb selected worlds' legs and advance only their random sequences."""
    world_id = wp.tid()
    if not reset_mask_in[world_id]:
        return

    # One thread owns a whole world's draws, so its state has no concurrent writer.
    state = random_states_out[world_id]
    if reseed:
        state = wp.rand_init(wp.int32(seed), world_id)  # pyright: ignore[reportArgumentType]
    for leg_id in range(qpos_ids_in.shape[0]):
        position_noise = wp.randf(state, -position_limit, position_limit)  # pyright: ignore[reportArgumentType]
        velocity_noise = velocity_std * wp.randn(state)  # pyright: ignore[reportArgumentType]
        qpos_out[world_id, qpos_ids_in[leg_id]] += position_noise  # pyright: ignore[reportIndexIssue]
        qvel_out[world_id, qvel_ids_in[leg_id]] = velocity_noise  # pyright: ignore[reportIndexIssue]
    random_states_out[world_id] = state  # pyright: ignore[reportIndexIssue]


@wp.kernel(enable_backward=False)
def _find_nonfinite_worlds(
    # Model:
    nq: int,
    nv: int,
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    # Out:
    nonfinite_out: wp.array[bool],
):
    """Flag worlds whose integrated state contains NaN or infinite values."""
    world_id = wp.tid()
    # One thread owns each world's flag, so no atomic update is required.
    nonfinite_out[world_id] = False  # pyright: ignore[reportIndexIssue]
    for i in range(nq):
        if not wp.isfinite(qpos_in[world_id, i]):  # pyright: ignore[reportIndexIssue]
            nonfinite_out[world_id] = True  # pyright: ignore[reportIndexIssue]
    for i in range(nv):
        if not (
            wp.isfinite(qvel_in[world_id, i])  # pyright: ignore[reportIndexIssue]
            and wp.isfinite(qacc_in[world_id, i])  # pyright: ignore[reportIndexIssue]
        ):
            nonfinite_out[world_id] = True  # pyright: ignore[reportIndexIssue]


@wp.kernel(enable_backward=False)
def _extract_segment_state(
    # Model:
    site_bodyid: wp.array[int],
    body_rootid: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    xquat_in: wp.array2d[wp.quat],
    site_xpos_in: wp.array2d[wp.vec3],
    site_xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cvel_in: wp.array2d[wp.spatial_vector],
    # In:
    segment_body_ids_in: wp.array[int],
    segment_center_site_ids_in: wp.array[int],
    leg_qpos_ids_in: wp.array[int],
    leg_qvel_ids_in: wp.array[int],
    head_tip_site_id: int,
    # Out:
    body_height_out: wp.array2d[float],
    body_quaternion_out: wp.array3d[float],
    leg_joint_position_out: wp.array3d[float],
    leg_joint_velocity_out: wp.array3d[float],
    body_linear_velocity_out: wp.array3d[float],
    body_angular_velocity_out: wp.array3d[float],
    body_planar_position_out: wp.array3d[float],
    head_tip_position_out: wp.array2d[float],
):
    """Copy one segment's pose, leg state, and local velocity for one world."""
    world_id, segment_id = wp.tid()  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    site_id = segment_center_site_ids_in[segment_id]  # pyright: ignore[reportIndexIssue]
    center = site_xpos_in[world_id, site_id]  # pyright: ignore[reportIndexIssue]

    body_height_out[world_id, segment_id] = center[2]  # pyright: ignore[reportIndexIssue]
    body_planar_position_out[world_id, segment_id, 0] = center[0]  # pyright: ignore[reportIndexIssue]
    body_planar_position_out[world_id, segment_id, 1] = center[1]  # pyright: ignore[reportIndexIssue]

    # MJWarp stores MuJoCo's (w, x, y, z) order in wp.quat; copy it unchanged
    # and never pass it to Warp's quaternion helpers, which assume (x, y, z, w).
    quaternion = xquat_in[world_id, segment_body_ids_in[segment_id]]  # pyright: ignore[reportIndexIssue]
    for component in range(4):
        body_quaternion_out[world_id, segment_id, component] = quaternion[component]  # pyright: ignore[reportIndexIssue]

    # Leg ID tables are flat in segment-major action order.
    action_count = leg_joint_position_out.shape[2]
    for action_id in range(action_count):
        leg_id = segment_id * action_count + action_id
        leg_joint_position_out[world_id, segment_id, action_id] = qpos_in[  # pyright: ignore[reportIndexIssue]
            world_id, leg_qpos_ids_in[leg_id]  # pyright: ignore[reportIndexIssue]
        ]
        leg_joint_velocity_out[world_id, segment_id, action_id] = qvel_in[  # pyright: ignore[reportIndexIssue]
            world_id, leg_qvel_ids_in[leg_id]  # pyright: ignore[reportIndexIssue]
        ]

    # cvel holds angular then linear velocity, the linear part measured at the
    # root's subtree center of mass, both in world axes. A rigid body turns as
    # one piece, so v_center = v_reference + w x (center - reference); both
    # vectors are then rotated into the site's axes, as mj_objectVelocity does.
    body_id = site_bodyid[site_id]  # pyright: ignore[reportIndexIssue]
    velocity = cvel_in[world_id, body_id]  # pyright: ignore[reportIndexIssue]
    angular = wp.spatial_top(velocity)
    linear = wp.spatial_bottom(velocity)
    reference = subtree_com_in[world_id, body_rootid[body_id]]  # pyright: ignore[reportIndexIssue]
    world_to_site = wp.transpose(site_xmat_in[world_id, site_id])  # pyright: ignore[reportIndexIssue]
    local_angular = world_to_site @ angular
    local_linear = world_to_site @ (linear - wp.cross(center - reference, angular))
    for axis in range(3):
        body_angular_velocity_out[world_id, segment_id, axis] = local_angular[axis]  # pyright: ignore[reportIndexIssue]
        body_linear_velocity_out[world_id, segment_id, axis] = local_linear[axis]  # pyright: ignore[reportIndexIssue]

    # The head tip belongs to the head, so the head segment's thread copies it.
    if segment_id == 0:
        head_tip = site_xpos_in[world_id, head_tip_site_id]  # pyright: ignore[reportIndexIssue]
        for axis in range(3):
            head_tip_position_out[world_id, axis] = head_tip[axis]  # pyright: ignore[reportIndexIssue]


@wp.kernel(enable_backward=False)
def _classify_contacts(
    # Data in:
    nacon_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_worldid_in: wp.array[int],
    # In:
    geom_owners_in: wp.array[int],
    geom_categories_in: wp.array[int],
    foot_geom_ids_in: wp.array2d[int],
    floor_geom_id: int,
    # Out:
    left_foot_ground_contact_out: wp.array2d[bool],
    right_foot_ground_contact_out: wp.array2d[bool],
    body_ground_contact_out: wp.array2d[bool],
    leg_leg_contact_out: wp.array2d[bool],
):
    """Mark the segment flags implied by one slot of the shared contact pool."""
    contact_id = wp.tid()
    # The pool has a fixed capacity; only its first nacon slots hold contacts.
    if contact_id >= nacon_in[0]:  # pyright: ignore[reportIndexIssue]
        return

    # Several contacts may set one flag, but every writer stores True.
    world_id = contact_worldid_in[contact_id]  # pyright: ignore[reportIndexIssue]
    geoms = contact_geom_in[contact_id]  # pyright: ignore[reportIndexIssue]
    first_geom = geoms[0]
    second_geom = geoms[1]

    if first_geom == floor_geom_id or second_geom == floor_geom_id:
        other_geom = first_geom
        if first_geom == floor_geom_id:
            other_geom = second_geom
        owner = geom_owners_in[other_geom]  # pyright: ignore[reportIndexIssue]
        category = geom_categories_in[other_geom]  # pyright: ignore[reportIndexIssue]
        # Floor contact with a non-foot leg part sets no flag.
        if category == _BODY_CATEGORY:
            body_ground_contact_out[world_id, owner] = True  # pyright: ignore[reportIndexIssue]
        elif category == _FOOT_CATEGORY:
            if other_geom == foot_geom_ids_in[owner, 0]:  # pyright: ignore[reportIndexIssue]
                left_foot_ground_contact_out[world_id, owner] = True  # pyright: ignore[reportIndexIssue]
            elif other_geom == foot_geom_ids_in[owner, 1]:  # pyright: ignore[reportIndexIssue]
                right_foot_ground_contact_out[world_id, owner] = True  # pyright: ignore[reportIndexIssue]
        return

    # Leg parts include feet; both owners are marked, even within one segment.
    first_category = geom_categories_in[first_geom]  # pyright: ignore[reportIndexIssue]
    second_category = geom_categories_in[second_geom]  # pyright: ignore[reportIndexIssue]
    first_is_leg = first_category == _LEG_CATEGORY or first_category == _FOOT_CATEGORY
    second_is_leg = (
        second_category == _LEG_CATEGORY or second_category == _FOOT_CATEGORY
    )
    if first_is_leg and second_is_leg:
        leg_leg_contact_out[world_id, geom_owners_in[first_geom]] = True  # pyright: ignore[reportIndexIssue]
        leg_leg_contact_out[world_id, geom_owners_in[second_geom]] = True  # pyright: ignore[reportIndexIssue]


@dataclass(frozen=True)
class PhysicalState:
    """Device arrays describing every world after the latest step or reset.

    The simulation allocates one instance and overwrites it in place at the end
    of every ``step()`` and ``reset()``. Values stay valid until the next such
    call; readers never write to these arrays and copy anything they must keep.
    Shapes use W worlds and S segments; leg fields follow the action order.
    """

    body_height: wp.array2d[wp.float32]  # (W, S), metres
    body_quaternion: wp.array3d[wp.float32]  # (W, S, 4), (w, x, y, z)
    leg_joint_position: wp.array3d[wp.float32]  # (W, S, 6), radians
    leg_joint_velocity: wp.array3d[wp.float32]  # (W, S, 6), radians/second
    body_linear_velocity: wp.array3d[wp.float32]  # (W, S, 3), segment axes
    body_angular_velocity: wp.array3d[wp.float32]  # (W, S, 3), segment axes
    body_planar_position: wp.array3d[wp.float32]  # (W, S, 2), world x and y
    left_foot_ground_contact: wp.array2d[wp.bool]  # (W, S)
    right_foot_ground_contact: wp.array2d[wp.bool]  # (W, S)
    body_ground_contact: wp.array2d[wp.bool]  # (W, S)
    leg_leg_contact: wp.array2d[wp.bool]  # (W, S)
    head_tip_position: wp.array2d[wp.float32]  # (W, 3), world frame


class CentipedeSimulation:
    """Own one compiled model and the device state of several independent worlds.

    ``host_model`` is the ordinary MuJoCo model used for schema inspection.
    ``model`` and ``data`` are MJWarp objects on the selected CUDA device.
    ``physical_state`` is refreshed by every ``step()`` and ``reset()`` and is
    meaningful after the first reset. This is a physical simulation, not a
    Gymnasium or PettingZoo task wrapper.
    """

    def __init__(
        self,
        model_path: str | Path,
        world_count: int,
        *,
        nconmax: int,
        njmax: int,
        device: str = "cuda:0",
    ) -> None:
        """Validate the model once and allocate the requested batch on one GPU.

        ``nconmax`` and ``njmax`` keep MJWarp's names. ``nconmax`` budgets
        contacts per world on average, because MJWarp shares one contact
        allocation across worlds; ``njmax`` limits constraint rows in each
        world. They are required: MJWarp's heuristic defaults (48 and 64 for
        this model) are below the CPU reference peak of 256 constraint rows.
        """
        self.model_path = Path(model_path).expanduser().resolve()
        self.host_model = MjModel.from_xml_path(str(self.model_path))
        self.world_count = world_count
        self.frame_skip = FRAME_SKIP
        self.dt = float(self.host_model.opt.timestep) * self.frame_skip

        self.segment_ids = self._discover_segment_ids()
        self.leg_actuator_ids, self.spine_actuator_ids = self._resolve_actuator_ids()
        self._resolve_physical_ids()
        self.action_shape = (
            self.world_count,
            len(self.segment_ids),
            len(LEG_ACTUATOR_ROLES),
        )

        self.device = wp.get_device(device)

        # ScopedDevice selects placement without changing the caller's default
        # device. Unsupported model features are reported by MJWarp itself.
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.host_model)
            self.data = mjw.make_data(
                self.host_model,
                nworld=self.world_count,
                nconmax=nconmax,
                njmax=njmax,
            )
            self._device_leg_actuator_ids = wp.array(
                self.leg_actuator_ids, dtype=wp.int32
            )
            self._device_leg_qpos_ids = wp.array(
                self.leg_qpos_ids.ravel(), dtype=wp.int32
            )
            self._device_leg_qvel_ids = wp.array(
                self.leg_qvel_ids.ravel(), dtype=wp.int32
            )
            self._device_segment_body_ids = wp.array(
                self.segment_body_ids, dtype=wp.int32
            )
            self._device_segment_center_site_ids = wp.array(
                self.segment_center_site_ids, dtype=wp.int32
            )
            self._device_foot_geom_ids = wp.array(self.foot_geom_ids, dtype=wp.int32)
            self._device_geom_owners = wp.array(self._geom_owners, dtype=wp.int32)
            self._device_geom_categories = wp.array(
                self._geom_categories, dtype=wp.int32
            )
            self.physical_state = self._allocate_physical_state()
            self._all_worlds = wp.ones(self.world_count, dtype=wp.bool)
            # Warp annotates zeros(), unlike empty(), with a general dtype type.
            self._nonfinite_worlds = wp.zeros(self.world_count, dtype=wp.bool)
            # Entropy is acquired once. Subsequent random draws stay on-device;
            # an explicit reset seed replaces the selected worlds' RNG states.
            self._reset_random_states = wp.zeros(self.world_count, dtype=wp.uint32)
            wp.launch(
                _initialize_reset_random_states,
                dim=self.world_count,
                inputs=[int(np.random.SeedSequence().generate_state(1)[0])],
                outputs=[self._reset_random_states],
            )

    def reset(
        self,
        reset_mask: wp.array[wp.bool] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """Restore selected worlds to the XML pose, with small leg-only noise.

        ``reset_mask`` is a device boolean array of shape (world_count,); None
        selects all worlds. A supplied unsigned 32-bit seed restarts only the
        selected worlds' random sequences. Without a seed those sequences advance.
        Unselected worlds retain their integration state and random streams.

        MJWarp owns clearing controls, time, applied forces, and solver history.
        This method does not advance time or return observations. It ends by
        refreshing ``physical_state`` for every world, because MJWarp's forward
        pass has no world mask; unselected worlds keep identical values.
        """
        mask = self._all_worlds if reset_mask is None else reset_mask
        with wp.ScopedDevice(self.device):
            # The library validates mask shape and owns physical reset semantics.
            mjw.reset_data(self.model, self.data, reset=mask)
            wp.launch(
                _randomize_reset_legs,
                dim=self.world_count,
                inputs=[
                    mask,
                    self._device_leg_qpos_ids,
                    self._device_leg_qvel_ids,
                    0 if seed is None else seed,
                    seed is not None,
                    LEG_POSITION_NOISE_LIMIT,
                    LEG_VELOCITY_NOISE_STD,
                ],
                outputs=[self.data.qpos, self.data.qvel, self._reset_random_states],
            )
            self._refresh_physical_state()
        self._raise_on_invalid_physics()

    def step(self, leg_actions: wp.array3d[wp.float32]) -> None:
        """Hold one batch of leg actions for a complete control interval.

        Input is a trusted float32 Warp array on this simulation's device,
        shaped as ``action_shape`` (world, segment, six leg controls), with
        finite values in [-1, 1]. It is neither validated nor clipped here.

        State is updated in place and ``physical_state`` describes the end of
        the transition. No observations, rewards, or episode endings are
        produced here. Calls use the current stream on the selected device;
        callers producing actions on another stream must order that work first.
        """
        with wp.ScopedDevice(self.device):
            # Clear every motor first so spine controls cannot retain old values.
            self.data.ctrl.zero_()
            wp.launch(
                _write_leg_controls,
                dim=self.action_shape,
                inputs=[leg_actions, self._device_leg_actuator_ids],
                outputs=[self.data.ctrl],
            )
            for _ in range(self.frame_skip):
                mjw.step(self.model, self.data)
            self._refresh_physical_state()
        self._raise_on_invalid_physics()

    def _allocate_physical_state(self) -> PhysicalState:
        """Allocate the fixed device arrays that every refresh overwrites."""
        worlds_and_segments = (self.world_count, len(self.segment_ids))
        action_count = len(LEG_ACTUATOR_ROLES)

        def float_array(*trailing: int) -> wp.array:
            return wp.zeros((*worlds_and_segments, *trailing), dtype=wp.float32)

        def flag_array() -> wp.array:
            return wp.zeros(worlds_and_segments, dtype=wp.bool)

        return PhysicalState(
            body_height=float_array(),
            body_quaternion=float_array(4),
            leg_joint_position=float_array(action_count),
            leg_joint_velocity=float_array(action_count),
            body_linear_velocity=float_array(3),
            body_angular_velocity=float_array(3),
            body_planar_position=float_array(2),
            left_foot_ground_contact=flag_array(),
            right_foot_ground_contact=flag_array(),
            body_ground_contact=flag_array(),
            leg_leg_contact=flag_array(),
            head_tip_position=wp.zeros((self.world_count, 3), dtype=wp.float32),
        )

    def _refresh_physical_state(self) -> None:
        """Recompute derived quantities at the final state and fill its arrays.

        ``mjw.step`` computes positions and contacts before integrating, so they
        describe the state one physics step earlier until this forward pass.
        Callers select the device and check the results afterwards.
        """
        state = self.physical_state
        mjw.forward(self.model, self.data)
        wp.launch(
            _find_nonfinite_worlds,
            dim=self.world_count,
            inputs=[
                self.host_model.nq,
                self.host_model.nv,
                self.data.qpos,
                self.data.qvel,
                self.data.qacc,
            ],
            outputs=[self._nonfinite_worlds],
        )
        wp.launch(
            _extract_segment_state,
            dim=(self.world_count, len(self.segment_ids)),
            inputs=[
                self.model.site_bodyid,
                self.model.body_rootid,
                self.data.qpos,
                self.data.qvel,
                self.data.xquat,
                self.data.site_xpos,
                self.data.site_xmat,
                self.data.subtree_com,
                self.data.cvel,
                self._device_segment_body_ids,
                self._device_segment_center_site_ids,
                self._device_leg_qpos_ids,
                self._device_leg_qvel_ids,
                self.head_tip_site_id,
            ],
            outputs=[
                state.body_height,
                state.body_quaternion,
                state.leg_joint_position,
                state.leg_joint_velocity,
                state.body_linear_velocity,
                state.body_angular_velocity,
                state.body_planar_position,
                state.head_tip_position,
            ],
        )

        # Contact threads only set flags, so every refresh starts from False.
        contact_flags = (
            state.left_foot_ground_contact,
            state.right_foot_ground_contact,
            state.body_ground_contact,
            state.leg_leg_contact,
        )
        for flags in contact_flags:
            flags.zero_()
        # One thread per pool slot: nacon lives on the device, so reading it
        # to size the launch would make the host wait for the GPU.
        wp.launch(
            _classify_contacts,
            dim=self.data.naconmax,
            inputs=[
                self.data.nacon,
                self.data.contact.geom,
                self.data.contact.worldid,
                self._device_geom_owners,
                self._device_geom_categories,
                self._device_foot_geom_ids,
                self.floor_geom_id,
            ],
            outputs=list(contact_flags),
        )

    def _raise_on_invalid_physics(self) -> None:
        """Stop the run on capacity overflows or non-finite physical state."""
        # Small diagnostic transfers deliberately synchronize this
        # correctness-first path; Stage 7 may defer them. Invalid physics stops
        # the run; never treat it as an episode ending or silently reset the
        # affected world. MJWarp keeps overflow flags until reset, so every
        # check also reports earlier overflows.
        overflow = self.data.overflow.numpy() & _INVALID_OVERFLOWS
        if np.any(overflow):
            failures = "; ".join(
                f"world {world_id}: "
                + ", ".join(
                    str(flag.name) for flag in mjw.OverflowType(int(overflow[world_id]))
                )
                for world_id in np.flatnonzero(overflow)
            )
            raise RuntimeError(f"MJWarp capacity overflow ({failures})")

        nonfinite = self._nonfinite_worlds.numpy()
        if np.any(nonfinite):
            raise RuntimeError(
                f"Non-finite physical state in worlds {np.flatnonzero(nonfinite).tolist()}"
            )

    def _discover_segment_ids(self) -> tuple[int, ...]:
        """Read segment owners from the XML and require consecutive identifiers."""
        if self.host_model.nuser_actuator < 1:
            raise ValueError("The model must define actuator ownership metadata")

        owners = np.unique(self.host_model.actuator_user[:, 0])
        expected = np.arange(len(owners))

        if not np.array_equal(owners, expected):
            raise ValueError("Segment owners must be consecutive IDs starting at 0")

        return tuple(int(owner) for owner in owners)

    def _model_id(self, element: str, name: str) -> int:
        """Resolve one named MuJoCo element and report a readable schema error."""
        try:
            return int(getattr(self.host_model, element)(name).id)
        except KeyError as error:
            raise ValueError(f"Missing {element}: {name}") from error

    def _model_ids(self, element: str, names: Iterable[str]) -> NDArray[np.int32]:
        """Resolve several named MuJoCo elements into one integer array."""
        return np.fromiter(
            (self._model_id(element, name) for name in names),
            dtype=np.int32,
        )

    def _resolve_actuator_id(
        self,
        actuator_name: str,
        joint_name: str,
        segment_id: int,
    ) -> int:
        """Resolve one named actuator and validate its joint, owner, and range."""
        # MuJoCo's named view exposes this actuator's own fields as small arrays.
        actuator = self.host_model.actuator(self._model_id("actuator", actuator_name))
        joint_id = self._model_id("joint", joint_name)

        if actuator.user[0] != segment_id:
            raise ValueError(
                f"{actuator_name} has owner {actuator.user[0]}, expected {segment_id}"
            )

        if actuator.trnid[0] != joint_id:
            raise ValueError(f"{actuator_name} does not control {joint_name}")

        if not actuator.ctrllimited[0]:
            raise ValueError(f"{actuator_name} must have a limited control range")

        if not np.array_equal(actuator.ctrlrange, (-1.0, 1.0)):
            raise ValueError(f"{actuator_name} must use control range [-1, 1]")

        return actuator.id

    def _resolve_actuator_ids(
        self,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Map leg motors and spine motors reserved for zero control."""
        leg_ids = np.array(
            [
                [
                    self._resolve_actuator_id(
                        actuator_name=f"segment_{segment_id:02d}_{side}_{role}_motor",
                        joint_name=f"segment_{segment_id:02d}_{side}_{role}",
                        segment_id=segment_id,
                    )
                    for side, role in LEG_ACTUATOR_ROLES
                ]
                for segment_id in self.segment_ids
            ],
            dtype=np.int32,
        )
        spine_ids = np.array(
            [
                self._resolve_actuator_id(
                    actuator_name=f"segment_{segment_id:02d}_yaw_motor",
                    joint_name=f"segment_{segment_id:02d}_yaw",
                    segment_id=segment_id,
                )
                for segment_id in self.segment_ids[1:]
            ],
            dtype=np.int32,
        )

        mapped_ids = np.concatenate((leg_ids.ravel(), spine_ids))

        if not np.array_equal(np.sort(mapped_ids), np.arange(self.host_model.nu)):
            raise ValueError("Every MuJoCo actuator must be mapped exactly once")

        return leg_ids, spine_ids

    def _resolve_physical_ids(self) -> None:
        """Cache joint, body, site, and geometry IDs needed after construction."""
        leg_joint_ids = self.host_model.actuator_trnid[self.leg_actuator_ids, 0]
        self.leg_qpos_ids = self.host_model.jnt_qposadr[leg_joint_ids].astype(np.int32)
        self.leg_qvel_ids = self.host_model.jnt_dofadr[leg_joint_ids].astype(np.int32)

        prefixes = tuple(f"segment_{segment_id:02d}" for segment_id in self.segment_ids)
        self.segment_body_ids = self._model_ids("body", prefixes)
        self.segment_center_site_ids = self._model_ids(
            "site", (f"{prefix}_center" for prefix in prefixes)
        )
        self.segment_body_geom_ids = self._model_ids(
            "geom", (f"{prefix}_body" for prefix in prefixes)
        )
        self.foot_geom_ids = self._model_ids(
            "geom",
            (
                f"{prefix}_{side}_foot"
                for prefix in prefixes
                for side in ("left", "right")
            ),
        ).reshape((-1, 2))
        self.head_tip_site_id = self._model_id("site", "head_tip")
        self.floor_geom_id = self._model_id("geom", "floor")

        if self.host_model.nuser_geom < 2:
            raise ValueError("The model must define geometry ownership and categories")

        # XML user fields are floating-point values, but these two fields encode
        # integer IDs. Validate before converting so malformed IDs cannot round.
        geometry_metadata = self.host_model.geom_user[:, :2]
        if not np.all(np.isfinite(geometry_metadata)) or not np.array_equal(
            geometry_metadata, np.floor(geometry_metadata)
        ):
            raise ValueError("Geometry owners and categories must be finite integers")
        self._geom_owners = geometry_metadata[:, 0].astype(np.int32)
        self._geom_categories = geometry_metadata[:, 1].astype(np.int32)
        self._validate_geometry_metadata()

    def _validate_geometry_metadata(self) -> None:
        """Validate contact owners and categories stored in the loaded XML."""
        owners = self._geom_owners
        categories = self._geom_categories
        segment_ids = np.asarray(self.segment_ids, dtype=np.int32)

        if (
            owners[self.floor_geom_id] != -1
            or categories[self.floor_geom_id] != _FLOOR_CATEGORY
        ):
            raise ValueError("The floor geometry must use owner -1 and category 0")

        robot = np.isin(categories, (_BODY_CATEGORY, _LEG_CATEGORY, _FOOT_CATEGORY))
        decorative = categories == _DECORATIVE_CATEGORY
        non_floor = np.arange(self.host_model.ngeom) != self.floor_geom_id
        collisions = (self.host_model.geom_contype != 0) | (
            self.host_model.geom_conaffinity != 0
        )

        if np.any(non_floor & ~(robot | decorative)):
            raise ValueError("Every geometry must use a known contact category")
        if np.any(robot & ~np.isin(owners, segment_ids)):
            raise ValueError("Every physical geometry must have a segment owner")
        if np.any(decorative & collisions):
            raise ValueError("Decorative geometries must not collide")
        if not np.array_equal(owners[self.segment_body_geom_ids], segment_ids):
            raise ValueError("Body geometry ownership does not match segment IDs")
        if np.any(categories[self.segment_body_geom_ids] != _BODY_CATEGORY):
            raise ValueError("Segment body geometries must use the body category")
        expected_foot_owners = np.repeat(segment_ids, 2)
        if not np.array_equal(owners[self.foot_geom_ids].ravel(), expected_foot_owners):
            raise ValueError("Foot geometry ownership does not match segment IDs")
        if np.any(categories[self.foot_geom_ids] != _FOOT_CATEGORY):
            raise ValueError("Segment foot geometries must use the foot category")
