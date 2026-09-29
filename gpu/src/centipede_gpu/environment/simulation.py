"""Load and advance a batch of centipede worlds with MuJoCo Warp.

The simulation validates the shared XML and advances device state under batched
leg commands and selectively resets worlds. Physical-state extraction is pending.
Targets, rewards, episode rules, policies, and rendering belong elsewhere.

Mapping helpers preserve the CPU implementation's named XML contract without
importing the CPU package. Host arrays are initialization-time metadata; the
actuator mapping used during stepping is copied onto the device only once.

Constructor arguments and runtime inputs come from trusted application layers;
configuration is validated where it is loaded. Only the XML contract and the
physical results are checked here.
"""

from collections.abc import Iterable
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjModel  # pyright: ignore[reportAttributeAccessIssue]
from numpy.typing import NDArray

# Geometry user fields store the owning segment and its contact category.
_FLOOR_CATEGORY = 0
_BODY_CATEGORY = 1
_LEG_CATEGORY = 2
_FOOT_CATEGORY = 3
_DECORATIVE_CATEGORY = 4

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


class CentipedeSimulation:
    """Own one compiled model and the device state of several independent worlds.

    ``host_model`` is the ordinary MuJoCo model used for schema inspection.
    ``model`` and ``data`` are MJWarp objects on the selected CUDA device.
    This is a physical simulation, not a Gymnasium or PettingZoo task wrapper.
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
        This method does not step physics or return observations. Derived state
        and shared contact caches require a refresh before physical extraction.
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

    def step(self, leg_actions: wp.array3d[wp.float32]) -> None:
        """Hold one batch of leg actions for a complete control interval.

        Input is a trusted float32 Warp array on this simulation's device,
        shaped as ``action_shape`` (world, segment, six leg controls), with
        finite values in [-1, 1]. It is neither validated nor clipped here.

        State is updated in place. No observations, rewards, or episode endings
        are produced here. Calls use the current stream on the selected device;
        callers producing actions on another stream must order that work first.
        Derived positions and contacts need a forward refresh before extraction;
        the physical-state extraction operation is a separate pending increment.
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

        # Small diagnostic transfers deliberately synchronize this initial
        # stepping path. Invalid physics stops the run; never treat it as an
        # episode ending or silently reset the affected world. MJWarp keeps
        # overflow flags until reset, so every step reports earlier overflows.
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
