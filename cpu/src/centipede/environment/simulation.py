"""MuJoCo-backed physical simulation used by the Centipede task environment."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray

# Reset varies only leg joints around the XML pose; the spine and body stay fixed.
LEG_POSITION_NOISE_LIMIT = np.deg2rad(2.0)
LEG_VELOCITY_NOISE_STD = 0.05

# Integer categories mirror the two geom_user metadata fields stored in the XML.
_FLOOR_CATEGORY = 0
_BODY_CATEGORY = 1
_LEG_CATEGORY = 2
_FOOT_CATEGORY = 3
_DECORATIVE_CATEGORY = 4
_LEG_CATEGORIES = (_LEG_CATEGORY, _FOOT_CATEGORY)

# This order is the stable six-value policy action contract for every segment.
LEG_ACTUATOR_ROLES = (
    ("left", "sweep"),
    ("left", "lift"),
    ("left", "knee"),
    ("right", "sweep"),
    ("right", "lift"),
    ("right", "knee"),
)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    """Copy an array and prevent accidental mutation of snapshot data."""
    result = values.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicalSnapshot:
    """Copied physical state and contact flags for every centipede segment."""

    # Pose and motion are ordered by segment ID; leg fields follow action order.
    body_height: NDArray[np.float64]
    body_quaternion: NDArray[np.float64]
    leg_joint_position: NDArray[np.float64]
    leg_joint_velocity: NDArray[np.float64]
    body_linear_velocity: NDArray[np.float64]
    body_angular_velocity: NDArray[np.float64]
    body_planar_position: NDArray[np.float64]
    # Contact arrays mark the owning segment in the final transition state.
    left_foot_ground_contact: NDArray[np.bool_]
    right_foot_ground_contact: NDArray[np.bool_]
    body_ground_contact: NDArray[np.bool_]
    leg_leg_contact: NDArray[np.bool_]
    # The task layer uses this world-space point to place and detect targets.
    head_tip_position: NDArray[np.float64]


class CentipedeSimulation(MujocoEnv):
    """Manage the complete centipede model as one internal Gymnasium simulation.

    The class loads the configured XML, validates and caches actuator mappings,
    converts enabled leg actions into full MuJoCo controls, and advances physics.
    Targets, partial observations, rewards, and episodes belong to the public
    PettingZoo environment rather than this physical layer.
    """

    # Gymnasium uses this metadata to configure rendering and playback timing.
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        model_path: str | Path,
        render_mode: str | None = None,
    ) -> None:
        """Load one model and prepare its fixed state and actuator interfaces."""
        resolved_model_path = Path(model_path).expanduser().resolve()
        super().__init__(
            model_path=str(resolved_model_path),
            frame_skip=200,
            observation_space=None,
            render_mode=render_mode,
        )

        # MujocoEnv needs a full-state space although policies receive partial
        # observations assembled later by the public environment.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float64,
        )

        # Resolve the XML contract once so stepping requires no name lookups.
        self.segment_ids = self._discover_segment_ids()
        (self.leg_actuator_ids, self.spine_actuator_ids) = self._resolve_actuator_ids()
        self._resolve_physical_ids()

        # Internal actions concatenate every segment's six leg controls.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.segment_ids) * len(LEG_ACTUATOR_ROLES),),
            dtype=np.float32,
        )

    def _discover_segment_ids(self) -> tuple[int, ...]:
        """Read segment owners from the XML and require consecutive identifiers."""
        if self.model.nuser_actuator < 1:
            raise ValueError("The model must define actuator ownership metadata")

        owners = np.unique(self.model.actuator_user[:, 0])
        expected = np.arange(len(owners))

        if not np.array_equal(owners, expected):
            raise ValueError("Segment owners must be consecutive IDs starting at 0")

        return tuple(int(owner) for owner in owners)

    def _model_id(self, element: str, name: str) -> int:
        """Resolve one named MuJoCo element and report a readable schema error."""
        try:
            return int(getattr(self.model, element)(name).id)
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
        actuator_id = self._model_id("actuator", actuator_name)
        joint_id = self._model_id("joint", joint_name)

        owner = self.model.actuator_user[actuator_id, 0]
        target_joint = self.model.actuator_trnid[actuator_id, 0]
        control_range = self.model.actuator_ctrlrange[actuator_id]

        if owner != segment_id:
            raise ValueError(
                f"{actuator_name} has owner {owner}, expected {segment_id}"
            )

        if target_joint != joint_id:
            raise ValueError(f"{actuator_name} does not control {joint_name}")

        if not self.model.actuator_ctrllimited[actuator_id]:
            raise ValueError(f"{actuator_name} must have a limited control range")

        if not np.array_equal(control_range, (-1.0, 1.0)):
            raise ValueError(f"{actuator_name} must use control range [-1, 1]")

        return actuator_id

    def _resolve_actuator_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """Build complete cached mappings for enabled leg and disabled spine motors."""
        leg_ids = np.array(
            [
                [
                    self._resolve_actuator_id(
                        actuator_name=(f"segment_{segment_id:02d}_{side}_{role}_motor"),
                        joint_name=(f"segment_{segment_id:02d}_{side}_{role}"),
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

        if len(np.unique(mapped_ids)) != self.model.nu:
            raise ValueError("Every MuJoCo actuator must be mapped exactly once")

        return leg_ids, spine_ids

    def _resolve_physical_ids(self) -> None:
        """Cache joint, body, site, and geometry IDs needed after construction."""
        leg_joint_ids = self.model.actuator_trnid[self.leg_actuator_ids, 0]
        self.leg_qpos_ids = self.model.jnt_qposadr[leg_joint_ids].astype(np.int32)
        self.leg_qvel_ids = self.model.jnt_dofadr[leg_joint_ids].astype(np.int32)

        if (
            len(np.unique(self.leg_qpos_ids)) != self.leg_actuator_ids.size
            or len(np.unique(self.leg_qvel_ids)) != self.leg_actuator_ids.size
        ):
            raise ValueError("Every leg actuator must control one distinct joint")

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

        if self.model.nuser_geom < 2:
            raise ValueError("The model must define geometry ownership and categories")

        self._geom_owners = self.model.geom_user[:, 0].astype(np.int32)
        self._geom_categories = self.model.geom_user[:, 1].astype(np.int32)
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
        non_floor = np.arange(self.model.ngeom) != self.floor_geom_id
        collisions = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)

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

    def _raw_state(self) -> NDArray[np.float64]:
        """Copy MuJoCo positions and velocities into one independent state vector."""
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    def _assemble_control(self, leg_controls: np.ndarray) -> np.ndarray:
        """Map flat leg commands into full controls while leaving spine motors zero."""
        control = np.zeros(self.model.nu, dtype=np.float64)
        control[self.leg_actuator_ids.ravel()] = leg_controls.ravel()
        return control

    def _classify_contacts(
        self,
        geom_pairs: NDArray[np.int32],
    ) -> tuple[
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Convert geometry pairs into per-segment ground and leg contact flags."""
        segment_count = len(self.segment_ids)
        left_foot_ground = np.zeros(segment_count, dtype=np.bool_)
        right_foot_ground = np.zeros(segment_count, dtype=np.bool_)
        body_ground = np.zeros(segment_count, dtype=np.bool_)
        leg_leg = np.zeros(segment_count, dtype=np.bool_)

        for first_geom, second_geom in geom_pairs:
            first_geom = int(first_geom)
            second_geom = int(second_geom)

            if self.floor_geom_id in (first_geom, second_geom):
                other_geom = (
                    second_geom if first_geom == self.floor_geom_id else first_geom
                )
                owner = int(self._geom_owners[other_geom])
                category = int(self._geom_categories[other_geom])

                if category == _BODY_CATEGORY and 0 <= owner < segment_count:
                    body_ground[owner] = True
                elif category == _FOOT_CATEGORY and 0 <= owner < segment_count:
                    if other_geom == self.foot_geom_ids[owner, 0]:
                        left_foot_ground[owner] = True
                    elif other_geom == self.foot_geom_ids[owner, 1]:
                        right_foot_ground[owner] = True
                continue

            first_owner = int(self._geom_owners[first_geom])
            second_owner = int(self._geom_owners[second_geom])
            first_category = int(self._geom_categories[first_geom])
            second_category = int(self._geom_categories[second_geom])
            if (
                0 <= first_owner < segment_count
                and 0 <= second_owner < segment_count
                and first_category in _LEG_CATEGORIES
                and second_category in _LEG_CATEGORIES
            ):
                leg_leg[first_owner] = True
                leg_leg[second_owner] = True

        return left_foot_ground, right_foot_ground, body_ground, leg_leg

    def _contact_flags(
        self,
    ) -> tuple[
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Classify contacts present in the current final MuJoCo state."""
        geom_pairs = np.array(
            [contact.geom for contact in self.data.contact],
            dtype=np.int32,
        ).reshape((-1, 2))
        return self._classify_contacts(geom_pairs)

    def _local_body_velocities(self) -> NDArray[np.float64]:
        """Return angular then linear center velocities in each segment frame."""
        velocities = np.empty((len(self.segment_ids), 6), dtype=np.float64)

        for row, site_id in enumerate(self.segment_center_site_ids):
            mujoco.mj_objectVelocity(  # pyright: ignore[reportAttributeAccessIssue]
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_SITE,  # pyright: ignore[reportAttributeAccessIssue]
                int(site_id),
                velocities[row],
                1,
            )

        return velocities

    def _validate_physics(self) -> None:
        """Reject non-finite physical state or any MuJoCo warning."""
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.ctrl,
            self.data.act,
        )
        if any(not np.isfinite(values).all() for values in arrays):
            raise RuntimeError("MuJoCo produced a non-finite physical state")

        warning_ids = [
            index for index, warning in enumerate(self.data.warning) if warning.number
        ]
        if warning_ids:
            raise RuntimeError(f"MuJoCo reported warnings at indices {warning_ids}")

    def snapshot(self) -> PhysicalSnapshot:
        """Copy the current complete state into the environment-facing format."""
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self._validate_physics()

        center_positions = self.data.site_xpos[self.segment_center_site_ids]
        local_velocities = self._local_body_velocities()
        (
            left_foot_ground,
            right_foot_ground,
            body_ground,
            leg_leg,
        ) = self._contact_flags()

        return PhysicalSnapshot(
            body_height=_readonly_copy(center_positions[:, 2]),
            body_quaternion=_readonly_copy(self.data.xquat[self.segment_body_ids]),
            leg_joint_position=_readonly_copy(self.data.qpos[self.leg_qpos_ids]),
            leg_joint_velocity=_readonly_copy(self.data.qvel[self.leg_qvel_ids]),
            body_linear_velocity=_readonly_copy(local_velocities[:, 3:]),
            body_angular_velocity=_readonly_copy(local_velocities[:, :3]),
            body_planar_position=_readonly_copy(center_positions[:, :2]),
            left_foot_ground_contact=_readonly_copy(left_foot_ground),
            right_foot_ground_contact=_readonly_copy(right_foot_ground),
            body_ground_contact=_readonly_copy(body_ground),
            leg_leg_contact=_readonly_copy(leg_leg),
            head_tip_position=_readonly_copy(
                self.data.site_xpos[self.head_tip_site_id]
            ),
        )

    def reset_model(self) -> np.ndarray:
        """Randomize only leg state around the nominal XML pose."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[self.leg_qpos_ids] += self.np_random.uniform(
            -LEG_POSITION_NOISE_LIMIT,
            LEG_POSITION_NOISE_LIMIT,
            size=self.leg_qpos_ids.shape,
        )
        qvel[self.leg_qvel_ids] += self.np_random.normal(
            0.0,
            LEG_VELOCITY_NOISE_STD,
            size=self.leg_qvel_ids.shape,
        )
        self.set_state(qpos, qvel)
        self._validate_physics()
        return self._raw_state()

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[
        NDArray[np.float64],
        np.float64,
        bool,
        bool,
        dict[str, np.float64],
    ]:
        """Apply one trusted leg action for a complete 20 ms control interval.

        A neutral Gymnasium transition is returned because task rewards and
        episode endings are calculated by the public PettingZoo environment.
        """
        control = self._assemble_control(action)
        self.do_simulation(control, self.frame_skip)
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self._validate_physics()

        observation = self._raw_state()

        if self.render_mode == "human":
            self.render()

        return observation, np.float64(0.0), False, False, {}
