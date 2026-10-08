from dataclasses import dataclass

import mujoco
import numpy as np

FLOOR_CATEGORY = 0
BODY_CATEGORY = 1
LEG_CATEGORY = 2
LEFT_FOOT_CATEGORY = 3
RIGHT_FOOT_CATEGORY = 4
MEMBRANE_CATEGORY = 5
ALL_CATEGORIES = range(6)

LEG_ACTION_ORDER = (
    ("left", "sweep"),
    ("left", "lift"),
    ("left", "knee"),
    ("right", "sweep"),
    ("right", "lift"),
    ("right", "knee"),
)


class ModelContractError(ValueError):
    """The model does not satisfy the segment contract."""


@dataclass(frozen=True)
class HeadOutline:
    """The head's body shape seen from above, as the rectangle that encloses it.

    Distances in metres from the head's centre, along the head (``rear_m``
    negative, ``front_m`` positive, forward) and to each side (``half_width_m``).
    """

    rear_m: float
    front_m: float
    half_width_m: float


@dataclass(frozen=True)
class ModelMapping:
    """Index tables linking each segment to its parts in a loaded model.

    The parts are its motors, joints, body, centre site, and collision shapes;
    ``head_outline`` is the head's shape seen from above. The spine tables have
    one entry per spine joint, ``(N - 1,)``: entry ``i`` is the yaw joint that
    joins segment ``i`` to segment ``i + 1`` behind it.
    """

    segment_count: int
    leg_actuator_ids: np.ndarray
    leg_qpos_addresses: np.ndarray
    leg_dof_addresses: np.ndarray
    spine_actuator_ids: np.ndarray
    spine_qpos_addresses: np.ndarray
    spine_dof_addresses: np.ndarray
    body_ids: np.ndarray
    center_site_ids: np.ndarray
    head_tip_site_id: int
    root_qpos_address: int
    geom_owner_indices: np.ndarray
    geom_categories: np.ndarray
    head_outline: HeadOutline

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "ModelMapping":
        """Check the segment contract once and build the index tables."""
        segment_count = _count_segments(model)
        leg_actuator_ids, leg_qpos_addresses, leg_dof_addresses = _map_leg_motors(
            model, segment_count
        )
        spine_actuator_ids, spine_qpos_addresses, spine_dof_addresses = (
            _map_spine_motors(model, segment_count)
        )
        _check_all_motors_found(model, leg_actuator_ids, spine_actuator_ids)
        body_ids, center_site_ids, head_tip_site_id = _map_bodies_and_sites(
            model, segment_count
        )
        root_qpos_address = _find_root_joint(model, int(body_ids[0]))
        geom_owner_indices, geom_categories = _read_geom_metadata(model, segment_count)
        is_head_shape = (geom_owner_indices == 0) & (geom_categories == BODY_CATEGORY)
        head_shape_id = int(np.flatnonzero(is_head_shape)[0])
        head_outline = _head_outline(
            model, head_shape_id, int(body_ids[0]), int(center_site_ids[0])
        )
        return cls(
            segment_count=segment_count,
            leg_actuator_ids=leg_actuator_ids,
            leg_qpos_addresses=leg_qpos_addresses,
            leg_dof_addresses=leg_dof_addresses,
            spine_actuator_ids=spine_actuator_ids,
            spine_qpos_addresses=spine_qpos_addresses,
            spine_dof_addresses=spine_dof_addresses,
            body_ids=body_ids,
            center_site_ids=center_site_ids,
            head_tip_site_id=head_tip_site_id,
            root_qpos_address=root_qpos_address,
            geom_owner_indices=geom_owner_indices,
            geom_categories=geom_categories,
            head_outline=head_outline,
        )


def _find_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, kind: str, name: str
) -> int:
    """Helper to convert a name to id in a loaded model."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ModelContractError(f"Model has no {kind} named {name}")

    return object_id


def _count_segments(model: mujoco.MjModel) -> int:
    """Number of segments, read from the owner stored with every motor."""
    if model.nuser_actuator < 1:
        raise ModelContractError("Motors carry no owning segment (actuator user)")

    owner_indices = model.actuator_user[:, 0].astype(int)
    if owner_indices.min() < 0:
        raise ModelContractError("A motor has a negative owning segment")
    segment_count = int(owner_indices.max()) + 1

    missing = set(range(segment_count)) - set(owner_indices.tolist())
    if missing:
        raise ModelContractError(f"Segments without motors: {sorted(missing)}")

    return segment_count


def _check_motor(
    model: mujoco.MjModel, motor_id: int, segment_index: int, joint_name: str
) -> int:
    """Check one motor's owner, joint, and command range; return its joint ID."""
    motor_name = model.actuator(motor_id).name

    owner_index = int(model.actuator_user[motor_id, 0])
    if owner_index != segment_index:
        raise ModelContractError(
            f"{motor_name} is owned by segment {owner_index}, expected {segment_index}"
        )

    transmission_type = model.actuator_trntype[motor_id]
    if transmission_type != mujoco.mjtTrn.mjTRN_JOINT:
        raise ModelContractError(f"{motor_name} does not drive a joint")

    expected_joint_id = _find_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint", joint_name)
    driven_joint_id = int(model.actuator_trnid[motor_id, 0])
    if driven_joint_id != expected_joint_id:
        driven_joint_name = model.joint(driven_joint_id).name
        raise ModelContractError(
            f"{motor_name} drives {driven_joint_name}, expected {joint_name}"
        )

    if not model.actuator_ctrllimited[motor_id]:
        raise ModelContractError(f"{motor_name} has no command limits")

    low, high = model.actuator_ctrlrange[motor_id]
    if low != -1 or high != 1:
        raise ModelContractError(
            f"{motor_name} accepts commands from {low} to {high}, expected -1 to 1"
        )

    return driven_joint_id


def _map_leg_motors(
    model: mujoco.MjModel, segment_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find and check every leg motor; return three (segments, 6) index tables.

    Row ``s``, column ``k`` of each table describes segment ``s``'s ``k``-th
    action in ``LEG_ACTION_ORDER``: the motor's ID (where the command goes in
    ``data.ctrl``), and its joint's addresses in ``data.qpos`` and ``data.qvel``.
    """
    motor_ids = np.zeros((segment_count, 6), dtype=np.int32)
    qpos_addresses = np.zeros((segment_count, 6), dtype=np.int32)
    qvel_addresses = np.zeros((segment_count, 6), dtype=np.int32)

    for segment_index in range(segment_count):
        for action_index, (side, role) in enumerate(LEG_ACTION_ORDER):
            joint_name = f"segment_{segment_index:02d}_{side}_{role}"
            motor_name = f"{joint_name}_motor"

            motor_id = _find_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", motor_name
            )
            joint_id = _check_motor(model, motor_id, segment_index, joint_name)

            motor_ids[segment_index, action_index] = motor_id
            qpos_addresses[segment_index, action_index] = model.jnt_qposadr[joint_id]
            qvel_addresses[segment_index, action_index] = model.jnt_dofadr[joint_id]

    return motor_ids, qpos_addresses, qvel_addresses


def _map_spine_motors(
    model: mujoco.MjModel, segment_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find and check every spine yaw motor; return three (segments - 1,) tables.

    The model stores each spine motor with the segment behind its joint, the
    unit it sits in, so ``segment_01_yaw_motor`` joins segment 0 to segment 1.
    Entry ``i`` describes the joint between segments ``i`` and ``i + 1``: the
    motor's ID and its joint's addresses in ``data.qpos`` and ``data.qvel``.
    Which agent commands it is the environment's choice.
    """
    motor_ids = np.zeros(segment_count - 1, dtype=np.int32)
    qpos_addresses = np.zeros(segment_count - 1, dtype=np.int32)
    qvel_addresses = np.zeros(segment_count - 1, dtype=np.int32)
    for joint_index in range(segment_count - 1):
        joint_name = f"segment_{joint_index + 1:02d}_yaw"
        motor_id = _find_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", f"{joint_name}_motor"
        )
        joint_id = _check_motor(model, motor_id, joint_index + 1, joint_name)
        motor_ids[joint_index] = motor_id
        qpos_addresses[joint_index] = model.jnt_qposadr[joint_id]
        qvel_addresses[joint_index] = model.jnt_dofadr[joint_id]
    return motor_ids, qpos_addresses, qvel_addresses


def _check_all_motors_found(
    model: mujoco.MjModel, leg_motor_ids: np.ndarray, spine_motor_ids: np.ndarray
) -> None:
    """Check that the model has no motors besides the legs' and the spine's."""
    found_motor_ids = set(leg_motor_ids.ravel().tolist()) | set(
        spine_motor_ids.tolist()
    )
    leftover_motor_ids = set(range(model.nu)) - found_motor_ids
    if leftover_motor_ids:
        leftover_names = sorted(model.actuator(i).name for i in leftover_motor_ids)
        raise ModelContractError(
            f"Motors that belong to no leg or spine joint: {', '.join(leftover_names)}"
        )


def _map_bodies_and_sites(
    model: mujoco.MjModel, segment_count: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the segment body IDs, the segment centre site IDs, and the head tip."""
    body_ids = np.zeros(segment_count, dtype=np.int32)
    center_site_ids = np.zeros(segment_count, dtype=np.int32)
    for segment_index in range(segment_count):
        prefix = f"segment_{segment_index:02d}"
        body_ids[segment_index] = _find_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "body", prefix
        )
        center_site_ids[segment_index] = _find_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "site", f"{prefix}_center"
        )
    head_tip_site_id = _find_id(model, mujoco.mjtObj.mjOBJ_SITE, "site", "head_tip")
    return body_ids, center_site_ids, head_tip_site_id


def _find_root_joint(model: mujoco.MjModel, head_body_id: int) -> int:
    """Return the position address of the free joint that carries the body.

    The head's body must be the root of the body tree, with a free joint (its
    position and orientation in the world), so that turning that joint turns
    the whole centipede.
    """
    first_joint_id = model.body_jntadr[head_body_id]
    if (
        model.body_parentid[head_body_id] != 0
        or first_joint_id < 0
        or model.jnt_type[first_joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ModelContractError(
            "segment_00 must be a child of the world with a free joint"
        )
    return int(model.jnt_qposadr[first_joint_id])


def _head_outline(
    model: mujoco.MjModel, shape_id: int, head_body_id: int, center_site_id: int
) -> HeadOutline:
    """The rectangle that encloses the head's body shape, seen from above.

    MuJoCo gives each shape a box around it in the shape's own frame; its
    corners are turned into the head body's frame and measured from the head's
    centre site, whose axes are the body's.
    """
    if model.geom_bodyid[shape_id] != head_body_id:
        raise ModelContractError("The head's body shape must belong to segment_00")
    box_center, box_half_size = (
        model.geom_aabb[shape_id, :3],
        model.geom_aabb[shape_id, 3:],
    )
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[shape_id])
    corners = (box_center + signs * box_half_size) @ rotation.reshape(3, 3).T
    corners += model.geom_pos[shape_id] - model.site_pos[center_site_id]
    return HeadOutline(
        rear_m=float(corners[:, 0].min()),
        front_m=float(corners[:, 0].max()),
        half_width_m=float(np.abs(corners[:, 1]).max()),
    )


def _read_geom_metadata(
    model: mujoco.MjModel, segment_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return every shape's owning segment and category, after checking them.

    Each shape stores ``user="owner category"`` in the XML. The floor's owner
    is -1; every other shape belongs to a segment, and every segment owns
    exactly one body, one left foot, and one right foot.
    """
    if model.nuser_geom < 2:
        raise ModelContractError(
            "Shapes do not carry an owner and a category (geom user)"
        )

    owner_indices = model.geom_user[:, 0].astype(np.int32)
    categories = model.geom_user[:, 1].astype(np.int32)

    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        owner_index, category = owner_indices[geom_id], categories[geom_id]
        if category not in ALL_CATEGORIES:
            raise ModelContractError(f"Shape {name} has unknown category {category}")
        expected_owner_indices = (
            {-1} if category == FLOOR_CATEGORY else set(range(segment_count))
        )
        if owner_index not in expected_owner_indices:
            raise ModelContractError(f"Shape {name} has invalid owner {owner_index}")

    for segment_index in range(segment_count):
        owned = categories[owner_indices == segment_index]
        for category, part in (
            (BODY_CATEGORY, "body"),
            (LEFT_FOOT_CATEGORY, "left foot"),
            (RIGHT_FOOT_CATEGORY, "right foot"),
        ):
            count = int((owned == category).sum())
            if count != 1:
                raise ModelContractError(
                    f"Segment {segment_index} owns {count} shapes of category {part}, "
                    "expected 1"
                )

    return owner_indices, categories
