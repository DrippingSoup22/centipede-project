from dataclasses import dataclass

import mujoco
import numpy as np

FLOOR_CATEGORY = 0
BODY_CATEGORY = 1
LEG_CATEGORY = 2
LEFT_FOOT_CATEGORY = 3
RIGHT_FOOT_CATEGORY = 4
MEMBRANE_CATEGORY = 5
HEAD_CATEGORY = 6
ALL_CATEGORIES = range(7)
HEAD_OWNER = -1  # the owner a head's motors store: no segment

# A leg's joints, from the body outward. Every leg has the first three; a
# model's legs may also have an ankle, all of them or none.
LEG_ROLES = ("sweep", "lift", "knee", "ankle")
LEG_SIDES = ("left", "right")


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
    ``head_outline`` is the head's shape seen from above. ``leg_roles`` are
    the joints every leg has, in ``LEG_ROLES`` order. The leg tables have one
    column per leg motor, ``(N, M)`` with ``M = 2 × len(leg_roles)``, in leg
    action order: the left leg's joints from the body outward, then the
    right leg's. The spine tables have one entry per spine joint, ``(N - 1,)``:
    entry ``i`` is the yaw joint that joins segment ``i`` to segment ``i + 1``
    behind it. ``head_actuator_ids`` are the motors of a head that is a body
    of its own (v6's neck), owned by no segment; none on v1 to v5.
    """

    segment_count: int
    leg_roles: tuple[str, ...]
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
    head_actuator_ids: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "ModelMapping":
        """Check the segment contract once and build the index tables."""
        segment_count = _count_segments(model)
        leg_roles = _leg_roles(model)
        leg_actuator_ids, leg_qpos_addresses, leg_dof_addresses = _map_leg_motors(
            model, segment_count, leg_roles
        )
        spine_actuator_ids, spine_qpos_addresses, spine_dof_addresses = (
            _map_spine_motors(model, segment_count)
        )
        body_ids, center_site_ids, head_tip_site_id = _map_bodies_and_sites(
            model, segment_count
        )
        root_qpos_address = _find_root_joint(model, body_ids)
        geom_owner_indices, geom_categories = _read_geom_metadata(model, segment_count)
        # The head's own shape where the head is a body of its own (v6), else
        # the first segment's body shape, which carries the head (v1 to v5).
        is_head_shape = geom_categories == HEAD_CATEGORY
        if not is_head_shape.any():
            is_head_shape = (geom_owner_indices == 0) & (
                geom_categories == BODY_CATEGORY
            )
        head_shape_id = int(np.flatnonzero(is_head_shape)[0])
        head_outline = _head_outline(
            model, head_shape_id, int(body_ids[0]), int(center_site_ids[0])
        )
        head_actuator_ids = _map_head_motors(model, head_shape_id, int(body_ids[0]))
        _check_all_motors_found(
            model, leg_actuator_ids, spine_actuator_ids, head_actuator_ids
        )
        return cls(
            segment_count=segment_count,
            leg_roles=leg_roles,
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
            head_actuator_ids=head_actuator_ids,
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
    """Number of segments, read from the owner stored with every motor; a
    head's motors are owned by -1, no segment."""
    if model.nuser_actuator < 1:
        raise ModelContractError("Motors carry no owning segment (actuator user)")

    owner_indices = model.actuator_user[:, 0].astype(int)
    if owner_indices.min() < HEAD_OWNER:
        raise ModelContractError("A motor has a negative owning segment")
    owner_indices = owner_indices[owner_indices != HEAD_OWNER]
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


def _leg_roles(model: mujoco.MjModel) -> tuple[str, ...]:
    """The joints every leg has: the first three of ``LEG_ROLES``, and the
    ankle when the head's left leg has one. ``_map_leg_motors`` then requires
    exactly these on every leg; an ankle elsewhere leaves its motor unmapped,
    which ``_check_all_motors_found`` rejects."""
    head_ankle = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "segment_00_left_ankle"
    )
    return LEG_ROLES if head_ankle != -1 else LEG_ROLES[:3]


def _map_leg_motors(
    model: mujoco.MjModel, segment_count: int, leg_roles: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find and check every leg motor; return three (segments, M) index tables.

    Row ``s``, column ``k`` of each table describes segment ``s``'s ``k``-th
    leg action: the left leg's ``leg_roles`` then the right leg's. Each entry
    is the motor's ID (where the command goes in ``data.ctrl``), or its
    joint's address in ``data.qpos`` or ``data.qvel``.
    """
    leg_actions = [(side, role) for side in LEG_SIDES for role in leg_roles]
    motor_ids = np.zeros((segment_count, len(leg_actions)), dtype=np.int32)
    qpos_addresses = np.zeros_like(motor_ids)
    qvel_addresses = np.zeros_like(motor_ids)

    for segment_index in range(segment_count):
        for action_index, (side, role) in enumerate(leg_actions):
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


def _map_head_motors(
    model: mujoco.MjModel, head_shape_id: int, first_body_id: int
) -> np.ndarray:
    """The motors the head owns: each drives a joint of the head's own body,
    the one carrying its shape (v6's neck), and takes commands from -1 to 1.
    A head that is part of the first segment's body (v1 to v5) has none."""
    owners = model.actuator_user[:, 0].astype(int)
    motor_ids = np.flatnonzero(owners == HEAD_OWNER).astype(np.int32)
    head_body_id = int(model.geom_bodyid[head_shape_id])
    for motor_id in motor_ids:
        name = model.actuator(motor_id).name
        drives_head = (
            head_body_id != first_body_id
            and model.actuator_trntype[motor_id] == mujoco.mjtTrn.mjTRN_JOINT
            and model.jnt_bodyid[model.actuator_trnid[motor_id, 0]] == head_body_id
        )
        if not drives_head:
            raise ModelContractError(
                f"{name} is the head's but drives no joint of a head body of its own"
            )
        if not model.actuator_ctrllimited[motor_id] or list(
            model.actuator_ctrlrange[motor_id]
        ) != [-1, 1]:
            raise ModelContractError(f"{name} must accept commands from -1 to 1")
    return motor_ids


def _check_all_motors_found(
    model: mujoco.MjModel,
    leg_motor_ids: np.ndarray,
    spine_motor_ids: np.ndarray,
    head_motor_ids: np.ndarray,
) -> None:
    """Check that the model has no motors besides the legs', the spine's and
    the head's."""
    found_motor_ids = (
        set(leg_motor_ids.ravel().tolist())
        | set(spine_motor_ids.tolist())
        | set(head_motor_ids.tolist())
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


def _find_root_joint(model: mujoco.MjModel, body_ids: np.ndarray) -> int:
    """Return the position address of the free joint that carries the body.

    One segment's body must be the root of the body tree, a child of the world
    with a free joint (its position and orientation in the world), so that
    turning that joint turns the whole centipede: the head's in models v1 to
    v4, the middle unit's in v5.
    """
    roots = [
        int(body_id)
        for body_id in body_ids
        if model.body_parentid[body_id] == 0
        and model.body_jntadr[body_id] >= 0
        and model.jnt_type[model.body_jntadr[body_id]] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(roots) != 1:
        raise ModelContractError(
            "one segment must be a child of the world with a free joint"
        )
    return int(model.jnt_qposadr[model.body_jntadr[roots[0]]])


def _head_outline(
    model: mujoco.MjModel, shape_id: int, first_body_id: int, center_site_id: int
) -> HeadOutline:
    """The rectangle that encloses the head's shape, seen from above.

    MuJoCo gives each shape a box around it in the shape's own frame; its
    corners are turned into the first segment's frame and measured from its
    centre site, whose axes are the body's. The shape belongs to the first
    segment's body (v1 to v5) or to the head's, which v6 attaches to it in
    the same frame: the outline is then the head's with its neck straight.
    """
    body_id = model.geom_bodyid[shape_id]
    in_first_frame = body_id == first_body_id or (
        model.body_parentid[body_id] == first_body_id
        and not model.body_pos[body_id].any()
        and np.array_equal(model.body_quat[body_id], [1.0, 0.0, 0.0, 0.0])
    )
    if not in_first_frame:
        raise ModelContractError(
            "The head's shape must belong to segment_00 or to a body in its frame"
        )
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
    exactly one body, one left foot, and one right foot. A head with a body
    of its own (v6) has at most one shape, owned by the first segment, which
    carries it.
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
        if category == FLOOR_CATEGORY:
            expected_owner_indices = {-1}
        elif category == HEAD_CATEGORY:
            expected_owner_indices = {0}
        else:
            expected_owner_indices = set(range(segment_count))
        if owner_index not in expected_owner_indices:
            raise ModelContractError(f"Shape {name} has invalid owner {owner_index}")
    head_shape_count = int((categories == HEAD_CATEGORY).sum())
    if head_shape_count > 1:
        raise ModelContractError(
            f"The model has {head_shape_count} head shapes, expected at most 1"
        )

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
