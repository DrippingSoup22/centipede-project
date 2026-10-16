from dataclasses import dataclass

import mujoco
import numpy as np

from centipede.environment.simulation.model_mapping import (
    BODY_CATEGORY,
    LEFT_FOOT_CATEGORY,
    ModelMapping,
)


@dataclass(frozen=True)
class BodyMeasures:
    """Sizes of the body that tasks and rewards are measured in.

    ``unit_weight_n`` is the whole model's weight shared by its segments.
    ``leg_push_n`` is the strongest push a leg's sweep motor makes at its
    foot: its maximum torque over the foot's flat distance from the joint, in
    the model's pose. ``body_length_m`` runs from the head's tip to the rear
    end of the last segment's body shape, and ``turning_circle_m`` is the
    diameter of the circle the segments' centres lie on with every spine yaw
    joint bent to its limit. ``segment_lengths_m`` are each segment's length
    from its front end to its rear end in the plane: the head's from its tip
    to its spine joint, the others' from joint to joint, the rear segment's
    twice the way from its joint to its centre. ``segment_weights_n`` are
    each segment's own weight, its body and legs without the segments behind.
    """

    unit_weight_n: float
    leg_push_n: float
    body_length_m: float
    turning_circle_m: float
    segment_lengths_m: tuple[float, ...]
    segment_weights_n: tuple[float, ...] = ()


def measure_body(model: mujoco.MjModel, mapping: ModelMapping) -> BodyMeasures:
    """Measure the body in the model's own pose, as the XML defines it."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    segment_count = mapping.segment_count

    gravity = np.linalg.norm(model.opt.gravity)
    weight_n = model.body_mass.sum() * gravity
    # A segment's subtree is itself, its legs and every segment behind it.
    subtree_masses = model.body_subtreemass[mapping.body_ids]
    own_masses = subtree_masses - np.append(subtree_masses[1:], 0.0)

    # The left sweep of the first segment stands for every leg: the model's
    # legs are copies of one another.
    sweep_motor_id = mapping.leg_actuator_ids[0, 0]
    sweep_joint_id = model.actuator_trnid[sweep_motor_id, 0]
    max_torque = (
        model.actuator_gear[sweep_motor_id, 0]
        * np.abs(model.actuator_ctrlrange[sweep_motor_id]).max()
    )
    foot_shape_id = _owned_shape(mapping, 0, LEFT_FOOT_CATEGORY)
    reach = np.linalg.norm(
        (data.geom_xpos[foot_shape_id] - data.xanchor[sweep_joint_id])[:2]
    )

    # Along the body, from the rear segment's centre toward the head's tip.
    head_tip = data.site_xpos[mapping.head_tip_site_id]
    rear_center = data.site_xpos[mapping.center_site_ids[-1]]
    forward = (head_tip - rear_center) / np.linalg.norm(head_tip - rear_center)
    rear_corners = _box_corners(
        model, data, _owned_shape(mapping, segment_count - 1, BODY_CATEGORY)
    )
    body_length = ((head_tip - rear_corners) @ forward).max()

    # Centres a unit apart, each turned by the joint's limit from the last,
    # lie on a circle whose chord of one unit spans that angle.
    centers = data.site_xpos[mapping.center_site_ids]
    unit_spacing = np.linalg.norm(np.diff(centers, axis=0), axis=1).mean()
    spine_joint_ids = model.actuator_trnid[mapping.spine_actuator_ids, 0]
    bend_limit = np.abs(model.jnt_range[spine_joint_ids]).min()
    turning_circle = unit_spacing / np.sin(bend_limit / 2)

    front_ends = np.vstack([head_tip, data.xanchor[spine_joint_ids]])[:, :2]
    rear_length = 2 * np.linalg.norm(front_ends[-1] - centers[-1, :2])
    segment_lengths = [
        *np.linalg.norm(np.diff(front_ends, axis=0), axis=1),
        rear_length,
    ]

    return BodyMeasures(
        unit_weight_n=float(weight_n / segment_count),
        leg_push_n=float(max_torque / reach),
        body_length_m=float(body_length),
        turning_circle_m=float(turning_circle),
        segment_lengths_m=tuple(float(length) for length in segment_lengths),
        segment_weights_n=tuple(float(mass * gravity) for mass in own_masses),
    )


def _owned_shape(mapping: ModelMapping, segment_index: int, category: int) -> int:
    """The id of the one shape of ``category`` that the segment owns."""
    return int(
        np.flatnonzero(
            (mapping.geom_categories == category)
            & (mapping.geom_owner_indices == segment_index)
        )[0]
    )


def _box_corners(
    model: mujoco.MjModel, data: mujoco.MjData, shape_id: int
) -> np.ndarray:
    """The world positions of the eight corners of the box MuJoCo keeps
    around a shape, in the shape's own frame."""
    center, half_size = model.geom_aabb[shape_id, :3], model.geom_aabb[shape_id, 3:]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
    rotation = data.geom_xmat[shape_id].reshape(3, 3)
    return data.geom_xpos[shape_id] + (center + signs * half_size) @ rotation.T
