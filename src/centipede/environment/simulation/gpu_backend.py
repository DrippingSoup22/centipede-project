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
    spine_joint_force_bodies,
)
from centipede.environment.simulation.physical_state import (
    EFFORT_FIELDS,
    PhysicalState,
)

# Warp otherwise prints a banner when it starts and a line for every kernel it
# loads, which bury the training's own output. The training profile measures
# the start-up time itself.
wp.config.quiet = True

SOLVERS = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}
# MuJoCo Warp's per-world overflow bits that mean data was dropped. The solver's
# iteration-limit bits are only notices, so they are not checked.
CONSTRAINT_OVERFLOWS = mjw.OverflowType.NEFC | mjw.OverflowType.NJMAX_NNZ
CONTACT_OVERFLOWS = mjw.OverflowType.BROADPHASE | mjw.OverflowType.NARROWPHASE
CAPACITY_OVERFLOWS = CONSTRAINT_OVERFLOWS | CONTACT_OVERFLOWS
# What an action starts from, kept before its physics so that a world it makes
# non-finite can go back there (``_restart_worlds``).
ACTION_START_FIELDS = ("qpos", "qvel", "qacc_warmstart", "ctrl", "xfrc_applied", "time")


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
    one action to the motor that drives its joint.
    """
    world_index, segment_index, leg_joint_index = wp.tid()
    motor_id = leg_actuator_ids[segment_index, leg_joint_index]
    ctrl[world_index, motor_id] = leg_actions[
        world_index, segment_index, leg_joint_index
    ]


@wp.kernel
def _apply_spine_actions(
    # In
    spine_actuator_ids: wp.array1d[wp.int32],
    spine_actions: wp.array2d[float],
    # Out
    ctrl: wp.array2d[float],
) -> None:
    """Write each world's spine actions into its row of ``ctrl``.

    Launched with one thread per (world, spine joint); joint ``i`` joins
    segments ``i`` and ``i + 1``.
    """
    world_index, joint_index = wp.tid()
    motor_id = spine_actuator_ids[joint_index]
    ctrl[world_index, motor_id] = spine_actions[world_index, joint_index]


@wp.kernel
def _apply_body_forces(
    # In
    body_ids: wp.array1d[wp.int32],
    body_forces: wp.array2d[wp.vec3],
    # Out
    xfrc_applied: wp.array2d[wp.spatial_vector],
) -> None:
    """Make each segment's force its body's applied force, with no torque.

    One thread per (world, segment). MuJoCo Warp keeps MuJoCo's order: the
    force in the spatial vector's top, the torque in its bottom.
    """
    world_index, segment_index = wp.tid()
    xfrc_applied[world_index, body_ids[segment_index]] = wp.spatial_vector(
        body_forces[world_index, segment_index], wp.vec3(0.0)
    )


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
    leg_motor_count = leg_qpos_addresses.shape[1]
    thread_number = (
        world_index * segment_count + segment_index
    ) * leg_motor_count + leg_joint_index
    random_state = wp.rand_init(reset_seed, thread_number)

    qpos_address = leg_qpos_addresses[segment_index, leg_joint_index]
    dof_address = leg_dof_addresses[segment_index, leg_joint_index]
    qpos[world_index, qpos_address] += wp.randf(random_state, -angle_noise, angle_noise)
    qvel[world_index, dof_address] = wp.randn(random_state) * speed_noise


@wp.kernel
def _turn_reset_heading(
    # In
    world_mask: wp.array1d[wp.bool],
    reset_seeds: wp.array1d[wp.int32],
    reset_counts: wp.array1d[wp.int32],
    root_quaternion_address: int,
    leg_thread_count: int,
    heading_range: float,
    # Out
    qpos: wp.array2d[float],
) -> None:
    """Turn the whole body of each masked world about the vertical.

    Launched with one thread per world, after ``_add_leg_reset_noise``. Each
    turns its world's root orientation, MuJoCo's (w, x, y, z) quaternion in
    ``qpos``, by a uniform angle within ``heading_range``. Its random stream is
    numbered after the leg joints' threads, so the legs draw the same numbers
    with or without the turn.
    """
    world_index = wp.tid()
    if not world_mask[world_index]:
        return

    reset_seed = wp.int32(
        wp.rand_init(reset_seeds[world_index], reset_counts[world_index])
    )
    random_state = wp.rand_init(reset_seed, leg_thread_count + world_index)
    half_angle = 0.5 * wp.randf(random_state, -heading_range, heading_range)
    turn_w = wp.cos(half_angle)
    turn_z = wp.sin(half_angle)

    # The turn (cos, 0, 0, sin) times the root's orientation (w, x, y, z).
    w = qpos[world_index, root_quaternion_address]
    x = qpos[world_index, root_quaternion_address + 1]
    y = qpos[world_index, root_quaternion_address + 2]
    z = qpos[world_index, root_quaternion_address + 3]
    qpos[world_index, root_quaternion_address] = turn_w * w - turn_z * z
    qpos[world_index, root_quaternion_address + 1] = turn_w * x - turn_z * y
    qpos[world_index, root_quaternion_address + 2] = turn_w * y + turn_z * x
    qpos[world_index, root_quaternion_address + 3] = turn_w * z + turn_z * w


@wp.kernel
def _read_segment_state(
    # In: tables
    center_site_ids: wp.array1d[int],
    body_ids: wp.array1d[int],
    leg_qpos_addresses: wp.array2d[int],
    leg_dof_addresses: wp.array2d[int],
    spine_qpos_addresses: wp.array1d[int],
    spine_dof_addresses: wp.array1d[int],
    spine_joint_ids: wp.array1d[int],
    head_tip_site_id: int,
    # In: model
    site_bodyid: wp.array1d[int],
    body_rootid: wp.array1d[int],
    # In: data
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    xquat: wp.array2d[wp.quat],
    xanchor: wp.array2d[wp.vec3],
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
    spine_yaw_position: wp.array2d[float],
    spine_yaw_velocity: wp.array2d[float],
    front_end_position: wp.array2d[wp.vec2],
) -> None:
    """Copy each segment's position, orientation, joints and velocity.

    Launched with one thread per (world, segment); each thread writes only its
    own segment's row. The velocity is what ``mj_objectVelocity`` returns for
    the segment's centre site in the site's own axes: MuJoCo stores each body's
    velocity at a shared reference point (the centre of mass of its whole
    tree), so it is first moved to the site, then turned into site axes.
    The spine joint read is the one behind the segment; the rear segment has
    none, and its entries stay zero. The front end is the head's tip, and
    for every other segment its spine joint with the segment ahead.
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

    for leg_joint_index in range(leg_qpos_addresses.shape[1]):
        qpos_address = leg_qpos_addresses[segment_index, leg_joint_index]
        dof_address = leg_dof_addresses[segment_index, leg_joint_index]
        leg_joint_position[world_index, segment_index, leg_joint_index] = qpos[
            world_index, qpos_address
        ]
        leg_joint_velocity[world_index, segment_index, leg_joint_index] = qvel[
            world_index, dof_address
        ]
    if segment_index < spine_qpos_addresses.shape[0]:
        spine_yaw_position[world_index, segment_index] = qpos[
            world_index, spine_qpos_addresses[segment_index]
        ]
        spine_yaw_velocity[world_index, segment_index] = qvel[
            world_index, spine_dof_addresses[segment_index]
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
        tip = site_xpos[world_index, head_tip_site_id]
        head_tip_position[world_index] = tip
        front_end_position[world_index, 0] = wp.vec2(tip[0], tip[1])
    else:
        anchor = xanchor[world_index, spine_joint_ids[segment_index - 1]]
        front_end_position[world_index, segment_index] = wp.vec2(anchor[0], anchor[1])


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


@wp.kernel
def _clear_effort(
    # In
    world_mask: wp.array1d[wp.bool],
    # Out
    foot_normal_force: wp.array3d[float],
    foot_forward_force: wp.array3d[float],
    belly_normal_force: wp.array2d[float],
    floor_forward_force: wp.array2d[float],
    leg_leg_force: wp.array2d[float],
    leg_leg_force_squared: wp.array2d[float],
    motor_positive_work: wp.array2d[float],
    motor_negative_work: wp.array2d[float],
) -> None:
    """Zero the masked worlds' effort measures; one thread per (world, segment)."""
    world_index, segment_index = wp.tid()
    if not world_mask[world_index]:
        return
    for side in range(2):
        foot_normal_force[world_index, segment_index, side] = 0.0
        foot_forward_force[world_index, segment_index, side] = 0.0
    belly_normal_force[world_index, segment_index] = 0.0
    floor_forward_force[world_index, segment_index] = 0.0
    leg_leg_force[world_index, segment_index] = 0.0
    leg_leg_force_squared[world_index, segment_index] = 0.0
    motor_positive_work[world_index, segment_index] = 0.0
    motor_negative_work[world_index, segment_index] = 0.0


@wp.kernel
def _add_contact_forces(
    # In: data
    nacon: wp.array1d[int],
    contact_geom: wp.array1d[wp.vec2i],
    contact_worldid: wp.array1d[int],
    contact_frame: wp.array1d[wp.mat33],
    contact_efc_address: wp.array2d[int],
    efc_force: wp.array2d[float],
    xmat: wp.array2d[wp.mat33],
    # In: tables
    geom_owner_indices: wp.array1d[int],
    geom_categories: wp.array1d[int],
    body_ids: wp.array1d[int],
    world_mask: wp.array1d[wp.bool],
    scale: float,
    # Out
    foot_normal_force: wp.array3d[float],
    foot_forward_force: wp.array3d[float],
    belly_normal_force: wp.array2d[float],
    floor_forward_force: wp.array2d[float],
    leg_leg_pressed: wp.array2d[float],
) -> None:
    """Add one physics step's contact forces of the masked worlds.

    Launched with one thread per pool slot, like ``_read_contact_flags``. The
    arithmetic of the CPU backend's ``_add_effort``: a contact's first three
    constraint rows are its normal and two tangential forces in its frame,
    acting on its second shape as they are and on the first reversed. Every
    shape on the floor adds ``scale`` times the force's part along its
    segment's forward axis laid flat on the floor to ``floor_forward_force``
    (along the tilted axis, a pitched segment would read part of its support
    as a push); a foot or body shape also its normal force, and a foot the
    part along the tilted axis, as the segment feels it; two legs touching
    add their normal force, unscaled, to both owners' ``leg_leg_pressed``,
    which ``_add_segment_effort`` then takes in. Several contacts may touch
    one shape, so the forces are added atomically.
    """
    contact_index = wp.tid()
    if contact_index >= nacon[0]:
        return
    world_index = contact_worldid[contact_index]
    if contact_efc_address[contact_index, 0] < 0 or not world_mask[world_index]:
        return
    local_force = wp.vec3(
        efc_force[world_index, contact_efc_address[contact_index, 0]],
        efc_force[world_index, contact_efc_address[contact_index, 1]],
        efc_force[world_index, contact_efc_address[contact_index, 2]],
    )
    normal_force = local_force[0]
    pair = contact_geom[contact_index]
    first_on_floor = geom_categories[pair[0]] == FLOOR_CATEGORY
    if not first_on_floor and geom_categories[pair[1]] != FLOOR_CATEGORY:
        for end in range(2):
            wp.atomic_add(
                leg_leg_pressed,
                world_index,
                geom_owner_indices[pair[end]],
                normal_force,
            )
        return

    # The shape on the floor, and the floor's force on it.
    force = wp.transpose(contact_frame[contact_index]) @ local_force
    shape_id = pair[1]
    if not first_on_floor:
        shape_id = pair[0]
        force = -force
    category = geom_categories[shape_id]
    owner_index = geom_owner_indices[shape_id]
    rotation = xmat[world_index, body_ids[owner_index]]
    forward = wp.vec3(rotation[0, 0], rotation[1, 0], rotation[2, 0])
    flat = wp.normalize(wp.vec3(rotation[0, 0], rotation[1, 0], 0.0))
    wp.atomic_add(
        floor_forward_force, world_index, owner_index, scale * wp.dot(force, flat)
    )
    if category == BODY_CATEGORY:
        wp.atomic_add(
            belly_normal_force, world_index, owner_index, scale * normal_force
        )
    elif category == LEFT_FOOT_CATEGORY or category == RIGHT_FOOT_CATEGORY:
        side = category - LEFT_FOOT_CATEGORY
        wp.atomic_add(
            foot_normal_force, world_index, owner_index, side, scale * normal_force
        )
        wp.atomic_add(
            foot_forward_force,
            world_index,
            owner_index,
            side,
            scale * wp.dot(force, forward),
        )


@wp.func
def _add_motor_work(
    actuator_force: wp.array2d[float],
    actuator_velocity: wp.array2d[float],
    motor_id: wp.int32,
    world_index: int,
    segment_index: int,
    timestep: float,
    motor_positive_work: wp.array2d[float],
    motor_negative_work: wp.array2d[float],
):
    """Add one motor's work over ``timestep`` to its segment's pushing work
    where positive, to its braking work where negative."""
    work = (
        timestep
        * actuator_force[world_index, motor_id]
        * actuator_velocity[world_index, motor_id]
    )
    if work > 0.0:
        motor_positive_work[world_index, segment_index] += work
    else:
        motor_negative_work[world_index, segment_index] -= work


@wp.kernel
def _add_segment_effort(
    # In: data
    actuator_force: wp.array2d[float],
    actuator_velocity: wp.array2d[float],
    # In: tables
    leg_actuator_ids: wp.array2d[wp.int32],
    spine_actuator_ids: wp.array1d[wp.int32],
    world_mask: wp.array1d[wp.bool],
    scale: float,
    timestep: float,
    # In and out
    leg_leg_pressed: wp.array2d[float],
    leg_leg_force: wp.array2d[float],
    leg_leg_force_squared: wp.array2d[float],
    motor_positive_work: wp.array2d[float],
    motor_negative_work: wp.array2d[float],
) -> None:
    """Add one physics step's legs pressed and motor work of the masked worlds.

    One thread per (world, segment), after ``_add_contact_forces``: the
    segment's pressed total adds ``scale`` times itself and its square, then
    starts again from zero; each of its leg motors and the spine motor behind
    it adds torque times speed times ``timestep`` to its pushing work where
    positive, to its braking work where negative.
    """
    world_index, segment_index = wp.tid()
    if not world_mask[world_index]:
        return
    pressed = leg_leg_pressed[world_index, segment_index]
    leg_leg_force[world_index, segment_index] += scale * pressed
    leg_leg_force_squared[world_index, segment_index] += scale * pressed * pressed
    leg_leg_pressed[world_index, segment_index] = 0.0

    for leg_motor_index in range(leg_actuator_ids.shape[1]):
        _add_motor_work(
            actuator_force,
            actuator_velocity,
            leg_actuator_ids[segment_index, leg_motor_index],
            world_index,
            segment_index,
            timestep,
            motor_positive_work,
            motor_negative_work,
        )
    if segment_index < spine_actuator_ids.shape[0]:
        _add_motor_work(
            actuator_force,
            actuator_velocity,
            spine_actuator_ids[segment_index],
            world_index,
            segment_index,
            timestep,
            motor_positive_work,
            motor_negative_work,
        )


@wp.kernel
def _add_joint_forces(
    # In: data
    cfrc_int: wp.array2d[wp.spatial_vector],
    xmat: wp.array2d[wp.mat33],
    # In: tables
    body_ids: wp.array1d[wp.int32],
    joint_force_bodies: wp.array1d[wp.int32],
    joint_force_signs: wp.array1d[float],
    world_mask: wp.array1d[wp.bool],
    scale: float,
    start: bool,
    # In and out
    front_joint_force: wp.array2d[float],
    net_joint_force: wp.array2d[wp.vec2],
) -> None:
    """Add ``scale`` times each segment's front joint force of the masked
    worlds, along its forward axis, and its net joint force, along it and up;
    ``start`` replaces the values instead.

    One thread per (world, segment). ``cfrc_int`` holds, for every body, the
    force its parent exerts on the subtree it heads, its linear part last, in
    world axes (MuJoCo Warp fills it after the solver with
    ``run_rne_postconstraint``). Each joint's force, the front side's on the
    rear side, is its child's ``cfrc_int`` with the sign the tree's
    direction there gives (``spine_joint_force_bodies``). The head has no
    joint ahead, and its front value stays zero. The net force on a segment
    alone is what its front joint gives it less what its rear joint passes
    on to the segments behind.
    """
    world_index, segment_index = wp.tid()
    if not world_mask[world_index]:
        return
    body_id = body_ids[segment_index]
    rotation = xmat[world_index, body_id]
    forward = wp.vec3(rotation[0, 0], rotation[1, 0], rotation[2, 0])
    front = wp.vec3(0.0)
    if segment_index > 0:
        front = joint_force_signs[segment_index - 1] * wp.spatial_bottom(
            cfrc_int[world_index, joint_force_bodies[segment_index - 1]]
        )
    net = front
    if segment_index < body_ids.shape[0] - 1:
        net = front - joint_force_signs[segment_index] * wp.spatial_bottom(
            cfrc_int[world_index, joint_force_bodies[segment_index]]
        )
    along = scale * wp.dot(front, forward)
    net_along_up = scale * wp.vec2(wp.dot(net, forward), net[2])
    if start:
        front_joint_force[world_index, segment_index] = along
        net_joint_force[world_index, segment_index] = net_along_up
    else:
        front_joint_force[world_index, segment_index] += along
        net_joint_force[world_index, segment_index] += net_along_up


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
        start_heading_range_rad: float = 0.0,
        measure_effort: bool = False,
        measure_joint_forces: bool = False,
        max_restarted_worlds_per_action: int = 0,
    ) -> None:
        """Copy the model to the GPU and allocate everything the worlds need.

        ``physics_steps_per_action`` is how many of the model's timesteps make
        one 20 ms action, and ``start_heading_range_rad`` how far a reset may
        turn the body about the vertical. ``solver`` is "newton" or "cg"; it is
        set on the host model before the copy, because MuJoCo Warp reads it
        while building the GPU model. The contact and constraint capacities are
        reserved here once: contacts in one pool shared by all worlds,
        constraints separately for each world. With ``measure_effort`` the
        physical state's effort fields are measured after every physics step
        (``_measure_effort``), and with ``measure_joint_forces`` its joint
        forces (``_measure_joint_forces``), for which MuJoCo Warp computes the
        forces between bodies after every solve.
        Up to ``max_restarted_worlds_per_action`` worlds made non-finite in one
        action are put back to its start instead of stopping the run
        (``_check_worlds``).
        Each physical-state tensor is also kept as a Warp view sharing its
        memory, so kernels writing a view fill the tensor directly.
        """
        model.opt.solver = SOLVERS[solver]
        self.gpu_model = mjw.put_model(model)
        self.gpu_model.opt.run_rne_postconstraint = measure_joint_forces
        self.gpu_data = mjw.make_data(
            model,
            nworld=world_count,
            nconmax=contacts_per_world,
            njmax=constraints_per_world,
        )
        self.world_count = world_count
        self.segment_count = mapping.segment_count
        self.leg_motor_count = mapping.leg_actuator_ids.shape[1]
        self.physics_steps_per_action = physics_steps_per_action
        self.start_heading_range_rad = start_heading_range_rad
        self.measure_effort = measure_effort
        self.measure_joint_forces = measure_joint_forces
        self.max_restarted_worlds_per_action = max_restarted_worlds_per_action
        self.timestep = float(model.opt.timestep)
        self.root_quaternion_address = mapping.root_qpos_address + 3

        # The Warp device that holds the model and data, and its PyTorch name.
        self.device = wp.get_device()
        torch_device = wp.device_to_torch(self.device)

        def to_gpu(table: np.ndarray) -> wp.array:
            return wp.array(table, dtype=wp.int32, device=self.device)

        self.leg_actuator_ids = to_gpu(mapping.leg_actuator_ids)
        self.leg_qpos_addresses = to_gpu(mapping.leg_qpos_addresses)
        self.leg_dof_addresses = to_gpu(mapping.leg_dof_addresses)
        self.spine_actuator_ids = to_gpu(mapping.spine_actuator_ids)
        self.spine_qpos_addresses = to_gpu(mapping.spine_qpos_addresses)
        self.spine_dof_addresses = to_gpu(mapping.spine_dof_addresses)
        # The spine's yaw joints, from the head back: the joints its motors drive.
        self.spine_joint_ids = to_gpu(
            model.actuator_trnid[mapping.spine_actuator_ids, 0]
        )
        self.body_ids = to_gpu(mapping.body_ids)
        joint_force_bodies, joint_force_signs = spine_joint_force_bodies(
            model, mapping.body_ids
        )
        self.joint_force_bodies = to_gpu(joint_force_bodies)
        self.joint_force_signs = wp.array(
            joint_force_signs, dtype=float, device=self.device
        )
        self.center_site_ids = to_gpu(mapping.center_site_ids)
        self.geom_owner_indices = to_gpu(mapping.geom_owner_indices)
        self.geom_categories = to_gpu(mapping.geom_categories)
        self.head_tip_site_id = mapping.head_tip_site_id

        self.physical_state = PhysicalState.allocate(
            world_count, mapping.segment_count, self.leg_motor_count, torch_device
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
        self._spine_yaw_position = wp.from_torch(state.spine_yaw_position)
        self._spine_yaw_velocity = wp.from_torch(state.spine_yaw_velocity)
        self._left_foot_ground_contact = wp.from_torch(state.left_foot_ground_contact)
        self._right_foot_ground_contact = wp.from_torch(state.right_foot_ground_contact)
        self._body_ground_contact = wp.from_torch(state.body_ground_contact)
        self._leg_leg_contact = wp.from_torch(state.leg_leg_contact)
        self._effort = [wp.from_torch(getattr(state, name)) for name in EFFORT_FIELDS]
        self._front_joint_force = wp.from_torch(state.front_joint_force)
        self._net_joint_force = wp.from_torch(state.net_joint_force, dtype=wp.vec2)
        self._front_end_position = wp.from_torch(
            state.front_end_position, dtype=wp.vec2
        )
        # Each segment's legs pressed by others during one physics step.
        self._leg_leg_pressed = wp.zeros(
            (world_count, self.segment_count), dtype=float, device=self.device
        )
        self._every_world = wp.from_torch(
            torch.ones(world_count, dtype=torch.bool, device=torch_device)
        )

        # The simulation's category, as views of MuJoCo Warp's own arrays: the
        # physics refreshes them, so nothing is copied per step.
        self.diagnostics = SimulationDiagnostics(
            SimulationFacts(
                qpos=wp.to_torch(self.gpu_data.qpos),
                contact_count=wp.to_torch(self.gpu_data.nacon),
                constraint_rows=wp.to_torch(self.gpu_data.nefc),
                solver_iterations=wp.to_torch(self.gpu_data.solver_niter),
                restarted_worlds=torch.zeros(
                    world_count, dtype=torch.bool, device=torch_device
                ),
            )
        )
        self._restarted_worlds = self.diagnostics.facts.restarted_worlds

        # The physics steps of one action, recorded as a CUDA graph by the
        # first step; see ``step``.
        self._physics_graph: wp.Graph | None = None
        # Every world's state at the start of the current action, for a
        # failed world's restart.
        self._action_start = {
            name: wp.clone(getattr(self.gpu_data, name)) for name in ACTION_START_FIELDS
        }

        # Each world's random sequence: the seed it was last given and how many
        # resets it has had since. Together they replace the CPU generators.
        self.reset_seeds = torch.zeros(
            world_count, dtype=torch.int32, device=torch_device
        )
        self.reset_counts = torch.zeros(
            world_count, dtype=torch.int32, device=torch_device
        )
        # The spine commands of a step that gives none.
        self.passive_spine_actions = torch.zeros(
            (world_count, self.segment_count - 1), device=torch_device
        )

    def step(
        self,
        leg_actions: torch.Tensor,
        spine_actions: torch.Tensor | None = None,
        body_forces: torch.Tensor | None = None,
    ) -> None:
        """Hold the (W, N, M) leg and (W, N - 1) spine actions for 20 ms.

        Without spine actions, the spine motors receive zero. ``body_forces``,
        (W, N, 3) newtons in world axes, become each segment body's applied
        force, its torque zero, before the physics runs.

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
        wp.launch(
            kernel=_apply_leg_actions,
            dim=(self.world_count, self.segment_count, self.leg_motor_count),
            inputs=[self.leg_actuator_ids, wp.from_torch(leg_actions.contiguous())],
            outputs=[self.gpu_data.ctrl],
            device=self.device,
        )
        if spine_actions is None:
            spine_actions = self.passive_spine_actions
        wp.launch(
            kernel=_apply_spine_actions,
            dim=(self.world_count, self.segment_count - 1),
            inputs=[
                self.spine_actuator_ids,
                wp.from_torch(spine_actions.contiguous()),
            ],
            outputs=[self.gpu_data.ctrl],
            device=self.device,
        )
        if body_forces is not None:
            wp.launch(
                kernel=_apply_body_forces,
                dim=(self.world_count, self.segment_count),
                inputs=[
                    self.body_ids,
                    wp.from_torch(body_forces.contiguous(), dtype=wp.vec3),
                ],
                outputs=[self.gpu_data.xfrc_applied],
                device=self.device,
            )
        for name, kept in self._action_start.items():
            wp.copy(kept, getattr(self.gpu_data, name))
        if self._physics_graph is None:
            self._run_physics_steps()
            # Recording only writes the launches down; nothing runs twice.
            with wp.ScopedCapture(device=self.device) as capture:
                self._run_physics_steps()
            self._physics_graph = capture.graph
        else:
            wp.capture_launch(self._physics_graph)
        restarted = self._check_worlds()
        mjw.forward(self.gpu_model, self.gpu_data)
        self._read_state()
        if restarted:
            self._measure_start(wp.from_torch(self._restarted_worlds))

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
        With a start heading range, the whole body is then turned about the
        vertical by a uniform angle within it.
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
            dim=(self.world_count, self.segment_count, self.leg_motor_count),
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
        if self.start_heading_range_rad > 0:
            wp.launch(
                _turn_reset_heading,
                dim=self.world_count,
                inputs=[
                    mask_view,
                    self.reset_seeds,
                    self.reset_counts,
                    self.root_quaternion_address,
                    self.world_count * self.segment_count * self.leg_motor_count,
                    self.start_heading_range_rad,
                ],
                outputs=[self.gpu_data.qpos],
                device=self.device,
            )
        self.reset_counts += mask
        mjw.forward(self.gpu_model, self.gpu_data)
        self._read_state()
        self._measure_start(mask_view)

    def _measure_start(self, world_mask: wp.array) -> None:
        """The masked worlds' effort at the pose they start from, after
        ``forward``: its forces; no motor has worked yet."""
        if self.measure_effort:
            self._clear_effort(world_mask)
            self._measure_effort(world_mask, 1.0)
        if self.measure_joint_forces:
            self._measure_joint_forces(world_mask, 1.0, start=True)

    def _run_physics_steps(self) -> None:
        """The physics steps of one action, launched from Python, each
        followed by its share of the effort when it is measured."""
        steps = self.physics_steps_per_action
        if self.measure_effort:
            self._clear_effort(self._every_world)
        for step_index in range(steps):
            mjw.step(self.gpu_model, self.gpu_data)
            if self.measure_effort:
                self._measure_effort(self._every_world, 1 / steps)
            if self.measure_joint_forces:
                self._measure_joint_forces(
                    self._every_world, 1 / steps, start=step_index == 0
                )

    def _clear_effort(self, world_mask: wp.array) -> None:
        """Zero the masked worlds' effort measures."""
        wp.launch(
            _clear_effort,
            dim=(self.world_count, self.segment_count),
            inputs=[world_mask],
            outputs=self._effort,
            device=self.device,
        )

    def _measure_effort(self, world_mask: wp.array, scale: float) -> None:
        """Add the masked worlds' effort of the physics state just computed:
        ``scale`` times its forces, and its motors' work over a timestep."""
        (
            foot_normal,
            foot_forward,
            belly,
            floor_forward,
            leg_leg,
            leg_leg_squared,
            *work,
        ) = self._effort
        wp.launch(
            _add_contact_forces,
            dim=self.gpu_data.naconmax,
            inputs=[
                self.gpu_data.nacon,
                self.gpu_data.contact.geom,
                self.gpu_data.contact.worldid,
                self.gpu_data.contact.frame,
                self.gpu_data.contact.efc_address,
                self.gpu_data.efc.force,
                self.gpu_data.xmat,
                self.geom_owner_indices,
                self.geom_categories,
                self.body_ids,
                world_mask,
                scale,
            ],
            outputs=[
                foot_normal,
                foot_forward,
                belly,
                floor_forward,
                self._leg_leg_pressed,
            ],
            device=self.device,
        )
        wp.launch(
            _add_segment_effort,
            dim=(self.world_count, self.segment_count),
            inputs=[
                self.gpu_data.actuator_force,
                self.gpu_data.actuator_velocity,
                self.leg_actuator_ids,
                self.spine_actuator_ids,
                world_mask,
                scale,
                self.timestep,
            ],
            outputs=[self._leg_leg_pressed, leg_leg, leg_leg_squared, *work],
            device=self.device,
        )

    def _measure_joint_forces(
        self, world_mask: wp.array, scale: float, start: bool
    ) -> None:
        """Add ``scale`` times the masked worlds' front joint forces of the
        physics state just computed, or, with ``start``, replace them."""
        wp.launch(
            _add_joint_forces,
            dim=(self.world_count, self.segment_count),
            inputs=[
                self.gpu_data.cfrc_int,
                self.gpu_data.xmat,
                self.body_ids,
                self.joint_force_bodies,
                self.joint_force_signs,
                world_mask,
                scale,
                start,
            ],
            outputs=[self._front_joint_force, self._net_joint_force],
            device=self.device,
        )

    def _check_worlds(self) -> bool:
        """Stop the run if any world's physics went wrong during the last step,
        or restart the worlds made non-finite if they are few; whether it did.

        MuJoCo Warp records, per world, every capacity overflow since the last
        reset, but it does not detect non-finite values, so both are checked
        here. An overflow always stops the run; up to
        ``max_restarted_worlds_per_action`` non-finite worlds go back to the
        action's start (``_restart_worlds``), and more stop it. The ``if`` is
        the step's only wait for the GPU; the failing world is looked up only
        after something went wrong.
        """
        self._restarted_worlds.zero_()
        overflow_bits = wp.to_torch(self.gpu_data.overflow)
        overflowed_worlds = (overflow_bits & CAPACITY_OVERFLOWS) != 0
        finite_qpos = torch.isfinite(wp.to_torch(self.gpu_data.qpos)).all(dim=1)
        finite_qvel = torch.isfinite(wp.to_torch(self.gpu_data.qvel)).all(dim=1)
        non_finite_worlds = ~(finite_qpos & finite_qvel)
        if not (overflowed_worlds | non_finite_worlds).any():
            return False

        if overflowed_worlds.any():
            world_index = int(torch.nonzero(overflowed_worlds)[0].item())
            if int(overflow_bits[world_index].item()) & CONTACT_OVERFLOWS:
                raise RuntimeError(
                    "Physics failed: the worlds together needed more than the "
                    f"{self.gpu_data.naconmax} contacts reserved for them, so "
                    "some contacts were dropped. Raise contacts_per_world in "
                    "[environment.simulation] (now "
                    f"{self.gpu_data.naconmax // self.world_count})."
                )
            raise RuntimeError(
                f"Physics failed in world {world_index}: it needed more than "
                f"the {self.gpu_data.njmax} constraints reserved per world, so "
                "some were dropped. Raise constraints_per_world in "
                "[environment.simulation]."
            )
        if int(non_finite_worlds.sum().item()) <= self.max_restarted_worlds_per_action:
            self._restart_worlds(non_finite_worlds)
            return True
        world_index = int(torch.nonzero(non_finite_worlds)[0].item())
        raise RuntimeError(
            f"Physics failed in world {world_index}: MuJoCo Warp produced a "
            "non-finite position or velocity (the simulation became unstable)."
        )

    def _restart_worlds(self, world_mask: torch.Tensor) -> None:
        """Put the masked worlds back to their state at the action's start and
        flag them, so that the environment ends their episodes there; the
        step's ``forward`` then gives them finite positions and forces."""
        for name, kept in self._action_start.items():
            values = wp.to_torch(getattr(self.gpu_data, name))
            values[world_mask] = wp.to_torch(kept)[world_mask]
        self._restarted_worlds.copy_(world_mask)

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
                self.spine_qpos_addresses,
                self.spine_dof_addresses,
                self.spine_joint_ids,
                self.head_tip_site_id,
                self.gpu_model.site_bodyid,
                self.gpu_model.body_rootid,
                self.gpu_data.site_xpos,
                self.gpu_data.site_xmat,
                self.gpu_data.xquat,
                self.gpu_data.xanchor,
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
                self._spine_yaw_position,
                self._spine_yaw_velocity,
                self._front_end_position,
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
