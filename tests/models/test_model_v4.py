"""Validation of model v4: twelve units with four-joint legs.

v4 keeps v3's contacts, spine, and the joints and maximum torques its legs
share with v3's; the ankle's torque is half the knee's. Since 2026-10-15 its
physics differ from v3's in three ways (docs/model.md, "Physics of v4"): leg
armature that gives every leg joint a 2 ms transient, a 0.5 ms timestep (40
steps per action), and contacts and joint limits that respond in twice the
timestep. The checks below repeat the parts of its validation that a change
could break.
"""

import mujoco
import numpy as np
import pytest

# A physics check (pyproject.toml): run with -m physics when a model changes.
pytestmark = pytest.mark.physics

FLOOR, BODY, LEG, LEFT_FOOT, RIGHT_FOOT = range(5)
LIMBS = (LEG, LEFT_FOOT, RIGHT_FOOT)
LEG_ROLES = ("sweep", "lift", "knee", "ankle")
STEPS_PER_ACTION = 40
LEG_TRANSIENT_S = 0.002


@pytest.fixture(scope="module")
def v3():
    return mujoco.MjModel.from_xml_path("models/assembly_v3.xml")


@pytest.fixture(scope="module")
def v4():
    return mujoco.MjModel.from_xml_path("models/assembly_v4.xml")


def joint_properties(model):
    """Each joint role (the name's last word) with the ranges, stiffnesses,
    dampings and, for the spine, armatures its joints have."""
    properties = {}
    for joint_id in range(1, model.njnt):  # 0 is the root
        dof = model.jnt_dofadr[joint_id]
        role = model.joint(joint_id).name.rsplit("_", 1)[1]
        values = (
            *model.jnt_range[joint_id],
            model.jnt_stiffness[joint_id],
            model.dof_damping[dof],
            None if role in LEG_ROLES else model.dof_armature[dof],
        )
        properties.setdefault(role, set()).add(values)
    return properties


def motor_torques(model):
    """Each motor role with the maximum torques its motors have."""
    torques = {}
    for motor_id in range(model.nu):
        role = model.actuator(motor_id).name.removesuffix("_motor").rsplit("_", 1)[1]
        torques.setdefault(role, set()).add(model.actuator_gear[motor_id, 0])
    return torques


def test_v3s_physics_carry_over_but_the_timestep_responses_and_leg_armature(v3, v4):
    for name in ("integrator", "iterations", "tolerance", "cone"):
        assert getattr(v4.opt, name) == getattr(v3.opt, name), name
    assert STEPS_PER_ACTION * v4.opt.timestep == pytest.approx(0.020, abs=1e-12)
    # Contacts and joint limits respond in twice the timestep, the shortest
    # MuJoCo allows, so none is softened further.
    response = (2 * v4.opt.timestep, 1.0)
    assert {tuple(row) for row in v4.geom_solref} == {response}
    assert {tuple(row) for row in v4.jnt_solref[v4.jnt_limited == 1]} == {response}

    # Every leg joint takes LEG_TRANSIENT_S to reach the speed its torque and
    # damping set: its inertia, armature included, over its damping.
    data = mujoco.MjData(v4)
    mujoco.mj_forward(v4, data)
    inertia = np.zeros((v4.nv, v4.nv))
    mujoco.mj_fullM(v4, data, inertia)
    for joint_id in range(1, v4.njnt):
        if v4.joint(joint_id).name.endswith(LEG_ROLES):
            dof = v4.jnt_dofadr[joint_id]
            transient = inertia[dof, dof] / v4.dof_damping[dof]
            assert transient == pytest.approx(LEG_TRANSIENT_S, rel=0.05)

    for field in (
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solimp",
        "geom_condim",
    ):
        for category in range(6):
            v3_values = getattr(v3, field)[v3.geom_user[:, 1] == category]
            v4_values = getattr(v4, field)[v4.geom_user[:, 1] == category]
            assert {tuple(np.atleast_1d(row)) for row in v4_values} == {
                tuple(np.atleast_1d(row)) for row in v3_values
            }, (field, category)

    v3_joints, v4_joints = joint_properties(v3), joint_properties(v4)
    ankle = v4_joints.pop("ankle")
    assert v4_joints == v3_joints
    # The ankle bends like the knee: the same damping.
    assert {values[3:] for values in ankle} == {
        values[3:] for values in v3_joints["knee"]
    }

    v3_torques, v4_torques = motor_torques(v3), motor_torques(v4)
    assert v4_torques.pop("ankle") == {next(iter(v3_torques["knee"])) / 2}
    assert v4_torques == v3_torques


def ground_contacts(model, data):
    """How many feet and bodies touch the floor."""
    categories = model.geom_user[:, 1].astype(int)
    # A body can touch the floor at several points: count each shape once.
    touching = {
        int(b if categories[a] == FLOOR else a)
        for a, b in zip(data.contact.geom1, data.contact.geom2, strict=True)
        if FLOOR in (categories[a], categories[b])
    }
    feet = sum(categories[geom] in (LEFT_FOOT, RIGHT_FOOT) for geom in touching)
    return feet, sum(categories[geom] == BODY for geom in touching)


def held(model, lift_command, seconds):
    """The data after ``seconds`` with every lift motor at ``lift_command``."""
    data = mujoco.MjData(model)
    for motor_id in range(model.nu):
        if model.actuator(motor_id).name.endswith("_lift_motor"):
            data.ctrl[motor_id] = lift_command
    deepest_penetration = 0.0
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.ncon:
            deepest_penetration = max(deepest_penetration, -data.contact.dist.min())
    return data, deepest_penetration


def test_motors_off_the_body_sinks_and_the_lift_motors_hold_it_up(v4):
    """With every motor off the body settles onto its belly, as v1 to v3 do;
    with the lift motors pressing down at 40% it stands on all 24 feet. The
    1 ms contacts sink about 0.05 mm (v3's 0.3 ms ones 0.02 mm)."""
    settled, penetration = held(v4, 0.0, 1.0)
    assert ground_contacts(v4, settled)[1] == 12
    assert penetration < 1e-4

    standing, penetration = held(v4, -0.4, 1.0)
    assert ground_contacts(v4, standing) == (24, 0)
    assert penetration < 1e-4


def test_no_leg_passes_through_a_body_under_random_commands(v4):
    """Every motor flails, with a new command every 20 ms as in training."""
    categories = v4.geom_user[:, 1].astype(int)
    bodies = np.flatnonzero(categories == BODY)
    limbs = np.flatnonzero(np.isin(categories, LIMBS))

    def parent_filtered(a, b):
        body_a, body_b = v4.geom_bodyid[a], v4.geom_bodyid[b]
        return v4.body_parentid[body_a] == body_b or v4.body_parentid[body_b] == body_a

    pairs = [
        (int(limb), int(body))
        for limb in limbs
        for body in bodies
        if not parent_filtered(limb, body)
    ] + [(int(a), int(b)) for i, a in enumerate(bodies) for b in bodies[i + 2 :]]
    data = mujoco.MjData(v4)
    rng = np.random.default_rng(0)
    closest = np.inf

    for step in range(100 * STEPS_PER_ACTION):
        if step >= 10 * STEPS_PER_ACTION and step % STEPS_PER_ACTION == 0:
            data.ctrl[:] = rng.uniform(-1.0, 1.0, v4.nu)
        mujoco.mj_step(v4, data)
        if step % 2 == 0:  # every millisecond
            for a, b in pairs:
                closest = min(
                    closest, mujoco.mj_geomDistance(v4, data, a, b, 1e-3, None)
                )

    assert closest > 0.0
