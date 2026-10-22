"""Validation of model v6 headless: v5's headless tester with a more mobile
trunk and target actuators.

Every link between units gains a passive roll hinge beside its yaw and
pitch hinges, and the pitch hinge's range widens from 8 to 10 degrees either
way with a softer spring. Each motor keeps v5's joint, strength (gear) and
command range, but a command u is a target angle in the joint's range,
m + c u, and the motor pulls toward it as a spring and a damper, at full
strength a quarter of the range away and critically damped at the starting
pose (archive/model_v6_spine/make_model_v6.py in the study's folder).
"""

import mujoco
import numpy as np
import pytest

# A physics check (pyproject.toml): run with -m physics when a model changes.
pytestmark = pytest.mark.physics

FLOOR, BODY, LEFT_FOOT, RIGHT_FOOT = 0, 1, 3, 4
UNITS = 10
# The trunk's hinges: each range either way, in degrees, and each spring,
# in N m/rad, a factor of the first v6's (v5's) pitch spring.
FIRST_PITCH_SPRING = 0.001114
PITCH_RANGE, PITCH_SPRING = 10.0, 0.7 * FIRST_PITCH_SPRING
ROLL_RANGE, ROLL_SPRING = 5.0, 4.5 * FIRST_PITCH_SPRING


@pytest.fixture(scope="module")
def v5():
    return mujoco.MjModel.from_xml_path("models/assembly_v5_headless.xml")


@pytest.fixture(scope="module")
def v6():
    return mujoco.MjModel.from_xml_path("models/assembly_v6_headless.xml")


def target_command(model, motor_id, angle):
    """The command that aims a motor's joint at ``angle``."""
    low, high = model.jnt_range[model.actuator_trnid[motor_id, 0]]
    return (angle - (low + high) / 2) / ((high - low) / 2)


def test_v6_is_v5s_body_with_a_roll_hinge_on_every_link(v5, v6):
    """Bodies, masses and every v5 joint are kept, the pitch hinges widened
    and softened; each link adds a roll hinge, damped like its pitch hinge.
    A positive roll turns the unit behind about the forward axis on both
    sides of the root, as every other hinge keeps its sign."""
    assert v6.opt.timestep == v5.opt.timestep
    assert np.array_equal(v6.body_mass, v5.body_mass)
    assert v6.nv == v5.nv + UNITS - 1
    for old in range(v5.njnt):
        name = v5.joint(old).name
        new = v6.joint(name).id
        old_dof, new_dof = v5.jnt_dofadr[old], v6.jnt_dofadr[new]
        assert v6.dof_damping[new_dof] == v5.dof_damping[old_dof], name
        assert v6.dof_armature[new_dof] == v5.dof_armature[old_dof], name
        if name.endswith("_pitch"):
            assert np.degrees(v6.jnt_range[new]) == pytest.approx(
                [-PITCH_RANGE, PITCH_RANGE]
            )
            assert v6.jnt_stiffness[new] == pytest.approx(PITCH_SPRING)
        else:
            assert np.array_equal(v6.jnt_range[new], v5.jnt_range[old]), name
            assert v6.jnt_stiffness[new] == v5.jnt_stiffness[old], name

    rolls = [j for j in range(v6.njnt) if v6.joint(j).name.endswith("_roll")]
    assert len(rolls) == UNITS - 1
    for roll in rolls:
        pitch = v6.joint(v6.joint(roll).name.replace("_roll", "_pitch")).id
        roll_dof, pitch_dof = v6.jnt_dofadr[roll], v6.jnt_dofadr[pitch]
        assert np.degrees(v6.jnt_range[roll]) == pytest.approx(
            [-ROLL_RANGE, ROLL_RANGE]
        )
        assert v6.jnt_stiffness[roll] == pytest.approx(ROLL_SPRING)
        assert v6.dof_damping[roll_dof] == v6.dof_damping[pitch_dof]

    # Every roll hinge at +2 degrees: each unit is turned 2 degrees more
    # about its forward axis than the unit ahead of it.
    data = mujoco.MjData(v6)
    data.qpos[v6.jnt_qposadr[rolls]] = np.radians(2.0)
    mujoco.mj_kinematics(v6, data)
    units = [v6.body(f"segment_{k:02d}").id for k in range(UNITS)]
    roll_deg = [
        np.degrees(np.arctan2(m[2, 1], m[2, 2]))
        for m in data.xmat[units].reshape(-1, 3, 3)
    ]
    assert np.diff(roll_deg) == pytest.approx(2.0, abs=1e-6)


def test_motors_pull_toward_their_targets(v5, v6):
    """Each motor keeps v5's joint, strength and ranges; the force, in shares
    of the strength, is kp (target - q) - kv dq/dt within -1 and 1, kp
    reaching 1 a quarter of the range away and kv damping the joint
    critically."""
    for field in ("actuator_gear", "actuator_ctrlrange", "actuator_forcerange"):
        assert np.array_equal(getattr(v6, field), getattr(v5, field)), field
    assert [v6.actuator(a).name for a in range(v6.nu)] == [
        v5.actuator(a).name for a in range(v5.nu)
    ]
    assert [v6.joint(v6.actuator_trnid[a, 0]).name for a in range(v6.nu)] == [
        v5.joint(v5.actuator_trnid[a, 0]).name for a in range(v5.nu)
    ]

    data = mujoco.MjData(v6)
    mujoco.mj_forward(v6, data)
    inertia = np.zeros((v6.nv, v6.nv))
    mujoco.mj_fullM(v6, data, inertia)
    rng = np.random.default_rng(0)
    data.qpos[7:] += rng.normal(0.0, 0.2, v6.nq - 7)
    data.qvel[6:] = rng.normal(0.0, 5.0, v6.nv - 6)
    data.ctrl[:] = rng.uniform(-1.0, 1.0, v6.nu)
    mujoco.mj_forward(v6, data)
    for motor_id in range(v6.nu):
        joint_id = v6.actuator_trnid[motor_id, 0]
        dof, gear = v6.jnt_dofadr[joint_id], v6.actuator_gear[motor_id, 0]
        low, high = v6.jnt_range[joint_id]
        kp = -v6.actuator_biasprm[motor_id, 1] * gear
        kv = -v6.actuator_biasprm[motor_id, 2] * gear
        assert kp * (high - low) / 4 == pytest.approx(1.0, rel=1e-5)
        assert gear * kv == pytest.approx(
            2 * np.sqrt(gear * kp * inertia[dof, dof]), rel=1e-5
        )
        target = (low + high) / 2 + (high - low) / 2 * data.ctrl[motor_id]
        q = data.qpos[v6.jnt_qposadr[joint_id]]
        force = kp * (target - q) - kv * data.qvel[dof]
        assert data.actuator_force[motor_id] == pytest.approx(
            np.clip(force, -1.0, 1.0), rel=1e-4, abs=1e-6
        )


def test_lifts_aimed_down_hold_the_body_up_on_its_motors(v6):
    """Every lift aimed 10 degrees down and every other joint straight, the
    body stands on its 20 feet within 3 degrees of the aim, its lifts pulling
    at a fifth of their strength or less, off their stops."""
    data = mujoco.MjData(v6)
    lifts = [a for a in range(v6.nu) if v6.actuator(a).name.endswith("_lift_motor")]
    for motor_id in range(v6.nu):
        angle = np.radians(-10.0) if motor_id in lifts else 0.0
        data.ctrl[motor_id] = target_command(v6, motor_id, angle)
    for _ in range(round(1.0 / v6.opt.timestep)):
        mujoco.mj_step(v6, data)

    categories = v6.geom_user[:, 1].astype(int)
    touching = {
        int(b if categories[a] == FLOOR else a)
        for a, b in zip(data.contact.geom1, data.contact.geom2, strict=True)
        if FLOOR in (categories[a], categories[b])
    }
    assert sum(categories[g] in (LEFT_FOOT, RIGHT_FOOT) for g in touching) == 2 * UNITS
    assert not any(categories[g] == BODY for g in touching)
    lift_angles = data.qpos[v6.jnt_qposadr[v6.actuator_trnid[lifts, 0]]]
    assert np.degrees(lift_angles) == pytest.approx(-10.0, abs=3.0)
    assert np.abs(data.actuator_force[lifts]).max() < 0.2
