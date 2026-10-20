"""Validation of model v5: the body of Scolopendra heros, in three files.

v5 is a head without legs and 21 leg-bearing units, 152 mm and 11.56 g as
the animals measured; its physics are v4's scaled to that size by dynamic
similarity, simulated at the settings revised for speed (docs/model.md,
"Model v5"). Its half is the same body's first ten units, and the headless
tester the half without its head. The checks below repeat the parts of its
validation that a change could break.
"""

import mujoco
import numpy as np
import pytest

# A physics check (pyproject.toml): run with -m physics when a model changes.
pytestmark = pytest.mark.physics

FLOOR, BODY, LEG, LEFT_FOOT, RIGHT_FOOT = range(5)
LIMBS = (LEG, LEFT_FOOT, RIGHT_FOOT)
LEG_ROLES = ("sweep", "lift", "knee", "ankle")
UNITS = 21
HALF_UNITS = 10
STEPS_PER_ACTION = 15
LEG_REACH = 3


@pytest.fixture(scope="module")
def v4():
    return mujoco.MjModel.from_xml_path("models/assembly_v4.xml")


@pytest.fixture(scope="module")
def v5():
    return mujoco.MjModel.from_xml_path("models/assembly_v5_full.xml")


@pytest.fixture(scope="module")
def half():
    return mujoco.MjModel.from_xml_path("models/assembly_v5_half.xml")


@pytest.fixture(scope="module")
def headless():
    return mujoco.MjModel.from_xml_path("models/assembly_v5_headless.xml")


def by_role(model, names, values):
    """Each role (the name's last word) with the set of its values."""
    roles = {}
    for name, value in zip(names, values, strict=True):
        roles.setdefault(name.rsplit("_", 1)[1], set()).add(value)
    return roles


def test_v5_is_s_heros_with_v4s_physics_scaled_to_its_size(v4, v5):
    """The animal's size and mass; v4's joint ranges but the knee's, which
    bends 10 degrees less; every torque one factor, M L, times v4's, and
    the leg transient and damping, over that factor, the square root of
    the length scale L; 15 steps per action, with the contacts and limits
    responding in twice the timestep."""
    data = mujoco.MjData(v5)
    mujoco.mj_forward(v5, data)
    assert v5.body_subtreemass[1] == pytest.approx(0.01156, rel=1e-3)
    tip = data.site_xpos[v5.site("head_tip").id]
    rear = v5.geom(f"segment_{UNITS - 1:02d}_body").id
    rear_end = data.geom_xpos[rear, 0] + v5.geom_aabb[rear, 0] - v5.geom_aabb[rear, 3]
    assert tip[0] - rear_end == pytest.approx(0.152, rel=0.02)
    names = [v5.joint(j).name for j in range(v5.njnt)]
    # The first unit's legs, and no other joint: the head has no legs.
    assert sum("segment_00" in name for name in names) == len(LEG_ROLES) * 2

    assert STEPS_PER_ACTION * v5.opt.timestep == pytest.approx(0.020, abs=1e-12)
    response = (2 * v5.opt.timestep, 1.0)
    assert {tuple(row) for row in v5.geom_solref} == {response}
    assert {tuple(row) for row in v5.jnt_solref[v5.jnt_limited == 1]} == {response}
    assert {tuple(row) for row in v5.pair_solref} == {response}

    def motors(model):
        names = [model.actuator(m).name.removesuffix("_motor") for m in range(model.nu)]
        return by_role(model, names, model.actuator_gear[:, 0].round(12))

    v4_torques, v5_torques = motors(v4), motors(v5)
    (sweep4,), (sweep5,) = v4_torques["sweep"], v5_torques["sweep"]
    torque_scale = sweep5 / sweep4
    mass_scale = (v5.body_subtreemass[1] / UNITS) / (v4.body_subtreemass[1] / 12)
    time_scale = np.sqrt(torque_scale / mass_scale)
    for role, (torque,) in v4_torques.items():
        (torque5,) = v5_torques[role]
        assert torque5 == pytest.approx(torque * torque_scale, rel=1e-3)

    def joints(model):
        hinges = range(1, model.njnt)
        names = [model.joint(j).name for j in hinges]
        values = [
            (*model.jnt_range[j].round(6), model.dof_damping[model.jnt_dofadr[j]])
            for j in hinges
        ]
        return by_role(model, names, values)

    v4_joints, v5_joints = joints(v4), joints(v5)
    for role, values in v4_joints.items():
        ((low, high, damping),) = values
        ((low5, high5, damping5),) = v5_joints[role]
        bend = np.radians(10) if role == "knee" else 0.0
        assert (low5, high5) == pytest.approx((low, high - bend), abs=1e-6)
        assert damping5 / damping / torque_scale == pytest.approx(time_scale, rel=0.01)

    inertia = np.zeros((v5.nv, v5.nv))
    mujoco.mj_fullM(v5, data, inertia)
    for joint_id in range(1, v5.njnt):
        if v5.joint(joint_id).name.endswith(LEG_ROLES):
            dof = v5.jnt_dofadr[joint_id]
            transient = inertia[dof, dof] / v5.dof_damping[dof]
            assert transient == pytest.approx(0.002 * time_scale, rel=0.05)


def test_the_full_body_and_its_testers_share_their_units_and_settings(
    v5, half, headless
):
    """The free joint on the middle unit; the solver's tolerance 1e-4; a leg
    touches the floor, and the legs of its own side up to three units away
    through contact pairs; the half is the full body's first ten units, every
    part alike, and the headless tester the same but for the first unit,
    whose body is a trunk unit's."""
    for model, units in ((v5, UNITS), (half, HALF_UNITS), (headless, HALF_UNITS)):
        root = model.jnt_bodyid[0]
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert model.body(root).name == f"segment_{units // 2:02d}"
        assert model.body_parentid[root] == 0
        assert model.opt.tolerance == pytest.approx(1e-4)
        categories = model.geom_user[:, 1].astype(int)
        assert set(model.geom_conaffinity[np.isin(categories, LIMBS)]) == {1}
        pairs = {
            tuple(sorted(tuple(model.geom(g).name.split("_")[1:3]) for g in pair))
            for pair in zip(model.pair_geom1, model.pair_geom2, strict=True)
        }
        expected = {
            ((f"{a:02d}", side), (f"{b:02d}", side))
            for side in ("left", "right")
            for a in range(units)
            for b in range(a + 1, min(units, a + LEG_REACH + 1))
        }
        assert pairs == expected
    trunk_unit = v5.body_mass[v5.body("segment_01").id]
    for tester in (half, headless):
        assert tester.opt.timestep == v5.opt.timestep
        for b in range(1, tester.nbody):
            name = tester.body(b).name
            mass = v5.body_mass[v5.body(name).id]
            if tester is headless and name == "segment_00":
                mass = trunk_unit
            assert tester.body_mass[b] == pytest.approx(mass, rel=1e-9)
        for j in range(1, tester.njnt):
            name = tester.joint(j).name
            assert np.array_equal(tester.jnt_range[j], v5.jnt_range[v5.joint(name).id])
        for a in range(tester.nu):
            name = tester.actuator(a).name
            gear = v5.actuator_gear[v5.actuator(name).id, 0]
            assert tester.actuator_gear[a, 0] == gear
    assert not any(
        headless.geom(g).name.startswith("head") for g in range(headless.ngeom)
    )


def ground_contacts(model, data):
    """How many feet and bodies touch the floor."""
    categories = model.geom_user[:, 1].astype(int)
    touching = {
        int(b if categories[a] == FLOOR else a)
        for a, b in zip(data.contact.geom1, data.contact.geom2, strict=True)
        if FLOOR in (categories[a], categories[b])
    }
    feet = sum(categories[geom] in (LEFT_FOOT, RIGHT_FOOT) for geom in touching)
    return feet, sum(categories[geom] == BODY for geom in touching)


def held(model, lift_command, seconds):
    """The data after ``seconds`` with every lift motor at ``lift_command``,
    and the deepest any shape sank into another."""
    data = mujoco.MjData(model)
    for motor_id in range(model.nu):
        if model.actuator(motor_id).name.endswith("_lift_motor"):
            data.ctrl[motor_id] = lift_command
    deepest = 0.0
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.ncon:
            deepest = max(deepest, -data.contact.dist.min())
    return data, deepest


def test_motors_off_the_body_sinks_and_the_lift_motors_hold_it_up(v5):
    """With every motor off the body settles onto its belly, as v1 to v4 do;
    with the lift motors pressing down at 40% it stands on all 42 feet."""
    settled, penetration = held(v5, 0.0, 1.0)
    assert ground_contacts(v5, settled)[1] == UNITS
    assert penetration < 2e-4

    standing, penetration = held(v5, -0.4, 1.0)
    assert ground_contacts(v5, standing) == (2 * UNITS, 0)
    assert penetration < 2e-4


def test_no_leg_passes_through_a_body_under_random_commands(v5):
    """Every motor flails, with a new command every 20 ms as in training;
    a leg can reach only the bodies of the segments next to its own and
    the next ones."""
    categories = v5.geom_user[:, 1].astype(int)
    owners = v5.geom_user[:, 0].astype(int)
    bodies = np.flatnonzero(categories == BODY)
    limbs = np.flatnonzero(np.isin(categories, LIMBS))

    def parent_filtered(a, b):
        body_a, body_b = v5.geom_bodyid[a], v5.geom_bodyid[b]
        return v5.body_parentid[body_a] == body_b or v5.body_parentid[body_b] == body_a

    pairs = [
        (int(limb), int(body))
        for limb in limbs
        for body in bodies
        if abs(owners[limb] - owners[body]) <= 2 and not parent_filtered(limb, body)
    ] + [
        (int(a), int(b))
        for a in bodies
        for b in bodies
        if 2 <= owners[b] - owners[a] <= 3
    ]
    data = mujoco.MjData(v5)
    rng = np.random.default_rng(0)
    closest = np.inf

    for step in range(60 * STEPS_PER_ACTION):
        if step >= 10 * STEPS_PER_ACTION and step % STEPS_PER_ACTION == 0:
            data.ctrl[:] = rng.uniform(-1.0, 1.0, v5.nu)
        mujoco.mj_step(v5, data)
        if step % 2 == 0:  # every 2.7 ms
            for a, b in pairs:
                distance = mujoco.mj_geomDistance(v5, data, a, b, 1e-3, None)
                closest = min(closest, distance)

    assert closest > 0.0
