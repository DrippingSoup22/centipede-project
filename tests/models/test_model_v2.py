"""Validation of model v2 against the frozen v1 baseline.

v2 keeps v1's body and physics. It differs only in which shapes may collide,
in separate categories for left and right feet, and in the solver tolerance.
Shape categories themselves are checked by the model mapping's tests.
"""

import mujoco
import numpy as np
import pytest

FLOOR, BODY, LEG, LEFT_FOOT, RIGHT_FOOT, MEMBRANE = range(6)
LIMBS = (LEG, LEFT_FOOT, RIGHT_FOOT)


@pytest.fixture(scope="module")
def v1():
    return mujoco.MjModel.from_xml_path("models/assembly.xml")


@pytest.fixture(scope="module")
def v2():
    return mujoco.MjModel.from_xml_path("models/assembly_v2.xml")


def categories(model):
    return model.geom_user[:, 1].astype(int)


def collide(model, geom_a, geom_b):
    return bool(
        (model.geom_contype[geom_a] & model.geom_conaffinity[geom_b])
        or (model.geom_contype[geom_b] & model.geom_conaffinity[geom_a])
    )


def parent_filtered(model, geom_a, geom_b):
    """MuJoCo never tests shapes on a body and its direct parent body."""
    body_a, body_b = model.geom_bodyid[geom_a], model.geom_bodyid[geom_b]
    return (
        model.body_parentid[body_a] == body_b or model.body_parentid[body_b] == body_a
    )


def test_only_the_intended_settings_differ_from_v1(v1, v2):
    for field in [
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_condim",
        "geom_solref",
        "geom_solimp",
        "jnt_type",
        "jnt_range",
        "jnt_stiffness",
        "dof_damping",
        "dof_armature",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_trnid",
        "actuator_user",
        "site_pos",
        "qpos0",
    ]:
        assert np.array_equal(getattr(v1, field), getattr(v2, field)), field
    assert v2.opt.timestep == v1.opt.timestep
    assert v2.opt.iterations == v1.opt.iterations
    assert (v1.opt.tolerance, v2.opt.tolerance) == (1e-10, 1e-6)


def test_collision_rules_leave_only_fast_pairs(v2):
    """Bodies touch only the floor; legs and feet touch the floor and each other.

    The remaining pairs never put a mesh against anything but the floor plane,
    so none needs MuJoCo Warp's general convex collision code.
    """
    shape_categories = categories(v2)
    mesh, plane = mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_PLANE

    for a in range(v2.ngeom):
        for b in range(a + 1, v2.ngeom):
            kinds = {shape_categories[a], shape_categories[b]}
            expected = MEMBRANE not in kinds and (FLOOR in kinds or kinds <= set(LIMBS))
            assert collide(v2, a, b) == expected, (v2.geom(a).name, v2.geom(b).name)

            types = {v2.geom_type[a], v2.geom_type[b]}
            if expected and not parent_filtered(v2, a, b) and mesh in types:
                assert types == {mesh, plane}


def settle_with_motors_off(model):
    data = mujoco.MjData(model)
    deepest_penetration = 0.0
    for _ in range(20_000):
        mujoco.mj_step(model, data)
        if data.ncon:
            deepest_penetration = max(deepest_penetration, -data.contact.dist.min())
    contact_pairs = {
        (min(a, b), max(a, b))
        for a, b in zip(data.contact.geom1, data.contact.geom2, strict=True)
    }
    return data, contact_pairs, deepest_penetration


def test_settling_with_motors_off_matches_v1(v1, v2):
    """With zero commands the body sinks onto its belly, in both versions.

    Settling is not chaotic, so v2 must end where v1 ends: the same contacts and
    body positions within 0.05 mm despite the looser solver tolerance.
    """
    v1_data, v1_contacts, _ = settle_with_motors_off(v1)
    v2_data, v2_contacts, v2_penetration = settle_with_motors_off(v2)

    assert v2_contacts == v1_contacts
    assert np.abs(v2_data.xpos - v1_data.xpos).max() < 5e-5
    assert v2_penetration < 5e-5


def test_no_leg_passes_through_a_body_under_random_torques(v2):
    """The removed pairs must stay apart even when every motor flails."""
    shape_categories = categories(v2)
    bodies = np.flatnonzero(shape_categories == BODY)
    limbs = np.flatnonzero(np.isin(shape_categories, LIMBS))
    pairs = [
        (int(limb), int(body))
        for limb in limbs
        for body in bodies
        if not parent_filtered(v2, limb, body)
    ] + [(int(a), int(b)) for i, a in enumerate(bodies) for b in bodies[i + 2 :]]
    data = mujoco.MjData(v2)
    rng = np.random.default_rng(0)
    closest = np.inf

    for step in range(20_000):
        if step >= 2_000 and step % 200 == 0:
            data.ctrl[:] = rng.uniform(-1.0, 1.0, v2.nu)
        mujoco.mj_step(v2, data)
        if step % 10 == 0:
            for a, b in pairs:
                distance = mujoco.mj_geomDistance(v2, data, a, b, 1e-3, None)
                closest = min(closest, distance)

    assert closest > 0.0
