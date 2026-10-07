"""Validation of model v3 against v2.

v3 is v2 with a 0.149 ms timestep (134 physics steps per 20 ms action instead
of 200), chosen in Stage 8.2 because it makes the physics about 1.3 times
faster while the contacts, which respond in 0.3 ms, keep their stiffness.
"""

import mujoco
import numpy as np
import pytest

BODY, LEG, LEFT_FOOT, RIGHT_FOOT = 1, 2, 3, 4
LIMBS = (LEG, LEFT_FOOT, RIGHT_FOOT)
SETTLING_SECONDS = 2.0


@pytest.fixture(scope="module")
def v2():
    return mujoco.MjModel.from_xml_path("models/assembly_v2.xml")


@pytest.fixture(scope="module")
def v3():
    return mujoco.MjModel.from_xml_path("models/assembly_v3.xml")


def test_only_the_timestep_differs_from_v2(v2, v3):
    for name in dir(v2):
        value = getattr(v2, name)
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, getattr(v3, name)), name
    for name in ("integrator", "iterations", "tolerance", "cone", "solver"):
        assert getattr(v2.opt, name) == getattr(v3.opt, name), name
    assert round(0.020 / v3.opt.timestep) == 134
    assert 134 * v3.opt.timestep == pytest.approx(0.020, abs=1e-12)
    # Contacts respond in 0.3 ms, and MuJoCo keeps a response at least twice
    # the timestep, so at 0.149 ms none is softened.
    assert v3.geom_solref[:, 0].min() >= 2 * v3.opt.timestep


def settle_with_motors_off(model):
    data = mujoco.MjData(model)
    deepest_penetration = 0.0
    for _ in range(round(SETTLING_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.ncon:
            deepest_penetration = max(deepest_penetration, -data.contact.dist.min())
    contact_pairs = {
        (min(a, b), max(a, b))
        for a, b in zip(data.contact.geom1, data.contact.geom2, strict=True)
    }
    return data, contact_pairs, deepest_penetration


def test_settling_with_motors_off_matches_v2(v2, v3):
    """Settling is not chaotic, so v3 must end where v2 ends after 2 s."""
    v2_data, v2_contacts, _ = settle_with_motors_off(v2)
    v3_data, v3_contacts, v3_penetration = settle_with_motors_off(v3)

    assert v3_contacts == v2_contacts
    assert np.abs(v3_data.xpos - v2_data.xpos).max() < 5e-5
    assert v3_penetration < 5e-5


def test_no_leg_passes_through_a_body_under_random_commands(v3):
    """Longer steps must not let a leg pass through a body while every motor
    flails, with a new command every 20 ms as in training."""
    categories = v3.geom_user[:, 1].astype(int)
    bodies = np.flatnonzero(categories == BODY)
    limbs = np.flatnonzero(np.isin(categories, LIMBS))

    def parent_filtered(a, b):
        body_a, body_b = v3.geom_bodyid[a], v3.geom_bodyid[b]
        return v3.body_parentid[body_a] == body_b or v3.body_parentid[body_b] == body_a

    pairs = [
        (int(limb), int(body))
        for limb in limbs
        for body in bodies
        if not parent_filtered(limb, body)
    ] + [(int(a), int(b)) for i, a in enumerate(bodies) for b in bodies[i + 2 :]]
    data = mujoco.MjData(v3)
    rng = np.random.default_rng(0)
    steps_per_action = round(0.020 / v3.opt.timestep)
    closest = np.inf

    for step in range(round(SETTLING_SECONDS / v3.opt.timestep)):
        if step >= 10 * steps_per_action and step % steps_per_action == 0:
            data.ctrl[:] = rng.uniform(-1.0, 1.0, v3.nu)
        mujoco.mj_step(v3, data)
        if step % 7 == 0:
            for a, b in pairs:
                closest = min(
                    closest, mujoco.mj_geomDistance(v3, data, a, b, 1e-3, None)
                )

    assert closest > 0.0
