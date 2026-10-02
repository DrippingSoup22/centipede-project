"""Tests for the CPU physics backend.

Expected values are read from MuJoCo by element name, independently of the
mapping the backend uses, so the two paths check each other.
"""

import mujoco
import numpy as np
import pytest
import torch

from centipede.environment.simulation.cpu_backend import (
    LEG_ANGLE_NOISE_RAD,
    PHYSICS_STEPS_PER_ACTION,
    CPUBackend,
)
from centipede.environment.simulation.model_mapping import (
    LEG_ACTION_ORDER,
    ModelMapping,
)

SEGMENT_COUNT = 8


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path("models/assembly_v2.xml")


@pytest.fixture(scope="module")
def mapping(model):
    return ModelMapping.from_model(model)


def make_backend(model, mapping, world_count, seed=0):
    backend = CPUBackend(model, mapping, world_count)
    backend.reset(seed=seed)
    return backend


def random_actions(world_count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(world_count, SEGMENT_COUNT, 6, generator=generator) * 2 - 1


def leg_joint_names():
    return [
        f"segment_{segment:02d}_{side}_{role}"
        for segment in range(SEGMENT_COUNT)
        for side, role in LEG_ACTION_ORDER
    ]


def leg_qpos_addresses(model):
    return np.array([model.joint(name).qposadr[0] for name in leg_joint_names()])


def leg_dof_addresses(model):
    return np.array([model.joint(name).dofadr[0] for name in leg_joint_names()])


# -- Reset ---------------------------------------------------------------------


def test_reset_varies_only_the_legs_within_the_noise_limits(model, mapping):
    backend = make_backend(model, mapping, world_count=3, seed=1)
    qpos_addresses = leg_qpos_addresses(model)
    dof_addresses = leg_dof_addresses(model)

    for data in backend.world_data:
        leg_offsets = data.qpos[qpos_addresses] - model.qpos0[qpos_addresses]
        assert np.abs(leg_offsets).max() <= LEG_ANGLE_NOISE_RAD
        assert np.abs(leg_offsets).min() > 0.0
        assert np.array_equal(
            np.delete(data.qpos, qpos_addresses), np.delete(model.qpos0, qpos_addresses)
        )
        assert np.abs(data.qvel[dof_addresses]).max() > 0.0
        assert not np.delete(data.qvel, dof_addresses).any()
        assert data.time == 0.0
        assert not data.ctrl.any()

    first, second = backend.world_data[0], backend.world_data[1]
    assert not np.array_equal(first.qpos, second.qpos)


def test_seeds_make_resets_reproducible(model, mapping):
    """The same seed repeats the same resets; without a seed they continue."""
    first = make_backend(model, mapping, world_count=2, seed=5)
    second = make_backend(model, mapping, world_count=2, seed=5)
    different = make_backend(model, mapping, world_count=2, seed=6)
    for a, b, c in zip(
        first.world_data, second.world_data, different.world_data, strict=True
    ):
        assert np.array_equal(a.qpos, b.qpos)
        assert np.array_equal(a.qvel, b.qvel)
        assert not np.array_equal(a.qpos, c.qpos)

    first_reset = first.world_data[0].qpos.copy()
    first.reset()
    second.reset()
    assert not np.array_equal(first.world_data[0].qpos, first_reset)
    assert np.array_equal(first.world_data[0].qpos, second.world_data[0].qpos)


def test_a_mask_resets_only_the_selected_worlds(model, mapping):
    backend = make_backend(model, mapping, world_count=3)
    backend.step(random_actions(3, seed=0))
    before = [data.qpos.copy() for data in backend.world_data]
    state_before = backend.physical_state.leg_joint_position.clone()

    backend.reset(torch.tensor([False, True, False]))

    assert np.array_equal(backend.world_data[0].qpos, before[0])
    assert np.array_equal(backend.world_data[2].qpos, before[2])
    assert backend.world_data[1].time == 0.0
    assert torch.equal(backend.physical_state.leg_joint_position[0], state_before[0])
    assert not torch.equal(
        backend.physical_state.leg_joint_position[1], state_before[1]
    )


# -- Step ----------------------------------------------------------------------


def test_step_holds_the_actions_for_twenty_milliseconds(model, mapping):
    backend = make_backend(model, mapping, world_count=2)
    actions = random_actions(2, seed=1)

    backend.step(actions)

    leg_motor_ids = [model.actuator(f"{name}_motor").id for name in leg_joint_names()]
    spine_motor_ids = [
        model.actuator(f"segment_{segment:02d}_yaw_motor").id
        for segment in range(1, SEGMENT_COUNT)
    ]
    for world_index, data in enumerate(backend.world_data):
        assert data.time == pytest.approx(PHYSICS_STEPS_PER_ACTION * 1e-4)
        assert np.allclose(
            data.ctrl[leg_motor_ids], actions[world_index].numpy().ravel()
        )
        assert not data.ctrl[spine_motor_ids].any()


def test_worlds_do_not_affect_each_other(model, mapping):
    """World 0 behaves the same whether or not world 1 runs beside it."""
    actions = random_actions(2, seed=2)
    pair = make_backend(model, mapping, world_count=2)
    alone = make_backend(model, mapping, world_count=1)

    for _ in range(3):
        pair.step(actions)
        alone.step(actions[:1])

    assert np.array_equal(pair.world_data[0].qpos, alone.world_data[0].qpos)
    assert torch.equal(
        pair.physical_state.body_linear_velocity[0],
        alone.physical_state.body_linear_velocity[0],
    )


def test_physical_state_matches_mujoco(model, mapping):
    backend = make_backend(model, mapping, world_count=2, seed=3)
    for step_index in range(3):
        backend.step(random_actions(2, seed=10 + step_index))
    state = backend.physical_state
    qpos_addresses = leg_qpos_addresses(model).reshape(SEGMENT_COUNT, 6)
    dof_addresses = leg_dof_addresses(model).reshape(SEGMENT_COUNT, 6)

    for world_index, data in enumerate(backend.world_data):
        for segment in range(SEGMENT_COUNT):
            site = model.site(f"segment_{segment:02d}_center").id
            body = model.body(f"segment_{segment:02d}").id
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 1
            )
            row = world_index, segment
            expected = {
                "body_height": data.site_xpos[site, 2],
                "body_planar_position": data.site_xpos[site, :2],
                "body_quaternion": data.xquat[body],
                "leg_joint_position": data.qpos[qpos_addresses[segment]],
                "leg_joint_velocity": data.qvel[dof_addresses[segment]],
                "body_angular_velocity": velocity[:3],
                "body_linear_velocity": velocity[3:],
            }
            for field, value in expected.items():
                actual = getattr(state, field)[row].numpy()
                assert np.allclose(actual, value, rtol=1e-6, atol=1e-7), field

        head_tip = data.site_xpos[model.site("head_tip").id]
        assert np.allclose(state.head_tip_position[world_index].numpy(), head_tip)


def flags_from_contact_names(model, data):
    """The four contact flags, worked out from the shapes' names."""
    flags = {
        name: np.zeros(SEGMENT_COUNT, dtype=bool)
        for name in ("left_foot", "right_foot", "body", "leg_leg")
    }
    for geom_a, geom_b in zip(data.contact.geom1, data.contact.geom2, strict=True):
        name_a, name_b = model.geom(geom_a).name, model.geom(geom_b).name
        if "floor" in (name_a, name_b):
            other = name_b if name_a == "floor" else name_a
            segment = int(other.split("_")[1])
            for suffix, flag in (
                ("_left_foot", "left_foot"),
                ("_right_foot", "right_foot"),
                ("_body", "body"),
            ):
                if other.endswith(suffix):
                    flags[flag][segment] = True
        else:
            for name in (name_a, name_b):
                flags["leg_leg"][int(name.split("_")[1])] = True
    return flags


def test_contact_flags_follow_the_contact_list(model, mapping):
    """Motors off lets the bodies sink to the floor; random actions bring legs
    together. Every flag type must be seen, and always agree with MuJoCo."""
    backend = make_backend(model, mapping, world_count=1, seed=4)
    state = backend.physical_state
    seen = {"left_foot": False, "right_foot": False, "body": False, "leg_leg": False}
    actions = [torch.zeros(1, SEGMENT_COUNT, 6)] * 5 + [
        random_actions(1, seed=20 + step_index) for step_index in range(25)
    ]

    for step_actions in actions:
        backend.step(step_actions)
        expected = flags_from_contact_names(model, backend.world_data[0])
        actual = {
            "left_foot": state.left_foot_ground_contact[0].numpy(),
            "right_foot": state.right_foot_ground_contact[0].numpy(),
            "body": state.body_ground_contact[0].numpy(),
            "leg_leg": state.leg_leg_contact[0].numpy(),
        }
        for flag, values in expected.items():
            assert np.array_equal(actual[flag], values), flag
            seen[flag] |= bool(values.any())

    assert all(seen.values()), seen


# -- Failure -------------------------------------------------------------------


def test_invalid_physics_stops_the_run(model, mapping):
    """MuJoCo's own failure report, and a non-finite state, both stop the run."""
    backend = make_backend(model, mapping, world_count=2)
    backend.world_data[1].qvel[10] = np.nan
    with pytest.raises(RuntimeError, match="world 1: MuJoCo reported"):
        backend.step(torch.zeros(2, SEGMENT_COUNT, 6))

    backend.reset()
    backend.world_data[0].qpos[20] = np.inf
    with pytest.raises(RuntimeError, match="world 0: non-finite"):
        backend._check_world(0)
