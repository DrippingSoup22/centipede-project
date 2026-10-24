"""Tests for the CPU physics backend.

Expected values are read from MuJoCo by element name, independently of the
mapping the backend uses, so the two paths check each other.
"""

import mujoco
import numpy as np
import pytest
import torch

from centipede.environment.simulation.constants import (
    ACTION_DURATION_S,
    LEG_ANGLE_NOISE_RAD,
)
from centipede.environment.simulation.cpu_backend import CPUBackend
from centipede.environment.simulation.model_mapping import (
    LEG_ROLES,
    LEG_SIDES,
    ModelMapping,
)
from centipede.environment.simulation.settings import SimulationSettings
from centipede.environment.simulation.simulation import (
    model_spec,
    physics_steps_per_action,
)

SEGMENT_COUNT = 8


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path("models/assembly_v3.xml")


@pytest.fixture(scope="module")
def mapping(model):
    return ModelMapping.from_model(model)


def make_backend(model, mapping, world_count, seed=0, heading_range=0.0):
    backend = CPUBackend(
        model, mapping, world_count, physics_steps_per_action(model), heading_range
    )
    backend.reset(seed=seed)
    return backend


def random_actions(world_count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(world_count, SEGMENT_COUNT, 6, generator=generator) * 2 - 1


def leg_joint_names():
    return [
        f"segment_{segment:02d}_{side}_{role}"
        for segment in range(SEGMENT_COUNT)
        for side in LEG_SIDES
        for role in LEG_ROLES[:3]  # v3's legs have three joints
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


def test_a_start_heading_range_turns_the_whole_body(model, mapping):
    """Only the root's orientation changes, by a turn about the vertical within
    the range; the legs draw the same noise as without it."""
    straight = make_backend(model, mapping, world_count=4, seed=2)
    turned = make_backend(model, mapping, world_count=4, seed=2, heading_range=0.5)
    address = model.joint("root").qposadr[0] + 3
    headings = []
    for plain, data in zip(straight.world_data, turned.world_data, strict=True):
        w, x, y, z = data.qpos[address : address + 4]
        assert x == y == 0.0
        headings.append(2 * np.arctan2(z, w))
        assert np.array_equal(
            np.delete(data.qpos, range(address, address + 4)),
            np.delete(plain.qpos, range(address, address + 4)),
        )
        assert np.array_equal(data.qvel, plain.qvel)
    assert np.abs(headings).max() <= 0.5
    assert len(set(headings)) == 4
    # The physical state follows: the head faces its new heading.
    head = turned.physical_state.body_quaternion[0, 0].numpy()
    assert np.isclose(2 * np.arctan2(head[3], head[0]), headings[0], atol=1e-5)


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
    spine_actions = torch.rand(2, SEGMENT_COUNT - 1) * 2 - 1

    backend.step(actions, spine_actions)

    leg_motor_ids = [model.actuator(f"{name}_motor").id for name in leg_joint_names()]
    spine_motor_ids = [
        model.actuator(f"segment_{segment:02d}_yaw_motor").id
        for segment in range(1, SEGMENT_COUNT)
    ]
    for world_index, data in enumerate(backend.world_data):
        assert data.time == pytest.approx(ACTION_DURATION_S)
        assert np.allclose(
            data.ctrl[leg_motor_ids], actions[world_index].numpy().ravel()
        )
        assert np.allclose(data.ctrl[spine_motor_ids], spine_actions[world_index])

    backend.step(actions)  # without spine actions, the spine is passive
    assert not any(data.ctrl[spine_motor_ids].any() for data in backend.world_data)


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
        spine_actions = torch.rand(2, SEGMENT_COUNT - 1) * 2 - 1
        backend.step(random_actions(2, seed=10 + step_index), spine_actions)
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
                # The joint behind the segment; the rear has none.
                "spine_yaw_position": 0.0,
                "spine_yaw_velocity": 0.0,
            }
            if segment < SEGMENT_COUNT - 1:
                spine = model.joint(f"segment_{segment + 1:02d}_yaw")
                expected["spine_yaw_position"] = data.qpos[spine.qposadr[0]]
                expected["spine_yaw_velocity"] = data.qvel[spine.dofadr[0]]
            # The front end: the head's tip, then the joint with the one ahead.
            expected["front_end_position"] = (
                data.site_xpos[model.site("head_tip").id, :2]
                if segment == 0
                else data.xanchor[model.joint(f"segment_{segment:02d}_yaw").id, :2]
            )
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


def settled_backend_measuring_effort(model, mapping):
    """One world measuring effort, settled for half a second with the motors
    off: the body rests on the floor."""
    backend = CPUBackend(
        model, mapping, 1, physics_steps_per_action(model), measure_effort=True
    )
    backend.reset(seed=0)
    for _ in range(25):
        backend.step(torch.zeros(1, SEGMENT_COUNT, 6))
    return backend


def test_effort_measures_the_floors_support_the_legs_pressing_and_the_work(
    model, mapping
):
    backend = settled_backend_measuring_effort(model, mapping)
    state = backend.physical_state
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    support = state.foot_vertical_force.sum() + state.belly_vertical_force.sum()
    assert support.item() == pytest.approx(weight, rel=1e-3)
    assert state.motor_positive_work.sum() == state.motor_negative_work.sum() == 0

    # Segments 2 and 3 sweep their left legs toward each other: both feel the
    # same press, and only they; a mean of squares is at least the squared mean.
    actions = torch.zeros(1, SEGMENT_COUNT, 6)
    actions[0, 2, 0], actions[0, 3, 0] = -1.0, 1.0
    for _ in range(3):
        backend.step(actions)
    pressed = state.leg_leg_force[0]
    assert pressed[2] == pressed[3] > 0.1 * weight / SEGMENT_COUNT
    assert pressed.sum() == pytest.approx(2 * pressed[2].item())
    assert state.leg_leg_force_squared[0, 2] >= pressed[2] ** 2
    assert (state.motor_positive_work[0, 2:4] > 0).all()
    # A torque motor's force is its command, as a share of its strength.
    assert torch.allclose(state.motor_force[0, :, :-1], actions[0])

    # Every sweep pushes back: the floor pushes the feet forward.
    actions[..., 0] = actions[..., 3] = -0.8
    backend.step(actions)
    assert state.foot_forward_force.sum() > 0


TILTED_GROUND = """<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="1 1 0.01" euler="0 -20 0" />
  </worldbody>
</mujoco>"""


def tilted_model(tmp_path):
    """Model v3 on ground tilted 20 degrees (a terrain file)."""
    (tmp_path / "tilted.xml").write_text(TILTED_GROUND, encoding="utf-8")
    settings = SimulationSettings.from_section(
        {
            "model_path": "models/assembly_v3.xml",
            "backend": "cpu",
            "terrain_path": str(tmp_path / "tilted.xml"),
        }
    )
    return model_spec(settings).compile()


def test_on_tilted_ground_the_feet_and_belly_carry_the_whole_weight(tmp_path):
    """At rest on a 20 degree slope, the floor's vertical push on the feet
    and belly is the body's weight; their normal force would be its cosine,
    0.94, the rest held by friction."""
    model = tilted_model(tmp_path)
    backend = settled_backend_measuring_effort(model, ModelMapping.from_model(model))
    state = backend.physical_state
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    support = state.foot_vertical_force.sum() + state.belly_vertical_force.sum()
    assert support.item() == pytest.approx(weight, rel=1e-2)


def test_the_floor_push_counts_every_shape_on_the_floor_along_it():
    """At rest with its motors off, the tester lies on more than its feet:
    the floor holds it up and pushes it nowhere along the floor (along the
    units' own pitched axes, part of the support would read as a push).
    Pushed gently forward along the floor, the body is held by the floor
    alone: the floor's push on all its shapes cancels the push."""
    model = mujoco.MjModel.from_xml_path("models/assembly_v5_headless.xml")
    mapping = ModelMapping.from_model(model)
    backend = CPUBackend(
        model, mapping, 1, physics_steps_per_action(model), measure_effort=True
    )
    backend.reset(seed=0)
    actions = torch.zeros(1, mapping.segment_count, mapping.leg_actuator_ids.shape[1])
    for _ in range(25):
        backend.step(actions)
    state = backend.physical_state
    data = backend.world_data[0]
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    assert abs(state.floor_forward_force.sum().item()) < 0.002 * weight

    push = 0.05 * weight / mapping.segment_count
    flat_axes = data.xmat[mapping.body_ids][:, [0, 3, 6]] * [1.0, 1.0, 0.0]
    flat_axes /= np.linalg.norm(flat_axes, axis=1, keepdims=True)
    data.xfrc_applied[mapping.body_ids, :3] = push * flat_axes
    backend.step(actions)

    held = state.floor_forward_force.sum().item()
    assert held == pytest.approx(-push * mapping.segment_count, rel=0.1)


@pytest.mark.parametrize(
    "model_path", ["models/assembly_v3.xml", "models/assembly_v5_half.xml"]
)
def test_joint_forces_carry_a_push_from_behind_up_the_body(model_path):
    """The rear segment, pushed forward with three body weights, pushes the
    segment ahead with most of it, and each joint further up carries less:
    each joint moves everything ahead of it. The head has no joint ahead.
    Net of both its joints, the rear segment is held back by most of the
    push, and the joints' forces on the whole body cancel. So whichever
    segment roots MuJoCo's body tree: the head in v3, the middle unit in v5."""
    model = mujoco.MjModel.from_xml_path(model_path)
    mapping = ModelMapping.from_model(model)
    backend = CPUBackend(
        model, mapping, 1, physics_steps_per_action(model), measure_joint_forces=True
    )
    backend.reset(seed=0)
    push = 3 * model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    backend.world_data[0].xfrc_applied[mapping.body_ids[-1], 0] = push

    backend.step(
        torch.zeros(1, mapping.segment_count, mapping.leg_actuator_ids.shape[1])
    )

    force = backend.physical_state.front_joint_force[0]
    assert force[0] == 0
    assert force[-1] < -0.5 * push
    assert (force[1:].diff() < 0).all()
    net = backend.physical_state.net_joint_force[0]
    assert net[-1, 0] == force[-1]
    assert net[:, 0].sum().abs() < 0.01 * push


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
