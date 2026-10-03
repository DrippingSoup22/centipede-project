"""Tests for the GPU physics backend.

They use the default Newton solver on Volta or newer GPUs and the
conjugate-gradient solver on older ones, which cannot compile Newton, and take
as few physics steps as possible.
"""

import dataclasses

import mujoco
import numpy as np
import pytest
import torch

mjw = pytest.importorskip("mujoco_warp")
if not torch.cuda.is_available():
    pytest.skip("The GPU backend needs a CUDA GPU", allow_module_level=True)

import warp as wp  # noqa: E402

from centipede.environment.simulation.constants import (  # noqa: E402
    LEG_ANGLE_NOISE_RAD,
)
from centipede.environment.simulation.cpu_backend import CPUBackend  # noqa: E402
from centipede.environment.simulation.gpu_backend import (  # noqa: E402
    SOLVERS,
    GPUBackend,
)
from centipede.environment.simulation.model_mapping import ModelMapping  # noqa: E402

SOLVER = "newton" if torch.cuda.get_device_capability() >= (7, 0) else "cg"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path("models/assembly_v2.xml")


@pytest.fixture(scope="module")
def mapping(model):
    return ModelMapping.from_model(model)


def make_backend(model, mapping, world_count, seed=0):
    backend = GPUBackend(model, mapping, world_count, SOLVER, 128, 512)
    backend.reset(seed=seed)
    return backend


def positions(backend):
    return backend.gpu_data.qpos.numpy()


def test_construction_prepares_the_gpu_worlds(model, mapping):
    backend = GPUBackend(model, mapping, 2, "cg", 128, 512)

    assert backend.gpu_model.opt.solver == mujoco.mjtSolver.mjSOL_CG
    assert (backend.gpu_data.naconmax, backend.gpu_data.njmax) == (2 * 128, 512)

    for table in (
        "leg_actuator_ids",
        "leg_qpos_addresses",
        "leg_dof_addresses",
        "body_ids",
        "center_site_ids",
        "geom_owner_indices",
        "geom_categories",
    ):
        assert np.array_equal(getattr(backend, table).numpy(), getattr(mapping, table))

    for field in dataclasses.fields(backend.physical_state):
        tensor = getattr(backend.physical_state, field.name)
        assert tensor.is_cuda, field.name
        assert getattr(backend, f"_{field.name}").ptr == tensor.data_ptr(), field.name


def test_step_holds_each_worlds_actions_for_twenty_milliseconds(model, mapping):
    """A step takes about 10 s on a pre-Volta GPU, so physics runs sparingly here."""
    backend = GPUBackend(model, mapping, 2, SOLVER, 128, 512)
    actions = torch.rand(2, mapping.segment_count, 6, device="cuda") * 2 - 1

    backend.step(actions)

    ctrl = backend.gpu_data.ctrl.numpy()
    leg_motor_ids = mapping.leg_actuator_ids
    for world_index in range(2):
        world_actions = actions[world_index].cpu().numpy()
        assert np.array_equal(ctrl[world_index, leg_motor_ids], world_actions)
    assert not np.delete(ctrl, leg_motor_ids.ravel(), axis=1).any()
    assert np.allclose(backend.gpu_data.time.numpy(), 200 * model.opt.timestep)


# -- Reset ---------------------------------------------------------------------


def test_reset_varies_only_the_legs_within_the_noise_limits(model, mapping):
    backend = make_backend(model, mapping, world_count=3, seed=1)
    qpos_addresses = mapping.leg_qpos_addresses.ravel()
    dof_addresses = mapping.leg_dof_addresses.ravel()
    qpos, qvel = positions(backend), backend.gpu_data.qvel.numpy()

    for world_index in range(3):
        leg_offsets = qpos[world_index, qpos_addresses] - model.qpos0[qpos_addresses]
        assert np.abs(leg_offsets).max() <= LEG_ANGLE_NOISE_RAD
        assert np.abs(leg_offsets).min() > 0.0
        assert np.array_equal(
            np.delete(qpos[world_index], qpos_addresses),
            np.delete(model.qpos0, qpos_addresses).astype(np.float32),
        )
        assert np.abs(qvel[world_index, dof_addresses]).max() > 0.0
        assert not np.delete(qvel[world_index], dof_addresses).any()
    assert not np.array_equal(qpos[0], qpos[1])


def test_seeds_make_resets_reproducible(model, mapping):
    """The same seed repeats the same resets; without a seed they continue."""
    first = make_backend(model, mapping, world_count=2, seed=5)
    second = make_backend(model, mapping, world_count=2, seed=5)
    different = make_backend(model, mapping, world_count=2, seed=6)
    assert np.array_equal(positions(first), positions(second))
    assert not np.array_equal(positions(first), positions(different))

    first_reset = positions(first).copy()
    first.reset()
    second.reset()
    assert not np.array_equal(positions(first), first_reset)
    assert np.array_equal(positions(first), positions(second))


def test_a_mask_resets_only_the_selected_worlds(model, mapping):
    backend = make_backend(model, mapping, world_count=3)
    backend.gpu_data.time.fill_(1.0)
    before = positions(backend).copy()

    backend.reset(torch.tensor([False, True, False], device="cuda"))

    after = positions(backend)
    assert np.array_equal(after[[0, 2]], before[[0, 2]])
    assert not np.array_equal(after[1], before[1])
    assert backend.gpu_data.time.numpy().tolist() == [1.0, 0.0, 1.0]


# -- Physical state ------------------------------------------------------------


def test_physical_state_matches_cpu_mujoco(model, mapping):
    """Each world's state is recomputed by CPU MuJoCo from the same positions and
    speeds. Every speed is random, so the bodies move, not only the legs."""
    backend = make_backend(model, mapping, world_count=2, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(0)
    wp.to_torch(backend.gpu_data.qvel)[:] = torch.randn(
        2, model.nv, device="cuda", generator=generator
    )
    mjw.forward(backend.gpu_model, backend.gpu_data)
    backend._read_state()
    state = backend.physical_state

    for world_index in range(2):
        data = mujoco.MjData(model)
        data.qpos[:] = backend.gpu_data.qpos.numpy()[world_index]
        data.qvel[:] = backend.gpu_data.qvel.numpy()[world_index]
        mujoco.mj_forward(model, data)
        for segment_index in range(mapping.segment_count):
            site_id = mapping.center_site_ids[segment_index]
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 1
            )
            expected = {
                "body_height": data.site_xpos[site_id, 2],
                "body_planar_position": data.site_xpos[site_id, :2],
                "body_quaternion": data.xquat[mapping.body_ids[segment_index]],
                "leg_joint_position": data.qpos[
                    mapping.leg_qpos_addresses[segment_index]
                ],
                "leg_joint_velocity": data.qvel[
                    mapping.leg_dof_addresses[segment_index]
                ],
                "body_angular_velocity": velocity[:3],
                "body_linear_velocity": velocity[3:],
            }
            for field, value in expected.items():
                actual = getattr(state, field)[world_index, segment_index]
                assert np.allclose(actual.cpu().numpy(), value, atol=1e-5), field

        head_tip = data.site_xpos[mapping.head_tip_site_id]
        assert np.allclose(state.head_tip_position[world_index].cpu().numpy(), head_tip)


def flags_from_contact_pool(model, data, world_count):
    """The four contact flags of every world, worked out from shape names."""
    flags = {
        name: np.zeros((world_count, 8), dtype=bool)
        for name in ("left_foot", "right_foot", "body", "leg_leg")
    }
    contact_count = int(data.nacon.numpy()[0])
    pairs = data.contact.geom.numpy()[:contact_count]
    world_indices = data.contact.worldid.numpy()[:contact_count]
    for (geom_a, geom_b), world_index in zip(pairs, world_indices, strict=True):
        name_a, name_b = model.geom(int(geom_a)).name, model.geom(int(geom_b)).name
        if "floor" in (name_a, name_b):
            other = name_b if name_a == "floor" else name_a
            for flag in ("left_foot", "right_foot", "body"):
                if other.endswith(f"_{flag}"):
                    flags[flag][world_index, int(other.split("_")[1])] = True
        else:
            for name in (name_a, name_b):
                flags["leg_leg"][world_index, int(name.split("_")[1])] = True
    return flags


def test_contact_flags_follow_the_contact_pool(model, mapping):
    """States made quickly on the CPU are loaded into two GPU worlds: motors off
    lets the bodies sink to the floor, random actions bring legs together. Every
    flag type must be seen, and always agree with the pool's shape names."""
    source = CPUBackend(model, mapping, world_count=2)
    source.reset(seed=4)
    backend = make_backend(model, mapping, world_count=2)
    state = backend.physical_state
    generator = torch.Generator().manual_seed(0)
    seen = dict.fromkeys(("left_foot", "right_foot", "body", "leg_leg"), False)

    for step_index in range(30):
        actions = torch.zeros(2, mapping.segment_count, 6)
        if step_index >= 5:
            actions = torch.rand(actions.shape, generator=generator) * 2 - 1
        source.step(actions)
        for world_index, data in enumerate(source.world_data):
            wp.to_torch(backend.gpu_data.qpos)[world_index] = torch.from_numpy(
                data.qpos
            )
            wp.to_torch(backend.gpu_data.qvel)[world_index] = torch.from_numpy(
                data.qvel
            )
        mjw.forward(backend.gpu_model, backend.gpu_data)
        backend._read_state()

        expected = flags_from_contact_pool(model, backend.gpu_data, world_count=2)
        actual = {
            "left_foot": state.left_foot_ground_contact,
            "right_foot": state.right_foot_ground_contact,
            "body": state.body_ground_contact,
            "leg_leg": state.leg_leg_contact,
        }
        for flag, values in expected.items():
            assert np.array_equal(actual[flag].cpu().numpy(), values), flag
            seen[flag] |= bool(values.any())

    assert all(seen.values()), seen


def test_gpu_physics_matches_cpu_physics_while_settling():
    """From the same state, with motors off, both backends let the body sink
    onto the floor in the same way. Settling is not chaotic, unlike random
    flailing, so positions and orientations must agree after 60 ms; speeds and
    contact flags at the moment of impact are too sensitive to compare."""
    cpu_model = mujoco.MjModel.from_xml_path("models/assembly_v2.xml")
    cpu_model.opt.solver = SOLVERS[SOLVER]
    gpu_model = mujoco.MjModel.from_xml_path("models/assembly_v2.xml")
    mapping = ModelMapping.from_model(cpu_model)
    cpu = CPUBackend(cpu_model, mapping, world_count=2)
    cpu.reset(seed=1)
    gpu = make_backend(gpu_model, mapping, world_count=2)
    for world_index, data in enumerate(cpu.world_data):
        wp.to_torch(gpu.gpu_data.qpos)[world_index] = torch.from_numpy(data.qpos)
        wp.to_torch(gpu.gpu_data.qvel)[world_index] = torch.from_numpy(data.qvel)
    mjw.forward(gpu.gpu_model, gpu.gpu_data)

    for _ in range(3):
        cpu.step(torch.zeros(2, mapping.segment_count, 6))
        gpu.step(torch.zeros(2, mapping.segment_count, 6, device="cuda"))

    for field, tolerance in (
        ("body_height", 1e-4),
        ("body_planar_position", 1e-4),
        ("head_tip_position", 1e-4),
        ("body_quaternion", 1e-2),
        ("leg_joint_position", 1e-2),
    ):
        expected = getattr(cpu.physical_state, field)
        actual = getattr(gpu.physical_state, field).cpu()
        assert torch.allclose(actual, expected, atol=tolerance), field


# -- Failure -------------------------------------------------------------------


def test_invalid_physics_stops_the_run(model, mapping):
    """Running out of reserved space, and a non-finite state, stop the run;
    the solver's iteration-limit notice does not."""
    crowded = GPUBackend(model, mapping, 2, SOLVER, 1, 512)
    crowded.reset()
    with pytest.raises(RuntimeError, match="Raise contacts_per_world"):
        crowded.step(torch.zeros(2, mapping.segment_count, 6, device="cuda"))

    backend = make_backend(model, mapping, world_count=2)
    overflow_bits = wp.to_torch(backend.gpu_data.overflow)
    overflow_bits[0] = int(mjw.OverflowType.ITERATIONS)
    backend._check_worlds()

    overflow_bits[1] = int(mjw.OverflowType.NEFC)
    with pytest.raises(RuntimeError, match="world 1: .* Raise constraints_per_world"):
        backend._check_worlds()

    backend.reset()
    wp.to_torch(backend.gpu_data.qvel)[0, 10] = float("nan")
    with pytest.raises(RuntimeError, match="world 0: .*non-finite"):
        backend._check_worlds()
