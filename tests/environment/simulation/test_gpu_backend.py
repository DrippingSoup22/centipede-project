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
# A physics check (pyproject.toml): run with -m physics when the simulation changes.
pytestmark = pytest.mark.physics

import warp as wp  # noqa: E402

from centipede.environment.simulation.constants import (  # noqa: E402
    ACTION_DURATION_S,
    LEG_ANGLE_NOISE_RAD,
)
from centipede.environment.simulation.cpu_backend import CPUBackend  # noqa: E402
from centipede.environment.simulation.gpu_backend import (  # noqa: E402
    SOLVERS,
    GPUBackend,
)
from centipede.environment.simulation.model_mapping import ModelMapping  # noqa: E402
from centipede.environment.simulation.physical_state import (  # noqa: E402
    EFFORT_FIELDS,
)
from centipede.environment.simulation.simulation import (  # noqa: E402
    physics_steps_per_action,
)

SOLVER = "newton" if torch.cuda.get_device_capability() >= (7, 0) else "cg"
# Physics steps per action of the model the tests use, model v3.
STEPS = physics_steps_per_action(mujoco.MjModel.from_xml_path("models/assembly_v3.xml"))


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path("models/assembly_v3.xml")


@pytest.fixture(scope="module")
def mapping(model):
    return ModelMapping.from_model(model)


def make_backend(model, mapping, world_count, seed=0, heading_range=0.0):
    backend = GPUBackend(
        model, mapping, world_count, STEPS, SOLVER, 128, 512, heading_range
    )
    backend.reset(seed=seed)
    return backend


def positions(backend):
    return backend.gpu_data.qpos.numpy()


def test_construction_prepares_the_gpu_worlds(model, mapping):
    backend = GPUBackend(model, mapping, 2, STEPS, "cg", 128, 512)

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

    effort_views = dict(zip(EFFORT_FIELDS, backend._effort, strict=True))
    for field in dataclasses.fields(backend.physical_state):
        tensor = getattr(backend.physical_state, field.name)
        view = effort_views.get(field.name) or getattr(backend, f"_{field.name}")
        assert tensor.is_cuda, field.name
        assert view.ptr == tensor.data_ptr(), field.name

    # The simulation's category is MuJoCo Warp's own arrays, seen as tensors.
    facts = backend.diagnostics.facts
    assert facts.qpos.data_ptr() == backend.gpu_data.qpos.ptr
    assert facts.constraint_rows.data_ptr() == backend.gpu_data.nefc.ptr
    assert facts.solver_iterations.data_ptr() == backend.gpu_data.solver_niter.ptr
    assert facts.contact_count.data_ptr() == backend.gpu_data.nacon.ptr


def test_step_holds_each_worlds_actions_for_twenty_milliseconds(model, mapping):
    """A step takes about 10 s on a pre-Volta GPU, so physics runs sparingly here."""
    backend = GPUBackend(model, mapping, 2, STEPS, SOLVER, 128, 512)
    actions = torch.rand(2, mapping.segment_count, 6, device="cuda") * 2 - 1
    spine_actions = torch.rand(2, mapping.segment_count - 1, device="cuda") * 2 - 1

    backend.step(actions, spine_actions)

    ctrl = backend.gpu_data.ctrl.numpy()
    leg_motor_ids = mapping.leg_actuator_ids
    for world_index in range(2):
        world_actions = actions[world_index].cpu().numpy()
        assert np.array_equal(ctrl[world_index, leg_motor_ids], world_actions)
        assert np.array_equal(
            ctrl[world_index, mapping.spine_actuator_ids],
            spine_actions[world_index].cpu().numpy(),
        )
    assert np.allclose(backend.gpu_data.time.numpy(), ACTION_DURATION_S)


def test_replayed_physics_steps_match_steps_launched_from_python(model, mapping):
    """The second step replays the graph the first recorded; from the same
    state, it must move the worlds as MuJoCo Warp's steps launched one by one.

    MuJoCo Warp on a GPU is not deterministic: it adds some values in parallel,
    in no fixed order. Two launches from the same state usually end within
    1e-5 (metres or radians), but a contact or a limit reached in one and
    missed in the other can part them by 5e-3 within the action. The action
    moves the legs by 0.2 rad (the median), so a replay with wrong or missing
    commands would part from the reference by far more than the tolerance.
    """
    backend = make_backend(model, mapping, 2)
    reference = make_backend(model, mapping, 2)
    generator = torch.Generator(device="cuda").manual_seed(0)
    leg_motor_ids = torch.as_tensor(mapping.leg_actuator_ids.ravel(), device="cuda")

    def random_actions():
        shape = (2, mapping.segment_count, 6)
        return torch.rand(shape, generator=generator, device="cuda") * 2 - 1

    backend.step(random_actions())
    for name in ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "time"):
        wp.copy(getattr(reference.gpu_data, name), getattr(backend.gpu_data, name))
    actions = random_actions()
    backend.step(actions)
    wp.to_torch(reference.gpu_data.ctrl)[:, leg_motor_ids] = actions.flatten(1)
    for _ in range(STEPS):
        mjw.step(reference.gpu_model, reference.gpu_data)

    assert backend._physics_graph is not None
    assert np.allclose(positions(backend), positions(reference), rtol=0, atol=2e-2)


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


def test_a_start_heading_range_turns_the_whole_body(model, mapping):
    """Only the root's orientation changes, by a turn about the vertical within
    the range; the legs draw the same noise as without it."""
    straight = make_backend(model, mapping, world_count=4, seed=2)
    turned = make_backend(model, mapping, world_count=4, seed=2, heading_range=0.5)
    address = mapping.root_qpos_address + 3
    plain, qpos = positions(straight), positions(turned)
    w, x, y, z = (qpos[:, address + index] for index in range(4))
    assert not x.any() and not y.any()
    headings = 2 * np.arctan2(z, w)
    assert np.abs(headings).max() <= 0.5 + 1e-6
    assert len(set(headings.tolist())) == 4
    root_quaternion = range(address, address + 4)
    others = [index for index in range(qpos.shape[1]) if index not in root_quaternion]
    assert np.array_equal(qpos[:, others], plain[:, others])
    head = turned.physical_state.body_quaternion[:, 0].cpu().numpy()
    assert np.allclose(2 * np.arctan2(head[:, 3], head[:, 0]), headings, atol=1e-5)


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


@pytest.mark.parametrize(
    "model_path",
    ["models/assembly_v3.xml", "models/assembly_v4.xml", "models/assembly_v5_half.xml"],
)
def test_physical_state_matches_cpu_mujoco(model_path):
    """Each world's state is recomputed by CPU MuJoCo from the same positions and
    speeds. Every speed is random, so the bodies move, not only the legs. v3's
    legs have three joints and v4's four; v5's first segment carries a head
    without legs, and its free joint is on its middle unit (the half's, as the
    full model's)."""
    model = mujoco.MjModel.from_xml_path(model_path)
    mapping = ModelMapping.from_model(model)
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
                # The joint behind the segment; the rear has none.
                "spine_yaw_position": 0.0,
                "spine_yaw_velocity": 0.0,
            }
            if segment_index < mapping.segment_count - 1:
                expected["spine_yaw_position"] = data.qpos[
                    mapping.spine_qpos_addresses[segment_index]
                ]
                expected["spine_yaw_velocity"] = data.qvel[
                    mapping.spine_dof_addresses[segment_index]
                ]
            # The front end: the head's tip, then the joint with the one ahead.
            spine_joint_ids = model.actuator_trnid[mapping.spine_actuator_ids, 0]
            expected["front_end_position"] = (
                data.site_xpos[mapping.head_tip_site_id, :2]
                if segment_index == 0
                else data.xanchor[spine_joint_ids[segment_index - 1], :2]
            )
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
    source = CPUBackend(model, mapping, world_count=2, physics_steps_per_action=STEPS)
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
    cpu_model = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    cpu_model.opt.solver = SOLVERS[SOLVER]
    gpu_model = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    mapping = ModelMapping.from_model(cpu_model)
    cpu = CPUBackend(cpu_model, mapping, world_count=2, physics_steps_per_action=STEPS)
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


def test_gpu_effort_matches_cpu_effort():
    """From the same settled state, with two neighbours' legs swept together,
    both backends measure the same effort: the floor's support, the legs'
    press, the feet's forward forces, and the motors' work."""
    cpu_model = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    cpu_model.opt.solver = SOLVERS[SOLVER]
    mapping = ModelMapping.from_model(cpu_model)
    segment_count = mapping.segment_count
    cpu = CPUBackend(cpu_model, mapping, 1, STEPS, measure_effort=True)
    cpu.reset(seed=0)
    for _ in range(25):
        cpu.step(torch.zeros(1, segment_count, 6))
    gpu_model = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    gpu = GPUBackend(
        gpu_model, mapping, 1, STEPS, SOLVER, 128, 512, measure_effort=True
    )
    gpu.reset(seed=0)
    wp.to_torch(gpu.gpu_data.qpos)[0] = torch.from_numpy(cpu.world_data[0].qpos)
    wp.to_torch(gpu.gpu_data.qvel)[0] = torch.from_numpy(cpu.world_data[0].qvel)
    mjw.forward(gpu.gpu_model, gpu.gpu_data)

    actions = torch.zeros(1, segment_count, 6)
    actions[0, 2, 0], actions[0, 3, 0] = -1.0, 1.0
    actions[0, 5:, 0] = actions[0, 5:, 3] = -0.8
    for _ in range(2):
        cpu.step(actions)
        gpu.step(actions.cuda())

    def measured(field: str) -> tuple[torch.Tensor, torch.Tensor]:
        expected = getattr(cpu.physical_state, field)
        assert expected.abs().max() > 0, field
        return getattr(gpu.physical_state, field).cpu(), expected

    # The legs' press and the motors' work belong to each segment exactly.
    for field in EFFORT_FIELDS[4:]:
        actual, expected = measured(field)
        assert torch.allclose(actual, expected, atol=0.05 * expected.abs().max())
    # A resting body's load and friction are shared among its shapes as each
    # solver finds (they are statically indeterminate, and the laptop's CG
    # solver stops at its iteration limit), so only the totals compare;
    # friction, the least converged, only roughly.
    for field, tolerance in (
        ("foot_normal_force", 0.03),
        ("belly_normal_force", 0.03),
        ("foot_forward_force", 0.5),
        ("floor_forward_force", 0.5),
    ):
        actual, expected = measured(field)
        assert actual.sum().item() == pytest.approx(
            expected.sum().item(), rel=tolerance
        )


@pytest.mark.parametrize(
    "model_path", ["models/assembly_v3.xml", "models/assembly_v5_half.xml"]
)
def test_gpu_joint_forces_match_cpu_joint_forces(model_path):
    """With the rear segment pushed forward by three body weights, as the
    step's body forces, both backends measure each joint's push alike,
    whichever segment roots the body tree (v5's middle unit). Near the head
    the push is small beside the friction, which each solver shares among
    the feet as it finds, so the joints are compared against the largest
    push."""
    model = mujoco.MjModel.from_xml_path(model_path)
    mapping = ModelMapping.from_model(model)
    steps = physics_steps_per_action(model)
    cpu = CPUBackend(model, mapping, 1, steps, measure_joint_forces=True)
    cpu.reset(seed=0)
    gpu = GPUBackend(
        model, mapping, 1, steps, SOLVER, 128, 512, measure_joint_forces=True
    )
    gpu.reset(seed=0)
    body_forces = torch.zeros(1, mapping.segment_count, 3)
    body_forces[0, -1, 0] = (
        3 * model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    )
    actions = torch.zeros(1, mapping.segment_count, mapping.leg_actuator_ids.shape[1])

    cpu.step(actions, body_forces=body_forces)
    gpu.step(actions.cuda(), body_forces=body_forces.cuda())

    expected = cpu.physical_state.front_joint_force[0]
    actual = gpu.physical_state.front_joint_force[0].cpu()
    assert actual[0] == expected[0] == 0
    assert torch.allclose(actual, expected, atol=0.05 * expected.abs().max())
    expected = cpu.physical_state.net_joint_force[0]
    actual = gpu.physical_state.net_joint_force[0].cpu()
    assert torch.allclose(actual, expected, atol=0.05 * expected.abs().max())


# -- Failure -------------------------------------------------------------------


def test_invalid_physics_stops_the_run(model, mapping):
    """Running out of reserved space, and a non-finite state with no restart
    allowed, stop the run; the solver's iteration-limit notice does not."""
    crowded = GPUBackend(model, mapping, 2, STEPS, SOLVER, 1, 512)
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
    backend.gpu_data.overflow.zero_()
    wp.to_torch(backend.gpu_data.qvel)[1, 10] = float("nan")
    with pytest.raises(RuntimeError, match="world 1: .*non-finite"):
        backend._check_worlds()


def test_allowed_non_finite_worlds_restart_from_the_actions_start(
    model, mapping, monkeypatch
):
    """With one restart allowed per action, a world the physics makes
    non-finite goes back to the action's start, flagged for the environment,
    with finite state and effort; the other world steps on."""
    backend = GPUBackend(model, mapping, 2, STEPS, SOLVER, 128, 512, 0.0, True, True, 1)
    backend.reset(seed=0)
    actions = torch.rand(2, mapping.segment_count, 6, device="cuda") * 2 - 1
    backend.step(actions)
    start = positions(backend).copy()
    replay = wp.capture_launch

    def replay_spoiling_world_1(graph):
        replay(graph)
        wp.to_torch(backend.gpu_data.qvel)[1, 10] = float("nan")

    monkeypatch.setattr(wp, "capture_launch", replay_spoiling_world_1)
    backend.step(actions)

    assert backend.diagnostics.facts.restarted_worlds.tolist() == [False, True]
    assert np.array_equal(positions(backend)[1], start[1])
    assert not np.array_equal(positions(backend)[0], start[0])
    for field in dataclasses.fields(backend.physical_state):
        values = getattr(backend.physical_state, field.name)
        assert torch.isfinite(values.float()).all(), field.name
