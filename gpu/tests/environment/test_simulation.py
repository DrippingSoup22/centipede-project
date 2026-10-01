"""Focused GPU setup, stepping, reset, and extraction checks; no training is run."""

from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import warp as wp
from centipede_gpu.environment import simulation as simulation_module
from centipede_gpu.environment.simulation import CentipedeSimulation

# Both implementations load the same frozen model outside their source trees.
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "assembly.xml"

# Test headroom above the CPU reference peak (40 contacts and 256 constraint
# rows at rest); these are not tuned training capacities.
NCONMAX = 128
NJMAX = 512

# Geometry categories stored in the XML user fields.
LEG_CATEGORY = 2


def make_simulation(world_count: int) -> CentipedeSimulation:
    """Construct a simulation with the shared test capacities."""
    return CentipedeSimulation(
        MODEL_PATH, world_count=world_count, nconmax=NCONMAX, njmax=NJMAX
    )


def replace_mjwarp_physics(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Replace MJWarp integration and forward passes with call counters.

    Both run MJWarp's collision code, which the local MX330 cannot compile.
    Tests using this exercise the project's own kernels, not physics.
    """
    physics_step, forward_pass = Mock(), Mock()
    monkeypatch.setattr(simulation_module.mjw, "step", physics_step)
    monkeypatch.setattr(simulation_module.mjw, "forward", forward_pass)
    return physics_step, forward_pass


def zero_actions(simulation: CentipedeSimulation) -> wp.array:
    """Return a device action batch of zeros for the simulation's worlds."""
    return wp.zeros(simulation.action_shape, dtype=wp.float32, device=simulation.device)


@pytest.fixture(scope="module")
def simulation() -> CentipedeSimulation:
    """Share one read-only world so setup tests allocate GPU memory only once."""
    return make_simulation(world_count=1)


def test_model_dimensions_and_control_interval(simulation: CentipedeSimulation) -> None:
    """Preserve the CPU baseline's dimensions and 20 ms decision interval."""
    assert simulation.segment_ids == tuple(range(8))
    assert (simulation.host_model.nq, simulation.host_model.nv) == (69, 68)
    assert simulation.host_model.nu == 55
    assert simulation.frame_skip == 200
    assert simulation.dt == pytest.approx(0.02)
    assert simulation.action_shape == (1, 8, 6)


def test_actuator_and_joint_mappings(simulation: CentipedeSimulation) -> None:
    """Cover every motor once and preserve the six leg controls in each segment."""
    mapped_ids = np.concatenate(
        (simulation.leg_actuator_ids.ravel(), simulation.spine_actuator_ids)
    )
    np.testing.assert_array_equal(np.sort(mapped_ids), np.arange(55))
    assert simulation.leg_actuator_ids.shape == (8, 6)
    assert simulation.spine_actuator_ids.shape == (7,)

    # Spell out the contract independently of the implementation's role tuple.
    roles = (
        "left_sweep",
        "left_lift",
        "left_knee",
        "right_sweep",
        "right_lift",
        "right_knee",
    )
    model = simulation.host_model
    for segment_id in simulation.segment_ids:
        for action_index, role in enumerate(roles):
            name = f"segment_{segment_id:02d}_{role}"
            actuator_id = int(simulation.leg_actuator_ids[segment_id, action_index])
            assert model.actuator(actuator_id).name == f"{name}_motor"
            joint_id = model.joint(name).id
            assert (
                simulation.leg_qpos_ids[segment_id, action_index]
                == (model.jnt_qposadr[joint_id])
            )
            assert (
                simulation.leg_qvel_ids[segment_id, action_index]
                == (model.jnt_dofadr[joint_id])
            )


def test_state_is_allocated_on_cuda(simulation: CentipedeSimulation) -> None:
    """Require real GPU allocation, rather than a mock or silent CPU fallback."""
    wp.synchronize_device(simulation.device)
    assert simulation.device.is_cuda
    for array, shape in (
        (simulation.data.qpos, (1, 69)),
        (simulation.data.qvel, (1, 68)),
        (simulation.data.ctrl, (1, 55)),
    ):
        assert array.device == simulation.device
        assert array.shape == shape
        assert np.isfinite(array.numpy()).all()

    state = simulation.physical_state
    for array, shape, dtype in (
        (state.body_height, (1, 8), wp.float32),
        (state.body_quaternion, (1, 8, 4), wp.float32),
        (state.leg_joint_position, (1, 8, 6), wp.float32),
        (state.leg_joint_velocity, (1, 8, 6), wp.float32),
        (state.body_linear_velocity, (1, 8, 3), wp.float32),
        (state.body_angular_velocity, (1, 8, 3), wp.float32),
        (state.body_planar_position, (1, 8, 2), wp.float32),
        (state.left_foot_ground_contact, (1, 8), wp.bool),
        (state.right_foot_ground_contact, (1, 8), wp.bool),
        (state.body_ground_contact, (1, 8), wp.bool),
        (state.leg_leg_contact, (1, 8), wp.bool),
        (state.head_tip_position, (1, 3), wp.float32),
    ):
        assert array.device == simulation.device
        assert array.shape == shape
        assert array.dtype == dtype


def test_step_places_controls_and_refreshes_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check real GPU control placement independently of MJWarp physics kernels."""
    simulation = make_simulation(world_count=2)
    actions = np.linspace(-1.0, 1.0, 96, dtype=np.float32).reshape(2, 8, 6)
    physics_step, forward_pass = replace_mjwarp_physics(monkeypatch)

    simulation.data.ctrl.fill_(1.0)
    simulation.step(wp.array(actions, device=simulation.device))

    controls = simulation.data.ctrl.numpy()
    np.testing.assert_array_equal(controls[:, simulation.leg_actuator_ids], actions)
    np.testing.assert_array_equal(controls[:, simulation.spine_actuator_ids], 0.0)
    assert physics_step.call_count == simulation.frame_skip
    physics_step.assert_called_with(simulation.model, simulation.data)
    # One forward pass refreshes derived quantities at the final state.
    forward_pass.assert_called_once_with(simulation.model, simulation.data)


@pytest.mark.physics
def test_step_advances_worlds_with_separate_controls() -> None:
    """Advance two worlds, replace leg commands, and keep spine motors at zero."""
    simulation = make_simulation(world_count=2)
    actions = np.zeros(simulation.action_shape, dtype=np.float32)
    actions[1] = np.linspace(-0.1, 0.1, 48, dtype=np.float32).reshape(8, 6)
    device_actions = wp.array(actions, device=simulation.device)
    previous_time = simulation.data.time.numpy().copy()

    for _ in range(2):
        # Deliberately seed stale commands, including the disabled spine motors.
        simulation.data.ctrl.fill_(1.0)
        simulation.step(device_actions)
        np.testing.assert_allclose(
            simulation.data.time.numpy() - previous_time,
            simulation.dt,
            rtol=1e-5,
            atol=1e-7,
        )
        controls = simulation.data.ctrl.numpy()
        np.testing.assert_array_equal(controls[:, simulation.leg_actuator_ids], actions)
        np.testing.assert_array_equal(controls[:, simulation.spine_actuator_ids], 0.0)
        assert np.isfinite(simulation.data.qpos.numpy()).all()
        assert np.isfinite(simulation.data.qvel.numpy()).all()
        np.testing.assert_array_equal(simulation.data.overflow.numpy(), 0)

        # Reuse the input buffer to check that the next call reads new actions.
        previous_time = simulation.data.time.numpy().copy()
        actions *= -1
        device_actions.assign(actions)


def test_reset_restores_only_selected_worlds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear selected physics state and perturb legs without moving root/spine."""
    simulation = make_simulation(world_count=3)
    _, forward_pass = replace_mjwarp_physics(monkeypatch)
    data = simulation.data
    # Artificial old-episode state avoids needing the unsupported motion kernel.
    fields = (
        data.qpos,
        data.qvel,
        data.ctrl,
        data.time,
        data.qacc_warmstart,
        data.qfrc_applied,
        data.xfrc_applied,
        data.overflow,
    )
    for field in fields:
        field.fill_(1)
    # A continuing world can only carry tolerated solver flags: a capacity flag
    # would have stopped the run at its last check.
    data.overflow.fill_(int(simulation_module.mjw.OverflowType.ITERATIONS))
    before = [field.numpy().copy() for field in fields]
    selected = np.array([True, False, True])

    simulation.reset(wp.array(selected, device=simulation.device), seed=42)

    # The continuing world's primary state, inputs, and solver history survive.
    for field, previous in zip(fields, before, strict=True):
        np.testing.assert_array_equal(field.numpy()[1], previous[1])
    for field in fields[2:]:
        np.testing.assert_array_equal(field.numpy()[selected], 0)
    forward_pass.assert_called_once_with(simulation.model, simulation.data)

    qpos, qvel = data.qpos.numpy()[selected], data.qvel.numpy()[selected]
    leg_qpos = simulation.leg_qpos_ids.ravel()
    leg_qvel = simulation.leg_qvel_ids.ravel()
    non_leg_qpos = np.setdiff1d(np.arange(qpos.shape[1]), leg_qpos)
    non_leg_qvel = np.setdiff1d(np.arange(qvel.shape[1]), leg_qvel)
    nominal = simulation.host_model.qpos0.astype(np.float32)
    np.testing.assert_array_equal(
        qpos[:, non_leg_qpos], np.tile(nominal[non_leg_qpos], (2, 1))
    )
    np.testing.assert_array_equal(qvel[:, non_leg_qvel], 0)
    offsets = qpos[:, leg_qpos] - nominal[leg_qpos]
    assert np.all(np.abs(offsets) <= np.deg2rad(2.0) + 1e-7)
    assert np.any(offsets != 0)
    assert np.isfinite(qvel).all() and np.any(qvel[:, leg_qvel] != 0)
    # Broad scale check, not a claim that a small sample proves normality.
    assert 0.02 < np.std(qvel[:, leg_qvel]) < 0.08
    assert not np.array_equal(qpos[0], qpos[1])


def test_reset_random_sequences_are_repeatable_and_world_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reseeding reproduces poses; resetting another world consumes no draws."""
    simulation = make_simulation(world_count=2)
    replace_mjwarp_physics(monkeypatch)
    # Exercise the initial entropy-based sequence, before any explicit seed.
    simulation.reset()
    unseeded = simulation.data.qpos.numpy()
    simulation.reset()
    assert np.isfinite(unseeded).all()
    assert not np.array_equal(unseeded, simulation.data.qpos.numpy())
    simulation.reset(seed=2**32 - 1)
    first = (simulation.data.qpos.numpy(), simulation.data.qvel.numpy())
    simulation.reset()
    second = (simulation.data.qpos.numpy(), simulation.data.qvel.numpy())
    assert not np.array_equal(first[0], second[0])

    simulation.reset(seed=2**32 - 1)
    # An empty selection must change neither physical state nor random sequences.
    simulation.reset(wp.array([False, False], dtype=wp.bool, device=simulation.device))
    np.testing.assert_array_equal(simulation.data.qpos.numpy(), first[0])
    np.testing.assert_array_equal(simulation.data.qvel.numpy(), first[1])

    for selected_world in (0, 1):
        mask = wp.array(
            np.arange(2) == selected_world, dtype=wp.bool, device=simulation.device
        )
        simulation.reset(mask)
    np.testing.assert_array_equal(simulation.data.qpos.numpy(), second[0])
    np.testing.assert_array_equal(simulation.data.qvel.numpy(), second[1])

    # Reseeding just one world must also leave the other world's stream intact.
    simulation.reset(seed=2**32 - 1)
    simulation.reset(
        wp.array([True, False], dtype=wp.bool, device=simulation.device), seed=7
    )
    simulation.reset(wp.array([False, True], dtype=wp.bool, device=simulation.device))
    np.testing.assert_array_equal(simulation.data.qpos.numpy()[1], second[0][1])
    np.testing.assert_array_equal(simulation.data.qvel.numpy()[1], second[1][1])


def test_step_rejects_invalid_physics_but_not_solver_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop on overflows and non-finite state; accept unconverged solver flags."""
    simulation = make_simulation(world_count=2)
    replace_mjwarp_physics(monkeypatch)
    actions = zero_actions(simulation)
    overflow = simulation_module.mjw.OverflowType

    # MJWarp keeps these flags until reset, so they persist across the mock step.
    solver_limits = overflow.ITERATIONS | overflow.LS_ITERATIONS
    simulation.data.overflow.assign(np.array([0, solver_limits], dtype=np.int32))
    simulation.step(actions)

    capacity = overflow.NEFC | overflow.CCD
    simulation.data.overflow.assign(np.array([0, capacity], dtype=np.int32))
    with pytest.raises(RuntimeError, match="world 1: NEFC, CCD"):
        simulation.step(actions)

    simulation.data.overflow.zero_()
    qvel = simulation.data.qvel.numpy()
    qvel[0, 0] = np.nan
    simulation.data.qvel.assign(qvel)
    with pytest.raises(
        RuntimeError, match=r"Non-finite physical state in worlds \[0\]"
    ):
        simulation.step(actions)


def scrambled_host_state(
    simulation: CentipedeSimulation, rng: np.random.Generator
) -> mujoco.MjData:
    """Return CPU MuJoCo data at a random non-symmetric pose with velocities.

    A random root orientation and bent joints keep every rotation matrix far
    from the identity, so a transposed or misindexed frame cannot pass.
    """
    model = simulation.host_model
    host_data = mujoco.MjData(model)
    for joint_id in range(model.njnt):
        address = model.jnt_qposadr[joint_id]
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            host_data.qpos[address : address + 3] += rng.normal(0.0, 0.01, 3)
            orientation = rng.normal(size=4)
            host_data.qpos[address + 3 : address + 7] = orientation / np.linalg.norm(
                orientation
            )
        else:
            host_data.qpos[address] += rng.uniform(-0.2, 0.2)
    host_data.qvel[:] = rng.normal(0.0, 0.5, model.nv)
    mujoco.mj_forward(model, host_data)
    return host_data


def expected_physical_state(
    simulation: CentipedeSimulation, host_data: mujoco.MjData
) -> dict[str, np.ndarray]:
    """Compute the physical fields for one world with CPU MuJoCo's own calls."""
    model = simulation.host_model
    centers = host_data.site_xpos[simulation.segment_center_site_ids]
    local_velocities = np.zeros((len(simulation.segment_ids), 6))
    for row, site_id in enumerate(simulation.segment_center_site_ids):
        mujoco.mj_objectVelocity(
            model,
            host_data,
            mujoco.mjtObj.mjOBJ_SITE,
            int(site_id),
            local_velocities[row],
            1,
        )
    return {
        "body_height": centers[:, 2],
        "body_quaternion": host_data.xquat[simulation.segment_body_ids],
        "leg_joint_position": host_data.qpos[simulation.leg_qpos_ids],
        "leg_joint_velocity": host_data.qvel[simulation.leg_qvel_ids],
        "body_linear_velocity": local_velocities[:, 3:],
        "body_angular_velocity": local_velocities[:, :3],
        "body_planar_position": centers[:, :2],
        "head_tip_position": host_data.site_xpos[simulation.head_tip_site_id],
    }


def test_extraction_matches_cpu_mujoco_derived_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract CPU MuJoCo's derived quantities with the project kernel.

    CPU MuJoCo computes positions, orientations, and velocities for two
    different worlds; they are copied into MJWarp's data, and the kernel's
    output must match mj_objectVelocity and the CPU fields. This checks the
    kernel's indexing and velocity formula, not MJWarp's forward pass.
    """
    simulation = make_simulation(world_count=2)
    replace_mjwarp_physics(monkeypatch)
    rng = np.random.default_rng(3)
    host_datas = [scrambled_host_state(simulation, rng) for _ in range(2)]

    data = simulation.data
    for field in ("qpos", "qvel", "xquat", "site_xpos", "subtree_com", "cvel"):
        getattr(data, field).assign(
            np.stack([getattr(host, field) for host in host_datas]).astype(np.float32)
        )
    data.site_xmat.assign(
        np.stack([host.site_xmat.reshape(-1, 3, 3) for host in host_datas]).astype(
            np.float32
        )
    )
    data.nacon.zero_()

    simulation.step(zero_actions(simulation))

    state = simulation.physical_state
    for world_id, host_data in enumerate(host_datas):
        for name, expected in expected_physical_state(simulation, host_data).items():
            actual = getattr(state, name).numpy()[world_id]
            if "velocity" in name and "joint" not in name:
                # Computed in float32 from rounded inputs of order 1 (qvel is
                # drawn with a 0.5 standard deviation). float32 spacing at 1 is
                # 1.2e-7, so each term carries about 1e-7 absolute error even
                # when the result is near zero; 1e-6 allows a few such terms.
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            else:
                # Copied fields must equal the float32 rounding of CPU values.
                np.testing.assert_array_equal(actual, expected.astype(np.float32))
    for flags in (
        state.left_foot_ground_contact,
        state.right_foot_ground_contact,
        state.body_ground_contact,
        state.leg_leg_contact,
    ):
        assert not flags.numpy().any()


def test_contact_classification_follows_cpu_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify a hand-written contact pool covering every CPU rule."""
    simulation = make_simulation(world_count=2)
    replace_mjwarp_physics(monkeypatch)
    geom_user = simulation.host_model.geom_user
    floor = simulation.floor_geom_id
    body = simulation.segment_body_geom_ids
    feet = simulation.foot_geom_ids

    def leg_part(segment_id: int) -> int:
        """Return a non-foot leg geometry owned by one segment."""
        owned = (geom_user[:, 0] == segment_id) & (geom_user[:, 1] == LEG_CATEGORY)
        return int(np.flatnonzero(owned)[0])

    # (world, first geometry, second geometry); the floor appears on both sides.
    contacts = [
        (0, floor, body[2]),  # body on floor: body 2
        (0, feet[3, 0], floor),  # left foot on floor: segment 3
        (1, floor, feet[6, 1]),  # right foot on floor: segment 6
        (1, floor, feet[6, 1]),  # a second point of the same contact
        (0, floor, leg_part(4)),  # non-foot leg part on floor: no flag
        (1, leg_part(1), feet[2, 0]),  # leg with foot: segments 1 and 2
        (1, leg_part(5), feet[5, 1]),  # both legs of segment 5
        (0, leg_part(7), body[6]),  # leg with body: no flag
    ]
    # A stale entry past nacon must be ignored.
    stale_contact = (1, floor, body[7])

    pool_geoms = np.zeros((simulation.data.naconmax, 2), dtype=np.int32)
    pool_worlds = np.zeros(simulation.data.naconmax, dtype=np.int32)
    for slot, (world_id, first, second) in enumerate([*contacts, stale_contact]):
        pool_geoms[slot] = (first, second)
        pool_worlds[slot] = world_id
    simulation.data.contact.geom.assign(pool_geoms)
    simulation.data.contact.worldid.assign(pool_worlds)
    simulation.data.nacon.assign(np.array([len(contacts)], dtype=np.int32))

    simulation.step(zero_actions(simulation))

    expected = {
        "left_foot_ground_contact": [(0, 3)],
        "right_foot_ground_contact": [(1, 6)],
        "body_ground_contact": [(0, 2)],
        "leg_leg_contact": [(1, 1), (1, 2), (1, 5)],
    }
    state = simulation.physical_state
    for name, marked in expected.items():
        flags = np.zeros((2, 8), dtype=bool)
        for world_id, segment_id in marked:
            flags[world_id, segment_id] = True
        np.testing.assert_array_equal(getattr(state, name).numpy(), flags)

    # Flags describe only the latest refresh: an empty pool clears them.
    simulation.data.nacon.zero_()
    simulation.step(zero_actions(simulation))
    for name in expected:
        assert not getattr(state, name).numpy().any()


@pytest.mark.physics
def test_forward_refresh_matches_cpu_mujoco_at_same_state() -> None:
    """Compare MJWarp's refreshed state with CPU MuJoCo at one settled pose.

    CPU MuJoCo settles the body for 0.1 s so that feet touch the floor; the same
    positions and velocities are written to the GPU, and an empty reset mask
    refreshes every world without resetting any. Tolerances allow float32
    kinematics; contact pairs are compared as sets because MJWarp generates at
    most one contact per capsule-mesh pair.
    """
    simulation = make_simulation(world_count=1)
    model = simulation.host_model
    host_data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, host_data)
    mujoco.mj_forward(model, host_data)

    simulation.data.qpos.assign(host_data.qpos[None].astype(np.float32))
    simulation.data.qvel.assign(host_data.qvel[None].astype(np.float32))
    simulation.reset(wp.array([False], dtype=wp.bool, device=simulation.device))

    state = simulation.physical_state
    tolerances = {
        "body_height": 1e-7,
        "body_quaternion": 1e-6,
        "leg_joint_position": 1e-7,
        "leg_joint_velocity": 1e-6,
        "body_linear_velocity": 1e-6,
        "body_angular_velocity": 1e-5,
        "body_planar_position": 1e-7,
        "head_tip_position": 1e-7,
    }
    for name, expected in expected_physical_state(simulation, host_data).items():
        np.testing.assert_allclose(
            getattr(state, name).numpy()[0], expected, rtol=0, atol=tolerances[name]
        )

    contact_count = int(simulation.data.nacon.numpy()[0])
    gpu_pairs = {
        tuple(sorted(pair))
        for pair in simulation.data.contact.geom.numpy()[:contact_count].tolist()
    }
    cpu_pairs = {tuple(sorted(contact.geom.tolist())) for contact in host_data.contact}
    assert gpu_pairs == cpu_pairs
    assert state.left_foot_ground_contact.numpy().any()
