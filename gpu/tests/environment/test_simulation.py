"""Focused GPU setup, stepping, and selective-reset checks; no training is run."""

from pathlib import Path
from unittest.mock import Mock

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


def make_simulation(world_count: int) -> CentipedeSimulation:
    """Construct a simulation with the shared test capacities."""
    return CentipedeSimulation(
        MODEL_PATH, world_count=world_count, nconmax=NCONMAX, njmax=NJMAX
    )


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


def test_step_places_controls_without_running_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check real GPU control placement independently of MJWarp physics kernels."""
    simulation = make_simulation(world_count=2)
    actions = np.linspace(-1.0, 1.0, 96, dtype=np.float32).reshape(2, 8, 6)
    physics_step = Mock()
    monkeypatch.setattr(simulation_module.mjw, "step", physics_step)

    simulation.data.ctrl.fill_(1.0)
    simulation.step(wp.array(actions, device=simulation.device))

    controls = simulation.data.ctrl.numpy()
    np.testing.assert_array_equal(controls[:, simulation.leg_actuator_ids], actions)
    np.testing.assert_array_equal(controls[:, simulation.spine_actuator_ids], 0.0)
    assert physics_step.call_count == simulation.frame_skip
    physics_step.assert_called_with(simulation.model, simulation.data)


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


def test_reset_restores_only_selected_worlds() -> None:
    """Clear selected physics state and perturb legs without moving root/spine."""
    simulation = make_simulation(world_count=3)
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
    before = [field.numpy().copy() for field in fields]
    selected = np.array([True, False, True])

    simulation.reset(wp.array(selected, device=simulation.device), seed=42)

    # The continuing world's primary state, inputs, and solver history survive.
    for field, previous in zip(fields, before, strict=True):
        np.testing.assert_array_equal(field.numpy()[1], previous[1])
    for field in fields[2:]:
        np.testing.assert_array_equal(field.numpy()[selected], 0)

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


def test_reset_random_sequences_are_repeatable_and_world_local() -> None:
    """Reseeding reproduces poses; resetting another world consumes no draws."""
    simulation = make_simulation(world_count=2)
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
    monkeypatch.setattr(simulation_module.mjw, "step", Mock())
    actions = wp.zeros(
        simulation.action_shape, dtype=wp.float32, device=simulation.device
    )
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
