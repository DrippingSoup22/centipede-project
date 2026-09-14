"""Tests for the internal MuJoCo simulation boundary."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from centipede.simulation import (
    LEG_ACTUATOR_ROLES,
    LEG_POSITION_NOISE_LIMIT,
    CentipedeSimulation,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "assembly.xml"


@pytest.fixture
def simulation() -> Iterator[CentipedeSimulation]:
    """Provide a fresh simulation and always release its rendering resources."""
    instance = CentipedeSimulation(model_path=MODEL_PATH)
    try:
        yield instance
    finally:
        instance.close()


def test_relative_model_path_resolves_from_working_directory(monkeypatch) -> None:
    monkeypatch.chdir(MODEL_PATH.parents[1])
    instance = CentipedeSimulation(model_path=Path("models/assembly.xml"))
    try:
        assert Path(instance.fullpath) == MODEL_PATH
    finally:
        instance.close()


def test_model_loads_with_expected_dimensions(
    simulation: CentipedeSimulation,
) -> None:
    assert simulation.model.nq == 69
    assert simulation.model.nv == 68
    assert simulation.model.nu == 55
    assert simulation.frame_skip == 200
    assert simulation.dt == pytest.approx(0.02)


def test_seeded_reset_is_finite(simulation: CentipedeSimulation) -> None:
    state, info = simulation.reset(seed=0)

    assert state.shape == (137,)
    assert np.isfinite(state).all()
    assert info == {}


def test_reset_randomizes_only_leg_state_reproducibly(
    simulation: CentipedeSimulation,
) -> None:
    first, _ = simulation.reset(seed=17)
    advanced, _ = simulation.reset()
    repeated, _ = simulation.reset(seed=17)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, advanced)

    qpos_delta = first[: simulation.model.nq] - simulation.init_qpos
    qvel_delta = first[simulation.model.nq :] - simulation.init_qvel
    qpos_mask = np.zeros(simulation.model.nq, dtype=np.bool_)
    qvel_mask = np.zeros(simulation.model.nv, dtype=np.bool_)
    qpos_mask[simulation.leg_qpos_ids] = True
    qvel_mask[simulation.leg_qvel_ids] = True

    np.testing.assert_array_equal(qpos_delta[~qpos_mask], 0.0)
    np.testing.assert_array_equal(qvel_delta[~qvel_mask], 0.0)
    assert np.abs(qpos_delta[qpos_mask]).max() <= LEG_POSITION_NOISE_LIMIT
    assert np.any(qpos_delta[qpos_mask] != 0.0)
    assert np.any(qvel_delta[qvel_mask] != 0.0)


def test_actuator_mapping_is_complete_and_unique(
    simulation: CentipedeSimulation,
) -> None:
    assert simulation.leg_actuator_ids.shape == (8, 6)
    assert simulation.spine_actuator_ids.shape == (7,)

    mapped_ids = np.concatenate(
        (simulation.leg_actuator_ids.ravel(), simulation.spine_actuator_ids)
    )
    assert len(np.unique(mapped_ids)) == simulation.model.nu
    assert set(mapped_ids.tolist()) == set(range(simulation.model.nu))


def test_physical_mapping_has_one_entry_per_segment_and_leg_joint(
    simulation: CentipedeSimulation,
) -> None:
    assert simulation.leg_qpos_ids.shape == (8, 6)
    assert simulation.leg_qvel_ids.shape == (8, 6)
    assert simulation.segment_body_ids.shape == (8,)
    assert simulation.segment_center_site_ids.shape == (8,)
    assert simulation.segment_body_geom_ids.shape == (8,)
    assert simulation.foot_geom_ids.shape == (8, 2)
    assert len(np.unique(simulation.leg_qpos_ids)) == 48
    assert len(np.unique(simulation.leg_qvel_ids)) == 48


def test_leg_actuator_order_matches_the_contract(
    simulation: CentipedeSimulation,
) -> None:
    for segment_id in simulation.segment_ids:
        for action_index, (side, role) in enumerate(LEG_ACTUATOR_ROLES):
            actuator_id = int(
                simulation.leg_actuator_ids[segment_id, action_index]
            )
            actual_name = simulation.model.actuator(actuator_id).name
            expected_name = f"segment_{segment_id:02d}_{side}_{role}_motor"
            assert actual_name == expected_name


def test_control_assembly_maps_legs_and_disables_spine(
    simulation: CentipedeSimulation,
) -> None:
    leg_controls = np.linspace(
        -1.0,
        1.0,
        simulation.leg_actuator_ids.size,
        dtype=np.float32,
    )
    control = simulation._assemble_control(leg_controls)

    assert control.shape == (simulation.model.nu,)
    np.testing.assert_array_equal(
        control[simulation.leg_actuator_ids.ravel()], leg_controls
    )
    np.testing.assert_array_equal(control[simulation.spine_actuator_ids], 0.0)


def test_step_advances_one_control_interval(
    simulation: CentipedeSimulation,
) -> None:
    simulation.reset(seed=0)
    time_before = simulation.data.time
    action = np.zeros(simulation.leg_actuator_ids.size, dtype=np.float32)

    observation, reward, terminated, truncated, info = simulation.step(action)

    assert simulation.action_space.shape == (48,)
    assert simulation.data.time == pytest.approx(time_before + simulation.dt)
    assert observation.shape == (137,)
    assert np.isfinite(observation).all()
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info == {}
    np.testing.assert_array_equal(
        simulation.data.ctrl[simulation.spine_actuator_ids], 0.0
    )


def test_snapshot_copies_complete_segment_state(
    simulation: CentipedeSimulation,
) -> None:
    simulation.reset(seed=3)
    snapshot = simulation.snapshot()
    expected_shapes = {
        "body_height": (8,),
        "body_quaternion": (8, 4),
        "leg_joint_position": (8, 6),
        "leg_joint_velocity": (8, 6),
        "body_linear_velocity": (8, 3),
        "body_angular_velocity": (8, 3),
        "body_planar_position": (8, 2),
        "left_foot_ground_contact": (8,),
        "right_foot_ground_contact": (8,),
        "body_ground_contact": (8,),
        "leg_leg_contact": (8,),
        "head_tip_position": (3,),
    }

    for field, shape in expected_shapes.items():
        values = getattr(snapshot, field)
        assert values.shape == shape
        assert np.isfinite(values).all()
        assert not values.flags.writeable

    np.testing.assert_array_equal(
        snapshot.leg_joint_position,
        simulation.data.qpos[simulation.leg_qpos_ids],
    )
    np.testing.assert_array_equal(
        snapshot.leg_joint_velocity,
        simulation.data.qvel[simulation.leg_qvel_ids],
    )
    np.testing.assert_allclose(
        np.linalg.norm(snapshot.body_quaternion, axis=1), 1.0
    )


def test_contact_classification_uses_geometry_metadata(
    simulation: CentipedeSimulation,
) -> None:
    geom_pairs = np.array(
        [
            [simulation.floor_geom_id, simulation.foot_geom_ids[0, 0]],
            [simulation.foot_geom_ids[1, 1], simulation.floor_geom_id],
            [simulation.floor_geom_id, simulation.segment_body_geom_ids[2]],
            [
                simulation.model.geom("segment_03_left_upper_geom").id,
                simulation.model.geom("segment_04_right_lower_geom").id,
            ],
            [
                simulation.model.geom("segment_05_left_upper_geom").id,
                simulation.segment_body_geom_ids[6],
            ],
        ],
        dtype=np.int32,
    )

    actual = simulation._classify_contacts(geom_pairs)
    expected = np.zeros((4, 8), dtype=np.bool_)
    expected[0, 0] = True
    expected[1, 1] = True
    expected[2, 2] = True
    expected[3, [3, 4]] = True

    for actual_flags, expected_flags in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_flags, expected_flags)


def test_snapshot_reports_contacts_from_the_final_state(
    simulation: CentipedeSimulation,
) -> None:
    simulation.reset(seed=0)
    simulation.set_state(simulation.init_qpos, simulation.init_qvel)
    simulation.step(np.zeros(simulation.action_space.shape, dtype=np.float32))
    snapshot = simulation.snapshot()

    expected_foot_contacts = np.zeros(8, dtype=np.bool_)
    expected_foot_contacts[[1, 7]] = True
    np.testing.assert_array_equal(
        snapshot.left_foot_ground_contact, expected_foot_contacts
    )
    np.testing.assert_array_equal(
        snapshot.right_foot_ground_contact, expected_foot_contacts
    )


def test_numerical_validation_rejects_invalid_state_and_warnings(
    simulation: CentipedeSimulation,
) -> None:
    simulation.reset(seed=0)
    simulation.data.qpos[0] = np.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        simulation._validate_physics()

    simulation.data.qpos[0] = simulation.init_qpos[0]
    simulation.data.warning[0].number = 1

    with pytest.raises(RuntimeError, match="warnings at indices"):
        simulation._validate_physics()
