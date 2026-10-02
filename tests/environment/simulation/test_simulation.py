"""Tests for the physics simulation's front file.

The backend's behaviour is tested in test_cpu_backend.py; these tests cover
only what the front adds: loading, backend choice, and forwarding.
"""

import pytest
import torch

from centipede.environment.simulation import PhysicsSimulation, SimulationSettings


def settings(**changes) -> SimulationSettings:
    values = {"model_path": "models/assembly_v2.xml", "backend": "cpu"} | changes
    return SimulationSettings.from_section(values)


def test_builds_from_settings_and_forwards_reset_and_step():
    simulation = PhysicsSimulation(settings(world_count=2))
    state = simulation.physical_state

    assert simulation.segment_count == 8
    assert simulation.world_count == 2
    assert simulation.step_duration_s == pytest.approx(0.02)

    simulation.reset(seed=1)
    simulation.step(torch.full((2, 8, 6), 0.5))
    moved = state.leg_joint_position.clone()
    simulation.reset(torch.tensor([True, False]))

    assert simulation.physical_state is state
    assert not torch.equal(state.leg_joint_position[0], moved[0])
    assert torch.equal(state.leg_joint_position[1], moved[1])


def test_missing_model_file_fails_clearly():
    with pytest.raises(ValueError, match="no_such_model.xml"):
        PhysicsSimulation(settings(model_path="models/no_such_model.xml"))


def test_gpu_backend_is_not_available_yet():
    with pytest.raises(NotImplementedError, match="Stage 7"):
        PhysicsSimulation(settings(backend="gpu"))
