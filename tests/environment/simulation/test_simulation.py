"""Tests for the physics simulation's front file.

The backends' behaviour is tested in their own test files; these tests cover
only what the front adds: loading, the action's duration, backend choice,
and forwarding.
"""

from pathlib import Path

import pytest
import torch

from centipede.environment.simulation import PhysicsSimulation, SimulationSettings
from centipede.environment.simulation.model_mapping import ModelContractError


def settings(**changes) -> SimulationSettings:
    values = {"model_path": "models/assembly_v3.xml", "backend": "cpu"} | changes
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


def test_a_timestep_that_does_not_fit_the_action_is_rejected(tmp_path):
    xml = Path("models/assembly_v3.xml").read_text(encoding="utf-8")
    uneven = tmp_path / "uneven_timestep.xml"
    uneven.write_text(
        xml.replace('timestep="0.000149253731343284"', 'timestep="0.00015"'),
        encoding="utf-8",
    )

    with pytest.raises(ModelContractError, match="timestep"):
        PhysicsSimulation(settings(model_path=str(uneven)))


def test_missing_model_file_fails_clearly():
    with pytest.raises(ValueError, match="no_such_model.xml"):
        PhysicsSimulation(settings(model_path="models/no_such_model.xml"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_backend_is_chosen_with_its_settings():
    """No physics steps: a step takes about 10 s on a pre-Volta GPU."""
    simulation = PhysicsSimulation(
        settings(backend="gpu", world_count=2, gpu_solver="cg", contacts_per_world=64)
    )
    backend = simulation._backend

    assert type(backend).__name__ == "GPUBackend"
    assert backend.gpu_data.naconmax == 2 * 64
    assert simulation.physical_state is backend.physical_state
    assert simulation.physical_state.body_height.is_cuda

    simulation.reset(seed=1)
    assert torch.all(simulation.physical_state.body_height > 0)
