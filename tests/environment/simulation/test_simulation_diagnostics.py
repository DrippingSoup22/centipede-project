"""Tests for the physics simulation's diagnostics category.

The category is filled by the backends; these tests check, on the CPU, that
its values are the ones MuJoCo holds after a step and after a partial reset.
The GPU backend's wiring is checked in its own test file.
"""

import mujoco
import numpy as np
import torch

from centipede.environment.simulation.cpu_backend import CPUBackend
from centipede.environment.simulation.model_mapping import ModelMapping
from centipede.environment.simulation.simulation import physics_steps_per_action


def test_the_category_holds_mujocos_values_after_a_step_and_a_reset():
    model = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    backend = CPUBackend(
        model, ModelMapping.from_model(model), 2, physics_steps_per_action(model)
    )
    backend.reset(seed=1)
    facts = backend.diagnostics.facts
    assert facts.qpos.shape == (2, model.nq)

    backend.step(torch.full((2, 8, 6), 0.7))
    _check_against_mujoco(backend)

    backend.reset(torch.tensor([True, False]))
    _check_against_mujoco(backend)
    assert facts.contact_count[0] > 0 and facts.constraint_rows.min() > 0


def _check_against_mujoco(backend):
    facts = backend.diagnostics.facts
    for world_index, data in enumerate(backend.world_data):
        assert np.allclose(facts.qpos[world_index].numpy(), data.qpos)
        assert facts.constraint_rows[world_index] == data.nefc
        assert facts.solver_iterations[world_index] == data.solver_niter.max()
    assert facts.contact_count[0] == sum(data.ncon for data in backend.world_data)
