import mujoco
import torch

from centipede.environment.simulation.constants import PHYSICS_STEPS_PER_ACTION
from centipede.environment.simulation.cpu_backend import CPUBackend
from centipede.environment.simulation.model_mapping import ModelMapping
from centipede.environment.simulation.settings import SimulationSettings


class PhysicsSimulation:
    """The physics simulation's front: the only part the environment uses.

    It loads the model named in the settings, checks it against the segment
    contract, and runs every world on the chosen backend. ``physical_state``
    is refreshed in place by every ``reset`` and ``step``, and so is
    ``diagnostics.facts``, the simulation's diagnostics category.
    """

    def __init__(self, settings: SimulationSettings) -> None:
        """Load and check the model, then create the chosen backend."""
        model = mujoco.MjModel.from_xml_path(str(settings.model_path))
        mapping = ModelMapping.from_model(model)

        if settings.backend == "cpu":
            self._backend = CPUBackend(model, mapping, settings.world_count)
        else:
            # Imported here so that a CPU-only installation, without the
            # optional GPU packages, can still use the CPU backend.
            from centipede.environment.simulation.gpu_backend import GPUBackend

            self._backend = GPUBackend(
                model,
                mapping,
                settings.world_count,
                settings.gpu_solver,
                settings.contacts_per_world,
                settings.constraints_per_world,
            )

        self.segment_count = mapping.segment_count
        self.world_count = settings.world_count
        self.step_duration_s = PHYSICS_STEPS_PER_ACTION * model.opt.timestep
        self.physical_state = self._backend.physical_state
        self.diagnostics = self._backend.diagnostics

    def reset(
        self, world_mask: torch.Tensor | None = None, seed: int | None = None
    ) -> None:
        """Reset the selected worlds (all if no mask); see the backends' ``reset``."""
        self._backend.reset(world_mask, seed)

    def step(self, leg_actions: torch.Tensor) -> None:
        """Hold the (W, N, 6) leg actions for one step in every world."""
        self._backend.step(leg_actions)
