"""The physics simulation's diagnostics: positions, and how hard the physics works.

One category, ``SimulationFacts``, allocated once and refreshed in place. Its
``qpos`` is every world's position vector, kept as it is so that the
interaction loop's recorder can copy it for replays; the three health values
show how close a run comes to the memory the GPU backend reserves and how hard
the solver works. All four describe the moment after the ``forward`` call that
follows a step's last physics step, or a reset, so both backends report the
same thing. The CPU backend fills the category with ``fill_from_cpu`` at the
end of ``step`` and ``reset``; the GPU backend builds it from views of MuJoCo
Warp's arrays, which refresh themselves. The values are listed in
docs/diagnostics.md.
"""

from dataclasses import dataclass

import mujoco
import torch

from centipede.diagnostics_category import measure


@dataclass(frozen=True)
class SimulationFacts:
    """What the physics simulation knows after each step, in every world."""

    qpos: torch.Tensor = measure(
        "Every world's position coordinates in MuJoCo's qpos layout, (W, nq)",
        summary="recorded",
    )
    contact_count: torch.Tensor = measure(
        "Contacts in all worlds together after the step, (1,); the GPU reserves "
        "contacts_per_world times the number of worlds",
        summary="maximum",
    )
    constraint_rows: torch.Tensor = measure(
        "Constraint rows in each world after the step, (W,); the GPU reserves "
        "constraints_per_world",
        summary="maximum",
    )
    solver_iterations: torch.Tensor = measure(
        "Solver iterations each world needed on the last physics call, (W,)",
        summary="maximum",
    )


class SimulationDiagnostics:
    """Holds the simulation's category and fills it on the CPU."""

    def __init__(self, facts: SimulationFacts) -> None:
        self.facts = facts
        # NumPy views sharing the tensors' memory, for the CPU fill.
        if facts.qpos.device.type == "cpu":
            self._qpos = facts.qpos.numpy()
            self._contact_count = facts.contact_count.numpy()
            self._constraint_rows = facts.constraint_rows.numpy()
            self._solver_iterations = facts.solver_iterations.numpy()

    @classmethod
    def allocate(
        cls, world_count: int, position_count: int, device: str
    ) -> "SimulationDiagnostics":
        """A category of zeros, to be filled in place."""
        return cls(
            SimulationFacts(
                qpos=torch.zeros((world_count, position_count), device=device),
                contact_count=torch.zeros(1, dtype=torch.int32, device=device),
                constraint_rows=torch.zeros(
                    world_count, dtype=torch.int32, device=device
                ),
                solver_iterations=torch.zeros(
                    world_count, dtype=torch.int32, device=device
                ),
            )
        )

    def fill_from_cpu(self, world_data: list[mujoco.MjData]) -> None:
        """Copy every world's values from its MuJoCo data, after ``mj_forward``.

        ``solver_niter`` is per constraint island; with islands off only the
        first entry is used, so the largest entry is right either way.
        """
        contacts = 0
        for world_index, data in enumerate(world_data):
            self._qpos[world_index] = data.qpos
            self._constraint_rows[world_index] = data.nefc
            self._solver_iterations[world_index] = data.solver_niter.max()
            contacts += data.ncon
        self._contact_count[0] = contacts
