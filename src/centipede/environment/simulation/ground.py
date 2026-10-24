"""The ground's height anywhere, and the terrain's start areas, read once from
the model as tensors on the simulation's device."""

import mujoco
import numpy as np
import torch

from centipede.environment.simulation.model_mapping import FLOOR_CATEGORY

# A terrain file's start areas are box sites named start_<ground>; the joiner
# prefixes them (simulation.model_spec).
START_AREA_PREFIX = "terrain_start_"


class Ground:
    """The terrain's height field as a grid of heights, or a flat floor.

    ``heights`` and ``highest_near`` read the grid's points (a height field
    is drawn through them; between them MuJoCo cuts each cell into two
    triangles, which the bilinear blend here stands in for). Points beyond
    the grid take its edge. Without a height field the ground is the plane
    at height 0. ``start_areas`` maps each ground's name to its area's
    centre and half sizes, ``(2,)`` m each.
    """

    def __init__(self, model: mujoco.MjModel, device: torch.device | str) -> None:
        shapes = [
            shape
            for shape in range(model.ngeom)
            if model.geom_type[shape] == mujoco.mjtGeom.mjGEOM_HFIELD
            and model.geom_user[shape, 1] == FLOOR_CATEGORY
        ]
        self.flat = not shapes
        if not self.flat:
            shape = shapes[0]
            field = model.geom_dataid[shape]
            rows, cols = model.hfield_nrow[field], model.hfield_ncol[field]
            size = model.hfield_size[field]
            start = model.hfield_adr[field]
            grid = model.hfield_data[start : start + rows * cols].reshape(rows, cols)
            self.grid = torch.tensor(
                model.geom_pos[shape, 2] + grid * size[2], dtype=torch.float32
            ).to(device)
            self.corner = torch.tensor(
                model.geom_pos[shape, :2] - size[:2], dtype=torch.float32
            ).to(device)
            self.cell = torch.tensor(
                2 * size[:2] / (np.array([cols, rows]) - 1), dtype=torch.float32
            ).to(device)
        self.start_areas = {
            model.site(site).name.removeprefix(START_AREA_PREFIX): (
                torch.tensor(model.site_pos[site, :2], dtype=torch.float32).to(device),
                torch.tensor(model.site_size[site, :2], dtype=torch.float32).to(device),
            )
            for site in range(model.nsite)
            if model.site(site).name.startswith(START_AREA_PREFIX)
        }

    def heights(self, points: torch.Tensor) -> torch.Tensor:
        """The ground's height at ``points`` ``(..., 2)`` m (x, y): ``(...)``."""
        if self.flat:
            return torch.zeros(points.shape[:-1], device=points.device)
        place = (points - self.corner) / self.cell
        rows, cols = self.grid.shape
        place = torch.stack(
            (
                place[..., 0].clamp(0, cols - 1.001),
                place[..., 1].clamp(0, rows - 1.001),
            ),
            dim=-1,
        )
        first = place.floor().long()
        share = place - first
        col, row = first[..., 0], first[..., 1]
        x, y = share[..., 0], share[..., 1]
        grid = self.grid
        return (
            (1 - x) * (1 - y) * grid[row, col]
            + x * (1 - y) * grid[row, col + 1]
            + (1 - x) * y * grid[row + 1, col]
            + x * y * grid[row + 1, col + 1]
        )

    def highest_near(self, points: torch.Tensor, radius_m: float) -> torch.Tensor:
        """The highest grid point within ``radius_m`` of each of ``points``
        ``(W, 2)`` m along x and y (a square around it): ``(W,)``."""
        if self.flat:
            return torch.zeros(points.shape[0], device=points.device)
        rows, cols = self.grid.shape
        reach = torch.ceil(radius_m / self.cell).long()
        steps_x = torch.arange(-reach[0], reach[0] + 1, device=points.device)
        steps_y = torch.arange(-reach[1], reach[1] + 1, device=points.device)
        centre = torch.round((points - self.corner) / self.cell).long()
        col = (centre[:, 0, None, None] + steps_x[None, None, :]).clamp(0, cols - 1)
        row = (centre[:, 1, None, None] + steps_y[None, :, None]).clamp(0, rows - 1)
        return self.grid[row, col].flatten(start_dim=1).max(dim=1).values
