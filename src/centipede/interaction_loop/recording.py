"""Recording the poses of a window, for replay in MujocoReplay.

``WindowRecorder`` belongs to the loop's diagnostics. The experiment arms it
before a window it wants recorded; the loop's diagnostics start it when the
window starts and feed it after every step; the experiment takes the result
between cycles and writes the file. While a window is recorded, every step
copies the simulation's ``qpos`` of every world into a device buffer and adds
the step's rewards to each world's score: a few small tensor operations that
never wait for the GPU. Only the worlds chosen by rank are copied to the CPU,
after the window, outside the timed parts. The file format and the rank rule
are MujocoReplay's (its docs/recording-format.md).
"""

from dataclasses import dataclass

import numpy as np
import torch
from mujoco_replay.selection import level_of_ranks, selected_ranks

from centipede.environment.diagnostics import EnvironmentDiagnostics

SELECTIONS = ("ranked", "first", "all")


@dataclass(frozen=True)
class RecordedWindow:
    """The chosen worlds' frames of one window, as NumPy arrays on the CPU.

    ``K`` worlds in rank order (best first) over ``T`` frames: ``qpos (T, K,
    nq)``, ``score (K,)``, ``world_ids (K,)``, ``level (K,)``, ``episode_start
    (T, K)``, and ``target (T, K, 2)``.
    """

    qpos: np.ndarray
    score: np.ndarray
    world_ids: np.ndarray
    level: np.ndarray
    episode_start: np.ndarray
    target: np.ndarray


class WindowRecorder:
    """Captures every world's poses over one window, on the device."""

    def __init__(self, diagnostics: EnvironmentDiagnostics, world_count: int) -> None:
        """Prepare to record from the environment's categories; allocate nothing."""
        self._simulation = diagnostics.simulation
        self._step_facts = diagnostics.step
        self._episode = diagnostics.episode
        self._world_count = world_count
        self._device = diagnostics.simulation.qpos.device
        self._armed_frames = 0
        self._recording = False
        self._frame = 0
        self._capacity = 0
        self._qpos = torch.empty(0)
        self._episode_start = torch.empty(0)
        self._target = torch.empty(0)
        self._score = torch.empty(0)

    @property
    def recording(self) -> bool:
        """Whether the current window is being recorded."""
        return self._recording

    def arm(self, frames: int) -> None:
        """Record the next window, with room for ``frames`` steps."""
        self._armed_frames = frames

    def start_window(self) -> None:
        """Begin recording if armed; otherwise this window is skipped."""
        if not self._armed_frames:
            self._recording = False
            return
        if self._armed_frames > self._capacity:
            self._allocate(self._armed_frames)
        self._armed_frames = 0
        self._frame = 0
        self._score.zero_()
        self._recording = True

    def step_taken(self) -> None:
        """Keep the step just taken, if this window is recorded."""
        if not self._recording:
            return
        frame = self._frame
        self._qpos[frame].copy_(self._simulation.qpos)
        self._episode_start[frame].copy_(self._episode.episode_ended)
        self._target[frame].copy_(self._step_facts.target_position)
        self._score += self._step_facts.reward_parts.sum(dim=(1, 2))
        self._frame += 1

    def take(self, levels: int, per_level: int, selection: str) -> RecordedWindow:
        """The recorded window's chosen worlds, copied to the CPU; ends recording.

        ``"ranked"`` keeps the worlds at MujocoReplay's ``selected_ranks`` of
        the summed reward; ``"first"`` keeps the first worlds, as many as the
        ranked choice would, for continuity across windows; ``"all"`` keeps
        every world. Worlds come out in rank order, best first.
        """
        self._recording = False
        frames = self._frame
        order = torch.argsort(self._score, descending=True, stable=True)
        if selection == "all":
            chosen = order
            ranks = np.arange(self._world_count)
        elif selection == "ranked":
            ranks, _ = selected_ranks(self._world_count, levels, per_level)
            chosen = order[torch.as_tensor(ranks, device=self._device)]
        else:
            kept = min(self._world_count, levels * per_level)
            rank_of_world = torch.empty_like(order)
            rank_of_world[order] = torch.arange(self._world_count, device=self._device)
            chosen = torch.argsort(rank_of_world[:kept])
            ranks = rank_of_world[chosen].cpu().numpy()
        level = level_of_ranks(ranks, self._world_count, levels)
        return RecordedWindow(
            qpos=self._qpos[:frames, chosen].cpu().numpy(),
            score=self._score[chosen].cpu().numpy(),
            world_ids=chosen.cpu().numpy(),
            level=level,
            episode_start=self._episode_start[:frames, chosen].cpu().numpy(),
            target=self._target[:frames, chosen].cpu().numpy(),
        )

    def _allocate(self, frames: int) -> None:
        """Buffers for ``frames`` steps of every world, on the category's device."""
        world_count, position_count = self._simulation.qpos.shape
        device = self._device
        self._qpos = torch.zeros((frames, world_count, position_count), device=device)
        self._episode_start = torch.zeros(
            (frames, world_count), dtype=torch.bool, device=device
        )
        self._target = torch.zeros((frames, world_count, 2), device=device)
        self._score = torch.zeros(world_count, device=device)
        self._capacity = frames
