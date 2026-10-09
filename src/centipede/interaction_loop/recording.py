"""Recording the poses of a stretch of steps, for replay in MujocoReplay.

``WindowRecorder`` belongs to the loop's diagnostics. The experiment arms it
for a number of steps before the window where they begin; the loop's
diagnostics start it when that window starts and feed it after every step, and
a recording that needs more steps carries on through the following windows
until it is ``complete``. The experiment takes the result between cycles and
writes the file: in training a whole episode length, which spans several
windows. While a recording runs, every step
copies the simulation's ``qpos`` of every world into a device buffer and adds
the step's rewards to each world's score: a few small tensor operations that
never wait for the GPU. Only the worlds chosen by rank are copied to the CPU,
after the recording, outside the timed parts. The file format and the rank rule
are MujocoReplay's (its docs/recording-format.md).
"""

from dataclasses import dataclass

import numpy as np
import torch
from mujoco_replay.selection import level_of_ranks, selected_ranks

from centipede.environment.diagnostics import EnvironmentDiagnostics

SELECTIONS = ("ranked", "first")


@dataclass(frozen=True)
class RecordedWindow:
    """The chosen worlds' frames of one recording, as NumPy arrays on the CPU.

    ``K`` worlds in rank order (best first) over ``T`` frames: ``qpos (T, K,
    nq)``, ``score (K,)``, ``world_ids (K,)``, ``level (K,)``, ``episode_start
    (T, K)``, ``target (T, K, 2)``, and ``range_radius (T, K)``, the radius of
    each world's range circle around its target (infinite without one).
    ``rank (K,)`` is each world's rank
    among all ``ranked_worlds`` worlds, 1 for the best, and ``level_count`` is
    the number of levels the worlds were split into.
    """

    qpos: np.ndarray
    score: np.ndarray
    world_ids: np.ndarray
    level: np.ndarray
    episode_start: np.ndarray
    target: np.ndarray
    range_radius: np.ndarray
    rank: np.ndarray
    ranked_worlds: int
    level_count: int


class WindowRecorder:
    """Captures every world's poses over the armed steps, on the device."""

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
        self._frames = 0
        self._capacity = 0
        self._qpos = torch.empty(0)
        self._episode_start = torch.empty(0)
        self._target = torch.empty(0)
        self._range_radius = torch.empty(0)
        self._score = torch.empty(0)

    @property
    def recording(self) -> bool:
        """Whether a recording is running: started, and not yet taken."""
        return self._recording

    @property
    def complete(self) -> bool:
        """Whether the running recording holds all the steps it was armed for."""
        return self._recording and self._frame == self._frames

    def arm(self, frames: int) -> None:
        """Record ``frames`` steps, from the start of the next window."""
        self._armed_frames = frames

    def start_window(self) -> None:
        """Begin recording if armed, or carry on with an unfinished recording;
        otherwise this window is skipped."""
        if self._recording and not self.complete:
            return
        if not self._armed_frames:
            self._recording = False
            return
        if self._armed_frames > self._capacity:
            self._allocate(self._armed_frames)
        self._frames = self._armed_frames
        self._armed_frames = 0
        self._frame = 0
        self._score.zero_()
        self._recording = True

    def step_taken(self, counted_worlds: torch.Tensor | None = None) -> None:
        """Keep the step just taken, if a recording is running and not complete.

        ``counted_worlds`` is the ``(W,)`` mask of the worlds whose rewards
        count toward their score, for evaluation; in training every world's do.
        """
        if not self._recording or self.complete:
            return
        frame = self._frame
        self._qpos[frame].copy_(self._simulation.qpos)
        self._episode_start[frame].copy_(self._episode.episode_ended)
        self._target[frame].copy_(self._step_facts.target_position)
        self._range_radius[frame].copy_(self._step_facts.range_radius)
        rewards = self._step_facts.reward_parts.sum(dim=(1, 2))
        if counted_worlds is not None:
            rewards = rewards * counted_worlds
        self._score += rewards
        self._frame += 1

    def take(self, levels: int, worlds: int | str, selection: str) -> RecordedWindow:
        """The recording's chosen worlds, copied to the CPU; ends it, complete or not.

        ``worlds`` is how many worlds to keep, or ``"all"``; every world is
        kept when there are no more than that. Otherwise ``"ranked"`` splits
        the worlds by the rank of their summed reward into as many bands as
        it keeps, from the best to the worst, and keeps the best world of each
        (MujocoReplay's ``selected_ranks`` with one world per band, the rule
        its viewer uses); ``"first"`` keeps the first worlds, for continuity
        across recordings. Worlds come out in rank order, best first, each with its
        level among ``levels``.
        """
        self._recording = False
        frames = self._frame
        order = torch.argsort(self._score, descending=True, stable=True)
        kept = self._world_count if worlds == "all" else min(worlds, self._world_count)
        if kept == self._world_count:
            chosen = order
            ranks = np.arange(self._world_count)
        elif selection == "ranked":
            ranks, _ = selected_ranks(self._world_count, kept, 1)
            chosen = order[torch.as_tensor(ranks, device=self._device)]
        else:
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
            range_radius=self._range_radius[:frames, chosen].cpu().numpy(),
            rank=np.asarray(ranks, dtype=np.int64) + 1,
            ranked_worlds=self._world_count,
            level_count=levels,
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
        self._range_radius = torch.zeros((frames, world_count), device=device)
        self._score = torch.zeros(world_count, device=device)
        self._capacity = frames
