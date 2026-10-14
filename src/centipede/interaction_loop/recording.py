"""Recording whole episodes of the first worlds, for replay in MujocoReplay.

``EpisodeRecorder`` belongs to the loop's diagnostics. The experiment arms it
with a number of frames and of worlds, the first ones; the loop's diagnostics
feed it after every step. Each recorded world starts at its own next episode
start, so that its frames hold one whole episode however the worlds' episodes
are staggered, or at once, for an evaluation that starts from a reset. A frame
holds the poses after a step, whether an episode started there, and the values
the environment's step facts keep for recordings, such as the target; each
world's score adds up the environment's ``recording_score``. Every step's
copies are a few small tensor operations that never wait for the GPU; only
``complete`` and ``take`` do, between cycles. The file format is
MujocoReplay's (its docs/recording-format.md).
"""

from dataclasses import dataclass

import numpy as np
import torch

from centipede.diagnostics_category import descriptions


@dataclass(frozen=True)
class RecordedEpisodes:
    """The recorded worlds' frames as NumPy arrays on the CPU, best score first.

    ``K`` worlds over ``T`` frames: ``qpos (T, K, nq)``, ``score (K,)``,
    ``world_ids (K,)``, ``episode_start (T, K)``, and ``kept``, each value the
    step facts keep for recordings by name, ``(T, K, ...)``.
    """

    qpos: np.ndarray
    score: np.ndarray
    world_ids: np.ndarray
    episode_start: np.ndarray
    kept: dict[str, np.ndarray]


class EpisodeRecorder:
    """Captures the first worlds' poses over one episode each, on the device."""

    def __init__(self, diagnostics) -> None:
        """Prepare to record from the environment's diagnostics; allocate nothing."""
        self._diagnostics = diagnostics
        self._kept_names = [
            name
            for name, description in descriptions(diagnostics.step).items()
            if description.summary == "recorded"
        ]
        self._recording = False
        self._frames = 0
        self._buffers: dict[str, torch.Tensor] = {}

    @property
    def recording(self) -> bool:
        """Whether a recording is running: armed, and not yet taken."""
        return self._recording

    def arm(self, frames: int, worlds: int, at_next_episode: bool = True) -> None:
        """Record ``frames`` steps of each of the first ``worlds`` worlds, from
        the next step on: each from its next episode start, or at once."""
        sources = self._sources()
        device = sources["qpos"].device
        worlds = min(worlds, sources["qpos"].shape[0])
        # One extra frame takes the writes of the worlds not recording.
        self._buffers = {
            name: torch.zeros(
                (frames + 1, worlds, *value.shape[1:]), dtype=value.dtype, device=device
            )
            for name, value in sources.items()
        }
        self._frames = frames
        self._frame = torch.zeros(worlds, dtype=torch.long, device=device)
        self._started = torch.full(
            (worlds,), not at_next_episode, dtype=torch.bool, device=device
        )
        self._score = torch.zeros(worlds, device=device)
        self._recording = True

    def complete(self) -> bool:
        """Whether every recorded world holds all its frames; waits for the GPU."""
        return self._recording and bool((self._frame == self._frames).all())

    def step_taken(self, counted_worlds: torch.Tensor | None = None) -> None:
        """Keep the step just taken in the worlds recording.

        ``counted_worlds`` is the ``(W,)`` mask of the worlds whose score still
        counts, for evaluation; in training every world's does.
        """
        if not self._recording:
            return
        sources = self._sources()
        worlds = self._score.shape[0]
        self._started |= sources["episode_start"][:worlds]
        active = self._started & (self._frame < self._frames)
        row = torch.where(active, self._frame, self._frames)
        column = torch.arange(worlds, device=row.device)
        for name, buffer in self._buffers.items():
            buffer[row, column] = sources[name][:worlds]
        score = self._diagnostics.recording_score()[:worlds]
        if counted_worlds is not None:
            score = score * counted_worlds[:worlds]
        self._score += torch.where(active, score, 0.0)
        self._frame += active

    def take(self) -> RecordedEpisodes:
        """The recording so far, copied to the CPU, best score first; ends it."""
        self._recording = False
        frames = int(self._frame.max())
        order = torch.argsort(self._score, descending=True, stable=True)
        arrays = {
            name: buffer[:frames, order].cpu().numpy()
            for name, buffer in self._buffers.items()
        }
        self._buffers = {}
        return RecordedEpisodes(
            qpos=arrays.pop("qpos"),
            score=self._score[order].cpu().numpy(),
            world_ids=order.cpu().numpy(),
            episode_start=arrays.pop("episode_start"),
            kept=arrays,
        )

    def _sources(self) -> dict[str, torch.Tensor]:
        """Each recorded tensor's source, ``(W, ...)``, by name."""
        diagnostics = self._diagnostics
        return {
            "qpos": diagnostics.simulation.qpos,
            "episode_start": diagnostics.episode.episode_ended,
            **{name: getattr(diagnostics.step, name) for name in self._kept_names},
        }
