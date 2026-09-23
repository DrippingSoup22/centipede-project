"""Environment-replica boundary used by synchronous rollout collection.

The rollout coordinator owns policy and trajectory state, while a pool owns the
complete PettingZoo environments. The serial implementation preserves the
existing in-process behavior and will serve as the correctness reference for the
spawned process pool.
"""

import multiprocessing
from collections.abc import Mapping, Sequence
from contextlib import suppress
from multiprocessing.process import BaseProcess
from pathlib import Path
from time import monotonic
from traceback import format_exc
from typing import Any, Protocol, TypeAlias, TypeVar, cast

from centipede.environment import (
    Action,
    AgentID,
    CentipedeParallelEnv,
    Observation,
)

# Pool calls retain PettingZoo's per-agent dictionaries. Replica containers are
# always ordered by the stable environment index assigned by the experiment.
AgentObservations: TypeAlias = dict[AgentID, Observation]
AgentActions: TypeAlias = dict[AgentID, Action]
EnvironmentStep: TypeAlias = tuple[
    AgentObservations,
    dict[AgentID, float],
    dict[AgentID, bool],
    dict[AgentID, bool],
    dict[AgentID, dict[str, Any]],
]

_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_ValueT = TypeVar("_ValueT")


class _WorkerConnection(Protocol):
    """Describe the common API of platform-specific multiprocessing endpoints."""

    def send(self, obj: Any) -> None: ...

    def recv(self) -> Any: ...

    def close(self) -> None: ...


class EnvironmentPool(Protocol):
    """Define the operations required by synchronous rollout collection."""

    @property
    def environment_count(self) -> int:
        """Return the number of complete environment replicas in the pool."""
        ...

    def reset_all(self, seeds: Sequence[int | None]) -> list[AgentObservations]:
        """Reset every replica and return observations in replica order."""
        ...

    def step_all(self, actions: Sequence[AgentActions]) -> list[EnvironmentStep]:
        """Advance every replica once and return results in replica order."""
        ...

    def reset_indices(self, indices: Sequence[int]) -> dict[int, AgentObservations]:
        """Reset selected ended replicas and return their new observations."""
        ...

    def close(self) -> None:
        """Release every environment resource owned by the pool."""
        ...


def _group_splitter(
    environment_count: int,
    worker_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Divide global environment indices into balanced contiguous groups."""

    base_size, remaining = divmod(environment_count, worker_count)
    assignments: list[tuple[int, ...]] = []
    first_index = 0

    for worker_idx in range(worker_count):
        # The first ``remaining`` workers receive one additional replica.
        group_size = base_size + int(worker_idx < remaining)
        last_index = first_index + group_size
        assignments.append(tuple(range(first_index, last_index)))
        first_index = last_index

    return tuple(assignments)


def _environment_worker(
    connection: _WorkerConnection,
    model_path: Path,
    max_episode_steps: int,
    assigned_indices: tuple[int, ...],
) -> None:
    """Own assigned environments and execute commands from the parent process."""

    environments: dict[int, CentipedeParallelEnv] = {}

    try:
        for index in assigned_indices:
            environment = CentipedeParallelEnv(
                model_path=model_path,
                max_episode_steps=max_episode_steps,
            )
            environments[index] = environment

        # The parent treats construction as successful only after every assigned
        # MuJoCo environment exists inside this worker.
        connection.send(("ready", None))

        while True:
            command, payload = connection.recv()

            if command == "reset_all":
                seeds = payload
                observations: list[AgentObservations] = []

                for index, seed in zip(assigned_indices, seeds, strict=True):
                    observation, _ = environments[index].reset(seed=seed)
                    observations.append(observation)
                connection.send(("ok", observations))

            elif command == "step_all":
                actions = payload
                step_results: list[EnvironmentStep] = []
                for index, action in zip(assigned_indices, actions, strict=True):
                    result = environments[index].step(action)
                    step_results.append(result)
                connection.send(("ok", step_results))

            elif command == "reset_indices":
                indices = payload
                reset_observations: dict[int, AgentObservations] = {}

                for index in indices:
                    observation, _ = environments[index].reset()
                    reset_observations[index] = observation
                connection.send(("ok", reset_observations))

            elif command == "close":
                break

            else:
                raise RuntimeError(f"Unknown worker command: {command!r}")

    except BaseException:
        # format_exc() captures the active exception's type, message, and stack.
        # Sending that text lets the parent report failures raised in this process.
        error_traceback = format_exc()
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("error", error_traceback))

    finally:
        for environment in environments.values():
            with suppress(Exception):
                environment.close()
        with suppress(OSError):
            connection.close()


class ProcessEnvironmentPool:
    """Own persistent spawned workers that contain complete environments."""

    def __init__(
        self,
        model_path: Path,
        max_episode_steps: int,
        environment_count: int,
        worker_count: int,
    ) -> None:
        """Start workers and wait until every assigned environment is ready."""

        self._environment_count = environment_count
        self._worker_count = worker_count
        self._assignments = _group_splitter(environment_count, worker_count)
        self._environment_owners = {
            environment_index: worker_index
            for worker_index, assignment in enumerate(self._assignments)
            for environment_index in assignment
        }
        self._connections: list[_WorkerConnection] = []
        self._processes: list[BaseProcess] = []
        self._closed = False

        context = multiprocessing.get_context("spawn")

        try:
            # Start every worker before waiting, allowing model construction to
            # overlap across the available CPU processes.
            for worker_idx, assignment in enumerate(self._assignments):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_environment_worker,
                    args=(
                        child_connection,
                        model_path,
                        max_episode_steps,
                        assignment,
                    ),
                    name=f"centipede-worker-{worker_idx}",
                )

                try:
                    process.start()
                except BaseException:
                    parent_connection.close()
                    child_connection.close()
                    raise

                # Each process retains its child endpoint after spawn. The parent
                # keeps only the endpoint used to send commands and receive data.
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)

            # A ready message means every environment assigned to that worker was
            # constructed successfully, not merely that the process was created.
            for worker_idx in range(len(self._connections)):
                self._receive_response(
                    worker_idx,
                    operation="startup",
                    expected_status="ready",
                )

        except BaseException:
            self._abort_workers()
            raise

    @property
    def environment_count(self) -> int:
        """Return the total number of replicas across every worker."""

        return self._environment_count

    def reset_all(self, seeds: Sequence[int | None]) -> list[AgentObservations]:
        """Reset every worker replica and restore global environment order."""

        responses = cast(
            Mapping[int, Sequence[AgentObservations]],
            self._exchange("reset_all", self._group_values(seeds)),
        )
        return self._flatten_responses(responses)

    def step_all(self, actions: Sequence[AgentActions]) -> list[EnvironmentStep]:
        """Step every worker concurrently and restore global environment order."""

        responses = cast(
            Mapping[int, Sequence[EnvironmentStep]],
            self._exchange("step_all", self._group_values(actions)),
        )
        return self._flatten_responses(responses)

    def reset_indices(self, indices: Sequence[int]) -> dict[int, AgentObservations]:
        """Reset selected global replicas through only their owning workers."""

        worker_indices: dict[int, list[int]] = {
            worker_index: [] for worker_index in range(self._worker_count)
        }
        for environment_index in indices:
            worker_indices[self._environment_owners[environment_index]].append(
                environment_index
            )

        active_payloads = {
            worker_index: selected_indices
            for worker_index, selected_indices in worker_indices.items()
            if selected_indices
        }
        responses = cast(
            Mapping[int, Mapping[int, AgentObservations]],
            self._exchange("reset_indices", active_payloads),
        )
        merged = {
            environment_index: observations
            for worker_index in active_payloads
            for environment_index, observations in responses[worker_index].items()
        }
        return {
            environment_index: merged[environment_index]
            for environment_index in indices
        }

    def close(self) -> None:
        """Close workers gracefully and terminate any that miss the deadline."""

        if self._closed:
            return

        # Mark first so repeated cleanup from an outer finally block is harmless.
        self._closed = True
        self._shutdown_workers(grace_period=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)

    def _exchange(
        self,
        command: str,
        payloads: Mapping[int, Any],
    ) -> dict[int, Any]:
        """Send one command to all selected workers before receiving results."""

        if self._closed:
            raise RuntimeError("Environment pool is closed")

        try:
            # Complete the dispatch phase first so workers execute concurrently.
            for worker_index, payload in payloads.items():
                try:
                    self._connections[worker_index].send((command, payload))
                except (BrokenPipeError, EOFError, OSError) as error:
                    raise RuntimeError(
                        f"Worker {worker_index} disconnected while dispatching "
                        f"command {command!r}"
                    ) from error

            return {
                worker_index: self._receive_response(
                    worker_index,
                    operation=command,
                    expected_status="ok",
                )
                for worker_index in payloads
            }

        except BaseException:
            self._abort_workers()
            raise

    def _receive_response(
        self,
        worker_index: int,
        *,
        operation: str,
        expected_status: str,
    ) -> Any:
        """Receive one worker response and convert process failures to errors."""

        try:
            status, payload = self._connections[worker_index].recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"Worker {worker_index} exited during {operation}"
            ) from error

        if status == "error":
            raise RuntimeError(
                f"Worker {worker_index} failed during {operation}:\n{payload}"
            )
        if status != expected_status:
            raise RuntimeError(
                f"Worker {worker_index} returned status {status!r} during "
                f"{operation}; expected {expected_status!r}"
            )
        return payload

    def _group_values(
        self,
        values: Sequence[_ValueT],
    ) -> dict[int, list[_ValueT]]:
        """Route globally ordered values to their assigned workers."""

        return {
            worker_index: [values[index] for index in assignment]
            for worker_index, assignment in enumerate(self._assignments)
        }

    def _flatten_responses(
        self,
        responses: Mapping[int, Sequence[_ValueT]],
    ) -> list[_ValueT]:
        """Flatten worker-local sequences back into global replica order."""

        return [
            value
            for worker_index in range(self._worker_count)
            for value in responses[worker_index]
        ]

    def _abort_workers(self) -> None:
        """Immediately stop workers after startup or communication failure."""

        self._closed = True
        self._shutdown_workers(grace_period=0.0)

    def _shutdown_workers(self, *, grace_period: float) -> None:
        """Release all worker resources, optionally allowing a graceful exit."""

        for connection in self._connections:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("close", None))

        if grace_period > 0.0:
            self._join_until(
                self._processes,
                deadline=monotonic() + grace_period,
            )

        stragglers: list[BaseProcess] = []
        for process in self._processes:
            with suppress(OSError, ValueError):
                if process.is_alive():
                    process.terminate()
                    stragglers.append(process)

        # Startup failures skip the graceful join, so every process still needs
        # reaping. Normal shutdown only needs to join terminated stragglers again.
        processes_to_reap = self._processes if grace_period == 0.0 else stragglers
        self._join_until(
            processes_to_reap,
            deadline=monotonic() + _WORKER_SHUTDOWN_TIMEOUT_SECONDS,
        )

        for connection in self._connections:
            with suppress(OSError):
                connection.close()

        for process in self._processes:
            with suppress(OSError, ValueError):
                process.close()

    @staticmethod
    def _join_until(
        processes: Sequence[BaseProcess],
        *,
        deadline: float,
    ) -> None:
        """Join several processes within one shared absolute deadline."""

        for process in processes:
            remaining = max(0.0, deadline - monotonic())
            with suppress(AssertionError, OSError, ValueError):
                process.join(timeout=remaining)


class SerialEnvironmentPool:
    """Own and step environment replicas sequentially in the current process."""

    def __init__(self, environments: Sequence[CentipedeParallelEnv]) -> None:
        """Retain the trusted environment replicas in their stable index order."""
        self._environments = tuple(environments)
        self._closed = False

    @property
    def environment_count(self) -> int:
        """Return the number of owned replicas."""
        return len(self._environments)

    def reset_all(self, seeds: Sequence[int | None]) -> list[AgentObservations]:
        """Reset all replicas without retaining reset-only diagnostic infos."""
        return [
            environment.reset(seed=seed)[0]
            for environment, seed in zip(self._environments, seeds, strict=True)
        ]

    def step_all(self, actions: Sequence[AgentActions]) -> list[EnvironmentStep]:
        """Step each replica once without automatically resetting ended episodes."""
        return [
            environment.step(environment_actions)
            for environment, environment_actions in zip(
                self._environments, actions, strict=True
            )
        ]

    def reset_indices(self, indices: Sequence[int]) -> dict[int, AgentObservations]:
        """Continue each selected replica's random stream after an episode end."""
        return {index: self._environments[index].reset()[0] for index in indices}

    def close(self) -> None:
        """Close every owned environment exactly once."""
        if self._closed:
            return
        self._closed = True
        for environment in self._environments:
            environment.close()
