"""Focused checks for the serial environment-pool boundary."""

from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest

from centipede.environment import CentipedeParallelEnv
from centipede.training import environment_pool
from centipede.training.environment_pool import (
    AgentActions,
    AgentObservations,
    EnvironmentStep,
    ProcessEnvironmentPool,
    SerialEnvironmentPool,
    _environment_worker,
    _group_splitter,
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "assembly.xml"


def _observations(value: float) -> AgentObservations:
    """Build a distinguishable one-agent observation dictionary."""
    return {0: np.array([value], dtype=np.float32)}


def _step_result(value: float, *, ended: bool = False) -> EnvironmentStep:
    """Build one complete PettingZoo step result for ordering checks."""
    return (
        _observations(value),
        {0: value},
        {0: ended},
        {0: False},
        {0: {"value": value}},
    )


def _assert_observations_equal(
    first: AgentObservations,
    second: AgentObservations,
) -> None:
    """Compare every agent observation without ambiguous array equality."""

    assert first.keys() == second.keys()
    for agent in first:
        np.testing.assert_array_equal(first[agent], second[agent])


@pytest.mark.parametrize(
    ("environment_count", "worker_count", "expected_sizes"),
    [
        (32, 4, (8, 8, 8, 8)),
        (32, 8, (4, 4, 4, 4, 4, 4, 4, 4)),
        (10, 4, (3, 3, 2, 2)),
    ],
)
def test_group_splitter_balances_contiguous_global_indices(
    environment_count: int,
    worker_count: int,
    expected_sizes: tuple[int, ...],
) -> None:
    """Assign every replica once while keeping worker loads nearly equal."""

    assignments = _group_splitter(environment_count, worker_count)

    assert tuple(map(len, assignments)) == expected_sizes
    assert tuple(index for group in assignments for index in group) == tuple(
        range(environment_count)
    )


def test_environment_worker_executes_commands_and_closes_owned_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the child-side protocol without starting an operating-system process."""

    first = Mock()
    second = Mock()
    first.reset.side_effect = [(_observations(1.0), {}), (_observations(3.0), {})]
    second.reset.return_value = (_observations(2.0), {})
    first.step.return_value = _step_result(4.0)
    second.step.return_value = _step_result(5.0)
    environment_factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(
        environment_pool,
        "CentipedeParallelEnv",
        environment_factory,
    )

    first_actions = {0: np.array([0.1], dtype=np.float32)}
    second_actions = {0: np.array([0.2], dtype=np.float32)}
    connection = Mock()
    connection.recv.side_effect = [
        ("reset_all", [101, 202]),
        ("step_all", [first_actions, second_actions]),
        ("reset_indices", [10]),
        ("close", None),
    ]

    _environment_worker(connection, Path("model.xml"), 2_048, (10, 11))

    assert environment_factory.call_args_list == [
        call(model_path=Path("model.xml"), max_episode_steps=2_048),
        call(model_path=Path("model.xml"), max_episode_steps=2_048),
    ]
    assert connection.send.call_args_list == [
        call(("ready", None)),
        call(("ok", [_observations(1.0), _observations(2.0)])),
        call(("ok", [first.step.return_value, second.step.return_value])),
        call(("ok", {10: _observations(3.0)})),
    ]
    first.reset.assert_has_calls([call(seed=101), call()])
    second.reset.assert_called_once_with(seed=202)
    first.step.assert_called_once_with(first_actions)
    second.step.assert_called_once_with(second_actions)
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_environment_worker_reports_a_formatted_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return enough context for the parent to diagnose a child-process failure."""

    environment = Mock()
    monkeypatch.setattr(
        environment_pool,
        "CentipedeParallelEnv",
        Mock(return_value=environment),
    )
    connection = Mock()
    connection.recv.return_value = ("invalid", None)

    _environment_worker(connection, Path("model.xml"), 2_048, (0,))

    status, traceback_text = connection.send.call_args_list[-1].args[0]
    assert status == "error"
    assert "Traceback (most recent call last)" in traceback_text
    assert "RuntimeError: Unknown worker command: 'invalid'" in traceback_text
    environment.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_environment_worker_closes_partial_startup_after_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release earlier replicas and report a later model-construction failure."""

    first_environment = Mock()
    monkeypatch.setattr(
        environment_pool,
        "CentipedeParallelEnv",
        Mock(
            side_effect=[
                first_environment,
                RuntimeError("model construction failed"),
            ]
        ),
    )
    connection = Mock()

    _environment_worker(connection, Path("model.xml"), 2_048, (0, 1))

    status, traceback_text = connection.send.call_args.args[0]
    assert status == "error"
    assert "RuntimeError: model construction failed" in traceback_text
    connection.recv.assert_not_called()
    first_environment.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_process_pool_constructor_starts_all_workers_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain parent endpoints after every spawned worker reports ready."""

    parents = [Mock(), Mock()]
    children = [Mock(), Mock()]
    processes = [Mock(), Mock()]
    for parent in parents:
        parent.recv.return_value = ("ready", None)
    for process in processes:
        process.is_alive.return_value = False

    context = Mock()
    context.Pipe.side_effect = list(zip(parents, children, strict=True))
    context.Process.side_effect = processes
    monkeypatch.setattr(
        environment_pool.multiprocessing,
        "get_context",
        Mock(return_value=context),
    )

    pool = ProcessEnvironmentPool(
        model_path=Path("model.xml"),
        max_episode_steps=2_048,
        environment_count=5,
        worker_count=2,
    )

    assert pool.environment_count == 5
    assert pool._assignments == ((0, 1, 2), (3, 4))
    assert pool._environment_owners == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    assert [process.start.call_count for process in processes] == [1, 1]
    assert [child.close.call_count for child in children] == [1, 1]
    assert [parent.recv.call_count for parent in parents] == [1, 1]
    assert context.Process.call_args_list == [
        call(
            target=_environment_worker,
            args=(children[0], Path("model.xml"), 2_048, (0, 1, 2)),
            name="centipede-worker-0",
        ),
        call(
            target=_environment_worker,
            args=(children[1], Path("model.xml"), 2_048, (3, 4)),
            name="centipede-worker-1",
        ),
    ]

    pool.close()
    pool.close()

    for parent, process in zip(parents, processes, strict=True):
        parent.send.assert_called_once_with(("close", None))
        parent.close.assert_called_once_with()
        assert process.join.call_count == 1
        process.terminate.assert_not_called()
        process.close.assert_called_once_with()


def test_process_pool_constructor_aborts_all_workers_after_startup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not leave processes or pipe endpoints alive after failed readiness."""

    parents = [Mock(), Mock()]
    children = [Mock(), Mock()]
    processes = [Mock(), Mock()]
    parents[0].recv.return_value = ("ready", None)
    parents[1].recv.return_value = ("error", "model load traceback")
    for process in processes:
        process.is_alive.return_value = True

    context = Mock()
    context.Pipe.side_effect = list(zip(parents, children, strict=True))
    context.Process.side_effect = processes
    monkeypatch.setattr(
        environment_pool.multiprocessing,
        "get_context",
        Mock(return_value=context),
    )

    with pytest.raises(
        RuntimeError,
        match="Worker 1 failed during startup",
    ) as error:
        ProcessEnvironmentPool(
            model_path=Path("model.xml"),
            max_episode_steps=2_048,
            environment_count=4,
            worker_count=2,
        )

    assert "model load traceback" in str(error.value)
    for parent, child, process in zip(parents, children, processes, strict=True):
        parent.send.assert_called_once_with(("close", None))
        parent.close.assert_called_once_with()
        child.close.assert_called_once_with()
        process.terminate.assert_called_once_with()
        assert process.join.call_count == 1
        process.close.assert_called_once_with()


def test_process_pool_routes_commands_and_restores_global_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch commands by worker while preserving the public replica ordering."""

    parents = [Mock(), Mock()]
    children = [Mock(), Mock()]
    processes = [Mock(), Mock()]
    for process in processes:
        process.is_alive.return_value = False

    reset_responses = [
        [_observations(0.0), _observations(1.0)],
        [_observations(2.0)],
    ]
    step_responses = [
        [_step_result(10.0), _step_result(11.0)],
        [_step_result(12.0)],
    ]
    parents[0].recv.side_effect = [
        ("ready", None),
        ("ok", reset_responses[0]),
        ("ok", step_responses[0]),
        ("ok", {0: _observations(20.0)}),
    ]
    parents[1].recv.side_effect = [
        ("ready", None),
        ("ok", reset_responses[1]),
        ("ok", step_responses[1]),
        ("ok", {2: _observations(22.0)}),
    ]

    context = Mock()
    context.Pipe.side_effect = list(zip(parents, children, strict=True))
    context.Process.side_effect = processes
    monkeypatch.setattr(
        environment_pool.multiprocessing,
        "get_context",
        Mock(return_value=context),
    )
    pool = ProcessEnvironmentPool(
        model_path=Path("model.xml"),
        max_episode_steps=2_048,
        environment_count=3,
        worker_count=2,
    )

    calls = Mock()
    calls.attach_mock(parents[0], "worker_0")
    calls.attach_mock(parents[1], "worker_1")

    observations = pool.reset_all([101, 102, 103])

    assert calls.mock_calls == [
        call.worker_0.send(("reset_all", [101, 102])),
        call.worker_1.send(("reset_all", [103])),
        call.worker_0.recv(),
        call.worker_1.recv(),
    ]
    assert [float(result[0][0]) for result in observations] == [0.0, 1.0, 2.0]

    actions: list[AgentActions] = [
        {0: np.array([value], dtype=np.float32)} for value in (0.1, 0.2, 0.3)
    ]
    results = pool.step_all(actions)

    assert parents[0].send.call_args_list[-1] == call(
        ("step_all", [actions[0], actions[1]])
    )
    assert parents[1].send.call_args_list[-1] == call(("step_all", [actions[2]]))
    for result, expected in zip(
        results,
        [*step_responses[0], *step_responses[1]],
        strict=True,
    ):
        assert result is expected

    reset_observations = pool.reset_indices([2, 0])

    assert parents[0].send.call_args_list[-1] == call(("reset_indices", [0]))
    assert parents[1].send.call_args_list[-1] == call(("reset_indices", [2]))
    assert list(reset_observations) == [2, 0]
    _assert_observations_equal(reset_observations[2], _observations(22.0))
    _assert_observations_equal(reset_observations[0], _observations(20.0))
    pool.close()


def test_process_pool_aborts_after_runtime_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface a child traceback and release the complete pool immediately."""

    parent = Mock()
    child = Mock()
    process = Mock()
    process.is_alive.return_value = True
    parent.recv.side_effect = [
        ("ready", None),
        ("error", "step traceback"),
    ]
    context = Mock()
    context.Pipe.return_value = (parent, child)
    context.Process.return_value = process
    monkeypatch.setattr(
        environment_pool.multiprocessing,
        "get_context",
        Mock(return_value=context),
    )
    pool = ProcessEnvironmentPool(
        model_path=Path("model.xml"),
        max_episode_steps=2_048,
        environment_count=1,
        worker_count=1,
    )

    with pytest.raises(RuntimeError, match="Worker 0 failed during reset_all") as error:
        pool.reset_all([101])

    assert "step traceback" in str(error.value)
    assert pool._closed is True
    process.terminate.assert_called_once_with()
    parent.close.assert_called_once_with()


def test_process_pool_matches_serial_reset_and_step() -> None:
    """Preserve deterministic MuJoCo results across the process boundary."""

    serial_environments = [
        CentipedeParallelEnv(model_path=MODEL_PATH, max_episode_steps=10)
        for _ in range(2)
    ]
    serial_pool = SerialEnvironmentPool(serial_environments)
    process_pool = ProcessEnvironmentPool(
        model_path=MODEL_PATH,
        max_episode_steps=10,
        environment_count=2,
        worker_count=2,
    )

    try:
        seeds = [301, 302]
        serial_observations = serial_pool.reset_all(seeds)
        process_observations = process_pool.reset_all(seeds)
        for serial, processed in zip(
            serial_observations,
            process_observations,
            strict=True,
        ):
            _assert_observations_equal(serial, processed)

        actions = [
            {
                agent: np.zeros(environment.action_space(agent).shape, dtype=np.float32)
                for agent in environment.possible_agents
            }
            for environment in serial_environments
        ]
        serial_steps = serial_pool.step_all(actions)
        process_steps = process_pool.step_all(actions)

        for serial, processed in zip(serial_steps, process_steps, strict=True):
            _assert_observations_equal(serial[0], processed[0])
            assert serial[1:] == processed[1:]
    finally:
        serial_pool.close()
        process_pool.close()


def test_serial_pool_resets_replicas_in_stable_order() -> None:
    """Keep replica identity and seed assignment independent of object timing."""
    first = Mock()
    second = Mock()
    first.reset.return_value = (_observations(1.0), {0: {}})
    second.reset.return_value = (_observations(2.0), {0: {}})
    pool = SerialEnvironmentPool([first, second])

    observations = pool.reset_all([101, 202])

    assert pool.environment_count == 2
    assert np.array_equal(observations[0][0], np.array([1.0], dtype=np.float32))
    assert np.array_equal(observations[1][0], np.array([2.0], dtype=np.float32))
    first.reset.assert_called_once_with(seed=101)
    second.reset.assert_called_once_with(seed=202)


def test_serial_pool_steps_all_replicas_without_hidden_reset() -> None:
    """Return terminal observations before the coordinator requests any reset."""
    first = Mock()
    second = Mock()
    first.step.return_value = _step_result(1.0)
    second.step.return_value = _step_result(2.0, ended=True)
    pool = SerialEnvironmentPool([first, second])
    actions: list[AgentActions] = [
        {0: np.array([0.1], dtype=np.float32)},
        {0: np.array([0.2], dtype=np.float32)},
    ]

    results = pool.step_all(actions)

    assert results[0] is first.step.return_value
    assert results[1] is second.step.return_value
    first.step.assert_called_once_with(actions[0])
    second.step.assert_called_once_with(actions[1])
    first.reset.assert_not_called()
    second.reset.assert_not_called()


def test_serial_pool_resets_only_selected_replicas() -> None:
    """Restart ended replicas without disturbing continuing physical episodes."""
    environments = [Mock() for _ in range(3)]
    environments[0].reset.return_value = (_observations(0.0), {0: {}})
    environments[2].reset.return_value = (_observations(2.0), {0: {}})
    pool = SerialEnvironmentPool(environments)

    observations = pool.reset_indices([2, 0])

    assert list(observations) == [2, 0]
    assert np.array_equal(observations[2][0], np.array([2.0], dtype=np.float32))
    assert np.array_equal(observations[0][0], np.array([0.0], dtype=np.float32))
    assert [environment.reset.call_count for environment in environments] == [1, 0, 1]
    environments[2].reset.assert_called_once_with()
    environments[0].reset.assert_called_once_with()


def test_serial_pool_closes_owned_environments_once() -> None:
    """Make repeated cleanup safe without closing a replica more than once."""
    environments = [Mock() for _ in range(2)]
    pool = SerialEnvironmentPool(environments)

    pool.close()
    pool.close()

    for environment in environments:
        environment.close.assert_called_once_with()
