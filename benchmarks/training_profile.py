"""Where the time of training goes, part by part, on the CPU and the GPU.

The interaction loop's own timing says how long a window's collecting and
learning take. This script splits that time into every part of the pipeline
that does work, and shows for each whether the CPU or the GPU limits it, so
that speed work aims at what measurements show. It builds the real components
from a training configuration file, as the experiment does, and runs a few
short windows of real training at one or more world counts:

1. Start-up: building the environment and the agents, and the first step,
   which compiles the GPU kernels (they are cached on disk afterwards).
2. Collecting and learning as training runs: the loop's own timing.
3. Every part of a step, timed twice: once waiting for the GPU before and
   after it (its full cost), and once without waiting (the CPU's time to issue
   its work). A part whose two times are close keeps the GPU waiting for the
   CPU. A part that itself waits for the GPU also absorbs, without waiting,
   all the work queued before it; the trace tells these apart.
4. A PyTorch profiler trace of a few steps: kernels per step, the CPU's waits
   for the GPU, how busy the GPU was, and the most expensive kernels. With
   ``--output`` the trace is saved; it opens in https://ui.perfetto.dev.
5. The physics alone: the physics steps of one action launched from Python
   and replayed as a recorded CUDA graph, and how often MuJoCo Warp's solver
   makes the CPU wait for the GPU.
6. Learning: the advantage estimation, and one minibatch at several sizes,
   full and as issued by the CPU.
7. Between cycles: the log line, checkpoint, recording, and the report at the
   configured run's final log length.
8. A projection of the whole configured run from these measurements.

The pipeline's methods are wrapped with timers at run time; no pipeline code
changes. Some private methods are reached on purpose: this is a diagnostic,
not pipeline code. Run from the repository root, for example on the GPU
desktop:

    python benchmarks/training_profile.py configs/baseline.toml \\
        --worlds 64 256 1024 --output benchmarks/results/<date>_<machine>

on the CPU backend, with the probe's settings:

    python benchmarks/training_profile.py configs/probe.toml

and locally on a GPU older than Volta, with the CG solver and small windows:

    python benchmarks/training_profile.py configs/training.toml --gpu-solver cg \\
        --worlds 16 --window-steps 4 --minibatch-sizes 64 1024
"""

import argparse
import functools
import gzip
import json
import math
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from centipede.agents.agents import Agents
from centipede.agents.segment_agent import SegmentAgent
from centipede.environment.environment import Environment
from centipede.experiment import experiment as experiment_module
from centipede.experiment.configuration import Configuration, read_configuration
from centipede.experiment.recordings import RecordingScene, training_recording
from centipede.experiment.run_folder import RunFolder
from centipede.interaction_loop.interaction_loop import InteractionLoop
from centipede.interaction_loop.settings import InteractionLoopSettings

SESSION_HOURS = 12  # Kaggle's limit for one notebook session
# The loop windows run per world count: warm-up, as training runs, waiting
# around every part, issuing only, and the profiler trace.
WINDOW_NAMES = ("warm-up", "plain", "waiting", "issuing", "trace")
# Profiled steps in the trace window, after one warm-up step. One step of the
# physics is tens of thousands of kernels, so the trace file is large already.
TRACE_STEPS = 1
MAX_MINIBATCHES = 64  # per minibatch size in the learning sweep
CPU_MINIBATCHES = 4  # the same sweep on the CPU, which is slower per minibatch
RULE = "=" * 78
# CPU-side calls in a trace that copy between the CPU and the GPU or wait for
# the GPU. PyTorch's are traced; Warp waits through the driver
# (cuStreamSynchronize), which the trace may not show.
TRANSFER_CALLS = re.compile(
    r"^(cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|"
    r"cuStreamSynchronize|cuCtxSynchronize|cuEventSynchronize|cudaMemcpy|"
    r"cuMemcpy)"
)


# -- Measuring tools ------------------------------------------------------------------


class Output:
    """Prints report lines and keeps them for the saved report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)


class PhaseTimers:
    """Times wrapped methods by their place in the call stack.

    ``mode`` is "off" (calls pass straight through), "waiting" (wait for the
    GPU before and after each call: its full cost), "issuing" (no waiting: the
    CPU's time to issue it), or "trace" (no timing; each call becomes a named
    range in the profiler's trace). A part is named by its callers, such as
    ``environment step / physics step / failure check``, so the same method
    called from two places is timed separately.
    """

    def __init__(self, synchronise) -> None:
        self.mode = "off"
        self.synchronise = synchronise
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.order: list[str] = []  # parts in the order they were first entered
        self._stack: list[str] = []

    def clear(self) -> None:
        self.seconds.clear()
        self.calls.clear()

    def wrap(self, owner: Any, attribute: str, label: str) -> None:
        """Replace ``owner.attribute`` with a timed version of itself."""
        original = getattr(owner, attribute)
        timers = self

        @functools.wraps(original)
        def timed(*args, **kwargs):
            if timers.mode == "off":
                return original(*args, **kwargs)
            timers._stack.append(label)
            key = " / ".join(timers._stack)
            if key not in timers.order:
                timers.order.append(key)
            range_name = (
                record_function(label) if timers.mode == "trace" else nullcontext()
            )
            try:
                if timers.mode == "waiting":
                    timers.synchronise()
                start = time.perf_counter()
                with range_name:
                    result = original(*args, **kwargs)
                if timers.mode == "waiting":
                    timers.synchronise()
                timers.seconds[key] += time.perf_counter() - start
                timers.calls[key] += 1
            finally:
                timers._stack.pop()
            return result

        setattr(owner, attribute, timed)


class ModuleProxy:
    """Stands in for a module, so that some of its functions can be wrapped."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class GPUSampler:
    """Samples the GPU's utilisation and memory with nvidia-smi, in the background.

    Utilisation is the share of the last sample period in which any kernel
    ran, as the driver reports it.
    """

    def __init__(self, device_index: int) -> None:
        self.samples: list[tuple[float, float, float]] = []
        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        self._command = [
            "nvidia-smi",
            "-i",
            f"GPU-{uuid}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while self._running:
            try:
                output = subprocess.run(
                    self._command, capture_output=True, text=True, timeout=5
                ).stdout
                utilisation, memory = (float(value) for value in output.split(","))
            except (OSError, ValueError, subprocess.SubprocessError):
                return  # no nvidia-smi: no samples
            self.samples.append((time.perf_counter(), utilisation, memory))
            time.sleep(0.1)

    def mean_between(self, start: float, end: float) -> float | None:
        """Mean utilisation of the samples taken between two clock readings."""
        inside = [sample[1] for sample in self.samples if start <= sample[0] <= end]
        return sum(inside) / len(inside) if inside else None

    def stop(self) -> None:
        self._running = False


def synchroniser(device: str):
    """A function that waits until the device has finished all queued work."""
    if device == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def timed_seconds(function, synchronise) -> float:
    """Wall time of ``function()``, waiting for the device before and after."""
    synchronise()
    start = time.perf_counter()
    function()
    synchronise()
    return time.perf_counter() - start


# -- Building and instrumenting the pipeline ------------------------------------------


def configuration_for(
    configuration: Configuration,
    world_count: int,
    window_steps: int,
    gpu_solver: str | None,
) -> Configuration:
    """The file's configuration with the measured world count, window, and solver."""
    simulation = replace(configuration.environment.simulation, world_count=world_count)
    if gpu_solver is not None:
        simulation = replace(simulation, gpu_solver=gpu_solver)
    return replace(
        configuration,
        environment=replace(configuration.environment, simulation=simulation),
        interaction_loop=InteractionLoopSettings(
            rollout_window_steps=window_steps, update_cycles=len(WINDOW_NAMES)
        ),
    )


def instrument(
    timers: PhaseTimers,
    environment: Environment,
    agents: Agents,
    loop: InteractionLoop,
) -> None:
    """Wrap every part of the pipeline that does work during training."""
    wrap = timers.wrap
    simulation = environment.simulation
    backend = simulation._backend

    wrap(agents, "act", "agents act")
    wrap(agents, "record", "agents record")
    wrap(agents, "update", "agents update")
    wrap(agents.diagnostics, "record_update", "learning diagnostics")
    for segment_agent in agents.segment_agents:
        wrap(segment_agent, "act", "segment agent")
        wrap(segment_agent, "record", "segment agent")
        wrap(segment_agent, "update", "segment agent")
        wrap(
            segment_agent.observation_normaliser, "normalize", "normalise observations"
        )
        wrap(segment_agent.ppo, "sample_action", "sample action")
        wrap(segment_agent.ppo, "state_value", "critic value")
        wrap(segment_agent.ppo, "update", "PPO update")
        wrap(segment_agent.ppo, "_update_minibatch", "minibatch")
        storage = segment_agent.rollout_storage
        wrap(storage, "store_action", "store")
        wrap(storage, "store_outcome", "store")
        wrap(storage, "training_batch", "advantage estimation")

    wrap(environment, "step", "environment step")
    wrap(environment, "_place_targets", "place targets")
    wrap(environment.reward_function, "compute", "rewards")
    wrap(environment.observation_builder, "build", "observations")
    wrap(environment.diagnostics, "record_step", "step diagnostics")
    wrap(environment.diagnostics, "start_episodes", "start episodes")
    wrap(simulation, "step", "physics step")
    wrap(simulation, "reset", "reset worlds")
    if hasattr(backend, "gpu_data"):
        wrap(backend, "_check_worlds", "failure check")
        wrap(backend, "_read_state", "read physical state")
        install_mujoco_warp_proxy(timers)
    else:
        wrap(backend.diagnostics, "fill_from_cpu", "simulation diagnostics")

    wrap(loop.diagnostics, "step_taken", "loop diagnostics")
    wrap(loop.diagnostics.recorder, "step_taken", "recorder")
    wrap(loop.diagnostics.step_window, "add", "step summary")
    wrap(loop.diagnostics.episode_window, "add", "episode summary")
    wrap(loop.diagnostics.simulation_window, "add", "physics summary")


def install_mujoco_warp_proxy(timers: PhaseTimers) -> None:
    """Time MuJoCo Warp's step and forward as the GPU backend calls them.

    Each world count has its own timers, so each gets a new proxy around the
    real module.
    """
    import mujoco_warp

    from centipede.environment.simulation import gpu_backend

    proxy = ModuleProxy(mujoco_warp)
    proxy.step = mujoco_warp.step
    proxy.forward = mujoco_warp.forward
    proxy.reset_data = mujoco_warp.reset_data
    timers.wrap(proxy, "step", "MuJoCo Warp step")
    timers.wrap(proxy, "forward", "MuJoCo Warp forward")
    timers.wrap(proxy, "reset_data", "MuJoCo Warp reset")
    gpu_backend.mjw = proxy


# -- One world count ------------------------------------------------------------------


def profile_world_count(
    configuration: Configuration,
    configuration_path: Path,
    world_count: int,
    arguments: argparse.Namespace,
    output_folder: Path | None,
    out: Output,
) -> dict[str, Any]:
    """Every measurement at one world count; returns them as plain values."""
    run_configuration = configuration_for(
        configuration, world_count, arguments.window_steps, arguments.gpu_solver
    )
    device = run_configuration.agents.device
    synchronise = synchroniser(device)
    window_steps = arguments.window_steps
    seed = run_configuration.run.seed
    result: dict[str, Any] = {"worlds": world_count, "window_steps": window_steps}
    out(f"\n{RULE}\n{world_count} worlds, windows of {window_steps} steps\n{RULE}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. Start-up.
    startup = {}
    start = time.perf_counter()
    environment = Environment(run_configuration.environment)
    synchronise()
    startup["environment"] = time.perf_counter() - start
    start = time.perf_counter()
    agents = Agents(
        environment.segment_count,
        environment.observation_size,
        world_count,
        window_steps,
        run_configuration.agents,
        seed,
    )
    synchronise()
    startup["agents"] = time.perf_counter() - start
    loop = InteractionLoop(environment, agents, run_configuration.interaction_loop)
    result["segment_count"] = environment.segment_count
    zero_actions = torch.zeros(
        (world_count, environment.segment_count, 6), device=environment.device
    )
    startup["first reset"] = timed_seconds(lambda: environment.reset(seed), synchronise)
    startup["first step"] = timed_seconds(
        lambda: environment.step(zero_actions), synchronise
    )
    startup["second step"] = timed_seconds(
        lambda: environment.step(zero_actions), synchronise
    )
    result["startup_seconds"] = startup
    out("\nStart-up (kernels compile on the first steps of a fresh machine)")
    for name, seconds in startup.items():
        out(f"  {name:<14} {seconds:8.2f} s")

    timers = PhaseTimers(synchronise)
    instrument(timers, environment, agents, loop)
    sampler = GPUSampler(torch.cuda.current_device()) if device == "cuda" else None
    update_intervals: list[tuple[float, float]] = []
    original_update = agents.update

    def update_with_interval() -> None:
        start = time.perf_counter()
        original_update()
        update_intervals.append((start, time.perf_counter()))

    agents.update = update_with_interval

    # 2. to 4. The loop's windows; each yields after its update.
    cycles = loop.train(seed)
    window_timing: dict[str, dict[str, float]] = {}
    window_bounds: dict[str, tuple[float, float]] = {}
    phase_seconds: dict[str, dict[str, float]] = {}
    phase_calls: dict[str, int] = {}
    trace_summary: dict[str, Any] = {}
    recorder = loop.diagnostics.recorder
    for window_name in WINDOW_NAMES:
        if window_name == "issuing" and device != "cuda":
            continue  # without a GPU, issuing is the full cost
        timers.clear()
        timers.mode = {
            "waiting": "waiting",
            "issuing": "issuing",
            "trace": "trace",
        }.get(window_name, "off")
        if window_name in ("waiting", "issuing"):
            recorder.arm(window_steps)  # recorded windows cost the most
        window_start = time.perf_counter()
        if window_name == "trace":
            trace_summary = trace_window(
                cycles, loop, device, window_steps, output_folder, world_count
            )
        else:
            next(cycles)
        window_bounds[window_name] = (window_start, time.perf_counter())
        timing = loop.diagnostics.timing
        window_timing[window_name] = {
            "collecting_seconds": float(timing.collecting_seconds),
            "learning_seconds": float(timing.learning_seconds),
        }
        if window_name in ("waiting", "issuing"):
            phase_seconds[window_name] = dict(timers.seconds)
            phase_calls.update(timers.calls)
        if window_name == "waiting":
            # 7. Between cycles, while this window's recording is ready.
            result["between_cycles_seconds"] = between_cycles(
                run_configuration,
                configuration,
                configuration_path,
                environment,
                agents,
                loop,
                synchronise,
            )
    timers.mode = "off"
    if sampler is not None:
        sampler.stop()

    plain = window_timing["plain"]
    collect_step = plain["collecting_seconds"] / window_steps
    result["window_timing"] = window_timing
    result["collect_seconds_per_step"] = collect_step
    result["transitions_per_second"] = world_count / collect_step
    result["learning_seconds"] = plain["learning_seconds"]
    out("\nCollecting and learning as training runs (the loop's own timing)")
    for name in ("warm-up", "plain"):
        timing = window_timing[name]
        out(
            f"  {name:<8} window: collecting {timing['collecting_seconds']:8.2f} s"
            f"  ({timing['collecting_seconds'] / window_steps * 1000:9.1f} ms/step)"
            f"   learning {timing['learning_seconds']:8.2f} s"
        )
    out(f"  {world_count / collect_step:,.1f} transitions per second while collecting")

    if sampler is not None and sampler.samples:
        utilisation = {}
        collect_start = window_bounds["plain"][0]
        update_start, update_end = update_intervals[1]
        utilisation["collecting"] = sampler.mean_between(collect_start, update_start)
        utilisation["learning"] = sampler.mean_between(update_start, update_end)
        result["gpu_utilisation_percent"] = utilisation
        out(
            "  GPU utilisation (nvidia-smi): "
            + ", ".join(
                f"{name} {value:.0f}%" if value is not None else f"{name} no sample"
                for name, value in utilisation.items()
            )
        )
        out(
            "  (the share of time with any kernel running, however small; the"
            " trace's busy share is stricter)"
        )

    result["parts"] = report_parts(
        timers.order, phase_seconds, phase_calls, window_steps, out
    )
    if trace_summary:
        result["trace"] = trace_summary
        report_trace(trace_summary, out)

    # 5. The physics alone, and 6. learning.
    if hasattr(environment.simulation._backend, "gpu_data"):
        result["physics"] = profile_physics(environment, out)
    result["learning"] = profile_learning(
        agents,
        run_configuration,
        arguments.minibatch_sizes,
        synchronise,
        output_folder,
        world_count,
        out,
    )

    result["memory"] = measure_memory(agents, device, out)
    report_between_cycles(result["between_cycles_seconds"], out)
    result["projection"] = project(result, configuration, out)
    return result


def trace_window(
    cycles,
    loop: InteractionLoop,
    device: str,
    window_steps: int,
    output_folder: Path | None,
    world_count: int,
) -> dict[str, Any]:
    """Run one window with the profiler active on a few of its steps."""
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    active_steps = min(TRACE_STEPS, max(window_steps - 1, 1))
    profiler = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=active_steps, repeat=1),
    )
    diagnostics = loop.diagnostics
    original_step_taken = diagnostics.step_taken

    def step_taken_and_profiler_step(*args, **kwargs):
        original_step_taken(*args, **kwargs)
        profiler.step()

    diagnostics.step_taken = step_taken_and_profiler_step
    with profiler:
        next(cycles)
    diagnostics.step_taken = original_step_taken
    if output_folder is not None:
        export_trace(profiler, output_folder / f"trace_{world_count}_worlds.json")
    return summarise_trace(profiler, active_steps)


def export_trace(profiler, path: Path) -> None:
    """Save a trace as gzipped JSON, which https://ui.perfetto.dev opens."""
    profiler.export_chrome_trace(str(path))
    with path.open("rb") as source, gzip.open(f"{path}.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()


def summarise_trace(profiler, steps: int) -> dict[str, Any]:
    """Kernels, transfers, and GPU busy time per step, from a profiler trace."""
    kernels: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    transfers: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    gpu_busy_us = 0.0
    first_start, last_end = math.inf, -math.inf
    for event in profiler.events():
        start, end = event.time_range.start, event.time_range.end
        if event.device_type == torch.autograd.DeviceType.CUDA:
            # Named ranges also appear on the GPU's timeline; they are not work.
            if event.is_user_annotation:
                continue
            name = kernel_name(event.name)
            kernels[name][0] += 1
            kernels[name][1] += end - start
            gpu_busy_us += end - start
        elif TRANSFER_CALLS.match(event.name):
            transfers[event.name][0] += 1
            transfers[event.name][1] += end - start
        if event.device_type == torch.autograd.DeviceType.CPU and event.name.startswith(
            "ProfilerStep"
        ):
            first_start, last_end = min(first_start, start), max(last_end, end)
    wall_us = last_end - first_start if last_end > first_start else math.nan
    warp_kernels = sum(count for name, (count, _) in kernels.items() if is_warp(name))
    torch_kernels = sum(
        count for name, (count, _) in kernels.items() if not is_warp(name)
    )
    top = sorted(kernels.items(), key=lambda item: item[1][1], reverse=True)[:20]
    return {
        "steps": steps,
        "wall_ms_per_step": wall_us / steps / 1000,
        "gpu_busy_ms_per_step": gpu_busy_us / steps / 1000,
        "gpu_busy_share": gpu_busy_us / wall_us if wall_us == wall_us else None,
        "warp_kernels_per_step": warp_kernels / steps,
        "torch_kernels_per_step": torch_kernels / steps,
        "transfers_per_step": {
            name: {"count": count / steps, "ms": total / steps / 1000}
            for name, (count, total) in transfers.items()
        },
        "top_kernels": [
            {
                "name": name,
                "calls_per_step": count / steps,
                "ms_per_step": total / steps / 1000,
                "warp": is_warp(name),
            }
            for name, (count, total) in top
        ],
    }


def kernel_name(name: str) -> str:
    """A kernel's name without Warp's hash or template arguments, shortened."""
    name = re.sub(r"_[0-9a-f]{6,}_cuda_kernel_forward$", " [warp]", name)
    name = re.sub(r"<.*>", "<>", name)
    return name if len(name) <= 70 else name[:67] + "..."


def is_warp(name: str) -> bool:
    return name.endswith("[warp]")


def profile_physics(environment: Environment, out: Output) -> dict[str, Any]:
    """The physics steps alone: from Python, as a graph, and the solver's waits."""
    import mujoco_warp as mjw
    import warp as wp

    backend = environment.simulation._backend
    gpu_model, gpu_data = backend.gpu_model, backend.gpu_data
    steps = backend.physics_steps_per_action
    result: dict[str, Any] = {}

    # Each solver iteration outside a graph reads a flag on the CPU, so the
    # loop's iterations (the slowest world's) are the waits of one solve.
    iterations = []
    for _ in range(steps):
        mjw.step(gpu_model, gpu_data)
        iterations.append(int(wp.to_torch(gpu_data.solver_niter).max()))
    result["solver_iterations_per_physics_step"] = {
        "mean": sum(iterations) / len(iterations),
        "max": max(iterations),
    }

    def python_steps() -> None:
        for _ in range(steps):
            mjw.step(gpu_model, gpu_data)

    repeats = 2
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        python_steps()
    wp.synchronize()
    result["python_seconds"] = (time.perf_counter() - start) / repeats
    result["graph_seconds"] = None
    try:
        with wp.ScopedCapture(device=backend.device) as capture:
            python_steps()
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        result["graph_seconds"] = (time.perf_counter() - start) / repeats
    except Exception as error:  # report it and keep profiling
        result["graph_error"] = str(error)
    backend._check_worlds()

    out(f"\nPhysics alone: the {steps} physics steps of one action")
    iterations_summary = result["solver_iterations_per_physics_step"]
    waits_per_action = iterations_summary["mean"] * steps
    out(
        f"  solver iterations per physics step: mean {iterations_summary['mean']:.1f},"
        f" max {iterations_summary['max']}; outside a graph each one is a wait,"
        f" about {waits_per_action:,.0f} per action"
    )
    out(f"  launched from Python   {result['python_seconds']:8.3f} s")
    if result["graph_seconds"] is not None:
        out(
            f"  replayed as a graph    {result['graph_seconds']:8.3f} s"
            f"   ({result['python_seconds'] / result['graph_seconds']:.1f}x faster;"
            " the difference is the GPU waiting for the CPU)"
        )
    else:
        out(f"  graph capture failed: {result['graph_error']}")
    return result


def profile_learning(
    agents: Agents,
    configuration: Configuration,
    minibatch_sizes: list[int],
    synchronise,
    output_folder: Path | None,
    world_count: int,
    out: Output,
) -> dict[str, Any]:
    """Advantage estimation and minibatch times, on the first agent's last window."""
    agent = agents.segment_agents[0]
    ppo_settings = configuration.agents.ppo
    storage = agent.rollout_storage
    window_steps = storage.rewards.shape[0]

    def advantages():
        return storage.training_batch(
            discount=ppo_settings.discount, gae_lambda=ppo_settings.gae_lambda
        )

    gae_seconds = min(timed_seconds(advantages, synchronise) for _ in range(3))
    batch = advantages()
    sample_count = batch[0].shape[0]
    result: dict[str, Any] = {
        "advantage_seconds_per_row": gae_seconds / window_steps,
        "minibatch": {},
    }
    out("\nLearning, one segment agent")
    out(
        f"  advantage estimation: {gae_seconds * 1000:.1f} ms for {window_steps} rows"
        f" ({gae_seconds / window_steps * 1000:.2f} ms per row;"
        " a Python loop over the rows)"
    )
    # The same minibatches on a copy of the agent on the CPU, when the agents
    # are on the GPU: whether learning would be faster there.
    cpu_agent = cpu_batch = None
    if batch[0].is_cuda:
        cpu_agent = SegmentAgent(
            segment_index=0,
            world_count=1,
            observation_size=batch[0].shape[1],
            rollout_window_steps=1,
            settings=replace(configuration.agents, device="cpu"),
            seed=0,
        )
        cpu_batch = tuple(values.cpu() for values in batch)
    out("  one minibatch (actor and critic step):")
    out(
        f"  {'size':>7} {'full ms':>9} {'issued ms':>10} {'limit':>6}"
        + (f" {'on CPU ms':>10}" if cpu_agent else "")
    )
    sizes = sorted(set(minibatch_sizes) | {ppo_settings.minibatch_size})
    generator = torch.Generator(device=batch[0].device).manual_seed(0)
    for size in sizes:
        count = min(max(sample_count // size, 1), MAX_MINIBATCHES)
        # Drawn with repetition, so a size above the window's samples still
        # measures a minibatch of that size; the time depends on the size only.
        chunks = [
            torch.randint(
                sample_count, (size,), generator=generator, device=batch[0].device
            )
            for _ in range(count)
        ]

        def minibatches(chunks=chunks, ppo=agent.ppo, values_batch=batch) -> None:
            for indices in chunks:
                ppo._update_minibatch(*(values[indices] for values in values_batch))

        minibatches()  # warm-up: allocations and kernel selection
        synchronise()
        start = time.perf_counter()
        minibatches()
        issued = time.perf_counter() - start
        synchronise()
        full = time.perf_counter() - start
        limit = "CPU" if issued > 0.8 * full else "GPU"
        result["minibatch"][size] = {
            "full_ms": full / count * 1000,
            "issued_ms": issued / count * 1000,
            "limit": limit,
        }
        line = (
            f"  {size:>7} {full / count * 1000:>9.2f} {issued / count * 1000:>10.2f}"
            f" {limit:>6}"
        )
        if cpu_agent is not None:
            cpu_chunks = [indices.cpu() for indices in chunks[:CPU_MINIBATCHES]]
            cpu_minibatches = functools.partial(
                minibatches, cpu_chunks, cpu_agent.ppo, cpu_batch
            )
            cpu_minibatches()  # warm-up
            start = time.perf_counter()
            cpu_minibatches()
            cpu_ms = (time.perf_counter() - start) / len(cpu_chunks) * 1000
            result["minibatch"][size]["on_cpu_ms"] = cpu_ms
            line += f" {cpu_ms:>10.2f}"
        out(line)

    if output_folder is not None and batch[0].is_cuda:
        indices = torch.arange(
            min(ppo_settings.minibatch_size, sample_count), device=batch[0].device
        )
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            advantages()
            for _ in range(3):
                agent.ppo._update_minibatch(*(values[indices] for values in batch))
            synchronise()
        export_trace(trace, output_folder / f"trace_{world_count}_learning.json")
    return result


def between_cycles(
    run_configuration: Configuration,
    file_configuration: Configuration,
    configuration_path: Path,
    environment: Environment,
    agents: Agents,
    loop: InteractionLoop,
    synchronise,
) -> dict[str, float]:
    """The experiment's work between two cycles, in a temporary run folder."""
    run = run_configuration.run
    total_cycles = file_configuration.interaction_loop.update_cycles
    categories = experiment_module._training_categories(loop, agents, environment)
    result = {}
    with tempfile.TemporaryDirectory() as folder_path:
        folder = RunFolder.create(Path(folder_path), "profile")
        folder.write_configuration(run_configuration)
        folder.add_session(
            experiment_module._session_facts(
                configuration_path, run_configuration.agents.device, 0, None
            )
        )
        start = time.perf_counter()
        record = {"cycle": 1, "transitions": 0}
        record |= {category.name: category.plain_values() for category in categories}
        folder.append_log(record)
        result["log line"] = time.perf_counter() - start

        start = time.perf_counter()
        window = loop.diagnostics.recorder.take(
            run.record_levels, run.record_per_level, run.record_selection
        )
        scene = RecordingScene.from_run(run_configuration, environment)
        folder.write_recording(
            "cycle_0001",
            training_recording(
                scene, window, 1, total_cycles, run_configuration, folder
            ),
        )
        result["recording"] = time.perf_counter() - start

        start = time.perf_counter()
        folder.save_checkpoint(1, agents.state_dict())
        result["checkpoint"] = time.perf_counter() - start

        for cycle in range(2, total_cycles + 1):
            folder.append_log(record | {"cycle": cycle})
        start = time.perf_counter()
        if file_configuration.run.report:
            experiment_module._write_training_report(
                folder, run_configuration, categories
            )
        result["report"] = time.perf_counter() - start
    synchronise()
    return result


def measure_memory(agents: Agents, device: str, out: Output) -> dict[str, float]:
    """Memory in use after the windows, and the agents' stored data."""
    storage_bytes = sum(
        value.numel() * value.element_size()
        for agent in agents.segment_agents
        for value in vars(agent.rollout_storage).values()
        if isinstance(value, torch.Tensor)
    )
    result = {"agents_storage_mb": storage_bytes / 2**20}
    if device == "cuda":
        import warp as wp

        free, total = torch.cuda.mem_get_info()
        result["torch_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        result["warp_mb"] = wp.get_mempool_used_mem_current("cuda:0") / 2**20
        result["device_used_mb"] = (total - free) / 2**20
        result["device_total_mb"] = total / 2**20
    out("\nMemory")
    for name, value in result.items():
        out(f"  {name:<20} {value:10,.1f} MB")
    return result


# -- Reports --------------------------------------------------------------------------


def report_parts(
    order: list[str],
    phase_seconds: dict[str, dict[str, float]],
    calls: dict[str, int],
    window_steps: int,
    out: Output,
) -> dict[str, dict[str, float]]:
    """The per-part table: full and issuing time, per step or per update."""
    waiting = phase_seconds.get("waiting", {})
    issuing = phase_seconds.get("issuing", {})
    step_total = (
        sum(
            waiting.get(name, 0.0)
            for name in (
                "agents act",
                "environment step",
                "agents record",
                "loop diagnostics",
            )
        )
        / window_steps
    )
    result = {}
    out("\nEvery part of a step: its full time, waiting for the GPU before and after.")
    if issuing:
        out("'issued' is the CPU's time to issue the part without waiting; near 100%")
        out("of the full time, the GPU waits for the CPU, unless the part itself waits")
        out("for the GPU (the failure check, ended.any(), the solver's iterations).")
    for section, roots, per in (
        (
            "Collecting, per step",
            ("agents act", "environment step", "agents record", "loop diagnostics"),
            window_steps,
        ),
        ("Learning, per update", ("agents update",), 1),
    ):
        out(f"\n  {section}")
        out(
            f"  {'part':<52} {'calls':>6} {'full ms':>10} {'issued ms':>10}"
            f" {'issued':>7} {'share':>6}"
        )
        section_total = sum(waiting.get(root, 0.0) for root in roots) / per
        for name in order:
            if name.split(" / ")[0] not in roots or name not in waiting:
                continue
            depth = name.count(" / ")
            label = "  " * depth + name.split(" / ")[-1]
            full = waiting[name] / per * 1000
            issued = issuing.get(name, math.nan) / per * 1000
            share = full / (section_total * 1000) if section_total else math.nan
            result[name] = {
                "calls": calls[name] / per,
                "full_ms": full,
                "issued_ms": issued,
            }
            issued_columns = (
                f" {issued:>10.2f} {issued / full if full else math.nan:>7.0%}"
                if issuing
                else f" {'-':>10} {'-':>7}"
            )
            out(
                f"  {label[:52]:<52} {calls[name] / per:>6.0f} {full:>10.2f}"
                f"{issued_columns} {share:>6.0%}"
            )
    out(f"\n  Sum of the step's parts while waiting: {step_total * 1000:.1f} ms/step")
    return result


def report_trace(summary: dict[str, Any], out: Output) -> None:
    out(f"\nProfiler trace of {summary['steps']} step(s), in the middle of a window")
    out(f"  wall time {summary['wall_ms_per_step']:.1f} ms per step")
    if not summary["top_kernels"]:
        return  # no GPU
    out(
        f"  GPU busy {summary['gpu_busy_ms_per_step']:.1f} ms per step"
        f" ({summary['gpu_busy_share']:.0%}): the summed time of its kernels"
    )
    out(
        f"  kernels per step: {summary['warp_kernels_per_step']:,.0f} Warp (physics),"
        f" {summary['torch_kernels_per_step']:,.0f} PyTorch"
    )
    if summary["transfers_per_step"]:
        out("  CPU-side copies and waits, per step (CPU time of the calls; Warp's own")
        out("  waits go through the driver and may be missing here):")
        for name, transfer in summary["transfers_per_step"].items():
            out(
                f"    {name:<28} {transfer['count']:>8,.0f} times"
                f" {transfer['ms']:>10.1f} ms"
            )
    out("  most expensive kernels, per step:")
    for kernel in summary["top_kernels"]:
        out(
            f"    {kernel['ms_per_step']:>9.2f} ms {kernel['calls_per_step']:>7,.0f}x"
            f"  {kernel['name']}"
        )


def report_between_cycles(seconds: dict[str, float], out: Output) -> None:
    out("\nBetween cycles (the report at the configured run's final log length)")
    for name, value in seconds.items():
        out(f"  {name:<12} {value:8.2f} s")


def project(result: dict[str, Any], configuration: Configuration, out: Output) -> dict:
    """The configured run's time, from this world count's measurements."""
    world_count = result["worlds"]
    window_steps = configuration.interaction_loop.rollout_window_steps
    cycles = configuration.interaction_loop.update_cycles
    run = configuration.run
    ppo = configuration.agents.ppo
    learning = result["learning"]
    agent_count = result["segment_count"]

    minibatch_ms = learning["minibatch"][ppo.minibatch_size]["full_ms"]
    minibatches = ppo.update_epochs * math.ceil(
        world_count * window_steps / ppo.minibatch_size
    )
    update = agent_count * (
        learning["advantage_seconds_per_row"] * window_steps
        + minibatches * minibatch_ms / 1000
    )
    measured_window = result["window_steps"]
    measured_minibatches = ppo.update_epochs * math.ceil(
        world_count * measured_window / ppo.minibatch_size
    )
    predicted_measured = agent_count * (
        learning["advantage_seconds_per_row"] * measured_window
        + measured_minibatches * minibatch_ms / 1000
    )
    collect = result["collect_seconds_per_step"] * window_steps
    between = result["between_cycles_seconds"]
    checkpoints = math.ceil(cycles / run.checkpoint_every_cycles)
    recordings = (
        math.ceil(cycles / run.record_every_cycles) if run.record_every_cycles else 0
    )
    startup = sum(result["startup_seconds"].values())
    total = (
        startup
        + cycles * (collect + update + between.get("log line", 0.0))
        + checkpoints * (between.get("checkpoint", 0.0) + between.get("report", 0.0))
        + recordings * between.get("recording", 0.0) * window_steps / measured_window
    )
    transitions = world_count * window_steps * cycles
    projection = {
        "cycles": cycles,
        "window_steps": window_steps,
        "transitions": transitions,
        "collect_seconds_per_cycle": collect,
        "update_seconds_per_cycle": update,
        "minibatches_per_update": minibatches * agent_count,
        "total_hours": total / 3600,
        "sessions": math.ceil(total / 3600 / SESSION_HOURS),
        "hours_per_10M_transitions": total / transitions * 1e7 / 3600,
        "update_check": {
            "predicted": predicted_measured,
            "measured": result["learning_seconds"],
        },
        "agents_storage_mb_at_window": result["memory"]["agents_storage_mb"]
        * window_steps
        / measured_window,
    }
    out(
        f"\nProjection of the configured run: {cycles} cycles of {window_steps} steps,"
        f" {transitions:,} transitions, minibatch {ppo.minibatch_size}"
    )
    out(
        f"  update model check at the measured window: predicted"
        f" {predicted_measured:.2f} s, measured {result['learning_seconds']:.2f} s"
    )
    out(
        f"  per cycle: collecting {collect / 60:6.1f} min,"
        f" learning {update / 60:6.1f} min ({minibatches * agent_count:,} minibatches)"
    )
    out(
        f"  total {total / 3600:.1f} h"
        f" ({projection['hours_per_10M_transitions']:.1f} h per 10M transitions),"
        f" {projection['sessions']} session(s) of {SESSION_HOURS} h"
    )
    # The update at this window for every minibatch size measured.
    update_by_size = {}
    for size, minibatch in learning["minibatch"].items():
        count = ppo.update_epochs * math.ceil(world_count * window_steps / size)
        update_by_size[size] = agent_count * (
            learning["advantage_seconds_per_row"] * window_steps
            + count * minibatch["full_ms"] / 1000
        )
    projection["update_seconds_by_minibatch_size"] = update_by_size
    out(
        "  update per cycle by minibatch size: "
        + ", ".join(
            f"{size:,}: {seconds / 60:.1f} min"
            if seconds >= 60
            else f"{size:,}: {seconds:.0f} s"
            for size, seconds in update_by_size.items()
        )
    )
    out(
        f"  agents' stored data at this window: "
        f"{projection['agents_storage_mb_at_window']:,.0f} MB"
    )
    return projection


# -- Main -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configuration", type=Path, help="a training file")
    parser.add_argument(
        "--worlds",
        type=int,
        nargs="*",
        help="world counts to measure; default: the file's",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=8,
        help="steps of each measured window (default 8)",
    )
    parser.add_argument(
        "--minibatch-sizes", type=int, nargs="*", default=[64, 1024, 4096, 16384, 65536]
    )
    parser.add_argument("--gpu-solver", choices=("newton", "cg"))
    parser.add_argument(
        "--output", type=Path, help="folder for the report, the numbers, and the traces"
    )
    arguments = parser.parse_args()

    configuration = read_configuration(arguments.configuration)
    if configuration.mode != "train" or configuration.continue_from is not None:
        parser.error("the profile needs a training file of a new run")
    world_counts = arguments.worlds or [
        configuration.environment.simulation.world_count
    ]
    if arguments.output is not None:
        arguments.output.mkdir(parents=True, exist_ok=True)

    out = Output()
    simulation = configuration.environment.simulation
    out(f"Training profile of {arguments.configuration}")
    out(
        f"  backend {simulation.backend}, agents on {configuration.agents.device},"
        f" solver {arguments.gpu_solver or simulation.gpu_solver}"
    )
    if configuration.agents.device == "cuda":
        out(
            f"  GPU {torch.cuda.get_device_name()}, {torch.cuda.device_count()} visible"
        )
    out(
        f"  CPU {platform.processor() or platform.machine()},"
        f" {torch.get_num_threads()} PyTorch threads"
    )
    out(
        f"  measured windows of {arguments.window_steps} steps; the projection uses"
        f" the file's {configuration.interaction_loop.rollout_window_steps}"
    )

    results = []
    for world_count in world_counts:
        results.append(
            profile_world_count(
                configuration,
                arguments.configuration,
                world_count,
                arguments,
                arguments.output,
                out,
            )
        )

    out(f"\n{RULE}\nSummary\n{RULE}")
    out(
        f"  {'worlds':>7} {'steps/s':>9} {'collect min':>12} {'learn min':>10}"
        f" {'total h':>8} {'h/10M':>7} {'sessions':>9}"
    )
    for result in results:
        projection = result["projection"]
        out(
            f"  {result['worlds']:>7} {result['transitions_per_second']:>9,.0f}"
            f" {projection['collect_seconds_per_cycle'] / 60:>12.1f}"
            f" {projection['update_seconds_per_cycle'] / 60:>10.1f}"
            f" {projection['total_hours']:>8.1f}"
            f" {projection['hours_per_10M_transitions']:>7.1f}"
            f" {projection['sessions']:>9}"
        )

    if arguments.output is not None:
        (arguments.output / "report.txt").write_text("\n".join(out.lines), "utf-8")
        (arguments.output / "profile.json").write_text(
            json.dumps(results, indent=1, default=str), "utf-8"
        )
        out(f"\nSaved in {arguments.output}")


if __name__ == "__main__":
    main()
