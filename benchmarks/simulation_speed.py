"""Speed of the physics simulation on the CPU and GPU backends.

A transition is one 20 ms step of one world: what each of the eight segment
agents turns into one learning sample. This script measures how many
transitions per second each backend produces with random actions, the way the
interaction loop will drive it, and estimates how long a training budget would
take in physics alone (learning updates come on top).

For the GPU it also times the 200 physics steps of a transition twice: launched
from Python as usual, and replayed as one recorded CUDA graph. Outside a graph,
MuJoCo Warp's solver copies its "still solving" flag to the CPU after every
iteration, so the GPU idles while the CPU reads it and launches the next one.
The difference between the two times is that waiting, and shows what recording
the steps as a graph would gain.

Run from the repository root, for example on a Kaggle T4:

    python benchmarks/simulation_speed.py

and locally on a GPU older than Volta, with fewer and smaller runs:

    python benchmarks/simulation_speed.py --gpu-solver cg --gpu-worlds 64 \\
        --transitions 2
"""

import argparse
import gc
import time

import torch

from centipede.environment.simulation import PhysicsSimulation, SimulationSettings
from centipede.environment.simulation.constants import PHYSICS_STEPS_PER_ACTION

MODEL_PATH = "models/assembly_v2.xml"
TRAINING_BUDGETS = (10_000_000, 100_000_000)  # transitions


def make_simulation(backend: str, world_count: int, gpu_solver: str):
    settings = SimulationSettings.from_section(
        {
            "model_path": MODEL_PATH,
            "backend": backend,
            "world_count": world_count,
            "gpu_solver": gpu_solver,
        }
    )
    return PhysicsSimulation(settings)


def random_actions(simulation, generator, device) -> torch.Tensor:
    shape = (simulation.world_count, simulation.segment_count, 6)
    return torch.rand(shape, generator=generator, device=device) * 2 - 1


def seconds_per_transition(simulation, transitions, generator, device) -> float:
    """Average wall time of one ``step`` for all worlds, with new actions each."""
    synchronize(device)
    start = time.perf_counter()
    for _ in range(transitions):
        simulation.step(random_actions(simulation, generator, device))
    synchronize(device)
    return (time.perf_counter() - start) / transitions


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def gpu_time_split(simulation, repeats) -> dict:
    """Time of 200 physics steps launched from Python and replayed as a graph.

    Reaches into the backend on purpose: this is a diagnostic, not pipeline
    code. The extra steps continue the same worlds with their last actions.
    """
    import mujoco_warp as mjw
    import warp as wp

    backend = simulation._backend
    gpu_model, gpu_data = backend.gpu_model, backend.gpu_data

    wp.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mjw.step(gpu_model, gpu_data)
    wp.synchronize()
    split = {"python": (time.perf_counter() - start) / repeats, "graph": None}

    try:
        with wp.ScopedCapture(device=backend.device) as capture:
            for _ in range(PHYSICS_STEPS_PER_ACTION):
                mjw.step(gpu_model, gpu_data)
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        split["graph"] = (time.perf_counter() - start) / repeats
    except Exception as error:  # report and keep benchmarking
        print(f"    graph capture failed: {error}")
    backend._check_worlds()
    return split


def benchmark(backend, world_count, arguments, generator) -> dict | None:
    device = "cuda" if backend == "gpu" else "cpu"
    if backend == "gpu":
        import warp as wp

        memory_before = wp.get_mempool_used_mem_current("cuda:0")
    try:
        simulation = make_simulation(backend, world_count, arguments.gpu_solver)
        simulation.reset(seed=0)

        start = time.perf_counter()
        simulation.step(random_actions(simulation, generator, device))
        synchronize(device)
        first_step = time.perf_counter() - start

        if backend == "gpu":  # kernels compile and caches fill on the first steps
            seconds_per_transition(simulation, 2, generator, device)
        step_seconds = seconds_per_transition(
            simulation, arguments.transitions, generator, device
        )
        result = {
            "backend": backend,
            "worlds": world_count,
            "first_step": first_step,
            "step": step_seconds,
            "rate": world_count / step_seconds,
        }
        if backend == "gpu":
            result["memory_mb"] = (
                wp.get_mempool_used_mem_current("cuda:0") - memory_before
            ) / 2**20
            result["split"] = gpu_time_split(simulation, repeats=3)
        del simulation
        return result
    except Exception as error:  # an overflow or out-of-memory ends one run only
        print(f"  {backend} with {world_count} worlds failed: {error}")
        return None
    finally:
        gc.collect()


def report(results) -> None:
    print("\nThroughput (random actions, after warm-up)")
    print(
        f"{'backend':>8} {'worlds':>7} {'s/step':>8} {'transitions/s':>14} "
        f"{'GPU MB':>7} {'first step s':>13}"
    )
    for result in results:
        memory = f"{result['memory_mb']:.0f}" if "memory_mb" in result else "-"
        print(
            f"{result['backend']:>8} {result['worlds']:>7} {result['step']:>8.3f} "
            f"{result['rate']:>14,.0f} {memory:>7} {result['first_step']:>13.1f}"
        )

    gpu_results = [result for result in results if "split" in result]
    if gpu_results:
        print("\nGPU: 200 physics steps launched from Python, and as a graph")
        print(
            f"{'worlds':>7} {'python s':>9} {'graph s':>8} {'waiting s':>10} "
            f"{'waiting share':>14} {'graph speed-up':>15}"
        )
        for result in gpu_results:
            python, graph = result["split"]["python"], result["split"]["graph"]
            if graph is None:
                print(f"{result['worlds']:>7} {python:>9.3f} {'failed':>8}")
                continue
            print(
                f"{result['worlds']:>7} {python:>9.3f} {graph:>8.3f} "
                f"{python - graph:>10.3f} {(python - graph) / python:>14.0%} "
                f"{python / graph:>14.1f}x"
            )

    print("\nPhysics time for a training budget, at each measured speed")
    header = "".join(f"{budget / 1e6:>10.0f}M" for budget in TRAINING_BUDGETS)
    print(f"{'backend':>8} {'worlds':>7}{header}   (hours)")
    for result in results:
        hours = "".join(
            f"{budget / result['rate'] / 3600:>11.1f}" for budget in TRAINING_BUDGETS
        )
        print(f"{result['backend']:>8} {result['worlds']:>7}{hours}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cpu-worlds", type=int, nargs="*", default=[1, 16])
    parser.add_argument(
        "--gpu-worlds", type=int, nargs="*", default=[64, 256, 1024, 4096]
    )
    parser.add_argument("--gpu-solver", choices=("newton", "cg"), default="newton")
    parser.add_argument("--transitions", type=int, default=10)
    arguments = parser.parse_args()

    generator_cpu = torch.Generator().manual_seed(0)
    results = []
    for world_count in arguments.cpu_worlds:
        print(f"CPU, {world_count} worlds ...", flush=True)
        results.append(benchmark("cpu", world_count, arguments, generator_cpu))
    if arguments.gpu_worlds:
        print(f"GPU: {torch.cuda.get_device_name()}, {arguments.gpu_solver} solver")
        generator_gpu = torch.Generator(device="cuda").manual_seed(0)
        for world_count in arguments.gpu_worlds:
            print(f"GPU, {world_count} worlds ...", flush=True)
            results.append(benchmark("gpu", world_count, arguments, generator_gpu))
    report([result for result in results if result is not None])


if __name__ == "__main__":
    main()
