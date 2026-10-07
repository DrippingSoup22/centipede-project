"""Speed and soundness of the physics under other solver settings and timesteps.

After the CUDA graph, the GPU's own work per physics step limits training at
1,024 worlds and more, so the remaining speed levers are the physics options:
the solver (Newton or conjugate gradient), its tolerance and line search, the
friction cone, and the timestep. Except for the solver, they belong to the
frozen model, so any change would become a new model version (see
docs/model.md). This script measures what each would gain, and whether the
physics stays sound, before any such decision.

It drives MuJoCo Warp directly, outside the pipeline, and changes the model's
options in memory only: the model file is never changed. Every variant holds
each action for 20 ms, as the environment does, so a different timestep means
a different number of physics steps per action. For each variant it reports:

- the time of one action at ``--speed-worlds`` worlds, launched from Python
  and replayed as a recorded CUDA graph;
- on ``--check-worlds`` worlds, the checks of docs/model.md: the deepest
  contact penetration and the largest joint-limit excess under random
  full-strength leg commands (each world its own), the solver's iterations and
  whether it hit its limit, the largest number of constraint rows, and any
  failure;
- settling with motors off from the model's pose: how far the bodies end, after
  the same simulated time, from where the model as it is puts them;
- under random commands, the mean height and speed of the root body, as ratios
  to the model as it is. Flailing is chaotic, so single trajectories cannot be
  compared; these averages can.

A note on timesteps: the model's contacts respond in 0.3 ms, and MuJoCo never
lets a constraint respond faster than twice the timestep. Above 0.15 ms the
contacts therefore become softer than the model specifies; the 0.2 ms variant
shows by how much that matters.

Run from the repository root on a Kaggle T4:

    python benchmarks/physics_options.py --output /kaggle/working/physics_options

and locally on a GPU older than Volta, where only the CG solver compiles:

    python benchmarks/physics_options.py --base-solver cg --speed-worlds 16 \\
        --check-worlds 4 --settle-actions 5 --random-actions 10 \\
        --variants reference "timestep 0.2 ms"
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from centipede.environment.simulation.constants import LEG_ANGLE_NOISE_RAD
from centipede.environment.simulation.model_mapping import ModelMapping

MODEL_PATH = "models/assembly_v2.xml"
ACTION_SECONDS = 0.020
# The pipeline's reserved memory per world (docs/environment.md, Settings).
CONTACTS_PER_WORLD = 128
CONSTRAINTS_PER_WORLD = 512
SOLVERS = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}
CONES = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
CAPACITY_OVERFLOWS = (
    mjw.OverflowType.NEFC
    | mjw.OverflowType.NJMAX_NNZ
    | mjw.OverflowType.BROADPHASE
    | mjw.OverflowType.NARROWPHASE
)

# Each variant changes some of the model's options; the rest stay as they are.
# "substeps" is the number of physics steps per 20 ms action.
VARIANTS: dict[str, dict[str, Any]] = {
    "reference": {},
    "cg": {"solver": "cg"},
    "tolerance 1e-5": {"tolerance": 1e-5},
    "tolerance 1e-4": {"tolerance": 1e-4},
    "line search 10": {"ls_iterations": 10},
    "pyramidal cone": {"cone": "pyramidal"},
    "timestep 0.125 ms": {"substeps": 160},
    "timestep 0.149 ms": {"substeps": 134},
    "timestep 0.2 ms": {"substeps": 100},
    "0.149 ms, tolerance 1e-5": {"substeps": 134, "tolerance": 1e-5},
}


class Variant:
    """One set of physics options, on the GPU, for a given number of worlds."""

    def __init__(self, options: dict[str, Any], base_solver: str, world_count: int):
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.reference_timestep = model.opt.timestep
        self.substeps = options.get(
            "substeps", round(ACTION_SECONDS / model.opt.timestep)
        )
        model.opt.timestep = ACTION_SECONDS / self.substeps
        model.opt.solver = SOLVERS[options.get("solver", base_solver)]
        model.opt.tolerance = options.get("tolerance", model.opt.tolerance)
        model.opt.ls_iterations = options.get("ls_iterations", model.opt.ls_iterations)
        model.opt.cone = CONES[options.get("cone", "elliptic")]
        self.model = model
        self.mapping = ModelMapping.from_model(model)
        self.gpu_model = mjw.put_model(model)
        # The overflow bits are read here, so MuJoCo Warp need not print them.
        self.gpu_model.opt.warn_overflow = 0
        self.data = mjw.make_data(
            model,
            nworld=world_count,
            nconmax=CONTACTS_PER_WORLD,
            njmax=CONSTRAINTS_PER_WORLD,
        )
        self.world_count = world_count
        self.device = wp.get_device()
        self.torch_device = wp.device_to_torch(self.device)
        self.leg_motor_ids = torch.as_tensor(
            self.mapping.leg_actuator_ids.ravel(), device=self.torch_device
        )
        self.graph = None

    def reset(self, leg_noise_seed: int | None) -> None:
        """All worlds to the model's pose, with each its own leg angles if seeded."""
        mjw.reset_data(self.gpu_model, self.data)
        if leg_noise_seed is not None:
            generator = np.random.default_rng(leg_noise_seed)
            addresses = self.mapping.leg_qpos_addresses.ravel()
            noise = generator.uniform(
                -LEG_ANGLE_NOISE_RAD,
                LEG_ANGLE_NOISE_RAD,
                (self.world_count, addresses.size),
            )
            qpos = wp.to_torch(self.data.qpos)
            qpos[:, torch.as_tensor(addresses, device=self.torch_device)] += (
                torch.as_tensor(noise, dtype=qpos.dtype, device=self.torch_device)
            )
        mjw.forward(self.gpu_model, self.data)

    def set_leg_commands(self, commands: torch.Tensor | None) -> None:
        """Every world's leg motor commands, (W, 48); None turns all motors off."""
        ctrl = wp.to_torch(self.data.ctrl)
        ctrl.zero_()
        if commands is not None:
            ctrl[:, self.leg_motor_ids] = commands

    def python_action(self) -> None:
        """One 20 ms action, each physics step launched from Python."""
        for _ in range(self.substeps):
            mjw.step(self.gpu_model, self.data)

    def graph_action(self) -> None:
        """One 20 ms action, replayed from a CUDA graph recorded on first use."""
        if self.graph is None:
            with wp.ScopedCapture(device=self.device) as capture:
                self.python_action()
            self.graph = capture.graph
        wp.capture_launch(self.graph)

    def failure(self) -> str | None:
        """A capacity overflow or a non-finite state since the last reset."""
        overflow = wp.to_torch(self.data.overflow)
        if bool(((overflow & int(CAPACITY_OVERFLOWS)) != 0).any()):
            return "reserved contact or constraint memory overflowed"
        for name in ("qpos", "qvel"):
            if not bool(torch.isfinite(wp.to_torch(getattr(self.data, name))).all()):
                return f"non-finite {name}"
        return None

    def hit_iteration_limit(self) -> bool:
        overflow = wp.to_torch(self.data.overflow)
        return bool(((overflow & int(mjw.OverflowType.ITERATIONS)) != 0).any())


def joint_limit_table(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """The qpos addresses and ranges of the limited hinge joints."""
    limited = (model.jnt_limited == 1) & (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    return model.jnt_qposadr[limited], model.jnt_range[limited]


def measure_speed(variant: Variant, actions: int) -> dict[str, Any]:
    """Seconds per action launched from Python and as a graph."""
    variant.reset(leg_noise_seed=0)
    generator = torch.Generator(device=variant.torch_device).manual_seed(0)

    def commands() -> torch.Tensor:
        shape = (variant.world_count, variant.leg_motor_ids.numel())
        return (
            torch.rand(shape, generator=generator, device=variant.torch_device) * 2 - 1
        )

    variant.set_leg_commands(commands())
    variant.python_action()  # compiles kernels and fills caches
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(actions):
        variant.set_leg_commands(commands())
        variant.python_action()
    wp.synchronize()
    result: dict[str, Any] = {"python_s": (time.perf_counter() - start) / actions}
    try:
        variant.graph_action()  # records the graph
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(actions):
            variant.set_leg_commands(commands())
            variant.graph_action()
        wp.synchronize()
        result["graph_s"] = (time.perf_counter() - start) / actions
    except Exception as error:  # report it and keep measuring
        result["graph_s"] = None
        result["graph_error"] = str(error)
    result["failure"] = variant.failure()
    return result


def check_physics(
    variant: Variant, settle_actions: int, random_actions: int
) -> dict[str, Any]:
    """The docs/model.md checks, settling, and averages under random commands."""
    run = variant.graph_action
    try:
        variant.reset(leg_noise_seed=None)
        run()
    except Exception:  # no graph: launch from Python
        run = variant.python_action

    # Settling with motors off, from the model's exact pose.
    variant.reset(leg_noise_seed=None)
    variant.set_leg_commands(None)
    for _ in range(settle_actions):
        run()
    settled_body_positions = wp.to_torch(variant.data.xpos)[0].cpu().numpy().copy()
    settle_failure = variant.failure()

    # Random full-strength leg commands, each world its own, from noisy poses.
    variant.reset(leg_noise_seed=1)
    limit_addresses, limit_ranges = joint_limit_table(variant.model)
    addresses = torch.as_tensor(limit_addresses, device=variant.torch_device)
    lower = torch.as_tensor(limit_ranges[:, 0], device=variant.torch_device)
    upper = torch.as_tensor(limit_ranges[:, 1], device=variant.torch_device)
    generator = torch.Generator(device=variant.torch_device).manual_seed(1)
    deepest_penetration = 0.0
    largest_limit_excess = 0.0
    most_constraint_rows = 0
    most_iterations = 0
    root_heights, root_speeds = [], []
    qpos = wp.to_torch(variant.data.qpos)
    previous_root = qpos[:, :2].clone()
    for _ in range(random_actions):
        shape = (variant.world_count, variant.leg_motor_ids.numel())
        variant.set_leg_commands(
            torch.rand(shape, generator=generator, device=variant.torch_device) * 2 - 1
        )
        run()
        contact_count = min(
            int(wp.to_torch(variant.data.nacon)[0]), variant.data.naconmax
        )
        if contact_count:
            distances = wp.to_torch(variant.data.contact.dist)[:contact_count]
            deepest_penetration = max(deepest_penetration, float(-distances.min()))
        joint_angles = qpos[:, addresses]
        excess = torch.maximum(lower - joint_angles, joint_angles - upper).clamp_min(0)
        largest_limit_excess = max(largest_limit_excess, float(excess.max()))
        most_constraint_rows = max(
            most_constraint_rows, int(wp.to_torch(variant.data.nefc).max())
        )
        most_iterations = max(
            most_iterations, int(wp.to_torch(variant.data.solver_niter).max())
        )
        root_heights.append(float(qpos[:, 2].mean()))
        root_speeds.append(
            float((qpos[:, :2] - previous_root).norm(dim=1).mean()) / ACTION_SECONDS
        )
        previous_root = qpos[:, :2].clone()
    return {
        "settled_body_positions": settled_body_positions,
        "settle_failure": settle_failure,
        "failure": variant.failure(),
        "hit_iteration_limit": variant.hit_iteration_limit(),
        "deepest_penetration_mm": deepest_penetration * 1000,
        "largest_limit_excess_deg": math.degrees(largest_limit_excess),
        "most_constraint_rows": most_constraint_rows,
        "most_solver_iterations_last_step": most_iterations,
        "mean_root_height_mm": sum(root_heights) / len(root_heights) * 1000,
        "mean_root_speed_mm_s": sum(root_speeds) / len(root_speeds) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--base-solver", choices=tuple(SOLVERS), default="newton")
    parser.add_argument("--speed-worlds", type=int, default=1024)
    parser.add_argument("--speed-actions", type=int, default=3)
    parser.add_argument("--check-worlds", type=int, default=64)
    parser.add_argument("--settle-actions", type=int, default=50, help="1 s")
    parser.add_argument("--random-actions", type=int, default=100, help="2 s")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    unknown = set(arguments.variants) - set(VARIANTS)
    if unknown:
        parser.error(f"unknown variants {sorted(unknown)}; known: {list(VARIANTS)}")
    if "reference" not in arguments.variants:
        arguments.variants.insert(0, "reference")  # every comparison needs it

    print(f"GPU {torch.cuda.get_device_name()}, base solver {arguments.base_solver}")
    print(
        f"speed at {arguments.speed_worlds} worlds; checks at {arguments.check_worlds}"
        f" worlds, {arguments.settle_actions} actions settling and"
        f" {arguments.random_actions} under random commands"
    )
    results: dict[str, dict[str, Any]] = {}
    for name in arguments.variants:
        options = VARIANTS[name]
        print(f"\n{name} ...", flush=True)
        result: dict[str, Any] = {"options": options}
        try:
            variant = Variant(options, arguments.base_solver, arguments.speed_worlds)
            result["substeps"] = variant.substeps
            result["timestep_ms"] = variant.model.opt.timestep * 1000
            result["speed"] = measure_speed(variant, arguments.speed_actions)
            del variant
            gc.collect()
            variant = Variant(options, arguments.base_solver, arguments.check_worlds)
            result["checks"] = check_physics(
                variant, arguments.settle_actions, arguments.random_actions
            )
            del variant
            gc.collect()
        except Exception as error:  # one variant failing ends only that variant
            result["error"] = str(error)
            print(f"  failed: {error}")
        results[name] = result
        print(f"  {json.dumps(summary_row(result, results.get('reference')))}")

    lines = report(results)
    print("\n" + "\n".join(lines))
    if arguments.output is not None:
        arguments.output.mkdir(parents=True, exist_ok=True)
        (arguments.output / "report.txt").write_text("\n".join(lines), "utf-8")
        (arguments.output / "physics_options.json").write_text(
            json.dumps(
                {
                    name: summary_row(result, results["reference"])
                    | {"options": result["options"]}
                    for name, result in results.items()
                },
                indent=1,
            ),
            "utf-8",
        )
        print(f"\nSaved in {arguments.output}")


def summary_row(
    result: dict[str, Any], reference: dict[str, Any] | None
) -> dict[str, Any]:
    """The plain numbers of one variant, with its comparison to the reference."""
    row: dict[str, Any] = {"substeps": result.get("substeps")}
    if "error" in result:
        return row | {"error": result["error"]}
    speed, checks = result.get("speed", {}), result.get("checks", {})
    row |= {
        "python_s": speed.get("python_s"),
        "graph_s": speed.get("graph_s"),
        "speed_failure": speed.get("failure"),
    }
    row |= {
        key: value for key, value in checks.items() if key != "settled_body_positions"
    }
    if reference is not None and "checks" in reference and "checks" in result:
        difference = np.linalg.norm(
            checks["settled_body_positions"]
            - reference["checks"]["settled_body_positions"],
            axis=1,
        )
        row["settled_difference_mm"] = float(difference.max() * 1000)
        for key in ("mean_root_height_mm", "mean_root_speed_mm_s"):
            row[key.replace("mean_", "ratio_")] = (
                checks[key] / reference["checks"][key]
                if reference["checks"][key]
                else None
            )
        reference_time = reference["speed"].get("graph_s") or reference["speed"].get(
            "python_s"
        )
        own_time = speed.get("graph_s") or speed.get("python_s")
        if reference_time and own_time:
            row["speed_up"] = reference_time / own_time
    return row


def report(results: dict[str, dict[str, Any]]) -> list[str]:
    """Two tables: speed, and soundness compared with the model as it is."""
    reference = results["reference"]
    lines = ["Speed of one 20 ms action (graph: replayed CUDA graph)"]
    lines.append(
        f"  {'variant':<26} {'steps':>5} {'python s':>9} {'graph s':>8} {'gain':>6}"
    )
    rows = {name: summary_row(result, reference) for name, result in results.items()}
    for name, row in rows.items():
        if "error" in row:
            lines.append(f"  {name:<26} failed: {row['error']}")
            continue

        def seconds(value):
            return f"{value:.3f}" if value else "-"

        gain = f"{row['speed_up']:.2f}x" if row.get("speed_up") else "-"
        lines.append(
            f"  {name:<26} {row['substeps']:>5} {seconds(row['python_s']):>9}"
            f" {seconds(row['graph_s']):>8} {gain:>6}"
        )
    lines.append("")
    lines.append("Soundness (docs/model.md checks; ratios to the model as it is)")
    lines.append(
        f"  {'variant':<26} {'penetr. mm':>10} {'limit deg':>9} {'settle mm':>9}"
        f" {'height':>7} {'speed':>7} {'rows':>5} {'iter':>5} {'limit hit':>9}"
        "  failure"
    )
    for name, row in rows.items():
        if "error" in row or "deepest_penetration_mm" not in row:
            continue

        def ratio(value):
            return f"{value:.2f}" if value is not None else "-"

        failure = (
            row.get("failure") or row.get("settle_failure") or row.get("speed_failure")
        )
        lines.append(
            f"  {name:<26} {row['deepest_penetration_mm']:>10.3f}"
            f" {row['largest_limit_excess_deg']:>9.2f}"
            f" {row.get('settled_difference_mm', 0.0):>9.3f}"
            f" {ratio(row.get('ratio_root_height_mm')):>7}"
            f" {ratio(row.get('ratio_root_speed_mm_s')):>7}"
            f" {row['most_constraint_rows']:>5}"
            f" {row['most_solver_iterations_last_step']:>5}"
            f" {'yes' if row['hit_iteration_limit'] else 'no':>9}"
            f"  {failure or 'none'}"
        )
    lines.append("")
    lines.append(
        "The model's own validation (docs/model.md): 0.176 mm deepest penetration and"
        " 4.23 deg joint-limit excess under random full-strength commands; settling"
        " matched v1 within 0.016 mm."
    )
    return lines


if __name__ == "__main__":
    main()
