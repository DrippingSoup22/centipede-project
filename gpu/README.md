# Centipede GPU implementation

The simulation loads the shared XML, validates its mappings, allocates batched
MuJoCo Warp state, and implements batched leg-control stepping, selective
resets, and physical-state extraction into `physical_state`. Motion validation
is blocked on the local MX330 by an MJWarp collision-kernel compilation error.
The task environment and training remain to be implemented.

This implementation will have its own source, tests, configuration, tools, and
virtual environment. It will use the shared root-level models and scientific
documentation, and reuse generic reinforcement-learning functionality from
`RL_lib`. It must not require importing the CPU implementation to run.

See the [GPU development plan](plan.md) for the architecture and stages, starting
at Stage 1. The [previous plan](../cpu/plan.md) remains the CPU history and
migration reference.

## Local simulation environment

Use `$HOME\.venvs\Centipede-GPU` for GPU development. The existing
`Centipede` environment remains dedicated to the CPU implementation. Neither
environment belongs inside the repository.

The local MX330 uses NVIDIA's CUDA 12 compatibility wheel. This setup does not
require installing a system CUDA toolkit or changing the project architecture.
These are local simulation dependencies, not a Kaggle training environment.

To recreate the setup in PowerShell:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m venv "$HOME\.venvs\Centipede-GPU"
& "$HOME\.venvs\Centipede-GPU\Scripts\python.exe" -m pip install "https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-win_amd64.whl" "mujoco-warp==3.14.0" "mujoco==3.12.0" "numpy==2.5.3" "ruff==0.16.8"
& "$HOME\.venvs\Centipede-GPU\Scripts\python.exe" -m pip check
& "$HOME\.venvs\Centipede-GPU\Scripts\python.exe" -m pip install "pytest==9.1.1"
```

In Zed, open `gpu/` as its own project to keep its interpreter selection separate
from `cpu/`. Run **toolchain: select**, choose **Add toolchain** if necessary, and
select `$HOME\.venvs\Centipede-GPU\Scripts\python.exe`.
This is an editor selection, not a source-code change. Reopen the terminal after
selecting it so new commands use the new environment.

References: [Warp compatibility](https://github.com/NVIDIA/warp/blob/v1.17.0/docs/user_guide/compatibility.rst)
and [Zed toolchains](https://zed.dev/docs/toolchains).

### Setup verification

On 2026-09-23, dependency checks passed and the constructor allocated one world's
model and state on the MX330 (`cuda:0`, compute capability 6.1, 2 GiB), using
driver 576.83. The GPU position array had shape `(1, 69)`.

The subsequent stepping test failed before integration: MJWarp's convex collision
kernel requires 4,292 bytes of parameters, exceeding this GPU's 4,096-byte limit.
The larger CUDA parameter allowance requires Volta or newer hardware; changing
world count or contact capacity does not change this kernel-signature limit.
See [NVIDIA's parameter-limit documentation](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/).
Allocation is verified, but actual centipede motion is not. The shared XML and
collisions remain unchanged. A newer GPU is needed to attempt this same motion
check with the installed library versions; no training has been run.

MJWarp warned that capsule-mesh collision pairs do not support multiple contact
points and will generate at most one contact per pair. The XML was not changed.
Check the resulting contact behavior during physics validation before treating
the GPU simulation as equivalent to the CPU baseline.

## Simulation tests

From the repository root, run:

```powershell
& "$HOME\.venvs\Centipede-GPU\Scripts\python.exe" -m pytest -c gpu/pytest.ini gpu/tests -q
```

Eleven tests cover model dimensions and timing, actuator/joint mappings, real
CUDA allocation of the MJWarp state and `physical_state`, control placement,
two-world physics stepping, selective reset, rejection of invalid physical
results, extraction, contact classification, and a forward-pass comparison with
CPU MuJoCo.

Both `mjw.step` and `mjw.forward` run MJWarp's collision code, and every `step()`
and `reset()` ends with a forward pass. The locally supported tests therefore
replace both with call counters and exercise the real project kernels and
`mjw.reset_data` on the MX330; they are not evidence of working physics:

- Reset checks cover selected-world state clearing, leg-only noise, repeatable
  independent random sequences, and empty selections.
- The extraction check lets CPU MuJoCo compute positions, orientations, and
  velocities for two scrambled worlds, copies them into MJWarp's data, and
  requires the kernel output to match the CPU fields and `mj_objectVelocity`.
- The contact check classifies a hand-written contact pool covering every flag
  rule, a repeated contact, and a stale entry beyond `nacon`.

Two tests carry the `physics` marker: two-world stepping, and a comparison of
MJWarp's refreshed state and contact pairs with CPU MuJoCo at one settled pose.

Constructor arguments and runtime inputs are trusted, so no test supplies invalid
arguments. Configuration will be validated where it is loaded. The tests use
capacities of 128 contacts and 512 constraint rows per world, above the CPU
reference peak of 40 contacts and 256 rows measured at rest over one simulated
second; these are test headroom, not tuned training capacities.

On 2026-09-24, the local MX330 supported subset had **seven passed and one
deselected**. The physics test was rerun and still fails only with the compiler
limit above. It remains an ordinary test, not marked as expected to fail. The
capsule-mesh warning remains visible. No test trains a policy.

On 2026-09-29, after extraction was added, the supported subset had **nine
passed and two deselected**. The two `physics` tests have not yet run on a
compatible GPU.

Local development continues with the supported tests. To run that subset:

```powershell
& "$HOME\.venvs\Centipede-GPU\Scripts\python.exe" -m pytest -c gpu/pytest.ini gpu/tests -q -m "not physics"
```

This intentionally deselects the tests that need MJWarp's collision code; it does
not validate physics. Run the full suite on a compatible Kaggle GPU before substantial
training. The [plan](plan.md#local-development-and-validation) records this
validation split and the next implementation step.

### Kernel review

The project kernels place leg commands, initialize random states, add reset
noise, flag worlds with non-finite state, extract per-segment physical fields,
and classify contacts. MJWarp's public APIs still own integration, forward
computation, and physical reset. The finiteness, extraction, and contact kernels
follow the same naming conventions but were added after the analyzer review
below and have not yet been analyzed.
The `physics` marker identifies tests that integrate the complete frozen model;
the default test command includes them and still exposes the known local failure.

On 2026-09-24, the simulation passed the official
[MJWarp kernel analyzer](https://github.com/google-deepmind/mujoco_warp/tree/521edd1859ccf80cdb425742de869cc3d183a761/contrib/kernel_analyzer)
using its installed MJWarp `types.py`. The analyzer ran from a temporary review
copy, not a new project dependency or editor extension. Its checks cover kernel
conventions, not numerical accuracy or a general proof of race freedom.

The review aligned input/output names, parameter groups, and scalar type aliases
with MJWarp conventions. Each actuator destination is unique after XML mapping
validation; each reset thread owns one world's coordinates and random state.
Warp's `rand_init(seed, world_id)` initializes both default and explicitly seeded
streams; updated states are saved for the next reset. No kernel needs gradients,
so backward code generation is disabled explicitly. See
[Warp 1.17 runtime guidance](https://nvidia.github.io/warp/v1.17/user_guide/runtime.html)
and [kernel settings](https://nvidia.github.io/warp/v1.17/user_guide/configuration.html).

The seven supported tests also passed with Warp debug compilation and CUDA
error verification enabled. On 2026-09-29 the nine supported tests, including
extraction and contact classification, passed again in that mode, whose array
bounds checking was confirmed to stop a deliberate out-of-range read. These
checks validate the exercised operations on the MX330, not full motion, real
contacts, or performance on Kaggle.

Each extraction thread writes only its own world and segment. Contact threads
may write the same flag, but only the value `True`, so their order does not
matter. The contact pool is launched at its fixed capacity, and threads beyond
`nacon` exit immediately.
