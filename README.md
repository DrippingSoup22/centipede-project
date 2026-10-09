# Centipede

A reinforcement-learning project in which a simulated centipede learns to walk.
Each of its eight body segments is an independent agent that controls only its
own pair of legs. The segments never share networks or data: they cooperate only
through the shared body, what each can feel of its neighbours, and a shared
reward when the head reaches a target.

The physics runs in [MuJoCo](https://mujoco.org/), on the CPU or, with MuJoCo
Warp, on the GPU. The learning algorithms come from the sibling `../RL_lib`
library, and recorded runs are watched with the sibling `../MujocoReplay`
viewer.

## Goals

1. Learn to walk and to steer the head toward targets on flat ground.
2. See whether the legs develop a wave-like rhythm that travels along the body,
   as in a real centipede.

Walking ability, navigation, and resemblance to a real gait are evaluated
separately.

## Status

- **Model: complete and frozen.** The current body, v3, is in
  [`models/assembly_v3.xml`](models/assembly_v3.xml) and described in
  [`docs/model.md`](docs/model.md).
- **Design: agreed.** The program structure, the environment, and the agents are
  described in the documents below.
- **Code: Stage 7 complete.** The physics simulation runs batches of worlds on
  the CPU and, with MuJoCo Warp, on the GPU, and is covered by tests that pass
  locally and on a Kaggle T4. The environment builds on it: targets, episodes,
  observations, rewards, and diagnostics, tested locally. The agents, one
  independent PPO learner per segment using RL_lib's batched PPO, act, store
  their data, learn, and save, tested locally; they have not been trained yet.
  The interaction loop connects them for training and evaluation, tested
  locally with a short real run. The experiment runs everything from one
  configuration file and writes each run's log, checkpoints, and an HTML report;
  it is tested locally, and a first CPU probe run was trained, continued, and
  evaluated. Training and evaluation also record the poses of chosen worlds
  for replay in the sibling `../MujocoReplay` project, a viewer for recorded
  MuJoCo poses that is developed separately. Nothing has learned to walk yet. On 2026-10-01 the project was restarted with
  a new structure. A first implementation reached early learning on the CPU,
  with the rear segments learning to avoid ground contact, but no standing or
  walking. It is preserved in Git history (tag `cpu-stage8`) and in the local
  archive.

## Documentation

| Document | Describes |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | How the program is organised: layers, components, and how they connect |
| [`docs/model.md`](docs/model.md) | The centipede's body: dimensions, joints, motors, mass, contacts, physics settings |
| [`docs/configuration.md`](docs/configuration.md) | The TOML files that start every run, and what a run writes: run folder, log, checkpoints, and report |
| [`docs/environment.md`](docs/environment.md) | The task: physics simulation, actions, observations, targets, episodes, rewards |
| [`docs/agents.md`](docs/agents.md) | The learners: independent segment agents, networks, PPO settings, checkpoints |
| [`docs/diagnostics.md`](docs/diagnostics.md) | What each component measures about a run, for the training log and the report |

## Project folders

```text
Centipede/
├─ docs/             Architecture, model, environment, and agents documents
├─ models/           Frozen MuJoCo XML models
├─ src/centipede/    Program code, one folder per component
├─ configs/          TOML files for the smoke test, probe, training, and evaluation
├─ results/          The report's data: small files of the runs it uses, by study
├─ scripts/          Controlling the GPU desktop and Kaggle from the laptop
├─ benchmarks/       Speed measurements, such as the physics simulation's
│  └─ results/       Their saved results, one folder per measurement; local only
├─ tests/            Automated tests, one file per component
└─ runs/             Training and evaluation results; local only
```

`runs/`, `benchmarks/results/`, and `archive/` stay on this computer and are not
part of the repository.
Back up results that must survive separately. The runs the report uses are
kept in `results/`, which is part of the repository, without their recordings
and checkpoints; [`results/README.md`](results/README.md) explains its layout.

## Setup

A fresh Windows computer first needs Python 3.13, Git, and, for the GPU, a
current NVIDIA driver. Install the first two with `winget install
Python.Python.3.13` and `winget install Git.Git` (or from python.org, ticking
"Add python.exe to PATH", and git-scm.com), then open a new PowerShell. Clone
this repository, `RL_lib`, and `MujocoReplay` side by side into one folder,
since the setup installs the other two from `..\RL_lib` and `..\MujocoReplay`.

One Python environment runs everything: the CPU and GPU simulation backends,
the learning code, and the tests. It lives outside the repository. From the
repository root, in PowerShell (`py -3.13` is Windows' Python launcher, which
works even where plain `python` is not found):

```powershell
py -3.13 -m venv "$HOME\.venvs\Centipede"
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ..\RL_lib
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ..\MujocoReplay
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ".[gpu,dev]"
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pybind11_stubgen mujoco -o typings --ignore-all-errors
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pytest
```

MuJoCo's functions are compiled code without type descriptions, so editors
cannot see them. The `pybind11_stubgen` line generates those descriptions into
the ignored `typings/` folder, where the editor's type checker finds them.
In the editor, select `$HOME\.venvs\Centipede\Scripts\python.exe` as the
Python interpreter.

PyTorch is installed first from its CUDA 12.6 build, so that tensors can live on
the GPU; plain `pip install torch` gives a CPU-only build on Windows. This build
still supports older GPUs such as the local MX330 (compute capability 6.1). The
`gpu` extra adds NVIDIA Warp (its CUDA 12 build, for the same reason) and MuJoCo
Warp. Verified on 2026-10-03 with Python 3.13.15, MuJoCo 3.12.0, MuJoCo Warp
3.14.0, Warp 1.17.0, PyTorch 2.14.1+cu126, and NumPy 2.5.3: MuJoCo Warp's GPU
arrays become PyTorch GPU tensors without copying.

Two benchmarks measure speed on a GPU:
[`benchmarks/training_profile.py`](benchmarks/training_profile.py) measures
where the time of a training cycle goes, part by part, and projects how long a
configured run will take; run it with a training file before launching a long
run, for example
`python benchmarks/training_profile.py configs/baseline.toml --worlds 64 256 1024`.
[`benchmarks/physics_options.py`](benchmarks/physics_options.py) compares the
physics' speed and soundness under other solver settings and timesteps.

## Training machine

Training runs on a desktop with an RTX 3080, set up as above, with this
repository, `RL_lib`, and `MujocoReplay` side by side. It collects about 1,250
transitions per second at 1,024 worlds, two and a half times a Kaggle T4. The
laptop, whose MX330 cannot run the default solver, is used to write code, run
the CPU tests, and read the results. While the desktop cannot be reached,
training runs on Kaggle (below).

The laptop controls the desktop over SSH, through
[`scripts/gpu_desktop.py`](scripts/gpu_desktop.py), run from the repository
root with the project's Python:

```powershell
python scripts/gpu_desktop.py run configs/baseline.toml
python scripts/gpu_desktop.py queue configs/overnight
python scripts/gpu_desktop.py watch
python scripts/gpu_desktop.py status
```

`run` makes the desktop pull the pushed code (this project's, RL_lib's and
MujocoReplay's, never while a run or queue is running), sends it the configuration file
as it is on the laptop, starts the run there, independently of the SSH
connection, and shows its output as it is written: one line per window, with
a bar of `#` and `.` that fills as the window is collected. The folder's
`launched/` holds each launch's configuration file and console output, so a
training run and its evaluations keep theirs apart. Ctrl+C stops watching, not
the run. `watch` follows the latest run again (or the one whose name contains
a given text), `fetch` copies a run's folder at any time, `status` lists the
running and recent runs with how each ended and what is being copied, `stop`
stops the running ones, and `tests` runs the tests on the desktop. A second
run is refused while one is running, unless `--alongside` is given, since two
runs share the GPU.

Every training that `run` or `queue` launches is copied into the laptop's
`runs/` as soon as it ends, with its evaluation, however it was started and
whether or not anyone is watching. A copier does it: a process of its own on
the laptop, started with the launch, that goes on when the terminal stops
watching or is closed, and tries again every minute while the desktop cannot
be reached. It logs what it copied in `runs/copies/<launch>.txt`, and the
terminal that watches shows those lines among the output. A copier stopped
before the end, for example by restarting the laptop, is started again by the
next command of the script.

`queue` starts a whole folder of training files, which
[`scripts/run_queue.py`](scripts/run_queue.py) runs on the desktop one after
another, in the order of their names, for example overnight. A run that fails
does not stop the queue. If the folder holds an `evaluation.toml`, each
finished run is evaluated right after it, with that file's `source` replaced
by the run's folder. Every run of the queue is a launched run of its own, so
`status` lists it. Its copier copies each run to the laptop as soon as it and
its evaluation are done, while the queue goes on, and at the end the queue's
output, which ends with a table of how each run ended, to `runs/queues/`;
`fetch queue` copies all its runs at any time. `stop` stops the queue first,
then its running run.

Setting this up once: on the desktop, install Windows' OpenSSH Server, start
the `sshd` service automatically, make Windows PowerShell its default shell,
and set the network profile to Private (the SSH firewall rule covers only
private networks). On the laptop, create a key with `ssh-keygen -t ed25519`,
add its public half to the desktop's
`C:\ProgramData\ssh\administrators_authorized_keys` (the file Windows reads
for administrator accounts), and name the desktop `gpu` in `~/.ssh/config`
with its address and user. The script expects the desktop's projects in
`~\Projects` and its environment in `~\.venvs\Centipede`; another host name
can be given in `CENTIPEDE_GPU_HOST`.

### Training on Kaggle

[`scripts/kaggle_gpu.py`](scripts/kaggle_gpu.py) trains a run and its
evaluation on each of a Kaggle machine's two T4s, in one session of the private notebook
`centipede-training`, launched and followed from the laptop. It uses Kaggle's
command-line tool, installed in the project's environment and signed in once
(the sign-in opens the browser):

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install kaggle==2.2.4
& "$HOME\.venvs\Centipede\Scripts\kaggle.exe" auth login
```

Then, from the repository root with the project's Python:

```powershell
python scripts/kaggle_gpu.py run configs/reward_study/01_baseline.toml --seeds 1 2
python scripts/kaggle_gpu.py run configs/pink_study/01_gae_lambda09.toml configs/pink_study/04_hidden256.toml
python scripts/kaggle_gpu.py watch
python scripts/kaggle_gpu.py status
python scripts/kaggle_gpu.py fetch
```

`run` checks that the code is pushed, since the session clones it at the
laptop's commit, uploads the run that the file starts from as a private
dataset the first time, and sends the configuration file as it is on the
laptop, with the `evaluation.toml` beside it. With `--seeds`, the session
trains the file once per seed at the same time, one run on each of the
machine's two T4s, named with `_seed<N>`; given two files from one folder, it
trains both at the same time, one on each T4, each evaluated with that
folder's `evaluation.toml`. The session,
[`scripts/kaggle_session.py`](scripts/kaggle_session.py), runs the tests
(`--skip-tests` leaves them out), trains, and evaluates each new run. `run`
shows its output as Kaggle streams it, one line per window, marked with its
seed or its file's name, and when it ends copies the run folders into `runs/` and the console
output of every step into
`runs/kaggle/<launch>/`, so Kaggle's web page is not needed. Ctrl+C stops
watching, not the session; `watch` follows it again, `status` tells whether it
is queued, running, or ended, and `fetch` copies its results at any time,
waiting while it runs. Only the latest session is followed or fetched, so a
new one is refused until it has ended. Run folders made on Kaggle are named
by its clock, in UTC.

## Archive

Material that is no longer active is moved to the local `archive/` folder rather
than deleted, when it records a decision, a comparison, or a result. Archived
files are never imported or used by active code. An archived idea that is revived
is rebuilt as a new version and checked against the current baseline.
