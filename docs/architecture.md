# Project architecture

Centipede trains a simulated centipede to walk. Each of its eight body segments
is an independent reinforcement-learning agent, and the segments cooperate only
through the shared body.

The program follows the basic model of reinforcement learning: **agents** choose
actions, an **environment** responds with observations and rewards, and the two
repeat this exchange. It is built as a pipeline of independent components. Each
component has one clear job, uses only the component below it, and offers a
small set of operations to the component above.

You work only at the two ends of the pipeline. At the top, you run one command
with a TOML configuration file that holds every setting of a run. At the bottom,
the model files describe the body; the configuration says which one to load.
Everything in between runs on its own.

## Layers

![The four layers of the pipeline](images/architecture-layers.svg)

**Layer 4: Experiment.** Turns your command into a run: reads the
configuration, starts training or evaluation, and saves the results. It is the
only layer that deals with files.

**Layer 3: Interaction loop.** Makes the agents and the environment
communicate, one step at a time, and decides when the agents learn from what
they have collected.

**Layer 2: Agents and environment.** The two sides of the reinforcement-learning
problem, side by side. The environment is the problem; the agents try to solve
it. They never talk to each other directly.

**Layer 1: Physics simulation.** Moves the body. It runs on the CPU or the GPU,
and the layers above cannot tell which. Only the environment uses it.

## Components

![Components of the pipeline](images/architecture-components.svg)

A box inside another box is an internal part of that component. Blue boxes are
the parts the layer above talks to: the front of each component. Dashed boxes
are outside the pipeline. Arrows point from a component to the one it calls.
What passes along each connection:

| Connection | Going down | Coming back |
| --- | --- | --- |
| Experiment ↔ Interaction loop | Settings | Agent state for checkpoints |
| Interaction loop ↔ Agents | Observations, rewards, episode ends | Joint action |
| Interaction loop ↔ Environment | Joint action | Observations, rewards, episode ends |
| Environment ↔ Physics simulation | Leg actions, reset requests | Physical state |

Each component is one folder under `src/centipede/`. In every folder, the file
named like the folder is the **front file**: it offers the component's operations
to the layer above and manages the other files, which are internal. Each
component owns and checks its own section of the configuration file, using the
shared helper `src/centipede/settings_section.py` so that the checks are written
once. Data passed
between components is PyTorch tensors; inside a component, NumPy may be used
where simpler.

### Experiment

The entry point of the program. It reads the TOML configuration, hands each
component its section, runs training or evaluation through the interaction loop,
and saves what the run produces. Each run gets its own folder under `runs/`,
holding the configuration it used, its logs, and its checkpoints. The logs and
the run's report are built from the [diagnostics](diagnostics.md) that every
component measures about its own work.

Every configuration file has a `mode`. **Train** files start a new run,
possibly with the agents of an earlier run (`start_from`, for training in
stages), or continue an earlier run (`continue_from`); smoke tests, probe
tests, and full training use the same code and differ only in their values.
**Evaluate** files test a saved run on fixed seeds, optionally alongside
zero-action and random-action baselines, and write the results into that run's
folder. Training
writes one log line per window and a report of the whole run; the layout of the
files and of the run folder is described in [configuration.md](configuration.md).
Recorded windows are written as replay files for the sibling `../MujocoReplay`
project, a general viewer for recorded MuJoCo poses that knows nothing about
this project; Centipede uses its package only to write the files.

The experiment uses the other components only through their front files: it
creates the environment, the agents, and the interaction loop, steps through
`train()` one cycle at a time, and reads the diagnostics between cycles. It
never looks inside them, so a new kind of run needs only a new configuration
file, or a new function next to `train` and `evaluate`.

**Files:**

| File | Covers |
| --- | --- |
| `experiment.py` | Front file: `run(path)` trains or evaluates as the file's mode says |
| `configuration.py` | Reading the three kinds of file, the checks across sections, and saving the complete configuration |
| `run_folder.py` | A run's folder: creating it, the log, checkpoints, session facts, recordings, and evaluation results |
| `report.py`, `report_page.html` | The training and evaluation reports: the data, and the page that draws it |
| `recordings.py` | Turning a recorded window into a replay recording for the sibling MujocoReplay project |

The command `python -m centipede CONFIG.toml` lives in `src/centipede/__main__.py`.
Reusable configuration files live in `configs/`.

### Interaction loop

The exchange between agents and environment. It gives observations to the
agents, passes their joint action to the environment, and returns the results to
the agents. It counts steps, tells the agents to learn at the end of every
collection window (which never resets the environment), and measures the time
spent collecting and learning. It calculates nothing itself.

`train(seed)` resets the environment once with the seed, then runs
`update_cycles` cycles. Each cycle collects `rollout_window_steps` steps in every
world (the agents act, the environment steps, the agents record) and ends with
one update. It hands control back to the experiment after every cycle, so the
experiment can write the log and save checkpoints between cycles; that time is
not counted as collecting or learning.

`evaluate(actor, seed)` resets the environment with the seed and runs until
every world has finished its first episode. The actor is the trained agents or
one of the baselines; it acts without exploration, and nothing is stored or
learned. Worlds that finish early start new episodes as usual, but only each
world's first episode counts in the diagnostics. The experiment calls it once
per seed and reads the results from the loop's diagnostics.

`diagnostics` also carries the recorder: the experiment arms it before a
window it wants recorded, the loop's diagnostics feed it every step, and the
experiment takes the chosen worlds' poses afterwards
(see [diagnostics.md](diagnostics.md#recordings)).

**Files:** `interaction_loop.py` (front file: `train()` and `evaluate()`),
`settings.py`, which holds `rollout_window_steps` (steps per world before each
update, first value 256) and `update_cycles` (number of collect-and-learn
cycles, first value 128), `diagnostics.py`, which times each window and
summarises the environment's and the simulation's diagnostics over it
(see [diagnostics.md](diagnostics.md#timing)), and `recording.py`, the
recorder.

### Agents

One segment agent per body segment (eight in v1), managed as one group. Each
has its own networks, optimizers, observation normaliser, and stored data, and
learns only from its own data with PPO from the shared `RL_lib` library. The
group splits the observations among its segment agents and joins their actions
into one joint action.

**Details:** [agents.md](agents.md)

### Environment

The problem the agents must solve. `reset` starts new episodes, and `step` takes a
joint action and returns the next observations, one reward per segment, and which
episodes ended. Its front file keeps the episode state (targets, step counts,
previous positions) and coordinates its parts: the observation builder, the
reward function, its diagnostics, and the physics simulation. It does not store
data for learning.

**Details:** [environment.md](environment.md)

### Physics simulation

Part of the environment, in `environment/simulation/`. It loads the model, finds
each segment's motors and contact shapes by name, applies leg actions, advances
the physics, resets worlds, and reports the physical state. A CPU backend uses
MuJoCo and a GPU backend uses MuJoCo Warp; the configuration chooses one, and
nothing outside the simulation can tell the difference.

**Details:** [environment.md](environment.md#physics-simulation)

### Model

The centipede's body: MuJoCo XML files describing its shape, joints, motors,
masses, and contacts. It contains no program logic and is not part of the
pipeline. It is frozen; the configuration names the file to load.

**Details:** [model.md](model.md)
