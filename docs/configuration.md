# Configuration files

Every run starts from one TOML file and one command:

```text
python -m centipede CONFIG.toml
```

The file is organised like the [architecture](architecture.md): each component
owns one section, named after it, and reads and checks only that section.
Sub-sections are used only for groups of settings that are edited together. The
exact fields may grow or change; this structure stays.

Every setting has a default equal to the first value given in the component's
document, so a file only needs the values it is about. The run folder always
saves the complete configuration with every value filled in, so nothing is
hidden afterwards.

## Sections

| Section | Owned by | Holds |
| --- | --- | --- |
| `[run]` | Experiment | What to do (`mode`), the run's name and seed, checkpoints, the report, and which earlier run to start from, continue, or evaluate |
| `[environment]` | Environment | Episode length and observation radius |
| `[environment.simulation]` | Physics simulation | Model file, backend, number of worlds, GPU memory |
| `[environment.target]` | Environment | Where targets are placed and when the head has arrived |
| `[environment.rewards]` | Environment | Reward and cost values |
| `[agents]` | Agents | Device, networks, learning rate, normalisation |
| `[agents.ppo]` | Agents | PPO settings |
| `[interaction_loop]` | Interaction loop | Steps per window and number of update cycles |
| `[evaluation]` | Experiment | Seeds, episodes per seed, and optional baselines; evaluation files only |

The settings in each section, with their meanings and first values, are listed
in [environment.md](environment.md#settings), [agents.md](agents.md#settings),
and the [architecture](architecture.md) for the interaction loop. The `[run]` and
`[evaluation]` settings are listed below.

## Checks across sections

Each component checks its own section. The experiment adds the checks that
need two sections at once:

- The physics and the agents run on the same kind of device: `backend = "cpu"`
  requires `[agents] device = "cpu"`, and `backend = "gpu"` requires
  `device = "cuda"`. The other two combinations are rejected for now. They may
  be allowed later, for example a single world on the CPU with large networks
  on the GPU; the interaction loop would then move tensors between devices.

## Kinds of file

There are three kinds of file, told apart by their `[run]` section:

| Kind | `[run]` holds | The file contains |
| --- | --- | --- |
| Training | `mode = "train"` and the run's own settings | Every section of a new run; only the values that differ from the defaults need writing |
| Continuing | `mode = "train"` and `continue_from` | Only `[run]`, optionally with the session's `time_limit_hours`, and optionally `[interaction_loop]`, such as a larger `update_cycles` |
| Evaluation | `mode = "evaluate"` and `source` | `[run]`, `[evaluation]`, and optionally `[environment]` sections |

Continuing and evaluation files name an existing run and start from its saved
configuration: their own sections replace only the matching values, and every
result is checked by the components as usual. Any other section is rejected
with an error naming it, so nothing in such a file is silently ignored.

## Training files

`mode = "train"` starts a new run. Smoke tests, probes, the Kaggle levels, and full
training all use this mode and the same code; they differ only in their values, mainly
the number of worlds, episode length, and number of update cycles.

| File | Purpose | Size | Report |
| --- | --- | --- | --- |
| `configs/smoke.toml` | Check that everything runs from start to finish | Tiny, seconds on the CPU | No |
| `configs/probe.toml` | See how a choice of settings behaves on the CPU | Light, about half an hour on the CPU | Yes |
| `configs/baseline.toml` | The first, smallest training run, against which later changes are compared | About 17 minutes on a Kaggle T4 (64 worlds, 8 cycles of 256 steps, 1,024-step episodes), estimated | Yes |
| `configs/kaggle_quick.toml` | Check that a GPU run works and measure its speed | Quick, about 8 minutes on a Kaggle T4 (1,024 worlds, 3 cycles of 64 steps) | Yes |
| `configs/kaggle_medium.toml` | See how a choice of settings starts to behave on the GPU | Medium, about 25 minutes on a Kaggle T4 (256 worlds, 8 cycles of 256 steps) | Yes |
| `configs/training.toml` | The actual experiment | Long, about 19 hours on a Kaggle T4 (1,024 worlds, 128 cycles of 256 steps), in two sessions | Yes |

The Kaggle times were measured with the training profile on a T4 (Stage 8.2 in
`plan.md`): a step takes 0.67 s at 256 worlds and 2.1 s at 1,024, and a fresh
machine adds about a minute compiling the kernels. Two runs side by side, one
per T4, each keep their speed.

The `[run]` settings of a training file:

| Setting | Default | Meaning |
| --- | --- | --- |
| `name` | required | Part of the run folder's name; letters, digits, `-` and `_` |
| `seed` | 0 | Starts every random sequence of the run: the agents' and the environment's |
| `checkpoint_every_cycles` | 16 | A checkpoint is saved after every this many cycles, and always after the last |
| `report` | `true` | Whether to write the run's report; the smoke test turns it off |
| `runs_folder` | `"runs"` | Where run folders are created; created if missing |
| `start_from` | none | A run folder (its latest checkpoint) or a checkpoint file: the new run's agents begin from it |
| `record_every_cycles` | `checkpoint_every_cycles` | The window of every cycle that is a multiple of this, and always the last, is recorded for replay; `0` records nothing |
| `record_levels` | 4 | Worlds are ranked by their summed reward over the window and split into this many levels |
| `record_per_level` | 8 | How many worlds of each level are kept, evenly spaced from the level's best to its worst |
| `record_selection` | `"ranked"` | `"ranked"` keeps the worlds chosen by level; `"first"` keeps the first worlds, as many, so that consecutive recordings show the same worlds |
| `time_limit_hours` | none | Bounds one training session: after each cycle, if one more cycle (at this session's average time per cycle) would end after the limit, the run saves a checkpoint and its report and stops, to be continued in a new session. A continuing file may set a new value |

A complete training file, with every section written out:

```toml
[run]
mode = "train"
name = "probe"                        # run folder: runs/<date>_<time>_<name>
seed = 1
checkpoint_every_cycles = 16
report = true
# runs_folder = "runs"
# start_from = "runs/2026-10-07_1432_easy"
# record_every_cycles = 16           # defaults to checkpoint_every_cycles
# record_levels = 4
# record_per_level = 8
# record_selection = "ranked"

[environment]
max_episode_steps = 8192
observation_radius = 1

[environment.simulation]
model_path = "models/assembly_v3.xml" # required
backend = "cpu"                       # required: "cpu" or "gpu"
world_count = 4
# gpu_solver = "newton"               # GPU only: "newton" or "cg"
# contacts_per_world = 128            # GPU only
# constraints_per_world = 512         # GPU only

[environment.target]
distance_range_m = [0.010, 0.020]
bearing_range_deg = [-15, 15]
arrival_radius_m = 0.001

[environment.rewards]
arrival_reward = 1.0
efficiency_cost = 0.003
body_contact_cost = 0.010
leg_contact_cost = 0.005
distance_ratio_epsilon_m = 1e-6

[agents]
device = "cpu"                        # "cpu" or "cuda", matching the backend
hidden_layers = [64, 64]
initial_action_std = 0.5
learning_rate = 3e-4
normaliser_epsilon = 1e-8
observation_clip = 10.0

[agents.ppo]
discount = 0.999
gae_lambda = 0.95
clip_ratio = 0.2
update_epochs = 4
minibatch_size = 64
max_gradient_norm = 0.5
entropy_coefficient = 0.001

[interaction_loop]
rollout_window_steps = 256
update_cycles = 128
```

### Starting from another run

A training file with `start_from` creates a **new run** whose agents begin with
another run's learned networks, optimizers, normalisers, and random
generators. Everything else is the new file's own, so the task can change: for
example, targets placed farther away once the centipede reaches near ones. The
new run counts its cycles from zero and keeps its own log and report, which
names the checkpoint it started from. Training in stages is a chain of such
runs.

Two agent settings must match the checkpoint, and a difference is rejected:
`hidden_layers`, because the networks must have the same shape, and
`learning_rate`, because the optimizers' saved state brings back the rate they
were created with. The number of segments and the observation size must match
too; the agents check those when loading.

### Continuing a run

A continuing file resumes an existing run, for example after a Kaggle session
ended, or to train it for more cycles:

```toml
[run]
mode = "train"
continue_from = "runs/2026-10-07_1432_probe"

[interaction_loop]
update_cycles = 256                   # optional: the new total
```

Training resumes in the same folder from the latest checkpoint and counts on
from its cycle. Windows logged after that checkpoint, by a session that stopped
between checkpoints, are removed from the log and trained again. Checkpoints
hold no unfinished episodes (see [agents.md](agents.md#checkpoints)), so the
environment starts new ones, drawn with the run's seed plus the number of
completed cycles so that they differ from the run's first episodes. The saved
configuration is updated, and each session is recorded in `run_info.json`.

## Evaluation files

`mode = "evaluate"` tests a checkpoint of a saved run. The task settings come
from that run's saved configuration, so by default evaluation tests the problem
the agents were trained on. Results are written inside the evaluated run's
folder.

```toml
[run]
mode = "evaluate"
source = "runs/2026-10-07_1432_probe" # a run folder (its latest checkpoint) or a checkpoint file

[evaluation]
seeds = [101, 102, 103, 104]
episodes_per_seed = 8
baselines = ["zero", "random"]        # optional; leave out for none

[environment.simulation]              # optional: replaces the run's own values
backend = "cpu"
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `seeds` | required | Each seed resets the environment once and runs one episode in every world |
| `episodes_per_seed` | 8 | The number of worlds, so the number of episodes per seed |
| `baselines` | none | `"zero"`, `"random"`, or both, run on the same seeds as the agents |
| `record` | `true` | Write a replay recording of every world for each actor and seed |

An evaluation file may also contain `[environment]` sections, whose values
replace the run's: `backend = "cpu"` evaluates a run trained on a GPU on the
CPU, and the agents' device follows the backend; farther targets test how far
the learned walking carries. Every changed value is listed in the results. The
number of worlds cannot be changed this way, since `episodes_per_seed` sets it.
CPU and GPU physics drift apart quickly (Stage 2 of the plan measured this), so
results from the two backends are not identical.

Each world counts only its first episode; worlds that finish early start new
episodes, which are not counted. An evaluation lasts up to `max_episode_steps`
steps per seed and per actor, so long episodes on the CPU take a while.

Baselines are off unless listed: `"zero"` always sends zero actions, which shows
whether the agents do better than doing nothing; `"random"` sends uniformly
random actions, which becomes a useful comparison once the centipede moves. The
agents act with their policy's mean action, without exploration noise, and the
checkpoint is never changed.

## What a run writes

```text
runs/2026-10-07_1432_probe/
├─ configuration.toml   the complete settings; can be run again as they are
├─ run_info.json        each training session: start, first cycle, settings its file set, code version, packages, device
├─ training_log.jsonl   one line of diagnostics per window
├─ report.html          the run's summary (unless report = false)
├─ checkpoints/         cycle_0016.pt, cycle_0032.pt, ...
├─ recordings/          cycle_0016.npz, ...: replay recordings of training windows
└─ evaluations/         <date>_<time>_<checkpoint>.json and .html for each evaluation,
                        and <the same>_<actor>_seed<seed>.npz recordings
```

Folders are created only when missing, and a run is never overwritten: a new run
whose folder name is taken, by a second run in the same minute, gets a numbered
suffix. The terminal shows one short line per window: the cycle, the speed, and
how the episodes that ended went.

The log and `run_info.json` hold what the reports need. Each log line has the
cycle, the world steps collected so far, and every diagnostics category,
including the bin counts of the episode histograms (`episode_distributions`,
see [diagnostics.md](diagnostics.md#episode-summary)). Each session in
`run_info.json` also lists the settings its configuration file set itself
(`settings_written`), so that a report can tell chosen values from defaults.

The **recordings** hold the poses of the recorded windows for replay in the
sibling MujocoReplay project: `recordings/cycle_NNNN.npz` is the window
collected in cycle `NNNN`, by the agents after `NNNN − 1` updates, with the
worlds chosen by level; an evaluation's recordings hold every world of each
actor and seed, from the first episode's first step until the last world's
first episode ended. Each file carries the model, the targets as markers, and
the run's setup, so it replays on its own
(see [diagnostics.md](diagnostics.md#recordings)). With the defaults, a
training run records one window of 32 worlds before each checkpoint, 2.3 MB
each for models v2 and v3. To watch a run's recordings, in order, with
MujocoReplay installed:

```powershell
mujoco-replay runs\<run>\recordings\*.npz
```

The **report** is one HTML file that opens in any browser, offline. It is
refreshed at every checkpoint, so an interrupted run still has one. It works the
same for a CPU run with a few worlds and a GPU run with thousands: the worlds
are summarised on the device before anything is logged, so the page grows with
the number of windows, not of worlds.

### How the report is laid out

The report is ordered by priority, and the size of each part follows it.

- **Header.** The run's name; one line with the cycles done, world steps,
  episodes, training time, and sessions; and one chip per group of settings
  (backend and worlds, model, episode limit, targets, observation radius,
  rewards, window and cycles, network, PPO, seed). Chips that include a setting
  the configuration file set itself are highlighted; the others show defaults.
  Every setting is listed in the details.
- **1 · Results.** Five tiles: arrival share, distance closed, reward per step,
  return, and episode length, each with its latest value, its value at the
  start, how many episodes the latest value rests on, and a small trend line.
  Episode values are means over the episodes of the first or last few windows
  that have any, so a window with many episodes counts for more than one with
  a single episode.
  Below them, the largest chart: the head's distance to its target in every
  window, with the distances at the start and the end of the episodes that
  ended and the gap between them shaded. Then the reward per step split into
  its terms, next to the return of each segment, and the three episode
  histograms, comparing the episodes of the first and the last quarter of the
  windows in which episodes ended.
- **2 · Behaviour** (smaller). Eight values along the body (body and legs
  touching, each foot on the ground, height, uprightness, speed, speed toward
  the goal), each early, midway, and late in training; the useful share of
  movement (speed toward the goal ÷ speed) along the body in the same way;
  heading error; and the share of time the head is upside down.
- **3 · Learning** (small). Policy change (KL), clipped samples, critic
  accuracy, and action spread, each as the segments' mean with their range.
- **4 · Run** (small). Steps per second, and the time each window spent
  collecting and learning.
- **Details,** behind a "Show details" button: the other learning values, each
  behaviour value per segment over training, the number of episodes that ended
  per window and the three histograms window by window (readable only with many
  worlds), a table of the body at the end of training, every other logged
  value, a table of every value at the start and at the end, the training
  sessions, the run's facts, and all settings.

Charts over training run along the world steps collected, so that runs with
different numbers of worlds can be compared; hovering shows the cycle too.
Dotted vertical lines mark where a later session resumed the run. The first
window of each session also holds its start-up, such as compiling the GPU
kernels (about 50 s on a fresh Kaggle machine), so it is drawn apart and left
out of the speed's median.

Every point of an episode value is a mean over the episodes that ended in that
window, and every point of a step value a mean over its worlds and steps. When
few stand behind each point (fewer than 100 episodes, or 10,000 world steps, in
a typical window), the points are drawn faint under a moving average; with many
worlds, as on the GPU, the line is drawn as it is. Worlds that start together
also time out together, so long episodes give few windows with episode values;
the values measured at every step, such as the head's distance and the reward
per step, have a point in every window, so they lead the results.

Values are shown in readable units: millimetres, millimetres per second,
degrees, seconds of simulated time, and percentages. Percentages never go
below 0 on an axis, and uprightness is always drawn on its whole range from −1
to 1, so that small differences are not magnified. Heatmaps of values that have a better
way (a higher return, height, uprightness, or speed toward the goal; less body
contact and fewer legs touching) run from red for the worst values shown to
blue for the best, through grey at zero or at the middle; their colour key says
which end is better. Values that are neither good nor bad, such as speed or
feet on the ground, run from light to dark blue. Clicking a chart's title
opens a short explanation of what it shows and how to read it, and hovering
over any mark shows its exact value. The reward per step and the mean return
are the only values the report computes itself, as sums and means of logged
values.

### Which chart for which question

Each kind of chart is used for one kind of question:

| Question | Chart | Why |
| --- | --- | --- |
| How did a value change over training? | Line over world steps | Positions along one axis are read most precisely; world steps make runs of different sizes comparable |
| What makes up a total? | Stacked areas, positive parts above zero and costs below, with the total as a line | The reward terms add up to the reward, so stacking shows the total and what drives it at once |
| How did each segment change over training? | Heatmap: one row per segment, one column per window | Eight lines would tangle; colour shows the pattern along the body and over time together |
| How did the episodes spread? | Histogram: early against late in training, or one per actor | A mean hides two groups of outcomes or a few failures; counting the episodes of many windows together keeps the bars readable with few worlds |
| How does a value differ along the body? | Line along the segments, head first, early, midway, and late in training | One axis for all moments, so the change can be read directly |
| How do two values relate? | Their ratio, as a line along the body | The useful share of movement (speed toward the goal ÷ speed) answers the question directly; a scatter of the two speeds was tried, but speed toward the goal is about a tenth of the speed, so its reference diagonal was unreadable |
| What is the current value? | Tile with a trend line | The headline number and its direction |
| How do the actors compare? | Bars from zero | Lengths compare quantities; used only when there are at least two actors |
| What is the exact value? | Table, in the details only | Tables are for looking values up, not for seeing patterns |

A pair plot is not used: over training every pair of values moves with time, so
its panels would suggest relationships that are not there.

### Changing the report

The layout and the explanations are declared at the start of the page's script
in `src/centipede/experiment/report_page.html` (`METRICS`, `SETTING_CHIPS`,
`TRAINING_LAYOUT`, `EVALUATION_LAYOUT`), so changing the report means editing
those lists. A value added to a diagnostics category that no layout places
appears among the details automatically. The layout is a baseline for later
work: values may be moved or resized and new ones added, but what it shows
stays.

### The evaluation report

An **evaluation report** follows the same priorities and does not need
baselines. Its header lists the seeds and episodes, its chips mark the settings
changed for the evaluation, and its results show the five headline values as
tiles (the mean over the seeds and the lowest to highest seed), the three
episode histograms, and bars of the return per segment and of the reward terms.
When baselines are listed, each tile adds one bar per actor, the histograms
show every actor side by side, and the charts draw the baselines in grey. Its
behaviour section has the eight values along the body and the useful share of
movement, each with one line per actor, and the steering values,
as bars when there are actors to compare and as numbers otherwise. Its details
hold every value per actor, one table of the body per actor (coloured on shared
scales so that they can be compared), the evaluation's facts, and the settings.
