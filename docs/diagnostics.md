# Diagnostics

Diagnostics are measurements that show how a run is going: what the body does,
how episodes end, how learning progresses, and how fast the program runs. They
feed the training log and the report of a run. They are never used for learning:
no policy observes them and no reward depends on them.

## How they work

- **Each component measures only its own work.** It keeps its measurements in a
  small set of tensors, its **category**, allocated once and refreshed in place,
  like the physical state. No other component writes them.
- **The experiment reads them all in one place.** It creates the components, so
  it can collect every category when a run starts, in one list. Logs and
  reports read through that list and never ask components directly. The only
  category that is passed on is the physics simulation's, which the environment
  offers next to its own because only the environment can see the simulation.
- **They never slow the pipeline down.** Categories are filled on the device
  where the data already is, from values the component has computed anyway, and
  no component waits for the GPU because of them. They are copied to the CPU
  only when a log line or the report is written.
- **Every value is described and used.** Each value comes with its meaning, its
  unit, and how it is summarised over worlds and steps. The log and the report
  are built from these descriptions, so every value appears in them. A value
  that would not be worth showing is not measured.

## Categories

| Category | Measured by | Contents |
| --- | --- | --- |
| Step facts | Environment | What happened in the last 20 ms step |
| Episode summary | Environment | How each finished episode went |
| Simulation facts | Physics simulation | Every world's positions, and how close the physics came to its limits |
| Learning | Agents | How each segment agent's last update went |
| Timing | Interaction loop | Where the run's time goes |

`W` is the number of worlds and `N` the number of segments, as in
[environment.md](environment.md#interface).

### Step facts

Refreshed on every step, for every world.

| Value | Shape | Meaning | Summary |
| --- | --- | --- | --- |
| `reward_parts` | `(W, N, T)` | Each reward term, weighted: arrival, progress, step cost, body contact, leg contact, and movement when it has parts (runs before 2026-10-08: arrival, efficiency, body contact, leg contact); they add up to the reward | Mean |
| `contact_flags` | `(W, N, 4)` | Left foot, right foot, and body on the ground; legs touching | Share of steps |
| `segment_progress` | `(W, N)` | Metres gained toward the segment's goal: the target for the head, the spot where the segment ahead was for the others | Mean |
| `segment_moved` | `(W, N)` | Metres the segment's centre moved, measured flat on the ground | Mean |
| `joint_movement` | `(W, N)` | How far the joints the segment commands moved on the step, rad: the root of their mean squared movement, which the movement cost charges | Mean |
| `spine_bend` | `(W, N)` | Angle of the spine joint behind the segment, either way, rad; 0 for the rear | Mean |
| `body_height` | `(W, N)` | Height of the segment's centre, m | Mean |
| `uprightness` | `(W, N)` | How upright the segment is: 1 upright, 0 on its side, −1 upside down | Mean |
| `head_distance` | `(W,)` | Flat distance from the head's tip to the target, m | Mean |
| `heading_error` | `(W,)` | Angle between the head's forward direction and the target, rad, from 0 to π | Mean |
| `target_position` | `(W, 2)` | Each world's target, world x and y, m; refreshed again for the worlds a step resets, so it always matches the pose | Recorded |
| `range_radius` | `(W,)` | Radius of each world's range circle around its target, m (infinite without one); refreshed like the target | Recorded |

Comparing `segment_progress` with `segment_moved` shows how much of a segment's
movement brings it closer to its goal. Gait measures, such as how long each foot
stays on the ground and how a step travels from segment to segment, are worked
out later from `contact_flags` over time.

### Episode summary

Written when an episode ends, for the worlds whose episode ended on that step.
Over a window, `episode_ended` is summarised as the number of episodes that
ended, and every other value as a mean over those episodes.

| Value | Shape | Meaning |
| --- | --- | --- |
| `episode_ended` | `(W,)` | Which worlds' episodes ended on this step; only their rows are summarised |
| `arrived` | `(W,)` | Whether the head reached the target, rather than running out of time |
| `left_range` | `(W,)` | Whether the episode was cut because the head's tip left the range circle |
| `ending` | `(W,)` | How the episode ended, as the middle of its bin: arrived (0.5); ran out of time within a quarter of the start distance (1.5), within half (2.5), closer than at the start (3.5), or not closer (4.5); left the range circle (5.5) |
| `length_steps` | `(W,)` | Episode length in steps of 20 ms |
| `segment_return` | `(W, N)` | Sum of each segment's rewards over the episode |
| `start_distance`, `final_distance` | `(W,)` | Head's distance to the target at the start and at the end, m |
| `distance_closed` | `(W,)` | Share of the start distance closed by the end: 1 − final ÷ start; 1 at the target, below 0 when the head moved away |
| `head_path_length` | `(W,)` | Total distance the head's tip travelled, m; with the net progress it shows how direct the path was |
| `segment_total_progress` | `(W, N)` | Sum of `segment_progress`: what each segment contributed |
| `body_contact_share`, `leg_contact_share` | `(W, N)` | Share of steps with the body on the ground, or legs touching |
| `foot_contact_share` | `(W, N, 2)` | Share of steps each foot was on the ground |
| `upside_down_share` | `(W,)` | Share of steps with the head upside down |

Five of these values are also counted in **histograms**, so that the report
can show how the episodes spread and not only their mean. With many worlds, a
mean hides whether every episode went halfway or some arrived while the rest
flipped over. The bins are fixed, the same for every run:

| Value | Bins |
| --- | --- |
| `ending` | One bin per ending, edges 0 to 6: measured against each episode's own start distance, so the classes hold for any target distance; the report's "Where episodes ended" uses it |
| `final_distance` | Edges at 0, 1, 5, 10, 15, 20, 30, 50, 100, and 200 mm: below 1 mm were the arrivals with the tip's 1 mm radius; the others tell how far the episodes that ran out of time ended. Reports of runs without `ending` use it |
| `distance_closed` | From −100% to 100% in steps of 10% |
| `length_steps` | Edges at 0, 16, 24, 32, 48, 64, … 12,288, 16,384 steps: each about 1.5 times the last, including every power of two, so that a time limit such as 1,024 or 8,192 steps starts its own bin and time-outs are not mixed with arrivals |
| `upside_down_share` | From 0% to 100% in steps of 10% |

Values beyond the outer edges count in the first or the last bin.

### Simulation facts

Refreshed by every step and reset, for every world. The physical state carries
what the environment needs; this category carries what the recorder and the
speed work need. All four values describe the moment after the `forward` call
that follows a step's last physics step, or a reset, so the CPU and GPU
backends report the same thing; they are not peaks over the physics steps of
a transition (134 for model v3). On the GPU the category is made of views of MuJoCo Warp's own
arrays, so filling it costs nothing; on the CPU each world's values are copied
after the step.

| Value | Shape | Meaning | Summary |
| --- | --- | --- | --- |
| `qpos` | `(W, nq)` | Every world's position coordinates, in MuJoCo's `qpos` layout (69 for models v2 and v3) | Recorded |
| `contact_count` | `(1,)` | Contacts in all worlds together; the GPU reserves `contacts_per_world × W` | Maximum |
| `constraint_rows` | `(W,)` | Constraint rows in each world; the GPU reserves `constraints_per_world` | Maximum |
| `solver_iterations` | `(W,)` | Solver iterations each world needed on the last physics call | Maximum |

The three maxima are logged as the `physics` category and show how close a
run comes to the memory reserved on the GPU and how hard the solver works. The
CPU benchmark (`benchmarks/simulation_speed.py`, since archived, 4 worlds, 20 transitions,
on the laptop) measured 0.132 s per step before the category existed and
0.112 to 0.138 s over three runs after it: the run-to-run spread is wider than
any difference.

### Learning

Written after each update, one row per segment agent. The first six come from
RL_lib's update summary; each is averaged over the update's minibatches, except
the explained variance, which describes the batch before the update.

| Value | Shape | Meaning |
| --- | --- | --- |
| `actor_loss` | `(N,)` | Clipped policy loss, including the entropy bonus |
| `critic_loss` | `(N,)` | Half the mean squared error of the values |
| `entropy` | `(N,)` | Estimated entropy of the policy: how widely it explores |
| `approximate_kl` | `(N,)` | How far the update moved the policy |
| `clip_fraction` | `(N,)` | Share of samples whose probability ratio was clipped |
| `explained_variance` | `(N,)` | How much of the returns' variation the critic predicted: 1 exactly, 0 nothing |
| `learning_rate` | `()` | The optimizers' learning rate in the update, the same for every agent; set by the experiment from the schedule |
| `action_std` | `(N, 6)` | Each leg action's learned spread, before squashing |
| `spine_action_std` | `(N,)` | The learned spread of the command for the spine joint behind the segment, before squashing; 0 for a segment without one |

### Timing

Measured by the interaction loop for each window, as single values. The clock
waits for the GPU only when collecting or learning starts and ends, so the
times include the GPU's work.

| Value | Meaning |
| --- | --- |
| `collecting_seconds` | Time spent collecting the window: acting, stepping, and recording |
| `learning_seconds` | Time spent on the update after the window; 0 in evaluation |
| `transitions_per_second` | World steps collected per second of collecting |

## Window summaries

The log has one line per window, so the environment's step facts and episode
summaries are summarised over each window as they are refreshed. The interaction
loop adds every step to two window summaries, one per category, and each
starts empty when a window starts. A summary keeps running totals on the
category's device, so it never waits for the GPU, and removes the worlds:
a `(W, N)` value becomes `(N,)`.

Each value follows its own summary: a **mean** or a **share** averages over
the counted rows, a **count** adds up true flags, and a **maximum** keeps the
largest value. A **recorded** value is never summarised: it is kept as it is
for the recorder and left out of the window summary, the log, and the
report. A value declared with histogram edges is also counted bin by
bin: each counted row adds one to the bin its value falls in, a bin including
its lower edge. The counts are kept on the device like the totals, so they too
never wait for the GPU. The episode summary counts only the worlds whose episode ended.
In evaluation, only each world's first episode counts, so the loop passes the
worlds still in it. A window with no ended episode gives NaN means for the
episode summary.

## Recordings

A recording keeps the poses of a stretch of steps for replay, so that what the
worlds did can be watched afterwards without rendering anything during the
run. In training the stretch is one episode length of windows: all worlds
start together, so it holds every world's whole episode from its first step
(or several episodes for the worlds that arrived early). The interaction
loop's recorder (`interaction_loop/recording.py`) is armed by the experiment
for that many steps before the first of those windows, and carries on through
the following windows until it is complete. While it runs, every step copies
the simulation's `qpos` of every world into a buffer on the device and adds
the step's rewards to each world's score; the step facts' target and the
episode summary's `episode_ended` are kept alongside. These are a few small
tensor operations that never wait for the GPU. When the recording is complete,
or the session ends, the experiment takes the kept worlds between cycles and
writes the file: the worlds are ranked by their summed reward over the whole
recording and written in rank order, best first. By default every world is
kept; a number of worlds (`record_worlds` in
[configuration.md](configuration.md#training-files)) keeps that many, evenly
spaced over the ranks with MujocoReplay's `selected_ranks` (one band), so that
the file holds the best, the middle, and the worst. Only the kept worlds are
copied to the CPU. Each file also gives every world's rank among all the run's
worlds and its level among `record_levels`, so the viewer shows, for example,
"rank 37 of 1,024" and "level 2 of 4". With every world in the file, the
viewer's highlight reaches each of them, whichever worlds it draws.

An evaluation's recording holds every world, ranked by the summed reward of its
first episode only, as in the evaluation's results; the frames continue until
the last world's first episode has ended.

The file format belongs to the sibling MujocoReplay project, which replays
the files (`../MujocoReplay/docs/recording-format.md`); `experiment/recordings.py`
builds each file from the recorded steps, the model, and the run's facts.

## Files

| File | Covers |
| --- | --- |
| `src/centipede/diagnostics_category.py` | Shared helper: describing values and summarising them over a window, written once for every category |
| `diagnostics.py` in a component's folder | That component's category and how it is filled |
| `src/centipede/environment/simulation/diagnostics.py` | The simulation's category, filled by the backends |
| `src/centipede/interaction_loop/recording.py` | The recorder: the poses of the armed steps, across windows, and the choice of worlds |
| `src/centipede/experiment/recordings.py` | Turning the recorded steps into a MujocoReplay recording file |

A component fills its category with one call, at the point where it already
has the values. A value whose last dimension has named entries, such as the
four contact flags, lists them as its `parts`, so that logs and reports can
label them, and a value counted in a histogram lists its `histogram_edges`. Reading, logging, and the report belong to the experiment: it
collects every category when a run starts, writes one log line per window, and
draws the report from the descriptions (see
[configuration.md](configuration.md#what-a-run-writes)).
