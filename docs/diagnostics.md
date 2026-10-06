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
| Physics health | Physics simulation | How close the physics came to its limits |
| Learning | Agents | How each segment agent's last update went |
| Timing | Interaction loop | Where the run's time goes |

`W` is the number of worlds and `N` the number of segments, as in
[environment.md](environment.md#interface).

### Step facts

Refreshed on every step, for every world.

| Value | Shape | Meaning | Summary |
| --- | --- | --- | --- |
| `reward_parts` | `(W, N, 4)` | Arrival, efficiency, body-contact cost, leg-contact cost; they add up to the reward | Mean |
| `contact_flags` | `(W, N, 4)` | Left foot, right foot, and body on the ground; legs touching | Share of steps |
| `segment_progress` | `(W, N)` | Metres gained toward the segment's goal: the target for the head, the spot where the segment ahead was for the others | Mean |
| `segment_moved` | `(W, N)` | Metres the segment's centre moved, measured flat on the ground | Mean |
| `body_height` | `(W, N)` | Height of the segment's centre, m | Mean |
| `uprightness` | `(W, N)` | How upright the segment is: 1 upright, 0 on its side, −1 upside down | Mean |
| `head_distance` | `(W,)` | Flat distance from the head's tip to the target, m | Mean |
| `heading_error` | `(W,)` | Angle between the head's forward direction and the target, rad, from 0 to π | Mean |

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
| `length_steps` | `(W,)` | Episode length in steps of 20 ms |
| `segment_return` | `(W, N)` | Sum of each segment's rewards over the episode |
| `start_distance`, `final_distance` | `(W,)` | Head's distance to the target at the start and at the end, m |
| `head_path_length` | `(W,)` | Total distance the head's tip travelled, m; with the net progress it shows how direct the path was |
| `segment_total_progress` | `(W, N)` | Sum of `segment_progress`: what each segment contributed |
| `body_contact_share`, `leg_contact_share` | `(W, N)` | Share of steps with the body on the ground, or legs touching |
| `foot_contact_share` | `(W, N, 2)` | Share of steps each foot was on the ground |
| `upside_down_share` | `(W,)` | Share of steps with the head upside down |

### Physics health

Defined together with the work on simulation speed, so that measuring it never
changes the speed measured before. The intended values are single numbers for
the whole run, the same on both backends, each taken from the busiest physics
step: contacts per world, constraint rows in one world, and solver iterations.
They show how close a run comes to the memory reserved on the GPU and how hard
the solver works.

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
| `action_std` | `(N, 6)` | Each action's learned spread, before squashing |

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
largest value. The episode summary counts only the worlds whose episode ended.
In evaluation, only each world's first episode counts, so the loop passes the
worlds still in it. A window with no ended episode gives NaN means for the
episode summary.

## Files

| File | Covers |
| --- | --- |
| `src/centipede/diagnostics_category.py` | Shared helper: describing values and summarising them over a window, written once for every category |
| `diagnostics.py` in a component's folder | That component's category and how it is filled |

A component fills its category with one call, at the point where it already
has the values. Reading, logging, and the report belong to the experiment.
