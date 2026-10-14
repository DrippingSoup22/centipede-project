# The reward and the tools

<!-- Formulas go in ```math blocks only, never inline $...$: GitHub stops
     rendering the blocks of a <details> section that also holds inline math. -->

What the agents are paid for now, and every tool built around the reward.
This page describes the workspace branch `oscillators` as it trains today.

- **Part 1, the reward:** only the terms in use, grouped by kind, each with its
  idea, its math, and what to expect from it.
- **Part 2, the tools:** everything that is not a reward term in use, in use
  or tried before, so that any of them can be tried again.

**Current as of:** cycle 3's step-value run (`configs/oscillators/57_...` in
CentipedeTraining), the user's redesign of 2026-10-13: each agent is paid
only for what it can see and cause. Every segment's step is judged by the
[step value](#2-the-step-value), in one physical unit; the followers learn
their step and the [tempo chain](#tempo-chain), nothing from the target; the
head learns its step and the [task](#1-task-terms), each with a [critic of
its own](#a-task-critic-for-the-head). Each segment decides its legs' curves
and its tempo once per turn and [holds them for the whole
turn](#curves-held-for-a-turn), and learns from each turn as a whole. Model v5, model B, the circle
task from 10 cm, white noise for the turns and pink for the neck, no CAPS.
Two tools were built for run 58, an appendix test the user then dropped
(2026-10-14) to rethink the reward around cooperation: a [fixed
tempo](#a-fixed-tempo) and [learning from whole
turns](#learning-from-whole-turns); both are off unless set. Before run 57, runs 51 to 56 paid every
segment the head's task and charged the [retired](#retired-reward-terms)
health costs (foot hold, carry, leg pushing): the followers' critics
explained none of their returns, because the shared task they could neither
see nor cause drowned their own signal (the study notes, "Why the followers
do not learn their legs"). The values marked *measured* come from checks on
the laptop's CPU, not from training.

> **Kept current.** Each cycle has a **theory phase** (design the reward or
> another part, implement and test it) and an **experiment phase** (train and
> read the results). At the end of every theory phase, the sections of this
> page that changed are rewritten, the tools' marks updated, and the commit is
> pushed. Terms no longer in use are removed from Part 1, not kept.

---

## Part 1. The reward

### At a glance

Twelve agents on model v5, one per segment: the **head** (segment 0) and
eleven **followers** (segments 1 to 11). Every term is computed on every
20 ms step, for every segment.

**The principle** (the user's, 2026-10-13): an agent is paid only for what
it can see and cause, or its critic cannot learn what its actions are worth.
A follower sees its own body and feet and its neighbours', never the target:
it is paid for its step and for keeping the tempo of the segment ahead. The
head is a follower that sees the target: the same step value, plus the task.

| Group | Term | Head | Followers | Weight |
| --- | --- | :---: | :---: | --- |
| 1. Task | [Arrival](#arrival) | ✔ (task critic) | – | 0.5 per arrival |
| 1. Task | [Progress](#progress) | ✔ (task critic) | – | 0.100 per halving |
| 2. Step value | [Carried forward](#the-four-checks) (C1) | ✔ | ✔ | +s per segment weight × mm |
| 2. Step value | [Foot sliding](#the-four-checks) (C2) | ✔ | ✔ | −s |
| 2. Step value | [Carried by others](#the-four-checks) (C3) | ✔ | ✔ | −s |
| 2. Step value | [Leg pressing](#the-four-checks) (C4) | ✔ | ✔ | −s |
| 3. Rhythm | [Tempo chain](#tempo-chain), ahead | – | ✔ | 3.5 parts, −0.00164 at most |
| 3. Rhythm | [Tempo chain](#tempo-chain), behind | ✔ | ✔ (not the rear) | 7 parts, −0.00327 at most |

`s = 0.0015` per segment weight carried one millimetre (`step_value_scale`).
Each segment's reward has two parts, its walk's and the task's:

```math
r_i^{\text{walk}} \;=\; s\,\big(C_{1,i} - C_{2,i} - C_{3,i} - C_{4,i}\big) \;-\; 0.00164\,A_i \;-\; 0.00327\,B_i, \qquad
r_0^{\text{task}} \;=\; 0.5\,\mathbf{1}[\text{arrived}] \;+\; 0.100\,P
```

with `r^task = 0` for the followers. A follower learns its walk with one
critic. The head learns its walk with one critic and the task with another,
and its policy learns from

```math
\hat A_0 \;=\; \operatorname{norm}\big(A^{\text{walk}}_0\big) \;+\; 0.2\,\operatorname{norm}\big(A^{\text{task}}_0\big)
```

each advantage normalised over the batch on its own: the task counts a fifth
of the walk, whatever the distance to the target or the arrivals' rarity
(`head_task_critic_weight = 0.2`, the user's balance: walking first,
progress about a fifth to a tenth of it).

<details>
<summary><b>How the weights are set</b></summary>

- **The step value has a physical unit:** one segment's own weight carried
  one millimetre. All four checks are measured in it and all weigh 1 in it,
  so the physics, not our choice, sets the balance between them; each can
  still be weighed on its own (`carried_forward_weight` and the others).
- **Its scale** sets it against the tempo chain, the followers' only other
  term. It is 0.0015, from the pricing below: run 56's followers' walk,
  judged by the step value, then costs about what its health costs charged
  it (−0.00074 per step at cycle 64), so the tempo chain keeps the weight
  beside the step that it had beside the health costs, where it worked. A
  decent walk then earns about +0.00008 per step; the chain costs about
  0.00027 at an episode's start and 0.00002 to 0.00005 once settled.
- **The task's weights** follow the cost budget as before. With episodes of
  `T = 512` steps the agents discount by γ = 2^(−1/T) = 0.998647; over a cost
  horizon of `H = 256` steps the budget, split into `cost_budget_parts = 7`,
  makes **one part 0.000467 per step**. One halving of the head's distance is
  worth `progress_parts = 0.84` parts of the horizon, 0.100; an arrival
  pays `arrival_payout = 0.5`. The tempo chain takes 3.5 and 7 parts.
- **The head's balance is set by the advantage weight, not the reward
  scales.** With one critic, a reward's share of the learning signal is its
  share of the return's variation, which nothing fixes: progress per
  millimetre grows as the target nears (0.00144 per mm at 10 cm, three times
  that at 3 cm), and once arrivals start they multiplied the return's
  variation 70 to 88 times (measured on run 54, cycle 288), so they would
  drown the step as they drowned the followers'. Normalised on their own, the
  two advantages keep the ratio the weight sets (van Seijen et al., NeurIPS
  2017, learn one value function per reward component; Mysore et al., ICLR
  2022, one critic per task).

Settings: `[environment.rewards]` and `[agents] head_task_critic_weight` in
the run's file; the program prints every weight under the settings when a run
starts. Code: `src/centipede/environment/settings.py`,
`RewardSettings.weights`; RL_lib's PPO for the critics.
</details>

<details>
<summary><b>What to expect</b>: the pricing (measured)</summary>

On the laptop's CPU, model v5, 8 worlds, 2 s to settle and 8 s measured,
each check at weight 1 (`scratchpad` check of 2026-10-14, not a training
result). Per segment, in segment weights × mm per second; the last column
is the step value per step at `s = 0.0015`, in hundred-thousandths. The
speed is the segment centres' path, in any direction.

| Walk | Speed (mm/s) | C1 | C2 | C3 | C4 | Step value | Per step | Carried itself |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hand-made wave: the default 15° step at 2 Hz, load feedback 3, follow nudge 8 rad/s | 13.0 | +10.3 | −5.1 | −2.5 | −0.01 | **+2.7** | +8 | 0.75 to 0.91 |
| The same legs at random phases, no reflexes | 11.4 | +7.5 | −5.2 | −3.7 | −1.4 | **−2.8** | −8 | 0.60 to 0.76 |
| Standing still | 0 | 0 | 0 | 0 | 0 | **0.00** | 0 | 0.95 to 1 |
| Run 56's agents, cycle 128 (mean action) | 20.4 | +0.3 | −14.9 | −9.2 | −1.4 | **−25** | −76 | 0.29 to 0.76 |
| Run 54's agents, cycle 288 (mean action) | 42.4 | +6.6 | −38.8 | −26.4 | −17.9 | **−77** | −230 | 0.34 to 0.43 |

The step value ranks the walks as wanted: the coordinated wave above
standing still, and standing still above the same legs at random phases,
almost as fast but colliding (C4) and less supported (C3). So the wave is
paid without being named. The trained agents score far below 0: their
bodies move a lot (run 54's path, 42 mm/s, against a closing speed of 2 to 4)
on feet that slide and segments that hang, which is what the user saw. Even
the hand-made wave's loaded feet slide half as far as they carry (C2 against
C1): on v5's floor (friction 0.8) a sweeping leg drags its own loaded foot,
so a better step has to push less hard or keep its feet lighter.
</details>

### 1. Task terms

What the centipede is for: reaching targets. Only the head sees the target,
so only the head is paid for it, with a critic of its own
([below](#a-task-critic-for-the-head)). Until run 56 every follower earned
the head's arrival and progress too (`follower_arrival_share` and
`follower_progress_share` at 1): a follower could neither see nor change
them, and they made up 82 to 98% of its return's variation (measured, runs 54
and 56), so its critic learned nothing (explained variance near 0 from cycle
64). Now both shares are 0.

#### Arrival

**Idea.** The goal itself: the head brings the target under its body.

<details>
<summary>Math, and what to expect</summary>

```math
\text{arrival} = \mathbf{1}[\text{the target lies under the head's outline, seen from above}], \qquad w_A = 0.5
```

Paid once, on the step it happens, to the head; that step ends the episode
for learning, and the body walks on to a new target with its clocks still
running.

**What to expect.** Its part per step is `w_A` times the arrivals per step:
it grows with the targets reached per minute. While the curriculum climbs,
about half of the episodes arrive by construction, so this part rises slowly;
it grows faster only once the targets stop getting farther.
</details>

#### Progress

**Idea.** Every step that brings the head closer pays, in **halvings** of its
distance: ending closer is better than not trying, wiggling earns nothing,
and walking past the target gives back what was earned. With arrival and
discounting, it also makes an early arrival worth more, so the head needs no
time cost (the user's, 2026-10-13).

<details>
<summary>Math, and what to expect</summary>

```math
P \;=\; \log_2 \frac{\max(d_{t-1},\, r)}{\max(d_{t},\, r)}, \qquad w_P = 0.100 \text{ per halving}
```

`d` is the flat distance from the head's tip to the target, before and after
the step, and `r = 4 mm`, half the head's width: closer than that, nothing pays
more. Over an episode the steps add up to

```math
\sum_t P \;=\; \log_2 \frac{d_{\text{start}}}{\max(d_{\text{end}},\, r)}
```

whatever the path. It is potential-based shaping (Ng, Harada and Russell,
ICML 1999): it hints at the direction without changing which policy is best.

**What to expect.** Positive while the head walks toward the target, 0 when it
stands still, negative when it walks away; a fifth to a half of an arrival's
worth per approach.
</details>

### 2. The step value

**Idea.** The user's design (2026-10-13): judge each segment's own step, the
one thing a follower sees and causes, in one number with no template and no
hand-chosen penalty. A good step **carries the segment's weight, holds the
ground while it pushes the body forward, and returns through the air**. In
one physical currency, a segment's weight times millimetres:

> **step value = how far the segment carried itself forward on feet that
> held − how far others carried it**

Zero means not walking, positive walking well, negative walking badly. The
only modelling choice is the cap at one body weight, so that the segments
cooperate with neither greed nor laziness (the user's).

#### The four checks

<details>
<summary>Math</summary>

On every physics step `k` of a 20 ms step (134 on v5), each foot's load is
its floor force over the segment's own weight, `N_f = F_f / (m_i g)`, and
the segment's carried share is

```math
c_i \;=\; \frac{1}{134}\sum_{k} \min\!\big(1,\ N_{\text{left},k} + N_{\text{right},k}\big)
```

With `Δx` the segment centre's movement on the step in mm and `h` its
forward direction:

```math
C_1 = c_i\,(\Delta x \cdot h), \qquad
C_2 = \sum_{k}\sum_{\text{feet}} N_{f,k}\,\lVert \Delta p_{f,k} \rVert, \qquad
C_3 = (1 - c_i)\,\lVert \Delta x \rVert, \qquad
C_4 = \frac{F^{\text{legs}}_i}{m g / N}\,\lVert \Delta x \rVert
```

- **C1, carried forward** (paid): the share of its own weight its feet
  carried, times how far it moved forward. Backward counts negative, so
  rocking cancels out.
- **C2, foot sliding** (charged): each foot's load times how far it slid on
  each physics step, `Δp` in mm. A loaded foot dragged by the body or
  slipping back under too hard a push wastes the stride; a foot in the air
  bears nothing.
- **C3, carried by others** (charged): the share of its weight no foot
  carried, times how far it moved: taps, both feet up, a loaded foot lifted
  before its weight passed to the other.
- **C4, leg pressing** (charged): the force between its legs and other legs,
  in segment weights (`m g / N`, 0.86 mN on v5), times how far it moved.
  The one thing the others cannot see: a leg can hold the ground while it
  shoves its neighbour's. Both segments of a contact pay.

Everything is counted inside the physics steps (`PhysicalState`'s
`carried_share` and `loaded_sliding_m`), so a foot that lands and lifts
between two 20 ms readings still counts, and a landing foot's flight never
counts as sliding. The body's own movement is read per 20 ms step.
</details>

#### What it covers without a term

| A good step… | Why it follows |
| --- | --- |
| makes one long contact per cycle, no taps | a tap carries almost nothing (C1) and leaves the segment carried by others (C3) |
| spends about half the cycle on the ground | it follows from needing support; 0.5 is not imposed |
| alternates left and right | otherwise both feet are up and others carry it (C3) |
| sweeps back at the body's speed | any mismatch is sliding (C2) |
| lands in front | a longer stance carries the body farther (C1) |
| lifts only once its weight has moved off | lifting a loaded foot leaves the weight unsupported (C3) |

A travelling wave is not named either. Physics gives two reasons for one:
the body cannot roll, so the legs of one side must not all lift together
(C2, C3), and neighbours alternating exactly sweep into each other once their
strokes are long (C4), Kuroda's leg-crossing argument (Kuroda et al.,
bioRxiv 2018). Whether it is enough is measured at every check.

#### Shortcuts checked

- **Standing still** is worth 0; but when the head moves the body, a follower
  that stands is dragged (C2) or hangs (C3).
- **Rocking, or walking backward**: forward counts positive and backward
  negative.
- **Carrying a neighbour** earns nothing past one body weight.
- **Jerking when a curve is switched** costs sliding or lost support.
- **Riding passively while others push** is allowed for now; the diagnostics'
  `carried_movement` over `segment_moved` gives each segment's share of its
  movement it carried itself, which shows it.

#### Distance or share

The version in use is the **distance**: a longer stride earns more, so at
one tempo a faster walk earns more. The **share** version, the step value
divided by the distance the segment moved, would make it speed-neutral and
leave the speed to the head; it is defined only over a whole cycle (a still
segment has no share), so it needs the step value summed over each segment's
turn.
The user's choice: test both, distance first.

### 3. Rhythm terms

#### Tempo chain

**Idea.** The user's rule (2026-10-13), in place of [tempo
ahead](#retired-reward-terms): each segment copies the tempo of the segment
ahead and minds the segment behind. It is fair to the followers: it compares
their tempo with their neighbours', which they hear, so their critic can
predict it; it is the one mechanism that worked in every run.
The gap ahead costs in proportion, so that even a small gap is worth
closing; the gap behind costs its square, so that a small lead is almost
free and a large one dear. With the behind weight twice the ahead weight, a
follower copies the segment ahead until it is an octave from the segment
behind, then waits for it; the head, which has only a segment behind, may
lead the body but not run away from it. It judges only tempos, never
phases, so the wave stays free to emerge.

<details>
<summary>Math, and what to expect</summary>

```math
A_i \;=\; \min\!\Big(1,\ \frac{\lvert \log_2 f_i - \log_2 f_{i-1} \rvert}{4}\Big), \qquad
B_i \;=\; \min\!\Big(1,\ \frac{\lvert \log_2 f_i - \log_2 f_{i+1} \rvert}{4}\Big)^{2}, \qquad
w_A = 3.5 \times 0.000467, \quad w_B = 7 \times 0.000467
```

with `A_0 = 0` for the head and `B = 0` for the rear segment. Both reach
their whole weight at four octaves, beyond any gap the body reaches (the
12.5 Hz ceiling over a 1 Hz neighbour is 3.6), so neither ever goes flat:
their sum has exactly one cheapest tempo for every pair of neighbours, and
from anywhere else the cost falls steadily toward it. The first scale, two
octaves (run 42), went flat under the racing head, 2.3 octaves above
segment 1, so neither of them felt any pull. A follower stops copying where
the two slopes meet, 2 w_A / w_B = 1 octave from the segment behind. The
gaps seen: at an episode's start 0.67 octave on average between neighbours
(a quarter over 1); once settled, 0.05 to 0.13 between runs 39 to 41's
followers. At the largest gap reached, 2 octaves, a middle follower pays
1.75 + 1.75 parts, today's 3.5; the 10.5 of four octaves is never reached.

| Gap | ¼ octave | ½ octave | 1 octave | 2 octaves |
| --- | ---: | ---: | ---: | ---: |
| Ahead, per step | −0.00010 | −0.00020 | −0.00041 | −0.00082 |
| Behind, per step | −0.00001 | −0.00005 | −0.00020 | −0.00082 |

**What to expect.** Run 39's head, 2.3 octaves above segment 1, would pay
−0.0011 per step behind, about two time costs, less with every octave it
comes down; segment 1 −0.00094 ahead. A head leading by a quarter octave
pays −0.00001.
</details>

---

## Part 2. The tools

Everything around the reward: how the legs are driven, what each segment
sees, how the agents explore, and how training is scheduled.
✅ is in use now (run 57); ⬜ is not. The runs are those of the study notes
(`results/oscillators/README.md` in CentipedeTraining).

| | Tool | Kind | Setting | Runs |
| :---: | --- | --- | --- | --- |
| ⬜ | [Model A, the network draws](#model-a-the-network-draws) | Leg control | `leg_drive = "swing"` | pilot |
| ✅ | [Model B, the clock is the engine](#model-b-the-clock-is-the-engine) | Leg control | `leg_drive = "waves"`, `waves = 2` | 31– |
| ✅ | [The heart](#the-heart) | Leg control | `tempo_rate_octaves_per_s = 1`, `start_tempo_range_hz = [1, 4]`, `max_tempo_hz = 12.5` | 31–57 (ceiling 35–) |
| ⬜ | [A fixed tempo](#a-fixed-tempo) | Leg control | `[environment.clock] fixed_tempo = true`, `middle_tempo_hz = 6` | none (built for 58) |
| ✅ | [Targets along the curve](#targets-along-the-curve) | Leg control | always with models A and B | 31– |
| ✅ | [Curves held for a turn](#curves-held-for-a-turn) | Leg control | `hold_curves = true` | 57– |
| ⬜ | [Learning from whole turns](#learning-from-whole-turns) | Leg control | `hold_spine = true` | none (built for 58) |
| ✅ | [A clock per leg](#a-clock-per-leg) | Leg control | `leg_clocks = true`, `segment_tempo = true` | 08– |
| ✅ | [Passive follower spine](#passive-follower-spine) | Leg control | `passive_follower_spine = true` | all |
| ⬜ | [The step shape](#the-step-shape) | Leg control | `leg_drive = "step_shape"` | 08–28 |
| ⬜ | [The clock's lag](#the-clocks-lag) (the action filter) | Leg control | `clock_time_constant_s = 0.5` | 11–13, 15–28 |
| ⬜ | [Load feedback](#load-feedback) | Leg control | `load_feedback_rad_per_s = 3` | 09–28 |
| ⬜ | [The follow nudge](#the-follow-nudge) | Leg control | `follow_nudge_rad_per_s = 12` | 28 |
| ⬜ | [A clock per segment](#a-clock-per-segment) | Leg control | `clocks = true` | 01–04 |
| ✅ | [The feet's physics](#the-feets-physics) | What a segment sees | `feet_physics = true` | 31– |
| ✅ | [Tilt and bend](#tilt-and-bend) | What a segment sees | `tilt = true` | 31– |
| ✅ | [Observation radius 2](#observation-radius) | What a segment sees | `observation_radius = 2` | 03– (1 in 01–02) |
| ✅ | [Leg accelerations](#leg-accelerations) | What a segment sees | `leg_accelerations = true` | 15– |
| ⬜ | [Neighbours' clocks](#neighbours-clocks) | What a segment sees | `neighbour_clocks = true` | 08–14, 17 |
| ✅ | [Neighbours' tempos](#neighbours-tempos) | What a segment sees | `neighbour_tempos = true` | 41– |
| ✅ | [Coloured exploration noise](#coloured-exploration-noise) | Exploration | `exploration_noise_beta = 1`, the neck only | 1, neck only, turns white: 57–; 1: 52, 54–56 (and 02, 04, 08, 09, 13); 2: 31–51, 53; 0.5: 10–12, 14–28 |
| ✅ | [Tempo noise that follows the gap](#tempo-noise-that-follows-the-gap) | Exploration | `[agents] follower_tempo_noise_gap_octaves = 0.17` | 51– |
| ⬜ | [CAPS, smooth mean actions](#caps-smooth-mean-actions) | Learning | `temporal_smoothness_coefficient` | 0.1: 14; 0, 0.3, 1, 3: 31–34; 1: 44–50 |
| ✅ | [A task critic for the head](#a-task-critic-for-the-head) | Learning | `[agents] head_task_critic_weight = 0.2` | 57– |
| ✅ | [A random first guess](#a-random-first-guess) | Learning | `initial_action_offset_spread = 0.2` | 31– |
| ✅ | [Curriculum](#curriculum) | Training | `[curriculum]`; `level_bonus = 0.05` (39–); the circle task (41–) | all |
| ✅ | [Plateau stop](#plateau-stop) | Training | `plateau_cycles = 48` | all |
| ✅ | [Range circle](#range-circle) | Task | `range_circle_ratio = 2.5` | all |
| ✅ | [Walking on after an arrival](#walking-on-after-an-arrival) | Task | `after_arrival = "new_target"` | all |
| ⬜ | [Retired reward terms](#retired-reward-terms) | Reward | each its own `_parts` | to 56 |

### Leg control

Cycle 2's principle (the user's, 2026-10-12): what the network outputs
reaches the joints unchanged, and the walk is learned, not configured. No
lag, no reflex, no fixed pattern between legs; the two models differ only in
how a joint's angle is made.

#### Model A, the network draws

✅ **Idea.** Each leg swings under its clock's hand, a simple step of
15 degrees of sweep and lift (the default step of run 28), and the network
adds its own angle to every joint, every 20 ms. The swing gives every leg a
rhythm from the start; the network bends it into a walk. Close to Policies
Modulating Trajectory Generators (Iscen et al., CoRL 2018). 9 actions per
follower (the heart and 8 angles), 10 for the head (and its neck).

<details>
<summary>Math</summary>

For each leg, with its hand `φ` and the network's angles `a`, in action units
of each joint's half range:

```math
\text{sweep} = \tfrac{15}{40}\cos\varphi + a_s, \qquad
\text{lift} = \tfrac{15}{32.5}\max(0,\,-\sin\varphi) + a_l, \qquad
\text{knee} = a_k, \qquad \text{ankle} = a_a
```

Stance is `φ` from 0 to π, swing from π to 2π.
</details>

#### Model B, the clock is the engine

✅ **Idea.** Each joint's angle is a curve of its leg's hand: a centre plus
two waves, at once and twice the heartbeat, a Fourier series. The network
sets the curve's 5 amounts per joint; the clock plays it. Until run 56 it
set them every 20 ms, so a walking leg played about 25 curves stitched
together per turn; from run 57 the curves are [held for a whole
turn](#curves-held-for-a-turn) of their segment. 41 actions per follower (the heart and 8
joints × 5), 42 for the head.

<details>
<summary>Math, and why two waves</summary>

For each joint, with its leg's hand `φ`:

```math
\theta \;=\; c \;+\; \sum_{k=1}^{2} \big(a_k \cos k\varphi \;+\; b_k \sin k\varphi\big)
```

in action units of the joint's half range. Two waves are enough: drawn
with 1 to 5 waves, run 28's constant-speed step slid 0.530, 0.394, 0.379,
0.382 and 0.383 against the step's own 0.373, and walked 11.5, 16.6, 17.4,
16.7 and 16.5 mm/s (`analysis/wave_step.py` in CentipedeTraining). The sweep
and the lift need both waves; the knee and the ankle keep them too, for
freedom (the user's choice).
</details>

#### The heart

✅ **Idea.** Each segment has one tempo, its heartbeat, shared by its two
legs. The segment's first action does not set the tempo, it changes it: at
most doubling or halving it per second, so the tempo cannot jump; with
held curves, the change comes at once as each turn starts. Each
episode every segment starts at its own random tempo between 1 and 4 Hz, and
each leg's hand at its own random place: nothing about the rhythm or the
phases between legs is given. Since run 35 the tempo has a ceiling of
12.5 Hz, a quarter turn per 20 ms step: without one (runs 31 to 34), the
heads, which pay no tempo cost, raced, up to 112 Hz in run 31's evaluation,
where a step jumps more than half a turn and samples the curve like a strobe
light, and the legs moved fast and without coordination. Real *S.
polymorpha* steps at about 6 Hz on flat ground (Diaz et al., J. Exp. Biol.
2023). There is no lower limit.

<details>
<summary>Math</summary>

```math
\log_2 f \;\leftarrow\; \min\!\Big(\log_2 f + a_{\text{tempo}} \cdot 1\ \tfrac{\text{octave}}{\text{s}} \cdot \Delta t,\ \log_2 12.5\Big), \qquad
\varphi \;\leftarrow\; \varphi + 2\pi f\,\Delta t, \qquad \Delta t = 0.02\ \text{s}
```

with the action from −1 to 1 and the start drawn uniformly in octaves. A
segment sees each of its hands as `cos φ`, `sin φ`, and its tempo in octaves
around 2 Hz.
</details>

#### A fixed tempo

⬜ **Idea.** The user's appendix test (2026-10-14, run 58): the heart
stops, and every clock turns at one tempo, 6 Hz, about *S. polymorpha*'s on
flat ground (Diaz et al., J. Exp. Biol. 2023). The segments lose their tempo
action, so nothing they output goes unread, and the rhythm leaves what they
learn: a body that cannot learn to walk with its rhythm given will not learn
it with its own. With every tempo equal, the [tempo chain](#tempo-chain),
the [neighbours' tempos](#neighbours-tempos) and the [tempo
noise](#tempo-noise-that-follows-the-gap) have nothing to do and are off.
Each segment's hand still starts at a random place. Run 57's heart had let
the tempo fall from 2.2 to 0.9 Hz, one way of moving less; at 6 Hz a turn
is 8.3 steps, so a 128-step window holds about 15 of each segment's turns
instead of 2.3.

#### Targets along the curve

✅ **Idea.** Within each 20 ms step a leg's target moves along its curve
instead of holding the step's first value: every physics step (134 per
action) sets the target a share of the way from the curve at the hand's
start to the curve at its end. The motors then follow a smooth line, not a
staircase of 50 jumps a second.

#### Curves held for a turn

✅ **Idea.** The user's rule (2026-10-13), made per segment (2026-10-14): a
step is judged as a whole, so it must be played whole. A segment's two legs'
curves are played by one hand, which starts at a random place when its
world restarts and turns at its tempo. On the step the hand starts a turn,
the segment takes both legs' curves (40 values) and its tempo action from
its actions, and plays them unchanged for the whole turn. One decision, one
turn, one judgement, and exploration that tries one whole step against
another. Each leg's timing is its own curve's: where its stance falls in
the turn is its curve's to choose, since a curve of two waves shifted in
time is another curve of two waves, so nothing ties the legs together, and
the starting curves are random. A hand per leg, at a random place each,
would only make the network turn its 20 values by an angle that changes
every episode; on one hand every curve starts at φ = 0, so starting a new
curve where the leg is means its centre plus its cosine amounts. A new
curve may still start away from where the leg is; no rule stops it, since
the lurch costs sliding or support.

The tempo action moves the [heart](#the-heart) only as a turn
starts, by as much as the heart would over one turn at the tempo it had
(action × 1 octave/s × the turn's length), so that every turn is played at
one tempo; the clocks' load feedback and follow nudge, which would turn one
leg's hand away from the other's, must be off.

What a segment sees is fitted to deciding a turn (the user's review,
2026-10-14), 210 values: its body and its 4 neighbours' (height, tilt and
bend, velocities, feet, leg angles and speeds; no leg accelerations, a 20 ms
snapshot of jerks that tells nothing about a whole turn), the target (the
head's), the spine joint behind, its hand and tempo once, its neighbours'
tempos, and last its turn's values: the 40 curve values it holds (the user's
"they see their own curve") and the step value's four checks averaged over
its last turn, how its last step went, which a critic that sees only one
instant of a rhythm could not tell.

<details>
<summary>How the agents learn from it</summary>

The network still acts every 20 ms, so that its critic sees every step,
but only the actions that took effect count: the environment's
`action_mask` marks all of a deciding segment's actions, and otherwise only
the head's neck command, which acts every step. PPO leaves the other actions
out of each sample's probability and entropy, and a sample that decided
nothing teaches only the critic (RL_lib's action masks). About one step in
25 is a decision at 2 Hz.

The rewards are still paid every step: the step value and the tempo chain
are sums over time, so a turn's steps add up to the turn's judgement. What
must match the turn is how the learning adds them up. With GAE's λ = 0.95
counted per step, a reward one turn after a decision (25 steps at 2 Hz)
would count only 0.27 directly, the rest left to the critic's guesses, so a
weak critic would judge a step mostly on its first 0.4 s. So λ is 1 inside
a turn and 0.95 only from one turn's start to the next: each decision's
advantage holds all its turn's discounted rewards plus the value where the
next turn starts, and turns are mixed with λ as steps were (variable λ,
Sutton and Barto, 2nd ed., 2018, section 12.8; a turn is one step of a
semi-Markov decision process, Sutton, Precup and Singh, 1999). The agents
find the turns' starts from the masks: the steps on which every action took
effect.
</details>

#### Learning from whole turns

⬜ **Idea.** The user's (2026-10-14, run 58): one action, held until the
next, so one sample per turn: what the segment saw as the turn started, the
actions it held, the turn's rewards added up, and where the next turn
starts. The head's neck is held for the turn too (`hold_spine`), so every
action is a turn's decision, and only the turns reach PPO: no masks, every
action explores with white noise, and the critic learns only the values at
turns' starts, the ones the decisions are judged by.

<details>
<summary>Math, and what it changes</summary>

For a turn that starts at step `t` and lasts `τ` steps:

```math
\hat A_t \;=\; \sum_{k=0}^{\tau-1} \gamma^{k} r_{t+k} \;+\; \gamma^{\tau} V(s_{t+\tau}) \;-\; V(s_t) \;+\; \gamma^{\tau} \lambda\, \hat A_{t+\tau}
```

GAE over turns, with the step's discount γ compounded over each turn
(a semi-Markov decision process, Sutton, Precup and Singh, 1999). For the
curves this is what run 57 already computed: with λ 1 inside a turn, the
values in its middle cancel out. What changes is everything around it:

- **Turns cut short are left out.** A turn the window's end or a restart
  cuts would be judged on part of its turn and a value in its middle, which
  the critic no longer learns; the turn before it then ends with
  `V(s_{t+τ})` alone. Run 57 judged the last decision of every world's
  window this way: at 0.9 Hz, 2.3 decisions per window, about 43% of them.
- **The critic learns where it is used**, at the turns' starts, instead of
  mostly in the middle of turns.
- **No masks reach PPO**, so the minibatches are the decisions themselves
  (run 57's fault, 128 Adam steps on about 150 decisions each, cannot
  return).

Code: `RolloutStorage.turn_batch` in the agents.
</details>

#### A clock per leg

✅ **Idea.** Each leg has a clock, a hand `φ` that turns at its segment's
tempo; with models A and B the hand drives the leg through the swing or the
curve. CPG-RL works this way (Bellegarda and Ijspeert, 2022).

#### Passive follower spine

✅ **Idea.** Only the head bends the spine joint behind it, its neck, to
steer; the followers' spine joints bend passively against their springs, so
the followers spend their actions on their legs.

#### The step shape

⬜ **Idea.** The legs of runs 08 to 28: the hand drives each leg through a
fixed step shape, sweeping back on the ground and coming forward lifted, and
the segment sets the tempo, the sweep and lift amplitudes and the joints'
centres. Replaced in cycle 2 because it decided the step's shape for the
network.

<details>
<summary>Math</summary>

```math
\text{sweep} = c_{s} + A_{s}\cos\varphi, \qquad
\text{lift} = c_{l} + A_{l}\,\max(0,\,-\sin\varphi), \qquad
\text{knee} = c_{k}\ (\text{and the ankle} = c_{a})
```

with the tempo `f = 2 · 2^a` Hz from the action, and in run 28 a default step
of 15 degrees added to the amplitudes.
</details>

#### The clock's lag

⬜ **Idea.** The clock in use followed the segment's actions slowly, with a
0.5 s time constant, so that the exploration noise did not make the feet
twitch (runs 11 to 28). Removed in cycle 2: the network's output reaches the
joints unchanged, and red noise and CAPS take over its job.

<details>
<summary>Math</summary>

```math
u \;\leftarrow\; u + \alpha\,(a - u), \qquad \alpha = 1 - e^{-\Delta t/\tau} = 0.039
```
</details>

#### Load feedback

⬜ **Idea.** The ground pushed the clocks: a loaded leg hurried toward
mid-stance and was held back after it (Owaki and Ishiguro, 2017). With the
step shape, a wave formed only with it (runs 15 against 16, 20 against 25).
Removed in cycle 2 with every reflex, so that the walk is learned; whether a
wave forms without it is the pilot's open question.

<details>
<summary>Math</summary>

```math
\frac{d\varphi}{dt} \;=\; \omega + \sigma N \cos\varphi, \qquad \sigma = 3\ \text{rad/s (run 28)}
```
</details>

#### The follow nudge

⬜ **Idea.** Each follower's leg timed its lift-off to the same-side leg
ahead, held back in late stance while that foot was down (Yasui et al., PLoS
ONE 2017). Run 28's wave passed down the whole body with it. Removed in
cycle 2 with the other reflex.

<details>
<summary>Math</summary>

```math
\frac{d\varphi_i}{dt} \;=\; \omega + \rho \,\max(0,\,-\cos\varphi_i)\,\mathbf{1}[\sin\varphi_i \ge 0]\,\big(1 - 2\,D_{i-1}\big), \qquad \rho = 12\ \text{rad/s (run 28)}
```
</details>

#### A clock per segment

⬜ **Idea.** One clock per segment, whose tempo the segment chose and whose
hand it saw, but which moved nothing: the legs followed it only if the
segment learned to. Runs 01 to 04 formed no wave; replaced by the clock per
leg.

### What a segment sees

#### The feet's physics

✅ **Idea.** Instead of a contact flag per foot, each segment feels what its
feet do: how much weight each carries, how fast each slides while planted,
and how well its feet carried it over its last stride. These are what the
health costs judge, so a segment can see what it pays for.

<details>
<summary>Math</summary>

Five values per segment. Each foot's load, its floor force averaged over the
step's physics steps, over the segment's own weight (its subtree's mass
minus the next segment's, times g):

```math
N_f = \frac{\bar F_f}{m_i\, g}
```

each foot's sliding speed while planted, in units of 16 mm/s, and the stride
support, the feet's summed load averaged over about one stride:

```math
S \;\leftarrow\; S + \big(1 - e^{-f\,\Delta t}\big)\,\big(N_{\text{left}} + N_{\text{right}} - S\big)
```

at the segment's tempo `f`. The force is averaged over the physics steps
because the step's last one alone read 1.39 body weights at rest, against
1.007 for the average.
</details>

#### Tilt and bend

✅ **Idea.** Instead of the world orientation, whose heading means nothing
to a segment, each segment sees how it tilts (two values: the world's
vertical in its frame) and how its spine joint ahead bends. Run 28's agents
used the world orientation (permutation importance 0.21 to 0.29) though its
heading was noise.

#### Observation radius

✅ **Idea.** Each segment observes its own body and its neighbours' within
two segments on each side (radius 1 in runs 01 and 02). A follower reads the
rhythm ahead only from these neighbours' legs and feet, never from their
clocks.

#### Leg accelerations

✅ **Idea.** Every observed segment's block ends with its leg joints'
accelerations over the last step, so that a follower can read where the leg
ahead is in its stride, not only where it is.

<details>
<summary>Math</summary>

```math
\ddot q \;=\; \frac{\dot q_{t} - \dot q_{t-1}}{\Delta t}
```
</details>

#### Neighbours' clocks

⬜ **Idea.** Each segment also saw the clocks of its neighbours, relative to
its own. In run 17 the right legs locked in phase and pressed instead of
stepping, so the clocks are read only from the legs; seeing them would also
give away a rhythm the segment should find.

#### Neighbours' tempos

✅ **Idea.** The user's choice (2026-10-13): each segment hears the tempo of
every neighbour it sees, as if it heard their heartbeats, so that it can
keep the [tempo chain](#tempo-chain) or [tempo ahead](#tempo-ahead) without
reading them from the legs. Only the tempo, never the hand: how fast a
neighbour steps, not when, so the offsets between neighbours stay free. With
radius 2, four values: the two segments ahead, then the two behind, each
tempo minus the segment's own in octaves, 0 where there is no neighbour.
They come last in the observation, so a run starts from one saved without
them: its networks are widened, the new inputs starting at zero weight.

### Exploration and learning

#### Coloured exploration noise

✅ **Idea.** The exploration noise drifts over time instead of changing
completely at every step, so an exploring leg tries a slightly different
movement for a while rather than jittering. Cycle 2 uses β = 2, red noise:
per step it changes less than the step's own motion, so model B's curves are
explored as curves (the twitching of runs 08 to 10 came from noise that
changed faster than the step). β = 1, pink noise (Eberhard et al., ICLR
2023), is the pilot's alternative. From run 57, with held curves, only
the actions that act every step keep the colour, the head's neck, which
steers: a held turn already keeps one curve for a whole step, which is what
the colour was for, and a turn spans about 25 steps of a coloured sequence,
so consecutive turns would explore alike. The held tempo and curves explore
with white noise, every turn a fresh trial (RL_lib's beta per action
component).

<details>
<summary>Math</summary>

Each action is sampled as u = μ(s) + σ(s)·ε; the sequence
of `ε` over 1,000 steps has power spectral density

```math
S(f) \;\propto\; \frac{1}{f^{\beta}}
```

while each step's `ε` stays standard normal, so the policy's probabilities are
unchanged. Code: RL_lib, `docs/colored-noise.md`.
</details>

#### Tempo noise that follows the gap

✅ **Idea.** The user's rule (2026-10-13): only the head explores the tempo
freely; a follower explores its tempo only while it is out of step with the
segment ahead, and its exploration fades as it falls into step. The tempo
chain still teaches it to copy the tempo ahead and to wait for the segment
behind. The reason is the heart: a follower's tempo noise adds up into a
drift of its tempo, about 0.3 octave per second with red noise at today's
spread, while a wave holds only when neighbours' tempos stay within about
0.006 octave (at 2 Hz, a 0.06-octave gap slides the phase offset a whole turn
every 12 s). The followers of runs 31 to 47 answered that drift by steering
to a fixed tempo of their own. People do something alike: their movements
vary more after a failure and less after a success (Pekny, Izawa and
Shadmehr, J Neurosci 2015).

<details>
<summary>Math</summary>

A follower's tempo action is drawn with its learned spread times

```math
s \;=\; \operatorname{clip}\!\Big(\frac{|g|}{g_0},\; 0.03,\; 1\Big), \qquad g_0 = 0.17 \text{ octave}
```

with `g` its tempo gap to the segment ahead, which it observes
([neighbours' tempos](#neighbours-tempos)), read at every step: after a
restart, when the clocks start at random tempos, and whenever the head
changes its tempo, the noise is whole; in step, it is 3% of it. At
`g_0 = 0.17`, with a spread near 0.35, the noise cannot move a follower's
tempo by more than its gap within one stride (about 0.5 s at 2 Hz). PPO
evaluates each sample again with the scale it was drawn with (RL_lib's
`SquashedGaussianPolicy`), so the scale is part of the policy at that
observation. The head's spread is never scaled. The learning log reports
each segment's mean scale over its last window (`tempo_noise_scale`).
</details>

#### CAPS, smooth mean actions

✅ **Idea.** A term in the actor's loss that keeps the policy's mean action
close from one step to the next (Mysore et al., ICRA 2021). In cycle 2 it
does the lag's old job by learning instead of filtering: model B's network
must keep a curve for a whole turn for the clock to play it. It smooths the
mean action, not the noise (run 14). Its strength is the pilot's first
question: 0, 0.3, 1 and 3 (runs 31 to 34); model A, with fewer and larger
actions, needs about half of B's strength for the same pull per action.

<details>
<summary>Math, and the pilot's rule</summary>

```math
\mathcal{L}_{\text{actor}} \;\mathrel{+}=\; \lambda_T \cdot \operatorname{mean}_i \big\lVert \pi(s_i) - \pi(s_i') \big\rVert_2
```

with `s'` the observation after `s`. The rule, fixed before the results: the
smallest `λ_T` whose curves hold, changing at most 10% of their size per
turn (`analysis/curve_hold.py` in CentipedeTraining, on the run's last
checkpoint), with a task level at cycle 64 at least 80% of the run without
CAPS. An untrained network's curves hold trivially (0.026 after 4 cycles,
its mean action barely depends on what it sees), so the measure means
something only once the network has learned. Code: RL_lib,
`docs/action-smoothness.md`.
</details>

#### A task critic for the head

✅ **Idea.** The user's (2026-10-13): the head has two rewards from two
jobs, its step and the task, so it learns each with a critic of its own.
The walk's critic predicts the step value and the tempo chain behind; the
task's predicts the arrivals and the progress. Each critic's advantage is
normalised over the batch on its own, and the policy learns from the walk's
plus 0.2 times the task's, so the weight alone sets the balance, whatever
the rewards' scales (van Seijen et al., NeurIPS 2017; Mysore et al., ICLR
2022). The followers have no task reward, so one critic each.

<details>
<summary>Math</summary>

```math
\hat A \;=\; \frac{A^{\text{walk}} - \mu^{\text{walk}}}{\sigma^{\text{walk}}} \;+\; 0.2\,\frac{A^{\text{task}} - \mu^{\text{task}}}{\sigma^{\text{task}}}
```

with each `A` from generalized advantage estimation of its own reward and
critic, and the means and spreads over the samples that decided something.
The learning log reports both critics' explained variance
(`explained_variance`, `task_explained_variance`).
</details>

#### A random first guess

✅ **Idea.** Model B's network starts with a curve of its own: each wave
amount starts at a random value within ±0.2, through the starting bias of
the actor's last layer, different for every segment. Nothing pulls it back;
the network changes it from the first update. Without it every curve would
start flat and every leg still.

### Training and task

#### Curriculum

✅ **Idea.** The targets get farther and wider as the agents arrive more
often. Cycle 2 starts easier, from 15–30 mm within ±45° (30–60 mm within
±90° before), so that a body still learning to step can reach its first
targets; the final ranges stay 30–200 mm all around. The level settles where
about half of the episodes arrive, so the level is the run's progress on the
task, and it also hardens the [health costs](#health-hardening).

**The circle task** (cycle 3, the user's idea, 2026-10-13). Every target of
a level lies at one distance, in any direction: 10 cm at level 0, growing
to 40 cm (8 body lengths) at level 1. Before, a level drew distances from a
range, so even at level 1 a third of the targets lay within 8 cm, 1.5 body
lengths, and the agents arrived 86% of the time without a wave. A target
allows 10 s, so each level asks for a speed toward the target, turn
included: 10 mm/s at level 0, 40 mm/s at level 1, more than twice run 35's
15 to 17. The level then reads as the body's speed.

<details>
<summary>Math</summary>

After every update cycle, with arrival share `a`:

```math
\ell \;\leftarrow\; \operatorname{clip}\big(\ell + 0.02\,(a - 0.5),\ 0,\ 1\big)
```

and each end of the target ranges lies a share `ℓ` of the way from the start
range to the final one.

**The level bonus** (`level_bonus = 0.05`, the user's idea, from the round
after runs 35 to 38; 0 is off). The rule above moves the level slowly: at
70% arrivals, 0.004 per cycle, 250 cycles from 0 to 1, though run 31,
trained at level 0.05, already arrived 58% and 84% at the final targets. So
the level also jumps by 0.05 whenever the agents keep arriving as the
targets get harder. At each check the arrival share `a` (the share of the
last episode length) must hold:

```math
a \;\ge\; 0.5 \quad\text{and}\quad a \;\ge\; a_{\text{last check}} - 0.02
```

The first check comes four episode lengths (32 cycles) into the run, so a
lucky run of easy targets cannot earn it; after a bonus the next comes two
episode lengths (16 cycles) later, one for the episodes started at the old
level to end and one to measure the new; after a failed check, 32 again.
0.02 is about the chance difference of two readings of about 1,000 episodes
each. Like the "bold driver" rule for learning rates (Vogl et al.,
Biological Cybernetics 1988; Battiti, Complex Systems 1989).
</details>

#### Plateau stop

✅ **Idea.** A run stops by itself, with a checkpoint, once it has stopped
improving, so no money goes to cycles that teach nothing. It watches the task
(arrivals plus level), the level alone, the return, and the wave.

<details>
<summary>Math</summary>

Each signal is averaged over its last episode lengths of readings. The run
stops when no average has risen by 0.02 above its best for 48 cycles, never
before cycle 128.
</details>

#### Range circle

✅ **Idea.** A head that leaves a circle around its target ends the episode,
like the time limit, so walking away does not go on forever.

<details>
<summary>Math</summary>

```math
R \;=\; 2.5\, d_{\text{start}} + 35\ \text{mm}
```

around the target.
</details>

#### Retired reward terms

⬜ Still in the code, each off unless its setting gives it parts; their
history is in the study notes.

| Term | Setting | Runs | What it charged |
| --- | --- | --- | --- |
| Time | `step_cost_parts` | to 56 | 1 part per step, every segment: the price of time; progress and arrival already make an early arrival worth more |
| Health hardening | `health_ramp_start`, `follower_health_scale` | 31–56 | the health costs grew from a share of their weight with the curriculum's level |
| Carry | `carry_cost_parts`, `carry_every_step` | 31–56 | the share of a segment's weight its feet did not carry; now C3 |
| Grip | `grip_cost_parts` | 31–50 | a loaded foot's sliding speed; now C2 |
| Foot hold | `foot_hold_cost_parts`, `foot_hold_looks_ahead` | 51–56 | grip plus the load a foot bore when it lifted, and with the look-ahead the load still on the feet ahead; the look-ahead decided an order between neighbours, so it was dropped |
| Leg pushing | `leg_pushing_cost_parts` | to 56 | the force between legs, in segment weights; now C4 |
| Tempo ahead | `tempo_ahead_cost_parts` | 31–39, 41, 43 | a follower's tempo gap to the segment ahead, an octave costing all; replaced by the tempo chain |
| Energy | `energy_cost_parts` | 43 | the leg motors' positive power, in mW; it changed nothing |

#### Walking on after an arrival

✅ **Idea.** After an arrival the body keeps walking toward a new target,
with its clocks running, as a real centipede would, and the targets reached
per minute count every arrival. Since run 57 the learning treats it so too
(`[agents] returns_continue_after_arrival`): an arrival ends no return, and
only a restart, at the time limit or outside the range circle, cuts them.
Until run 56 every arrival ended every segment's return as if the world had
stopped there, a cut the followers could neither see nor cause.
