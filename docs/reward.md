# The reward and the tools

<!-- Formulas go in ```math blocks only, never inline $...$: GitHub stops
     rendering the blocks of a <details> section that also holds inline math. -->

What the agents are paid for now, and every tool built around the reward.
This page describes the workspace branch `oscillators` as it trains today.

- **Part 1, the reward:** only the terms in use, grouped by kind, each with its
  idea, its math, and what to expect from it.
- **Part 2, the tools:** everything that is not a reward term, in use or
  tried before, so that any of them can be tried again.

**Current as of:** cycle 1's baseline, run 28
(`configs/oscillators/28_v5_cycle1.toml` in CentipedeTraining), on model v5.
The numbers measured at the end of a run are still run 26's (model v4), the
last baseline trained.

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
20 ms step, for every segment. Numbers are in units of one arrival reward.

| Group | Term | Head | Followers | Weight | Most per step | Run 26 at the end, per step |
| --- | --- | :---: | :---: | --- | --- | --- |
| 1. Task | [Arrival](#arrival) | ✔ | – | 0.5 per arrival | 0.5, on the arrival step | +0.00063 |
| 1. Task | [Progress](#progress) | ✔ | – | 0.359 per halving | about 0.003 while walking | +0.00278 |
| 2. Posture | [Body on the ground](#body-on-the-ground) | ✔ | ✔ | 2.5 parts | −0.00117 | head −0.000023, followers ≈ 0 |
| 2. Posture | [Leg pushing](#leg-pushing) | ✔ | ✔ | 2 parts | −0.00093 | new (legs touching, which it replaces: head −0.000023, followers −0.000013) |
| 3. Rhythm | [Following the clock ahead](#following-the-clock-ahead) | – | ✔ | 5 parts | −0.00234 | followers −0.00012, over six values (segment 1 −0.00037) |

So the head earns, and every segment pays for its posture; the followers
earn nothing and are judged only on their posture and on how well they copy
the clock of the segment ahead:

```math
r_0 \;=\; \underbrace{0.5\,\mathbf{1}[\text{arrived}] \;+\; 0.359\,P}_{\text{task}} \;-\; \underbrace{0.00117\,B_0 \;-\; 0.00093\,F_0}_{\text{posture}}
```

```math
r_i \;=\; -\,\underbrace{0.00117\,B_i \;-\; 0.00093\,F_i}_{\text{posture}} \;-\; \underbrace{0.00234\,C_i}_{\text{rhythm}} \qquad i = 1,\dots,11
```

`P` is the head's progress, `B` the body's contact flag (0 or 1), `F` the
force between the segment's legs and others (0 to 1), and `C` the follower's
clock mismatch (0 to 1), all defined below.

<details>
<summary><b>How the weights are set</b>: the cost budget, in parts</summary>

No weight is tuned by hand: each follows from a few rules and a number of
**parts**.

- **The discount fits the episode.** With episodes of `T = 512` steps, the
  agents discount by γ = 2^(−1/T) = 0.998647: an arrival on the last
  step of an episode is worth half as much, seen from its first.
- **The worst arrival balances to zero.** Over a **cost horizon** of
  `H = 256` steps (a typical approach, not the time limit), an approach that
  pays *every* cost on every step and arrives on its last step adds up to zero,
  seen from its start. The per-step cost `c` of all the costs together is
  then

```math
\sum_{t=0}^{H-1} \gamma^{t}\, c \;=\; \gamma^{H-1} A
\quad\Longrightarrow\quad
c \;=\; \frac{\gamma^{H-1}\,(1-\gamma)}{1-\gamma^{H}} \;=\; 0.003271 \text{ per step}
```

  with `A = 1`, the arrival reward. Over the horizon this budget is
  H·c = 0.837 arrivals.
- **The budget is split into parts.** `cost_budget_parts = 7`, so **one part
  costs 0.000467 per step** (0.120 over the horizon). A cost with `p` parts
  weighs

```math
w \;=\; p \cdot \frac{c}{7} \;=\; p \times 0.000467 \text{ per step}
```

- **Progress is counted in parts too.** One halving of the head's distance is
  worth `progress_parts = 3` parts of the whole horizon:
  3 × H·c / 7 = 0.359.
- **The arrival pays half.** `arrival_payout = 0.5`: an arrival pays 0.5, but
  the budget above is measured against a full arrival of 1, so the costs weigh
  twice as much against what an arrival actually pays.

The budget stays at 7 parts, so that adding or removing a cost leaves the
others' weights as they are. The head's costs take 4.5 parts of it; a
follower's take 9.5, more than the budget, so a follower that paid every cost
on every step would end below zero. The followers earn nothing anyway: for
them only the difference between one behaviour and another counts.

Settings: `[environment.rewards]` in the run's file; the program prints every
weight under the settings when a run starts. Code:
`src/centipede/environment/settings.py`, `RewardSettings.weights`.
</details>

### 1. Task terms

What the centipede is for: reaching targets. Only the head sees the target,
and only the head is paid for it (`follower_arrival_share = 0`,
`follower_progress_share = 0`).

#### Arrival

**Idea.** The goal itself: the head brings the target under its body.

<details>
<summary>Math, and what to expect</summary>

```math
\text{arrival}_0 = \mathbf{1}[\text{the target lies under the head's outline, seen from above}], \qquad w = 0.5
```

Paid once, on the step it happens; that step ends the episode for learning,
and the body walks on to a new target with its clocks still running. The
followers receive 0.

**What to expect.** Its part per step is 0.5 times the arrivals per step: it
grows with the targets reached per minute. While the curriculum climbs, about
half of the episodes arrive by construction, so this part rises slowly; it
grows faster only once the targets stop getting farther. Run 26 ended at
+0.00063 per step on the head.
</details>

#### Progress

**Idea.** Every step that brings the head closer pays, in **halvings** of its
distance: ending closer is better than not trying, wiggling earns nothing, and
walking past the target gives back what was earned.

<details>
<summary>Math, and what to expect</summary>

```math
P \;=\; \log_2 \frac{\max(d_{t-1},\, r)}{\max(d_{t},\, r)}, \qquad w = 0.359 \text{ per halving}
```

`d` is the flat distance from the head's tip to the target, before and after
the step, and `r = 4 mm`, half the head's width: closer than that, nothing pays
more. Over an episode the steps add up to

```math
\sum_t P \;=\; \log_2 \frac{d_{\text{start}}}{\max(d_{\text{end}},\, r)}
```

whatever the path. An approach from 45 mm is worth log₂(45/4) = 3.5 halvings, so 1.25;
from 200 mm, the curriculum's farthest, 5.6 halvings, so 2.0, four times the
arrival's 0.5.

**What to expect.** The largest part of the head's reward: about +0.003 per
step while it walks toward the target, 0 when it stands still, negative when
it walks away. Run 26 ended at +0.00278 on the head. It wins against every
cost: the head walks whatever the body does (lesson 5 of the study notes).
</details>

### 2. Posture terms

How the body carries itself. Every segment pays its own.

#### Body on the ground

**Idea.** A walking centipede carries its body on its legs; dragging the belly
costs.

<details>
<summary>Math, and what to expect</summary>

```math
B_i = \mathbf{1}[\text{segment } i\text{'s body touches the ground}], \qquad w = 2.5 \times 0.000467 = 0.00117
```

**What to expect.** Small, because the bodies stay up: run 26 ended at
−0.000023 on the head (2% of its most) and about 0 on the followers. A sudden
rise means the body has started to crawl.
</details>

#### Leg pushing

**Idea.** In the real animal legs touching is no problem; legs that push on
each other are, since they block the stride. So the cost is the force
between legs, not the touch: both segments of a contact pay, in proportion
to how hard they press, and legs that brush past each other pay almost
nothing. It replaces legs touching (runs 08 to 27), a flag that charged every
touch alike. The user's decision of 2026-10-12; a run without it is planned.

<details>
<summary>Math, and what to expect</summary>

```math
F_i = \min\!\Big(1,\ \frac{\sum \text{normal force of the contacts between one of segment } i\text{'s legs and another leg}}{m g / N}\Big), \qquad w = 2 \times 0.000467 = 0.00093
```

`m g / N` is the weight one segment carries, 0.86 mN on v5, about the
strongest push a leg's sweep motor makes at its foot. A full push costs what
a touch cost before, at the same 2 parts.

**What to expect.** Open loop on v5 (clocks at random hands, no reflexes),
the legs touched in 4% of segment-steps at a 15-degree swing and 29% at 25;
when they touched they pressed with 1.3 and 2.7 segment weights at the
median, so about two thirds of touching steps pay the whole cost and a third
less. With the reflexes the legs touch under 1.5% of the time at swings up
to 25 degrees, so the part should stay small unless the strides grow or the
body turns hard. Both backends sum the forces (`leg_leg_force` in the
physical state).
</details>

### 3. Rhythm terms

How the legs keep time. Each leg has a clock that drives it through a step
shape (Part 2); these terms judge the clocks, never the phase, so the offset
between neighbours, the wave, stays free to emerge.

#### Following the clock ahead

**Idea.** Each follower copies the clock **in use** of the same-side leg of
the segment ahead: its tempo and the size of its step, the sweep and lift
amplitudes, but not its phase and not its posture. The head, with nothing
ahead, sets the clock, and it passes down the body one segment at a time.
Charged on the absolute difference, so that small mismatches still cost (the
squared form left them nearly free). The joint centres left the comparison
in cycle 1 (`clock_tracking_centres = false`): copying the posture turned the
whole body at once and forbade useful differences along it.

<details>
<summary>Math, and what to expect</summary>

Each leg's clock in use holds the tempo (in octaves, over the range of one
octave) and the sweep and lift amplitudes, each from −1 to 1 (with the
default step added, so they may reach a little past 1). With one tempo per
segment both legs share the tempo. For follower `i`, over its two legs and
the three values:

```math
C_i \;=\; \frac{1}{6} \sum_{\ell \in \{\text{left},\,\text{right}\}} \;\sum_{k=1}^{3} \frac{\lvert u_{i,\ell,k} - u_{i-1,\ell,k} \rvert}{2},
\qquad C_0 = 0,
\qquad w = 5 \times 0.000467 = 0.00234
```

`C = 0` copies the segment ahead exactly; `C = 1` is the largest possible
mismatch.

**Why 5 parts.** The weight sets what a follower trades between copying the
stride ahead and pushing its neighbours' legs. A follower 23 degrees short
of a sweep ahead pays (23/40)/2 over three values, about 0.1 of the term,
0.48 parts per step, 12 part-steps over a 2 Hz step of 25 steps; copying a
stride that makes the legs push fully for a quarter of the step costs about
6 × 2 = 12 part-steps of leg pushing. So a follower copies unless copying
makes it push for more than about a quarter of every step.

**What to expect.** Run 26, over six values, ended at −0.00012 per step on
the followers on average: segment 1 −0.00037 (16% of its most), the others
3% to 4%, because the head's sweep was large and segment 1's small. Over
three values the same mismatches weigh twice as much. Close to 0 everywhere
means either a body that copies well or one whose clocks are all alike and
still: check the sweeps (lesson 9 of the study notes).
</details>

---

## Part 2. The tools

Everything around the reward: how the legs are driven, what each segment
sees, how the agents explore, and how training is scheduled.
✅ is in the baseline; ⬜ is not in use now. The runs are those of the study
notes (`results/oscillators/README.md` in CentipedeTraining).

| | Tool | Kind | Setting | Runs |
| :---: | --- | --- | --- | --- |
| ✅ | [A clock per leg, with its step shape](#a-clock-per-leg-with-its-step-shape) | Leg control | `leg_clocks = true` | 08–27 |
| ✅ | [The clock's lag](#the-clocks-lag) (the action filter) | Leg control | `clock_time_constant_s = 0.5` | 11–13, 15–27 |
| ✅ | [Load feedback](#load-feedback) | Leg control | `load_feedback_rad_per_s = 3`, `load_feedback_source = "force"` | contact: 09, 12, 15, 17, 18; force: 19–24, 26, 27 (6 rad/s), 28 (3) |
| ✅ | [The follow nudge](#the-follow-nudge) | Leg control | `follow_nudge_rad_per_s = 12` | 28 |
| ✅ | [One tempo per segment](#one-tempo-per-segment) | Leg control | `segment_tempo = true` | 28 |
| ✅ | [The default step](#the-default-step) | Leg control | `default_swing_deg = 15`, `default_lift_deg = 15` | 28 |
| ✅ | [Passive follower spine](#passive-follower-spine) | Leg control | `passive_follower_spine = true` | all |
| ⬜ | [A clock per segment](#a-clock-per-segment) | Leg control | `clocks = true` | 01–04 |
| ✅ | [Observation radius 2](#observation-radius) | What a segment sees | `observation_radius = 2` | 03–27 (1 in 01–02) |
| ✅ | [Leg accelerations](#leg-accelerations) | What a segment sees | `leg_accelerations = true` | 15–27 |
| ⬜ | [Neighbours' clocks](#neighbours-clocks) | What a segment sees | `neighbour_clocks = true`, `coupling = "ahead"` | 08–14, 17 |
| ✅ | [Coloured exploration noise](#coloured-exploration-noise) | Exploration | `exploration_noise_beta = 0.5` | 0.5: 10–12, 14–27; 1: 02, 04, 08, 09, 13 |
| ⬜ | [CAPS, smooth mean actions](#caps-smooth-mean-actions) | Learning | `temporal_smoothness_coefficient = 0.1` | 14 |
| ✅ | [Curriculum](#curriculum) | Training | `[curriculum]` | all |
| ✅ | [Plateau stop](#plateau-stop) | Training | `plateau_cycles = 48` | all |
| ✅ | [Range circle](#range-circle) | Task | `range_circle_ratio = 2.5` | all |
| ✅ | [Walking on after an arrival](#walking-on-after-an-arrival) | Task | `after_arrival = "new_target"` | all |

### Leg control

#### A clock per leg, with its step shape

✅ **Idea.** Each leg has a clock, a hand `φ` that turns at a tempo the
segment chooses, and the hand drives the leg through a fixed step shape: the
foot sweeps back on the ground (stance), then comes forward lifted (swing).
The segment sets each leg's tempo, step sizes and posture; how the legs keep
time together is learnt. CPG-RL works this way (Bellegarda and Ijspeert, 2022).

<details>
<summary>Math</summary>

Each leg's actions `a` set its tempo, sweep and lift amplitudes, and its
joints' centres: sweep, lift and knee, and on model v5 the ankle (with one
tempo per segment, the segment's first action sets both legs' tempo). The
tempo in Hz, from 1 to 4:

```math
f = 2 \cdot 2^{\,a_{\text{tempo}}} \qquad \varphi \leftarrow \varphi + 2\pi f\,\Delta t \quad (\Delta t = 0.02\ \text{s, plus load feedback})
```

The joint targets, in action units mapped onto each joint's range:

```math
\text{sweep} = c_{s} + A_{s}\cos\varphi, \qquad
\text{lift} = c_{l} + A_{l}\,\max(0,\,-\sin\varphi), \qquad
\text{knee} = c_{k}\ (\text{and the ankle} = c_{a})
```

Stance is `φ` from 0 to π, swing from π to 2π. A segment sees each of its
legs' hands as `cos φ`, `sin φ` and its tempo.
</details>

#### The clock's lag

✅ **Idea.** The clock in use follows the segment's actions slowly, all six
values alike, so the exploration noise added at every step does not make the
feet twitch 50 times a second. Without it (runs 08 to 10) the agents shrank
their steps to avoid the twitching; with it (from run 11) the foot slip
halved. This is the action filter of runs 11 to 13, which then lagged the
amplitudes and centres; it now lags the whole clock, the tempo too. Run 14
switched it off for CAPS.

<details>
<summary>Math</summary>

A first-order lag with time constant `τ = 0.5 s`:

```math
u \;\leftarrow\; u + \alpha\,(a - u), \qquad \alpha = 1 - e^{-\Delta t/\tau} = 0.039
```

so 63% of a sudden change is covered after 0.5 s. Each segment observes its
clocks in use, which keeps its inputs Markov (Mysore et al., 2021).
</details>

#### Load feedback

✅ **Idea.** The ground pushes the clocks: a leg that carries load hurries
toward mid-stance and is held back after it, so it stays on the ground while
the others take their turn. Our best friend so far: with clocks driving the
legs, a wave formed only with it (runs 15 against 16, 20 against 25). From Owaki and
Ishiguro's quadruped (2017), whose oscillators interact only through the body.
Along the body it pulls neighbours toward alternation, the short waves that
limited the stride; across it, with one tempo per segment, it alternates a
segment's two legs, as the shared weight holds one down while the other is
lifted. Since cycle 1, σ = 3 rad/s: the open-loop sweep on v5
(`analysis/open_loop_wave.py`, `reflex_*_v5*.json` in CentipedeTraining)
found that 6 nearly stops the clocks at 1 Hz, and that without it a
segment's two legs keep whatever gap they started with.

<details>
<summary>Math</summary>

```math
\frac{d\varphi}{dt} \;=\; \omega + \sigma N \cos\varphi, \qquad \omega = 2\pi f, \quad \sigma = 3\ \text{rad/s}
```

`N` is the foot's load at the end of the last step, from 0 to 1: with
`"force"`, its normal force over the weight one segment carries,
N = min(1, Fₙ / (m·g / segments)); with `"contact"`, 1 when the foot
touches the ground. Stance is centred at `φ = π/2`, where `cos φ` changes
sign. σ must stay below the slowest `ω`, 2π rad/s, or a loaded clock could
stop in stance for good.
</details>

#### The follow nudge

✅ **Idea.** Each follower's leg times its lift-off to the same-side leg of
the segment ahead: in the second half of its stance it is held back while
that foot is on the ground and hurried while it is lifted, so it lifts soon
after the leg ahead. The physics, not the reward, then makes a wave from head
to tail, and each segment still sees only its neighbours. Real *Scolopendra*
legs over a gap stop stepping until the legs ahead touch the ground, and
Yasui et al. (PLoS ONE 2017) model each leg reacting to its own and its
neighbours' foot forces. Open loop on v5 it makes a uniform wave of 7 to 11
legs at 1 to 4 Hz and swings of 8 to 25 degrees, with the legs almost never
touching; without it no case held a wave.

<details>
<summary>Math, and how ρ was chosen</summary>

```math
\frac{d\varphi_i}{dt} \;=\; \omega + \sigma N_i \cos\varphi_i \;+\; \rho \,\max(0,\,-\cos\varphi_i)\,\mathbf{1}[\sin\varphi_i \ge 0]\,\big(1 - 2\,D_{i-1}\big), \qquad \rho = 12\ \text{rad/s}
```

`D` is 1 while the same-side foot of the segment ahead is on the ground
(before the step), else 0; the head is not nudged. A held hand may stop, or
turn back a little, until the foot ahead lifts. ρ is the weakest strength
that held the wave in every case of the open-loop sweep (rule A, the user's
choice, fixed before the results): three tempos by three swings, and every
segment's tempo up to 10% off over four random draws, on the 12-unit body
and on a 21-unit test body. The stronger the nudge, the larger the tempo
difference it pulls into step; ρ = 10 failed one draw on 21 units.
</details>

#### One tempo per segment

✅ **Idea.** A segment works at one frequency, like a heartbeat: one action
sets the tempo of both its legs' clocks, while each leg keeps its own phase,
step size and posture. With a tempo per leg the two sides formed separate
chains and drifted apart (run 26: 1.8 to 2.0 Hz on the left, 2.1 to 2.4 on
the right). The head steers with its step sizes and its neck instead of a
tempo difference. The user's decision of 2026-10-12.

#### The default step

✅ **Idea.** An action of 0 asks for a step, 15 degrees of swing and of lift,
instead of standing still, so that every segment steps from the start and
none begins as an anchor. 15 degrees is the largest swing whose pinch stays
below the legs' touching limit in the reflexes' wave. Exploration noise
spreads the early steps around it, about ±20 degrees.

<details>
<summary>Math</summary>

The default is added to the amplitude actions before the clock's lag, in
action units of each joint's half range:

```math
A_{s} = a_{s} + \frac{15^\circ}{40^\circ}, \qquad A_{l} = a_{l} + \frac{15^\circ}{32.5^\circ}
```

on model v5, whose sweep spans ±40 degrees and lift −25 to 40.
</details>

#### Passive follower spine

✅ **Idea.** Only the head bends the spine joint behind it, its neck, to
steer; the followers' spine joints bend passively against their springs, so
the followers spend their actions on their legs.

#### A clock per segment

⬜ **Idea.** One clock per segment, whose tempo the segment chooses and whose
hand it sees, but which moves nothing: the legs follow it only if the segment
learns to. Runs 01 to 04 kept a common tempo but formed no wave; replaced by
the clock per leg.

<details>
<summary>Math</summary>

```math
f = 2 \cdot 2^{\,a}, \qquad \varphi \leftarrow \varphi + 2\pi f\,\Delta t
```

taken at once from the action, with no lag.
</details>

### What a segment sees

#### Observation radius

✅ **Idea.** Each segment observes its own body and its neighbours' within
two segments on each side (radius 1 in runs 01 and 02). A follower reads the
rhythm ahead only from these neighbours' legs.

#### Leg accelerations

✅ **Idea.** Every observed segment's block ends with its six leg joints'
accelerations over the last step, so that a follower can read where the leg
ahead is in its stride, not only where it is.

<details>
<summary>Math</summary>

```math
\ddot q \;=\; \frac{\dot q_{t} - \dot q_{t-1}}{\Delta t}
```

for each of the six leg joints.
</details>

#### Neighbours' clocks

⬜ **Idea.** Each segment also sees the clocks of its neighbours, relative to
its own: with `coupling = "ahead"`, only those ahead. Runs 08 to 14 and 17
used it; in run 17 the right legs locked in phase and pressed instead of
stepping, so the clocks are now read only from the legs.

<details>
<summary>Math</summary>

For neighbour `n` of segment `i`, each leg against the same side's:

```math
\big(\cos(\varphi_n - \varphi_i),\ \sin(\varphi_n - \varphi_i),\ (o_n - o_i)/2\big)
```

with `o` the tempo in octaves.
</details>

### Exploration and learning

#### Coloured exploration noise

✅ **Idea.** The exploration noise drifts over time instead of changing
completely at every step, so an exploring leg tries a slightly different
movement for a while rather than jittering. β = 0.5 is the PPO default of
Hollenstein et al. (2024); β = 1, pink noise (Eberhard et al., 2023), was
tried in runs 02, 04, 08, 09 and 13.

<details>
<summary>Math</summary>

Each action is sampled as u = μ(s) + σ(s)·ε; the sequence
of `ε` over 1,000 steps has power spectral density

```math
S(f) \;\propto\; \frac{1}{f^{\beta}}
```

while each step's `ε` stays standard normal, so the policy's probabilities are
unchanged. At β = 0.5 consecutive steps correlate by 0.36; at β = 1, by 0.77.
Code: RL_lib, `docs/colored-noise.md`.
</details>

#### CAPS, smooth mean actions

⬜ **Idea.** A term in the actor's loss that keeps the policy's mean action
close from one step to the next (Mysore et al., 2021). Run 14 used it instead
of the clock's lag: it smooths the mean action but not the noise, and the
foot slip did not fall as it did with the lag.

<details>
<summary>Math</summary>

```math
\mathcal{L}_{\text{actor}} \;\mathrel{+}=\; \lambda_T \cdot \operatorname{mean}_i \big\lVert \pi(s_i) - \pi(s_i') \big\rVert_2, \qquad \lambda_T = 0.1
```

with `s'` the observation after `s`. Code: RL_lib, `docs/action-smoothness.md`.
</details>

### Training and task

#### Curriculum

✅ **Idea.** The targets get farther and wider as the agents arrive more
often, from 30–60 mm within ±90° to 30–200 mm all around. The level settles
where about half of the episodes arrive, so the level is the run's progress
on the task.

<details>
<summary>Math</summary>

After every update cycle, with arrival share `a`:

```math
\ell \;\leftarrow\; \operatorname{clip}\big(\ell + 0.02\,(a - 0.5),\ 0,\ 1\big)
```

and each end of the target ranges lies a share `ℓ` of the way from the start
range to the final one.
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

#### Walking on after an arrival

✅ **Idea.** After an arrival the body keeps walking toward a new target,
with its clocks running, as a real centipede would, and the targets reached
per minute count every arrival.
