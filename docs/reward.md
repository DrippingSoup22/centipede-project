# The reward and the tools

<!-- Formulas go in ```math blocks only, never inline $...$: GitHub stops
     rendering the blocks of a <details> section that also holds inline math. -->

What the agents are paid for now, and every tool built around the reward.
This page describes the workspace branch `oscillators` as it trains today.

- **Part 1, the reward:** only the terms in use, grouped by kind, each with its
  idea, its math, and what to expect from it.
- **Part 2, the tools:** everything that is not a reward term, in use or
  tried before, so that any of them can be tried again.

**Current as of:** cycle 2's pilot (`configs/oscillators/31_cycle2_b_caps0.toml`
to `34_...` in CentipedeTraining), on model v5. The pilot compares two ways of
driving the legs (models A and B), CAPS strengths, noise colours and two
versions of the task reward; the values marked *measured* come from open-loop
checks on the laptop's CPU, not from training.

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

**The anchor rule** (the user's, 2026-10-12): a perfect walk earns about 1,
a walk that does everything wrong about 0, and standing still a little below
0. The task is the team's: every segment earns the head's arrival and
progress, though only the head sees the target. The costs judge how each
segment walks, and the health costs harden as the curriculum's targets get
harder, so that early on the agents may walk badly before they walk well.

| Group | Term | Head | Followers | Weight | Most per step |
| --- | --- | :---: | :---: | --- | --- |
| 1. Task | [Arrival](#arrival) | ✔ | ✔ (shared) | P: 0.5, T: 1 per arrival | the weight, on the arrival step |
| 1. Task | [Progress](#progress) | ✔ | ✔ (shared) | P: 0.100 per halving, T: none | about 0.0005 while walking |
| 1. Task | [Time](#time) | ✔ | ✔ | 1 part | −0.00047 |
| 2. Health | [Carry](#carry) | ✔ | ✔ (half) | 3 parts × h | −0.00140 × h |
| 2. Health | [Grip](#grip) | ✔ | ✔ (half) | 2 parts × h | −0.00093 × h |
| 2. Health | [Leg pushing](#leg-pushing) | ✔ | ✔ (half) | 2 parts × h | −0.00093 × h |
| 3. Rhythm | [Tempo ahead](#tempo-ahead) | – | ✔ | 3.5 parts | −0.00164 |

`h` is the [health hardening](#health-hardening), from 0.3 at the first
targets to 1 at the final ones. The two task versions, **P** (arrival 0.5 and
progress) and **T** (arrival 1 alone), are both in the pilot:

```math
r_i \;=\; \underbrace{w_A\,\mathbf{1}[\text{arrived}] \;+\; w_P\,P}_{\text{task, every segment}} \;-\; \underbrace{0.00047}_{\text{time}} \;-\; h_i\,\underbrace{\big(0.00140\,K_i \;+\; 0.00093\,G_i \;+\; 0.00093\,F_i\big)}_{\text{health}} \;-\; \underbrace{0.00164\,D_i}_{\text{rhythm, followers}}
```

with `w_A = 0.5`, `w_P = 0.100` in version P and `w_A = 1`, `w_P = 0` in
version T. `P` is the head's progress, `K` the share of its weight a segment's
feet did not carry, `G` its loaded feet's sliding, `F` the force between its
legs and others, and `D` its tempo's distance from the segment ahead's, all
from 0 to 1 and defined below. `h_0 = 0.3 + 0.7ℓ` for the head and half of it
for the followers, at the curriculum's level `ℓ`.

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

- **Each segment's worst costs one arrival.** The head's health costs take
  3 + 2 + 2 = 7 parts; a follower's, at half strength, 3.5, and its tempo
  cost 3.5 more: 7 parts each. The time cost adds 1 part for everyone. So a
  segment that does everything wrong on every step, at the final targets,
  ends near zero even when the head arrives: the anchor rule's "about 0".
- **Progress is counted in parts too** (version P). One halving of the
  head's distance is worth `progress_parts = 0.84` parts of the whole
  horizon: 0.84 × H·c / 7 = 0.100. An approach from 30 mm is worth 2.9
  halvings, 0.29; from 200 mm, the curriculum's farthest, 5.6 halvings, 0.56.
  With the arrival's 0.5, a perfect approach earns 0.8 to 1.06, the anchor
  rule's "about 1".

Settings: `[environment.rewards]` in the run's file; the program prints every
weight under the settings when a run starts. Code:
`src/centipede/environment/settings.py`, `RewardSettings.weights`.
</details>

<details>
<summary><b>What to expect</b>: two open-loop checks (measured)</summary>

On the laptop's CPU, 8 worlds, 256 steps, the pilot's settings, the first
targets (1.5 to 3 cm within ±45°); per step, in hundred-thousandths.
`scratchpad` checks of 2026-10-12, not training results.

| Term | Still body (model B, every amount 0), head / followers | Model A's default step (every action 0), head / followers |
| --- | ---: | ---: |
| Arrival (P) | 0 / 0 | +97.7 / +97.7 |
| Progress (P) | 0 / 0 | +23.1 / +23.1 |
| Time | −46.7 / −46.7 | −46.7 / −46.7 |
| Carry, h at level 0 → 1 | −0.7 → −2.4 / −0.9 → −3.0 | −3.4 → −11.3 / −2.1 → −6.9 |
| Grip, level 0 → 1 | −0.1 → −0.4 / 0 → −0.1 | −5.8 → −19.5 / −3.1 → −10.3 |
| Leg pushing, level 0 → 1 | 0 / 0 | −0.9 → −2.9 / −1.1 → −3.7 |
| Tempo ahead | 0 / −100.3 | 0 / −100.3 |
| **Over 256 steps** | **−0.12 / −0.38** | **+0.16 → +0.10 / −0.08 → −0.12** |

Standing still costs a follower −0.38 over the horizon, two thirds of it its
tempo's distance from the random tempo ahead, which it can close; the head
pays only time. The default step walks the head to some of the first,
nearest targets by chance. In every case the time and tempo costs are the
largest at the start, and the health costs grow with the level.
</details>

### 1. Task terms

What the centipede is for: reaching targets. Only the head sees the target,
but the whole team is paid for it (`follower_arrival_share = 1`,
`follower_progress_share = 1`): a follower that helps the head arrive earns
what the head earns. Before cycle 2 only the head earned, and standing still
was the followers' best choice (run 27; the cycle-2 design notes).

#### Arrival

**Idea.** The goal itself: the head brings the target under its body.

<details>
<summary>Math, and what to expect</summary>

```math
\text{arrival} = \mathbf{1}[\text{the target lies under the head's outline, seen from above}], \qquad w_A = 0.5 \text{ (P) or } 1 \text{ (T)}
```

Paid once, on the step it happens, to every segment; that step ends the
episode for learning, and the body walks on to a new target with its clocks
still running.

**What to expect.** Its part per step is `w_A` times the arrivals per step:
it grows with the targets reached per minute. While the curriculum climbs,
about half of the episodes arrive by construction, so this part rises slowly;
it grows faster only once the targets stop getting farther.
</details>

#### Progress

**Idea.** Version P only. Every step that brings the head closer pays, in
**halvings** of its distance: ending closer is better than not trying,
wiggling earns nothing, and walking past the target gives back what was
earned. Version T leaves it out (`progress_parts = 0`), so that only arriving
pays; the pilot tests whether the hint is needed.

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

#### Time

**Idea.** Every step costs a little, for every segment: a body that stands
still slowly loses, and a quick arrival keeps more. It is the only cost that
does not depend on how the segment moves.

<details>
<summary>Math, and what to expect</summary>

```math
-1 \text{ on every step}, \qquad w = 1 \times 0.000467
```

**What to expect.** −0.00047 per step on every segment, always: −0.12 over
the 256-step horizon.
</details>

### 2. Health terms

How each segment walks: does it carry itself, do its feet grip, do its legs
stay clear of the others'. The followers pay half (`follower_health_scale =
0.5`), so that their tempo matters as much as their health. All three grow
with the curriculum's level.

#### Health hardening

**Idea.** A beginner may walk clumsily: the health costs start at 30% of
their weight and grow with the curriculum's level to the full weight at the
final targets (`health_ramp_start = 0.3`). The same idea as Hwangbo et al.'s
cost curriculum (Science Robotics 2019, which starts at 0.3), tied to the
level instead of the training time, so that the costs harden only as the
walk improves. The time and tempo costs are at full weight from the start.

<details>
<summary>Math</summary>

```math
h_0 \;=\; 0.3 + 0.7\,\ell, \qquad h_{i \ge 1} \;=\; 0.5\,h_0
```

with `ℓ` the curriculum's level, 0 to 1, set after every update cycle.
</details>

#### Carry

**Idea.** Each segment carries its own weight on its own feet. A segment held
up by its neighbours, legs in the air, pays for the weight its feet did not
carry; one that carries more than its own pays nothing (that is its
neighbours' business).

<details>
<summary>Math, and what to expect</summary>

```math
K_i \;=\; \max\!\big(0,\ 1 - S_i\big), \qquad w = 3 \times 0.000467 = 0.00140
```

`S` is the segment's [stride support](#the-feets-physics): its feet's floor
force over its own weight, averaged over its last stride.

**What to expect.** Small for a body that stands or steps evenly (measured:
−0.000007 to −0.00011 per step); large for a segment that hangs from its
neighbours, as the anchored followers of runs 08 to 13 did.
</details>

#### Grip

**Idea.** A foot that carries weight should not slide: a loaded foot dragged
along the ground wastes the push. A foot that slides while bearing nothing,
brushing the floor in its swing, costs nothing.

<details>
<summary>Math, and what to expect</summary>

```math
G_i \;=\; \min\!\Big(1,\ \frac{\sum_{\text{feet}} N_f\, v_f}{16\ \text{mm/s}}\Big), \qquad w = 2 \times 0.000467 = 0.00093
```

`N` is each foot's load (its floor force over the segment's own weight) and
`v` its sliding speed, for a foot on the ground at both ends of the step.
16 mm/s is about a default stroke's foot speed.

**What to expect.** Near 0 for a still body; for the default step a few
hundred-thousandths at the first targets, three times more at the final ones
(measured). It falls as the stance foot learns to stay planted.
</details>

#### Leg pushing

**Idea.** In the real animal legs touching is no problem; legs that push on
each other are, since they block the stride. So the cost is the force
between legs, not the touch: both segments of a contact pay, in proportion
to how hard they press, and legs that brush past each other pay almost
nothing.

<details>
<summary>Math, and what to expect</summary>

```math
F_i = \min\!\Big(1,\ \frac{\sum \text{normal force of the contacts between one of segment } i\text{'s legs and another leg}}{m g / N}\Big), \qquad w = 2 \times 0.000467 = 0.00093
```

`m g / N` is the weight one segment carries, 0.86 mN on v5, about the
strongest push a leg's sweep motor makes at its foot.

**What to expect.** Model B's random first curves and its exploration noise
make the legs touch in about 35% of segment-steps at the start (the GPU
check's first cycles; 3% with the mean action alone), so this term should
shrink early as the curves sort themselves out.
</details>

### 3. Rhythm terms

#### Tempo ahead

**Idea.** Each follower keeps the tempo of the segment ahead, and so, down
the chain, the head's: the one rule about rhythm, and mandatory (the user's
decision, 2026-10-12). It judges only the tempo, never the phase or the step,
so the wave and the step's shape stay free to emerge. The head, with nothing
ahead, sets the tempo.

<details>
<summary>Math, and what to expect</summary>

```math
D_i \;=\; \min\!\Big(1,\ \big\lvert \log_2 f_i - \log_2 f_{i-1} \big\rvert\Big), \qquad D_0 = 0, \qquad w = 3.5 \times 0.000467 = 0.00164
```

`f` is a segment's tempo, its heartbeat: one octave apart (one tempo twice
the other) costs the whole weight. The absolute form keeps small gaps costly
(the squared form left them nearly free, run 24).

**What to expect.** The largest cost at the start: every segment starts at
its own random tempo between 1 and 4 Hz, so a follower pays about −0.0010
per step (measured, two thirds of an octave on average) until it matches. It
should fall within the first cycles; the [heart](#the-heart) lets a tempo
change by at most an octave per second.
</details>

---

## Part 2. The tools

Everything around the reward: how the legs are driven, what each segment
sees, how the agents explore, and how training is scheduled.
✅ is in cycle 2's pilot; ⬜ is not in use now. The runs are those of the
study notes (`results/oscillators/README.md` in CentipedeTraining).

| | Tool | Kind | Setting | Runs |
| :---: | --- | --- | --- | --- |
| ✅ | [Model A, the network draws](#model-a-the-network-draws) | Leg control | `leg_drive = "swing"` | pilot |
| ✅ | [Model B, the clock is the engine](#model-b-the-clock-is-the-engine) | Leg control | `leg_drive = "waves"`, `waves = 2` | 31– |
| ✅ | [The heart](#the-heart) | Leg control | `tempo_rate_octaves_per_s = 1`, `start_tempo_range_hz = [1, 4]`, `max_tempo_hz = 12.5` | 31– (ceiling 35–) |
| ✅ | [Targets along the curve](#targets-along-the-curve) | Leg control | always with models A and B | 31– |
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
| ✅ | [Coloured exploration noise](#coloured-exploration-noise) | Exploration | `exploration_noise_beta = 2` | 2: 31–; 1: 02, 04, 08, 09, 13; 0.5: 10–12, 14–28 |
| ✅ | [CAPS, smooth mean actions](#caps-smooth-mean-actions) | Learning | `temporal_smoothness_coefficient` | 0.1: 14; 0, 0.3, 1, 3: 31–34 |
| ✅ | [A random first guess](#a-random-first-guess) | Learning | `initial_action_offset_spread = 0.2` | 31– |
| ✅ | [Curriculum](#curriculum) | Training | `[curriculum]`; `level_bonus = 0.05` from the next round | all |
| ✅ | [Plateau stop](#plateau-stop) | Training | `plateau_cycles = 48` | all |
| ✅ | [Range circle](#range-circle) | Task | `range_circle_ratio = 2.5` | all |
| ✅ | [Walking on after an arrival](#walking-on-after-an-arrival) | Task | `after_arrival = "new_target"` | all |

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
sets the curve's 5 amounts per joint every 20 ms; the clock plays it. To
walk, the network must draw a curve and keep it long enough to be played
whole, which [CAPS](#caps-smooth-mean-actions) teaches. 41 actions per
follower (the heart and 8 joints × 5), 42 for the head.

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
most doubling or halving it per second, so the tempo cannot jump. Each
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

#### Targets along the curve

✅ **Idea.** Within each 20 ms step a leg's target moves along its curve
instead of holding the step's first value: every physics step (134 per
action) sets the target a share of the way from the curve at the hand's
start to the curve at its end. The motors then follow a smooth line, not a
staircase of 50 jumps a second.

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

### Exploration and learning

#### Coloured exploration noise

✅ **Idea.** The exploration noise drifts over time instead of changing
completely at every step, so an exploring leg tries a slightly different
movement for a while rather than jittering. Cycle 2 uses β = 2, red noise:
per step it changes less than the step's own motion, so model B's curves are
explored as curves (the twitching of runs 08 to 10 came from noise that
changed faster than the step). β = 1, pink noise (Eberhard et al., ICLR
2023), is the pilot's alternative.

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

#### Walking on after an arrival

✅ **Idea.** After an arrival the body keeps walking toward a new target,
with its clocks running, as a real centipede would, and the targets reached
per minute count every arrival.
