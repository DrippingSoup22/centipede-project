# Reward

The reward agreed for run 12 (the user's decisions, 2026-10-21, after
reviewing runs 01 to 11 against the literature, [research.md](research.md)),
with the carry measured per leg from run 14 (2026-10-22) and, from run 22,
the speed matched plainly and the line kept by each unit's own price
(2026-10-22).
Earlier rewards, and what each taught, are in the study's
`Training/results/follow-line/`.

## What it asks for

The body walks along the line at a commanded speed, every unit on the
line, spending as little energy as its units' neighbourhoods can, each unit
carrying its own weight.
How it walks is left to the agents: no term judges a step's shape (the
user's decision).

## The formula

Each 20 ms step, for unit i, in mm/s:

```text
team_i = v* − |v̄ − v*|  −  c · Ē_i
own_i  = − carry_i  −  p · L_i        (p · L_i from run 22)
```

| Symbol | What it is | Setting |
| --- | --- | --- |
| v̄ | The body's speed along the line: the units' mean forward line, each averaged over 1 s; from run 22 their mean plain speed along it ([keeping the line](#keeping-the-line-from-run-22)) | `match_plain_speed` |
| v* | The commanded speed, 67 mm/s; from stage 3 (run 23) the speeds' schedule's, the pay keeping 67 as its constant ([the commanded speed](#the-commanded-speed)) | `commanded_speed_mm_per_s`, `[environment.speeds]` |
| Ē_i | The mean energy spent on the step by the units within unit i's radius, itself and up to two each side (those present), mJ/s | `neighbourhood_energy` |
| c | The energy's price, in mm/s per mJ/s: 0.4 in stage 1 | `mm_per_mj` |
| L_i | From run 22: the mean, over unit i and the units behind it within its radius, of 1 − exp(−d / σ), d each one's distance from the line | `line_distance_scale_mm` (σ) |
| p | The line's price, in mm/s | `line_price_mm_per_s` |
| carry_i | Runs 12 and 13: the weight its own feet do not carry, in unit weights, averaged over 1 s, at 50 mm/s each. From run 14: each foot's load, averaged over 1 s, beyond a quarter either way of half the unit's weight, at the same price | `carry_price`, `carry_per_leg`, `carry_band` |

The speed is averaged before it is compared with v*, because |v̄ − v*| is
not linear: a stride's surges and dips must cancel first. The energy is
charged as it is spent: its cost is linear, so averaging it would only
delay it, and over a walk the total is the same.

Below the target the team is paid its speed, v̄, exactly as the line paid
before; above it each extra mm/s costs one. Every ending is a truncation
([task.md](task.md)), so a constant added to every step changes nothing
learned: written as − |v̄ − v*| − c Ē_i, it teaches the same.

Each account has its own critic ([critic.md](critic.md)).

## The commanded speed

The task sets how fast to go, and the body finds how to walk that fast.

- **Why a commanded speed.** Every study found with natural gaits commands
  a speed and leaves the gait to an energy (Fu et al. 2021, legged_gym).
  In this study, paying distance ran away (runs 05, 09 to 11), a cap with
  nothing lost above it was overrun two to three times (runs 03, 08), and
  the best waves (0.35 to 0.37) came with jerky motion priced (runs 07,
  08), one of them with a capped line (run 08). At a
  set speed, the only way to earn more is to spend less, so a fixed cost
  per stroke can no longer be spread over ever more distance, as runs 09 to
  11 did.
- **The priority made explicit.** Animals with time to spare travel near
  their cheapest pace (Hoyt & Taylor 1981), faster when time matters more;
  the command is that choice, made by the task.
- **The forward line.** For each unit, with Δs how far its centre went
  along the line on the step, d its distance from the line and θ its angle
  from the line's direction: Δs · max(0, 1 − (d / b)²) · max(0, cos θ) when
  Δs ≥ 0, and Δs in full when negative. The band b is the sight radius,
  16.5 mm: a unit's progress counts only while it sees the line and faces
  along it, and a step back counts against.
- **Averaged over 1 s**, about three strides, because a gait's speed swings
  within each stride: a per-step target would ask for an even speed within
  the stride, a constraint on form.
- **Symmetric**: as much for each mm/s too slow as too fast, as in the
  literature.
- **Slower in bends** (stage 2, `slow_in_bends`): on a bend the outer feet
  run on a wider circle than the centre, so the command falls just enough
  for them to keep v*: v* / (1 + r k), r the feet's reach to the side of
  the centre (17.7 mm on v5 and v6, measured from the model) and k the
  line's curvature, the units' mean, averaged over 1 s as v̄ is. The team
  is paid v* − |v̄ − command|: a bend lowers no pay, only the speed it
  pays for. Walkers slow on curved paths (Hicheur et al. 2005). A free range
  of speeds was set aside: with energy priced, the slowest speed in it would
  be the cheapest, on straights too.

| Radius | Command |
| --- | --- |
| Straight | 67 mm/s |
| 3 body lengths | 61.5 mm/s |
| 1 body length | 53 mm/s |

- **A changing command** (stage 3, from run 23, [task.md](task.md#the-speeds-stage-3)).
  The command is the speeds' schedule's speed, lowered in bends as above,
  and the team is paid 67 − |v̄ − command|: the constant stays, so a stop
  pays as much as walking at the command, and only the gap is charged.
  Both are averaged over 1 s, so the lag after a change counts the same on
  both sides; a restart is a change from rest, the command averaged from 0
  as the body's speed is. At c = 2 the pay keeps the body at the command
  up to 100 mm/s and beyond: each mm/s below it costs 1, while the energy a
  scripted wave saves by going 1 mm/s slower is worth 0.1 (near 90 mm/s) to
  0.25 (near 125) (the study's archive, `ledger/speed_steps.txt`).

## Keeping the line (from run 22)

The learned walks kept 1 to 2 mm from the line at the front and drifted at
the rear (the user, 2026-10-22). Each unit's mean distance in the noise-free
evaluations at cycle 512, front to rear, mm:

| Run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16, seed 1 | 1.6 | 1.6 | 1.9 | 1.8 | 1.5 | 1.3 | 1.4 | 1.4 | 1.1 | 1.7 |
| 16, seed 2 | 1.2 | 1.5 | 1.5 | 1.2 | 0.9 | 1.7 | 3.0 | 4.9 | 7.5 | 10.0 |
| 17, seed 1 | 1.0 | 0.8 | 0.9 | 1.1 | 1.3 | 1.7 | 2.1 | 2.7 | 3.4 | 4.1 |
| 17, seed 2 | 1.1 | 1.0 | 1.2 | 1.2 | 0.9 | 0.7 | 0.5 | 0.9 | 1.8 | 2.8 |

The forward line (runs 12 to 21) explains it three ways:

- **The line counted only through the speed.** Being off the line was the
  same as walking slower, and above the command it cancelled the overspeed:
  run 16 seed 2 walked 76 mm/s, its rear 3 to 10 mm off and turned 16 to
  26°, which the forward line counted as about 69 mm/s, nearly the command.
- **Flat near the line.** 1 − (d / 16.5 mm)² takes 1.5% of a unit's step at
  2 mm and 3.3% at 3 mm; the team's mean, a tenth of that.
- **Shared.** A unit's distance was a tenth of the reward all ten agents
  share, the hardest kind of reward for each agent to learn from (Agogino &
  Tumer 2005); and the rear unit has no spine motor: the joint the unit
  ahead commands places it.

So, from run 22 (the user's agreement, 2026-10-22):

- **The speed, plainly** (`match_plain_speed`): v̄ is the units' mean plain
  speed along the line, a step counted wherever the unit stands. Each unit
  also observes this speed in place of its forward line.
- **The line, in each unit's own account**: p · L_i, L_i the mean over unit
  i and the units behind it within its radius (itself and two, those
  present) of 1 − exp(−d / σ). The spine joint behind a unit places the
  units behind it, and the unit sees them. The exponential of the distance
  pulls hardest at the line and still pays a small improvement far from
  it: a vessel learning to follow a path with PPO was paid e^(−γ|e|),
  chosen over a Gaussian for its fatter tails (Meyer et al. 2020).

| What each shape takes from a unit's step at a distance | 1 mm | 2 mm | 3 mm | 5 mm | 10 mm |
| --- | --- | --- | --- | --- | --- |
| The forward line, 1 − (d / 16.5 mm)² (runs 12 to 21) | 0.4% | 1.5% | 3.3% | 9% | 37% |
| 1 − exp(−d / 4 mm) | 22% | 39% | 53% | 71% | 92% |

p and σ are set by the ledger on the built code (below).

## The energy

Each motor stands for the muscles across its joint. A muscle contracting
over and over pays three costs (van der Zee & Kuo 2021): work, switching its
force on and off, and holding it. For each unit, in mJ a second:

| Cost | Charged | Price |
| --- | --- | --- |
| Work | The motors' positive work, torque times joint speed, summed over the physics steps; negative work at a fifth (muscle efficiencies about 25% and −120%, Margaria 1968) | 1 |
| Switching | Each motor's force f, a share of its strength, its rate of change squared | 0.00352 mJ/s for a rate of 1 a second |
| Holding | Each motor's force, \|f\| | 0.418 mJ/s for a motor fully on |

f is the command on torque motors and the spring's and damper's pull on
target motors ([actions.md](actions.md)). The two prices split the scripted
wave's cost, at 3.5 strokes a second, as van der Zee & Kuo measured: 42%
work, 41% switching, 17% holding; on the body runs use, the calibration is
repeated. Swinging a leg quickly costs far more than its work (Doke & Kuo
2007), and small animals pay mostly for producing force in quick steps
(Kram & Taylor 1990): switching makes quick, short strokes dear, 4 to 13
times the scripted wave's energy for each millimetre in runs 10 and 11
(work alone, run 09: 1.5 times). It also does the job of the smoothness
terms of other rewards, which charge the change of the action between
decisions (legged_gym's action rate).

**Its price c** is an exchange rate: how many mm/s off the target the body
would accept to save 1 mJ/s. A small c lets clumsy beginners, whose
exploring alone switches 14 to 22 mJ/s a unit, still gain by trying to walk;
a large c separates the gaits more strongly. Training goes in stages:

- **Stage 1**, from scratch: **c = 0.4**, the middle of the range in which
  exploring still beats standing still under the tow (measured below). A
  gait holding the speed while spending 30 mJ/s instead of the wave's 2
  already loses 11 mm/s.
- **Stage 2**, from stage 1's checkpoint: **c of at least 0.55**, once the
  agents walk, so that such waste costs as much as a leg held up (15 to
  25 mm/s). Run 15 set **c = 2** at once on run 14's final checkpoint:
  the energy for each mm fell from 0.062 to 0.026 and 0.029, below the
  scripted wave's 0.035, and neighbouring legs locked their steps.
- **From run 16, both stages in one run**: c stays at 0.4 until the
  noise-free walk holds the command (its line paid at least 0.55 of the
  90 mm/s cap, `ramp_after_line_share`), then rises to 2 over 128 cycles
  (`ramp_start` 0.2, `ramp_cycles`). On a clumsy gait, near 0.4 mJ for
  each mm, a price of 2 would make moving less pay nearly as much as
  walking faster, so the price waits for the walk.

An energy cost is needed from the start: run 03 learned its walk with none,
and adding it at full price later changed nothing.

**The neighbourhood's energy.** Each unit is charged the mean energy of the
units within its radius, in the team account:

- **Same account, same scale as the gain.** The gain, the speed, is
  shared; a private cost set against it taught units to do less (the
  cooperators: one unit's exploring gained the body 51 mm in 3 s and itself
  5, against 7.5 of its own costs), and energy in the own account, whose
  advantages are normalised apart, paid free-riding (the cooperators' cycle
  16: a unit 20% weaker gained 30 to 55 mm/s).
- **Its causes within the critic's sight.** A unit's actions change mostly
  its own and its neighbours' effort, through the chain. Critics that see a
  neighbourhood are accurate when rewards and interactions are local (Qu,
  Wierman & Li 2022), and agents with local critics are given their
  neighbours' rewards (MA2C, Chu et al. 2019). The body's mean (runs 09 to
  11) put each unit's own energy at a tenth of its reward, among nine
  others' noise, with half of it out of its critic's sight.
- **No free ride, and a reason to coordinate.** A unit that leans on its
  neighbours or fights them through the chain pays for their extra effort;
  legs in step or in random phase spend more than a wave (the ledger).
- **The units behind weighed less** (run 17, `energy_weight_behind`, the
  user's idea, 2026-10-22). Weighed alike, the neighbourhood pays two waves
  leaving a middle unit as much as one wave (the study's `ledger_15.txt`:
  +0.3 mm/s), and run 15's units locked into such patches. Weighing the
  units behind less, each unit answers more for the units ahead, the front
  unit mostly for itself: the units behind adapt to those ahead, and the
  front leads. The weight is the strongest tilt the ledger finds safe: the
  less a unit pays for the units behind, the cheaper it is for it to be
  pushed by them. Every weight from 1 to 0 moves the ledger's margins by
  2 mm/s at most (the study's `ledger_16_17_w*.txt`), so run 17 uses 0:
  each unit pays for itself and the units ahead. The scripted gaits cannot
  show the tilt's real effect, which is on who adapts to whom while
  learning.

## The carry

Each unit's own account is charged 50 mm a second for each unit weight its
own feet, on average over 1 s, do not carry, whatever else holds it up: its
neighbours through the joints, its belly, or a leg resting on another's.

- **A duty of its own.** With a team reward independent learners breed
  lazy agents (Sunehag et al. 2018); kept local, the remedy is an
  individual reward with its own critic (Sheikh & Bölöni 2020; Wang et al.
  2022). Locomotion studies need none: one network cannot free-ride on
  itself.
- **The evidence.** Without it legs went idle (runs 05 and 09) and the
  ledger paid a leg held up more than the wave (run 09); with it that leg
  lost by 13 to 20 mm/s, and its critic explained 0.90 to 1.00 of its
  returns. Units 4 to 6 still leaned in runs 10 and 11.
- **Its strength** is not its price: its account holds only the carry and
  is normalised on its own, so the weight given to the own account against
  the team's decides it ([critic.md](critic.md)).

### Per leg (from run 14)

Run 12 showed the unit carry's two gaps (the study's
`archive/checks/run_12/support_check_384.txt`):

- **It charged the units that could not act.** The middle units hung on
  their neighbours (seed 1's units 3 to 6 carried 0.21 to 0.73 of their
  weight) with their lifts pressed to the low stop 93 to 100% of the time;
  the units that held them up (up to 2.1 unit weights) had their lifts free
  and paid nothing, the carry charging only a shortfall.
- **One foot satisfied it.** Three legs a seed stayed up, their units
  carried by the other foot and held level by their neighbours.

So each foot is charged instead, in either direction, for its load
averaged over 1 s beyond a quarter of half its unit's weight (0.375 to
0.625 of the weight free). Over a unit, the forces balance: the weight goes
through its two feet, its belly or its joints to its neighbours, and a foot
carrying more than the other is held level by the other foot or by the
trunk's roll. So feet each carrying half, on average, is the trunk carrying
neither weight nor twist: the user's rule, "the spine keeps a unit only for
a moment", read at the feet, where the signal is large (a unit on one foot
rolls 0.6° on v6; its feet read 1 and 0 against 0.5). It says nothing of
the rhythm: a foot may step as often and as briefly as it likes, and a
stride's single-foot stances average out.

| Gait, the body's total charge in unit weights | Unit carry | Per leg |
| --- | --- | --- |
| The scripted wave, first v6 | 0.72 | 0.11 |
| The scripted wave, v6 | 0.63 | 0.03 |
| Run 12's seed 1, final | 1.87 | 4.03 |
| Run 12's seed 2, final | 1.62 | 3.75 |

Left open: a foot that stays down and slides (the team's speed charges it),
and turns, where the inner legs carry more (the user, 2026-10-22: to
revisit when the line bends).

## Left out

- Terms that judge a step's shape (the user's decision): how to walk is the
  agents' to find.
- A separate smoothness term: switching charges it.
- The value of time and the energy's share (runs 09 to 11): the commanded
  speed replaces them.

## Numbers fixed by measurement

Computed from recorded walks before the build (2026-10-21, the study's
`archive/checks/reward_12/`: scripted walks, the learned gaits of runs 10
and 11, fresh networks; 4 worlds, 6 s each, on v5 and v6):

| Number | Found | Value |
| --- | --- | --- |
| v* | The scripted wave spends nearly the same for each millimetre at every speed on v6 (0.028 mJ/mm from 48 to 111 mm/s), so no cheapest pace sets it | **67 mm/s**, 1 body length a second, the middle of *S. heros*' 0.5 to 1.5 |
| Ē at the ends | The mean of the units present and their sum over five differ by under 1 mm/s in every comparison | **The mean of the units present** |
| The carry | Without it a leg held up or a unit dragged pays as much as the wave on both bodies (−0.4 to +4.5 mm/s): the neighbourhood's energy alone does not stop it. With it they lose by 15 to 35 | **Kept** |
| c, stage 1 | The faults barely depend on c: the scripted gaits spend 1.5 to 2.8 mJ/s, close to the wave's 1.9 (v6) and 2.5 (v5). What bounds it is the start: fresh networks' exploring, at the starting noise 0.3 on v6, spends 11.9 mJ/s, 85% of it switching, and its team account beats a body standing under the tow by 8.5 mm/s at c = 0, 5.6 at 0.25, 2.7 at 0.5 and −3.1 at 1. With the carry added, standing wins at the start at every c, as it did in runs 06 to 11, whose walks formed anyway under the tow | **0.4**, with the noise 0.3 ([critic.md](critic.md)) |

With v* = 67 mm/s and the carry, every fault loses on v5 at every c from
0 to 4: legs in step by 55, in random phase by 46, the rear half limp by
62, a leg held up by 16 to 25, wooden legs, the belly and quick tapping (4
times the rate, a quarter of the sweep) by 16 to 35. Runs 10's and 11's
learned gaits lose by 53 to 198, for their speed above the target and their
energy. A wave at twice the rate with half the sweep, 67 mm/s at about 5
strokes a second, is the closest: it loses by 2 to 6 on v5 and gains 2.4 on
v6, where it spends the same energy as the wave and its feet carry more. It
is a wave too: the reward leaves the stepping rate free, and rejects
uncoordinated legs, quick tapping that cannot hold the speed, legs that do
no work, and running past the target. The energy's price matters most
against a gait that holds the speed while wasting energy, as the learned
gaits did (27 to 35 mJ/s): stage 2's larger c.

On the built code (the study's `archive/ledger/ledger_12.txt`, run 12's
configuration, c = 0.4, own weight 1), a wave over 5 units at 69 mm/s
keeps the team 65 mm/s, and every fault loses: off the speed by 19 to 58,
a leg held up by 20 to 23, the narrowest the lifts pressing half as hard
(1.1) and one unit's feet skating (4.4). The wave running backward, a wave
too, loses by 2.0.

## Open

- **Turns** (stage 2): a band of speeds around v* that all count as on
  target, so that slowing in a bend costs nothing (the user's idea).
