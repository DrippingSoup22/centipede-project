# Reward

From run 09 (the user's design, 2026-10-21, after run 08): one account,
the team's: the line it walks, of which the body's energy takes a share
that the task's priority sets. Run 08 capped the line at 100 mm/s, let the
joints' accelerations take the share and charged each unit its carry; runs
06 and 07 subtracted the accelerations as a price, run 05 charged the
joints' speeds, and runs 01 to 04 used the cooperators' run 28 reward
(`docs/cooperators/reward.md` on that branch). Every run's lessons are in
the study's `Training/results/follow-line/README.md`.

## One account

Each 20 ms step every agent receives the same:

```text
team = forward line × h / (h + E) − steps back
```

Everything is counted in millimetres of line, and each agent has one
critic ([critic.md](critic.md)).

## The line

For each unit, with Δs how far its centre went along the line on the step
(mm), d its centre's distance from the line (mm) and θ its body's angle from
the line's direction:

> forward line = Δs · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0,
> and a step back, Δs < 0, charged in full

The team receives the mean over the units.

- **The band b**, the sight radius, 16.5 mm on v5: a unit is paid for the
  line only while it can see it ([observations.md](observations.md)).
- **Facing** along the line, and **a step back charged in full**, so that a
  body walking sideways earns less and rocking never pays.
- **The mean**, because the chain moves together: sharing the gain is what
  first moved the walk into the policy (cooperators, run 23).
- **No cap**: the priority sets the pace (below).

## The energy E

For each unit, in mJ a second, averaged over the last second; the team's
E is the units' mean, one body spending one store:

- **Work**: its motors' positive work, torque times joint speed, which the
  simulation adds up over its substeps.
- **Braking**: their negative work, at a fifth of its size: muscle does
  negative work at about a fifth of the cost of positive work (efficiencies
  about 25% and −120%, Margaria 1968). It is 2 to 7% of the positive work.
- **Holding**: each motor's command squared, 2.7 mJ/s for a motor fully on,
  however long it is held: what an electric motor turns into heat, and the
  cost of force a muscle pays (Kram & Taylor 1990). At 2.7, holding costs as
  much as the work on the scripted walk at its cheapest pace; in human
  walking, supporting the weight is about 28% of the net cost and pushing
  forward about 45% (Grabowski, Farley & Kram 2005).

Why this energy:

- **Faster costs more, for every gait.** Work grows with how fast and how
  hard the legs push. Runs 01 to 04 charged the same parts, but holding
  weighed six times more (16.6 mJ/s a motor) and was 86% of the scripted
  walk's cost, and holding is blind to how often a leg strokes.
- **Posture costs.** Runs 07 and 08's legs were held stiff and moved by the
  sweep alone: the lift pressed down 72 to 99% of the stance, the knee at
  its outer stop half the time. The joints' stops carry none of that
  torque; the motors hold it all, so holding charges it: those gaits spend
  2.7 times the scripted walk's holding for each millimetre. Animals place
  their legs so that the ground's push passes near the joints and the
  muscles need little force (Biewener 1989).
- **Squared force favours sharing the load.** N feet holding a weight W
  cost W²/N: on 18 feet instead of 20, 11% more. Muscles share a load by a
  cost rising faster than force (Crowninshield & Brand 1981); the carry
  asked for this before.
- **One store.** A cost in a unit's own account would let a unit save while
  the others carry it; the team pays the units' mean.

What a millimetre costs, measured on the gaits we have (2026-10-21):

| Walk | mJ for each mm of line |
| --- | --- |
| Scripted 35° wave at 76 / **102** / 121 / 148 mm/s | 0.031 / **0.030** / 0.039 / 0.053 |
| Scripted 20° and 50° waves, at their cheapest (73, 114 mm/s) | 0.036 |
| Runs 07 and 08's learned gaits, noise-free | 0.064 to 0.082 |

The body travels most cheaply near 100 mm/s, with work alone or holding
alone: the weight of holding sets how strongly posture is charged, not the
pace.

## The priority h

A trip of length D at speed v takes D / v seconds and costs E · D / v of
energy. If each second of it is worth h, its whole cost is (E + h) · D / v,
and the best speed is the one that makes v · h / (h + E) largest: the share
above. So h is the value of a second, in mJ, the task's priority:

- **h near 0, no hurry**: the most distance for the energy. Animals with
  time to spare travel at the speed of least energy for their distance
  (horses, Hoyt & Taylor 1981; people walking, Ralston 1958).
- **A larger h, a hurry**: people commuting walk faster than that, time
  having a value (Summerside, Kram & Ahmed 2018), and movements quicken as
  more reward is at stake (Shadmehr, Huang & Ahmed 2016).
- **A large h, an escape**: speed, whatever it spends.

On the scripted 35° walk:

| h, mJ/s | Best pace | At 121 mm/s | At 148 mm/s | Run 08's gait at 201 mm/s |
| --- | --- | --- | --- | --- |
| **1** | **102** | **−16%** | **−32%** | **half the best** |
| 3 | 102 to 110 | −8% | −19% | two thirds |
| 8 | 121 | best | about the same | 92% |
| 12 | 148 | | | as much as the best |

**h is 1 mJ/s** (the user's choice, 2026-10-21): the body walks at its
cheapest pace, and faster walks lose the most. The exploring noise's own
cost, mostly holding, hardly depends on the speed, and a constant energy
moves the best pace as a larger h would: the training worlds already feel
more hurry than h says.

## A share, not a price

Subtracted, the energy would need a price of at most 0.08 mJ for each
millimetre of line to keep the scripted walk's best pace near 100 mm/s. At
that price every gait our agents trained with, exploring noise included,
earns less than lying still on the belly, which spends nothing; to make
them pay, the price must reach 0.12, and the best pace jumps to 148 mm/s.
A clumsy beginner would earn most by not moving and never practise: runs
06 and 07's collapse. As a share, any forward walk earns more than lying
still, and before there is a walk there is nothing to take, so the
untrained agents' exploring costs nothing. The trip is to be made; the
priority decides how.

The DeepMind Control Suite builds its rewards the same way: the humanoid's
task terms are multiplied by a control factor (`small_control`, Tassa et
al. 2018).

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On run 09's reward (2026-10-21) the good walk, 103 mm/s, spends 3.06 mJ/s
a unit and keeps 26 of its 100 mm/s of line. Every fault loses but one:
0.7 times the step rate by 4.6 mm/s (68 mm/s), 1.4 times by 4.6 (120
mm/s), twice by 8.3 (148 mm/s); the wave running backward by 1.5, legs in
random phase by 13; wooden legs, the lift and knee pressed down and the
sweep alone stroking, by 39 (the body slides backward); the belly dragged
by 26, the lifts pressing half as hard by 18; six legs held up by 20;
standing and limp by 26. **One leg held up pays more**: unit 5's by 1.4
mm/s, the rear unit's by 0.8 (the front unit's loses by 3.5). The body
walks as fast on 19 legs and spends 4% less: the load the other legs take
on costs less than the idle leg saves. Run 08's carry made it lose; run
09 goes without it, and its feet are watched (the user, 2026-10-21: "we
will see"). Fresh networks under the tow walk 10 mm/s forward and 2
back, spending 4.3 mJ/s a unit: the share leaves about 2.3 of the forward
line and the steps back take 2, so they keep 0.4 mm/s (run 08's: 3.7). A
limp body under the tow does not move and earns nothing, so moving pays
from the start, if little.
