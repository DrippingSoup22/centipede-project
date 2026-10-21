# Reward

From run 10 (the user's decision, 2026-10-21, after run 09): the team is
paid the line it walks, of which the body's energy, as a muscle pays it,
takes a share that the task's priority sets; each unit's own account is
charged the weight its feet do not carry. Run 09 had the same share with
the motors' work and squared commands as the energy, and no carry; run 08
capped the line at 100 mm/s and let the joints' accelerations take the
share; runs 06 and 07 subtracted the accelerations as a price, run 05
charged the joints' speeds, and runs 01 to 04 used the cooperators' run
28 reward (`docs/cooperators/reward.md` on that branch). Every run's
lessons are in the study's `Training/results/follow-line/README.md`.

## Two accounts

Each 20 ms step every agent receives:

```text
team = forward line × h / (h + E) − steps back   (the same for all ten)
own  = − carry                                     (each unit's own)
```

Each account has its own critic ([critic.md](critic.md)). Everything is
counted in millimetres of line.

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
- **No cap**: the priority and the energy set the pace (below).

## The energy E

Each joint's motor stands for the muscles across that joint: the agent
commands its force, u, a share of its strongest torque, and the angle
follows from the physics. A muscle contracting over and over pays three
costs (van der Zee & Kuo 2021, who measured them in cyclic contractions):

| Cost | What it pays for | As the strokes quicken, at the same force | Its share in their measurement |
| --- | --- | --- | --- |
| **Work** | Shortening against a load | Grows in proportion | 42% |
| **Switching** | Turning the force on and off: pumping calcium at every contraction | Grows as the square | 41% |
| **Holding** | Keeping the force on, in proportion to the force | Does not grow | 17% |

Earlier, swinging the leg fast was found to cost far more than its work,
the extra following the rate at which force is produced (Doke & Kuo 2007).
For each unit, in mJ a second, averaged over the last second, the team's
E being the units' mean (one body, one store):

- **Work**: its motors' positive work, torque times joint speed, which the
  simulation adds up over its substeps; their negative work at a fifth of
  its size (muscle efficiencies about 25% and −120%, Margaria 1968).
- **Switching**: each motor's command's rate of change, squared, at 0.00352
  mJ/s for a rate of 1 a second (a full stroke of the command in a second).
- **Holding**: each motor's command, |u|, at 0.418 mJ/s for a motor fully
  on.

The two prices split the scripted walk's cost, at *S. heros*' fastest
tempo of 3.5 strokes a second, as van der Zee & Kuo measured: 42% work,
41% switching, 17% holding. Run 09 had no switching cost and held as u²,
the heat of an electric motor, at 2.7 mJ/s: its legs learned to buzz at
10 to 12 strokes a second, whose cost for each millimetre fell as they sped
up, and they ran to 300 mm/s.

What a millimetre costs, measured on the gaits we have (2026-10-21,
`Training/archive/checks/muscle_10/`):

| Walk | mJ for each mm, a muscle's energy | Run 09's energy | The share's pay at h = 1, mm/s |
| --- | --- | --- | --- |
| Scripted wave, 3.5 strokes a second, 102 mm/s | **0.036** | 0.030 | **22** |
| The same, 4.9 strokes a second, 121 mm/s | 0.053 | 0.039 | 16 |
| The same, 7 strokes a second, 148 mm/s | 0.099 | 0.053 | 9.5 |
| Run 09's buzz, noise-free, 245 to 309 mm/s | 0.13 to 0.15 | 0.043 to 0.049 | 7 (run 09's energy: 19 to 22) |
| Run 08's gait, noise-free, 189 to 201 mm/s | 0.27 to 0.29 | 0.074 | 3.5 |

The body travels most cheaply at 76 to 102 mm/s; the cost climbs steeply
above, and the buzz earns a third of the scripted walk, where run 09's
energy nearly tied them. The exploring noise switches the commands too,
about 12 mJ/s a unit on run 09's gaits: it acts as a larger priority would
(the training worlds' best scripted pace becomes about 110 to 120 mm/s)
until the noise narrows.

- **Posture costs.** Runs 07 and 08's legs were held stiff and moved by the
  sweep alone, the lift pressed down 72 to 99% of the stance; the joints'
  stops carry none of that torque, the motors hold it all, and holding
  charges it: those gaits held twice to three times the scripted walk's
  force. Animals place their legs so that the ground's push passes near
  the joints and the muscles need little force (Biewener 1989).
- **One store.** A cost in a unit's own account would let a unit save while
  the others carry it; the team pays the units' mean.

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

**h is 1 mJ/s** (the user's choice, 2026-10-21). The scripted walk's E is
3.6 mJ/s, so the share is close to the distance for the energy: the body
walks at its cheapest pace, and faster walks lose (above).

## A share, not a price

Subtracted, the energy would need a price low enough to keep the scripted
walk's best pace near 100 mm/s, and at that price every gait our agents
trained with, exploring noise included, earns less than lying still on
the belly, which spends nothing (run 09's energy: at most 0.08 mJ for each
millimetre, against their 0.083 to 0.13). A clumsy beginner would earn most
by not moving and never practise: runs 06 and 07's collapse. As a share,
any forward walk earns more than lying still, and before there is a walk
there is nothing to take, so the untrained agents' exploring costs nothing.
The trip is to be made; the priority decides how.

The DeepMind Control Suite builds its rewards the same way: the humanoid's
task terms are multiplied by a control factor (`small_control`, Tassa et
al. 2018).

## The carry

Each unit's own account is charged, at 50 mm a second for each unit weight,
what its two feet, on average over 1 s, do not carry of its weight beyond
its free support (on the headless tester, all of its weight). Without it
legs go idle: run 05's units held their legs crossed over their
neighbours' and lifted, and run 09's held a leg up or leaned on their
neighbours, carrying under half their weight, while the energy alone made
a leg held up pay a little more than the good walk (the ledger, run 09). A
unit must carry itself on its own feet whatever else holds it up: its
neighbours through the joints, its belly, or a leg resting on another. It
is its own duty, in its own account ("shared gains, own duties",
[README.md](README.md)): the charge falls on the unit that leans on the
others, not on the team. Feet that carry while skidding along pass: a unit
carried along on skidding feet is a known gap, left open until a run shows
it.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On run 10's reward (2026-10-21) the good walk, 103 mm/s, spends 3.64 mJ/s
a unit (work 1.53, switching 1.49, holding 0.62), keeps 23 of its 100 mm/s
of line, and its units pay 0 to 15 mm/s of carry. Every fault loses, at own
weight 0.8 and 4: 0.7, 1.4 and 2 times its step rate by 3.4 and 5.8, 6.4 and
6.9, 12.3 and 10.9 mm/s (68, 120, 148 mm/s); the wave running backward by
1.1 and 0.5, the narrowest; legs in random phase by 16 and 31; a leg held
up by 13 to 20 and 68 to 92 (the team alone gains 0.8 to 1.2, the carry
takes 17 to 23); wooden legs by 41 and 61; the belly dragged by 59 and 203;
the lifts pressing half as hard by 27 and 74; standing by 21 and 14. Fresh
networks under the tow spend 15 mJ/s a unit, 11.7 of it switching, the
exploring noise's: the share leaves 6% of their forward line, less than
their steps back take, so the team gets −0.4 mm/s (run 09: +0.4), and
their carry costs 12.9.
