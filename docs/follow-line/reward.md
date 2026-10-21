# Reward

Runs 10 and 11 (the user's decisions, 2026-10-21). Earlier runs' rewards,
and what each taught, are in the study's `Training/results/follow-line/`.

## Two accounts

Each 20 ms step every agent receives, in millimetres of line:

```text
team = forward line × h / (h + E) − steps back   (the same for all ten)
own  = − carry                                     (each unit's own)
```

Each account has its own critic ([critic.md](critic.md)).

## The line

For each unit, with Δs how far its centre went along the line on the step
(mm), d its centre's distance from the line and θ its body's angle from the
line's direction:

> forward line = Δs · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0,
> and a step back, Δs < 0, charged in full

The team receives the mean over the units. The band b is the sight radius,
16.5 mm on v5: a unit is paid only while it can see the line. Facing along
the line and charging a step back in full make walking sideways earn less
and rocking never pay.

## The energy E

Each motor stands for the muscles across its joint. A muscle contracting
over and over pays three costs (van der Zee & Kuo 2021): work, switching
its force on and off, and holding it. For each unit, in mJ a second,
averaged over the last second, the team's E being the units' mean:

| Cost | Charged | Price |
| --- | --- | --- |
| Work | The motors' positive work, torque times joint speed, summed over the physics steps; negative work at a fifth (muscle efficiencies about 25% and −120%, Margaria 1968) | 1 |
| Switching | Each motor's force f, a share of its strength, its rate of change squared | 0.00352 mJ/s for a rate of 1 a second |
| Holding | Each motor's force, \|f\| | 0.418 mJ/s for a motor fully on |

f is the command on torque motors and the spring's and damper's pull on
target motors ([actions.md](actions.md)), a mean over the step. The two
prices split the scripted walk's cost on v5, at 3.5 strokes a second, as
van der Zee & Kuo measured: 42% work, 41% switching, 17% holding.

## The priority h

A trip of length D at speed v takes D / v seconds and costs E · D / v of
energy. If each second is worth h, the trip's whole cost is (E + h) · D / v,
and the best speed makes v · h / (h + E) largest: the share above. h is the
value of a second, the task's priority: near 0 the most distance for the
energy, as animals with time to spare travel (Hoyt & Taylor 1981); large,
speed whatever it costs. **h = 1 mJ/s** (the user's choice). As a share,
not a price, any forward walk earns more than lying still, and an untrained
body's exploring costs nothing before there is a walk (the DeepMind Control
Suite multiplies its humanoid's task by a control factor the same way,
Tassa et al. 2018).

## The carry

Each unit's own account is charged 50 mm a second for each unit weight its
two feet, on average over 1 s, do not carry. A unit must carry itself on
its own feet, whatever else holds it up: its neighbours through the joints,
its belly, or a leg resting on another. Without it legs went idle (runs 05
and 09).

## The ledger

Scripted walks on the built code check, before a run, that the good wave
walk pays more than every fault known (the study's
`Training/analysis/ledger.py`). On run 11's v6 the good walk, 104 mm/s,
spends 2.63 mJ/s a unit and keeps 30 mm/s; every fault loses at own weight
0.8 (quick steps by 4 to 9 mm/s, the backward wave by 6, a leg held up by 11
to 21, wooden legs by 44), and 0.7 times its step rate ties. The ledger
checks where learning should end, not whether it gets there: run 11 passed
it and its legs tapped 12 to 14 times a second.
