# Reward

From run 06 (the user's design, 2026-10-20, after run 05): the team is paid
the line it walks, less what its joints' accelerations cost; each unit's
own account is charged the weight its feet do not carry. Run 05 charged the
joints' speeds instead and had no own account; runs 01 to 04 used the
cooperators' run 28 reward (`docs/cooperators/reward.md` on that branch).
Every run's lessons are in the study's `Training/results/follow-line/README.md`.

## Two accounts

Each 20 ms step every agent receives:

```text
team = line − accelerations        (the same for all ten)
own  = − carry                     (each unit's own)
```

Each account has its own critic ([critic.md](critic.md)). Everything is
counted in millimetres of line.

## The line

For each unit, with Δs how far its centre went along the line on the step
(mm), d its centre's distance from the line (mm) and θ its body's angle from
the line's direction:

> line = Δs · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0, and Δs for
> a step back

The team receives the mean over the units. There is no cap: walking always
pays, and the accelerations are what make walking faster stop paying.

- **The band b**, the sight radius, 16.5 mm on v5: a unit is paid for the
  line only while it can see it ([observations.md](observations.md)).
- **Facing** along the line, and **a step back charged in full**, so that a
  body walking sideways earns less and rocking never pays.
- **The mean**, because the chain moves together: sharing the gain is what
  first moved the walk into the policy (cooperators, run 23).

## The accelerations

Each unit's joints, its legs' and the spine joint behind it: how much each
changed its speed over the step, squared and added up, at 0.001 mm a second
for each (rad/s²)²; the team is charged the units' mean.

- **Wide, slow strokes cost less than short, fast ones.** The body's speed
  is about a stroke's width times its frequency. A joint's speed grows the
  same way, so a cost on the joints' speeds (run 05's motion) cannot tell
  the two apart: it costs the speed squared whichever way it is made. A
  joint's acceleration grows with the width but with the frequency squared,
  so for the same speed a stroke twice as wide and half as fast costs a
  quarter. On the scripted walk at 90 mm/s, an 80° sweep at 3 Hz costs the
  sweep joint 8 times less than a 63° sweep at 7 Hz
  (`Training/analysis/sweep_rate.py`). Jerky, erratic movements cost the
  most. The metabolic cost of swinging a leg rises about as its
  frequency's fourth power (Doke, Donelan & Kuo 2005), and legged_gym
  charges joint accelerations for smooth gaits (Rudin et al. 2021).
- **Faster costs more and more.** Once the stroke is as wide as the leg
  allows, only a higher frequency makes the body faster, and the cost grows
  as the speed to the fourth: each millimetre costs more the faster the
  body goes. A gait the agents make smoother goes only a cube root faster
  (twice as smooth, 1.26 times), and a longer stride goes faster, which is
  the way to speed the reward wants.
- **Priced on the best gait we know.** Run 05's agents found strokes far
  smoother for their speed than the scripted walk's; the price is set so
  that run 05's gait, every leg working, would peak near 100 mm/s
  (`Training/analysis/gait_costs.py`, measured on the noise-free policies
  of runs 01 to 05). There the walk nets about +73 mm/s a unit. The jerkier
  gaits of runs 01 to 04 would peak at 18 to 51 mm/s: under this price
  smooth strokes pay.

## The ramp

The accelerations are weighed 0.002 of their price at the first update
cycle, rising by the same factor each cycle, about doubling every 25, to
full at cycle 225 (the exponential form of Hwangbo et al. 2019's
curriculum on the costs of moving). Untrained agents' exploring is jerky:
it costs more than run 05's gait at 354 mm/s, and at full price 400 times
what it earns; at 0.002 the two about balance. While the weight is below
about 0.08 (the first 60% of the ramp), any walk as jerky as the scripted
one still pays; after that only smooth walking does.

## The carry

Each unit's own account is charged, at 50 mm a second for each unit weight,
what its two feet, on average over 1 s, do not carry of its weight beyond
its free support (on the headless tester, all of its weight). Run 05's
agents walked on a few units' legs: the others held theirs crossed over
their neighbours' and lifted, and saved their share of the cost while
others carried them. A unit must carry itself on its own feet whatever
else holds it up: its neighbours through the joints, its belly, or a leg
resting on another. It is its own duty, in its own account ("shared gains,
own duties", [README.md](README.md)): the charge falls on the unit that
leans on the others, not on the team. Feet that carry while skidding along pass:
a unit carried along on skidding feet is a known gap, left open until a
run shows it.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
The scripted walk is about ten times jerkier for its speed than run 05's
learned gait (its lift snaps at every step), so at the full price it costs
far more than it earns and cannot judge that price; the price rests on the
learned gaits instead. The ledger checks the reward's structure with the
accelerations priced at the scripted walk's level (2026-10-20): the good
walk earns +37 mm/s a unit, and every fault loses, among them a walk at
twice the step rate (by 6.8), six units each holding a leg up (by 70, by 96
at twice the rate) and skating feet (by 19 to 63). At the ramp's start,
when the accelerations are nearly free, faster walks still pay more, until
the weight reaches about 0.05.
