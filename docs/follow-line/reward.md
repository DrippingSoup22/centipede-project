# Reward

From run 06 (the user's design, 2026-10-20, after run 05): the team is paid
the line it walks, less what its joints' accelerations cost; each unit's
own account is charged the weight its feet do not carry. From run 07 the
accelerations' weight adapts to the walk. Run 05 charged the
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
  smooth strokes pay. This full price is the ceiling; how much of it
  applies is the adaptive weight's choice (below).

## The price adapts to the walk

From run 07 the accelerations' weight follows what the agents' walk does,
not the clock (the user's decision, 2026-10-20). After each of the tow's
windows ([task.md](task.md#the-tow)) it reads the worlds that act with the
agents' mean action and are never towed, the walk the policy really learned:

- it rises 1.1 times while they walk faster than 110 mm/s with their feet
  stepping, the price being then needed;
- it falls 1.1 times while they walk slower than 70 mm/s, so that the cost
  can never squeeze the body into a creep;
- it holds in between, and the agents refine their gait at a steady price.

Each step goes to one more power of 1.1 for each 40 mm/s beyond the band,
at most four (1.46 times): a proportional term, so that the price catches
up with a walk far outside the band without overshooting near it (Stooke
et al. 2020's PID Lagrangian; the price works as a Lagrange multiplier on
the speed). It starts at 0.002, where untrained agents' exploring breaks
even: their jerky movements cost 400 times what they earn at full price.
It stays within 0.002 and 1. The band is the pace a real centipede walks
at; the agents choose how they make it, and wide, smooth strokes are what
keep the price low. A price that only rises when the walk is good enough
and falls when it breaks is the common answer of locomotion work to costs
that stifle exploring: legged_gym widens its tasks only once tracking
passes 80% (Rudin et al. 2021), and Kim et al. (2024) tighten each
constraint from the policy's current level.

Run 06 raised the weight on a clock, exponentially from 0.002 to full at
cycle 225: the body walked best near 0.06 (96 mm/s, a wave of 0.27), then
crept as the weight rose past it, its jerky gait and its exploring noise
both charged far more than the line paid.

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
