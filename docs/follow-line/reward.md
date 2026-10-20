# Reward

The minimal reward (the user's design, 2026-10-20, after runs 01 to 04): the
team is paid the line it walks, less what moving its joints costs. Nothing
else is priced. Runs 01 to 04 used the cooperators' run 28 reward
(`docs/cooperators/reward.md` on that branch), with a squeeze, energy in
millijoules, and each unit's own force share, hanging and pain; their lessons
are in the study's `Training/results/follow-line/README.md`.

## One account

Each 20 ms step every agent receives:

```text
team = line − motion        (the same for all ten)
```

Every part counts in this one account, so each agent has one critic
([critic.md](critic.md)); runs 01 to 04 also had each unit's own account.
Everything is counted in millimetres of line.

## The line

For each unit, with Δs how far its centre went along the line on the step
(mm), d its centre's distance from the line (mm) and θ its body's angle from
the line's direction:

> line = Δs · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0, and Δs for
> a step back

The team receives the mean over the units. There is no cap: walking always
pays, and the motion cost is what makes walking faster stop paying.

- **The band b**, the sight radius, 16.5 mm on v5: a unit is paid for the
  line only while it can see it ([observations.md](observations.md)).
- **Facing** along the line, and **a step back charged in full**, so that a
  body walking sideways earns less and rocking never pays.
- **The mean**, because the chain moves together: sharing the gain is what
  first moved the walk into the policy (cooperators, run 23).

## The motion

Each unit's joints, its legs' and the spine joint behind it, turning: their
squared speeds added up, at 0.19 mm a second for each (rad/s)²; the team is
charged the units' mean.

- **Zero at rest, steep with speed.** A joint's squared speed is in
  proportion to its kinetic energy, so the cost grows with how fast and how
  wide the legs stroke. Walking faster earns the line in proportion but costs
  more than in proportion, so there is a speed beyond which faster pays less.
- **Priced so that the net peaks at a real centipede's pace.** On the
  scripted walk (`Training/analysis/energy_speed.py`), one more mm/s of speed
  costs one more mm/s of motion at 90 mm/s, 0.6 of the full body's length a
  second (*S. heros*: 0.5 to 1.5 body lengths a second, Anderson et al.
  1995). There the net is +14 mm/s a unit: line 85, motion 71; at 41, 68,
  103, 111 and 148 mm/s it is +2, +11, +12, −6 and −54.
- **Why not energy in millijoules.** Runs 01 to 04 priced the motors'
  pushing and braking work and each command squared for as long as it was
  held. Measured on the scripted walk, that energy grows with speed but far
  too slowly: near 90 mm/s one more mm/s cost 0.1 to 0.2 mm/s, so the
  fastest walk always paid most, and runs 03 and 04 learned a run at 7 to
  11 strokes a second. Raising its price does not help: most of it is a
  fixed part, the force of holding commands, so walks at every speed would
  cost more than they earn and standing would win (run 01's timid legs).

## The ramp

The motion is weighed 0.05 of its price at the first update cycle, rising
in a straight line to full at cycle 161 (Hwangbo et al. 2019's curriculum on
the costs of moving, which "results in a standing behavior" when high from
the start). Untrained agents' exploring noise moves the joints more than the
scripted walk does, so at full price exploring costs 17 times what it earns
(the fresh check: motion −147 mm/s a unit, line +8.5 under the tow); at
0.05 the two about balance (+1.2). Lowering the noise instead does not: at
a spread of 0.1 exploring still loses (−16 mm/s) and earns little.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
With the motion in full (2026-10-20), the good walk at the target pace, 3
steps a second and 90 mm/s, earns the team +14 mm/s a unit, standing 0.
Faster walks lose (1.4 times the step rate by 17.6, twice it by 50.6), and
so do slower ones (0.7 times by 4.5), the backward wave (5.9) and every
fault of the body as a whole. Two faults of one unit pay a little more than
the good walk: unit 0 dragged with its legs limp (+2.4) and unit 5 holding a
leg up (+2.8). A unit that stops moving saves its share of the motion while
the others still carry it; the own terms that closed this gap (force share,
hanging) are left out of this test on purpose.
