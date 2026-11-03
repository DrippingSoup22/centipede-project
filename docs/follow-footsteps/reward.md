# Reward

Agreed with the user on 2026-11-02: follow-line's proven walk, with the
line's cost replaced by the marks'. How a unit walks is left to it: no term
judges a step's shape.

## Every walker, units 1 to 10: one function

Each 20 ms step, for walker i, in mm/s:

```text
team_i = 67 − |v̄_i − v*_i|  −  c · Ē_i
own_i  = − carry_i  −  p_f · F_i
```

| Symbol | What it is | From |
| --- | --- | --- |
| v*_i | The command the walker holds, relayed from its leader ([task.md](task.md#prints-marks-and-the-speed)), averaged over 1 s | New: the relay |
| v̄_i | Its own speed along its heading, averaged over 1 s. The units are joined, so this is the body's speed, seen from each unit | follow-line's speed, measured along the unit instead of a line |
| Ē_i | The mean muscle energy spent on the step by itself and its leader, mJ/s; for unit 1, whose leader is the head, its own | follow-line's energy, its neighbourhood the leader |
| c | The energy's price: 0.4 at first, rising to 2 over 128 cycles once the walk holds its command | follow-line's ramp, its gate now on the speed |
| carry_i | Each foot's load averaged over 1 s beyond a quarter either way of half the unit's weight, at 50 mm/s for each unit weight | follow-line's carry per leg (from its run 14) |
| F_i | The marks: the mean, over its two feet, of 1 − e^(−d/σ_f) for each foot that carries weight, d the foot's distance across the ground to the nearer of its marks on its side (0 for a foot in the air) | New |
| p_f, σ_f | The marks' price and scale | Set by the ledger (below) |

The constant 67 keeps the pay where follow-line's stage 3 kept it: a stop
pays as much as walking at the command, and only the gap is charged.

**Why each term.**

- **The speed.** Every walker needs it. A body whose units were charged
  only costs, and paid nothing for moving, learned to stand (the
  cooperators study, `docs/cooperators/reward.md` on that branch, cycle 6);
  follow-line's walk formed with a commanded speed paid to every unit.
  Marks alone would pay standing still: every foot on its mark, nothing
  spent.
- **The energy**, with the leader's in each unit's account: a unit that
  lets its leader pull it pays for the leader's extra effort, so free
  riding costs (follow-line's [reward](../follow-line/reward.md#the-energy)).
  Units charged with their neighbours' energy locked their steps there.
- **The carry**: each unit carries its own weight (follow-line's
  [carry](../follow-line/reward.md#the-carry)).
- **The marks** are the Gartic phone: a foot is paid for standing where its
  leader's foot stood. It is charged over the whole stance, not only at
  touchdown: a foot that never lifts would otherwise never be judged, and a
  foot sliding with the body moves off its print and pays. The exponential
  pulls hardest near the mark and still pays a small improvement far from
  it, as follow-line's line cost did (Meyer et al. 2020).

**What each walker sees of it** (the pairing rule): its own speed, feet
loads and energy over 1 s, its leader's energy over 1 s, its command, its
marks and each foot's distance to them ([observations.md](observations.md)).

**Left out**: follow-line's line costs and its forward line; a cost for the
follower's misses charged to its leader (the user, 2026-11-02: the leader's
spine no longer places the follower, [actions.md](actions.md)).

## The head (stage B)

Agreed with the user on 2026-11-03: one account, its own.

```text
head = progress toward its target  +  arriving  −  p_h · failed footholds
```

- **Progress**: the body's speed toward the target, mm/s, and a bonus on
  arriving.
- **A failed foothold**: one unit 1's foot reached but could not stand on:
  it carried nothing, or slid off it (a hole, an edge). The head chose it,
  so the head pays. A foot that misses a good foothold is unit 1's miss and
  only unit 1 pays, through its marks. What counts as reaching, carrying
  and sliding, and p_h, are set before stage B.
- **Left out**: its collisions (a head driving into an obstacle stops
  making progress, which already costs), its energy (it commands no motor)
  and a carry rule (its neck is held straight, unit 1 carries it).

## The ledger

Before the first run, on the built code and scripted walks (follow-line's
`Training/analysis/ledger.py`, re-aimed at the marks): the scripted wave
whose lag matches the unit spacing over its stride lands every foot on the
print ahead and must beat standing still, a unit dragged by its leader, legs
in random phase, the wave run backward, steps of the wrong length and
stepping in place. The same measurements set p_f, σ_f, the script's stride
and the distance at which a world restarts.

## Sources

- Meyer, Robinson, Rasheed & San (2020), Taming an autonomous surface
  vehicle for path following and collision avoidance using deep
  reinforcement learning, IEEE Access 8 ([arXiv 1912.08578](https://arxiv.org/abs/1912.08578)).
- The rest as in [follow-line's reward](../follow-line/reward.md).
