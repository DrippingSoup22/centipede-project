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
| F_i | The marks: the mean, over its two feet, of 1 − e^(−d/σ_f) for each foot on the ground (touching it, as a print is read), d the foot's distance across the ground to the nearer of its marks on its side (0 for a foot in the air) | New |
| p_f, σ_f | The marks' price and scale: 30 mm/s and 4 mm | Set by the ledger (below) |

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

Before the first run, on the built code and scripted walks: the scripted
wave whose lag matches the unit spacing over its stride lands every foot on
the print ahead and must beat every fault. Run on 2026-11-03 with the
study's `Training/analysis/footsteps_ledger.py` (follow-line's scripted
centipede, re-aimed at the marks; its output in
`results/follow-footsteps/ledger_stage_a.txt`): 8 worlds, each case 6 s
after a 3 s warm-up, the configured own weight (1) and 4, the ramped costs
in full and at their start. Unit 1's footholds were laid a stride ahead of
its own landings, since the scripted body cannot aim; training's script
lays them along the line.

**The good walk**: 3.5 steps a second, each unit 0.316 of a cycle behind
the unit ahead (the unit spacing, 6.6 mm, over the stride), 69 mm/s, a
stride of 20 mm. Every unit's touchdowns land 1.2 to 1.8 mm from the
nearer mark (the median; at most 2.5 mm at the 90th percentile).

**The numbers it set:**

| Number | Value | Why |
| --- | --- | --- |
| σ_f | 4 mm | A good landing (1.4 mm) pays 0.30 of the price, a miss by the unit spacing 0.81, by half a stride 0.92 |
| p_f | 30 mm/s | At 10, follow-line's 9-unit wave at the good walk's speed, 30% cheaper in energy, lost by only 1.5 mm/s; at 30 by 6.1, while the good walk pays 3.7 mm/s for its marks |
| The script's stride | 20 mm | The good walk's, 19.7 to 20.9 mm |
| A swing (`swing_steps`) | 4 steps | Feet flicker off the ground for 1 to 3 steps; real swings last 6 steps or more, 8 the median |
| The restart distance | A foot's reach, 17.7 mm | A good walker's mean distance peaks at 12.4 mm, a foot mid-swing |

**At these, every fault pays less than the good walk**, on straight lines
and on curves at the tightest level (a body length's radius). The
closest, the culprits' change in mm/s at own weight 1, straight | curved:

| Case | Change |
| --- | --- |
| follow-line's wave, at the good walk's speed | −6.1 \| −6.7 |
| the backward wave, at the good walk's speed | −9.8 \| −9.1 |
| steps half as long, at the same speed (twice the rate) | −17.9 \| −17.6 |
| legs in random phase | −28.9 \| −22.0 |
| unit 5 dragged by its leader | −36.2 \| −36.2 |
| everyone standing | −76.8 \| −76.3 |

A foot in the air pays no marks, so a unit that lifts a leg or hangs limp
pays less for them than the good walk (unit 5 dragged: 1.3 mm/s against
3.4); its carry costs it far more (follow-line's carry, kept for this).

## Sources

- Meyer, Robinson, Rasheed & San (2020), Taming an autonomous surface
  vehicle for path following and collision avoidance using deep
  reinforcement learning, IEEE Access 8 ([arXiv 1912.08578](https://arxiv.org/abs/1912.08578)).
- The rest as in [follow-line's reward](../follow-line/reward.md).
