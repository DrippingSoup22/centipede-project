# Reward

The reward agreed for run 12 (the user's decisions, 2026-10-21, after
reviewing runs 01 to 11 against the literature, [research.md](research.md)).
Earlier rewards, and what each taught, are in the study's
`Training/results/follow-line/`.

## What it asks for

The body walks along the line at a commanded speed, spending as little
energy as its units' neighbourhoods can, each unit carrying its own weight.
How it walks is left to the agents: no term judges a step's shape (the
user's decision).

## The formula

Each 20 ms step, for unit i, in mm/s:

```text
team_i = v* − |v̄ − v*|  −  c · Ē_i
own_i  = − carry_i
```

| Symbol | What it is |
| --- | --- |
| v̄ | The body's speed along the line: the units' mean forward line, each averaged over 1 s |
| v* | The commanded speed |
| Ē_i | The mean energy of the units within unit i's radius, itself and up to two each side, each averaged over 1 s, mJ/s |
| c | The energy's price, in mm/s per mJ/s |
| carry_i | The weight its own feet do not carry, in unit weights, averaged over 1 s, at 50 mm/s each |

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

## The energy

Each motor stands for the muscles across its joint. A muscle contracting
over and over pays three costs (van der Zee & Kuo 2021): work, switching its
force on and off, and holding it. For each unit, in mJ a second, averaged
over the last second:

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

- **Stage 1**, from scratch: c at the middle of the range in which the
  scripted wave beats every fault and exploring beats standing still
  (measured below).
- **Stage 2**, from stage 1's checkpoint: a larger c, once the agents walk.
  Whether it jumps or rises gradually is decided then.

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
| c, stage 1 | The faults barely depend on c: the scripted gaits spend 1.5 to 2.8 mJ/s, close to the wave's 1.9 (v6) and 2.5 (v5). What bounds it is the start: fresh networks' exploring spends about 15 mJ/s, mostly switching, and must still beat a body standing under the tow, which holds only for c up to about 0.07 on v6 and 0.2 on v5 with the carry counted | **Set with the exploration** ([critic.md](critic.md)): a smaller noise spends less and widens the range |

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

## Open

- **Turns** (stage 2): a band of speeds around v* that all count as on
  target, so that slowing in a bend costs nothing (the user's idea).
