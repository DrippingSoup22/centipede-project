# The follow-footsteps design

This folder holds the design of the `follow-footsteps` workspace branch, one
file per part, as agreed with the user on 2026-11-02. The branch started
from `follow-line` (the user's decision), whose walk it keeps; that design is
in `docs/follow-line/`, and main's `docs/` describe the program on `main`.
Every run's results and lessons are in the study's
`Training/results/follow-footsteps/`.

## The project

A centipede gait can emerge from many agents, each seeing only its part of
the body. The constraints that define the project are follow-line's
([constraints.md](../follow-line/constraints.md)): one agent per unit, each
knowing only what lies within its radius, nothing shared between units.

## The idea: a chain, like a Gartic phone

In follow-line's 25 runs neighbouring legs locked their steps, but the waves
left wherever order first formed, often a middle unit: nothing in the task
said which way a wave should run ([Open questions](../follow-line/README.md#open-questions)).
Real centipedes take it from the front. The brain's command reaches the
first segment, and "each body segment basically follows the locomotor mode
of its anterior segment" (Yasui et al. 2019); each leg aims at the spot
where the leg ahead on its side touched down (Yasui et al. 2022). So here:

- **The head leads.** It chooses where unit 1 puts each of its feet next,
  and how fast the body goes. These are the body's only messages, values
  one agent chooses for another: the one exception to "no direct
  communication" (the user, 2026-11-02), as in the animal.
- **Every unit passes them on.** Each unit's leader is the unit ahead of
  it, and the head is unit 1's. A unit's feet are paid for stepping where
  its leader's feet stepped, and it holds the speed command its leader held
  one step earlier. No unit chooses what it passes on: a footprint is where
  its foot really landed, and the command is copied unchanged.
- **The constraints hold.** A unit receives only from its leader, one hop
  away and inside its radius. A command broadcast from the head to every
  unit would break the second constraint; the relay does not.

What this costs: stepping where the leg ahead stepped sets the wave's
direction. With the body moving forward, each unit steps a fixed share of a
stride after its leader, the unit spacing over the stride length, so the
wave runs from head to tail. The agents learn how to walk that way: placing
each foot on its mark with eight motors, the timing, carrying their weight,
holding the speed, saving energy. The head learns where to lead.

## Stages

| Stage | Body | The leader | Ground |
| --- | --- | --- | --- |
| **A**, the baseline | v6 headless | A script gives unit 1 its footholds and the speed ([task.md](task.md)) | Flat |
| **B** | v6 half, its head on a passive neck | The head, an agent of its own, in a growing share of worlds | Holes, bumps and obstacles in one height field, and targets only the head senses |
| Later | v6 full | | Basins |

Because the marks set the wave's direction, the ground need not be final
from the first cycle, as it had to be in follow-line: stage A stays flat.

## The parts

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | Prints, marks and the relayed speed; stage A's script; episodes |
| Reward | [reward.md](reward.md) | One function for every walker, the marks' cost its only new term; the head's terms for stage B |
| Observations | [observations.md](observations.md) | 65 values: the unit itself, its leader's feet, its marks and its command |
| Actions | [actions.md](actions.md) | Eight leg motors and the spine joint ahead |
| Critic and learning | [critic.md](critic.md) | follow-line's, each critic seeing what its actor sees |

## The bodies

Model v6 (`docs/model.md`, "Model v6"): stage A trains on its headless
tester, stage B on its half, whose head is its own body on a passive neck.
The head's antennae will need two root joints each, sideways 0 to 90° from
the head's axis and up and down −15° to +40° (rough angles from photos of
*S. heros* and *S. subspinipes*), in a later model. Measured on v6's half
standing, an antenna's root is 8.9 mm above the floor and the antenna 30.4
mm long: across that range its tip reaches the floor with its root about 7°
below the head's plane, and rises to about 22 mm at the top, so the head
can feel the ground and some height without moving its neck.

## Kept from follow-line

The muscle energy and its price ramp, the carry per leg, the tow, the
speeds' schedule, two critics added 1 : 1, gSDE with its floor, the
observation and value normalisation, and the PPO settings.

## Left out

The line and its curves (stage A's script lays its own path), the line's
costs, the front unit's view of the ground (the head's antennae take its
place in stage B), and the neighbours' last actions, values another agent
chose.

## Open

- **The neck** is passive for now (the user, 2026-11-02): no measured range
  or strength of a centipede's neck was found. An active neck, or motors
  that lift the first few units as real centipedes lift their front, may
  come later.
- **The head's inputs and reward** are sketched for stage B and settled
  before it.
- **The numbers the ledger sets** before the first run: the marks' price
  and scale, the script's stride, the restart distance ([reward.md](reward.md#the-ledger)).

## Sources

- Yasui et al., Scientific Reports 2019: the brain's command and
  each segment following its anterior segment ([PMC6889372](https://pmc.ncbi.nlm.nih.gov/articles/PMC6889372)).
- Yasui, Takano, Kano & Ishiguro, Frontiers in Robotics and AI 2022: each
  leg following the contact point of the leg ahead
  ([doi 10.3389/frobt.2022.797566](https://www.frontiersin.org/articles/10.3389/frobt.2022.797566/full)).
- Everything else as in [follow-line's research](../follow-line/research.md).
