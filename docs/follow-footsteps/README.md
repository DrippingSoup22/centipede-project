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
| **B** | v6 half, its neck held straight | The head, an agent of its own, in a growing share of worlds | Holes, bumps and obstacles in one height field, and targets only the head senses |
| Later | v6 full | | Basins |

Because the marks set the wave's direction, the ground need not be final
from the first cycle, as it had to be in follow-line: stage A stays flat.

## The parts

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | Prints, marks and the relayed speed; stage A's script; episodes |
| Reward | [reward.md](reward.md) | One function for every walker, the marks' cost its only new term; the head's terms for stage B |
| Observations | [observations.md](observations.md) | 65 values: the unit itself, its leader's feet, its marks and its command; the head's grid of heights |
| Actions | [actions.md](actions.md) | Eight leg motors and the spine joint ahead; the head's footholds and speed |
| Critic and learning | [critic.md](critic.md) | follow-line's, each critic seeing what its actor sees |

## The bodies

Model v6 (`docs/model.md`, "Model v6"): stage A trains on its headless
tester, stage B on its half, whose head is its own body on a neck with a
motor on each hinge. Here no agent commands the neck: its motors' zero
command holds it straight, bent about 1° under the head standing and 0.6 to
1.5° on the scripted walk. The antennae are drawn shapes that touch
nothing. The head senses the ground through a grid of heights over the
places unit 1's feet can step ([observations.md](observations.md#the-head-stage-b)),
the simplest stand-in, as follow-line's front unit's view of the ground
was (the user, 2026-11-03).

## Kept from follow-line

The muscle energy and its price ramp, the carry per leg, the tow, the
speeds' schedule, two critics added 1 : 1, gSDE with its floor, the
observation and value normalisation, and the PPO settings.

## Left out

The line and its curves (stage A's script lays its own path), the line's
costs, the front unit's view of the ground (the head's grid of heights
takes its place in stage B), and the neighbours' last actions, values another agent
chose.

## Open

- **The head's numbers**, set before stage B from the scripted walk: the
  grid's reach and spacing, a foothold step's range, the price of a failed
  foothold and what counts as one.
- **Motors that lift the first few units**, as real centipedes lift their
  front, may come later (the user, 2026-11-02).

## Later: the antennae

Antennae are worth adding only as sensing organs, which this baseline does
not need (the user, 2026-11-03). Two ways were weighed and set aside: real
antennae, flexible and touching along their whole length (many contacts on
every step, which may slow the simulation, and which MuJoCo Warp may not
support), and rigid antennae that record only where they touch the ground
(they miss every hole they do not cross).

The user's idea, the starting point of that study, is a **torch**: each
antenna senses all the ground within a radius of its shape, as a torch
lights what is near it. It needs no contacts: the ground's heights at
points inside each antenna's capsule, read from the terrain as the grid's
are, so the code that reads the grid reads the torch too. It asks the head
for two things the grid does not: aiming its antennae with its neck (v6's
neck motors, unused here), and remembering what they sensed, since their
tips reach 44 mm ahead of unit 1's feet, about 0.6 s of walking
(`docs/model.md`, "Validation of v6"). Stick insects use their antennae
this way, actively, to place their front legs (Schütz & Dürr 2011).

## Sources

- Yasui et al., Scientific Reports 2019: the brain's command and
  each segment following its anterior segment ([PMC6889372](https://pmc.ncbi.nlm.nih.gov/articles/PMC6889372)).
- Yasui, Takano, Kano & Ishiguro, Frontiers in Robotics and AI 2022: each
  leg following the contact point of the leg ahead
  ([doi 10.3389/frobt.2022.797566](https://www.frontiersin.org/articles/10.3389/frobt.2022.797566/full)).
- Schütz, C. & Dürr, V. (2011), Active tactile exploration for adaptive
  locomotion in the stick insect, Phil. Trans. R. Soc. B 366, 2996–3005.
- Everything else as in [follow-line's research](../follow-line/research.md).
