# Observations

Agreed with the user on 2026-10-20.

## What a unit sees

**Its sight is a sphere** around its centre, as far as the far ends of the
two units ahead and behind when the body is straight and still: 2.5 unit
spacings, 16.5 mm on v5 (the user's definition).

- **The body** is seen by position in the chain: always itself and the two
  units ahead and behind, whatever the posture (radius 2). A missing
  neighbour is all zeros.
- **The environment** is seen as it is: only what lies inside the sphere at
  that moment, the line for now.

## A unit's block: its body, 45 values

| Part | Values | Why |
| --- | ---: | --- |
| Gravity's direction, in the unit's frame | 3 | Its tilt, without its heading in the world (legged_gym's projected gravity) |
| Its velocity, in its own frame | 3 | The cause of the line's pay (the pairing rule) |
| Its turning speed | 3 | Turning and rocking |
| Leg angles, leg speeds | 8 + 8 | Where its legs are and how they move |
| The spine joint behind it: angle, speed | 2 | The joint it commands; zeros for the rear |
| Foot loads: each foot pressing down, and pushing fore or aft | 4 | Load sensing, the signal biology puts at the root of the wave (Yasui et al. 2017; Dallmann et al. 2017; Owaki & Ishiguro 2017) |
| Its belly's load | 1 | Resting or dragging |
| Its feet's load, averaged over 1 s | 1 | What its carry is charged on (the pairing rule) |
| Its energy, averaged over 1 s, mJ/s | 1 | What the team's share is drawn from (the pairing rule); with its neighbours', the team's |
| Joint forces of the step: the front joint's along the body, and its two joints' net force along | 2 | A reflex: being pulled now |
| Its last action | 9 | What it is doing; across the blocks, what its neighbours just did |

Forces are in unit weights, means over the step's physics steps. An agent
sees five blocks: its own, then the two ahead, nearest first, then the two
behind.

Settings leave out what a run's reward does not use: the forces only
earlier costs priced (`observe_priced_forces`), the averaged load
(`observe_carry`) and the averaged energy (`observe_energy`).

## The trail it sees: 15 values

Five points of the line, 6.6 mm apart (one unit spacing), centred on the
line's point nearest the unit's centre: at −13.2, −6.6, 0, +6.6 and
+13.2 mm along the line. Each point is three values: its position forward
and to the left in the unit's frame (metres), and 1 if it lies inside the
sphere, seen from above, else 0 with its position zeroed. On a straight
line they say only where the line is; when it curves, they show the bend
before the unit reaches it, which matters most for the front unit, with
nothing ahead of it.

## The tow: 1 value

Its world's tow level, a force each unit feels on itself, zero in every
evaluation ([task.md](task.md#the-tow)).

## In all

5 × 45 + 15 + 1 = **241 values** for each of the ten agents. Each agent
normalises its own inputs with running statistics.
