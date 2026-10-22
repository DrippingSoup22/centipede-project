# Observations

Agreed with the user on 2026-10-20; the forward line added for run 12
(2026-10-21).

## What a unit sees

**Its sight is a sphere** around its centre, as far as the far ends of the
two units ahead and behind when the body is straight and still: 2.5 unit
spacings, 16.5 mm on v5 (the user's definition).

- **The body** is seen by position in the chain: always itself and the two
  units ahead and behind, whatever the posture (radius 2). A missing
  neighbour is all zeros.
- **The environment** is seen as it is: only what lies inside the sphere at
  that moment, the line for now.
- **From run 16 (`actor_sees_behind` off) the actor does not use the units
  behind.** Seeing both sides, neighbouring units locked their steps but the
  body split into waves leaving a middle unit (run 15). A chain whose
  coupling runs mostly one way keeps one lag along its length (Kopell and
  Ermentrout's chains of oscillators; Williams & Sigvardt 1994 in the
  lamprey's cord), and centipede models get one wave from each leg
  following the leg ahead (Yasui et al. 2022). So each actor acts on
  itself, the two units ahead and its trail, and the front unit, with no
  one ahead, leads. Its first layer takes nothing from the units behind;
  its critics still see them ([critic.md](critic.md)). Who leads does not
  set the wave's direction, which the reward leaves free.

## A unit's block: its body, 46 values

| Part | Values | Why |
| --- | ---: | --- |
| Gravity's direction, in the unit's frame | 3 | Its tilt, without its heading in the world (legged_gym's projected gravity) |
| Its velocity, in its own frame | 3 | The cause of the line's pay (the pairing rule) |
| Its turning speed | 3 | Turning and rocking |
| Leg angles, leg speeds | 8 + 8 | Where its legs are and how they move |
| The spine joint behind it: angle, speed | 2 | The joint it commands; zeros for the rear |
| Foot loads: each foot pressing down, and pushing fore or aft | 4 | Load sensing, the signal biology puts at the root of the wave (Yasui et al. 2017; Dallmann et al. 2017; Owaki & Ishiguro 2017) |
| Its belly's load | 1 | Resting or dragging |
| Its feet's load, averaged over 1 s (from run 14 each foot's: 2) | 1 | What its carry is charged on (the pairing rule) |
| Its energy, averaged over 1 s, mJ/s | 1 | What its team account is charged, with its neighbours' (the pairing rule) |
| Its forward line, averaged over 1 s, mm/s | 1 | What the commanded speed compares (the pairing rule): with its neighbours', most of the body's speed. Its own velocity swings within every stride and cannot tell it |
| Joint forces of the step: the front joint's along the body, and its two joints' net force along | 2 | A reflex: being pulled now |
| Its last action | 9 | What it is doing; across the blocks, what its neighbours just did |

Forces are in unit weights, means over the step's physics steps. An agent
sees five blocks: its own, then the two ahead, nearest first, then the two
behind.

Settings leave out what a run's reward does not use: the forces only
earlier costs priced (`observe_priced_forces`), the averaged load
(`observe_carry`), the averaged energy (`observe_energy`) and the averaged
forward line (`observe_line_speed`).

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

5 × 46 + 15 + 1 = **246 values** for each of the ten agents; from run 14,
with each foot's averaged load, 5 × 47 + 15 + 1 = **251**. Each agent
normalises its own inputs with running statistics.

## Left out

- **A learned memory** (a recurrent actor), which could carry a step's
  phase: the next step if no wave forms (the user's decision, 2026-10-21).
