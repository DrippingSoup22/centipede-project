# Task

Agreed with the user on 2026-11-02. Stage A is the baseline: the headless
tester, a scripted leader, flat ground.

## The body

v6 headless: ten units alike, unit 1 at the front (`segment_00` in the
model), ten agents ([README.md](README.md#the-bodies)).

## Prints, marks and the speed

Every walker has one **leader**, the unit ahead of it. Unit 1's leader is
the head; in stage A, a script.

- **A print** is where a foot landed: when a foot touches the ground after
  a swing, the point of contact. Every walker keeps its last two prints on
  each side.
- **A walker's marks** on each side are its leader's last two prints on
  that side. When the leader lands its next step, the follower is usually
  still standing on the earlier print, so both count.
- **Unit 1's marks** are the leader's footholds on each side: the one it
  was given when its foot last touched down, frozen at that moment, and the
  live one, where its next step on that side should land. The leader may
  move the live one at any time, as an antenna's touch re-aims a stick
  insect's front leg in its swing (Schütz & Dürr 2011).
- **The speed** passes back along the body. Unit 1 holds its leader's
  command one step later, and every other walker holds its leader's
  command one step later, unchanged: 20 ms a unit, 0.2 s at the rear. Each
  walker sees the command it holds, as it stands and averaged over 1 s, and
  only its own average is compared with its speed, so the delays do not
  add up into a slower command.

## Stage A's leader: a script

- **Its path** runs straight ahead of the body at first, then curves:
  follow-line's curved lines, piece by piece with bounded curvature, at
  levels that rise as the walk forms ([follow-line's task](../follow-line/task.md#the-curves-stage-2)).
  Nobody sees the path; it only places the footholds.
- **Its footholds.** Each side's next foothold lies one stride s along the
  path beyond that side's previous foothold, at the foot's natural reach
  to the side of the path, the left and right sides half a stride apart. A
  foothold follows the one the script gave, not where the foot landed, so a
  missed step does not bend the path.
- **Its speed** is follow-line's speeds' schedule: one speed at first, 67
  mm/s, then speeds that change, stops included, as levels rise
  ([follow-line's task](../follow-line/task.md#the-speeds-stage-3)).

## A world's episode

| Setting | Value | Why |
| --- | --- | --- |
| Start | The model's pose, each leg angle moved a little at random (main's reset). Each walker's first prints are where its feet stand, the script's first footholds one stride ahead of unit 1's | Every unit starts on its marks |
| Episode | 20 s, 1,000 decisions | As follow-line |
| A walker loses its marks | The world restarts when a walker's feet stay farther from their marks than a foot's reach for 1 s | No batch fills with steps where the marks pay nothing; the distance is set by the ledger |
| The tow | follow-line's: a force each unit feels, fading as the walk forms, zero in every evaluation | Lets fresh networks move before they can walk |
| Ground | Flat | The marks set the wave's direction, so stage A need not have stage B's ground |

Every ending is a truncation, as in follow-line: the agents value what
would have followed with their critics, as if the walk went on (Pardo et
al. 2018).

## Stage B, later

The head, an agent of its own on v6's half, chooses unit 1's footholds and
the speed, toward a target that only it senses, over one height field with
holes, bumps and obstacles. It takes over from the script in a growing
share of worlds, so that the walkers' task looks the same before and after.
Its design is settled before stage B.

## Sources

- Schütz, C. & Dürr, V. (2011), Active tactile exploration for adaptive
  locomotion in the stick insect, Phil. Trans. R. Soc. B 366, 2996–3005.
- Pardo, F., Tavakoli, A., Levdik, V. & Kormushev, P. (2018), Time limits
  in reinforcement learning, ICML.
