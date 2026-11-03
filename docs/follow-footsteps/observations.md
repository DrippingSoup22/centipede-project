# Observations

Agreed with the user on 2026-11-02, as small as the job allows: each
walker sees itself and its leader, the unit ahead, and nothing else.
follow-line's view of five units, its line and the front unit's ground
view are gone.

## What a walker sees: 65 values

Priority 1 is what the pay or the control cannot do without; priority 2 is
useful, cheap, and proven in follow-line. Both are in the baseline; what
was left out is listed below, a candidate for later.

| Input | Values | Priority | Why |
| --- | ---: | :---: | --- |
| Its leg angles and leg speeds | 16 | 1 | Where its feet are and where they are going |
| Each foot pressing down | 2 | 1 | Stance or swing: the marks are charged during the stance |
| Gravity's direction in its frame | 3 | 1 | Balance |
| The spine joint it commands, the one ahead ([actions.md](actions.md)): angle and speed | 2 | 1 | Its heading against its leader's |
| Its next mark on each side: forward, left and height in its frame | 6 | 1 | The newest print of its leader on that side; for unit 1, the next foothold its leader gave |
| Each foot's distance to its nearer mark | 2 | 1 | What the marks charge (the pairing rule) |
| The command it holds, averaged over 1 s | 1 | 1 | What its speed is compared with |
| Its speed, each foot's load and its energy, each averaged over 1 s | 4 | 1 | What the other terms are charged on |
| Its leader's energy, averaged over 1 s | 1 | 1 | Its energy term includes it; 0 for unit 1 |
| Its leader's feet now: where (6), and whether each is down (2) | 8 | 2 | The next print forms where that foot lands: with long steps a follower must begin its swing before the print exists. Down tells it the print is still taken. Zeros for unit 1 |
| The command as it stands | 1 | 2 | A change shows at once, not 1 s later |
| Each foot pushing fore or aft | 2 | 2 | Load sensing, the signal centipede models build the wave on (Yasui et al. 2017) |
| Its own velocity, turning speed and belly load | 7 | 2 | Proven in follow-line |
| Its last action | 9 | 2 | Proven in follow-line; what it is doing |
| The tow | 1 | 2 | Training only, as in follow-line |

Positions are in metres in the walker's own frame, forces in unit weights,
as in follow-line. Each walker normalises its own inputs with statistics
over about the last 10 update cycles, as in follow-line's stage 3.

## Unit 1

The same 65 inputs: its marks are its leader's footholds, the one its
foot went for and the next one, and its leader's feet and energy are zeros, the head
having none to share. Inputs that are always zero never move its network.

## Left out

- **Its leader's body**: follow-line's block of its leg angles, speeds and
  loads. The leader's feet and prints carry what following needs.
- **Its neighbours' last actions**, values another agent chose.
- **The units behind.** Each unit commands the spine joint ahead of it and
  is charged only for itself, so nothing behind is its doing.
- **The joint forces**, follow-line's reflex: optional, for later.
- **More leaders.** Seeing the prints of two or more units ahead and
  averaging them could slow the growth of the marks' error with depth: a
  later study against this baseline (the user, 2026-11-02).

## The head (stage B)

Agreed with the user on 2026-11-03. Positions are in unit 1's frame: with
the neck held straight, the head moves with unit 1.

| Input | Values | Why |
| --- | ---: | --- |
| The ground's height at each point of its grid | set by the grid | Where the footholds can go |
| Its target: direction (cosine, sine) and distance | 3 | Where to lead |
| Unit 1's feet: where (6), and whether each is down (2) | 8 | When the next touchdown comes, and where the next step starts |
| The footholds it last gave, one a side: forward, left and height | 6 | What unit 1 is going for |
| Unit 1's velocity and turning speed | 3 | How the body moves |
| Gravity's direction | 3 | Slopes |
| Its last action | 5 | What it is doing |

### The grid of heights

The head's sensor: points on two level strips, one over each line of unit
1's feet, turning with unit 1's heading, from behind the feet to beyond the
farthest foothold the head can give. Each value is the ground's height at
its point less the height of unit 1's centre, read from the terrain's map
(`Ground.heights`, as follow-line's view of the ground ahead), not from a
ray: the head sees behind a bump, a stand-in for senses it does not have,
as follow-line's front unit's view was (the user, 2026-10-24). This is the
usual way a learned legged controller perceives the ground: Rudin et al.
(2021) give theirs 108 heights on a grid around the robot's body, each
less the body's height, Miki et al. (2022) 52 around each foot.

- **No memory.** The points lie where the footholds go, so the head sees
  the ground there at the moment it gives one.
- **Its size is a setting**: how far the strips reach ahead and behind,
  their width and the spacing between points; the number of points
  follows. The defaults are set before stage B from the scripted walk's
  stride and the foot's reach. On flat ground every value is the same, so
  a run on flat ground can leave the grid out.
- **The antennae later.** The grid is one set of points whose heights the
  ground gives; the torch antennae ([README.md](README.md#later-the-antennae))
  would be another, inside each antenna's capsule, read the same way.

## Sources

- Yasui, Sakai, Kano, Owaki & Ishiguro (2017), Decentralized control
  scheme for myriapod robot inspired by adaptive and resilient centipede
  locomotion, PLOS ONE ([PMC5289600](https://pmc.ncbi.nlm.nih.gov/articles/PMC5289600)).
- Rudin, Hoeller, Reist & Hutter (2021), Learning to walk in minutes using
  massively parallel deep reinforcement learning, CoRL 2021, PMLR 164
  ([arXiv 2109.11978](https://arxiv.org/abs/2109.11978)).
- Miki, Lee, Hwangbo, Wellhausen, Koltun & Hutter (2022), Learning robust
  perceptive locomotion for quadrupedal robots in the wild, Science
  Robotics 7(62) ([arXiv 2201.08117](https://arxiv.org/abs/2201.08117)).
