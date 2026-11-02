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
| Its next mark on each side: forward, left and height in its frame | 6 | 1 | The newest print of its leader on that side; for unit 1, the live foothold |
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

The same 65 inputs: its marks are its leader's footholds, the frozen one
and the live one, and its leader's feet and energy are zeros, the head
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

## Sources

- Yasui, Sakai, Kano, Owaki & Ishiguro (2017), Decentralized control
  scheme for myriapod robot inspired by adaptive and resilient centipede
  locomotion, PLOS ONE ([PMC5289600](https://pmc.ncbi.nlm.nih.gov/articles/PMC5289600)).
