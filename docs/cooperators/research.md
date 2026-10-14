# Research

The projects and mechanisms the design draws on (searches of 2026-10-14),
sorted against the constraints in [README.md](README.md).

## The common pattern

Every project that got cooperation did two things. It paid only for the
team's outcome (the body's speed, progress or task, never how a limb
moved), and it gave learning a way to tell who helped: through the critic,
the reward, or what each agent senses. No project was found where each
segment of a centipede is its own learner.

## The closest projects

| Project | Agents | Each sees | Reward |
| --- | --- | --- | --- |
| Hexapod (Schilling et al., 2020) | 6, a network per leg | Own leg; both neighbouring legs' joints, contact and last actions; the body's orientation | The body's mean velocity, the same for every leg. Learned faster than one central network, and walked better |
| Snake (Sartoretti et al., IEEE T-RO 2019) | Body windows, one shared policy | Local only | The robot's forward progress |
| MASQ (2024) | 4, one per leg of a quadruped | Own leg and shared body signals | Shared; one critic for the whole robot (MAPPO). No ablation isolates the critic |
| Multi-Agent MuJoCo (FACMAC, NeurIPS 2021) | Segments of one chain, e.g. "ManyAgent Swimmer" | Joints within k hops | The robot's single reward, copied to every agent |
| MA2C (Chu et al., 2019) | One per traffic light | Itself and its neighbours | Its own plus its neighbours', discounted by distance |

## Mechanisms, sorted

| Mechanism | Source | Verdict | Why |
| --- | --- | --- | --- |
| Critic of the agent's neighbourhood | Qu, Wierman & Li, Operations Research 2022 | Kept | Nearly exact when interactions and rewards are local; [critic.md](critic.md) |
| Critic that sees the neighbours' actions | Wu et al., ICLR 2018 | Kept | Removes the neighbours' part from an agent's advantage, unbiased |
| Spatial discount α | MA2C | Kept | The cooperation dial for per-segment terms; [reward.md](reward.md) |
| A team goal paid to every limb | Hexapod, snake, Multi-Agent MuJoCo | Kept | Made visible to every segment by the path layer |
| Effort as a cost | Fu et al., CoRL 2021; Hoyt & Taylor, Nature 1981 | Kept | Gaits emerge from moving as cheaply as possible |
| Several critics with weighted advantages | van Seijen et al., NeurIPS 2017 | Kept | The head's task critic |
| Counterfactual baseline | COMA, Foerster et al., AAAI 2018 | Parked | A harder critic, built for discrete actions; only for a measured credit problem |
| Sequential updates | HAPPO, Kuba et al., ICLR 2022 | Parked | Only for measured instability between agents |
| One critic for the whole body | MAPPO, Yu et al., NeurIPS 2022 | Dropped | [critic.md](critic.md); a possible later comparison |
| Attention critic | Iqbal & Sha, ICML 2019 | Dropped | An internal design of a central critic |
| Agents sharing value estimates | Schneider et al., ICML 1999 | Dropped | Shares learned knowledge |
| Difference rewards | Wolpert & Tumer; Tumer & Agogino | Dropped | A re-simulation per segment; "without me" is ill-defined in a joined body |
| One policy for all followers | MAPPO, MASQ, Shared Modular Policies | Dropped | Shares networks |
| Learned messages between modules | Huang et al., ICML 2020; Pathak et al., NeurIPS 2019 | Dropped | Shares learned knowledge |
| Learned motor groups | DEMOS, Guo et al., 2023 | Dropped | The project fixes one agent per segment |
| Rewards for sameness, rivalry or influence | Jaques et al., ICML 2019, and others | Dropped | Sameness pays synchrony (oscillators' runs 17 and 27), rivalry makes neighbours compete, influence needs a model of the neighbours' policies |
| Step-quality rewards | The oscillators, runs 8 to 57 | Dropped | They prescribe a gait, and were disproved |

## Central versus local critics

Lyu et al. (AAMAS 2021) prove that, after convergence, central and local
critics give decentralized actors the same expected gradient, and that the
central critic's gradient has at least as much variance. Independent PPO
matched joint methods on StarCraft (de Witt et al., 2020), while MAPPO's
lead over it grew with the number of agents in Hanabi (Yu et al., 2022).

## Biology: where the wave comes from

- **Real centipedes** (Yasui et al., PLOS ONE 2017): a leg over a gap stops,
  and moves again when the leg ahead touches the ground. In their model each
  leg reacts only to its own and its neighbours' horizontal ground force,
  and a backward-travelling wave emerges.
- **The leg follows the leg ahead's foothold** (Yasui et al., Frontiers
  2022): own contact, the front neighbour's contact and the belly's contact
  give a tail-ward wave about five legs long from any starting pose; it
  needs both the adaptive legs and a flexible body.
- **Stick insects** (Cruse's rules, Walknet): neighbouring legs delay or
  trigger each other's lift-off; gaits emerge from this local coupling and
  the mechanics, with no central clock.
- **A quadruped robot** (Owaki & Ishiguro, Scientific Reports 2017): local
  load sensing alone gave walk, trot and gallop.

The wave comes from what each leg senses of its own and the leg ahead's
load and contact, not from a reward for the wave.

## Energy: work and force

- **Motors** (Seok et al., 2015): in the MIT Cheetah, heat in the actuators
  was 76% of the energy spent.
- **Muscle work** (Cavagna and Kaneko, 1977; Abbott, Bigland and Ritchie,
  1952): positive work at about 25% efficiency; negative work several times
  cheaper per joule.
- **Muscle force** (Kram and Taylor, Nature 1990): the energy of running is
  set mainly by producing force, its rate by body weight over the time a
  foot is on the ground; small animals pay more because they make force
  with fast fibres in many quick steps.
- **Size** (Taylor, Heglund and Maloiy, 1982; Heglund et al., 1982): the
  mechanical work per kilogram and metre hardly depends on size, while the
  energy per kilogram rises as animals get smaller; for an animal of a gram,
  the cost of force is almost all of it.
- **Speed** (Hoyt and Taylor, 1981): energy per distance is U-shaped in
  speed for each gait.

## Training practice for PPO with many agents

MAPPO's advice (Yu et al., 2022): normalise the value targets; at most 10
epochs on hard tasks; few minibatches; a clip of at most 0.2; large
batches.

## Sources

- Schilling, Konen, Ohl & Korthals, 2020, [arXiv 2005.11164](https://arxiv.org/abs/2005.11164)
- Sartoretti et al., IEEE T-RO 2019, [arXiv 1901.08537](https://arxiv.org/abs/1901.08537)
- MASQ, 2024, [arXiv 2408.13759](https://arxiv.org/abs/2408.13759)
- Peng et al., FACMAC and Multi-Agent MuJoCo, NeurIPS 2021, [arXiv 2003.06709](https://arxiv.org/abs/2003.06709)
- Chu, Wang, Codecà & Li, MA2C, IEEE T-ITS 2019, [arXiv 1903.04527](https://arxiv.org/abs/1903.04527)
- Qu, Wierman & Li, Operations Research 2022, [arXiv 1912.02906](https://arxiv.org/abs/1912.02906)
- Wu et al., ICLR 2018, [arXiv 1803.07246](https://arxiv.org/abs/1803.07246)
- Fu, Kumar, Malik & Pathak, CoRL 2021, [arXiv 2111.01674](https://arxiv.org/abs/2111.01674)
- Hoyt & Taylor, Nature 292, 1981, [doi 10.1038/292239a0](https://doi.org/10.1038/292239a0)
- van Seijen et al., Hybrid Reward Architecture, NeurIPS 2017, [arXiv 1706.04208](https://arxiv.org/abs/1706.04208)
- Foerster et al., COMA, AAAI 2018, [arXiv 1705.08926](https://arxiv.org/abs/1705.08926)
- Kuba et al., HAPPO, ICLR 2022, [arXiv 2109.11251](https://arxiv.org/abs/2109.11251)
- Yu et al., MAPPO, NeurIPS 2022, [arXiv 2103.01955](https://arxiv.org/abs/2103.01955)
- Iqbal & Sha, ICML 2019, [arXiv 1810.02912](https://arxiv.org/abs/1810.02912)
- Schneider, Wong, Moore & Riedmiller, ICML 1999, [CMU](https://www.ri.cmu.edu/publications/distributed-value-functions)
- Tumer & Agogino, difference rewards, [NASA](https://ntrs.nasa.gov/api/citations/20040068179/downloads/20040068179.pdf)
- Huang, Mordatch & Pathak, ICML 2020, [arXiv 2007.04976](https://arxiv.org/abs/2007.04976)
- Pathak et al., NeurIPS 2019, [arXiv 1902.05546](https://arxiv.org/abs/1902.05546)
- Guo et al., DEMOS, 2023, [arXiv 2306.17411](https://arxiv.org/abs/2306.17411)
- Lyu, Xiao, Daley & Amato, AAMAS 2021, [arXiv 2102.04402](https://arxiv.org/abs/2102.04402)
- de Witt et al., 2020, [arXiv 2011.09533](https://arxiv.org/abs/2011.09533)
- Yasui, Sakai, Kano, Owaki & Ishiguro, PLOS ONE 2017, [PMC5289600](https://pmc.ncbi.nlm.nih.gov/articles/PMC5289600)
- Yasui, Takano, Kano & Ishiguro, Frontiers in Robotics and AI 2022, [doi 10.3389/frobt.2022.797566](https://www.frontiersin.org/articles/10.3389/frobt.2022.797566/full)
- Schilling & Cruse, PLOS Computational Biology 2020, [PMC7205325](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7205325/)
- Owaki & Ishiguro, Scientific Reports 2017, [doi 10.1038/s41598-017-00348-9](https://doi.org/10.1038/s41598-017-00348-9)
- Seok et al., IEEE/ASME Transactions on Mechatronics 20(3), 2015, [MIT DSpace](https://dspace.mit.edu/handle/1721.1/108096)
- Cavagna & Kaneko, J. Physiol. 268, 1977, [PMC1283673](https://pmc.ncbi.nlm.nih.gov/articles/PMC1283673)
- Abbott, Bigland & Ritchie, The physiological cost of negative work, J. Physiol. 117, 1952
- Kram & Taylor, Nature 346, 1990
- Taylor, Heglund & Maloiy, J. Exp. Biol. 97, 1982, and Heglund, Fedak, Taylor & Cavagna, J. Exp. Biol. 97, 1982, [JEB](https://cob.silverchair.com/jeb/article/97/1/57/34676/Energetics-and-mechanics-of-terrestrial-locomotion)
- Ackermann & van den Bogert, J. Biomech. 43, 2010, [PMC2849893](https://pmc.ncbi.nlm.nih.gov/articles/PMC2849893)
- Biewener, Biomechanical consequences of scaling, J. Exp. Biol. 208, 2005
- Tucker, The energetic cost of moving about, American Scientist 63, 1975
