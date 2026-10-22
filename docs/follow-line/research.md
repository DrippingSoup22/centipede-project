# Research

What the literature knows about the two problems this project combines:
learning to walk with reinforcement learning, and learning to cooperate when
each agent sees only its neighbours; and how real centipedes walk. Searched
and checked against the papers on 2026-10-21, after run 11. Quotes are the
papers' own words, checked in their text; everything else is a summary. The
cooperators study's earlier search (energy, PPO practice, exploration noise,
turning) is in `docs/cooperators/research.md` on that branch.

## In short

- **Tapping is the known default of locomotion learning.** Rewarded for
  moving, simulated legged bodies learn high-frequency tapping or vibrating
  legs unless something shapes the gait. Natural gaits appear only with
  added structure: a commanded speed, an energy cost, smoothness, varied
  terrain, or a gait prior (a clock, an oscillator, a trajectory generator,
  motion examples). None of the works found gets a natural gait from
  distance alone on flat ground.
- **Splitting a body among agents works, but almost every success shares
  something**: one network for all agents, or a critic that sees the whole
  body in training. Both are outside this project's constraints
  ([constraints.md](constraints.md)). The closest to them is a hexapod with
  six separate leg networks and local views (Schilling et al. 2020).
- **Independent learners have named failures**: relative overgeneralisation
  (a worse joint behaviour wins because it looks good whatever the partners
  do) and lazy agents. Theory says a critic of the agent's neighbourhood is
  enough when interactions are local (Qu, Wierman & Li 2022); the remedies
  that stay local work on how advantages are judged or on how the reward is
  built.
- **Real centipedes** walk with a wave running from front to rear, about
  five legs long, and change speed mostly by the stride's length and the
  time on the ground, not by stepping faster. Models produce the wave from
  each leg sensing its own and its front neighbour's contact or load, with
  no central clock.

## 1. Natural gaits versus tapping

| Study | Body, actions, rate | What shaped the gait | What happened |
| --- | --- | --- | --- |
| Fu, Kumar, Malik & Pathak, CoRL 2021 | A1 quadruped; offsets from a standing pose (±0.15 to ±0.4 rad) to PD targets; 100 Hz; one central policy | A commanded forward speed (penalised deviation), the motors' energy (torque × joint speed), a survival bonus; fractal (rough) terrain in training | With energy: walk, trot and gallop at different commanded speeds, like horses'. Without the energy term: "All policies show high-frequency tapping behaviors that are not natural and inefficient." On flat ground without the fractal terrain, even with energy: "All training trials converge to unnatural and unstable gaits." |
| Escontrela et al., 2022 | A1 quadruped; PD targets; 30 Hz | Speed tracking plus a style reward from a discriminator trained on real dog motion (adversarial motion priors) | Natural gaits switching with speed, cheaper than the hand-made reward's. Without the style reward the policy "learns to move by vibrating its legs at high speeds with large torques" |
| Rudin, Hoeller, Reist & Hutter, CoRL 2021 (legged_gym) | ANYmal; offsets from a default pose to PD targets; 50 Hz | Speed-command tracking, penalties on action rate (the change of the action between decisions), torques, joint accelerations, collisions, and a term that pays each foot's air time above 0.5 s, commented "Reward long steps" in the code | Always a trot, often with artifacts (a dragging leg) |
| Hwangbo et al., Science Robotics 2019 | ANYmal; PD targets | Cost terms (torque, joint speed, foot clearance and slip, smoothness) scaled from 0.3 up to full over training | Their reason: strong penalties on torque and speed from the start make standing still the answer, weak ones give unnatural motion. A trot emerged |
| Peng & van de Panne, SCA 2017 | 2D biped, raptor, dog; torques, muscle activations, PD targets or target speeds | Imitation of reference motions, with a phase input | PD targets learned fastest and most robustly, torques among the worst; muscle activations slowest to learn but the most natural and smoothest motion |
| Siekmann et al., ICRA 2021 | Cassie biped; PD offsets and gains; 40 Hz; recurrent policy | Periodic rewards on foot force and speed, gated by a phase clock observed as sines; speed tracking; smoothness | Every common bipedal gait on the real robot, chosen by the clock's parameters |
| Iscen et al. 2018 (PMTG); Lee et al., Science Robotics 2020 | Minitaur; ANYmal (400 Hz) | The policy modulates a periodic foot trajectory generator (its frequency, amplitude and residuals) | Without the generator, learning reached less reward and could not track changing speeds (PMTG). ANYmal over rough terrain with a trot only |
| Bellegarda & Ijspeert, RA-L 2022 (CPG-RL) | A1 quadruped; the policy sets each leg's oscillator (amplitude, frequency 0 to 4.5 Hz) | One oscillator per leg, uncoupled; speed tracking, work penalty | Compared with joint targets, "a continuum of more natural gaits that are less rigid, more efficient, more robust, and have lower frequency" |
| Margolis & Agrawal, CoRL 2022 (Walk These Ways) | Go1; PD targets; 50 Hz | Gait parameters (contact timing, frequency, foot swing) in the command and the reward | A policy without them settles on one gait, chosen at random |
| Heess et al. 2017 | Torque-driven walker, quadruped, humanoid | Simple rewards (forward speed, control cost) over varied obstacle courses with curricula | Robust behaviour emerged from simple rewards, but some gaits "visually unsatisfactory" |
| Mysore et al., ICRA 2021 (CAPS) | Gym tasks and a real drone | Penalties on the change of the policy's action between consecutive states, and between nearby states | RL controllers oscillate at high frequency; the penalties made every tested policy smoother, the drone's current fell from about 23 A to 5 A |
| Seyde et al., NeurIPS 2021 | DeepMind Control tasks | None | Gaussian policies tend toward extreme, bang-bang actions; action penalties reduce it (the paper does not measure switching frequency) |

**What it adds up to.** No study found gets a natural gait from paying
for distance on flat ground. Every natural gait came with at least one of:

| Ingredient | Where | Works through |
| --- | --- | --- |
| A commanded speed, not "the faster the better" | Fu; legged_gym; most of the above | Speed is set; the gait is chosen at that speed |
| An energy or effort cost | Fu; CPG-RL; legged_gym (torques) | Among gaits at the speed, the cheapest wins |
| Smoothness: the action's rate of change penalised | legged_gym; Siekmann; CAPS; Hwangbo | Quick switching costs |
| Varied or rough terrain | Fu; Heess | Tapping and shuffling fail on uneven ground |
| Steps rewarded for their length or air time | legged_gym | Directly against short steps (it shapes the gait's form) |
| A gait prior: clock, oscillator, trajectory generator, motion examples | Siekmann; PMTG; CPG-RL; Escontrela; Peng & van de Panne | The rhythm comes from the prior, the policy adjusts it |
| Costs scaled up over training | Hwangbo | Standing still is not learned first |

Most of these studies also act through small offsets around a standing
pose, not across the joints' whole range.

## 2. A body split among agents

| Study | Agents | Each sees | Shared | Reward | Result |
| --- | --- | --- | --- | --- | --- |
| Schilling, Konen, Ohl & Korthals, IROS 2020 | 6, one network per leg of a simulated hexapod; 25 Hz | Its own leg, both neighbouring legs (joints, contact, last actions), the body's orientation | Nothing found shared | The body's mean velocity, the same for every leg | Learned faster than one central network and walked as well or better; from crawling to walking lifted off the ground. Gait not compared to insects |
| Sartoretti et al., IEEE T-RO 2019 | Snake: windows of joints; hexapod: one per leg | Local only | One shared policy | Snake: the robot's progress; hexapod: each leg's own term | The distributed hexapod converged in about 176 episodes, the central one not in 500. Discrete actions: the snake's wave amplitude and frequency, the hexapod's shoulder offsets |
| Huang, Mordatch & Pathak, ICML 2020 (shared modular policies) | One module per limb | Its own limb and learned messages from its neighbours | One module network for every limb | The body's | Messages both ways beat decentralised modules without them; generalises to new bodies |
| Kurin et al., ICLR 2021 | One node per limb | Neighbours through a graph, or every limb through attention | Shared networks | The body's | The body's graph did not help: attention over all limbs did better |
| Liu et al. 2024 (MASQ) | 4, one per leg of a quadruped | Own leg and body signals, a gait timing signal | One actor; a critic of the whole robot (MAPPO) | Shared; many shaping terms | Faster than a single-agent baseline in simulation; no ablation |
| Multi-Agent MuJoCo (Peng et al., NeurIPS 2021) | Segments of one robot, e.g. 10 agents of 2 joints each on a long swimmer; many-segment ants | Its joints and those within k hops | The algorithms solving it (FACMAC, HAPPO, Multi-Agent Transformer) use a critic or value of the whole body | The robot's single reward, copied | The central or factored critics did best; gaits not judged |

No reinforcement-learning study of a true centipede or myriapod body was
found (search of 2026-10-21; a few robot papers known only by their
abstracts).

**What it adds up to.** Separate local leg controllers can learn to walk
(Schilling et al.), even faster than one central network. But nearly every
multi-agent success shares a network between agents or trains with a
whole-body critic, and none judges whether the gait is natural. Our
setting, separate networks and critics that see only a neighbourhood, is
the least explored.

## 3. Cooperation when each agent sees only its neighbours

### The failures independent learners meet

| Failure | Source | What it is | In our runs |
| --- | --- | --- | --- |
| Relative overgeneralisation | Wei & Luke, JMLR 2016 | A worse joint behaviour wins because each agent's part of it looks good on average against partners that are still exploring; the better joint behaviour pays only when the partners do their part | Every leg paddling on its own pays whatever its neighbours do; a wave pays only if the neighbours keep time. No wave above chance in runs 09 and 11 |
| Lazy agents | Sunehag et al., AAMAS 2018 | With a team reward, one agent works and another idles | Idle legs in runs 05, 09 and 11 |
| Non-stationarity, shadowed equilibria, alter-exploration | Matignon, Laurent & Le Fort-Piat, Knowledge Engineering Review 2012 (from its abstract) | Each learner's world changes as the others learn; the others' exploring hides good actions | The team critic losing sight while the walk forms (runs 06 to 09) |

### What theory and practice say about local critics

- **A neighbourhood critic is enough when interactions are local** (Qu,
  Wierman & Li, Operations Research 2022). How much an agent's value
  depends on a distant agent decays exponentially with their distance, so a
  critic that sees the states and actions within κ hops loses little, its
  error shrinking about as ρ^(κ+1). Their proof assumes discrete states and
  actions and local interactions.
- **Central and local critics give the same expected gradient once they
  have converged**; the central one has more variance, the local one more
  bias early (Lyu, Xiao, Daley & Amato, AAMAS 2021).
- **Independent PPO is strong** on StarCraft (de Witt et al. 2020) and in
  MAPPO's comparisons (Yu et al., NeurIPS 2022), but both share one network
  between agents, and sharing mattered: on one map MAPPO won 90.6% of games
  with sharing and 13.0% without. Sharing speeds learning in general (Terry
  et al. 2020) and hurts agents that must behave differently (Christianos
  et al., ICML 2021).

### Remedies that stay local

| Remedy | Source | How | Local? |
| --- | --- | --- | --- |
| Leniency | Palmer, Tuyls, Bloembergen & Savani, AAMAS 2018 | Early in learning, an agent ignores many of its bad outcomes, which its partners' exploring may have caused | Yes |
| Optimistic advantages | Zhao et al., ICML 2024 | Negative advantages clipped to zero; reached the joint optimum in matrix games, better on 13 of 19 tasks | The clipping is; they used a central value |
| Rewards each agent can learn from | Agogino & Tumer, JAAMAS 2008 | A reward is judged by factoredness (aligned with the team's) and learnability (how much of it the agent itself causes); the difference reward, the team's outcome minus the outcome without the agent, has both | The reward may use anything ([constraints.md](constraints.md)); "without the agent" is hard to define in a joined body |
| Neighbours' rewards, discounted with distance | Chu et al. (MA2C), IEEE T-ITS 2019 | Each agent's reward adds its neighbours', scaled by 0.75 per hop; neighbours' policies observed | Yes |
| Team and own accounts with their own critics | Sheikh & Bölöni 2020; Wang et al. (IRAT), ICML 2022 | A shared reward and an individual one, each learned by its own critic | Yes; this project's two critics |

## 4. How centipedes walk

| Source | Animal | Measured or shown |
| --- | --- | --- |
| Anderson, Shultz & Jayne, J. Exp. Biol. 1995 | *Scolopendra heros* on a treadmill, 0.5 to 1.5 body lengths a second | The body's bending waves travel from head to tail at every speed, actively driven by the axial muscles; bending grows with speed. Stride frequency about 1.9 to 3.45 a second over those speeds (from this study's earlier reading of the paper; not re-checked here) |
| Kuroda, Uchida & Nakagaki, bioRxiv 2018 (preprint) | *Scolopocryptops rubiginosus*, 46 mm | Walking on flat ground, the leg waves are retrograde, running from front to rear, under five legs long. Speed comes from the stride: "the velocity variations were provided by variations in the span and the duty factor", while "the rotation frequency was constant". Direct waves (rear to front) appear only in swimming and fast dashes |
| Yasui et al., PLOS ONE 2017 | *Scolopendra subspinipes*, 14 to 16 cm; a 20-legged model | A retrograde wave emerges in the model from each leg sensing its own and its neighbours' horizontal ground force, fully decentralised, no central pattern generator |
| Yasui, Takano, Kano & Ishiguro, Frontiers in Robotics and AI 2022 | *S. subspinipes mutilans*, 9 cm; a 21-legged model | Each leg follows the ground contact of the leg ahead on its side; a wave about five legs long emerges from any start; it needs both these legs and a flexible body. Live centipedes on scaffolds support the rule |
| Aoi, Egi & Tsuchiya, Physical Review E 2013 | Model and centipedes | The body's undulation is absent at low speeds and appears beyond a critical speed, as an instability of straight walking |
| Schilling & Cruse, PLOS Computational Biology 2020 (neuroWalknet) | Stick insect model | Local rules between neighbouring legs (a swinging leg inhibits the lift-off of the leg ahead; ending a swing excites it; position and load) give a continuum of gaits set by one speed input, no central pattern generator |
| Owaki & Ishiguro, Scientific Reports 2017 | Quadruped robot | One oscillator per leg, coupled only through each foot's load: walk, trot and gallop as speed rises |
| Ozkan-Aydin, Chong, Aydin & Goldman, RoboSoft 2020 | *S. heros*, 20 cm, on rough ground (a photo, Fig. 4a) | The trunk bends up and down as well as sideways; the body dips gently into hollows, spread over about ten segments (a few degrees a joint, read from the photo) |
| Diaz, Erickson, Chong, Soto & Goldman, J. Exp. Biol. 2023 | *S. polymorpha*, *Scolopocryptops sexspinosus* on rugose ground | On flat ground the body lifts while walking; over large gaps segments are held in the air; limbs bend passively when they hit a block |
| Pierce et al., Annals of the New York Academy of Sciences 2026 | *S. polymorpha*, 8.6 cm, through lattices of posts | In narrow lattices the body twists onto its side locally (at least four segments), and untwists again; complete twists not above 1.5 times the body's width |

No measurement of the trunk's stiffness or of its range up-down or in twist
was found (search of 2026-10-22): the papers show these bends, they do not
measure them. One model infers that *S. heros* stiffens its trunk to run
faster, sideways (Dionne, Giardina & Mahadevan 2026, preprint).

No recording of a central pattern generator in a centipede's nerve cord was
found in one search (a thesis reports head-removal experiments and a
model); the proposals are models.

**What it adds up to.** In real centipedes the wave runs front to rear,
about five legs long, and speed comes mostly from longer, shorter-loaded
strides at a nearly constant tempo. Models get the wave from what each leg
senses of its own and its front neighbour's contact or load. A rhythm in
these models comes from the leg's own dynamics or oscillator, set by
speed; local sensing coordinates it.

**Who sets a wave's lag** (searched 2026-10-22, after run 15's two waves
leaving a middle unit). In a chain of oscillators coupled to their
neighbours, coupling that runs mostly one way gives one lag along the whole
chain, except in a short boundary at one end; Williams & Sigvardt (1994)
found that boundary in the isolated lamprey cord, as Kopell and
Ermentrout's theory predicts, and read it as the coupling running up the
cord dominating (from their abstract; the theory is reviewed by Ashwin,
Coombes & Nicks 2016). The dominant coupling there runs tail to head while
the wave runs head to tail: who leads does not set the direction. A 2026
segmented robot of 6 to 16 legs gives each segment input from the segment
in front (Chen, Wang & Revzen, from its abstract; it uses state machines,
not learning).

## 5. Our runs against the literature

| What runs 01 to 11 showed | What the literature says |
| --- | --- |
| Quick tapping, 8 to 16 steps a second, in every run, whatever the reward or the motor | The known default of locomotion learning without gait structure (Fu; Escontrela; CAPS) |
| Runs paid for distance, uncapped from run 09; flat ground throughout | The structures that give natural gaits include a commanded speed and varied terrain (Fu) |
| The best waves (0.35 to 0.37, runs 07 and 08) came with jerky motion priced, run 08's also with a capped line | Smoothness, and a commanded speed, as in the standard recipe |
| No wave in runs 09 and 11; each leg paddles on its own | Relative overgeneralisation |
| Idle legs | Lazy agents |
| Targets across the whole range tapped faster than torques (run 11) | The studies using targets act through small offsets around a standing pose, with smoothness or energy terms |
| The walk born tapping, then sped up | Hwangbo: what is learned first shapes what follows |

## Sources

Locomotion learning
- Fu, Kumar, Malik & Pathak, Minimizing energy consumption leads to the emergence of gaits in legged robots, CoRL 2021, [arXiv 2111.01674](https://arxiv.org/abs/2111.01674)
- Escontrela et al., Adversarial motion priors make good substitutes for complex reward functions, 2022, [arXiv 2203.15103](https://arxiv.org/abs/2203.15103)
- Rudin, Hoeller, Reist & Hutter, Learning to walk in minutes using massively parallel deep reinforcement learning, CoRL 2021, [arXiv 2109.11978](https://arxiv.org/abs/2109.11978); legged_gym's [reward code](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) and [configuration](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- Hwangbo et al., Learning agile and dynamic motor skills for legged robots, Science Robotics 4, 2019, [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)
- Peng & van de Panne, Learning locomotion skills using DeepRL: does the choice of action space matter?, SCA 2017, [arXiv 1611.01055](https://arxiv.org/abs/1611.01055)
- Siekmann, Godse, Fern & Hurst, Sim-to-real learning of all common bipedal gaits via periodic reward composition, ICRA 2021, [arXiv 2011.01387](https://arxiv.org/abs/2011.01387)
- Iscen et al., Policies modulating trajectory generators, CoRL 2018, [arXiv 1910.02812](https://arxiv.org/abs/1910.02812)
- Lee, Hwangbo, Wellhausen, Koltun & Hutter, Learning quadrupedal locomotion over challenging terrain, Science Robotics 5, 2020, [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
- Bellegarda & Ijspeert, CPG-RL: learning central pattern generators for quadruped locomotion, IEEE RA-L 2022, [arXiv 2211.00458](https://arxiv.org/abs/2211.00458)
- Margolis & Agrawal, Walk these ways, CoRL 2022, [arXiv 2212.03238](https://arxiv.org/abs/2212.03238)
- Heess et al., Emergence of locomotion behaviours in rich environments, 2017, [arXiv 1707.02286](https://arxiv.org/abs/1707.02286)
- Mysore, Mabsout, Mancuso & Saenko, Regularizing action policies for smooth control with reinforcement learning, ICRA 2021, [arXiv 2012.06644](https://arxiv.org/abs/2012.06644)
- Seyde et al., Is bang-bang control all you need?, NeurIPS 2021, [arXiv 2111.02552](https://arxiv.org/abs/2111.02552)

A body split among agents
- Schilling, Konen, Ohl & Korthals, Decentralized deep reinforcement learning for a distributed and adaptive locomotion controller of a hexapod robot, IROS 2020, [arXiv 2005.11164](https://arxiv.org/abs/2005.11164)
- Sartoretti et al., Distributed learning of decentralized control policies for articulated mobile robots, IEEE T-RO 2019, [arXiv 1901.08537](https://arxiv.org/abs/1901.08537)
- Huang, Mordatch & Pathak, One policy to control them all, ICML 2020, [arXiv 2007.04976](https://arxiv.org/abs/2007.04976)
- Kurin, Igl, Rocktäschel, Böhmer & Whiteson, My body is a cage, ICLR 2021, [arXiv 2010.01856](https://arxiv.org/abs/2010.01856)
- Liu et al., MASQ: multi-agent reinforcement learning for single quadruped robot locomotion, 2024, [arXiv 2408.13759](https://arxiv.org/abs/2408.13759)
- Peng et al., FACMAC and Multi-Agent MuJoCo, NeurIPS 2021, [arXiv 2003.06709](https://arxiv.org/abs/2003.06709); [MaMuJoCo documentation](https://robotics.farama.org/envs/MaMuJoCo)
- Kuba et al., Trust region policy optimisation in multi-agent reinforcement learning (HAPPO), ICLR 2022, [arXiv 2109.11251](https://arxiv.org/abs/2109.11251)
- Wen et al., Multi-agent reinforcement learning is a sequence modeling problem (Multi-Agent Transformer), NeurIPS 2022, [arXiv 2205.14953](https://arxiv.org/abs/2205.14953)

Cooperation with local views
- Wei & Luke, Lenient learning in independent-learner stochastic cooperative games, JMLR 17, 2016, [JMLR](https://jmlr.org/papers/v17/15-417.html)
- Sunehag et al., Value-decomposition networks for cooperative multi-agent learning, AAMAS 2018, [arXiv 1706.05296](https://arxiv.org/abs/1706.05296)
- Matignon, Laurent & Le Fort-Piat, Independent reinforcement learners in cooperative Markov games: a survey regarding coordination problems, Knowledge Engineering Review 27(1), 2012, [doi 10.1017/S0269888912000057](https://doi.org/10.1017/S0269888912000057)
- Qu, Wierman & Li, Scalable reinforcement learning for multi-agent networked systems, Operations Research 2022, [arXiv 1912.02906](https://arxiv.org/abs/1912.02906)
- Lyu, Xiao, Daley & Amato, Contrasting centralized and decentralized critics in multi-agent reinforcement learning, AAMAS 2021, [arXiv 2102.04402](https://arxiv.org/abs/2102.04402)
- de Witt et al., Is independent learning all you need in the StarCraft multi-agent challenge?, 2020, [arXiv 2011.09533](https://arxiv.org/abs/2011.09533)
- Yu et al., The surprising effectiveness of PPO in cooperative multi-agent games, NeurIPS 2022, [arXiv 2103.01955](https://arxiv.org/abs/2103.01955)
- Terry, Grammel, Hari & Santos, Revisiting parameter sharing in multi-agent deep reinforcement learning, 2020, [arXiv 2005.13625](https://arxiv.org/abs/2005.13625)
- Christianos, Papoudakis, Rahman & Albrecht, Scaling multi-agent reinforcement learning with selective parameter sharing, ICML 2021, [arXiv 2102.07475](https://arxiv.org/abs/2102.07475)
- Palmer, Tuyls, Bloembergen & Savani, Lenient multi-agent deep reinforcement learning, AAMAS 2018, [arXiv 1707.04402](https://arxiv.org/abs/1707.04402)
- Zhao, Zhao, Li, Kannala & Pajarinen, Optimistic multi-agent policy gradient, ICML 2024, [arXiv 2311.01953](https://arxiv.org/abs/2311.01953)
- Agogino & Tumer, Analyzing and visualizing multiagent rewards in dynamic and stochastic domains, JAAMAS 17, 2008, [Oregon State](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_jaamas08.html)
- Chu, Wang, Codecà & Li, Multi-agent deep reinforcement learning for large-scale traffic signal control, IEEE T-ITS 2019, [arXiv 1903.04527](https://arxiv.org/abs/1903.04527)
- Sheikh & Bölöni, Multi-agent reinforcement learning for problems with combined individual and team reward, IJCNN 2020, [arXiv 2003.10598](https://arxiv.org/abs/2003.10598)
- Wang et al., Individual reward assisted multi-agent reinforcement learning (IRAT), ICML 2022, [PMLR 162](https://proceedings.mlr.press/v162/wang22ao.html)

Centipedes and decentralised walking
- Anderson, Shultz & Jayne, Axial kinematics and muscle activity during terrestrial locomotion of the centipede *Scolopendra heros*, J. Exp. Biol. 198, 1995, [JEB](https://journals.biologists.com/jeb/article/198/5/1185/6885/Axial-Kinematics-and-Muscle-Activity-During)
- Kuroda, Uchida & Nakagaki, Dynamic gait transition in the Scolopendromorpha *Scolopocryptops rubiginosus* L. Koch centipede, bioRxiv 2018 (preprint), [doi 10.1101/312280](https://www.biorxiv.org/content/10.1101/312280v1)
- Yasui, Sakai, Kano, Owaki & Ishiguro, Decentralized control scheme for myriapod robot inspired by adaptive and resilient centipede locomotion, PLOS ONE 2017, [PMC5289600](https://pmc.ncbi.nlm.nih.gov/articles/PMC5289600)
- Yasui, Takano, Kano & Ishiguro, Adaptive centipede walking via synergetic coupling between decentralized control and flexible body dynamics, Frontiers in Robotics and AI 9, 2022, [doi 10.3389/frobt.2022.797566](https://www.frontiersin.org/articles/10.3389/frobt.2022.797566/full)
- Aoi, Egi & Tsuchiya, Instability-based mechanism for body undulations in centipede locomotion, Physical Review E 87, 2013, [APS](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.012717)
- Schilling & Cruse, Decentralized control of insect walking: a simple neural network explains a wide range of behavioral and neurophysiological results, PLOS Computational Biology 2020, [PMC7205325](https://pmc.ncbi.nlm.nih.gov/articles/PMC7205325)
- Owaki & Ishiguro, A quadruped robot exhibiting spontaneous gait transitions from walking to trotting to galloping, Scientific Reports 7, 2017, [PMC5428244](https://pmc.ncbi.nlm.nih.gov/articles/PMC5428244)
- Ozkan-Aydin, Chong, Aydin & Goldman, A systematic approach to creating terrain-capable hybrid soft/hard myriapod robots, RoboSoft 2020, [Georgia Tech](https://crablab.gatech.edu/pages/publications/pdf/Robosoft2020_Centipede.pdf)
- Diaz, Erickson, Chong, Soto & Goldman, Active and passive mechanics for rugose terrain traversal in centipedes, J. Exp. Biol. 226, 2023, [doi 10.1242/jeb.244688](https://doi.org/10.1242/jeb.244688)
- Pierce et al., Legged locomotion in lattices: centipede traversal of obstacle-rich environments, Annals of the New York Academy of Sciences 1555, 2026, [PMC12831010](https://pmc.ncbi.nlm.nih.gov/articles/PMC12831010)
- Dionne, Giardina & Mahadevan, Embodied intelligence solves the centipede's dilemma, 2026 (preprint), [arXiv 2603.08409](https://arxiv.org/abs/2603.08409)
- Williams & Sigvardt, Intersegmental phase lags in the lamprey spinal cord: experimental confirmation of the existence of a boundary region, Journal of Computational Neuroscience 1, 1994, [doi 10.1007/BF00962718](https://link.springer.com/article/10.1007/BF00962718)
- Ashwin, Coombes & Nicks, Mathematical frameworks for oscillatory network dynamics in neuroscience, Journal of Mathematical Neuroscience 6, 2016, [arXiv 1506.05828](https://arxiv.org/abs/1506.05828)
- Chen, Wang & Revzen, Walking on rough terrain with any number of legs, 2026 (preprint), [arXiv 2603.09147](https://arxiv.org/abs/2603.09147)
