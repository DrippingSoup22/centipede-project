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
batches. Too much reuse of one batch hurts, which they relate to every
agent learning while the others change.

The GPU locomotion studies, per update (samples per world × worlds):

| Study | Batch | Minibatch | Epochs | Adam steps | Learning rate |
| --- | ---: | ---: | ---: | ---: | --- |
| Rudin et al., ANYmal, 50 Hz, 20 s episodes | 24 × 4,096 = 98,304 | 24,576 | 5 | 20 | Adapted to a KL of 0.01 |
| Isaac Gym, Humanoid | 32 × 4,096 = 131,072 | 32,768 | 5 | 20 | Adapted to 0.008 |
| Isaac Gym, ANYmal | 16 × 8,192 = 131,072 | 32,768 | 5 | 20 | Adapted to 0.01 |
| Isaac Gym, ANYmal on rough terrain | 24 × 4,096 = 98,304 | 24,576 | 10 | 40 | Adapted to 0.01 |
| Isaac Gym, Ant | 16 × 4,096 = 65,536 | 32,768 | 4 | 8 | Adapted to 0.008 |

Rudin et al. found that learning struggles with fewer than 25 steps per
robot and an update, that too many robots with too few steps each drop
sharply, and the best trade-off at 2,048 to 4,096 robots with batches of
about 100k to 200k. Andrychowicz et al. (ICLR 2021): go over each batch
several times; 3e-4 is a safe Adam learning rate; larger minibatches did not
hurt; shorter stretches of experience per environment, bootstrapped
earlier, can. Isaac Gym's runs trained ANYmal for 65 M steps, ANYmal on
rough terrain for 150 M, the humanoid for 327 M.

## Exploration

How a policy's noise is drawn, for the decision of cycle 8
([actions.md](actions.md#how-actions-explore)):

| Noise | What it is | Found |
| --- | --- | --- |
| White | Fresh at every step | The usual default; the body's inertia filters most of it out |
| Colored, 1/f^β | Drawn through time, drifting more as β grows: pink β = 1, red β = 2 | Pink best overall with MPO and SAC (Eberhard et al., ICLR 2023); with PPO, β about 0.5 (Hollenstein et al., AAAI 2024) |
| State-dependent (gSDE) | The policy's last-layer features times a random matrix held for some steps: the same state gets the same noise | Smooth enough to train real robots directly, and competitive with unstructured noise in simulation (Raffin, Kober and Stulp, CoRL 2021); in Stable-Baselines3's PPO (`use_sde`, a new matrix at each rollout by default, tanh features by default) |

Noise drawn through time is independent of the state, so a step it makes
cannot be copied by the mean action, which learns the average good action
of each state; state-dependent noise tries only what a policy can do.

## Standing still as a trap

Why the mean action stood through cycles 7 to 9, and what the literature
offers against it, for the decision of cycle 10 ([tow.md](tow.md)):

| Mechanism | Source | Verdict | Why |
| --- | --- | --- | --- |
| Relative overgeneralization | Wei & Luke, JMLR 2016 | The diagnosis | Independent learners judge their own cooperative action against their partners' random ones; when a step pays only if many step together, standing looks best to each |
| Assistance that fades | Yu, Turk & Liu, SIGGRAPH 2018; Cao et al., 2025 (preprint) | Kept: the tow | A helping force lets a few steps pay at the start and is taken away as the policy learns |
| A curriculum moved by success | Rudin et al., CoRL 2021 | Kept | The tow falls only while the mean action walks |
| Paying each car its own error | Jackiewicz, Railway Engineering Science 2021 | Kept: the own ground force | In trains with a motor in every car, each car's traction set from its own coupler's state brings the coupler forces near zero |
| Costs that grow from a fraction | Hwangbo et al., Science Robotics 2019 | Next variant | Their stated reason: a high penalty on motion makes standing a good local minimum; every cost but the objective's starts at 0.3. With each critic's advantages normalised alone, it must scale the health weight and the followers' own costs |
| Optimistic advantages | Zhao et al., ICML 2024 | Later | Negative advantages clipped, so partners' failures stop teaching an agent that its step is bad; a change in RL_lib |
| Energy shapes a gait once the body moves | Fu et al., CoRL 2021 | Kept | Their robot is given a speed to hold, so energy decides how to move, never whether |

## Shared rewards

Why each segment's own reward kept the body still, and how cooperative
learning shares a reward, for the decision of cycle 13
([reward.md](reward.md#cycle-13-the-shared-reward-the-users-design-2026-10-17)):

| Mechanism | Source | Verdict | Why |
| --- | --- | --- | --- |
| One team reward for every agent | Schilling et al. 2020 (the body's mean velocity, no costs); Multi-Agent MuJoCo; SMAC, Samvelyan et al. 2019 (damage, kills, wins) | Kept for the followers' goal | The gain of a step reaches the whole chain; so must its bill |
| Factoredness and learnability | Agogino & Tumer, JAAMAS 2008 | The measure | A reward aligned with the team's is noisy, one that is clear is selfish; both measured on paired worlds without training |
| Lazy and selfish agents | Mao, Gong & Xiao, 2020; Sunehag et al., VDN, AAMAS 2018 | The risk watched | A global reward breeds lazy agents, a local one selfish agents; mixed rewards did best |
| Own and team reward mixed | OpenAI Five, Berner et al. 2019 (the team's share raised from 0.3 to 1); Durugkar, Liebman & Stone, IJCAI 2020 | A dial kept (`goal_team_share`) | The share can stop short of the whole body if the followers turn lazy |
| Sharing with the neighbours | MA2C, Chu et al. 2019 | Kept for health | Energy, pain and hanging are local; shared, their learnability doubled |
| Learned sharing with the neighbours | LToS, Yi et al., NeurIPS 2022 | Later | Each agent learns how much reward to share; a change in the learning |
| Costs that grow from a fraction | Hwangbo et al., Science Robotics 2019 | Kept for pain | A high cost on motion makes standing a good local minimum |

## Team and own

How cooperative learning gives agents one job together and duties of their
own, for the decision of cycle 14
([reward.md](reward.md#cycle-14-the-teams-and-each-ones-own-the-users-split-2026-10-17)):

| Project | Shared by the team | Each agent's own |
| --- | --- | --- |
| Multi-Agent MuJoCo (Peng et al., NeurIPS 2021) | The single robot's reward, repurposed as the team's: ManyAgent Swimmer Δx/Δt − 0.0001 ‖u‖², every agent's commands in its control cost | Nothing |
| MASQ (2024), a quadruped's legs as agents | One reward, its torque and joint-speed costs summed over the legs; one central critic | Nothing |
| Escort (Sheikh and Bölöni, IJCNN 2020) | Protecting the VIP, learned by a central critic | Keeping its own distance from the VIP, 0 or −1, learned by a local critic; the two critics' gradients added, steadier and better than one critic of the mixed reward |
| IRAT (Wang et al., ICML 2022) | A sparse team reward | A dense individual reward, learned by a second policy kept close to the team's; it helped even where the individual reward conflicted |
| Mixing preferences and the task (Durugkar, Liebman and Stone, IJCAI 2020) | The task | A preference of its own, mixed in linearly; some mixes learned better than the task alone |
| Quadrupeds carrying a box (arXiv 2509.14342) | The box's height, speed and tilt | Its own place on the box, its torque and its joint motion |
| Ants carrying food (Gelblum et al., Nature Communications 2015) | The load's motion | Each ant feels the group's force through the load and aligns with it; neither conformity nor individuality alone, but a balance, steered best |
| Force-ANTS (Wang and Schwager, IJRR 2016) | The object | No messages: each robot reads the object's motion at its grip and aligns its force with it |

The pattern: the task's gain is shared, effort is often shared with it,
and each agent's duties stay its own, often judged by a critic of their
own; and agents that share a load coordinate through the force each feels.

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
- Raffin, Kober & Stulp, Smooth exploration for robotic reinforcement learning, CoRL 2021 (PMLR 164), [arXiv 2005.05719](https://arxiv.org/abs/2005.05719)
- Eberhard, Hollenstein, Pinneri & Martius, Pink noise is all you need, ICLR 2023, [ICLR](https://iclr.cc/virtual/2023/oral/12540)
- Hollenstein, Martius & Piater, Colored noise in PPO, AAAI 2024, [arXiv 2312.11091](https://arxiv.org/abs/2312.11091)
- Stable-Baselines3, PPO, [documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
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
- Dallmann, Dürr & Schmitz, Proc. R. Soc. B 283, 2016, [PMC4795010](https://pmc.ncbi.nlm.nih.gov/articles/PMC4795010)
- Guschlbauer, dissertation, University of Cologne, 2009, [KUPS](https://kups.ub.uni-koeln.de/3001/)
- Florensa, Held, Geng & Abbeel, Automatic goal generation for reinforcement learning agents, ICML 2018, [arXiv 1705.06366](https://arxiv.org/abs/1705.06366)
- Pardo, Tavakoli, Levdik & Kormushev, Time limits in reinforcement learning, ICML 2018, [arXiv 1712.00378](https://arxiv.org/abs/1712.00378)
- Rudin, Hoeller, Reist & Hutter, Learning to walk in minutes using massively parallel deep reinforcement learning, CoRL 2021, [arXiv 2109.11978](https://arxiv.org/abs/2109.11978)
- Diaz et al., Active and passive mechanics for rugose terrain traversal in centipedes, J. Exp. Biol. 226, 2023, [JEB](https://cob.silverchair.com/jeb/article/226/4/jeb244688/287689/Active-and-passive-mechanics-for-rugose-terrain)
- legged_gym (Rudin et al.'s code): 20 s episodes, 50 Hz control, discount 0.99, random first episode lengths, [GitHub](https://github.com/leggedrobotics/legged_gym)
- Makoviychuk et al., Isaac Gym, NeurIPS Datasets and Benchmarks 2021, [arXiv 2108.10470](https://arxiv.org/abs/2108.10470), Table 17
- Andrychowicz et al., What matters in on-policy reinforcement learning? A large-scale empirical study, ICLR 2021, [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- Anderson, Shultz & Jayne, Axial kinematics and muscle activity during terrestrial locomotion of the centipede Scolopendra heros, J. Exp. Biol. 198, 1995, [PDF](https://jwshultz.weebly.com/uploads/4/6/2/2/46222147/j_exp_biol-1995-anderson-1185-95.pdf)
- Dallmann, Hoinville, Dürr & Schmitz, A load-based mechanism for inter-leg coordination in insects, Proc. R. Soc. B 284, 2017, [PUB Bielefeld](https://pub.uni-bielefeld.de/record/2915640)
- Wei & Luke, Lenient learning in independent-learner stochastic cooperative games, JMLR 17, 2016
- Yu, Turk & Liu, Learning symmetric and low-energy locomotion, ACM Transactions on Graphics 37(4), SIGGRAPH 2018, [arXiv 1801.08093](https://arxiv.org/abs/1801.08093)
- Cao, Zhang, Nie, Lin, Li & Gao, Learning motion skills with adaptive assistive curriculum force in humanoid robots, 2025 (preprint), [arXiv 2506.23125](https://arxiv.org/abs/2506.23125)
- Jackiewicz, Coupler force reduction method for multiple-unit trains using a new hierarchical control system, Railway Engineering Science 29, 2021, [doi 10.1007/s40534-021-00239-w](https://doi.org/10.1007/s40534-021-00239-w)
- Hwangbo et al., Learning agile and dynamic motor skills for legged robots, Science Robotics 4, 2019, [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)
- Zhao, Zhao, Li, Kannala & Pajarinen, Optimistic multi-agent policy gradient, ICML 2024, [arXiv 2311.01953](https://arxiv.org/abs/2311.01953)
- Agogino & Tumer, Analyzing and visualizing multiagent rewards in dynamic and stochastic domains, JAAMAS 17(2), 2008, [Oregon State](https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_jaamas08.html)
- Mao, Gong & Xiao, Reward design in cooperative multi-agent reinforcement learning for packet routing, 2020, [arXiv 2003.03433](https://arxiv.org/abs/2003.03433)
- Sunehag et al., Value-decomposition networks for cooperative multi-agent learning, AAMAS 2018, [arXiv 1706.05296](https://arxiv.org/abs/1706.05296)
- Berner et al., Dota 2 with large scale deep reinforcement learning (OpenAI Five), 2019, [arXiv 1912.06680](https://arxiv.org/abs/1912.06680)
- Durugkar, Liebman & Stone, Balancing individual preferences and shared objectives in multiagent reinforcement learning, IJCAI 2020, [IJCAI](https://www.ijcai.org/proceedings/2020/347)
- Samvelyan et al., The StarCraft multi-agent challenge, AAMAS 2019, [arXiv 1902.04043](https://arxiv.org/abs/1902.04043)
- Multi-Agent MuJoCo (MaMuJoCo), Farama Foundation, [documentation](https://robotics.farama.org/envs/MaMuJoCo)
- Yi, Li, Wang & Lu, Learning to share in networked multi-agent reinforcement learning (LToS), NeurIPS 2022, [arXiv 2112.08702](https://arxiv.org/abs/2112.08702)
- Sheikh & Bölöni, Multi-agent reinforcement learning for problems with combined individual and team reward, IJCNN 2020, [arXiv 2003.10598](https://arxiv.org/abs/2003.10598)
- Wang, Zhang, Hu, Wang, Zhang, Gao, Hao, Lv & Fan, Individual reward assisted multi-agent reinforcement learning (IRAT), ICML 2022, [PMLR 162](https://proceedings.mlr.press/v162/wang22ao.html)
- Multi-quadruped cooperative object transport: learning decentralized pinch-lift-move, 2025, [arXiv 2509.14342](https://arxiv.org/abs/2509.14342)
- Liu, Guo, Lin, Ma, Zhu & Li, MASQ: multi-agent reinforcement learning for single quadruped robot locomotion, 2024, [arXiv 2408.13759](https://arxiv.org/abs/2408.13759)
- Gelblum, Pinkoviezky, Fonio, Ghosh, Gov & Feinerman, Ant groups optimally amplify the effect of transiently informed individuals, Nature Communications 6, 2015, [PMC4525283](https://pmc.ncbi.nlm.nih.gov/articles/PMC4525283/)
- Wang & Schwager, Force-amplifying N-robot transport system (Force-ANTS) for cooperative planar manipulation without communication, IJRR 35(13), 2016, [Stanford MSL](https://msl.stanford.edu/bibliography/wang_force-amplifying_2016.html)
- Gymnasium, HalfCheetah: reward = forward reward − 0.1 ‖a‖², [documentation](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
