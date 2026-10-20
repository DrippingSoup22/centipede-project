# Critic and learning

Agreed with the user on 2026-10-20. The cooperators' learning
(`docs/cooperators/critic.md` on that branch), without the baselines.

## What a critic does

A critic predicts how much reward usually follows the situation a unit is
in. After an action, what really followed minus that prediction is the
advantage: positive, the action did better than usual and becomes more
likely. What really followed holds the unit's own action, its neighbours'
and luck; the neighbours' share is noise to the unit, which averages out
over many samples (131,072 per agent per update) but slows learning.

## Two critics per agent

| Critic | Predicts | Sees |
| --- | --- | --- |
| Team | The team account: the line, of which the joints' accelerations take a share (from run 08; subtracted in runs 06 and 07) | What its actor sees |
| Own | The own account: its carry (from run 06) | What its actor sees, its feet's averaged load included |

- **One critic per account**, so that the clearer own signal is not drowned
  in the team's, and each weighs what its weight says whatever its size:
  RL_lib normalises each critic's advantages over the batch on their own,
  then adds them 1 : 0.8, the team first (Sheikh & Bölöni 2020;
  cooperators, cycle 14). Runs 01 to 04 had these two critics, the own
  account holding the force share and hanging.
- **One critic when there is one account.** Run 05's minimal reward had no
  own term, so its agents had one critic (`own_account` false): an account
  always 0 would give its critic nothing but noise to learn, normalised up
  to the team's size (the user's decision, 2026-10-20).
- **A local critic.** It sees its actor's view and nothing else: a critic of
  the whole body would make the cooperation come from a central judge, and
  does not make the agents cooperate better (Lyu et al. 2021).
- **No baselines.** The cooperators gave each critic a second network that
  also saw the neighbours' actions of the step. In runs 24 to 27 (eight
  runs) it explained the returns exactly as well as the critic, within
  0.01: the advantage looks about 19 steps ahead, and one step of the
  neighbours' actions is a sliver of their effect (as Tucker et al. 2018
  found for such baselines).

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Networks | Actor and each critic 2 × 64 | A local view of five units (the hexapod's leg networks, Schilling et al. 2020) |
| Value normalisation | Each critic learns its returns standardised by a mean and spread over about the last 10 updates | The units of a reward never set the scale; returns grow as the walk improves, and a scale over the whole run lagged them (run 25) |
| Exploration | gSDE: the noise is the actor's last hidden layer, at unit length, times a matrix each world draws and keeps for 4 decisions; spreads learned from 0.4; a tanh actor | Noise drawn through time was the gait and the noise-free policy stood (runs 13 to 18); from run 23 the walk moved into the policy (Raffin et al. 2021) |
| Worlds × steps | 4,096 × 32 per update | legged_gym's batch (Rudin et al. 2021) |
| Minibatches, passes | 4 of 32,768, 5 passes | 20 Adam steps an update, as theirs |
| Learning rate | Each actor's adapted after every update toward a KL of 0.01 (×1.5 or ÷1.5, between 1e-5 and 1e-2), from 3e-4; the critics 3e-4 | Rudin et al.'s rule: steady steps as the spread narrows |
| Discount, GAE λ | 0.998, 0.95 | Run 28's |
