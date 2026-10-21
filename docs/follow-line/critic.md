# Critic and learning

The cooperators' learning (`docs/cooperators/critic.md` on that branch),
without its baselines.

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
| Team | The team account ([reward.md](reward.md)) | What its actor sees, its own energy averaged over 1 s included |
| Own | The own account, its carry | What its actor sees, its feet's averaged load included |

- **One critic per account**, so that the clearer own signal is not drowned
  in the team's: RL_lib normalises each critic's advantages over the batch
  on their own, then adds them 1 : 0.8, the team first (Sheikh & Bölöni
  2020). With no own term, a run has one critic (`own_account` false): an
  account always 0 would give its critic only noise.
- **Local**: it sees its actor's view and nothing else
  ([constraints.md](constraints.md)). After convergence central and local
  critics give the same expected gradient (Lyu et al. 2021).
- **What a reward depends on must be seen** (the pairing rule): the energy
  and the carry are averaged over a second, so each unit observes its own
  averages.
- **No baselines.** In the cooperators' runs 24 to 27 a second network that
  also saw the neighbours' actions of the step explained the returns no
  better, within 0.01: one step of the neighbours' actions is a sliver of
  their effect (as Tucker et al. 2018 found).

**How they saw** (runs 06 to 11). The own critic explains 0.84 to 1.00 of
its returns throughout. The team critic loses sight while what the team
earns changes fast, as the walk forms (0.11 to 0.55, worst at the ends of
the body, whose units see three), and regains it as that settles (0.90 to
0.99).

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Networks | Actor and each critic 2 × 64 | A local view of five units (the hexapod's leg networks, Schilling et al. 2020) |
| Value normalisation | Each critic learns its returns standardised by a mean and spread over about the last 10 updates | Returns grow as the walk improves; a scale over the whole run lagged them |
| Exploration | gSDE: the noise is the actor's last hidden layer, at unit length, times a matrix each world draws and keeps for 4 decisions; spreads learned from 0.4; a tanh actor | With noise drawn through time, the gait was the noise and the noise-free policy stood (cooperators, runs 13 to 18) (Raffin et al. 2021) |
| Worlds × steps | 4,096 × 32 per update | legged_gym's batch (Rudin et al. 2021) |
| Minibatches, passes | 4 of 32,768, 5 passes | 20 Adam steps an update, as theirs |
| Learning rate | Each actor's adapted after every update toward a KL of 0.01 (×1.5 or ÷1.5, between 1e-5 and 1e-2), from 3e-4; the critics 3e-4 | Rudin et al.'s rule |
| Discount, GAE λ | 0.998, 0.95 | The cooperators' run 28 |
