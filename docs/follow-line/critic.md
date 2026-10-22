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
| Team | The team account ([reward.md](reward.md)) | What its actor sees, the forward lines and energies averaged over 1 s included |
| Own | The own account, its carry | What its actor sees, its feet's averaged load included |

- **One critic per account**, so that the clearer own signal is not drowned
  in the team's: RL_lib normalises each critic's advantages over the batch
  on their own, then adds them **1 : 1** (Sheikh & Bölöni 2020). Once
  normalised, the weight is the only thing that sets how much each account
  counts, and nothing measured favours either (runs 06 to 11 used 1 : 0.8,
  a choice never tested). With no own term, a run has one critic
  (`own_account` false): an account always 0 would give its critic only
  noise.
- **Local**: it sees its actor's view and nothing else
  ([constraints.md](constraints.md)). After convergence central and local
  critics give the same expected gradient (Lyu et al. 2021). When the actor
  leaves out the units behind (from run 16, [observations.md](observations.md)),
  the critics still see all five units: the team account charges each unit
  their energy, and a critic blind to part of its pay would judge every
  action against noise. A critic that sees more than its actor is the usual
  pattern of multi-agent learning (centralised training, decentralised
  execution); here it still sees only the neighbourhood.
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
| Exploration | gSDE: the noise is the actor's last hidden layer, at unit length, times a matrix each world draws and keeps for 4 decisions; spreads learned from 0.3; a tanh actor | With noise drawn through time, the gait was the noise and the noise-free policy stood (cooperators, runs 13 to 18) (Raffin et al. 2021) |
| The noise's start, 0.3 | Fresh networks under the tow: 8.9 mm/s at 11.9 mJ/s a unit, against 9.5 at 15.8 for 0.4 and 7.3 at 7.7 for 0.2 | Almost as much exploring as 0.4 for a quarter less energy; at 0.2 the body moves noticeably less ([reward.md](reward.md), c) |
| The matrix's hold, 4 | Holds of 1 to 32 decisions: the commands change at 12 to 6 Hz, the feet step 2.3 to 2.8 times a second, the start's margin over standing is the same from 1 to 16 and smaller at 32 | The noise follows the unit's quick senses whatever the hold, so a longer one gains nothing here; in Raffin et al.'s PPO test (Walker2D, 60 decisions a second) 4 was best and long holds explored worse. The study's archive, `checks/reward_12/hold_v6.txt` |
| The noise front (runs 18 to 20 only; off since) | Each unit's spreads stay at least 0.3 until a front reaches it; the front starts at the walk's gate, with the price, and lets one more unit go from the head every 30 cycles; a unit let go learns its spreads as before | In runs 16 and 17 the pairs stayed at chance while the noise was 0.25 to 0.30 and ordered as it fell; each unit joining an ordered piece stepped after it, so every first-ordered place became a source. Cooling from the head lets order start there alone and spread backward, as a front slower than order spreads leaves almost no defects (Dziarmaga, Laguna & Zurek 1999). 30 cycles a unit is slower than the 15 to 20 order took ([research.md](research.md), section 4). **It failed**: pairs ordered at 0.30 behind the front, seed 2's at the same cycle as without one; order starts as the walk is learned, whatever the noise (0.22 to 0.30) or the price (0.4 to 1.6) in runs 12 to 20 |
| Worlds × steps | 4,096 × 32 per update | legged_gym's batch (Rudin et al. 2021) |
| Minibatches, passes | 4 of 32,768, 5 passes | 20 Adam steps an update, as theirs |
| Learning rate | Each actor's adapted after every update toward a KL of 0.01 (×1.5 or ÷1.5, between 1e-5 and 1e-2), from 3e-4; the critics 3e-4 | Rudin et al.'s rule |
| Discount, GAE λ | 0.99, 0.95 | A horizon of about 2 s, as legged_gym's at the same 50 decisions a second (Rudin et al. 2021): what a step changes, the 1-s averages included, is paid within a stride or two and that second. 0.998, about 10 s, came from the cooperators' target several seconds away; here it would add mostly the other units' changing behaviour, noise to a unit's critic |
