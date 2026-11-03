# Critic and learning

follow-line's learning ([critic.md](../follow-line/critic.md)), unchanged
but for what the critics see.

## Two critics per agent

| Critic | Predicts | Sees |
| --- | --- | --- |
| Team | The team account: the speed against the command, less the energy of the unit and its leader ([reward.md](reward.md)) | What its actor sees, 65 values |
| Own | The own account: the carry and the marks | What its actor sees, 65 values |

- **One critic per account**, each critic's advantages normalised over the
  batch on their own and added **1 : 1**, as in follow-line (Sheikh &
  Bölöni 2020).
- **The same view as the actor.** Everything a walker is charged for comes
  from itself and its leader, and it sees both, so its critic needs nothing
  more. In follow-line the critics saw the units behind, whose distance from
  the line each unit was charged for; here nothing behind is charged.
- **Local** in the constraints' sense: an agent's actor and critic receive
  only what its radius holds, its leader one hop away.

## The head (stage B)

One critic for its one account ([reward.md](reward.md#the-head-stage-b)),
seeing what its actor sees ([observations.md](observations.md#the-head-stage-b)),
with the walkers' settings.

**Only the values it used are judged** (agreed with the user, 2026-11-03).
The head chooses its footholds at every step, but each is read only at its
side's touchdown ([actions.md](actions.md#the-head-stage-b)). At every
other step a foothold value changed nothing, yet PPO would still make it
more or less likely by how the step went, which adds noise to the head's
learning and can trip its clipping. So the environment marks which values
it read at each step (the speed always, a side's foothold at that side's
touchdown), and the head's update counts only those in each step's
probability. No step is dropped and nothing it sees changes. This needs a
change in RL_lib's `follow-footsteps` branch, whose policies add up every
value's log-probability; its scope is agreed before stage B.

## Settings

As follow-line's: networks of 2 × 64; each critic's returns standardised
over about the last 10 updates; each input's statistics over about the last
10 update cycles; gSDE from a spread of 0.3, its matrix held for 4
decisions, every spread kept above the floor; 4,096 worlds × 32 steps an
update, 4 minibatches, 5 passes, the actor's learning rate adapted toward a
KL of 0.01; discount 0.99, GAE λ 0.95.

## Sources

As in [follow-line's critic](../follow-line/critic.md).
