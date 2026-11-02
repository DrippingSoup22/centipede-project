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

## Settings

As follow-line's: networks of 2 × 64; each critic's returns standardised
over about the last 10 updates; each input's statistics over about the last
10 update cycles; gSDE from a spread of 0.3, its matrix held for 4
decisions, every spread kept above the floor; 4,096 worlds × 32 steps an
update, 4 minibatches, 5 passes, the actor's learning rate adapted toward a
KL of 0.01; discount 0.99, GAE λ 0.95.

## Sources

As in [follow-line's critic](../follow-line/critic.md).
