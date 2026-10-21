# Constraints

The project shows that a centipede gait can emerge from many agents, each
with an incomplete view of the whole body. These constraints define it (the
user's decisions, 2026-10-21). They are never relaxed and admit no
exceptions; every design keeps them.

1. **One agent per unit.** Each agent's actions command only its own unit's
   motors.
2. **Each agent knows only what lies within its radius.** Its actor and its
   critic receive only information from inside its unit's observation
   radius: its own body, the units within the radius, and the world within
   it. Nothing from outside the radius reaches either of them, in acting or
   in training: no critic of the whole body, no global state.
3. **The radius is local.** Its size is a design choice; today a unit sees
   itself and the two units ahead and behind ([observations.md](observations.md)).
4. **Nothing is shared between units.** Each agent has its own actor,
   critic, optimisers, normaliser and experience: no shared networks,
   pooled samples, gradients or losses.

Later, if the target task returns: only the head sees the target.

Everything else is a design choice, the reward first among them: it may
combine team and own terms.
