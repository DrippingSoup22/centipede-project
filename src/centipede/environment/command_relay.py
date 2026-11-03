"""The speed passed back along the body (docs/follow-footsteps/task.md,
prints, marks and the speed): every walker holds the command its leader
held one step earlier, unit 1 its leader's, unchanged, so the command
reaches the rear 0.2 s after it left the head. The environment steps it
with the leader's command; the reward and the observations read it."""

import torch


class CommandRelay:
    """Each unit's held command, ``held`` ``(W, N)`` mm/s, and the same
    averaged over about 1 s at ``averaging_rate`` a step, ``averaged``
    ``(W, N)``. Each walker's average is compared only with its own speed,
    so the delays do not add up into a slower command."""

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        averaging_rate: float,
        device: torch.device | str,
    ) -> None:
        self.averaging_rate = averaging_rate
        self.held = torch.zeros(world_count, segment_count, device=device)
        self.averaged = torch.zeros_like(self.held)
        # The leader's command on the last step, which unit 1 takes next.
        self.leader_command = torch.zeros(world_count, device=device)

    def restart(
        self,
        world_mask: torch.Tensor,
        command: torch.Tensor,
        averaged_start: torch.Tensor,
    ) -> None:
        """Every unit of the ``(W,)`` masked worlds holds the leader's
        ``command`` ``(W,)``, its average starting at ``averaged_start``
        ``(W,)``."""
        self.held[world_mask] = command[world_mask, None]
        self.leader_command[world_mask] = command[world_mask]
        self.averaged[world_mask] = averaged_start[world_mask, None]

    def step(self, command: torch.Tensor) -> None:
        """Pass every held command one unit back, unit 1 taking the leader's
        last, and remember the leader's ``command`` ``(W,)`` on this step."""
        self.held.copy_(torch.cat((self.leader_command[:, None], self.held[:, :-1]), 1))
        self.leader_command.copy_(command)
        self.averaged += self.averaging_rate * (self.held - self.averaged)
