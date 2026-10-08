"""Tests for the environment's front file, with the real model v3.

The parts have their own tests; these cover only what the front adds: episode
state (targets, step counts, previous positions), episode ends, and resetting
finished worlds. Target positions are read back through the head's two target
values, which give the target in the head's own directions.
"""

import pytest
import torch

from centipede.environment.environment import Environment
from centipede.environment.reward_function import joint_angles
from centipede.environment.settings import EnvironmentSettings
from centipede.settings_section import SettingsError

BACKENDS = ["cpu"] + (["gpu"] if torch.cuda.is_available() else [])


def environment(backend="cpu", world_count=3, **changes) -> Environment:
    simulation = {
        "model_path": "models/assembly_v3.xml",
        "backend": backend,
        "world_count": world_count,
        "gpu_solver": "cg",  # also runs on GPUs older than Volta
    }
    return Environment(
        EnvironmentSettings.from_section({"simulation": simulation} | changes)
    )


def target_distance_and_bearing(observations) -> tuple[torch.Tensor, torch.Tensor]:
    """Each world's target distance (m) and bearing (degrees, left positive)."""
    forward, left = observations[:, 0, -2:].unbind(dim=-1)
    return torch.hypot(forward, left), torch.rad2deg(torch.atan2(left, forward))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_seed_repeats_starts_and_targets_have_their_own_sequence(backend):
    """No physics steps, so it is quick on any GPU."""
    default_targets = environment(backend)
    other_targets = environment(
        backend,
        target={"distance_range_m": [0.01, 0.02], "bearing_range_deg": [20, 30]},
    )

    first = default_targets.reset(seed=3)
    other = other_targets.reset(seed=3)

    # Other target settings change the targets but never the starting poses.
    assert torch.equal(first[..., :-2], other[..., :-2])
    distance, bearing = target_distance_and_bearing(first)
    assert torch.all((distance >= 0.030) & (distance <= 0.060))
    assert torch.all(bearing.abs() <= 30)
    distance, bearing = target_distance_and_bearing(other)
    assert torch.all((distance >= 0.010) & (distance <= 0.020))
    assert torch.all((bearing >= 20) & (bearing <= 30))  # to the head's left

    assert torch.equal(default_targets.reset(seed=3), first)
    assert not torch.equal(default_targets.reset(seed=4)[:, 0, -2:], first[:, 0, -2:])


def test_episodes_end_by_arrival_time_limit_or_range_and_only_those_restart():
    """World 0 arrives on its last allowed step, with its target under the rear
    of the head, 6 mm from the tip; world 1 runs out of time; world 2's head is
    outside its range circle; world 3 is one step into its episode and goes on."""
    env = environment(world_count=4, max_episode_steps=2)
    env.reset(seed=1)
    zero_action = torch.zeros(4, env.segment_count, 6)
    _, _, terminated, truncated, _ = env.step(zero_action)
    assert not torch.any(terminated | truncated)

    state = env.simulation.physical_state
    head_centre, head_tip = state.body_planar_position[:, 0], state.head_tip_position
    behind_centre = head_centre[0] - 0.2 * (head_tip[0, :2] - head_centre[0])
    env.target_position[0] = behind_centre  # the head barely moves in 20 ms
    env.target_position[2] = head_tip[2, :2] + torch.tensor([1.0, 0.0])
    env.episode_steps[2:] = 0
    targets_before = env.target_position.clone()

    observations, rewards, terminated, truncated, final = env.step(zero_action)

    assert terminated.tolist() == [True, False, False, False]  # arrival wins
    assert truncated.tolist() == [False, True, True, False]
    assert torch.all(rewards[0] > 0) and torch.all(rewards[1:] < 0)
    assert env.episode_steps.tolist() == [0, 0, 0, 1]
    target_moved = (env.target_position != targets_before).any(dim=-1)
    assert target_moved.tolist() == [True, True, True, False]
    episodes = env.diagnostics.episode
    assert episodes.left_range[:3].tolist() == [False, False, True]
    assert episodes.ending[[0, 2]].tolist() == [0.5, 5.5]  # arrived, left the circle

    # Finished worlds return the observation their episode ended with: world 0's
    # target was 6 mm from the tip, and its new one is at least 30 mm away.
    restarted = (observations != final).flatten(start_dim=1).any(dim=-1)
    assert restarted.tolist() == [True, True, True, False]
    assert 0.005 < target_distance_and_bearing(final)[0][0] < 0.007
    assert target_distance_and_bearing(observations)[0][0] >= 0.030

    # Runs before the head arrival needed the tip within 1 mm, and had no circle.
    tip_env = environment(world_count=1, target={"arrival_radius_m": 0.001})
    tip_env.reset(seed=1)
    tip_env.target_position[0] = tip_env.simulation.physical_state.body_planar_position[
        0, 0
    ]
    _, _, terminated, truncated, _ = tip_env.step(zero_action[:1])
    assert not terminated.item() and not truncated.item()


def test_progress_is_measured_from_the_positions_before_the_step():
    """The physical state is overwritten by the step, so the front must have
    copied the earlier positions; otherwise every movement would measure zero."""
    env = environment(world_count=2)
    env.reset(seed=2)
    state = env.simulation.physical_state
    head_tip_before = state.head_tip_position[:, :2].clone()
    centres_before = state.body_planar_position.clone()

    env.step(torch.ones(2, env.segment_count, 6))

    target = env.target_position
    head_progress = (target - head_tip_before).norm(dim=-1) - (
        target - state.head_tip_position[:, :2]
    ).norm(dim=-1)
    moved = (state.body_planar_position - centres_before).norm(dim=-1)
    facts = env.diagnostics.step
    assert torch.all(moved > 0)
    assert torch.allclose(facts.segment_progress[:, 0], head_progress, atol=1e-9)
    assert torch.allclose(facts.segment_moved, moved, atol=1e-9)


def test_with_spine_control_each_segment_but_the_rear_bends_the_joint_behind_it():
    env = environment(
        world_count=2, spine_control=True, rewards={"movement_cost_parts": 1}
    )
    assert env.segment_action_sizes == [7] * 7 + [6] and env.action_size == 7
    assert env.observation_size == environment().observation_size + 2
    env.reset(seed=2)
    state = env.simulation.physical_state
    action = torch.zeros(2, env.segment_count, 7)
    action[0, :, 6] = 1.0  # world 0 bends every joint; the rear's 7th is padding

    for _ in range(5):
        angles_before = joint_angles(state).clone()
        observations, *_ = env.step(action)

    bend = state.spine_yaw_position
    assert torch.all(bend[0, :-1] > bend[1, :-1].abs())
    assert not bend[:, -1].any()  # the rear has no joint behind it
    assert torch.equal(observations[..., -2], bend)  # each sees its own, last
    # The movement is measured from the angles before the step, over the joints
    # each segment commands.
    squared = (joint_angles(state) - angles_before).square()
    expected = torch.cat(
        (squared[:, :-1].mean(dim=-1), squared[:, -1:, :6].mean(dim=-1)), dim=1
    )
    assert torch.allclose(env.diagnostics.step.joint_movement, expected.sqrt())


def test_an_observation_radius_reaching_past_the_body_is_rejected():
    with pytest.raises(SettingsError, match=r"observation_radius .* 8 segments"):
        environment(observation_radius=8)
