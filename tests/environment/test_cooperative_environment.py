"""Tests for the cooperators' environment and its task, observations, reward
and diagnostics.

The parts are fed hand-made physical states; one test runs the whole
environment on the CPU.
"""

import math
from pathlib import Path

import pytest
import torch

from centipede.diagnostics_category import WindowSummary
from centipede.environment.cooperative_diagnostics import CooperativeDiagnostics
from centipede.environment.cooperative_observations import (
    CooperativeObservationBuilder,
)
from centipede.environment.cooperative_reward import (
    PART_NAMES,
    CooperativeReward,
    energy_sharing_weights,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.environment import create_environment
from centipede.environment.reward_function import StepRewards
from centipede.environment.settings import CooperatorsSettings, EnvironmentSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.model_mapping import HeadOutline

SETTINGS = CooperatorsSettings.from_section({})
BODY = BodyMeasures(
    unit_weight_n=1e-3, leg_push_n=2e-3, body_length_m=0.05, turning_circle_m=0.03
)
SEGMENTS, LEGS = 5, 8


def resting_state(world_count: int) -> PhysicalState:
    """Upright segments facing +x, a centimetre apart, the head's tip at 5 mm."""
    state = PhysicalState.allocate(world_count, SEGMENTS, LEGS, "cpu")
    state.body_quaternion[..., 0] = 1.0
    state.body_planar_position[..., 0] = -0.01 * torch.arange(SEGMENTS)
    state.head_tip_position[:, 0] = 0.005
    return state


def test_targets_follow_the_level_and_the_stopwatch_moves_it():
    task = CooperativeTask(SETTINGS, BODY, 4096, episode_steps=400, device="cpu")
    state = resting_state(4096)
    every_world = torch.ones(4096, dtype=torch.bool)

    # The circle opens where the near end clears a body length and a turn.
    final_near = 2.7 * BODY.body_length_m
    assert task.full_circle_level == pytest.approx((0.08 - 0.02) / (final_near - 0.02))
    for level, near, half_bearing in ((0.0, 0.02, 45.0), (1.0, final_near, 180.0)):
        task.level = level
        task.place_targets(every_world, state)
        offset = task.target_position - state.head_tip_position[:, :2]
        distance = offset.norm(dim=-1)
        bearing = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0])).abs()
        assert distance.min() >= near and distance.max() <= 1.5 * near
        assert bearing.max() <= half_bearing + 1e-3
        assert bearing.max() > 0.95 * half_bearing

    # Over one episode length, a quarter of the targets reached within their
    # stopwatch of 350 steps: the level rises by 0.02 x (0.25 - 0.10).
    task.level = 0.5
    task.place_targets(every_world, state)
    reached = torch.arange(4096) < 1024
    for step in range(400):
        arrived = reached & (step == 100)
        task.time_step(arrived)
        task.place_targets(arrived, state)  # as the environment walks on
    task.adjust_level()
    assert task.in_time_share == pytest.approx(0.25)
    assert task.level == pytest.approx(0.5 + 0.02 * 0.15)


def test_energy_is_shared_with_the_neighbours_seen_and_pain_stays_with_the_pressed():
    weights = energy_sharing_weights(SEGMENTS, observation_radius=2, decay=0.5)
    torch.testing.assert_close(weights[2], torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1]))
    torch.testing.assert_close(weights[0], torch.tensor([4.0, 2.0, 1.0, 0, 0]) / 7)

    commanded = torch.ones(SEGMENTS, LEGS + 1)
    commanded[-1, -1] = 0.0
    reward = CooperativeReward(SETTINGS, SEGMENTS, 2, BODY, commanded)
    state = resting_state(1)
    state.motor_positive_work[0, 2] = 1e-6  # one microjoule pushing
    state.leg_leg_force_squared[0, 3] = BODY.leg_push_n**2  # a full push
    state.head_tip_position[0, 0] = 0.007  # 2 mm toward the target
    action = torch.zeros(1, SEGMENTS, LEGS + 1)
    action[0, 4, -1] = 1.0  # the rear's padding costs nothing
    previous = state.body_planar_position.clone()

    channels, parts, _ = reward.compute(
        state,
        previous,
        torch.tensor([[0.005, 0.0]]),
        torch.tensor([[0.007, 0.0]]),
        torch.tensor([True]),
        0.025,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        action,
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(by_name["progress"][0], torch.tensor(2.0))
    torch.testing.assert_close(by_name["arrival"], torch.tensor([25.0, 0, 0, 0, 0]))
    torch.testing.assert_close(by_name["pushing"], -1e-3 * weights[:, 2])
    assert not by_name["force"].any()
    # A full push for one step, priced as one motor fully on: force price x 20 ms.
    pain = -SETTINGS.force_price_w * 0.02 * 1e3
    torch.testing.assert_close(by_name["pain"], torch.tensor([0, 0, 0, pain, 0]))
    torch.testing.assert_close(
        channels[0, :, 0], by_name["progress"] + by_name["arrival"]
    )


def test_a_block_holds_what_the_segment_senses_and_only_the_head_sees_the_target():
    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, "cpu"
    )
    block = builder.block_size
    assert (block, builder.observation_size) == (42, 5 * 42 + 2)
    state = resting_state(1)
    state.foot_normal_force[0, 1] = torch.tensor([2e-3, 1e-3])
    last_action = torch.rand(1, SEGMENTS, LEGS + 1)

    observations = builder.build(state, torch.tensor([[0.025, 0.01]]), last_action)

    own = observations[0, 1, :block]
    torch.testing.assert_close(own[:3], torch.tensor([0.0, 0.0, -1.0]))  # gravity
    torch.testing.assert_close(own[27:29], torch.tensor([2.0, 1.0]))  # unit weights
    torch.testing.assert_close(own[-(LEGS + 1) :], last_action[0, 1])
    # The head sees nothing ahead of it, and the target from its tip.
    head = observations[0, 0]
    assert not head[block : 3 * block].any()
    torch.testing.assert_close(head[-2:], torch.tensor([0.02, 0.01]))
    assert not observations[0, 1:, -2:].any()


def test_the_gait_is_read_from_the_feet_and_each_episode_is_summarised():
    """Every foot stands four steps of eight, each segment's a step behind the
    one ahead and each right foot half a cycle behind its left: a wave from
    head to tail of 45 degrees per segment, with the sides alternating."""
    diagnostics = CooperativeDiagnostics(
        2, SEGMENTS, BODY.unit_weight_n, HeadOutline(-0.002, 0.002, 0.002), "cpu", None
    )
    window = WindowSummary(diagnostics.step)
    state = resting_state(2)
    target = torch.tensor([[0.1, 0.0], [0.1, 0.0]])
    every_world = torch.ones(2, dtype=torch.bool)
    diagnostics.new_targets(every_world, every_world, state, target)
    zeros = torch.zeros(2, SEGMENTS)
    step_rewards = StepRewards(zeros, torch.zeros(2, SEGMENTS, 6), zeros, zeros)
    delay = torch.arange(SEGMENTS)[:, None] + torch.tensor([0, 4])
    for step in range(40):
        standing = (step - delay) % 8 < 4
        state.foot_normal_force.copy_(standing * BODY.unit_weight_n)
        arrived = torch.tensor([step % 10 == 9, False])  # world 0, every 0.2 s
        diagnostics.record_step(
            state,
            state.body_planar_position,
            state.head_tip_position[:, :2],
            target,
            torch.zeros(2, SEGMENTS, 2),
            step_rewards,
            torch.zeros(2, SEGMENTS, 3),
            arrived,
            arrived,
            torch.full((2,), step == 39),  # the time limit
        )
        window.add()
        diagnostics.new_targets(arrived, torch.zeros_like(arrived), state, target)

    gait = window.result()
    torch.testing.assert_close(gait["foot_standing"], torch.full((SEGMENTS, 2), 0.5))
    torch.testing.assert_close(
        gait["neighbour_lag"], torch.full((SEGMENTS - 1, 2), math.pi / 4)
    )
    for name in ("neighbour_lag", "left_right_lag"):
        assert torch.allclose(gait[f"{name}_consistency"], torch.tensor(1.0))
        assert torch.allclose(gait[f"{name}_lock"], torch.tensor(1.0))
    assert torch.allclose(gait["left_right_lag"].cos(), torch.tensor(-1.0))
    episode = diagnostics.episode
    assert episode.episode_ended.all()
    assert episode.targets_reached.tolist() == [4.0, 0.0]
    assert episode.targets_by_bearing[:, 0].tolist() == [4.0, 1.0]  # all ahead
    assert episode.reached_by_bearing[:, 0].tolist() == [4.0, 0.0]


def test_after_an_arrival_the_body_walks_on_and_restarts_only_at_the_time_limit():
    settings = EnvironmentSettings.from_section(
        {
            "design": "cooperators",
            "max_episode_steps": 3,
            "observation_radius": 2,
            "simulation": {
                "model_path": str(Path("models/assembly_v4.xml").resolve()),
                "backend": "cpu",
                "world_count": 2,
            },
            "cooperators": {"stagger_first_episodes": False},
        }
    )
    environment = create_environment(settings)
    environment.reset(seed=0)
    # World 0's target lies under its head.
    state = environment.simulation.physical_state
    environment.task.target_position[0] = state.body_planar_position[0, 0]
    action = torch.zeros(2, 12, 9)

    observations, rewards, terminated, truncated, final = environment.step(action)

    assert rewards.shape == (2, 12, 2)
    assert not terminated.any() and not truncated.any()
    assert rewards[0, 0, 0] > 1.0 and environment.task.target_steps.tolist() == [0, 1]
    assert observations is final
    distance = environment.task.target_position[0] - state.head_tip_position[0, :2]
    assert distance.norm() >= 0.02

    environment.step(action)
    _, _, terminated, truncated, final = environment.step(action)
    assert truncated.all() and not terminated.any()
    assert environment.episode_steps.tolist() == [0, 0]
    assert not torch.equal(final, environment._observations())
