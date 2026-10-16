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
    AveragedForces,
    CooperativeReward,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.cooperative_trail import TrailMemory
from centipede.environment.environment import create_environment
from centipede.environment.reward_function import StepRewards
from centipede.environment.settings import CooperatorsSettings, EnvironmentSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.model_mapping import HeadOutline

SETTINGS = CooperatorsSettings.from_section({})
BODY = BodyMeasures(
    unit_weight_n=1e-3,
    leg_push_n=2e-3,
    body_length_m=0.05,
    turning_circle_m=0.03,
    segment_lengths_m=(0.01,) * 5,
)
SEGMENTS, LEGS = 5, 8


def resting_state(world_count: int) -> PhysicalState:
    """Upright segments facing +x, a centimetre apart, the head's tip at 5 mm
    and each joint halfway between two centres."""
    state = PhysicalState.allocate(world_count, SEGMENTS, LEGS, "cpu")
    state.body_quaternion[..., 0] = 1.0
    state.body_planar_position[..., 0] = -0.01 * torch.arange(SEGMENTS)
    state.head_tip_position[:, 0] = 0.005
    state.front_end_position[..., 0] = 0.005 - 0.01 * torch.arange(SEGMENTS)
    return state


def test_targets_follow_the_level_and_the_stopwatch_moves_it():
    task = CooperativeTask(SETTINGS, BODY, 4096, episode_steps=400, device="cpu")
    state = resting_state(4096)
    every_world = torch.ones(4096, dtype=torch.bool)

    # The circle opens where the near end clears a body length and a turn.
    final_near = 2.7 * BODY.body_length_m
    assert task.full_circle_level == pytest.approx((0.08 - 0.02) / (final_near - 0.02))
    for level, near, half_bearing in ((0.0, 0.02, 90.0), (1.0, final_near, 180.0)):
        task.level = level
        task.place_targets(every_world, state, torch.full((4096,), 400))
        offset = task.target_position - state.head_tip_position[:, :2]
        distance = offset.norm(dim=-1)
        bearing = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0])).abs()
        assert distance.min() >= near and distance.max() <= 1.5 * near
        assert bearing.max() <= half_bearing + 1e-3
        assert bearing.max() > 0.95 * half_bearing

    # Over one episode length, a third of the timed targets reached within
    # their stopwatch of 350 steps: the level rises by 0.02 x (1/3 - 0.10).
    # The last quarter of the worlds restart in 100 steps, too soon for a
    # stopwatch to run out: their targets are not counted.
    task.level = 0.5
    worlds = torch.arange(4096)
    task.place_targets(every_world, state, torch.where(worlds < 3072, 400, 100))
    reached = worlds < 1024
    for step in range(400):
        arrived = reached & (step == 100)
        task.time_step(arrived)
        # As the environment walks on: these targets come too late to count.
        task.place_targets(arrived, state, torch.full((4096,), 399 - step))
    task.adjust_level()
    assert task.in_time_share == pytest.approx(1 / 3)
    assert task.level == pytest.approx(0.5 + 0.02 * (1 / 3 - 0.1))


def at_yaw(state: PhysicalState, degrees: float) -> None:
    """Turn every segment of the state to face ``degrees`` from +x."""
    half = math.radians(degrees) / 2
    state.body_quaternion[:] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


def test_the_goal_pays_the_head_progress_and_the_followers_distance_on_the_trail():
    """The body steps 1 mm forward, the head 1 mm toward its target and onto
    it; trail offsets of 5, 0, 10, 0 and 20 degrees. Averaged forces, in unit
    weights: the head pushed forward by 0.4, held up by 0.8 and on the floor
    with 0.2; segment 1 pushing the head with 0.3; segment 2 held up by 1;
    the rear on the floor with 0.3, pressed down by its neighbour. Energy is
    shared 40% to itself and 30% to each side; pain stays with the pressed."""
    reward = CooperativeReward(
        SETTINGS, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, LEGS + 1)
    )
    torch.testing.assert_close(
        reward.sharing_weights[:3],
        torch.tensor(
            [
                [0.4, 0.4, 0.2, 0.0, 0.0],
                [0.3, 0.4, 0.2, 0.1, 0.0],
                [0.1, 0.2, 0.4, 0.2, 0.1],
            ]
        ),
    )
    state = resting_state(1)
    state.motor_positive_work[0, 2] = 1e-6  # one microjoule pushing
    state.leg_leg_force_squared[0, 3] = BODY.leg_push_n**2  # a full push
    state.net_joint_force[0] = 1e-3 * torch.tensor(
        [[0.4, 0.8], [-0.3, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, -0.4]]
    )
    state.front_joint_force[0] = 1e-3 * torch.tensor([0.0, -0.3, 0.0, 0.6, 0.0])
    state.belly_normal_force[0] = 1e-3 * torch.tensor([0.2, 0.0, 0.0, 0.0, 0.3])
    forces = AveragedForces(1, SEGMENTS, BODY.unit_weight_n, 0.5, "cpu")
    forces.restart(torch.tensor([True]), state)
    head_tip = state.head_tip_position[:, :2]
    previous_front_end = state.front_end_position.clone()
    previous_front_end[..., 0] -= 0.001
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)

    channels, parts, _ = reward.compute(
        state,
        state.body_planar_position - torch.tensor([0.001, 0.0]),
        head_tip - torch.tensor([[0.001, 0.0]]),
        previous_front_end,
        torch.tensor([[0.1, 0.0]]),
        torch.tensor([True]),
        0.025,
        zeros,
        zeros,
        torch.deg2rad(torch.tensor([[5.0, 0.0, 10.0, 0.0, 20.0]])),
        forces.values,
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(by_name["progress"], torch.tensor([1.0, 0, 0, 0, 0]))
    torch.testing.assert_close(by_name["arrival"], torch.tensor([25.0, 0, 0, 0, 0]))

    # Each follower's trail: the offset of the one behind and a quarter of
    # its own squared, the rear's own alone; an angle counts as its chord in
    # degrees. Segment 2 sits at the dead zone's edge, segments 1, 3 and the
    # rear about 5, 15 and 15 degrees beyond it.
    def chord(degrees: float) -> float:
        return math.degrees(2 * math.sin(math.radians(degrees) / 2))

    beyond = torch.tensor([chord(10) - 5, 0.0, chord(20) - 5, chord(20) - 5])
    torch.testing.assert_close(
        by_name["distance"],
        torch.tensor([0.0, *torch.exp(-((beyond / 20) ** 2))]),
        rtol=1e-3,
        atol=0,
    )
    # 300 mm/s per unit weight squeezed: 6 mm per 20 ms step.
    torch.testing.assert_close(
        by_name["squeeze"], -6 * torch.tensor([0.4, 0.3, 0.0, 0.0, 0.0])
    )
    one_motor = SETTINGS.force_price_w * 0.02 * 1e3
    torch.testing.assert_close(
        by_name["hanging"], -one_motor * torch.tensor([0.3 + 0.2, 0.0, 1.0, 0.0, 0.3])
    )
    torch.testing.assert_close(by_name["pushing"], -1e-3 * reward.sharing_weights[:, 2])
    # A full push for one step, priced as one motor fully on.
    torch.testing.assert_close(by_name["pain"], torch.tensor([0, 0, 0, -one_motor, 0]))
    goal = sum(by_name[name] for name in PART_NAMES[:4])
    torch.testing.assert_close(
        channels[0], torch.stack((goal, parts.rewards[0] - goal), -1)
    )

    state.net_joint_force[0, 2, 1] = 2e-3
    assert forces.update(state)[0, 2, 1] == pytest.approx(1.5)


def test_the_trail_is_where_the_front_end_went():
    """A segment a centimetre long walks 2 cm along x: it lies on its trail.
    Turned 30 degrees where it stands, it points 30 degrees left of it. Facing
    and walking a centimetre along y, it lies on its trail again; backing up a
    centimetre, it points the trail's opposite way."""
    trail = TrailMemory(1, (0.01,), "cpu")
    state = PhysicalState.allocate(1, 1, LEGS, "cpu")
    at_yaw(state, 0.0)
    trail.restart(torch.tensor([True]), state)

    def walk(steps: int, dx: float, dy: float) -> float:
        for _ in range(steps):
            state.front_end_position[0, 0] += torch.tensor([dx, dy])
            offset = trail.update(state)
        return math.degrees(offset[0, 0])

    assert walk(20, 0.001, 0.0) == pytest.approx(0.0, abs=1e-4)
    at_yaw(state, 30.0)
    assert walk(1, 0.0, 0.0) == pytest.approx(30.0, abs=1e-4)
    at_yaw(state, 90.0)
    assert walk(12, 0.0, 0.001) == pytest.approx(0.0, abs=1e-4)
    assert abs(walk(12, 0.0, -0.001)) == pytest.approx(180.0, abs=1e-3)


def test_a_block_holds_what_the_segment_senses_and_only_the_head_sees_the_target():
    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, "cpu"
    )
    block = builder.block_size
    assert (block, builder.observation_size) == (46, 5 * 46 + 2)
    state = resting_state(1)
    state.foot_normal_force[0, 1] = torch.tensor([2e-3, 1e-3])
    last_action = torch.rand(1, SEGMENTS, LEGS + 1)
    # Averaged forces: along, up, front joint, floor; segment 1 pushes the
    # head with 0.5 and pulls segment 2 with 2.
    averaged = torch.zeros(1, SEGMENTS, 4)
    averaged[0, :3, 2] = torch.tensor([0.0, -0.5, 2.0])
    averaged[0, 1, :2] = torch.tensor([0.3, 1.0])
    offset = torch.deg2rad(torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]]))

    observations = builder.build(
        state, torch.tensor([[0.025, 0.01]]), last_action, offset, averaged
    )

    own = observations[0, 1, :block]
    torch.testing.assert_close(own[:3], torch.tensor([0.0, 0.0, -1.0]))  # gravity
    torch.testing.assert_close(own[27:29], torch.tensor([2.0, 1.0]))  # unit weights
    # Front joint, net along and up, trail offset.
    torch.testing.assert_close(own[33:37], torch.tensor([-0.5, 0.3, 1.0, offset[0, 1]]))
    torch.testing.assert_close(own[-(LEGS + 1) :], last_action[0, 1])
    # Blocks: itself, the segments ahead, those behind; segment 1 sees its
    # rear joint in segment 2's block.
    torch.testing.assert_close(observations[0, 1, 3 * block + 33], torch.tensor(2.0))
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
    step_rewards = StepRewards(
        zeros, torch.zeros(2, SEGMENTS, len(PART_NAMES)), zeros, zeros
    )
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
            zeros,
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
    """The whole environment on v4: two channels, both for every segment, the
    spine motors receiving 0.35 of their command."""
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
    spine_commands = []
    step = environment.simulation.step

    def recorded_step(legs, spine):
        spine_commands.append(spine.clone())
        step(legs, spine)

    environment.simulation.step = recorded_step
    # World 0's target lies under its head.
    state = environment.simulation.physical_state
    environment.task.target_position[0] = state.body_planar_position[0, 0]
    action = torch.zeros(2, 12, 9)
    action[..., -1] = 1.0

    observations, rewards, terminated, truncated, final = environment.step(action)

    assert rewards.shape == (2, 12, 2) and torch.isfinite(rewards).all()
    assert environment.segment_reward_channels == [(0, 1)] * 12
    assert observations.shape == (2, 12, 5 * 46 + 2)
    torch.testing.assert_close(spine_commands[0], torch.full((2, 11), 0.35))
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
