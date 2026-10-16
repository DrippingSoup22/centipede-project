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
    energy_sharing_weights,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.cooperative_trail import TrailMemory
from centipede.environment.environment import create_environment
from centipede.environment.reward_function import StepRewards
from centipede.environment.settings import CooperatorsSettings, EnvironmentSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.model_mapping import HeadOutline
from centipede.settings_section import SettingsError

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
    for level, near, half_bearing in ((0.0, 0.02, 45.0), (1.0, final_near, 180.0)):
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
        torch.zeros(1, SEGMENTS),
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


def test_the_chain_charges_each_segment_its_own_joints_and_its_blocks_show_them():
    """Segment 1 pushes the head with half a unit weight and drags segment 2,
    which lags by two: segment 1 pays its push and the square of its pull,
    segment 2 its lag, and the head nothing for being pushed."""
    settings = CooperatorsSettings.from_section({"chain_price": 1.0})
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    reward = CooperativeReward(settings, SEGMENTS, 2, BODY, commanded)
    state = resting_state(1)
    state.front_joint_force[0] = torch.tensor([0.0, -0.5, 2.0, 0.0, 0.0]) * 1e-3
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)
    head_tip = state.head_tip_position[:, :2]
    _, parts, _ = reward.compute(
        state,
        state.body_planar_position.clone(),
        head_tip.clone(),
        head_tip + 0.1,
        torch.tensor([False]),
        0.025,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS),
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    one_motor = settings.force_price_w * 0.02 * 1e3
    expected = -one_motor * torch.tensor([0.0, 0.5 + 2.0**2, 2.0, 0.0, 0.0])
    torch.testing.assert_close(by_name["chain"], expected)

    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, True, "cpu"
    )
    assert builder.block_size == 43
    observations = builder.build(
        state, torch.tensor([[0.025, 0.01]]), zeros, torch.zeros(1, SEGMENTS)
    )
    blocks = observations[0, 1, :-2].reshape(5, 43)
    # Blocks: itself, the segments ahead, those behind. Segment 1 sees its
    # front joint in its own block and its rear joint in segment 2's.
    torch.testing.assert_close(blocks[[1, 0, 3], 33], torch.tensor([0.0, -0.5, 2.0]))


@pytest.mark.parametrize(
    ("rear_pull", "chain_paid"),
    [
        ("every", [1.0, 1.0 + 2.0**2, 2.0]),
        ("head", [1.0, 1.0, 2.0]),
        ("none", [0.0, 1.0, 2.0]),
    ],
)
def test_followers_without_a_goal_and_who_pays_the_rear_pull(rear_pull, chain_paid):
    """The body steps 1 mm forward while the head drags segment 1 by a unit
    weight and segment 1 drags segment 2 by two: only the head is paid
    progress, every follower pays its lag, and the rear pull's square is paid
    as ``chain_rear_pull`` says."""
    settings = CooperatorsSettings.from_section(
        {"chain_price": 1.0, "chain_rear_pull": rear_pull, "follower_goal": False}
    )
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    reward = CooperativeReward(settings, SEGMENTS, 2, BODY, commanded)
    state = resting_state(1)
    state.front_joint_force[0] = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0]) * 1e-3
    previous = state.body_planar_position.clone()
    previous[..., 0] -= 0.001
    head_tip = state.head_tip_position[:, :2]
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)
    _, parts, _ = reward.compute(
        state,
        previous,
        head_tip - torch.tensor([[0.001, 0.0]]),
        torch.tensor([[0.1, 0.0]]),
        torch.tensor([False]),
        0.025,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS),
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(by_name["progress"], torch.tensor([1.0, 0, 0, 0, 0]))
    one_motor = settings.force_price_w * 0.02 * 1e3
    expected = -one_motor * torch.tensor([*chain_paid, 0.0, 0.0])
    torch.testing.assert_close(by_name["chain"], expected)


def at_yaw(state: PhysicalState, degrees: float) -> None:
    """Turn every segment of the state to face ``degrees`` from +x."""
    half = math.radians(degrees) / 2
    state.body_quaternion[:] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


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


def test_the_trail_charges_the_steerer_and_carrying_oneself_the_carried():
    """Offsets of 5, 0, 10, 0 and 20 degrees: the head pays its own, each
    segment the one behind it fully and its own at a quarter. Segment 1 is
    pulled by half a unit weight and held up by one; the head, pushed and
    held up a little, pays only for being held up; segment 2, pressed down,
    pays nothing. Each block shows its segment's net force and offset."""
    settings = CooperatorsSettings.from_section(
        {"follower_goal": False, "follow_trail": True, "carry_price": 0.25}
    )
    reward = CooperativeReward(
        settings, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, LEGS + 1)
    )
    state = resting_state(1)
    state.net_joint_force[0, :3] = (
        torch.tensor([[-0.3, 0.2], [0.5, 1.0], [0.0, -0.4]]) * 1e-3
    )
    offset = torch.deg2rad(torch.tensor([[5.0, 0.0, 10.0, 0.0, 20.0]]))
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)
    head_tip = state.head_tip_position[:, :2]
    channels, parts, _ = reward.compute(
        state,
        state.body_planar_position.clone(),
        head_tip.clone(),
        head_tip + 0.1,
        torch.tensor([False]),
        0.025,
        zeros,
        zeros,
        offset,
    )

    def squared(degrees: float) -> float:
        return (180 / math.pi) ** 2 * 2 * (1 - math.cos(math.radians(degrees)))

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    expected = [
        squared(5),
        squared(10),
        0.25 * squared(10),
        squared(20),
        0.25 * squared(20),
    ]
    torch.testing.assert_close(
        by_name["trail"], -torch.tensor(expected), rtol=1e-4, atol=0
    )
    assert squared(10) == pytest.approx(100.0, rel=0.01)  # its square, in degrees
    one_motor = settings.force_price_w * 0.02 * 1e3
    carried = torch.tensor([0.2, 0.5 + 1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(by_name["carry"], -0.25 * one_motor * carried)
    assert channels.shape == (1, SEGMENTS, 3)
    torch.testing.assert_close(channels[0, :, 1], by_name["trail"])

    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, False, "cpu", trail=True, carry=True
    )
    assert builder.block_size == 45
    observations = builder.build(state, torch.tensor([[0.025, 0.01]]), zeros, offset)
    own = observations[0, 1, :45]
    torch.testing.assert_close(own[33:36], torch.tensor([0.5, 1.0, 0.0]))
    torch.testing.assert_close(observations[0, 2, 35], offset[0, 2])
    with pytest.raises(SettingsError, match="follower_goal"):
        CooperatorsSettings.from_section({"follow_trail": True})


def test_a_block_holds_what_the_segment_senses_and_only_the_head_sees_the_target():
    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, False, "cpu"
    )
    block = builder.block_size
    assert (block, builder.observation_size) == (42, 5 * 42 + 2)
    state = resting_state(1)
    state.foot_normal_force[0, 1] = torch.tensor([2e-3, 1e-3])
    last_action = torch.rand(1, SEGMENTS, LEGS + 1)

    observations = builder.build(
        state, torch.tensor([[0.025, 0.01]]), last_action, torch.zeros(1, SEGMENTS)
    )

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


def test_coordination_charges_who_drives_lags_or_hangs_and_the_trail_pays_time():
    """Cycle 5, averaged forces in unit weights. Segment 1 drives the body by
    0.4, 0.3 of it pushing the segment ahead, counted twice; segment 2 lags
    by 0.5 and hangs by 1; segment 3 drives 0.2 into pulling the one behind;
    the rear lies on the floor with 0.3, pressed down by its neighbour. The
    head, held up by 0.8 and on the floor with 0.2, pays the 0.3 beyond half
    its weight and the floor, priced, in its health. The trail pays each
    step's duration in full within 5 degrees, less beyond. Energy is shared
    40% to itself and 30% to each side."""
    settings = CooperatorsSettings.from_section(
        {
            "follower_goal": False,
            "follow_trail": True,
            "coordination": True,
            "carry_price": 1.0,
            "energy_sharing": "sides",
        }
    )
    reward = CooperativeReward(
        settings, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, LEGS + 1)
    )
    torch.testing.assert_close(
        reward.sharing_weights[:2],
        torch.tensor([[0.4, 0.4, 0.2, 0.0, 0.0], [0.3, 0.4, 0.2, 0.1, 0.0]]),
    )
    state = resting_state(1)
    state.net_joint_force[0] = 1e-3 * torch.tensor(
        [[-0.5, 0.8], [-0.4, 0.0], [0.5, 1.0], [-0.2, 0.0], [0.0, -0.4]]
    )
    state.front_joint_force[0] = 1e-3 * torch.tensor([0.0, -0.3, 0.0, 0.6, 0.0])
    state.belly_normal_force[0] = 1e-3 * torch.tensor([0.2, 0.0, 0.0, 0.0, 0.3])
    forces = AveragedForces(1, SEGMENTS, BODY.unit_weight_n, 0.5, "cpu")
    forces.restart(torch.tensor([True]), state)
    offset = torch.deg2rad(torch.tensor([[5.0, 0.0, 10.0, 0.0, 20.0]]))
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)
    head_tip = state.head_tip_position[:, :2]
    channels, parts, _ = reward.compute(
        state,
        state.body_planar_position.clone(),
        head_tip.clone(),
        head_tip + 0.1,
        torch.tensor([False]),
        0.025,
        zeros,
        zeros,
        offset,
        forces.values,
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(
        by_name["coordination"], -torch.tensor([0.0, 0.7, 1.5, 0.2, 0.3])
    )
    one_motor = settings.force_price_w * 0.02 * 1e3
    torch.testing.assert_close(
        by_name["carry"], -one_motor * torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0])
    )

    # The head: a quarter of its own 5 degrees, within the dead zone; segment
    # 2: a quarter of its own 10, at its edge; segments 1, 3 and the rear
    # 10, 20 and 20 degrees, about 5, 15 and 15 beyond it (an angle counts
    # as its chord, in degrees: the square root of the squared offset).
    def chord(degrees: float) -> float:
        return math.degrees(2 * math.sin(math.radians(degrees) / 2))

    beyond = torch.tensor([0.0, chord(10) - 5, 0.0, chord(20) - 5, chord(20) - 5])
    torch.testing.assert_close(
        by_name["trail"], 0.02 * torch.exp(-((beyond / 20) ** 2)), rtol=1e-3, atol=0
    )
    assert channels.shape == (1, SEGMENTS, 4)
    torch.testing.assert_close(channels[0, :, 1], by_name["coordination"])
    torch.testing.assert_close(channels[0, :, 2], by_name["trail"])

    state.net_joint_force[0, 2, 0] = 1.5e-3
    assert forces.update(state)[0, 2, 0] == pytest.approx(1.0)
    with pytest.raises(SettingsError, match="follow_trail"):
        CooperatorsSettings.from_section({"coordination": True})


@pytest.mark.parametrize(
    ("coordination", "channel_count", "followers", "block"),
    [(False, 3, (1, 2), 45), (True, 4, (1, 2, 3), 46)],
)
def test_with_the_trail_the_followers_learn_from_trail_and_cost(
    coordination, channel_count, followers, block
):
    """The whole environment on v4: three channels, the followers on two; with
    coordination four, every segment on three."""
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
            "cooperators": {
                "stagger_first_episodes": False,
                "follower_goal": False,
                "follow_trail": True,
                "carry_price": 0.25,
                "coordination": coordination,
                "averaging_s": 0.2,
            },
        }
    )
    environment = create_environment(settings)
    environment.reset(seed=0)

    observations, rewards, *_ = environment.step(torch.zeros(2, 12, 9))

    head = (0, 2, 3) if coordination else (0, 1, 2)
    assert environment.segment_reward_channels[:2] == [head, followers]
    assert rewards.shape == (2, 12, channel_count) and torch.isfinite(rewards).all()
    assert observations.shape == (2, 12, 5 * block + 2)
