"""Tests for reading the segment contract from a loaded MuJoCo model."""

from pathlib import Path

import mujoco
import pytest

from centipede.environment.simulation.model_mapping import (
    BODY_CATEGORY,
    FLOOR_CATEGORY,
    LEFT_FOOT_CATEGORY,
    LEG_ACTION_ORDER,
    MEMBRANE_CATEGORY,
    RIGHT_FOOT_CATEGORY,
    ModelContractError,
    ModelMapping,
)

MODEL_XML = Path("models/assembly_v3.xml").read_text(encoding="utf-8")
KNEE_MOTOR = (
    'name="segment_03_left_knee_motor" joint="segment_03_left_knee" '
    'gear="1e-05" ctrlrange="-1 1" ctrllimited="true" forcerange="-1 1" '
    'forcelimited="true" user="3"'
)


def load(xml: str = MODEL_XML) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def edited(old: str, new: str) -> mujoco.MjModel:
    """The v2 model with one exact piece of its XML replaced."""
    assert MODEL_XML.count(old) == 1, old
    return load(MODEL_XML.replace(old, new))


def test_mapping_matches_the_elements_found_by_name():
    model = load()
    mapping = ModelMapping.from_model(model)

    assert mapping.segment_count == 8
    for segment in range(8):
        prefix = f"segment_{segment:02d}"
        assert mapping.body_ids[segment] == model.body(prefix).id
        assert mapping.center_site_ids[segment] == model.site(f"{prefix}_center").id
        for action_index, (side, role) in enumerate(LEG_ACTION_ORDER):
            joint = model.joint(f"{prefix}_{side}_{role}")
            motor = model.actuator(f"{prefix}_{side}_{role}_motor")
            assert mapping.leg_actuator_ids[segment, action_index] == motor.id
            assert mapping.leg_qpos_addresses[segment, action_index] == joint.qposadr
            assert mapping.leg_dof_addresses[segment, action_index] == joint.dofadr
    # Spine joint i joins segment i to segment i + 1, whose unit holds its motor.
    for joint_index in range(7):
        name = f"segment_{joint_index + 1:02d}_yaw"
        assert (
            mapping.spine_actuator_ids[joint_index]
            == model.actuator(f"{name}_motor").id
        )
        assert mapping.spine_qpos_addresses[joint_index] == model.joint(name).qposadr
        assert mapping.spine_dof_addresses[joint_index] == model.joint(name).dofadr
    assert mapping.head_tip_site_id == model.site("head_tip").id
    assert mapping.root_qpos_address == model.joint("root").qposadr[0]
    # The head seen from above: 2.2 mm behind its centre to its tip 5 mm ahead,
    # and 8 mm wide (docs/model.md, Body).
    outline = mapping.head_outline
    assert (outline.rear_m, outline.front_m, outline.half_width_m) == pytest.approx(
        (-0.0022, 0.005, 0.004), abs=2e-5
    )

    for name, owner, category in [
        ("floor", -1, FLOOR_CATEGORY),
        ("segment_03_body", 3, BODY_CATEGORY),
        ("segment_03_left_foot", 3, LEFT_FOOT_CATEGORY),
        ("segment_03_right_foot", 3, RIGHT_FOOT_CATEGORY),
        ("segment_03_membrane", 3, MEMBRANE_CATEGORY),
    ]:
        geom_id = model.geom(name).id
        assert mapping.geom_owner_indices[geom_id] == owner
        assert mapping.geom_categories[geom_id] == category


def test_v1_is_rejected_because_its_feet_share_one_category():
    v1 = mujoco.MjModel.from_xml_path("models/assembly.xml")

    with pytest.raises(ModelContractError, match="2 shapes of category left foot"):
        ModelMapping.from_model(v1)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'name="segment_03_left_knee_motor"',
            'name="segment_03_left_knee_drive"',
            "no actuator named segment_03_left_knee_motor",
        ),
        (KNEE_MOTOR, KNEE_MOTOR.replace('user="3"', 'user="4"'), "owned by segment 4"),
        (
            KNEE_MOTOR,
            KNEE_MOTOR.replace(
                'joint="segment_03_left_knee"', 'joint="segment_03_left_lift"'
            ),
            "drives segment_03_left_lift, expected segment_03_left_knee",
        ),
        (
            KNEE_MOTOR,
            KNEE_MOTOR.replace('ctrlrange="-1 1"', 'ctrlrange="-2 2"'),
            "from -2.0 to 2.0, expected -1 to 1",
        ),
        (
            'name="segment_05_yaw_motor"',
            'name="segment_05_bend_motor"',
            "no actuator named segment_05_yaw_motor",
        ),
        (
            "  </actuator>",
            '    <motor name="segment_03_tail_motor" joint="segment_03_pitch" '
            'ctrlrange="-1 1" user="3" />\n  </actuator>',
            "belong to no leg or spine joint: segment_03_tail_motor",
        ),
        ('name="head_tip"', 'name="head_point"', "no site named head_tip"),
        ('<freejoint name="root" />', "", "segment_00 must be a child of the world"),
        (
            'contype="2" conaffinity="1" user="3 1"',
            'contype="2" conaffinity="1" user="3 9"',
            "segment_03_body has unknown category 9",
        ),
    ],
    ids=[
        "missing motor",
        "wrong owner",
        "wrong joint",
        "wrong range",
        "missing spine motor",
        "extra motor",
        "missing site",
        "fixed head",
        "unknown category",
    ],
)
def test_broken_xml_is_rejected(old, new, message):
    with pytest.raises(ModelContractError, match=message):
        ModelMapping.from_model(edited(old, new))


def test_broken_ownership_is_rejected():
    no_segment_1 = load()
    owners = no_segment_1.actuator_user[:, 0]
    owners[owners == 1] = 2
    with pytest.raises(ModelContractError, match=r"Segments without motors: \[1\]"):
        ModelMapping.from_model(no_segment_1)

    two_bodies = load()
    two_bodies.geom_user[two_bodies.geom("segment_04_body").id, 0] = 3
    with pytest.raises(ModelContractError, match="Segment 3 owns 2 shapes"):
        ModelMapping.from_model(two_bodies)

    unknown_owner = load()
    unknown_owner.geom_user[unknown_owner.geom("segment_03_left_foot").id, 0] = 12
    with pytest.raises(ModelContractError, match="has invalid owner 12"):
        ModelMapping.from_model(unknown_owner)
