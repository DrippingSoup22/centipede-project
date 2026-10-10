"""Validation of model v4 against v3.

v4 is v3 with position actuators on the legs, whose commands are target
angles, and weaker lift and knee motors. Their step response, standing and
carrying were measured once and are recorded in docs/model.md (Model v4);
this test checks that nothing else changed.
"""

import mujoco
import numpy as np

# The compiled arrays the new leg actuators change: their kind (bias type),
# gain and bias, angle range, torque limit and gear, and actuator_acc0, which
# MuJoCo derives from the gear.
LEG_ACTUATOR_ARRAYS = {
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_biastype",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_gear",
    "actuator_acc0",
}


def test_only_the_actuators_differ_from_v3():
    v3 = mujoco.MjModel.from_xml_path("models/assembly_v3.xml")
    v4 = mujoco.MjModel.from_xml_path("models/assembly_v4.xml")

    differing = {
        name
        for name in dir(v3)
        if isinstance(getattr(v3, name), np.ndarray)
        and not np.array_equal(getattr(v3, name), getattr(v4, name))
    }
    assert differing == LEG_ACTUATOR_ARRAYS
    for name in dir(v3.opt):
        value = getattr(v3.opt, name)
        if not name.startswith("_") and not callable(value):
            assert np.array_equal(value, getattr(v4.opt, name)), name
