"""Tests for the [interaction_loop] configuration section.

The generic value checks are tested with ``SettingsSection``; this covers only
the loop's keys and defaults.
"""

import pytest

from centipede.interaction_loop.settings import InteractionLoopSettings
from centipede.settings_section import SettingsError


def test_defaults_and_section_name():
    assert InteractionLoopSettings.from_section({}) == InteractionLoopSettings(
        rollout_window_steps=256, update_cycles=128
    )
    with pytest.raises(SettingsError, match=r"\[interaction_loop\] update_cycles"):
        InteractionLoopSettings.from_section({"update_cycles": 0})
