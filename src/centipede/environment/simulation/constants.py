"""Fixed values of the physics simulation, shared by both backends and the front.

They are design decisions, not configuration: changing one changes the task,
so they live here rather than in the settings file.
"""

import numpy as np

# One action is held for 20 ms of simulated time, as many physics steps as the
# model's timestep fits into it (134 for model v3, 200 for v2).
ACTION_DURATION_S = 0.020

# A reset offsets each leg joint angle uniformly within +-2 degrees and gives
# each leg joint a normally spread speed with this standard deviation.
LEG_ANGLE_NOISE_RAD = np.deg2rad(2)
LEG_SPEED_NOISE_RAD_S = 0.05
