"""Fixed values of the physics simulation, shared by both backends and the front.

They are design decisions, not configuration: changing one changes the task,
so they live here rather than in the settings file.
"""

import numpy as np

# One action is held for 200 physics steps of 0.1 ms: 20 ms of simulated time.
PHYSICS_STEPS_PER_ACTION = 200

# A reset offsets each leg joint angle uniformly within +-2 degrees and gives
# each leg joint a normally spread speed with this standard deviation.
LEG_ANGLE_NOISE_RAD = np.deg2rad(2)
LEG_SPEED_NOISE_RAD_S = 0.05
