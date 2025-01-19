# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation
from .Bug2 import Bug2


class Bug2CCW(Bug2):
    wall_following_direction = "CCW"


if __name__ == "__main__":
    Simulation.run(Bug2CCW())
