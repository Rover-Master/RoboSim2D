# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation
from .Bug1 import Bug1


class Bug1L(Bug1):
    wall_following_direction = "L"


if __name__ == "__main__":
    Simulation.run(Bug1L())
