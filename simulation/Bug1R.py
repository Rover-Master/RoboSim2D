# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from math import pi
from . import Simulation
from .Bug1 import Bug1


class Bug1R(Bug1):
    wall_following_direction = "R"

    def pad_loop(self, dp):
        return self.move(dp.angle + 0.5 * pi, self.padding_offset)


if __name__ == "__main__":
    Simulation.run(Bug1R())
