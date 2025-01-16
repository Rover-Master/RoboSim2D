# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation


class Bug1(Simulation):
    heading: 0.0  # Initial heading in radians

    def step(self, pos, dst):
        raise NotImplementedError


if __name__ == "__main__":
    Simulation.run(Bug1())
