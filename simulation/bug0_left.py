# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from . import Simulation
from math import pi
from numpy import linspace


class Bug0Left(Simulation):
    def step(self, pos, dst):
        r = (dst - pos).angle - 0.5 * pi
        # First try to move directly to the destination
        yield pos + self.move(r)
        # Then try to move along the obstacle
        for _ in linspace(r, r + 2 * pi, 360):
            yield pos + self.move(_)


if __name__ == "__main__":
    Simulation.run(Bug0Left())
