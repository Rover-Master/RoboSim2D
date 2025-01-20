# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from . import Simulation
from math import pi
from numpy import linspace
from lib.geometry import Point


class Bug0R(Simulation):
    p0: Point[float] | None = None
    p1: Point[float] | None = None

    def step(self, pos, dst):
        if self.p0 is not None and (self.p0 - pos).norm < self.step_length:
            raise Simulation.Abort("dead-loop")
        self.p0, self.p1 = self.p1, pos
        r = (dst - pos).angle
        # First try to move directly to the destination
        yield self.move(r)
        # Then try to move along the obstacle
        yield map(self.move, linspace(r, r - 2 * pi, 180))


if __name__ == "__main__":
    Simulation.run(Bug0R())
