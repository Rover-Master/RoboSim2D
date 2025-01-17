# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation
from math import pi, ceil
from numpy import linspace


class Bug2CCW(Simulation):
    def step(self, pos, dst):
        r0 = self.heading
        # The target direction
        r1 = (dst - pos).angle - 0.5 * pi
        # Lazy attempt to turn left towards the destination
        # dr ~ (0, 2 * pi) - CW
        dr = (r1 - r0) % (2 * pi)
        assert dr >= 0
        n = ceil(abs(dr) / (pi / 180.0))
        for r in linspace(r0, r0 + dr, n):
            yield pos + self.move(r), True
        # Then try to move along the obstacle (right turn)
        for r in linspace(r0, r0 - 2 * pi, 360):
            yield pos + self.move(r)


if __name__ == "__main__":
    Simulation.run(Bug2CCW())
