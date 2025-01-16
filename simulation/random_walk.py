# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from math import pi
from random import random
from . import Simulation


class RandomWalk(Simulation):
    heading: float = random() * 2 * pi

    def step(self, pos, dst):
        # Try to keep previous heading
        yield pos + self.move(self.heading)
        # Original heading no longer viable,
        # try new random headings until a viable one is found
        while True:
            self.heading = random() * 2 * pi
            yield pos + self.move(self.heading)


if __name__ == "__main__":
    Simulation.run(RandomWalk())
