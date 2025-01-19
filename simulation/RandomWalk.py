# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from math import pi
from random import random
from . import Simulation
from lib.util import repeat, limited


class RandomWalk(Simulation):
    heading: float = random() * 2 * pi

    def step(self, pos, dst):
        yield [
            # Try to keep previous heading
            self.move(self.heading),
            # Original heading no longer viable,
            # try new random headings until a viable one is found
            map(lambda r: self.move(r * 2 * pi), limited(repeat(random), 100)),
        ]


if __name__ == "__main__":
    Simulation.run(RandomWalk())
