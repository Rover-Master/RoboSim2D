# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from lib.arguments import register_arguments, Argument

register_arguments(
    seed=Argument(type=float, required=False, help="Initial random seed.")
)

from math import pi
from random import Random
from . import Simulation
from lib.util import repeat


class RandomWalk(Simulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = self.world.meta.get("seed", None)
        self.random = Random(seed).random
        self.heading = self.random() * 2 * pi

    def step(self, pos, dst):
        go_ahead = self.move(self.heading)
        yield [
            # Try to keep previous heading
            go_ahead,
            go_ahead / 2.0,
            go_ahead / 4.0,
            # Original heading no longer viable,
            # try new random headings until a viable one is found
            map(lambda r: self.move(r * 2 * pi), repeat(self.random)),
        ]


if __name__ == "__main__":
    Simulation.run(RandomWalk())
