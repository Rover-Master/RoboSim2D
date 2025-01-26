# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from . import Simulation
from lib.geometry import Point


class Bug0(Simulation):
    check_step_threshold: int = 10

    def __init__(self):
        super().__init__()
        self.hist = list[Point[float]]()

    def check(self, pos):
        """
        Bug 0 does not have any memory of visited points, it makes the same
        decisions at the same location. Therefore, if we revisit a previously
        visited point, we will repeat the same path as we did before - and that
        will cause a dead-loop.
        """
        self.hist.append(pos)
        to_check = self.hist[-self.check_step_threshold : 0 : -1]
        for p in to_check:
            if (p - pos).norm < self.step_length:
                raise Simulation.Abort("dead-loop")

    def hit_wall(self, r: float):
        raise NotImplementedError

    def step(self, pos, dst):
        self.check(pos)
        r = (dst - pos).angle
        yield (
            # First try to move directly to the destination
            self.move(r),
            # Then try to move along the obstacle
            self.hit_wall(r),
        )
