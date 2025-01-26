# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from lib.arguments import register_arguments, Argument

register_arguments(
    heading=Argument(
        type=float, required=False, help="Initial heading of the agent in radians."
    )
)

from . import Simulation, Hypothesis
from lib.geometry import Point


class WallBounce(Simulation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heading: float = self.world.meta.get("heading", 0.0)

    def step(self, pos: Point[float], dst):
        go_ahead = self.move(self.heading)
        yield [
            # Try to keep previous heading
            go_ahead,
            # Micro step to ensure robot is against the wall
            go_ahead / 2.0,
        ]
        # Find tangent vectors on both sides
        l: Point | None = yield Hypothesis(self.turn("left"))
        r: Point | None = yield Hypothesis(self.turn("right"))
        if l is None or r is None:
            raise Simulation.Abort("unable to determine wall normal")
        # Derive normal vector from tangent vectors
        a, b = (l - pos).angle, (r - pos).angle
        n = Point.Angular(a + (b - a) / 2)
        # Make reflection
        v: Point[float] = go_ahead - n * 2 * (go_ahead @ n)
        yield v
        yield self.wiggle(v.angle)


if __name__ == "__main__":
    Simulation.run(WallBounce())
