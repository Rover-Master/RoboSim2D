# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .arguments import register_arguments, Argument
from .util import tuple_of
from .geometry import Point


def point(s: str) -> Point[float]:
    return Point(*tuple_of(float)(s), type=float)


register_arguments(
    src=Argument(type=point, default=None, help="Simulation Starting Location"),
    dst=Argument(type=point, default=None, help="Simulation Destination"),
    threshold=Argument(
        "-t",
        type=float,
        default=0.5,
        help="Threshold distance near the target",
    ),
    radius=Argument(
        "-r",
        type=float,
        default=None,
        help="Collision radius for the robot, in meters",
    ),
    max_travel=Argument(
        "-M",
        type=float,
        default=None,
        help="Max travel distance before aborting the simulation",
    ),
    step_length=Argument(
        type=float, default=None, help="Simulation step length, in meters"
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
from dataclasses import dataclass, field

from .util import own_attrs
from .world import World
from .output import Output
from .visualization import Visualization
from .interact import prompt


@own_attrs
@dataclass(frozen=False)
class SimulationBase:
    world: World
    debug: bool

    src: Point[float]
    dst: Point[float]
    threshold: float
    radius: float
    max_travel: float
    step_length: float

    out: Output = field(init=False)
    vis: Visualization = field(init=False)

    def __post_init__(self):
        self.out = Output(**self.world.meta)
        self.vis = Visualization(**self.world.meta)
        if self.radius is None:
            self.radius = self.world.res * 4
        if self.step_length is None:
            self.step_length = self.world.res * 4
        if self.src is None or self.dst is None:
            self.src, self.dst = prompt(self.vis, self.src, self.dst, self.radius)


if __name__ == "__main__":
    from .arguments import parse

    SimulationBase(**parse())
