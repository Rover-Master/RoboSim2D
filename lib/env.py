# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from .world import World
from .geometry import Point


def parse_point(s: str) -> Point[float]:
    return Point(*s.split(","), type=float)


parser = ArgumentParser()

parser.add_argument("world", type=str, help="Path to the map file", nargs=1)
# Destination folder
parser.add_argument(
    "--prefix", type=str, help="Destination file or folder", default=None
)
# Simulation parameters
parser.add_argument("--src", type=parse_point, help="Source", default=None)
parser.add_argument("--dst", type=parse_point, help="Destination", default=None)
# Simulation conditions
parser.add_argument(
    "-t", "--threshold", type=float, help="Threshold Distance", default=0.5
)
parser.add_argument(
    "-r",
    "--radius",
    help="Collision radius for the robot, in meters",
    type=float,
    default=None,
)
# Visualization parameters
parser.add_argument(
    "-s", "--scale", type=float, help="DPI Scale for UI display", default=2.0
)
parser.add_argument(
    "-v",
    "--visualize",
    help="Visualize Simulation",
    default=False,
    action="store_true",
)
parser.add_argument(
    "-w",
    "--line-width",
    help="Line Width for Visualization, in Meters",
    type=float,
    default=0.05,
)
parser.add_argument(
    "-R",
    "--resolution",
    help="World pixel resolution in meters",
    type=float,
    default=0.025,
)
parser.add_argument(
    "--debug",
    help="Toggle debug tools",
    default=False,
    action="store_true",
)

args = parser.parse_args()
params = dict(
    prefix=args.prefix,
    radius=args.radius,
    threshold=float(args.threshold),
    dpi_scale=float(args.scale),
    visualize=bool(args.visualize),
    line_width=float(args.line_width),
    resolution=float(args.resolution),
    debug=bool(args.debug),
)

world = World(str(args.world[0]), **params)

src_pos: Point[float] = args.src
dst_pos: Point[float] = args.dst

if src_pos is None or dst_pos is None:
    from lib.interact import prompt

    src_pos, dst_pos = prompt(world, src_pos, dst_pos)

if __name__ == "__main__":
    print("SRC", src_pos)
    print("DST", dst_pos)
