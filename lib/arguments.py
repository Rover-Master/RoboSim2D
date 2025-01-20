# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from .geometry import Point


def parse_point(s: str) -> Point[float]:
    return Point(*s.split(","), type=float)


def tuple_of(t: type):
    def transform(s: str) -> tuple:
        return tuple(map(t, s.split(",")))

    return transform


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
    "-t",
    "--threshold",
    type=float,
    help="Threshold distance near the target",
    default=0.5,
)
parser.add_argument(
    "-r",
    "--radius",
    help="Collision radius for the robot, in meters",
    type=float,
    default=None,
)
parser.add_argument(
    "-M",
    "--max-travel",
    help="Max travel distance before aborting the simulation",
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
    "--no-wait",
    help="Do not wait for key stroke after simulation is complete",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--line-width",
    help="Line Width for Visualization, in Meters",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--line-color",
    help="Color of the trajectory line",
    type=tuple_of(float),
    default=(0, 0, 255),
)
parser.add_argument(
    "-R",
    "--resolution",
    help="World pixel resolution in meters (m/px)",
    type=float,
    default=None,
)
parser.add_argument(
    "-S",
    "--slice",
    help="Output Image Slice (x, y, w, h)",
    type=tuple_of(int),
    default=None,
)
parser.add_argument(
    "--debug",
    help="Toggle debug tools",
    default=False,
    action="store_true",
)


def parse():
    args = parser.parse_args()
    params = dict(
        prefix=args.prefix,
        radius=args.radius,
        threshold=args.threshold,
        max_travel=args.max_travel,
        dpi_scale=args.scale,
        visualize=args.visualize,
        no_wait=args.no_wait,
        line_width=args.line_width,
        line_color=args.line_color,
        resolution=args.resolution,
        slice=args.slice,
        debug=args.debug,
    )
    return args.world[0], params, args
