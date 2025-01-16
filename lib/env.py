# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from .world import World
from .geometry import Point


def parse_point(s: str) -> Point[float]:
    return Point(*map(float, s.split(",")))


parser = ArgumentParser()

parser.add_argument("map", type=str, help="Path to the map file", nargs=1)
parser.add_argument(
    "-s", "--scale", type=float, help="DPI Scale for UI display", default=2.0
)
parser.add_argument("--src", type=parse_point, help="Source", default=None)
parser.add_argument("--dst", type=parse_point, help="Destination", default=None)
parser.add_argument(
    "-t", "--threshold", type=float, help="Threshold Distance", default=0.5
)
parser.add_argument(
    "-v",
    "--visualize",
    help="Visualize Simulation",
    default=False,
    action="store_true",
)

args = parser.parse_args()
world = World(str(args.map[0]))
scale = float(args.scale)
threshold = float(args.threshold)
visualize = bool(args.visualize)

src_pos: Point[float] = args.src
dst_pos: Point[float] = args.dst

if src_pos is None or dst_pos is None:
    from lib.interactive import prompt

    src_pos, dst_pos = prompt(world, scale, src_pos, dst_pos)
