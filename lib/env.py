# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from math import ceil
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
parser.add_argument(
    "-w",
    "--line-width",
    help="Line Width for Visualization, in Meters",
    type=float,
    default=0.05,
)
parser.add_argument(
    "-r",
    "--resolution",
    help="Pixel resolution in meters",
    type=float,
    default=0.025,
)

args = parser.parse_args()
scale = float(args.scale)
threshold = float(args.threshold)
visualize = bool(args.visualize)
line_width = float(args.line_width)
resolution = float(args.resolution)
world = World(
    str(args.map[0]), resolution=resolution, dpi_scale=scale, line_width=line_width
)

import cv2, numpy as np


def show(title: str, img: np.ndarray):
    s = 1 / scale
    cv2.imshow(title, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR))


src_pos: Point[float] = args.src
dst_pos: Point[float] = args.dst

if src_pos is None or dst_pos is None:
    from lib.interact import prompt

    src_pos, dst_pos = prompt(world, src_pos, dst_pos)
