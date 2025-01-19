# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .world import World
from .geometry import Point
from .arguments import parse

name, params, args = parse()

world = World(str(name), **params)

src_pos: Point[float] = args.src
dst_pos: Point[float] = args.dst

if src_pos is None or dst_pos is None:
    from lib.interact import prompt

    src_pos, dst_pos = prompt(world, src_pos, dst_pos)

if __name__ == "__main__":
    print("SRC", src_pos)
    print("DST", dst_pos)
