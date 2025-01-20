# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from lib.arguments import parser

parser.add_argument(
    "--no-overlap", action="store_true", help="Add padding when moving back the loop"
)

from . import Simulation, WallFollowing
from math import pi
from numpy import argmin
from enum import Enum

from lib.geometry import Point


class Bug1(Simulation, WallFollowing):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from lib.env import args

        self.no_overlap: bool = args.no_overlap

    class Mode(Enum):
        MOVE_TO_DST = 0
        MOVE_ALONG_WALL = 1
        MOVE_TO_CLOSEST_POINT = 2

    mode: Mode = Mode.MOVE_TO_DST
    wall_loop: list[Point] = []

    @property
    def loop_closed(self):
        p1 = self.wall_loop[-1]
        for p0 in self.wall_loop[:-5][:10]:
            if (p0 - p1).norm < max(self.step_length * 2, self.world.radius):
                return True
        return False

    decay: float = 0.5
    prev_dp: Point | None = None
    padding: Point = Point(0, 0, type=int)
    loop_enter_heading: float = 0.0

    @property
    def padding_offset(self):
        return self.world.line_width_meters * 4

    def rolling(self, dp: Point) -> Point:
        if self.prev_dp is None:
            self.prev_dp = dp
        else:
            self.prev_dp = self.prev_dp * self.decay + dp * (1.0 - self.decay)
        return self.prev_dp

    def pad_loop(self, dp: Point) -> Point:
        raise NotImplementedError

    def step(self, pos, dst):
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                # The target direction
                r1 = (dst - pos).angle
                yield self.move(r1)
                # Hit the wall
                self.loop_enter_heading = r1
                self.mode = self.Mode.MOVE_ALONG_WALL
                self.wall_loop = [pos]
                # Left turn not viable, try right turn
                yield self.hit_wall
            case self.Mode.MOVE_ALONG_WALL:
                # Check for loop closure
                if self.loop_closed:
                    distances = [(dst - p).norm for p in self.wall_loop]
                    idx = int(argmin(distances))
                    self.wall_loop = self.wall_loop[idx:]
                    if len(self.wall_loop) > 1:
                        self.mode = self.Mode.MOVE_TO_CLOSEST_POINT
                        if self.no_overlap:
                            self.padding = self.move(
                                self.loop_enter_heading, -self.padding_offset
                            )
                            with self.no_check:
                                yield self.padding
                    else:
                        self.mode = self.Mode.MOVE_TO_DST
                    yield None
                else:
                    self.wall_loop.append(pos)
                    yield self.move_along_wall
            case self.Mode.MOVE_TO_CLOSEST_POINT:
                if len(self.wall_loop) == 0:
                    self.mode = self.Mode.MOVE_TO_DST
                    self.padding = Point(0, 0, type=int)
                    self.prev_dp = None
                    yield None
                else:
                    with self.no_check:
                        if self.no_overlap:
                            p1 = self.wall_loop.pop(-1)
                            dp = p1 - (pos - self.padding)
                            self.padding = self.pad_loop(self.rolling(dp))
                            yield p1 - pos + self.padding
                        else:
                            yield self.wall_loop.pop(-1) - pos
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")
