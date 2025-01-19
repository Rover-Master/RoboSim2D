# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation, WallFollowing
from math import pi
from numpy import argmin
from enum import Enum

from lib.geometry import Point


class Bug1(Simulation, WallFollowing):

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
            if (p0 - p1).norm < self.step_length * 2:
                return True
        return False

    def step(self, pos, dst):
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                # The target direction
                r1 = (dst - pos).angle - 0.5 * pi
                yield self.move(r1)
                # Hit the wall
                self.mode = self.Mode.MOVE_ALONG_WALL
                self.wall_loop = [pos]
                # Left turn not viable, try right turn
                yield self.hit_wall
            case self.Mode.MOVE_ALONG_WALL:
                self.wall_loop.append(pos)
                # Check for loop closure
                if self.loop_closed:
                    distances = [(dst - p).norm for p in self.wall_loop]
                    idx = int(argmin(distances))
                    self.wall_loop = self.wall_loop[idx:]
                    if len(self.wall_loop) > 1:
                        self.mode = self.Mode.MOVE_TO_CLOSEST_POINT
                    else:
                        self.mode = self.Mode.MOVE_TO_DST
                    yield None
                else:
                    yield self.move_along_wall
            case self.Mode.MOVE_TO_CLOSEST_POINT:
                if len(self.wall_loop) == 0:
                    self.mode = self.Mode.MOVE_TO_DST
                    yield None
                else:
                    yield self.wall_loop.pop(-1) - pos
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")
