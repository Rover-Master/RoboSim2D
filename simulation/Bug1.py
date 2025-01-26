# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from lib.arguments import register_arguments, Argument

register_arguments(
    no_overlap=Argument(
        action="store_true",
        help="Add padding when moving back the loop",
    )
)

from . import Simulation, WallFollowing, OffsetPath
from math import pi
from numpy import argmin
from enum import Enum
from typing import Iterator

from lib.geometry import Point


class Bug1(Simulation, WallFollowing):

    @property
    def no_overlap(self):
        return self.world.meta.get("no_overlap", False)

    class Mode(Enum):
        MOVE_TO_DST = 0
        MOVE_ALONG_WALL = 1
        MOVE_TO_CLOSEST_POINT = 2

    mode: Mode = Mode.MOVE_TO_DST
    wall_loop: list[Point] = []
    loop_back: Iterator[Point]

    @property
    def loop_closed(self):
        p1 = self.wall_loop[-1]
        cnt = int(10 * self.radius / self.step_length)
        cnt = max(cnt, 10)
        if len(self.wall_loop) < cnt:
            return False
        p0 = self.wall_loop[0]
        return (p0 - p1).norm < self.step_length * 2

    @property
    def padding_offset(self):
        k = -1 if self.wall_following_direction == "L" else 1
        return 2 * self.radius * k

    def pad_loop(self, dp: Point) -> Point:
        raise NotImplementedError

    def step(self, pos, dst):
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                self.check = True
                # The target direction
                r1 = (dst - pos).angle
                yield self.move(r1)
                # Hit the wall
                self.mode = self.Mode.MOVE_ALONG_WALL
                self.wall_loop = [pos]
                # Left turn not viable, try right turn
                yield self.hit_wall
            case self.Mode.MOVE_ALONG_WALL:
                self.check = True
                # Check for loop closure
                if self.loop_closed:
                    distances = [(dst - p).norm for p in self.wall_loop]
                    idx = int(argmin(distances))
                    trj = self.wall_loop[-1:idx:-1]
                    if len(trj) > 1:
                        self.mode = self.Mode.MOVE_TO_CLOSEST_POINT
                        if self.no_overlap:
                            trj = OffsetPath(
                                trj,
                                self.padding_offset,
                                step_length=self.step_length,
                                resolution=self.world.res,
                            )
                        self.loop_back = iter(trj)
                    else:
                        self.mode = self.Mode.MOVE_TO_DST
                    yield None
                else:
                    self.wall_loop.append(pos)
                    yield self.follow_wall
            case self.Mode.MOVE_TO_CLOSEST_POINT:
                self.check = False
                try:
                    with self.no_check:
                        yield next(self.loop_back) - pos
                except StopIteration:
                    self.mode = self.Mode.MOVE_TO_DST
                    yield None
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")
