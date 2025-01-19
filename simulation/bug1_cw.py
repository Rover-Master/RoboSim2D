# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
from . import Simulation
from math import pi
from numpy import argmin
from enum import Enum

from lib.geometry import Point


class Bug1CW(Simulation):

    class Mode(Enum):
        MOVE_TO_DST = 0
        MOVE_ALONG_WALL = 1
        MOVE_TO_CLOSEST_POINT = 2

    mode: Mode = Mode.MOVE_TO_DST
    wall_loop: list[Point] = []

    @property
    def loop_closed(self):
        if len(self.wall_loop) < 10:
            return False
        return (self.wall_loop[0] - self.wall_loop[-1]).norm < self.step_length * 2

    def step(self, pos, dst):
        print(f"Mode: {self.mode.name}")
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                # The target direction
                r1 = (dst - pos).angle - 0.5 * pi
                yield self.move(r1)
                # Hit the wall
                print(f"Hit the wall at {pos}")
                self.mode = self.Mode.MOVE_ALONG_WALL
                self.wall_loop = [pos]
                # Left turn not viable, try right turn
                yield from self.turn("left")
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
                    yield from self.turn("left")
                else:
                    # First try to turn CCW
                    yield self.turn("right")
                    # Left turn not viable, try right turn
                    yield from self.turn("left")
            case self.Mode.MOVE_TO_CLOSEST_POINT:
                if len(self.wall_loop) == 0:
                    self.mode = self.Mode.MOVE_TO_DST
                    yield from self.step(pos, dst)
                else:
                    yield self.wall_loop.pop(-1) - pos
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")


if __name__ == "__main__":
    Simulation.run(Bug1CW())
