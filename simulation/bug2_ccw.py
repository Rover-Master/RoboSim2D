# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
import numpy as np

from . import Simulation, Eager, Lazy
from math import pi, ceil
from enum import Enum
from lib.geometry import Point
from lib.util import sign, angDiff


class Bug2CCW(Simulation):

    class Mode(Enum):
        MOVE_TO_DST = 0
        MOVE_ALONG_WALL = 1

    mode: Mode = Mode.MOVE_TO_DST

    r: float | None = None  # Desired heading towards the target

    v0: Point[float] | None = None  # Previous delta vector
    r0: float | None = None  # Previous target direction
    d0: float | None = None  # Previous target distance

    def mLineIntersect(self, pos: Point[float], dst: Point[float]):
        v1 = dst - pos
        r0, d0 = self.r0, self.d0
        r1, d1 = v1.angle - 0.5 * np.pi, v1.norm
        self.r0, self.d0 = r1, d1
        if None in (r0, d0):
            return
        r0, r1 = angDiff(self.r, r0), angDiff(self.r, r1)
        if sign(r0) != sign(r1) and (abs(r0) + abs(r1)) < pi:
            d = max(d0, d1)
            n_steps = ceil(1 + d / self.step_length)
            self.r0, self.d0 = None, None
            return map(
                lambda x: dst - self.move(self.r, x) - pos, np.linspace(d, 0, n_steps)
            )

    def step(self, pos, dst):
        if self.r is None:
            self.r = (dst - pos).angle - 0.5 * pi
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                yield self.move(self.r)
                self.mode = self.Mode.MOVE_ALONG_WALL
                yield Eager(self.turn("right"))
            case self.Mode.MOVE_ALONG_WALL:
                intersect = self.mLineIntersect(pos, dst)
                # Check for m-line intersection after 3 steps
                if intersect is not None:
                    self.mode = self.Mode.MOVE_TO_DST
                    yield Eager(intersect)
                else:
                    yield (
                        # First try to turn CCW
                        Lazy(self.turn("left")),
                        # Left turn not viable, try right turn
                        Eager(self.turn("right")),
                    )
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")

    def visualize(self, pos, fg, bg=None, **kwargs):
        import cv2

        world = self.world
        if bg is None:
            bg = world.view
        cv2.line(
            bg,
            world.pixel_pos(self.src),
            world.pixel_pos(self.dst),
            (128, 128, 128),
            world.line_width,
            cv2.LINE_AA,
        )
        return super().visualize(pos, fg, bg, **kwargs)


if __name__ == "__main__":
    Simulation.run(Bug2CCW())
