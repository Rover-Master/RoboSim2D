# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================
import numpy as np
from . import Simulation, Eager, WallFollowing
from math import pi, ceil
from enum import Enum

from lib.geometry import Point
from lib.util import sign, angDiff


class Bug2(Simulation, WallFollowing):

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
        r1, d1 = v1.angle, v1.norm
        self.r0, self.d0 = r1, d1
        if None in (r0, d0):
            return
        r0, r1 = angDiff(self.r, r0), angDiff(self.r, r1)
        if sign(r0) != sign(r1) and (abs(r0) + abs(r1)) < pi:
            d = max(d0, d1)
            n_steps = ceil(1 + d / self.step_length)
            self.r0, self.d0 = None, None
            return Eager(
                map(
                    lambda x: dst - self.move(self.r, x) - pos,
                    np.linspace(d, 0, n_steps),
                )
            )

    src_pos: Point[float] = None
    n_steps: int = 0

    def step(self, pos, dst):
        if self.r is None:
            self.r = (dst - pos).angle
        if self.n_steps == 0:
            self.src_pos = pos
        self.n_steps += 1
        # Mode switch
        match self.mode:
            case self.Mode.MOVE_TO_DST:
                r_src = (self.src - pos).angle
                if (
                    self.n_steps > 10
                    and abs(angDiff(r_src, self.r)) < pi / 2
                    and (pos - self.src_pos).norm <= 2 * self.step_length
                ):
                    raise Simulation.Abort("dead-loop")
                yield self.move(self.r)
                self.mode = self.Mode.MOVE_ALONG_WALL
                yield self.hit_wall
            case self.Mode.MOVE_ALONG_WALL:
                intersect = self.mLineIntersect(pos, dst)
                # Check for m-line intersection after 3 steps
                if intersect is not None:
                    self.mode = self.Mode.MOVE_TO_DST
                    yield intersect
                # If intersection is not found, continue moving along the wall
                self.mode = self.Mode.MOVE_ALONG_WALL
                yield self.follow_wall
            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")

    def visualize(self, pos, fg, bg=None, **kwargs):
        if bg is None:
            bg = self.vis.view
        src = self.vis.pixel_pos(self.src)
        dst = self.vis.pixel_pos(self.dst)
        self.vis.line(bg, src, dst, (128, 128, 128))
        return super().visualize(pos, fg, bg, **kwargs)
