# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc), Adnan Abdullah
# License: MIT
# ==============================================================================

from . import Simulation
from math import pi, isclose
from enum import Enum
from lib.geometry import Point

class Bug2CW(Simulation):

    class Mode(Enum):
        MOVE_TO_DST = 0
        MOVE_ALONG_WALL = 1

    mode: Mode = Mode.MOVE_TO_DST
    hit_point = None
    count = 0

    def is_on_m_line(self,pos,dst):
        """
        Check if the robot is on the M-line.
        """
        if self.hit_point is None:
            return False
        m_line_slope = (dst - self.hit_point).angle - 0.5 * pi
        current_slope = (dst - pos).angle - 0.5 * pi
        return isclose(m_line_slope, current_slope, abs_tol=0.1)

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
                self.hit_point = pos
                self.mode = self.Mode.MOVE_ALONG_WALL
                 # Left turn not viable, try right turn
                yield from self.turn("left")
            case self.Mode.MOVE_ALONG_WALL:
                # Check for m-line intersection after 3 steps
                if self.count > 3 and self.is_on_m_line(pos, dst):
                    self.count = 0
                    self.mode = self.Mode.MOVE_TO_DST
                    # self.hit_point = None
                    yield from self.turn("left")
                else:
                    self.count +=1
                    # First try to turn CCW
                    yield self.turn("right")
                    # Left turn not viable, try right turn
                    yield from self.turn("left")

            case _:
                # Should never reach here
                raise RuntimeError(f"Invalid mode {self.mode}")


if __name__ == "__main__":
    Simulation.run(Bug2CW())
