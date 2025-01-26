# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .Bug0 import Bug0
from math import pi
from numpy import linspace


class Bug0R(Bug0):

    def hit_wall(self, r: float):
        return map(self.move, linspace(r, r - 2 * pi, 180))


if __name__ == "__main__":
    Bug0R.run()
