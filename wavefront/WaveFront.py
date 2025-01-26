# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from lib.arguments import register_arguments, Argument

register_arguments(
    gamma=Argument(
        type=float,
        required=False,
        help="Optional damping coefficient for wavefront. "
        "Use this value to avoid local standing waves.",
    )
)

import numpy as np
from lib.util import sliceOffsets
from . import WaveFront as WF

Deltas = [sliceOffsets(x, y) for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))]


class WaveFront(WF):

    # Damping coefficient, optional
    # Use this term to avoid local standing waves
    gamma: float | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Retrieve argument
        self.gamma = self.world.meta.get("gamma", None)
        # Cache for performance
        # (v * dt) ^ 2 / (dx ^ 2)
        self.k = (self.micro_step / self.world.res) ** 2
        # obstacle mask
        self.f = self.base > 0.5

    def tick(self, u0, du):
        # Laplacian Operator
        L = np.stack([u0] * len(Deltas), axis=0)
        for i, (s0, s1) in enumerate(Deltas):
            free_space = self.f[*s1]
            L[i][*s0][free_space] = u0[*s1][free_space]
        l = np.sum(L, axis=0) - 4 * u0
        if self.gamma is not None:
            du = (1 - self.gamma) * du
        return u0 + du + self.k * l


if __name__ == "__main__":
    WF.run(WaveFront())
