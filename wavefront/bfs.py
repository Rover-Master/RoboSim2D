import numpy as np
from lib.util import sliceOffsets
from . import WaveFront

Deltas = [sliceOffsets(x, y) for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))]


class BFS(WaveFront):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cache for performance
        # (v * dt) ^ 2 / (dx ^ 2)
        self.k = (self.velocity * self.time_step) ** 2 / (self.world.res**2)
        print("k:", self.k)
        # obstacle mask
        self.f = self.base > 0.5

    du: np.ndarray | None = None

    def tick(self, u0: np.ndarray, p0: float) -> np.ndarray:
        if self.du is None:
            du = np.zeros_like(u0)
        else:
            du = self.du
        # Laplacian Operator
        L = np.stack([u0] * len(Deltas), axis=0)
        for i, (s0, s1) in enumerate(Deltas):
            L[i][*s0][self.f[*s1]] = u0[*s1][self.f[*s1]]
        l = np.average(L, axis=0) - u0
        u1: np.ndarray = u0 + du + self.k * l
        # Normalize probability field
        # (1) Obstacles are not traversable, constrain to 0
        u1 = np.clip(u1 * self.base, 0.0, 1.0)
        # (2) Overall probability must equal previous iteration
        u1 *= p0 / u1.sum()
        self.du = u1 - u0
        return u1


if __name__ == "__main__":
    WaveFront.run(BFS())
