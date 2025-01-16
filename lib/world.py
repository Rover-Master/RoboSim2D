# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2, numpy as np
from yaml import safe_load
from pathlib import Path
from itertools import permutations
from math import floor, ceil
from typing import Iterable

from .util import repeat, readline
from .geometry import Point


class World:
    def __init__(self, map_name: Path | str, dpi_scale: float | None = None):
        path = Path(map_name)
        pgm = path.with_suffix(".pgm")
        yaml = path.with_suffix(".yaml")
        if not pgm.exists():
            raise FileNotFoundError(f"{pgm} not found")
        if not yaml.exists():
            raise FileNotFoundError(f"{yaml} not found")
        with yaml.open("r") as yaml:
            meta: dict = safe_load(yaml.read())
            self.res: float = meta.get("res", 0.05)
            self.origin = Point(*meta.get("origin", [0, 0])[:2])
            threshold = ceil(255 * (1.0 - meta.get("free_thresh", 0.25)))
            negate = bool(meta.get("negate", False))
        with pgm.open("rb") as pgm:
            stream = repeat(pgm.read, 1)
            assert readline(stream) == "P5\n"
            (w, h) = [int(i) for i in readline(stream).split()]
            depth = int(readline(stream))
            assert depth <= 255, f"Unsupported PGM depth ({depth})"
            count = w * h
            buffer = pgm.read(count)
            img = self.img = np.frombuffer(buffer, np.uint8, count).reshape((h, w))
            if not negate:
                self.occupancy: np.ndarray[tuple[int, int], bool] = img < threshold
            else:
                self.occupancy: np.ndarray[tuple[int, int], bool] = img > threshold
            # Width and height of the world, in meters
            h, w = self.occupancy.shape
            self.h, self.w = h * self.res, w * self.res

    def view(self, scale: float | None = None) -> np.ndarray:
        """
        Get the image of the world (RGB, uint8, grayscale)
        """
        img = self.img
        if scale is not None:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        return np.stack([img] * 3, axis=-1)

    def world_pos(self, p: Point[int], scale: float | None = None) -> Point[float]:
        """
        Get world location given the img point
        """
        k = self.res / scale if scale is not None else self.res
        x0, y0 = self.origin
        return Point(p.x * k + x0, (self.h - p.y * k) + y0, type=float)

    def pixel_pos(self, p: Point[float], scale: float | None = None) -> Point[int]:
        """
        Get img location given the world point
        """
        k = self.res / scale if scale is not None else self.res
        x0, y0 = self.origin
        return Point(int((p.x - x0) / k), int((y0 - p.y + self.h) / k), type=int)

    def checkPoint(self, pts: Iterable[Point[float]]) -> bool:
        """
        Get the type of location at the given point
        The point is in world coordinate (meters)
        True : free space
        False: obstacle, unknown, or out-of-bound
        """
        h, w = self.occupancy.shape
        for p in pts:
            x, y = self.pixel_pos(p)
            for a, b in permutations([floor, ceil], 2):
                i, j = a(x), b(y)
                if 0 <= i < w and 0 <= j < h:
                    if self.occupancy[j, i]:
                        return False
                else:
                    return False
        return True

    def checkLine(self, src: Point[float], dst: Point[float]) -> bool:
        """
        Check if a direct line between two points is free of obstacles
        The src and dst points are in world coordinate (meters)
        """
        l = (dst - src).norm / self.res
        if l < 1.0:
            return self.checkPoint([src, dst])
        steps = ceil(l)
        return self.checkPoint(
            map(
                lambda p: Point(*p, type=float),
                zip(np.linspace(src.x, dst.x, steps), np.linspace(src.y, dst.y, steps)),
            )
        )
