# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import os, cv2, numpy as np
from yaml import safe_load
from pathlib import Path
from math import ceil

from .util import repeat, readline
from .geometry import Point


class World:
    title: str = "World"

    def __init__(
        self,
        map_name: Path | str,
        resolution: float,
        dpi_scale: float,
        line_width: float,
        threshold: float,  # Distance threshold
        radius: float = None,
        visualize: bool = False,
        debug: bool = False,
    ):
        path = Path(map_name)
        pgm = path.with_suffix(".pgm")
        yaml = path.with_suffix(".yaml")
        self.name = pgm.name
        self.res = resolution
        self.dpi_scale = dpi_scale
        self.threshold = threshold
        self.visualize = visualize
        self.radius = radius
        self.debug = debug
        if not pgm.exists():
            raise FileNotFoundError(f"{pgm} not found")
        if not yaml.exists():
            raise FileNotFoundError(f"{yaml} not found")
        with yaml.open("r") as yaml:
            meta: dict = safe_load(yaml.read())
            scale = meta.get("resolution", 0.05) / resolution
            self.origin = Point(*meta.get("origin", [0, 0])[:2])
            if self.radius is None:
                self.radius = float(meta.get("radius", 0.25))
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
            img = np.frombuffer(buffer, np.uint8, count).reshape((h, w))
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
            self.img = img
            if not negate:
                self.occupancy: np.ndarray[tuple[int, int], bool] = img < threshold
            else:
                self.occupancy: np.ndarray[tuple[int, int], bool] = img > threshold
            # Width and height of the world, in meters
            h, w = self.occupancy.shape
            self.h, self.w = h * self.res, w * self.res
        # Rendering
        dx = int(0.2 / resolution * dpi_scale)
        self.text_offset = Point(dx * 2, dx, type=int)
        self.text_family = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.02 / resolution * dpi_scale
        self.lw = self.px(line_width)

    def px(self, d: float):
        return int(d / self.res * self.dpi_scale) or 1

    def line(
        self,
        img: np.ndarray,
        src: Point[float],
        dst: Point[float],
        color: tuple[int, int, int] | int = 255,
    ):
        cv2.line(img, src, dst, color, self.lw, cv2.LINE_AA)

    def draw_src(self, img: np.ndarray, pos: Point[int], color: tuple[int, int, int]):
        cv2.circle(img, pos, self.px(0.2), color, self.lw)
        cv2.putText(
            img,
            "SRC",
            pos + self.text_offset,
            self.text_family,
            self.text_scale,
            color,
            self.lw,
        )

    def draw_dst(self, img: np.ndarray, pos: Point[int], color: tuple[int, int, int]):
        cv2.drawMarker(img, pos, color, cv2.MARKER_SQUARE, self.px(0.4), self.lw)
        cv2.putText(
            img,
            "DST",
            pos + self.text_offset,
            self.text_family,
            self.text_scale,
            color,
            self.lw,
        )

    @property
    def handle(self):
        return f"{self.title} - {self.name}"

    def show(self, img: np.ndarray):
        if self.dpi_scale is not None:
            s = 1 / self.dpi_scale
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.handle, img)
        return self.handle

    @property
    def view(self) -> np.ndarray:
        """
        Get the image of the world (RGB, uint8, grayscale)
        """
        s = self.dpi_scale
        img = self.img
        if s is not None:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        return np.stack([img] * 3, axis=-1)

    def world_pos(self, p: Point[int]) -> Point[float]:
        """
        Get world location given the img point
        """
        k = self.res
        x0, y0 = self.origin
        return Point(p.x * k + x0, (self.h - p.y * k) + y0, type=float)

    def pixel_pos(self, p: Point[float], scale: float | None = None) -> Point[int]:
        """
        Get img location given the world point
        """
        if scale is None:
            scale = self.dpi_scale

        k = scale / self.res
        x0, y0 = self.origin
        return Point((p.x - x0) * k, (y0 - p.y + self.h) * k, type=int)

    def checkLine(self, src: Point[float], dst: Point[float]) -> bool:
        """
        Check if a direct line between two points is free of obstacles
        The src and dst points are in world coordinate (meters)
        """
        a, b = self.pixel_pos(src, 1.0), self.pixel_pos(dst, 1.0)
        h, w = self.occupancy.shape
        if min(a.x, b.x) < 0 or min(a.y, b.y) < 0:
            return False
        if max(a.x, b.x) >= w or max(a.y, b.y) >= h:
            return False
        mask = np.zeros_like(self.occupancy, dtype=np.uint8)
        w = ceil(self.radius / self.res)
        cv2.line(mask, a, b, 255, w, cv2.LINE_AA)
        mask = mask >= 128
        if self.debug:
            img = np.stack([self.img.copy()] * 3, axis=-1)
            img[mask] = [0, 192, 0]
            img[mask & self.occupancy] = [0, 0, 255]
            cv2.imshow(self.handle + " :: checkLine()", img)
        return not np.any(mask & self.occupancy)
