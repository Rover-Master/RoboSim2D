# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .arguments import register_arguments, Argument
from pathlib import Path

register_arguments(
    map_name=Argument(
        "map_name",
        opt_name=None,
        type=Path,
        nargs=1,
        help="Path to the world map (without extension)",
    ),
    res=Argument(
        "-R",
        opt_name="resolution",
        type=float,
        default=None,
        help="World pixel resolution in meters (m/px)",
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
import sys, cv2, numpy as np
from yaml import safe_load
from math import ceil
from dataclasses import dataclass, field

from .util import repeat, readline, own_attrs
from .geometry import Point


@own_attrs
@dataclass(frozen=False)
class World:
    # Snapshot of ALL arguments (not just the ones used in __init__)
    meta: dict

    debug: bool
    map_name: Path
    res: float
    # Raw image - read as-is
    raw_img: np.ndarray = field(init=False)
    raw_shape: tuple[int, int] = field(init=False)
    # Scaled to match desired pixel resolution
    origin: Point[float] = field(init=False)
    img: np.ndarray = field(init=False)
    occupancy: np.ndarray[tuple[int, int], bool] = field(init=False)

    @staticmethod
    def create(*, map_name: str, **kwargs):
        meta = kwargs
        world = World(**kwargs, meta=meta, map_name=map_name[0])
        meta["world"] = world
        return meta

    def __post_init__(self):
        path = Path(self.map_name)
        png = path.with_suffix(".png")
        pgm = path.with_suffix(".pgm")
        yaml = path.with_suffix(".yaml")
        if not yaml.exists():
            raise FileNotFoundError(f"{yaml} not found")
        with yaml.open("r") as yaml:
            desc: dict = safe_load(yaml.read())
            raw_res = desc.get("resolution", self.res)
            if raw_res is None:
                raise ValueError(
                    "No resolution specified in the map description or as an argument"
                )
            if self.res is None:
                self.res = raw_res
            scale = raw_res / self.res
            self.origin = Point(*desc.get("origin", [0, 0])[:2])
            threshold = ceil(255 * (1.0 - desc.get("free_thresh", 0.25)))
            negate = bool(desc.get("negate", False))
        if png.is_file():
            self.raw_img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        elif pgm.is_file():
            self.raw_img = loadPGM(pgm)
        else:
            raise FileNotFoundError(f"{png} or {pgm} not found")
        self.raw_shape = self.raw_img.shape
        # Resize image to match desired pixel resolution
        self.img = cv2.resize(
            self.raw_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        if not negate:
            self.occupancy: np.ndarray[tuple[int, int], bool] = self.img < threshold
        else:
            self.occupancy: np.ndarray[tuple[int, int], bool] = self.img > threshold
        # Width and height of the world, in meters
        h, w = self.occupancy.shape
        self.h, self.w = h * self.res, w * self.res
        # Rendering
        self.text_family = cv2.FONT_HERSHEY_DUPLEX

    @property
    def name(self):
        return self.map_name.name

    def world_pos(self, p: Point[int], scale: float = 1.0) -> Point[float]:
        """
        Get world location given the pixel coordinates on the view
        """
        k = self.res / scale
        x0, y0 = self.origin
        return Point(p.x * k + x0, (self.h - p.y * k) + y0, type=float)

    def pixel_pos(self, p: Point[float], scale: float = 1.0) -> Point[int]:
        """
        Get location on the view given the world coordinates in meters
        """
        k = scale / self.res
        x0, y0 = self.origin
        return Point((p.x - x0) * k, (y0 - p.y + self.h) * k, type=int)

    def checkLine(self, src: Point[float], dst: Point[float], radius: float) -> bool:
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
        w = ceil(radius / self.res)
        cv2.line(mask, a, b, 255, w, cv2.LINE_AA)
        mask = mask >= 128
        if self.debug:
            img = np.stack([self.img.copy()] * 3, axis=-1)
            img[mask] = [0, 192, 0]
            img[mask & self.occupancy] = [0, 0, 255]
            scale = self.meta.get("scale", None)
            if scale is not None:
                img = cv2.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
                )
            cv2.imshow(self.name + " :: DEBUG", img)
        return not np.any(mask & self.occupancy)


def loadPGM(path: Path):
    with path.open("rb") as pgm:
        stream = repeat(pgm.read, 1)
        assert readline(stream) == "P5\n"
        (w, h) = [int(i) for i in readline(stream).split()]
        depth = int(readline(stream))
        assert depth <= 255, f"Unsupported PGM depth ({depth})"
        count = w * h
        buffer = pgm.read(count)
        return np.frombuffer(buffer, np.uint8, count).reshape((h, w))
