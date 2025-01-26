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
        required=False,
        help="World pixel resolution in meters (m/px)",
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
import cv2, numpy as np
from yaml import safe_load
from math import ceil
from dataclasses import dataclass, field

from .util import repeat, readline, ownAttributes, sliceOffsets
from .geometry import Point


@ownAttributes
@dataclass(frozen=False)
class World:
    # Snapshot of ALL arguments (not just the ones used in __init__)
    meta: dict = field(default_factory=dict)

    debug: bool = False
    map_name: Path | None = None
    res: float | None = None
    # Flag to indicate post_init is done
    initialized: bool = False
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

    @property
    def clone(self):
        return World(**self.__dict__)

    def initialize(
        self,
        raw_img: np.ndarray,
        *,
        origin: Point[float],
        threshold: int = 128,
        scale: float | None = None,
        negate: bool = False,
    ):
        self.initialized = True
        self.origin = origin
        self.raw_img = raw_img
        self.raw_shape = raw_img.shape
        # Resize image to match desired pixel resolution
        img = raw_img
        if scale is not None:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
        self.img = img
        # Occupancy grid - boolean array indicating free space
        if not negate:
            occupancy: np.ndarray[tuple[int, int], bool] = img < threshold
        else:
            occupancy: np.ndarray[tuple[int, int], bool] = img > threshold
        self.occupancy = occupancy
        # Width and height of the world, in meters
        h, w = occupancy.shape
        self.h, self.w = h * self.res, w * self.res

    def __post_init__(self):
        self.meta["world"] = self
        if self.initialized:
            return
        # Initialize the world - load files according to it's name
        path = Path(self.map_name)
        png = path.with_suffix(".png")
        pgm = path.with_suffix(".pgm")
        yaml = path.with_suffix(".yaml")
        # Load the map description
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
            origin = Point(*desc.get("origin", [0, 0])[:2])
            threshold = ceil(255 * (1.0 - desc.get("free_thresh", 0.25)))
            negate = bool(desc.get("negate", False))
        # Load the map image
        if png.is_file():
            raw_img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        elif pgm.is_file():
            raw_img = loadPGM(pgm)
        else:
            raise FileNotFoundError(f"{png} or {pgm} not found")
        self.initialize(
            raw_img, origin=origin, threshold=threshold, scale=scale, negate=negate
        )

    @property
    def name(self):
        if self.map_name is not None:
            return self.map_name.name
        else:
            return "Unnamed World"

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

    def trajectory(
        self,
        src: Point[float],
        dst: Point[float],
        radius: float,
        *,
        check_boundary: bool = False,
    ) -> bool:
        """
        Generate a boolean occupancy map for a trajectory between two points
        """
        a, b = self.pixel_pos(src), self.pixel_pos(dst)
        if check_boundary:
            h, w = self.occupancy.shape
            if min(a.x, b.x) < 0 or min(a.y, b.y) < 0:
                return
            if max(a.x, b.x) >= w or max(a.y, b.y) >= h:
                return
        mask = np.zeros_like(self.occupancy, dtype=np.uint8)
        w = max(1, int(2 * radius / self.res + 0.5))
        cv2.line(mask, a, b, 255, w, cv2.LINE_AA)
        return mask >= 128

    def checkLine(self, src: Point[float], dst: Point[float], radius: float) -> bool:
        """
        Check if a direct line between two points is free of obstacles
        The src and dst points are in world coordinate (meters)
        """
        mask = self.trajectory(src, dst, radius, check_boundary=True)
        if mask is None:
            return False
        if self.debug:
            img = np.stack([self.img.copy()] * 3, axis=-1)
            img[mask] = [0, 192, 0]
            img[mask & self.occupancy] = [0, 0, 255]
            scale = self.meta.get("view_scale", None)
            if scale is not None:
                img = cv2.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
                )
            cv2.imshow(self.name + " :: DEBUG", img)
        return not np.any(mask & self.occupancy)

    def padMap(self, radius: float) -> np.ndarray[tuple[int, int], bool]:
        """
        Pad the occupancy grid with a given radius
        """
        o = self.occupancy
        m = o.copy()
        r = ceil(radius / self.res)
        k = np.zeros([2 * r + 1] * 2, dtype=np.uint8)
        cv2.circle(k, (r, r), r, 255, -1, cv2.FILLED)
        k[r, r] = 0
        s = slice(-r, r + 1)
        idx = np.mgrid[s, s].swapaxes(0, 2)
        # convolute
        for x, y in idx[k > 128]:
            s0, s1 = sliceOffsets(x, y)
            m[*s0] |= o[*s1]
        return m


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
