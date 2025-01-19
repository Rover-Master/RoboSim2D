# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, cv2, numpy as np
from yaml import safe_load
from pathlib import Path
from math import ceil

from .util import repeat, readline
from .geometry import Point


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


class World:
    def __init__(
        self,
        map_name: Path | str,
        *,
        resolution: float,
        dpi_scale: float,
        line_width: float,
        threshold: float,  # Threshold distance near the target
        max_travel: float | None,  # Max travel distance before aborting the simulation
        radius: float = None,
        prefix: str | None,
        slice: tuple[int, int, int, int] | None = None,
        visualize: bool = False,
        no_wait: bool = False,
        debug: bool = False,
    ):
        path = Path(map_name)
        png = path.with_suffix(".png")
        pgm = path.with_suffix(".pgm")
        yaml = path.with_suffix(".yaml")
        self.name = pgm.name
        self.dpi_scale = dpi_scale
        self.line_width_meters = line_width
        self.threshold = threshold
        self.max_travel = max_travel
        self.visualize = visualize
        self._slice = slice
        self.no_wait = no_wait
        self.radius = radius
        self.debug = debug
        if prefix is None:
            self.path_prefix = None
            self.file_prefix = None
        elif prefix.endswith("/"):
            self.path_prefix = Path(prefix)
            self.file_prefix = ""
        else:
            self.path_prefix = Path(prefix).parent
            self.file_prefix = Path(prefix).name
        if self.path_prefix:
            if not self.path_prefix.exists():
                self.path_prefix.mkdir(parents=True, exist_ok=True)
            elif self.path_prefix.is_file():
                raise FileExistsError(f"Prefix {self.path_prefix} is a file")
        if not yaml.exists():
            raise FileNotFoundError(f"{yaml} not found")
        with yaml.open("r") as yaml:
            meta: dict = safe_load(yaml.read())
            raw_res = meta.get("resolution", 0.05)
            if resolution is None:
                self.res = raw_res
                scale = 1.0
            else:
                self.res = resolution
                scale = raw_res / resolution
            self.origin = Point(*meta.get("origin", [0, 0])[:2])
            if self.radius is None:
                self.radius = float(meta.get("radius", 0.30))
            threshold = ceil(255 * (1.0 - meta.get("free_thresh", 0.25)))
            negate = bool(meta.get("negate", False))
        if png.is_file():
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        elif pgm.is_file():
            img = loadPGM(pgm)
        else:
            raise FileNotFoundError(f"{png} or {pgm} not found")
        self.raw_shape = img.shape
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        self.img = img
        if not negate:
            self.occupancy: np.ndarray[tuple[int, int], bool] = img < threshold
        else:
            self.occupancy: np.ndarray[tuple[int, int], bool] = img > threshold
        # Width and height of the world, in meters
        h, w = self.occupancy.shape
        self.h, self.w = h * self.res, w * self.res
        # Rendering
        self.text_family = cv2.FONT_HERSHEY_DUPLEX

    def slice(self, img: np.ndarray):
        if self._slice is None:
            return img
        x, y, w, h = self._slice
        h0, w0 = self.raw_shape
        h1, w1 = img.shape[:2]
        fx, fy = w1 / w0, h1 / h0
        _x, w = int(x * fx), int(w * fx)
        _y, h = int(y * fy), int(h * fy)
        x = max(int(x * fx), 0)
        y = max(int(y * fy), 0)
        if len(img.shape) == 2:
            return img[y : _y + h, x : _x + w]
        elif len(img.shape) == 3:
            return img[y : _y + h, x : _x + w, :]
        else:
            raise ValueError(f"Unsupported image shape {img.shape}")

    def saveImg(self, path: Path | str, img: np.ndarray):
        cv2.imwrite(str(path), self.slice(img))

    @property
    def marker_label_offset(self):
        dx = int(0.2 / self.res * self.dpi_scale)
        return Point(dx * 2, dx, type=int)

    @property
    def text_scale(self):
        return 0.02 / self.res * self.dpi_scale

    @property
    def line_width(self):
        return self.px(self.line_width_meters)

    def withPrefix(self, stem: str | None = None, suffix: str | None = None):
        if self.path_prefix is None or self.file_prefix is None:
            return None
        if self.file_prefix and stem:
            name = "-".join((self.file_prefix, stem))
        elif self.file_prefix or stem:
            name = self.file_prefix or stem
        else:
            name = self.path_prefix.name
        if suffix:
            if suffix.startswith("."):
                raise ValueError("suffix should not start with '.'")
            name = f"{name}.{suffix}"
        return self.path_prefix / name

    def px(self, d: float):
        return int(d / self.res * self.dpi_scale) or 1

    def line(
        self,
        img: np.ndarray,
        src: Point[float],
        dst: Point[float],
        color: tuple[int, int, int] | int = 255,
    ):
        cv2.line(img, src, dst, color, self.line_width, cv2.LINE_AA)

    def draw_src(self, img: np.ndarray, pos: Point[int], color=(255, 0, 0)):
        cv2.circle(img, pos, self.px(0.2), color, self.line_width)
        cv2.putText(
            img,
            "SRC",
            pos + self.marker_label_offset,
            self.text_family,
            self.text_scale,
            color,
            self.line_width,
        )

    def draw_dst(self, img: np.ndarray, pos: Point[int], color=(0, 192, 0)):
        cv2.drawMarker(
            img, pos, color, cv2.MARKER_SQUARE, self.px(0.4), self.line_width
        )
        cv2.putText(
            img,
            "DST",
            pos + self.marker_label_offset,
            self.text_family,
            self.text_scale,
            color,
            self.line_width,
        )

    def caption(
        self,
        img: np.ndarray,
        text: str,
        fg=(128, 128, 128),
        bg: tuple[int, int, int] | None = None,
    ):
        a = self.px(0.4)
        args = dict(
            img=img,
            text=text,
            org=(a, img.shape[0] - a),
            fontFace=self.text_family,
            fontScale=self.text_scale,
            lineType=cv2.LINE_AA,
        )
        if bg is not None:
            cv2.putText(**args, color=bg, thickness=self.line_width * 3)
        cv2.putText(**args, color=fg, thickness=self.line_width)

    @property
    def handle(self):
        return f"{self.title} - {self.name}"

    def show(self, img: np.ndarray, desc: str | None = None, scale=None):
        if scale is not None:
            s = 1 / scale
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        elif self.dpi_scale is not None:
            s = 1 / self.dpi_scale
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if desc is None:
            title = self.handle
        else:
            title = f"{self.handle} ({desc})"
        cv2.imshow(title, img)
        return title

    @property
    def grayscale(self) -> np.ndarray:
        """
        Get the image of the world (RGB, uint8, grayscale)
        """
        s = self.dpi_scale
        img = self.img
        if s is not None:
            return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        else:
            return img.copy()

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


if __name__ == "__main__":
    from .arguments import parse

    name, params, args = parse()
    world = World(str(name), **params)
    img_path = world.withPrefix(suffix="png")
    if world.visualize:
        s = world.dpi_scale

        x, y, w, h = world._slice or (0, 0, *world.raw_shape[::-1])

        def render():
            view = world.grayscale.astype(np.float32) / 255.0
            mask = np.ones_like(view) * 0.5
            mask_slice = world.slice(mask)
            mask_slice[:, :] = 1.0
            view = np.stack([view * mask, view * mask, view], axis=-1)
            caption = ", ".join([str(_).ljust(3) for _ in (x, y, w, h)])
            world.caption(view, caption, fg=(0, 0, 0), bg=[255] * 3)
            return world.show(view, scale=1.0)

        drag_origin: Point[int] | None = None

        def onMouse(event, cx, cy, flags, _: None):
            global x, y, w, h, drag_origin
            cx = int(cx / s)
            cy = int(cy / s)
            if event == cv2.EVENT_LBUTTONDOWN:
                x, y = cx, cy
                w, h = 1, 1
            elif event == cv2.EVENT_MBUTTONDOWN:
                drag_origin = Point(cx, cy)
            elif event == cv2.EVENT_MOUSEMOVE:
                if flags == cv2.EVENT_FLAG_LBUTTON:
                    w = max(1, cx - x)
                    h = max(1, cy - y)
                if flags == cv2.EVENT_FLAG_MBUTTON:
                    if drag_origin is not None:
                        dx, dy = cx - drag_origin.x, cy - drag_origin.y
                        x, y = x + dx, y + dy
                    drag_origin = Point(cx, cy)
            else:
                return
            world._slice = (x, y, w, h)
            render()

        handle = render()
        cv2.setMouseCallback(handle, onMouse)
        while cv2.waitKey(1) < 0:
            pass
    if world._slice is not None:
        print(
            "--slice=", ",".join(str(_) for _ in world._slice), sep="", file=sys.stderr
        )
    if img_path is not None:
        world.saveImg(img_path, world.grayscale)
