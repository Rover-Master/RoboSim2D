# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, cv2, numpy as np

from .world import World
from .util import repeat
from .geometry import Point

text_offset = Point(10, 10, type=int)
text_family = cv2.FONT_HERSHEY_SIMPLEX


def draw_src(img: np.ndarray, pos: Point[int], color: tuple[int, int, int]):
    cv2.circle(img, pos, 6, color, 2)
    cv2.putText(
        img,
        "SRC",
        pos + text_offset,
        text_family,
        0.5,
        color,
        2,
    )


def draw_dst(img: np.ndarray, pos: Point[int], color: tuple[int, int, int]):
    cv2.drawMarker(img, pos, color, cv2.MARKER_SQUARE, 16, 2)
    cv2.putText(
        img,
        "DST",
        pos + text_offset,
        text_family,
        0.5,
        color,
        2,
    )


def prompt(
    world: World, scale: float, src_pos: Point | None, dst_pos: Point | None
):
    src_pixel = None if src_pos is None else world.pixel_pos(src_pos, scale)
    dst_pixel = None if dst_pos is None else world.pixel_pos(dst_pos, scale)

    direct_navigable: bool = False
    cursor: Point[int] | None = None

    def render():
        img = world.view(scale)
        if cursor:
            if src_pixel is None:
                draw_src(img, cursor, (192, 192, 192))
            else:
                draw_dst(img, cursor, (192, 192, 192))
        if src_pixel:
            draw_src(img, src_pixel, (255, 0, 0))
        if dst_pixel:
            draw_dst(img, dst_pixel, (0, 192, 0))
        # Gray preview line
        if cursor and src_pixel:
            cv2.line(img, src_pixel, cursor, (128, 128, 128), 1, cv2.LINE_AA)
        # Direct connection line
        if src_pixel and dst_pixel:
            color = (0, 192, 0) if direct_navigable else (0, 0, 255)
            cv2.line(img, src_pixel, dst_pixel, color, 1, cv2.LINE_AA)
        cv2.imshow("Map", img)

    def onMouse(event, x, y, flags, _: None):
        nonlocal src_pixel, src_pos, dst_pixel, dst_pos, cursor, direct_navigable
        if event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            if src_pixel is None:
                src_pixel = Point(x, y, type=int)
                src_pos = world.world_pos(src_pixel, scale)
            else:
                dst_pixel = Point(x, y, type=int)
                dst_pos = world.world_pos(dst_pixel, scale)
                direct_navigable = world.checkLine(src_pos, dst_pos)
        cursor = Point(x, y, type=int)
        render()

    render()
    cv2.setMouseCallback("Map", onMouse)

    for keycode in repeat(cv2.waitKey, 10):
        match keycode:
            case 8 | 127:  # Backspace
                if dst_pixel is not None:
                    dst_pixel = None
                elif src_pixel is not None:
                    src_pixel = None
                render()
            case 13:  # Enter
                if src_pixel is not None and dst_pixel is not None:
                    break
            case _:
                if keycode == ord("q") or keycode == 27:
                    sys.exit(0)
                elif keycode >= 0:
                    print(f"Unknown keycode: {keycode}")
    cv2.setMouseCallback("Map", lambda *args: None)
    return src_pos, dst_pos
