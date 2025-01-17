# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, cv2

from .world import World
from .util import repeat
from .geometry import Point


def prompt(world: World, src_pos: Point | None, dst_pos: Point | None):
    src_pixel = None if src_pos is None else world.pixel_pos(src_pos)
    dst_pixel = None if dst_pos is None else world.pixel_pos(dst_pos)

    direct_navigable: bool = False
    cursor: Point[int] | None = None

    def render():
        img = world.view
        if cursor:
            if src_pixel is None:
                world.draw_src(img, cursor, (192, 192, 192))
            else:
                world.draw_dst(img, cursor, (192, 192, 192))
        if src_pixel:
            world.draw_src(img, src_pixel, (255, 0, 0))
        if dst_pixel:
            world.draw_dst(img, dst_pixel, (0, 192, 0))
        # Gray preview line
        if cursor and src_pixel:
            world.line(img, src_pixel, cursor, (128, 128, 128))
        # Direct connection line
        if src_pixel and dst_pixel:
            color = (0, 192, 0) if direct_navigable else (0, 0, 255)
            world.line(img, src_pixel, dst_pixel, color)
        return world.show(img)

    def onMouse(event, x, y, flags, _: None):
        nonlocal src_pixel, src_pos, dst_pixel, dst_pos, cursor, direct_navigable
        p = Point(x, y, type=int)
        cursor = p * world.dpi_scale
        if event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            if src_pixel is None:
                src_pixel = cursor
                src_pos = world.world_pos(p)
            else:
                dst_pixel = cursor
                dst_pos = world.world_pos(p)
                direct_navigable = world.checkLine(src_pos, dst_pos)
        render()

    handle = render()
    cv2.setMouseCallback(handle, onMouse)

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
    cv2.setMouseCallback(handle, lambda *args: None)
    return src_pos, dst_pos
