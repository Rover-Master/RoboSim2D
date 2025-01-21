# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2, numpy as np
from . import WaveFront

wf = WaveFront()

def scale(f: np.ndarray):
    s = wf.vis.dpi_scale
    return cv2.resize(f / f.max(), None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)


def render(v: int | None = None):
    if v is not None:
        wf.radius = v / 50.0
    view = wf.view.astype(wf.dtype) / 255.0
    view = wf.render(view, wf.source_field, [1.0, 0.0, 0.0])
    view = wf.render(view, wf.drain_field, [0.0, 0.8, 0.0])
    view = (view * 255).astype(np.uint8)
    world.draw_src(view, world.pixel_pos(wf.src), (0, 0, 255))
    world.draw_dst(view, world.pixel_pos(wf.dst), (0, 0, 255))
    world.show(view)


render()
cv2.createTrackbar("radius", world.handle, int(world.radius * 50.0), 50, render)
while cv2.waitKey(100) < 0:
    pass
cv2.destroyAllWindows()
