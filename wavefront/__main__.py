# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import cv2, numpy as np
from . import WaveFront

wf = WaveFront()
vis = wf.vis


def scale(f: np.ndarray):
    s = wf.vis.dpi_scale
    return cv2.resize(f / f.max(), None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)


def render(v: int | None = None):
    if v is not None:
        wf.radius = v / 100.0
        wf.__post_init__()
    view = np.clip(wf.view.astype(wf.dtype) / 255.0, 0, 1)
    view = wf.render(view, wf.source_field / wf.source_field.max(), [1.0, 0.0, 0.0])
    view = wf.render(view, wf.drain_field, [0.0, 0.8, 0.0])
    view = (view * 255).astype(np.uint8)
    vis.draw_src(view, vis.pixel_pos(wf.src), (0, 0, 255))
    vis.draw_dst(view, vis.pixel_pos(wf.dst), (0, 0, 255))
    vis.show(view)


render()
cv2.createTrackbar("radius", vis.handle, int(wf.radius * 100.0), 50, render)
while cv2.waitKey(100) < 0:
    pass
cv2.destroyAllWindows()
