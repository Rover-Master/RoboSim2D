# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import numpy as np, cv2
from lib.geometry import Point
from math import ceil
from lib.util import dup, sliceOffsets


class WaveFront:
    dtype = np.float32
    # Wave propagation speed in meters per second
    velocity: float = 1.0
    # Time step in seconds
    time_step: float = 0.01
    # Termination threshold - the remaining probability
    vanish_threshold: float = 0.0001

    def __init__(self, **kwargs):
        from lib.env import world, src_pos, dst_pos

        self.world = world
        self.src = src_pos
        self.dst = dst_pos
        self.__dict__.update(kwargs)
        # background
        o = world.occupancy
        r = ceil(world.radius / world.res)
        # kernel
        k = np.zeros([2 * r + 1] * 2, dtype=np.uint8)
        cv2.circle(k, (r, r), r, 255, -1, cv2.FILLED)
        k[r, r] = 0
        # prepare for convolution
        h, w = o.shape
        m = o.copy()
        s = slice(-r, r + 1)
        idx = np.mgrid[s, s].swapaxes(0, 2)
        # convolute
        for x, y in idx[k > 128]:
            s0, s1 = sliceOffsets(x, y)
            m[*s0] |= o[*s1]
        # The base field, 1.0 for free space, 0.0 for obstacles
        # `field * base` filters out probability distributions in obstacles
        self.base = 1.0 - m.astype(self.dtype)
        # Discrete cartesian coordinates (per-cell)
        Y, X = np.mgrid[:h, :w].astype(self.dtype) * world.res
        ox, oy = world.world_pos(Point(0, 0))
        self.X = +X + ox
        self.Y = -Y + oy
        # source_field:
        #   A gaussian distribution around the source.
        #   It's sigma equals the radius of the robot.
        self.source_field = self.norm(self.gaussian(src_pos, world.radius / 2))
        # drain_field:
        #   An inverse gaussian distribution around the destination
        #   It's sigma equals the threshold distance.
        self.drain_field = self.gaussian(dst_pos, world.threshold / 2)

    @property
    def mask(self):
        # rendering properties and assets
        r = self.world.dpi_scale
        return (
            cv2.resize(self.base, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
            > 0.5
        )

    @property
    def view(self):
        diff = (self.base <= 0.5) ^ self.world.occupancy
        r = self.world.dpi_scale
        diff = (
            cv2.resize(diff * 155, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
            >= 128
        )
        view = self.world.view
        view[diff, :] = 192
        return view

    def norm(self, field: np.ndarray) -> np.ndarray:
        f = np.maximum(field * self.base, 0.0).astype(self.dtype)
        return (f / f.sum()).astype(self.dtype)

    def gaussian(self, origin: Point[float], sigma) -> np.ndarray:
        X, Y = self.X - origin.x, self.Y - origin.y
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(self.dtype)

    def render(
        self,
        bg: np.ndarray,
        field: np.ndarray,
        color: list[float] = [0.0, 0.0, 1.0],
        invert=False,
    ) -> np.ndarray:
        # Requires float array of range [0.0, 1.0]
        if bg.max() > 1.0:
            raise ValueError("Background must be a float array of range [0.0, 1.0]")
        s = self.world.dpi_scale
        if invert:
            field = 1.0 - field
        field = np.clip(field, 0.0, 1.0)
        field = cv2.resize(field, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        field = np.stack([field] * 3, axis=-1)
        f0, f1 = 1.0 - field, field
        fg = np.ones_like(bg) * color
        return bg * f0 + fg * f1

    def heatmap(
        self,
        field: np.ndarray,
        c0: list[float] = [0.0, 0.0, 1.0],
        c1: list[float] = [1.0, 0.5, 0.0],
        invert: bool = False,
    ) -> np.ndarray:
        # Requires float array of range [0.0, 1.0]
        s = self.world.dpi_scale
        if invert:
            field = 1.0 - field
        field = np.clip(field, 0.0, 1.0)
        field = cv2.resize(field, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        field = np.stack([field] * 3, axis=-1)
        f0, f1 = 1.0 - field, field
        bg = np.ones_like(field) * c1
        fg = np.ones_like(field) * c0
        return bg * f0 + fg * f1

    # GPU Acceleration
    def tick(self, u0: np.ndarray, p0: float, du: np.ndarray) -> np.ndarray:
        """
        input: u0 - the prob. field at time t_{n-1}
        output: u1 - the prob. field at time t_{n}
        """
        raise NotImplementedError

    class Session:

        started: bool = False
        finished: bool = False

        def __init__(self, wf: "WaveFront"):
            self.wf = wf
            self.u0: np.ndarray = wf.source_field.copy()
            self.p0: float = self.u0.sum()
            self.du = np.zeros_like(self.u0)

        def __next__(self):
            if self.finished:
                raise StopIteration
            if self.started:
                # Apply next tick
                u1 = self.wf.tick(self.u0, self.p0, self.du)
                # Normalize probability field
                # (1) Obstacles are not traversable, constrain to 0
                u1 = np.clip(u1 * self.wf.base, 0.0, 1.0)
                # (2) Overall probability must equal previous iteration
                u1 *= self.p0 / u1.sum()
                # Compute delta
                du = u1 - self.u0
                # Apply drain
                drain: np.ndarray = u1 * self.wf.drain_field
                dp = float(drain.sum())
                p1 = self.p0 - dp
                u1 -= drain
                # Termination conditions
                if p1 < 0:
                    raise StopIteration
                if p1 < self.wf.vanish_threshold:
                    self.finished = True
                # Prepare for next rick
                self.u0, self.p0, self.du = u1, p1, du
            else:
                self.started = True
                dp = 0.0
            return self.u0, self.p0, dp

    def __iter__(self):
        return WaveFront.Session(self)

    @staticmethod
    def run(wf: "WaveFront"):
        name = wf.__class__.__name__
        out_list = wf.world.withPrefix(name, suffix="txt")
        out_img = wf.world.withPrefix(name, suffix="png")
        if out_list is not None:
            trj_list_file = open(out_list, "w")
            print = dup(trj_list_file)

        U = np.zeros_like(wf.base)
        P = list[float]()
        T = list[float]()
        t = 0.0
        dt = wf.velocity * wf.time_step

        if wf.world.visualize:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            (plot,) = ax.plot([], [], "r-", lw=2)
            last_vis_t = 0.0
            dp_max = 0.0
            bg = wf.view.astype(wf.dtype) / 255.0

            def visualize(u: np.ndarray, p: float, dp: float):
                nonlocal plot, dp_max
                dp_max = max(dp_max, dp)
                m = wf.render(bg, u / u.max(), color=[0.0, 0.0, 1.0])
                m = (m * 255.0).astype(np.uint8)
                wf.world.draw_src(m, wf.world.pixel_pos(wf.src))
                wf.world.draw_dst(m, wf.world.pixel_pos(wf.dst))
                caption = f"Travel {t:.2f}m | Progress {(1.0 - p) * 100.0:.2f}%"
                wf.world.caption(m, caption, fg=(0, 0, 0), bg=(255, 255, 255))
                wf.world.show(m)
                # T-P Plot
                ax.set_xlim(0, max(t, 1.0))
                ax.set_ylim(0, max(dp_max, 1e-10))
                plot.remove()
                (plot,) = ax.plot(T, P, "r-", lw=2)
                fig.show()

        n = 0
        for n, (u, p, dp) in enumerate(wf):
            U += u
            print(t, p, dp, sep=", ")
            P.append(dp)
            T.append(t)
            t += dt
            # Visualization
            if wf.world.visualize and t - last_vis_t > 0.1:
                last_vis_t = t
                visualize(u, p, dp)
                if cv2.waitKey(1) > 0:
                    break
        x = np.array(T)
        y = np.array(P)
        print(f"# coverage = {y.sum():.4f}")
        y /= y.sum()
        mean = np.sum(x * y)
        print(f"# travel   = {mean:.4f}")
        variance = np.sum((x - mean) ** 2 * y)
        std = np.sqrt(variance)
        print(f"# std      = {std:.4f}")

        wf.world.dpi_scale = max(wf.world.dpi_scale, 2.0)
        # u = U / float(n + 1)
        amp: float = 1.0
        bg = wf.view.astype(wf.dtype) / 255.0

        def vis():
            u = U * amp / U.max()
            v = bg.copy()
            v[wf.mask] = wf.heatmap(u)[wf.mask]
            v = (v * 255.0).astype(np.uint8)
            caption = f"Expected Travel: {mean:.2f} +/- {std:.2f}m"
            wf.world.draw_src(v, wf.world.pixel_pos(wf.src))
            wf.world.draw_dst(v, wf.world.pixel_pos(wf.dst))
            wf.world.caption(v, caption, fg=(0, 0, 0), bg=(255, 255, 255))
            return v

        def renderVis(_amp: float | None = None):
            nonlocal amp
            if _amp is not None and _amp >= 1.0:
                amp = _amp
            return wf.world.show(vis(), "probability distribution")

        if wf.world.visualize:
            if t != last_vis_t:
                visualize(u, p, dp)
                handle = renderVis()
                cv2.createTrackbar(
                    "AMP", handle, int(amp * 10), 100, lambda v: renderVis(v / 10.0)
                )
            try:
                while True:
                    if cv2.waitKey(1) > 0:
                        break
            except:
                pass

        if out_img is not None:
            cv2.imwrite(str(out_img), vis())
