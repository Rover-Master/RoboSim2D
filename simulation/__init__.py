# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import builtins, sys, numpy as np, cv2
from typing import Generator, Literal, Iterable, Union
from math import pi, ceil
from itertools import chain, pairwise
from contextlib import contextmanager, closing

from lib.geometry import Point
from lib.util import repeat, min_max, dup, Retry
from lib.arguments import auto_parse


class Hypothesis:
    """
    Dry-run the proposed point, return the first viable point.
    """

    def __init__(self, *hypos: Iterable[Point[float]]):
        self.hypos = hypos

    def __call__(self, p0: Point[float], sim: "Simulation"):
        for p1 in chain(*self.hypos):
            p1 = p0 + p1
            if sim.world.checkLine(p0, p1, sim.radius):
                return p1


class Proposal:
    def __init__(self, *iterables: Iterable["Point[float] | Proposal"]):
        self.queue = chain(*iterables)

    class Session:
        def __init__(self, p: "Proposal", p0: Point[float], sim: "Simulation"):
            self.queue = iter(p.queue)
            self.p0 = p0
            self.sim = sim

        def __iter__(self):
            return self

        def __next__(self):
            dp = next(self.queue)
            p0, sim = self.p0, self.sim
            if isinstance(dp, Point):
                p1 = p0 + dp
                if (not sim.check) or sim.world.checkLine(p0, p1, sim.radius):
                    return p1
            else:
                proposal = Proposal.construct(dp)
                p1 = proposal(p0, sim)
                if p1 is not None:
                    return p1

    def traverse(self, p0: Point[float], sim: "Simulation"):
        return Proposal.Session(self, p0, sim)

    def __call__(self, p0: Point[float], sim: "Simulation") -> Point[float] | None:
        raise NotImplementedError

    @staticmethod
    def construct(v: Point[float] | Iterable[Point[float]] | "Proposal") -> "Proposal":
        if isinstance(v, Proposal):
            return v
        elif isinstance(v, Point):
            return Eager([v])
        else:
            return Eager(v)


ProposalType = Point[float] | Iterable[Point[float]] | Proposal | Iterable[Proposal]


class Eager(Proposal):
    """
    Returns the FIRST viable candidate from all proposed points
    """

    def __call__(self, p0: Point[float], sim: "Simulation"):
        for p1 in self.traverse(p0, sim):
            if p1 is not None:
                return p1


class Lazy(Proposal):
    """
    Returns the LAST viable candidate from a set of consecutive viable points.
    e.g. [p1(Y), p2(Y), p3(N), p4(Y)] -> p2
    """

    def __call__(self, p0: Point[float], sim: "Simulation"):
        p1 = None
        for p in self.traverse(p0, sim):
            if p is None:
                break
            p1 = p
        return p1


from lib.simulation import SimulationBase


class __Simulation__(SimulationBase):
    heading: float = 0.0
    check: bool = True

    yield_src: bool = True
    yield_dst: bool = True

    @property
    @contextmanager
    def no_check(self):
        check = self.check
        self.check = False
        yield
        self.check = check

    def move(self, hdg: float, d: float | None = None) -> Point[float]:
        """
        Move the robot along a given heading by a certain length.
        Heading is in radians.
        """
        # zero heading pointing up (north)
        r = hdg
        if d is None:
            d = self.step_length
        return Point.Angular(r, d)

    def turn(
        self,
        direction: Literal["left", "right"] | int | float,
        delta=pi / 180.0,
    ) -> float:
        r0 = self.heading
        match direction:
            case "left":
                dr = +2 * np.pi
            case "right":
                dr = -2 * np.pi
            case _:
                dr = float(direction) % (2 * np.pi)
        n_steps = max(ceil(abs(dr / delta)), 1)
        return map(lambda r: self.move(r0 + r), np.linspace(0, dr, n_steps))

    def wiggle(
        self,
        heading: float,
        delta=pi / 180.0,
    ):
        n_steps = max(ceil(abs(pi / delta)), 1)
        return map(
            lambda r: Eager([self.move(heading + r), self.move(heading - r)]),
            np.linspace(0, pi, n_steps),
        )

    def step(
        self, pos: Point[float], dst: Point[float]
    ) -> Generator[Hypothesis | ProposalType | None, Point | None, any]:
        """
        Step the simulation forward.
        Implementation needs to yield the next position of the robot.
        If the yielded position is not viable, the step function need to try again.
        """
        raise NotImplementedError

    def isComplete(self, pos: Point[float], dst: Point[float]) -> bool:
        """
        Check if the simulation has reached a termination condition.
        """
        return (pos - dst).norm < self.threshold

    class Abort(Exception):
        def __init__(self, reason: str):
            self.reason = reason

    class Session:

        started: bool = False
        completed: bool = False
        finished: bool = False

        def __init__(self, sim: "__Simulation__"):
            self.sim = sim
            self.pos = sim.src
            self.completed = sim.isComplete(self.pos, sim.dst)

        @Retry()
        def __next__(self):
            if not self.started:
                self.started = True
                if self.sim.yield_src:
                    return self.sim.src
            if not self.completed:
                sim = self.sim
                pos = self.pos
                p1: Point[float] | None = None
                steps = sim.step(pos, sim.dst)
                try:
                    while True:
                        step = next(steps)
                        while isinstance(step, Hypothesis):
                            step = steps.send(step(self.pos, self.sim))
                        if step is None:
                            raise Retry
                        proposal = Proposal.construct(step)
                        p1 = proposal(self.pos, self.sim)
                        if p1 is not None:
                            break
                except StopIteration:
                    # sim.step() exhausted, no path found
                    raise Simulation.Abort(
                        f"{sim.__class__.__name__} exhausted at {pos}"
                    )
                self.sim.heading = (p1 - pos).angle
                self.pos = p1
                self.completed = sim.isComplete(pos, sim.dst)
                return self.pos
            if not self.finished and self.sim.yield_dst:
                self.finished = True
                return self.sim.dst
            else:
                raise StopIteration

    def __iter__(self):
        """
        Run the simulation
        """
        return __Simulation__.Session(self)

    def visualize(
        self,
        pos: Point[float] | None,
        fg: np.ndarray,
        bg: np.ndarray | None = None,
        travel: float | None = None,
    ):
        name = self.__class__.__name__
        vis = self.vis
        if bg is None:
            bg = vis.view
        bg[fg >= 128] = vis.line_color
        vis.draw_src(bg, vis.pixel_pos(self.src))
        vis.draw_dst(bg, vis.pixel_pos(self.dst))
        if pos is not None:
            cv2.circle(
                bg,
                vis.pixel_pos(pos),
                vis.px(0.16),
                (192, 0, 0),
                cv2.FILLED,
            )
        if travel is not None:
            vis.caption(
                bg, f"Travel: {travel:.2f} m ({name})", fg=(0, 0, 0), bg=(255, 255, 255)
            )
        return bg

    @classmethod
    def run(cls, sim: "Simulation" = None, /, **kwargs):
        """
        Run the simulation
        """
        if sim is None:
            sim = cls(**kwargs)
        elif len(kwargs) > 0:
            raise TypeError("Cannot specify both sim and kwargs")
        vis = sim.vis
        name = sim.__class__.__name__
        sim_img = sim.out(name, suffix="png")
        trj_list = sim.out(name, suffix="txt")
        trj_img = sim.out(f"{name}-trj", suffix="png")
        overlay_img = sim.out(f"{name}-overlay", suffix="png")
        if trj_list is not None:
            trj_list_file = open(trj_list, "w")
            print = dup(trj_list_file)
        else:
            print = builtins.print

        travel: float = 0.0
        p0 = sim.src
        p1 = sim.src

        trj = np.zeros(vis.grayscale.shape, dtype=np.uint8)

        def line(a: Point[float], b: Point[float]):
            cv2.line(
                trj,
                vis.pixel_pos(a),
                vis.pixel_pos(b),
                255,
                vis.line_width,
                cv2.LINE_AA,
            )

        def failure():
            cv2.drawMarker(
                trj,
                vis.pixel_pos(p1),
                0,
                cv2.MARKER_TILTED_CROSS,
                vis.px(0.3),
                vis.line_width * 3,
            )
            cv2.drawMarker(
                trj,
                vis.pixel_pos(p1),
                255,
                cv2.MARKER_TILTED_CROSS,
                vis.px(0.3),
                vis.line_width,
            )

        if vis.visualize:
            vis.show(sim.visualize(p0, trj, travel=0.0))

        flag_term = False
        try:
            for p1 in sim:
                travel += (p1 - p0).norm
                line(p0, p1)
                if vis.visualize:
                    vis.show(sim.visualize(p1, trj, travel=travel))
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        break
                print(*p0, (p1 - p0).angle, sep=",")
                p0 = p1
                if sim.max_travel is not None and travel > sim.max_travel:
                    raise __Simulation__.Abort("exceeded max travel distance")
            print(*p1, (p1 - p0).angle, sep=", ")
            print("# src   :", sim.src)
            print("# dst   :", sim.dst)
            print("# travel:", travel)
        except KeyboardInterrupt:
            flag_term = True
            failure()
            print("# abort : user aborted")
        except __Simulation__.Abort as e:
            failure()
            print("# abort :", e.reason)
        except Exception as e:
            import traceback

            failure()
            print("# abort :", e)
            traceback.print_exception(e, file=sys.stderr)

        if trj_list is not None:
            trj_list_file.close()
        if trj_img is not None:
            vis.saveImg(trj_img, trj)
        if sim_img is not None:
            vis.saveImg(sim_img, sim.visualize(None, trj, travel=travel))
        if overlay_img is not None:
            blank = np.zeros_like(trj)
            overlay = sim.visualize(
                None, blank, np.stack([blank] * 3, axis=-1), travel=travel
            )
            alpha = (np.max(overlay, axis=-1, keepdims=True) > 0).astype(np.uint8) * 255
            img = np.concatenate([overlay, alpha], axis=-1)
            vis.saveImg(overlay_img, img)
        if vis.visualize and not vis.no_wait:
            try:
                vis.show(sim.visualize(None, trj, travel=travel))
                for key in repeat(cv2.waitKey, 10):
                    if key > 0:
                        break
            except KeyboardInterrupt:
                flag_term = True
                pass
            cv2.destroyAllWindows()

        return not flag_term


@auto_parse()
class Simulation(__Simulation__):
    """
    Exported for convenience of use by specific implementations.
    No explicit argument parsing is needed.
    """


class WallFollowing:
    wall_following_direction: Literal["L", "R"] | None = None

    def __init__(self):
        if not isinstance(self, __Simulation__):
            raise TypeError(f"{self} does not implement Simulation")

    @property
    def hit_wall(self: Union[__Simulation__, "WallFollowing"]) -> ProposalType:
        match self.wall_following_direction:
            case "L":
                return Eager(self.turn("left"))
            case "R":
                return Eager(self.turn("right"))
        raise RuntimeError(
            f"Bad wall following direction {self.wall_following_direction}"
        )

    @property
    def follow_wall_left(self: Union[__Simulation__, "WallFollowing"]):
        return (
            # First try to turn clockwise
            Lazy(self.turn("right")),
            # Right turn not viable, try left turn
            Eager(self.turn("left")),
        )

    @property
    def follow_wall_right(self: Union[__Simulation__, "WallFollowing"]):
        return (
            # First try to turn counter-clockwise
            Lazy(self.turn("left")),
            # Left turn not viable, try right turn
            Eager(self.turn("right")),
        )

    @property
    def follow_wall(self: Union[__Simulation__, "WallFollowing"]) -> ProposalType:
        match self.wall_following_direction:
            case "L":
                return self.follow_wall_left
            case "R":
                return self.follow_wall_right
        raise RuntimeError(
            f"Bad wall following direction {self.wall_following_direction}"
        )


from lib.world import World


class OffsetPath(Simulation, WallFollowing):
    yield_src = False
    yield_dst = False

    prev: Point[float] | None = None
    travel: float = 0.0

    nearest_idx: int = 0

    def step(self, pos: Point[float], dst):
        nearest: float | None = None
        for i, (p, _, _, t) in self.scope(self.nearest_idx):
            d = (pos - p).norm
            if nearest is None:
                nearest = d
            elif d < nearest:
                nearest = d
                self.nearest_idx = i

        self.travel = self.vec[self.nearest_idx][3]

        self.prev = pos
        if self.look_back is not None or self.look_ahead is not None:
            self.world.img = self.render(
                pos,
                look_back=self.look_back,
                look_ahead=self.look_ahead,
                img=self.full_img,
            )
        yield self.follow_wall

    def scope(self, a: int = 0):
        for b, (_, _, _, t) in enumerate(self.vec[a:], a):
            if self.look_back is not None and t < self.travel - self.look_back:
                a = b + 1
            if self.look_ahead is not None and t > self.travel + self.look_ahead:
                break
        return enumerate(self.vec[a:b], a)

    def isComplete(self, pos, dst):
        if (
            self.look_ahead is not None
            and self.travel + self.look_ahead < self.trj_length
        ):
            return False
        return super().isComplete(pos, dst)

    def __init__(
        self,
        trj: Iterable[Point[float]],
        offset: float,
        *,
        resolution: float | None = None,
        step_length: float | None = None,
        look_back: float | None = None,
        look_ahead: float | None = None,
        **arguments,
    ):
        """
        Follow a path with an offset. Positive offset means to the left of the path.
        """
        self.look_back = look_back
        self.look_ahead = look_ahead
        # Cache vectors
        items = ((a, b, b - a) for a, b in pairwise(trj))
        t = 0.0

        def acc(dt: float):
            nonlocal t
            t += dt
            return t

        vec = self.vec = [(p0, p1, v, acc(v.norm)) for p0, p1, v in items if v.norm > 0]
        if len(vec) < 1:
            raise ValueError("Less than 2 points in the trajectory")
        # Derive wall following direction
        if offset > 0:
            self.wall_following_direction = "L"
        elif offset < 0:
            self.wall_following_direction = "R"
        else:
            raise ValueError("Offset cannot be zero")
        radius = abs(offset) / 2.0
        # Calculate bounding box
        x0, x1 = min_max(p.x for p in trj)
        y0, y1 = min_max(p.y for p in trj)
        # Auto determine step-length, if not specified
        if step_length is None:
            step_length = max(min(v[2].norm for v in vec), 0.05)
        # Auto-determine resolution, if not specified
        if resolution is None:
            resolution = min(radius, step_length) / 2
        # Create occupancy map
        pad = abs(offset) * 4.0
        w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
        blank = (
            np.ones((ceil(h / resolution), ceil(w / resolution)), dtype=np.uint8) * 255
        )
        # Compute start and end points
        k = 1 if self.wall_following_direction == "L" else -1
        p, _, v, l = vec[0]
        src = p + Point.Angular(v.angle + 0.5 * k * pi, 1.2 * abs(offset))
        self.heading = v.angle
        _, p, v, l = vec[-1]
        dst = p
        self.trj_length = l
        # Create world
        arguments = (
            dict(
                src=src,
                dst=dst,
                radius=radius,
                step_length=step_length,
                threshold=2 * step_length + abs(offset),
                res=resolution,
                visualize=True,
                no_wait=True,
            )
            | arguments
        )
        arguments["meta"] = arguments
        world = World(initialized=True, **arguments)
        arguments["world"] = world
        world.initialize(blank, origin=Point(x0 - pad, y0 - pad))
        # Construct Simulation
        __Simulation__.__init__(self, **arguments)
        WallFollowing.__init__(self)
        # Precompute full occupancy map
        self.full_img = self.render()
        self.full_img[self.full_img < 128] = 128

    def render(
        self,
        pos: Point[float] | None = None,
        look_back: float | None = None,
        look_ahead: float | None = None,
        img: np.ndarray | None = None,
    ):
        occupancy = np.zeros_like(self.world.occupancy)
        for p0, p1, v, t in self.vec:
            if look_back is not None and t < self.travel - look_back:
                continue
            if look_ahead is not None and t > self.travel + look_ahead:
                break
            mask = self.world.trajectory(p0, p1, self.radius)
            occupancy |= mask
        if pos is not None:
            occupancy[self.world.trajectory(pos, pos, self.radius)] = False
        self.world.occupancy = occupancy
        if img is not None:
            img = img.copy()
            img[occupancy] = 0
        else:
            img = ((1 - occupancy) * 255).astype(np.uint8)
        return img
