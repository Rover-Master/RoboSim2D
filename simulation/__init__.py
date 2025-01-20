# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import builtins, sys, numpy as np, cv2
from typing import Generator, Literal, Iterable, Union
from math import pi, cos, sin, ceil
from itertools import chain
from contextlib import contextmanager, closing

from lib.arguments import parser

parser.add_argument(
    "--step-length", type=float, help="Step length in meters", default=0.1
)

from lib.geometry import Point
from lib.world import World
from lib.util import repeat, dup, Retry


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
                if (not sim.check) or sim.world.checkLine(p0, p1):
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


ProposalType = (
    Point[float] | Iterable[Point[float]] | Proposal | Iterable[Proposal] | None
)


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


class Simulation:
    heading: float = 0.0

    def __init__(self, **kwargs):
        from lib.env import world, src_pos, dst_pos, args

        self.step_length: float = args.step_length
        self.world = world
        self.src = src_pos
        self.dst = dst_pos
        self.max_travel = world.max_travel
        self.__dict__.update(kwargs)

    check: bool = True

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
        return Point(d * cos(r), d * sin(r), type=float)

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

    def step(
        self, pos: Point[float], dst: Point[float]
    ) -> Generator[ProposalType, any, any]:
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
        return (pos - dst).norm < self.world.threshold

    class Abort(Exception):
        def __init__(self, reason: str):
            self.reason = reason

    class Session:

        started: bool = False
        completed: bool = False
        finished: bool = False

        def __init__(self, sim: "Simulation"):
            self.sim = sim
            self.pos = sim.src
            self.completed = sim.isComplete(self.pos, sim.dst)

        @Retry()
        def __next__(self):
            if not self.started:
                self.started = True
                return self.sim.src
            elif not self.completed:
                sim = self.sim
                pos = self.pos
                p1: Point[float] | None = None
                with closing(sim.step(pos, sim.dst)) as proposals:
                    for step in proposals:
                        if step is None:
                            raise Retry
                        proposal = Proposal.construct(step)
                        p1 = proposal(self.pos, self.sim)
                        if p1 is not None:
                            break
                    else:
                        # sim.step() exhausted, no path found
                        raise RuntimeError(
                            f"{sim.__class__.__name__}.step() exhausted at {pos}"
                        )
                self.sim.heading = (p1 - pos).angle
                self.pos = p1
                self.completed = sim.isComplete(pos, sim.dst)
                return self.pos
            elif not self.finished:
                self.finished = True
                return self.sim.dst
            else:
                raise StopIteration

    def __iter__(self):
        """
        Run the simulation
        """
        return Simulation.Session(self)

    def visualize(
        self,
        pos: Point[float] | None,
        fg: np.ndarray,
        bg: np.ndarray | None = None,
        travel: float | None = None,
    ):
        name = self.__class__.__name__
        world = self.world
        if bg is None:
            bg = world.view
        bg[fg >= 128] = world.line_color
        world.draw_src(bg, world.pixel_pos(self.src))
        world.draw_dst(bg, world.pixel_pos(self.dst))
        if pos is not None:
            cv2.circle(
                bg,
                world.pixel_pos(pos),
                world.px(0.16),
                (192, 0, 0),
                cv2.FILLED,
            )
        if travel is not None:
            world.caption(
                bg, f"Travel: {travel:.2f} m ({name})", fg=(0, 0, 0), bg=(255, 255, 255)
            )
        return bg

    @staticmethod
    def run(sim: "Simulation"):
        """
        Run the simulation
        """
        name = sim.__class__.__name__
        sim_img = sim.world.withPrefix(name, suffix="png")
        trj_list = sim.world.withPrefix(name, suffix="txt")
        trj_img = sim.world.withPrefix(f"{name}-trj", suffix="png")
        overlay_img = sim.world.withPrefix(f"{name}-overlay", suffix="png")
        if trj_list is not None:
            trj_list_file = open(trj_list, "w")
            print = dup(trj_list_file)
        else:
            print = builtins.print

        travel: float = 0.0
        p0 = sim.src

        trj = np.zeros(sim.world.grayscale.shape, dtype=np.uint8)

        def line(a: Point[float], b: Point[float]):
            cv2.line(
                trj,
                sim.world.pixel_pos(a),
                sim.world.pixel_pos(b),
                255,
                sim.world.line_width,
                cv2.LINE_AA,
            )

        def failure():
            cv2.drawMarker(
                trj,
                sim.world.pixel_pos(p1),
                0,
                cv2.MARKER_TILTED_CROSS,
                sim.world.px(0.3),
                sim.world.line_width * 3,
            )
            cv2.drawMarker(
                trj,
                sim.world.pixel_pos(p1),
                255,
                cv2.MARKER_TILTED_CROSS,
                sim.world.px(0.3),
                sim.world.line_width,
            )

        if sim.world.visualize:
            bg = sim.world.view
            sim.world.show(sim.visualize(p0, trj, travel=0.0))

        try:
            for p1 in sim:
                travel += (p1 - p0).norm
                line(p0, p1)
                if sim.world.visualize:
                    sim.world.show(sim.visualize(p1, trj, travel=travel))
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        break
                print(*p0, sim.heading - 0.5 * pi, sep=",")
                p0 = p1
                if sim.max_travel is not None and travel > sim.max_travel:
                    raise Simulation.Abort("exceeded max travel distance")
            print(*p1, sim.heading, sep=", ")
            print("# src   :", sim.src)
            print("# dst   :", sim.dst)
            print("# travel:", f"{travel:.2f}")
        except KeyboardInterrupt:
            failure()
            print("# abort : user aborted")
        except Simulation.Abort as e:
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
            sim.world.saveImg(trj_img, trj)
        if sim_img is not None:
            sim.world.saveImg(sim_img, sim.visualize(None, trj, travel=travel))
        if overlay_img is not None:
            blank = np.zeros_like(trj)
            overlay = sim.visualize(
                None, blank, np.stack([blank] * 3, axis=-1), travel=travel
            )
            alpha = (np.max(overlay, axis=-1, keepdims=True) > 0).astype(np.uint8) * 255
            img = np.concatenate([overlay, alpha], axis=-1)
            sim.world.saveImg(overlay_img, img)
        if sim.world.visualize and not sim.world.no_wait:
            try:
                for key in repeat(cv2.waitKey, 10):
                    if key > 0:
                        break
            except KeyboardInterrupt:
                pass
            cv2.destroyAllWindows()


class WallFollowing:
    wall_following_direction: Literal["L", "R"] | None = None

    def __init__(self):
        if not isinstance(self, Simulation):
            raise TypeError(f"{self} does not implement Simulation")

    @property
    def hit_wall(self: Union[Simulation, "WallFollowing"]) -> ProposalType:
        match self.wall_following_direction:
            case "L":
                return Eager(self.turn("left"))
            case "R":
                return Eager(self.turn("right"))
        raise RuntimeError(
            f"Bad wall following direction {self.wall_following_direction}"
        )

    @property
    def move_along_wall(self: Union[Simulation, "WallFollowing"]) -> ProposalType:
        match self.wall_following_direction:
            case "L":
                return (
                    # First try to turn clockwise
                    Lazy(self.turn("right")),
                    # Right turn not viable, try left turn
                    Eager(self.turn("left")),
                )
            case "R":
                return (
                    # First try to turn counter-clockwise
                    Lazy(self.turn("left")),
                    # Left turn not viable, try right turn
                    Eager(self.turn("right")),
                )
        raise RuntimeError(
            f"Bad wall following direction {self.wall_following_direction}"
        )
