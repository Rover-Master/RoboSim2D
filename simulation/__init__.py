# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import sys, numpy as np, cv2
from typing import Generator, Literal, Iterable
from math import pi, cos, sin, ceil
from itertools import chain
from lib.geometry import Point
from lib.world import World
from lib.util import repeat, dup


class Proposal:
    def __init__(self, *iterables: Iterable["Point[float] | Proposal"]):
        self.queue = chain(*iterables)

    class Session:
        def __init__(self, p: "Proposal", p0: Point[float], world: World):
            self.queue = iter(p.queue)
            self.p0 = p0
            self.world = world

        def __iter__(self):
            return self

        def __next__(self):
            dp = next(self.queue)
            p0, world = self.p0, self.world
            if isinstance(dp, Point):
                p1 = p0 + dp
                if self.world.checkLine(p0, p1):
                    return p1
            else:
                proposal = Proposal.construct(dp)
                p1 = proposal(p0, world)
                if p1 is not None:
                    return p1

    def traverse(self, p0: Point[float], world: World):
        return Proposal.Session(self, p0, world)

    def __call__(self, p0: Point[float], world: World) -> Point[float] | None:
        raise NotImplementedError

    @staticmethod
    def construct(v: Point[float] | Iterable[Point[float]] | "Proposal") -> "Proposal":
        if isinstance(v, Proposal):
            return v
        elif isinstance(v, Point):
            return Eager([v])
        else:
            return Eager(v)


class Eager(Proposal):
    """
    Returns the FIRST viable candidate from all proposed points
    """

    def __call__(self, p0: Point[float], world: World):
        for p1 in self.traverse(p0, world):
            if p1 is not None:
                return p1


class Lazy(Proposal):
    """
    Returns the LAST viable candidate from a set of consecutive viable points.
    e.g. [p1(Y), p2(Y), p3(N), p4(Y)] -> p2
    """

    def __call__(self, p0: Point[float], world: World):
        p1 = None
        for p in self.traverse(p0, world):
            if p is None:
                break
            p1 = p
        return p1


class Simulation:

    heading: float = 0.0

    max_attempts: int = 1000  # Maximum number of attempts per step
    max_travel: float = 1000.0  # Maximum travel distance
    step_length: float = 0.2  # Meters

    def __init__(self, **kwargs):
        import lib.env as env

        self.world = env.world
        self.src = env.src_pos
        self.dst = env.dst_pos
        self.__dict__.update(kwargs)

    def move(self, hdg: float, d: float | None = None) -> Point[float]:
        """
        Move the robot along a given heading by a certain length.
        Heading is in radians.
        """
        # zero heading pointing up (north)
        r = hdg + 0.5 * np.pi
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
    ) -> Generator[Point[float] | Iterable[Point[float]] | Proposal, any, any]:
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

    class Session:

        started: bool = False
        completed: bool = False
        finished: bool = False

        def __init__(self, sim: "Simulation"):
            self.sim = sim
            self.pos = sim.src
            self.completed = sim.isComplete(self.pos, sim.dst)

        def __next__(self):
            if not self.started:
                self.started = True
                return self.sim.src
            elif not self.completed:
                sim = self.sim
                pos = self.pos
                p1: Point[float] | None = None
                for cnt, step in enumerate(sim.step(pos, sim.dst)):
                    proposal = Proposal.construct(step)
                    p1 = proposal(self.pos, world=sim.world)
                    if p1 is not None:
                        break
                    if cnt > sim.max_attempts:
                        raise RuntimeError(f"Exceeded maximum attempts at {str(pos)}")
                else:
                    # sim.step() exhausted, no path found
                    raise RuntimeError(
                        f"{sim.__class__.__name__}.step() exhausted at {pos}"
                    )
                self.sim.heading = (p1 - pos).angle - 0.5 * pi
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
        pos: Point[float],
        fg: np.ndarray,
        bg: np.ndarray | None = None,
        travel: float | None = None,
    ):
        world = self.world
        if bg is None:
            bg = world.view
        bg[fg >= 128] = [0, 0, 255]
        world.draw_src(bg, world.pixel_pos(self.src))
        world.draw_dst(bg, world.pixel_pos(self.dst))
        cv2.circle(
            bg,
            world.pixel_pos(pos),
            world.px(0.1),
            (255, 0, 0),
            world.line_width,
        )
        if travel is not None:
            world.caption(
                bg, f"Travel: {travel:.2f} m", fg=(0, 0, 0), bg=(255, 255, 255)
            )
        return bg

    @staticmethod
    def run(sim: "Simulation"):
        """
        Run the simulation
        """
        name = sim.__class__.__name__
        trj_list = sim.world.withPrefix(name, suffix="txt")
        trj_img = sim.world.withPrefix(name, suffix="png")
        if trj_list is not None:
            trj_list_file = open(trj_list, "w")
            print = dup(trj_list_file)

        travel: float = 0.0
        p0 = sim.src
        if sim.world.visualize:
            bg = sim.world.view
            fg = np.zeros(bg.shape[:2], dtype=np.uint8)
            sim.world.show(sim.visualize(p0, fg, travel=0.0))

            def line(a: Point[float], b: Point[float]):
                cv2.line(
                    fg,
                    sim.world.pixel_pos(a),
                    sim.world.pixel_pos(b),
                    255,
                    sim.world.line_width,
                    cv2.LINE_AA,
                )

        try:
            for p1 in sim:
                travel += (p1 - p0).norm
                if sim.world.visualize:
                    line(p0, p1)
                    sim.world.show(sim.visualize(p1, fg, travel=travel))
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        break
                print(*p0, sim.heading, sep=",")
                p0 = p1
                if travel > sim.max_travel:
                    print("# Failed: Max travel distance reached")
                    break
            print(*p1, sim.heading, sep=", ")
        except KeyboardInterrupt:
            print("# Aborted by user")
        except Exception as e:
            import traceback

            print("# Failed:", e)
            traceback.print_exception(e, file=sys.stderr)
        finally:
            print("# src    =", sim.src)
            print("# dst    =", sim.dst)
            print("# travel =", round(travel, 2))
            if trj_list is not None:
                trj_list_file.close()
            if trj_img is not None:
                cv2.imwrite(str(trj_img), fg)
            if sim.world.visualize:
                try:
                    for key in repeat(cv2.waitKey, 10):
                        if key > 0:
                            break
                except KeyboardInterrupt:
                    pass
                cv2.destroyAllWindows()
