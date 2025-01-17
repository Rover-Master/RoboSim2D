# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Generator, Literal, Iterable
from contextlib import contextmanager
from lib.geometry import Point
from math import pi, cos, sin, ceil
import numpy as np, cv2
from lib.util import repeat


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
        print("Turn", direction, dr, n_steps)
        return map(lambda r: self.move(r0 + r), np.linspace(0, dr, n_steps))

    def step(
        self, pos: Point[float], dst: Point[float]
    ) -> Generator[Point[float] | Iterable[Point[float]], None, None]:
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
                cnt = 0
                p1: Point[float] | None = None
                for step in sim.step(pos, sim.dst):
                    if isinstance(step, Point):
                        step: Iterable[Point[float]] = [step]
                    for dp in step:
                        p = pos + dp
                        if sim.world.checkLine(pos, p):
                            p1 = p
                        else:
                            break
                        cnt += 1
                        if cnt > sim.max_attempts:
                            raise RuntimeError(
                                f"Exceeded maximum attempts at {str(pos)}"
                            )
                    if p1 is not None:
                        break
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

    @staticmethod
    def run(sim: "Simulation"):
        """
        Run the simulation
        """
        travel: float = 0.0
        p0 = sim.src
        if sim.world.visualize:
            bg = sim.world.view
            fg = np.zeros(bg.shape[:2], dtype=np.uint8)
            r0 = None
            p0v = p0

            def visualize():
                img = bg.copy()
                img[fg >= 128] = [0, 0, 255]
                sim.world.draw_src(img, sim.world.pixel_pos(sim.src), (255, 0, 0))
                sim.world.draw_dst(img, sim.world.pixel_pos(sim.dst), (0, 192, 0))
                cv2.circle(
                    img,
                    sim.world.pixel_pos(p1),
                    sim.world.px(0.1),
                    (255, 0, 0),
                    sim.world.lw,
                )
                sim.world.show(img)

        try:
            for p1 in sim:
                r1 = sim.heading
                if sim.world.visualize and r1 != r0:
                    r0 = r1
                    cv2.line(
                        fg,
                        sim.world.pixel_pos(p0v),
                        sim.world.pixel_pos(p1),
                        255,
                        sim.world.lw,
                        cv2.LINE_AA,
                    )
                    p0v = p1
                    visualize()
                    key = cv2.waitKey(1)
                    if key == 27 or key == ord("q"):  # ESC or 'q'
                        break
                travel += (p1 - p0).norm
                p0 = p1
                print(p1, sim.heading, sep=", ")
                if travel > sim.max_travel:
                    print("# Failed: Max travel distance reached")
                    break
        except KeyboardInterrupt:
            print("# Aborted by user")
        except Exception as e:
            print("# Failed:", e)
        else:
            if sim.world.visualize:
                visualize()
                try:
                    for key in repeat(cv2.waitKey, 10):
                        if key > 0:
                            break
                except KeyboardInterrupt:
                    pass
                cv2.destroyAllWindows()
        finally:
            print("# src    =", sim.src)
            print("# dst    =", sim.dst)
            print("# Travel =", round(travel, 2))
