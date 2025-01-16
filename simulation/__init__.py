# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Generator
from lib.geometry import Point
from math import cos, sin
import numpy as np, cv2
from lib.util import repeat
from lib.interactive import draw_src, draw_dst


class Simulation:

    heading: float = 0.0

    max_attempts: int = 1000  # Maximum number of attempts per step
    max_travel: float = 1000.0  # Maximum travel distance
    step_length: float = 0.1  # Meters

    def __init__(self):
        import lib.env as env

        self.world = env.world
        self.scale = env.scale
        self.threshold = env.threshold
        self.visualize = env.visualize
        self.src = env.src_pos
        self.dst = env.dst_pos

    def move(self, hdg: float, d: float | None = None) -> Point[float]:
        """
        Move the robot along a given heading by a certain length.
        Heading is in radians.
        """
        self.heading = hdg
        # zero heading pointing up (north)
        r = hdg + 0.5 * np.pi
        if d is None:
            d = self.step_length
        return Point(d * cos(r), d * sin(r), type=float)

    def step(
        self, pos: Point[float], dst: Point[float]
    ) -> Generator[Point[float], None, None]:
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

    class Session:

        started: bool = False
        completed: bool = False
        finished: bool = False

        def __init__(self, sim: "Simulation"):
            self.sim = sim
            self.pos = sim.src

        def __next__(self):
            if not self.started:
                self.started = True
                return self.sim.src
            elif not self.completed:
                sim = self.sim
                pos = self.pos
                for n, p1 in enumerate(sim.step(pos, sim.dst)):
                    if sim.world.checkLine(pos, p1):
                        break
                    if n >= sim.max_attempts:
                        raise RuntimeError(f"Trapped at {str(pos)}")
                else:
                    # sim.step() exhausted, no path found
                    raise RuntimeError(f"Exhausted at {str(pos)}")
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
        if sim.visualize:
            bg = sim.world.view(sim.scale)
            fg = np.zeros(bg.shape[:2], dtype=np.uint8)
            r0 = sim.heading
        for p1 in sim:
            r1 = sim.heading
            if sim.visualize:
                cv2.line(
                    fg,
                    sim.world.pixel_pos(p0, sim.scale),
                    sim.world.pixel_pos(p1, sim.scale),
                    255,
                    1,
                    cv2.LINE_AA,
                )
            travel += (p1 - p0).norm
            p0 = p1
            if sim.visualize and r1 != r0:
                r0 = r1
                img = bg.copy()
                img[fg > 0] = [0, 0, 255]
                draw_src(img, sim.world.pixel_pos(sim.src, sim.scale), (255, 0, 0))
                draw_dst(img, sim.world.pixel_pos(sim.dst, sim.scale), (0, 192, 0))
                cv2.circle(img, sim.world.pixel_pos(p1, sim.scale), 6, (255, 0, 0), 3)
                cv2.imshow("Map", img)
                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):  # ESC or 'q'
                    break
            print(p1, sim.heading, sep=", ")
        print("# Travel =", round(travel, 2))
        if sim.visualize:
            img = bg.copy()
            img[fg > 0] = [0, 0, 255]
            draw_src(img, sim.world.pixel_pos(sim.src, sim.scale), (255, 0, 0))
            draw_dst(img, sim.world.pixel_pos(sim.dst, sim.scale), (0, 192, 0))
            cv2.circle(img, sim.world.pixel_pos(p1, sim.scale), 6, (255, 0, 0), 3)
            cv2.imshow("Map", img)
            try:
                for key in repeat(cv2.waitKey, 10):
                    if key == ord(" "):
                        return Simulation.run(sim)
                    elif key >= 0:
                        break
            except KeyboardInterrupt:
                pass
            cv2.destroyAllWindows()
