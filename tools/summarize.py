#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser
from pathlib import Path
from sys import stdin, stderr, exit
from yaml import safe_load
import numpy as np

meta = safe_load(stdin.read())
ours = list[float]()
ours_total: int = 0
ours_failure: int = 0

parser = ArgumentParser()
parser.add_argument("root", type=Path, nargs="?", default=[Path("results")])
parser.add_argument("--latex", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--animate", action="store_true")
root: Path = parser.parse_args().root[0]
should_write_latex = bool(parser.parse_args().latex)
should_plot = bool(parser.parse_args().plot)
should_animate = bool(parser.parse_args().animate)

if not any([should_write_latex, should_plot, should_animate]):
    parser.print_usage()
    parser.print_help()

if not root.is_dir():
    print(f"{dir} is not a directory", file=stderr)
    exit(1)


def dirs(d: Path):
    return (x for x in d.iterdir() if x.is_dir())


class BugAlgorithm:
    def __init__(self):
        self.failures = 0
        self.data = []

    def __call__(self, p: Path, baseline: float):
        with p.open("rt") as f:
            content = (l[1:] for l in f if l.startswith("#"))
            meta = safe_load("\n".join(content))
        if "abort" in meta or "travel" not in meta:
            self.failures += 1
        else:
            self.data.append(meta["travel"] / baseline)

    def stat(self):
        v = np.array(sorted(self.data))
        x = len(v) / (len(v) + self.failures)
        return v.mean(), v.std(), x

    @property
    def result(self):
        v = 1 / np.array(sorted(self.data))
        x = len(v) / (len(v) + self.failures)
        avg = np.mean(v)
        std = np.std(v) / 2
        return tuple(map(float, (x, avg, std)))


class BatchSampler:
    def __init__(self):
        self.failures: int = 0
        self.data: list[float] = []

    def __call__(self, p: Path, baseline: float):
        with p.open("rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if "FAIL" in line.upper():
                    self.failures += 1
                    continue
                # Let ValueError propagate
                (d,) = list(map(float, line.split(",")))
                self.data.append(d / baseline)

    def stat(self, success_rate: float):
        i = int(len(self.data) * success_rate)
        v = np.array(list(sorted(self.data))[:i])
        return v.mean(), v.std(), success_rate

    @property
    def result(self):
        data = 1 / np.array(sorted(self.data))
        total = len(data) + self.failures
        x = list[float]()  # Success rate
        avg = list[float]()
        std = list[float]()
        for i in range(1, len(data) + 1):
            v = data[:i]
            success_rate = i / total
            x.append(success_rate)
            avg.append(np.mean(v))
            std.append(np.std(v * success_rate))
        return np.array(x), np.array(avg), np.array(std)


class WaveFront:
    def __init__(self):
        self.T = list[float]()  # Travel Distance
        self.P = list[float]()  # Cumulative Probability
        self.DP = list[float]()  # Transient Probability

        self.total: int = 0
        self.success_rate: float = 1.0

    def __call__(self, p: Path):
        baseline: float | None = None
        self.total += 1
        with p.open("rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                # Let ValueError propagate
                t, p, dp = list(map(float, line.split(",")))
                if p <= 1e-3:
                    continue
                elif baseline is None:
                    baseline = t
                self.T.append(t / baseline)
                self.P.append(p)
                self.DP.append(dp)
            self.success_rate = min(self.success_rate, p)
        return baseline

    def stat(self, success_rate: float):
        T, P, DP = np.array(self.T), np.array(self.P), np.array(self.DP)
        indices = P <= success_rate
        T, DP = T[indices], DP[indices]
        return (
            (T * DP).sum() / DP.sum(),
            np.std(len(DP) * T * DP / DP.sum()),
            success_rate,
        )

    @property
    def result(self):
        T, P, DP = 1 / np.array(self.T), np.array(self.P), np.array(self.DP)
        DP *= self.total
        x = list[float]([0.0])  # Success rate
        avg = list[float]([1.0])
        std = list[float]([0.0])
        for l in np.linspace(0.01, 1, 100):
            sel = P <= l
            if len(sel) == 0:
                continue
            t, dp = T[sel], DP[sel]
            dp /= dp.sum()
            x.append(l)
            avg.append(np.sum(t * dp))
            std.append(np.std(t * dp))
        sel = P < 0.01
        t, dp = T[sel], DP[sel]
        return np.array(x), np.array(avg), np.array(std)


RW = BatchSampler()
WB = BatchSampler()
WF = WaveFront()
Bug0, Bug1, Bug2 = BugAlgorithm(), BugAlgorithm(), BugAlgorithm()

for d in dirs(root):
    baseline = WF(d / "wavefront.txt")
    if baseline is None:
        raise ValueError(f"Wavefront @{d.name} has no baseline")
    RW(d / "RandomWalk.list", baseline)
    WB(d / "WallBounce.list", baseline)
    Bug0(d / "Bug0L.txt", baseline)
    Bug0(d / "Bug0R.txt", baseline)
    Bug1(d / "Bug1L.txt", baseline)
    Bug1(d / "Bug1R.txt", baseline)
    Bug2(d / "Bug2L.txt", baseline)
    Bug2(d / "Bug2R.txt", baseline)
    # Our results
    v = meta[d.name]
    ours_total += 3
    if len(v) < 3:
        ours_failure += 3 - len(v)
    ours.extend((t / baseline for t in meta[d.name]))


ours_success_rate = (ours_total - ours_failure) / ours_total

# DATA REPORTING
if should_write_latex:

    def pz(v: str, l: int):
        if len(v) == l:
            return "  " + v
        if len(v) > l:
            return v
        return "\\z" + v.rjust(l)

    def report(name: str, mean: float, std: float, success_rate: float):
        print(name, "&")
        s = pz(f"{success_rate * 100:.2f}", 6)
        a = pz(f"{mean:.2f}", 5)
        b = pz(f"{std:.2f}", 4)
        print(f"${s}\\%$", "&")
        print(f"${a}$\\small{{$\\pm {b}$}}", "&")
        print("--", "\\\\")

    report(
        "\\textbf{ClipRover\\,\\ding{72}}",
        np.mean(ours),
        np.std(ours),
        ours_success_rate,
    )
    print("\\hline")

    report("\\multirow{2}{*}{Random\\,Walk}", *RW.stat(0.5))
    print("\\cline{2-3}")
    report("", *RW.stat(0.8))
    print("\\hline")

    report("\\multirow{2}{*}{Wall\\,Bounce}", *WB.stat(0.5))
    print("\\cline{2-3}")
    report("", *WB.stat(0.8))
    print("\\hline")

    report("\\multirow{2}{*}{Wave\\,Front}", *WF.stat(0.5))
    print("\\cline{2-3}")
    report("", *WF.stat(0.8))

    print("\\Xhline{2\\arrayrulewidth}")

    report("Bug 0", *Bug0.stat())
    print("\\hline")
    report("Bug 1", *Bug1.stat())
    print("\\hline")
    report("Bug 2", *Bug2.stat())

# PLOTTING
if not should_plot and not should_animate:
    exit(0)

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rc

rc("font", family="Times New Roman", size=16)


def init_figure():
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in ax.spines.values():
        s.set_edgecolor("#CCC")
        s.set_linewidth(1)
    ax.set_xlabel("Success Rate")
    ax.set_xticks(np.linspace(0, 100, 6), minor=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, color="#CCC", linewidth=1)
    # ax.set_xscale("log")
    ax.set_xlim(0, 100)

    ax.set_ylabel("Efficiency")
    # ax.set_yscale("log")
    ax.set_ylim(0, 1)
    return fig, ax


def EPL(
    epsilon: float,
    alpha: float = 1.0,
    omega=4,
    y0=1.0,
    y1=0.09,
    X=np.linspace(0, 1, 100),
):
    X = np.array(X)
    _eps = 1.0 + omega * epsilon
    beta = alpha * _eps
    t = 1 / (y0 ** (1 / beta))
    t2 = 1 / (y1 ** (1 / beta))
    k = t2 - t
    _X = (k * X + t) ** beta
    # _eps = _eps ** beta
    Y = _eps / _X
    return X, Y


def find_OMG(alpha: float, y0=1.0, y1=0.09):
    epl = lambda omg: abs(EPL(1.0, alpha, omg, y0, y1, X=[1.0])[1][0] - 1)
    OMG = [(epl(omg), omg) for omg in np.linspace(9, 15, 1000)]
    OMG = sorted(OMG, key=lambda x: x[0])
    return OMG[0][1]


alpha = 0.75
omega = find_OMG(alpha)

if should_plot:
    fig, ax = init_figure()
    x, y, e = RW.result
    print(x[-1], y[-1], e[-1])
    ax.plot(100 * x, y, color="gray", linestyle="--")

    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="gray")

    x, y, e = WB.result
    ax.plot(100 * x, y, color="blue", linestyle="--")
    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="blue")

    x, y, e = WF.result
    ax.plot(100 * x, y, label="Wavefront", color="red", linestyle="-")
    ax.fill_between(100 * x, y - e, y + e, alpha=0.2, color="pink")

    def ebar(
        ax: plt.Axes,
        x,
        y,
        e,
        /,
        color: str,
        label: str | None = None,
        marker="D",
        markersize: int = 8,
        **kwargs,
    ):
        kwargs = dict(capsize=4, clip_on=False, zorder=10, color=color) | kwargs
        data, cap, bars = ax.errorbar([100 * x], [y], [e], **kwargs).lines
        data.set_marker(marker)
        data.set_markersize(markersize)
        for l in cap:
            l.set_linewidth(8)
            l.set_alpha(0.25)
        for l in bars:
            l.set_linewidth(2)
            l.set_alpha(0.25)
        if label is not None:
            ax.text(
                100 * x - 2,
                y,
                label,
                ha="right",
                va="center",
                fontsize=16,
                color=color,
                fontweight="bold",
            )

    ebar(ax, *Bug0.result, label="Bug0", color="blue")
    ebar(ax, *Bug1.result, label="Bug1", color="green")
    ebar(ax, *Bug2.result, label="Bug2", color="red")

    # Our results
    v = [1 / v for v in ours]
    x = ours_success_rate
    avg = np.mean(v)
    std = np.std(v) / 2

    ebar(ax, x, avg, std, color="black", marker="*", markersize=24)
    ax.text(
        100 * x - 4,
        avg,
        "ClipRover",
        ha="right",
        va="center",
        fontsize=16,
        color="black",
        fontweight="bold",
    )

    def steps(a, b, s):
        while a <= b:
            yield a
            a += s

    # Equipotential lines
    for epsilon in steps(0.1, 0.9, 0.1):
        X, Y = EPL(epsilon, alpha, omega, y0=1.0, y1=0.1)
        ax.plot(100 * X, Y, color="gray", linestyle="--", linewidth=0.8)

    fig.savefig("summary.pdf")

    fig.show()
    plt.show()

# ========== ANIMATION ==========


def interpolate(X, Y, x0):
    for x1, x2, y1, y2 in zip(X, X[1:], Y, Y[1:]):
        if x1 <= x0 <= x2:
            return y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)


def search(X: np.ndarray, Y: np.ndarray, y0):
    if min(Y) > y0:
        return X[0]
    if max(Y) < y0:
        return X[-1]
    for x1, x2, y1, y2 in zip(X, X[1:], Y, Y[1:]):
        if (y1 - y0) * (y2 - y1) > 0:
            return x1 + (x2 - x1) * (y0 - y1) / (y2 - y1)
    raise ValueError("No solution found")


if should_animate:
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent
    sys.path.append(str(root))

    from lib.video import Video
    from lib.util import fig2img

    outfile = Path("EQP.local.mp4")
    outfile.unlink(missing_ok=True)
    video = Video(outfile, 120)

    rc("text", usetex=True)

    fig, ax = init_figure()
    ax.patch.set_alpha(0)
    ax.grid(False)

    (line,) = ax.plot([], [], color="black", linestyle="--", linewidth=1)
    (dot,) = ax.plot([], [], color="black", marker="v", markersize=6, clip_on=False)
    label = None

    # Equipotential lines
    for epsilon in np.linspace(0, 1, 1200):
        X, Y = EPL(epsilon, alpha, omega, y0=1.0, y1=0.1)
        line.set_data(100 * X, Y)
        x0 = search(X, Y, 1.0)
        dot.set_data([100 * x0], [1.03])
        if label is not None:
            label.remove()
        label = ax.text(
            100 * x0 - 0.6,
            1.05,
            # "hello world",
            f"$\\varepsilon = {epsilon:.2f}$",
            ha="left",
            va="bottom",
            fontsize=16,
            color="black",
            clip_on=False,
        )
        # Find X value
        frame = fig2img(fig)
        video.write(frame)
        import cv2

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    video.release()
