#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from os import cpu_count
from sys import stdin, stdout, stderr, executable
from yaml import safe_load as parse
from typing import Iterable, Callable
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, Manager
from subprocess import Popen, PIPE
from time import time
from tqdm import tqdm

WATCH_LIST = [
    # ("SW", "C", "Bug1L"),
    # ("SW", "NE", "Bug1L"),
]

SLICE = {"slice": "19,7,320,360"}


def camel2dash(s: str) -> str:
    import re

    return re.sub(r"([a-z])([A-Z])", r"\1-\2", s).lower()


def kw2cmdline(**kwargs):
    for k, v in kwargs.items():
        if type(v) is bool:
            if v:
                yield f"--{camel2dash(k)}"
        else:
            yield f"--{camel2dash(k)}={v}"


class Python:
    def __init__(self, module: str, **env):
        self.module = module
        self.env = env

    def __call__(self, *args, **kwargs):
        args = executable, "-m", self.module, *args, *kw2cmdline(**kwargs)
        print(*args, file=stderr)
        return Popen(args, stdout=PIPE, env=self.env)


def parse_outputs(stream: Iterable[bytes]) -> dict[str, str]:
    from yaml import safe_load

    data = []
    for line in stream:
        try:
            line = line.decode("utf-8").strip()
            if line.startswith("#"):
                data.append(line[1:].strip())
        except:
            pass
    return safe_load("\n".join(data))


def format_message(desc: tuple[str, str, str], dt: float, meta: dict[str, str]) -> str:
    src, dst, module = desc
    msg = [
        f"{src.rjust(2)}-{dst.ljust(2)}",
        module.ljust(10),
        f"{dt:.2f} s".rjust(7),
    ]
    try:
        if "travel" in meta:
            travel = f"{meta['travel']:.2f}".rjust(6)
        else:
            travel = "--.--".rjust(6)
        t_msg = f"travel {travel}"
        if "std" in meta:
            std = f"{meta['std']:.2f}".rjust(6)
            t_msg += f" +/- {std}"
        t_msg += " m"
        msg.append(t_msg)
        if "success_rate" in meta:
            msg.append(f"success rate: {meta["success_rate"]:.4f}")
        if "abort" in meta:
            msg.append("aborted: " + meta["abort"])
    except Exception as e:
        msg.append(str(e))
    return " | ".join(msg)


def exec(
    world: str,
    src: str,
    dst: str,
    module: str,
    env: dict[str, str],
    kw: dict[str, str],
    filter: Callable | None = None,
    *filter_args,
):
    args = []
    triplet = src, dst, module.split(".")[-1]
    if triplet in WATCH_LIST:
        args += ["-v"]
    t0 = time()
    proc = Python(module, **env)(world, *args, **kw, **SLICE)
    if filter:
        output = filter(proc.stdout, *filter_args)
    else:
        output = proc.stdout
    meta = parse_outputs(output)
    proc.wait()
    t1 = time()
    return triplet, t1 - t0, meta


def unpack_exec(args):
    return exec(*args)


PList = dict[str, list[float]]


def combinations(
    world: str,
    SRC: PList,
    DST: PList,
    save: bool = False,
    **kw: dict[str, str],
):
    if SRC is None or DST is None:
        raise ValueError("Missing SRC or DST in batch configuration")

    for src, p0 in SRC.items():
        for dst, p1 in DST.items():
            if src == dst:
                continue
            local_kw = dict(
                **kw,
                src=",".join(map(str, p0)),
                dst=",".join(map(str, p1)),
            )
            if save:
                local_kw["prefix"] = f"results/{src}-{dst}/"
            yield world, src, dst, local_kw


def factory(f):
    def outer(*args, **kwargs):
        def inner(world: str, src: str, dst: str, kw: dict[str, str]):
            for module, env, _kw, *extras in f(*args, src, dst, **kwargs):
                yield world, src, dst, module, env, {**kw, **_kw}, *extras

        return inner

    return outer


@factory
def Bug(demo: bool, *_):
    for i, n in enumerate([1, 2, 0]):
        for d in "LR":
            if demo:
                r = 0.25 + 0.2 * i
                kw = dict(resolution=0.025, radius=r, stepLength=0.05)
                if n == 1:
                    kw |= dict(noOverlap=True)
            else:
                kw = dict()
            env = dict()
            yield f"simulation.Bug{n}{d}", env, kw


@factory
def RandomWalk(N: int = 100, *_):
    for n in range(N):
        env = dict(SEED=str(n))
        kw = dict()
        yield f"simulation.RandomWalk", env, kw


@factory
def WaveFront(queue, src: str, dst: str, *_):
    env = dict()
    kw = dict()
    filter = WaveFrontFilter
    desc = f"{src.rjust(2)}-{dst.ljust(2)}"
    yield f"wavefront.WaveFront", env, kw, filter, desc, queue


def WaveFrontFilter(lines: Iterable[bytes], desc: str, queue=None):
    slot: int = queue.get() if queue is not None else 1
    fmt = "{l_bar}{bar}| {n:.3f}/{total_fmt} [{elapsed}]"
    prog = tqdm(
        total=1.0, desc=desc.rjust(9), position=slot, bar_format=fmt, **ProgOpts
    )
    for b in lines:
        line = b.decode("utf-8").strip()
        if line.startswith("#"):
            yield b
        else:
            try:
                _, _, dp = line.split(",")
                prog.update(float(dp))
            except:
                pass
    prog.close()
    if queue is not None:
        queue.put(slot)


KW = dict(radius=0.25, resolution=0.025, scale=2.0)
META = dict[str, dict[str, str]]()

Combs = Iterable[tuple[str, str, str, dict[str, str]]]
ProgOpts = dict(leave=False, dynamic_ncols=True, file=stdout)


def runBugAlgorithms(*combs: Combs, demo=False):
    # Bug algorithms
    tasks = list(chain(*map(Bug(demo), *zip(*combs))))
    progress = tqdm(total=len(tasks), desc="Bug Algorithms", **ProgOpts)
    with Pool(max(1, cpu_count())) as pool:
        for triplet, dt, meta in pool.imap_unordered(unpack_exec, tasks):
            META["-".join(triplet)] = meta
            progress.write(format_message(triplet, dt, meta))
            progress.update(1)
    progress.clear()


def runRandomWalk(*combs: Combs):
    # Random Walk algorithms
    prog1 = tqdm(combs, desc="RandomWalk", position=0, **ProgOpts)
    for world, src, dst, kw in prog1:
        tasks = list(RandomWalk(100)(world, src, dst, kw))
        desc = f"{src.rjust(2)}-{dst.ljust(2)}".rjust(10)
        prog2 = tqdm(total=len(tasks), desc=desc, position=1, **ProgOpts)
        with Pool(max(1, cpu_count())) as pool:
            t = 0.0
            D = list[float]()
            n = 0
            for _, dt, meta in pool.imap_unordered(unpack_exec, tasks):
                n += 1
                t += dt
                if "travel" in meta:
                    D.append(float(meta["travel"]))
                prog2.update(1)
            triplet = src, dst, "RandomWalk"
            mean = sum(D) / len(D)
            std = sum((x - mean) ** 2 for x in D) ** 0.5 / len(D)
            meta = dict(travel=mean, std=std, success_rate=len(D) / n)
            META["-".join(triplet)] = meta
            prog1.write(format_message(triplet, t, meta))


def runWaveFrontPool(*combs: Combs):
    # WaveFront algorithms
    slots = max(1, cpu_count() // 4)
    queue = Manager().Queue()
    tasks = list(chain(*map(WaveFront(queue), *zip(*combs))))
    for n in range(slots):
        queue.put(n + 1)
    progress = tqdm(total=len(tasks), desc="WaveFront", position=0, **ProgOpts)
    with Pool(slots) as pool:
        for triplet, dt, meta in pool.imap_unordered(unpack_exec, tasks):
            META["-".join(triplet)] = meta
            progress.write(format_message(triplet, dt, meta))
            progress.update(1)
    progress.clear()


def runWaveFrontSync(*combs: Combs):
    # WaveFront algorithms
    tasks = list(chain(*map(WaveFront(None), *zip(*combs))))
    progress = tqdm(total=len(tasks), desc="WaveFront", position=0, **ProgOpts)
    for triplet, dt, meta in map(unpack_exec, tasks):
        META["-".join(triplet)] = meta
        progress.write(format_message(triplet, dt, meta))
        progress.update(1)
    progress.clear()


if __name__ == "__main__":
    parser = ArgumentParser(prog="python3 -m batch")
    parser.add_argument("world", type=str, nargs=1)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    world = str(args.world[0])
    demo = bool(args.demo)
    # Generate world view
    Python("lib.world")(world, prefix="results/world", scale=4, **SLICE).wait()
    config = parse(stdin)
    SRC, DST = config["SRC"], config["DST"]
    runBugAlgorithms(*combinations(world, SRC, DST, **KW, save=True), demo=demo)
    if demo:
        exit(0)
    runRandomWalk(*combinations(world, SRC, DST, **KW))
    runWaveFrontPool(*combinations(world, SRC, DST, **KW, save=True))
    # Save meta
    meta_path = Path("results/meta.yaml")
    with meta_path.open("w") as f:
        from yaml import dump

        dump(META, f)
