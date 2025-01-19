# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from os import cpu_count
from sys import stdin, stdout, stderr, executable
from yaml import safe_load as parse
from typing import Iterable
from itertools import chain
from argparse import ArgumentParser
from multiprocessing import Pool
from subprocess import Popen, PIPE
from time import time
from tqdm import tqdm

WATCH_LIST = [
    # ("SW", "C", "Bug1L"),
    # ("SW", "NE", "Bug1L"),
]

SLICE = {"slice": "19,7,320,320"}


class Python:
    def __init__(self, module: str, **env):
        self.module = module
        self.env = env

    def __call__(self, *args, **kwargs):
        kw = map(lambda k, v: f"--{k}={v}", *zip(*kwargs.items()))
        args = executable, "-m", self.module, *args, *kw
        print(*args, file=stderr)
        return Popen(args, stdout=PIPE, env=self.env)


def parse_outputs(stream: Iterable[bytes]):
    meta = dict[str, str]()
    for line in stream:
        try:
            line = line.decode("utf-8").strip()
            if line.startswith("#"):
                k, v = (s.strip() for s in line[1:].split("="))
                meta[k] = v
        except:
            pass
    return meta


def format_message(desc: tuple[str, str, str], dt: float, meta: dict[str, str]) -> str:
    src, dst, module = desc
    msg = [
        f"{src.rjust(2)}-{dst.ljust(2)}",
        module.ljust(10),
        f"{dt:.2f}s",
    ]
    try:
        travel = meta.get("travel", "--.--").rjust(6)
        t_msg = f"travel {travel}"
        if "std" in meta:
            std = meta.get("std", "N/A").ljust(6)
            t_msg += f" +/- {std}"
        t_msg += " m"
        msg.append(t_msg)
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
):
    args = []
    triplet = src, dst, module.split(".")[-1]
    if triplet in WATCH_LIST:
        args += ["-v"]
    t0 = time()
    proc = Python(module, **env)(world, *args, **kw, **SLICE)
    meta = parse_outputs(proc.stdout)
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
                local_kw["prefix"] = f"var/{src}-{dst}/"
            yield world, src, dst, local_kw


def factory(f):
    def outer(*args, **kwargs):
        def inner(world: str, src: str, dst: str, kw: dict[str, str]):
            for module, env, _kw in f(*args, **kwargs):
                yield world, src, dst, module, env, {**kw, **_kw}

        return inner

    return outer


@factory
def Bug():
    for n in range(3):
        for d in "LR":
            kw = dict()
            env = dict()
            yield f"simulation.Bug{n}{d}", env, kw


@factory
def RandomWalk(N: int = 100):
    for n in range(N):
        env = dict(SEED=str(n))
        kw = dict()
        yield f"simulation.RandomWalk", env, kw


KW = dict(radius=0.25, resolution=0.025, scale=2.0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("world", type=str, nargs=1)
    parser.add_argument("modules", type=str, nargs="*")
    args = parser.parse_args()
    world = str(args.world[0])
    # Generate world view
    Python("lib.world")(world, prefix="var/world", scale=4, **SLICE).wait()
    config = parse(stdin)
    SRC, DST = config["SRC"], config["DST"]
    # Bug algorithms
    tasks = list(
        chain(*map(Bug(), *zip(*combinations(world, SRC, DST, save=True, **KW))))
    )
    progress = tqdm(
        total=len(tasks),
        desc="Bug Algorithms",
        leave=False,
        dynamic_ncols=True,
        file=stdout,
    )
    with Pool(max(1, cpu_count() // 2)) as pool:
        for desc, dt, meta in pool.imap_unordered(unpack_exec, tasks):
            progress.write(format_message(desc, dt, meta))
            progress.update(1)
    progress.clear()
    # Random Walk algorithms
    prog1 = tqdm(
        list(combinations(world, SRC, DST, **KW)),
        desc="RandomWalk",
        leave=False,
        dynamic_ncols=True,
        file=stdout,
        position=0,
    )
    for world, src, dst, kw in prog1:
        tasks = list(RandomWalk(100)(world, src, dst, kw))
        prog2 = tqdm(
            total=len(tasks),
            desc=f"{src.rjust(2)}-{dst.ljust(2)}",
            leave=False,
            dynamic_ncols=True,
            file=stdout,
            position=1,
        )
        with Pool(max(1, cpu_count() // 2)) as pool:
            t = 0.0
            D = list[float]()
            n = 0
            for _, dt, meta in pool.imap_unordered(unpack_exec, tasks):
                n += 1
                t += dt
                if "travel" in meta:
                    D.append(float(meta["travel"]))
                prog2.update(1)
            mean = sum(D) / len(D)
            std = sum((x - mean) ** 2 for x in D) ** 0.5 / len(D)
            prog1.write(
                format_message(
                    (src, dst, "RandWalk"),
                    t / n,
                    dict(travel=f"{mean:.2f}", std=f"{std:.2f}"),
                )
            )
