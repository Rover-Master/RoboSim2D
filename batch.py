# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from os import cpu_count
from sys import stdin, stdout, stderr, executable as python
from yaml import safe_load as parse
from typing import Iterable
from argparse import ArgumentParser
from multiprocessing import Pool
from subprocess import Popen, PIPE, DEVNULL
from time import time
from tqdm import tqdm

WATCH_LIST = [
    # ("SW", "C", "Bug1L"),
    # ("SW", "NE", "Bug1L"),
]


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
    module, src, dst = desc
    msg = [
        f"{src.rjust(2)}-{dst.ljust(2)}::{module.split('.')[-1].ljust(8)}",
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


def simulation(
    world: str,
    module: str,
    src: str,
    dst: str,
    p0: list[float],
    p1: list[float],
    *args: str,
):
    prefix = f"var/{src}-{dst}/"
    args = (
        python,
        "-m",
        f"simulation.{module}",
        world,
        f"--prefix={prefix}",
        f"--src={','.join(map(str, p0))}",
        f"--dst={','.join(map(str, p1))}",
        # "-v",
        # "--no-wait",
        "--resolution=0.025",
        "--slice=19,7,320,320",
        "--scale=2.0",
        *args,
    )
    triplet = src, dst, module.split(".")[-1]
    if triplet in WATCH_LIST:
        args += ("-v",)
    # print(*args)
    t0 = time()
    proc = Popen(args, stdout=PIPE, stderr=None)
    print(*args, file=stderr)
    meta = parse_outputs(proc.stdout)
    proc.wait()
    t1 = time()
    return (module, src, dst), t1 - t0, meta


def simulation_unpack(args):
    return simulation(*args)


PList = dict[str, list[float]]


def combinations(world: str, modules: list[str], SRC: PList, DST: PList, *args: str):
    if SRC is None or DST is None:
        raise ValueError("Missing SRC or DST in batch configuration")

    for src, p0 in SRC.items():
        for dst, p1 in DST.items():
            if src == dst:
                continue
            for module in modules:
                yield world, module, src, dst, p0, p1, *args


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("world", type=str, nargs=1)
    parser.add_argument("modules", type=str, nargs="*")
    args = parser.parse_args()
    world = str(args.world[0])
    # modules = list[str](args.modules)
    modules = [
        "Bug0L",
        "Bug0R",
        "Bug1L",
        "Bug1R",
        "Bug2L",
        "Bug2R",
    ]
    config = parse(stdin)
    args = ["--radius=0.25"]
    tasks = list(combinations(world, modules, config["SRC"], config["DST"], *args))
    progress = tqdm(
        total=len(tasks),
        desc="Batch Simulation",
        leave=False,
        dynamic_ncols=True,
        file=stdout,
    )
    with Pool(max(1, cpu_count() // 2)) as pool:
        for desc, dt, meta in pool.imap_unordered(simulation_unpack, tasks):
            progress.write(format_message(desc, dt, meta))
            progress.update(1)
    progress.clear()
