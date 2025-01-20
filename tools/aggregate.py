#!/usr/bin/env python3
# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Iterable
from argparse import ArgumentParser
from pathlib import Path
from sys import stderr, exit
from yaml import safe_load
from json import dump

parser = ArgumentParser()
parser.add_argument("root", type=Path, nargs=1)
root: Path = parser.parse_args().root[0]

if not root.is_dir():
    print(f"{dir} is not a directory", file=stderr)
    exit(1)


def dirs(d: Path):
    return (x for x in d.iterdir() if x.is_dir())


def files(d: Path):
    return (x for x in d.iterdir() if x.is_file())


class Trajectory(list[tuple[float, float, float]]):

    def __init__(self, data: Iterable[str]):
        super().__init__()
        meta = []
        for line in data:
            line = line.strip()
            if line.startswith("#"):
                meta.append(line[1:].strip())
            else:
                self.append(tuple(map(float, line.split(","))))
        self.meta: dict[str, any] = safe_load("\n".join(meta))

    def __str__(self):
        import numpy as np
        from base64 import b64encode

        return b64encode(np.array(self, dtype=np.float32).tobytes()).decode("ascii")


offset = dict(x=0, y=0, r=0)


def aggregate(*runs: Path):
    trajectories = {}
    for run in runs:
        try:
            t = Trajectory(run.open("rt"))
        except Exception as e:
            print(f"Error processing {run}: {e}", file=stderr)
            exit(1)
        trajectories[run.stem] = dict(
            id=run.stem,
            success="abort" not in t.meta,
            range=len(t),
            stride=0.1,
            offset=offset,
            color=["red", "green", "blue"][int(run.stem[-2])],
            buffer=str(t),
        )
    return trajectories


for d in dirs(root):
    runs = [f for f in files(d) if f.suffix == ".txt" and f.stem.startswith("Bug")]
    runs.sort(key=lambda p: p.stem)
    with (d.parent / (d.name + "-Bug-L.json")).open("wt") as f:
        trj = aggregate(*(r for r in runs if r.stem.endswith("L")))
        dump(dict(base="base", trajectories=trj), f, indent="\t")
    with (d.parent / (d.name + "-Bug-R.json")).open("wt") as f:
        trj = aggregate(*(r for r in runs if r.stem.endswith("R")))
        dump(dict(base="base", trajectories=trj), f, indent="\t")
