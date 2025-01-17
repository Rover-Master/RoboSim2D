# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Iterable
from math import pi, fmod

def repeat(action: callable, *args, **kwargs):
    while True:
        yield action(*args, **kwargs)

def readline(stream: Iterable[bytes]) -> str:
    line: bytes = b""
    for byte in stream:
        line += byte
        if byte == b"\n":
            break
    return line.decode("utf-8")


def periodic(v, p: float = 2 * pi):
    res = fmod(v, p)
    p2 = p / 2
    if (res > +p2):
        return res - p
    if (res < -p2):
        return res + p
    return res

def angDiff(r1, r2):
    """
    Angle from r1 to r2, in radians
    """
    return periodic(r2 - r1)
