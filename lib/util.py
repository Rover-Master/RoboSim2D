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


def dup(dst, src=None):
    from sys import stdout

    if src is None:
        src = stdout

    def p(*args, file=stdout, **kwargs):
        if file is src:
            print(*args, file=dst, **kwargs)
        print(*args, file=file, **kwargs)

    return p


def sign(x: int | float):
    return (x > 0) - (x < 0)


def periodic(v, p: float = 2 * pi):
    res = fmod(v, p)
    p2 = p / 2
    if res > +p2:
        return res - p
    if res < -p2:
        return res + p
    return res


def angDiff(r1, r2):
    """
    Angle from r1 to r2, in radians
    """
    return periodic(r2 - r1)


def sliceOffsets(x: int, y: int):
    s0 = (
        slice(None, -y) if y > 0 else slice(-y, None),
        slice(None, -x) if x > 0 else slice(-x, None),
    )
    s1 = (
        slice(None, +y) if y < 0 else slice(+y, None),
        slice(None, +x) if x < 0 else slice(+x, None),
    )
    return s0, s1
