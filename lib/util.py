# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
# This module is a termination point. No imports should be made to other modules
# ==============================================================================
import sys
from functools import wraps
from typing import Iterable, Generic, TypeVar, Callable
from math import pi, fmod
from inspect import signature

T = TypeVar("T")


def own_attrs(cls: T) -> T:
    slots = signature(cls).parameters
    init = cls.__init__

    @wraps(init)
    def init_filter(self, *args, **kwargs):
        return init(self, *args, **{k: v for k, v in kwargs.items() if k in slots})

    cls.__init__ = init_filter
    return cls


def reflect(el: object):
    if isinstance(el, type):
        return el.__name__
    return repr(el)


def trace(fn: callable, name: str | None = None, file=sys.stderr):
    if name is None:
        name = fn.__name__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(name, end="(\n  ", file=file)
        print(
            *map(reflect, args),
            *(f"{k}={reflect(v)}" for k, v in kwargs.items()),
            sep=",\n  ",
            end="\n)\n",
            file=file,
            flush=True,
        )
        return fn(*args, **kwargs)

    return wrapper


def strip_parentheses(s: str) -> str:
    while True:
        for l, r in ("()", "[]", "{}", "<>"):
            s = s.strip()
            if s.startswith(l) and s.endswith(r):
                s = s[1:-1]
        else:
            break
    return s.strip()


class tuple_of(Generic[T]):
    def __init__(self, t: Callable[[str], T]):
        self.type = t

    def __call__(self, s: str) -> tuple[T]:
        return tuple(map(self.type, strip_parentheses(s).split(",")))

    def __repr__(self):
        return f"tuple_of({self.type.__name__})"


def repeat(action: Callable, *args, **kwargs):
    while True:
        yield action(*args, **kwargs)


def limited(it: Iterable, count: int):
    for i, item in enumerate(it):
        if i >= count:
            break
        yield item


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


from typing import TypeVar, Callable

T = TypeVar("T")


class Retry(Exception):

    def __call__(_, func: Callable[..., T]) -> Callable[..., T]:
        from functools import wraps

        def wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Retry:
                    pass

        return wraps(func)(wrapper)
