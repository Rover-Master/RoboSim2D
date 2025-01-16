# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from typing import Iterable

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
