# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from argparse import ArgumentParser, Action, FileType
from sys import version_info
from dataclasses import dataclass
from typing import Any, Iterable, TypeVar, Generic, Callable, Literal
from functools import wraps

T = TypeVar("T")
major, minor, *_ = version_info
parser = ArgumentParser(prog=f"python{major}.{minor} -m <module>")
arguments: dict[str, "Argument"] = {}


def transformArgument(cls: T) -> T:
    @wraps(cls)
    def wrap(*args, opt_name: str = ..., **kw):
        return cls(__args__=args, __opt_name__=opt_name, **kw)

    return wrap


@transformArgument
@dataclass
class Argument(Generic[T]):

    __args__: list[str]
    __opt_name__: str = ...
    action: str | type[Action] = ...
    nargs: int | Literal["*", "?", "+"] | None = ...
    const: Any = ...
    default: Any = ...
    type: Callable[[str], T] | FileType = ...
    choices: Iterable[T] | None = ...
    required: bool = ...
    help: str | None = ...
    metavar: str | tuple[str, ...] | None = ...
    dest: str | None = ...
    version: str = ...

    @property
    def kwargs(self):
        return {
            k: w
            for k, w in self.__dict__.items()
            if not k.startswith("__") and w is not ...
        }

    def register(self, parser: ArgumentParser, name: str):
        if self.__opt_name__ is not ...:
            name = self.__opt_name__
        if name is not None:
            args = (*self.__args__, f"--{name.replace('_', '-')}")
        else:
            args = self.__args__
        parser.add_argument(*args, **self.kwargs)

    def retrieve(self, src: dict, name: str) -> tuple[str, T]:
        if self.__opt_name__ is not ... and self.__opt_name__ is not None:
            name = self.__opt_name__
        return src.get(name)


def register_arguments(**kw: Argument):
    for k, arg in kw.items():
        arg.register(parser, k)
    arguments.update(kw)
    return kw


def retrieve_arguments(kwargs: dict, /, **kw: Argument):
    dst = {}
    for k, arg in kw.items():
        val = arg.retrieve(kwargs, k)
        if val is not None:
            dst[k] = val
    return dst


# No arguments shall be registered once parse() is called
def parse():
    from .world import World

    # Debug flag always comes last
    register_arguments(
        debug=Argument(
            action="store_true",
            help="Toggle debug tools",
        ),
    )
    kwargs = vars(parser.parse_args())
    kwargs = retrieve_arguments(kwargs, **arguments)
    if kwargs.get("debug"):
        l = max(len(k) for k in kwargs)
        for k, v in kwargs.items():
            print(f"{k.rjust(l)} = {repr(v)}")
    return World.create(**kwargs)


def auto_parse(**extra_kwargs):
    def decorator(cls: T) -> T:
        init = cls.__init__

        @wraps(init)
        def wrapper(*args, **kwargs):
            kw = parse()
            kw.update(**kwargs)
            kw.update(**extra_kwargs)
            return init(*args, **kw)

        cls.__init__ = wrapper
        return cls

    return decorator
