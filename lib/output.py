# ==============================================================================
# Author : Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from .arguments import register_arguments, Argument

register_arguments(
    prefix=Argument(
        type=str,
        default=None,
        help="Path prefix for outputs. If not specified, no output will be generated.",
    ),
)

# ==============================================================================
# End argument injection
# ==============================================================================
from pathlib import Path
from dataclasses import dataclass, field
from .util import own_attrs


@own_attrs
@dataclass(frozen=False)
class Output:
    debug: bool

    prefix: str | None

    path_prefix: Path | None = field(init=False)
    file_prefix: str | None = field(init=False)

    def __post_init__(self):
        prefix = self.prefix
        if prefix is None:
            self.path_prefix = None
            self.file_prefix = None
        elif prefix.endswith("/"):
            self.path_prefix = Path(prefix)
            self.file_prefix = ""
        else:
            self.path_prefix = Path(prefix).parent
            self.file_prefix = Path(prefix).name
        if self.path_prefix:
            if not self.path_prefix.exists():
                self.path_prefix.mkdir(parents=True, exist_ok=True)
            elif self.path_prefix.is_file():
                raise FileExistsError(f"Prefix {self.path_prefix} is a file")

    def __call__(self, stem: str | None = None, suffix: str | None = None):
        if self.path_prefix is None or self.file_prefix is None:
            return None
        if self.file_prefix and stem:
            name = "-".join((self.file_prefix, stem))
        elif self.file_prefix or stem:
            name = self.file_prefix or stem
        else:
            name = self.path_prefix.name
        if suffix:
            if suffix.startswith("."):
                raise ValueError("suffix should not start with '.'")
            name = f"{name}.{suffix}"
        return self.path_prefix / name
