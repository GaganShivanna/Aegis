"""Filesystem primitives shared by every storage tier."""

from __future__ import annotations

import os
import re
from pathlib import Path

_STEP_DIR = re.compile(r"^step_(\d{12})$")


class StorageFault(OSError):
    """A storage tier could not be read or written right now.

    Distinct from "the data is bad": a fault is a reason to retry, never a
    reason to discard a checkpoint.
    """


def step_dir_name(step: int) -> str:
    return f"step_{step:012d}"


def step_from_dir(name: str) -> int | None:
    match = _STEP_DIR.match(name)
    return int(match.group(1)) if match else None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Not every filesystem permits fsync on a directory handle. The rename
        # is still atomic; only the durability barrier is lost.
        pass
    finally:
        os.close(fd)
