"""Checkpoint storage.

Local filesystem for now; the interface is deliberately narrow (list, resolve,
publish, delete) so an S3-compatible implementation can slot in without the
validator or writer changing.
"""

from __future__ import annotations

import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from aegis.checkpoint.manifest import MANIFEST_NAME

_STEP_DIR = re.compile(r"^step_(\d{12})$")


class StorageFault(OSError):
    """Raised by the fault-injecting wrapper to simulate a storage outage."""


def step_dir_name(step: int) -> str:
    return f"step_{step:012d}"


def step_from_dir(name: str) -> int | None:
    match = _STEP_DIR.match(name)
    return int(match.group(1)) if match else None


class CheckpointStore(ABC):
    @abstractmethod
    def list_steps(self) -> list[int]:
        """Checkpoint steps present on disk, newest first.

        Includes incomplete checkpoints: deciding what is usable is the
        validator's job, not the store's.
        """

    @abstractmethod
    def path_for(self, step: int) -> Path: ...

    @abstractmethod
    def begin(self, step: int) -> Path:
        """Create (or clear) the directory for an in-progress checkpoint."""

    @abstractmethod
    def publish(self, step: int, manifest_json: str) -> None:
        """Atomically make the checkpoint visible as complete."""

    @abstractmethod
    def delete(self, step: int) -> None: ...

    def has_manifest(self, step: int) -> bool:
        return (self.path_for(step) / MANIFEST_NAME).is_file()


class LocalCheckpointStore(CheckpointStore):
    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_steps(self) -> list[int]:
        if not self.base_dir.is_dir():
            return []
        steps = []
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and (step := step_from_dir(entry.name)) is not None:
                steps.append(step)
        return sorted(steps, reverse=True)

    def path_for(self, step: int) -> Path:
        return self.base_dir / step_dir_name(step)

    def begin(self, step: int) -> Path:
        directory = self.path_for(step)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        return directory

    def publish(self, step: int, manifest_json: str) -> None:
        directory = self.path_for(step)
        tmp = directory / f".{MANIFEST_NAME}.tmp"
        with tmp.open("w") as fh:
            fh.write(manifest_json)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, directory / MANIFEST_NAME)
        _fsync_dir(directory)
        _fsync_dir(self.base_dir)

    def delete(self, step: int) -> None:
        shutil.rmtree(self.path_for(step), ignore_errors=True)


class FaultInjectingStore(CheckpointStore):
    """Wraps a store and fails a configurable number of operations.

    Exists to exercise the "temporary storage failure" acceptance case: the
    controller must retry rather than treat an unreadable store as a missing
    checkpoint.
    """

    def __init__(self, inner: CheckpointStore, fail_times: int = 0) -> None:
        self.inner = inner
        self.remaining_failures = fail_times
        self.calls = 0

    def _maybe_fail(self, op: str) -> None:
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise StorageFault(f"injected storage fault during {op}")

    def list_steps(self) -> list[int]:
        self._maybe_fail("list_steps")
        return self.inner.list_steps()

    def path_for(self, step: int) -> Path:
        return self.inner.path_for(step)

    def begin(self, step: int) -> Path:
        self._maybe_fail("begin")
        return self.inner.begin(step)

    def publish(self, step: int, manifest_json: str) -> None:
        self._maybe_fail("publish")
        self.inner.publish(step, manifest_json)

    def delete(self, step: int) -> None:
        self.inner.delete(step)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        # Not all filesystems permit fsync on a directory handle; the rename
        # itself is still atomic, we just lose the durability barrier.
        pass
    finally:
        os.close(fd)
