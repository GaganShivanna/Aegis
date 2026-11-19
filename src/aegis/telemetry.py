"""Append-only event log shared by the training job and the experiment harness.

Each event is one JSON line written with a single unbuffered `write` on an
O_APPEND descriptor. That keeps lines intact when several ranks write at once
and means the last events before a hard crash are not lost in a userspace
buffer -- which is exactly when they matter.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class MetricsLog:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self._fd: int | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    def emit(self, event: str, **fields: Any) -> None:
        if self._fd is None:
            return
        line = json.dumps({"t": time.time(), "event": event, **fields}, default=str) + "\n"
        os.write(self._fd, line.encode())

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def read_events(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    events = []
    for line in target.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events
