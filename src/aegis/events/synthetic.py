"""Synthetic health events.

Milestone 1 has no NVSentinel to listen to, so events are injected as JSON
lines on a file. The consumer tracks a byte offset, which means the CLI can
append events while the controller is running and nothing is replayed after a
controller restart.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis.events.base import HealthEventSource
from aegis.models import HealthEvent, SignalKind


def emit_event(
    path: str | os.PathLike[str],
    *,
    kind: SignalKind,
    job_id: str | None = None,
    node: str | None = None,
    source: str = "synthetic",
    detail: dict[str, Any] | None = None,
) -> HealthEvent:
    event = HealthEvent(
        kind=kind,
        source=source,
        observed_at=datetime.now(UTC),
        node=node,
        job_id=job_id,
        detail=detail or {},
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        fh.write(json.dumps(_to_payload(event)) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return event


class FileEventSource(HealthEventSource):
    name = "synthetic-file"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._offset = 0

    def poll(self, job_id: str) -> list[HealthEvent]:
        if not self.path.is_file():
            return []
        events: list[HealthEvent] = []
        with self.path.open("r") as fh:
            fh.seek(self._offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(_from_payload(json.loads(line)))
                except (ValueError, KeyError, TypeError):
                    continue
            self._offset = fh.tell()
        return [e for e in events if e.job_id in (None, "", job_id)]


def _to_payload(event: HealthEvent) -> dict[str, Any]:
    return {
        "kind": str(event.kind),
        "source": event.source,
        "observed_at": event.observed_at.isoformat(),
        "node": event.node,
        "job_id": event.job_id,
        "detail": event.detail,
    }


def _from_payload(payload: dict[str, Any]) -> HealthEvent:
    return HealthEvent(
        kind=SignalKind(payload["kind"]),
        source=payload.get("source", "synthetic"),
        observed_at=datetime.fromisoformat(payload["observed_at"]),
        node=payload.get("node"),
        job_id=payload.get("job_id"),
        detail=payload.get("detail", {}),
    )
