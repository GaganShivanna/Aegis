"""Progress heartbeat.

The training script publishes its step counter; the controller reads it to
tell "running" apart from "running and actually getting somewhere". The
heartbeat carries the run id so a stale file from a previous attempt is never
mistaken for progress by the current one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aegis.models import ProgressSample


@dataclass(frozen=True, slots=True)
class StallVerdict:
    stalled: bool
    seconds_since_progress: float | None
    detail: str = ""


def write_progress(
    path: str | os.PathLike[str],
    *,
    step: int,
    run_id: str,
    attempt: int,
    loss: float | None = None,
    finished: bool = False,
    step_seconds: float | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "run_id": run_id,
        "attempt": attempt,
        "loss": loss,
        "step_seconds": step_seconds,
        "finished": finished,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w") as fh:
        fh.write(json.dumps(payload))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


class ProgressWatcher:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def read(self, *, expected_run_id: str | None = None) -> ProgressSample | None:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            # A torn read of a file being replaced; the next tick will see it.
            return None

        run_id = payload.get("run_id", "")
        if expected_run_id and run_id != expected_run_id:
            return None
        try:
            observed_at = datetime.fromisoformat(payload["updated_at"])
        except (KeyError, ValueError):
            observed_at = datetime.now(UTC)
        return ProgressSample(
            step=int(payload.get("step", 0)),
            observed_at=observed_at,
            run_id=run_id,
            attempt=int(payload.get("attempt", 0)),
            loss=payload.get("loss"),
            step_seconds=payload.get("step_seconds"),
            finished=bool(payload.get("finished")),
        )

    def is_finished(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            return bool(json.loads(self.path.read_text()).get("finished"))
        except (OSError, ValueError):
            return False


def detect_stall(
    *,
    now: datetime,
    last_progress_at: datetime | None,
    timeout_seconds: float,
) -> StallVerdict:
    """A worker that is alive but not advancing is a failure Aegis must act on;
    it is the case plain Elastic cannot see."""
    if last_progress_at is None:
        return StallVerdict(False, None, "no progress observed yet")
    elapsed = (now - last_progress_at).total_seconds()
    if elapsed >= timeout_seconds:
        return StallVerdict(True, elapsed, f"no new step for {elapsed:.0f}s")
    return StallVerdict(False, elapsed)
