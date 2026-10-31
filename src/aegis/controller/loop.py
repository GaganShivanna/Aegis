"""Control loop.

A plain polling loop rather than a watch-driven framework: recovery decisions
depend on elapsed time (backoff, correlation windows, stall timeouts) as much
as on events, so the loop has to run on a tick regardless of whether anything
arrived.
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass

from aegis.clock import Clock, SystemClock
from aegis.controller.reconciler import Reconciler, TickResult
from aegis.models import TERMINAL_STATES, JobState
from aegis.store.repository import Repository

log = logging.getLogger(__name__)

Observer = Callable[[TickResult], None]


@dataclass(slots=True)
class LoopStats:
    ticks: int = 0
    transitions: int = 0


class ControlLoop:
    def __init__(
        self,
        repo: Repository,
        reconciler: Reconciler,
        *,
        interval_seconds: float = 2.0,
        clock: Clock | None = None,
        observer: Observer | None = None,
    ) -> None:
        self.repo = repo
        self.reconciler = reconciler
        self.interval = interval_seconds
        self.clock = clock or SystemClock()
        self.observer = observer
        self.stats = LoopStats()
        self._stop = False

    def request_stop(self, *_: object) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    def run(
        self,
        *,
        job_ids: list[str] | None = None,
        until_terminal: bool = False,
        max_ticks: int | None = None,
    ) -> LoopStats:
        while not self._stop:
            targets = job_ids or [j.job_id for j in self.repo.list_jobs(active_only=True)]
            if until_terminal and job_ids and self._all_terminal(job_ids):
                break
            if not targets:
                if until_terminal:
                    break
                self.clock.sleep(self.interval)
                continue

            for job_id in targets:
                try:
                    result = self.reconciler.tick(job_id)
                except Exception:
                    log.exception("tick failed for job %s", job_id)
                    continue
                self.stats.ticks += 1
                if result.changed:
                    self.stats.transitions += 1
                if self.observer:
                    self.observer(result)

            if until_terminal and self._all_terminal(targets):
                break
            if max_ticks is not None and self.stats.ticks >= max_ticks:
                break
            self.clock.sleep(self.interval)
        return self.stats

    def _all_terminal(self, job_ids: list[str]) -> bool:
        for job_id in job_ids:
            row = self.repo.get_job(job_id)
            if row is None or JobState(row.state) not in TERMINAL_STATES:
                return False
        return True
