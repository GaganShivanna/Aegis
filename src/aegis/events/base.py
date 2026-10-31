"""Health event sources.

Aegis consumes signals; it does not produce them. NVSentinel owns node health,
the backend reports worker exits, and the progress watcher reports stalls.
Each arrives through this interface so the controller treats them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegis.models import HealthEvent


class HealthEventSource(ABC):
    name: str = "source"

    @abstractmethod
    def poll(self, job_id: str) -> list[HealthEvent]:
        """Return events observed since the last poll and consume them."""


class InMemoryEventSource(HealthEventSource):
    name = "memory"

    def __init__(self) -> None:
        self._pending: list[HealthEvent] = []

    def push(self, event: HealthEvent) -> None:
        self._pending.append(event)

    def poll(self, job_id: str) -> list[HealthEvent]:
        matching = [e for e in self._pending if e.job_id in (None, job_id)]
        self._pending = [e for e in self._pending if e not in matching]
        return matching


class CompositeEventSource(HealthEventSource):
    name = "composite"

    def __init__(self, *sources: HealthEventSource) -> None:
        self.sources = list(sources)

    def add(self, source: HealthEventSource) -> None:
        self.sources.append(source)

    def poll(self, job_id: str) -> list[HealthEvent]:
        events: list[HealthEvent] = []
        for source in self.sources:
            events.extend(source.poll(job_id))
        return events
