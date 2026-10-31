"""Execution backend interface.

Aegis decides *whether* and *from where* to restart; a backend decides *how*.
Keeping that line sharp is what lets the same controller logic run against
local processes in a test and against Kubernetes Jobs in a cluster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aegis.models import JobSpec, RunHandle, RunStatus


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    spec: JobSpec
    action_id: str
    attempt: int
    run_id: str
    resume_step: int | None
    # Nodes NVSentinel has flagged; the backend translates these into whatever
    # placement exclusion it supports.
    excluded_nodes: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)


class ExecutionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def launch(self, request: LaunchRequest) -> RunHandle:
        """Start one run. Must be safe to call only once per action_id; the
        caller guarantees that via the action ledger."""

    @abstractmethod
    def status(self, handle: RunHandle) -> RunStatus: ...

    @abstractmethod
    def terminate(self, handle: RunHandle, *, grace_seconds: float = 20.0) -> None: ...

    @abstractmethod
    def discover(self, spec: JobSpec) -> list[RunHandle]:
        """Runs the backend currently believes are alive for this job.

        Consulted after a controller restart so the ledger can be reconciled
        against reality instead of trusted blindly.
        """

    def progress_path(self, spec: JobSpec) -> str:
        """Where the training script publishes its progress heartbeat."""
        raise NotImplementedError
