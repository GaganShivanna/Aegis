"""Domain types shared by the controller, store and backends."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    """Lifecycle of a training job as Aegis sees it."""

    PENDING = "pending"  # registered, nothing launched yet
    RUNNING = "running"  # healthy, making progress
    SUSPECT = "suspect"  # signals seen, correlating before acting
    RECOVERING = "recovering"  # checkpoint chosen, relaunch in flight
    VERIFYING = "verifying"  # relaunched, waiting for proof of progress
    SUCCEEDED = "succeeded"  # training finished
    EXHAUSTED = "exhausted"  # attempt limit or deadline hit; escalated
    STOPPED = "stopped"  # operator stopped it


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.EXHAUSTED, JobState.STOPPED})


class SignalKind(StrEnum):
    NODE_UNHEALTHY = "node_unhealthy"  # from NVSentinel node conditions
    NODE_DRAINING = "node_draining"
    WORKER_EXIT = "worker_exit"  # non-zero worker termination
    PROGRESS_STALLED = "progress_stalled"  # no new steps within timeout
    CHECKPOINT_INVALID = "checkpoint_invalid"
    STORAGE_UNAVAILABLE = "storage_unavailable"


class ActionKind(StrEnum):
    LAUNCH = "launch"
    TERMINATE = "terminate"
    ESCALATE = "escalate"


class ActionState(StrEnum):
    PENDING = "pending"  # recorded, not yet handed to the backend
    STARTED = "started"  # backend accepted it
    COMPLETED = "completed"
    FAILED = "failed"


class Outcome(StrEnum):
    RECOVERED = "recovered"
    EXHAUSTED = "exhausted"
    ABANDONED = "abandoned"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True, slots=True)
class HealthEvent:
    """A raw signal from NVSentinel, a worker, or the progress watcher.

    `dedup_key` collapses repeats: NVSentinel will report the same bad node on
    every poll, and a single node failure takes down several workers at once.
    All of that must become one incident.
    """

    kind: SignalKind
    source: str
    observed_at: datetime
    node: str | None = None
    job_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return f"{self.kind}:{self.node or '-'}:{self.job_id or '-'}"


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Knobs the learning layer will eventually rank. Versioned so an incident
    records which policy produced its outcome, and so rollback is possible."""

    name: str = "baseline"
    version: int = 1

    max_attempts: int = 3
    backoff_base_seconds: float = 10.0
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 300.0
    jitter_ratio: float = 0.2

    recovery_deadline_seconds: float = 3600.0
    # Correlation window: wait this long after the first signal so related
    # signals land in the same incident instead of triggering N recoveries.
    correlation_window_seconds: float = 5.0
    stall_timeout_seconds: float = 120.0

    required_progress_steps: int = 20
    require_new_checkpoint: bool = True
    verification_timeout_seconds: float = 900.0

    # How far back the selector may fall when the newest checkpoints are bad.
    max_checkpoint_candidates: int = 10
    keep_checkpoints: int = 5

    @property
    def policy_id(self) -> str:
        return f"{self.name}@v{self.version}"

    def backoff_for(self, attempt: int) -> float:
        """Exponential backoff for a 1-indexed attempt number, before jitter."""
        raw = self.backoff_base_seconds * (self.backoff_multiplier ** max(0, attempt - 1))
        return min(raw, self.backoff_max_seconds)


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Immutable description of the training job Aegis is guarding."""

    job_id: str
    name: str
    world_size: int
    checkpoint_dir: str
    workdir: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    total_steps: int = 1000
    namespace: str = "default"
    image: str = ""
    node_selector: dict[str, str] = field(default_factory=dict)
    policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)

    @staticmethod
    def new_id(name: str) -> str:
        return f"{name}-{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class RunHandle:
    """Backend-agnostic reference to one launched run of a job."""

    job_id: str
    action_id: str
    attempt: int
    backend: str
    reference: str  # pid for local, namespace/name for k8s
    started_at: datetime
    run_dir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class RunPhase(StrEnum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    GONE = "gone"  # backend has no record of it


@dataclass(frozen=True, slots=True)
class RunStatus:
    phase: RunPhase
    exit_code: int | None = None
    message: str = ""
    unhealthy_nodes: tuple[str, ...] = ()

    @property
    def is_active(self) -> bool:
        return self.phase in (RunPhase.PENDING, RunPhase.RUNNING)


@dataclass(frozen=True, slots=True)
class ProgressSample:
    step: int
    observed_at: datetime
    run_id: str = ""
    attempt: int = 0
    loss: float | None = None


def deterministic_action_id(job_id: str, attempt: int, kind: ActionKind) -> str:
    """Stable ID for (job, attempt, action kind).

    Two controller replicas, or one controller before and after a crash, derive
    the same ID for the same logical action. Combined with the primary key on
    the actions table, that is what prevents a duplicate launch.
    """
    raw = f"{job_id}|{attempt}|{kind}".encode()
    return hashlib.sha256(raw).hexdigest()[:32]
