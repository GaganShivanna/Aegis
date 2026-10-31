from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.controller.progress import write_progress
from aegis.models import JobSpec, RunHandle, RunPhase, RunStatus


class FakeBackend(ExecutionBackend):
    """In-memory backend.

    Holds the state a real cluster would: which runs exist, and what phase
    each is in. Tests drive failures through it rather than through real
    processes, so the controller's decisions can be asserted deterministically.
    """

    name = "fake"

    def __init__(self, progress_file: str) -> None:
        self._progress_file = progress_file
        self.runs: dict[str, RunHandle] = {}
        self.phases: dict[str, RunStatus] = {}
        self.launch_log: list[LaunchRequest] = []
        self.terminated: list[str] = []
        self.fail_next_launch = False
        # Register the run, then raise: models the controller dying after the
        # cluster accepted the work but before the handle was recorded.
        self.crash_after_launch = False
        # Termination always fails: a wedged process group that will not die.
        self.unkillable = False

    # -- backend interface

    def launch(self, request: LaunchRequest) -> RunHandle:
        if self.fail_next_launch:
            self.fail_next_launch = False
            raise RuntimeError("injected launch failure")
        self.launch_log.append(request)
        handle = RunHandle(
            job_id=request.spec.job_id,
            action_id=request.action_id,
            attempt=request.attempt,
            backend=self.name,
            reference=f"fake-{request.attempt}-{request.run_id}",
            started_at=datetime.now(UTC),
            run_dir=str(Path(request.spec.workdir) / f"attempt-{request.attempt}"),
            extra={"run_id": request.run_id},
        )
        self.runs[handle.action_id] = handle
        self.phases[handle.action_id] = RunStatus(RunPhase.RUNNING)
        if self.crash_after_launch:
            self.crash_after_launch = False
            raise RuntimeError("controller died after the cluster accepted the launch")
        return handle

    def status(self, handle: RunHandle) -> RunStatus:
        return self.phases.get(handle.action_id, RunStatus(RunPhase.GONE))

    def terminate(self, handle: RunHandle, *, grace_seconds: float = 20.0) -> None:
        if self.unkillable:
            raise RuntimeError(f"process group for {handle.reference} survived SIGKILL")
        self.terminated.append(handle.action_id)
        self.runs.pop(handle.action_id, None)
        self.phases[handle.action_id] = RunStatus(RunPhase.GONE, message="terminated")

    def discover(self, spec: JobSpec) -> list[RunHandle]:
        return [
            h
            for action_id, h in self.runs.items()
            if h.job_id == spec.job_id and self.phases[action_id].is_active
        ]

    def progress_path(self, spec: JobSpec) -> str:
        return self._progress_file

    # -- test controls

    @property
    def current(self) -> RunHandle:
        active = [h for h in self.runs.values() if self.phases[h.action_id].is_active]
        if not active:
            raise AssertionError("no active run")
        return max(active, key=lambda h: h.attempt)

    @property
    def recovery_launches(self) -> list[LaunchRequest]:
        """Launches that were recovery attempts, excluding the initial run."""
        return [r for r in self.launch_log if r.attempt > 0]

    @property
    def active_count(self) -> int:
        return len([h for h in self.runs.values() if self.phases[h.action_id].is_active])

    def fail_current(self, exit_code: int = 1, message: str = "worker exited") -> None:
        handle = self.current
        self.phases[handle.action_id] = RunStatus(RunPhase.FAILED, exit_code, message)
        self.runs.pop(handle.action_id, None)

    def succeed_current(self) -> None:
        handle = self.current
        self.phases[handle.action_id] = RunStatus(RunPhase.SUCCEEDED, 0)
        self.runs.pop(handle.action_id, None)

    def report_progress(self, step: int, *, run_id: str | None = None, attempt: int = 0) -> None:
        write_progress(
            self._progress_file,
            step=step,
            run_id=run_id if run_id is not None else self.current.extra["run_id"],
            attempt=attempt,
        )

    def orphan_current(self) -> RunHandle:
        """Simulate the controller dying with a run still alive."""
        return self.current
