"""The recovery state machine.

One `tick()` is a complete, crash-safe step for one job:

    observe  -> read backend status, progress heartbeat and health signals
    correlate-> fold signals into at most one open incident
    decide   -> advance the state machine under the recovery lease
    act      -> record the action, then perform it

Every side effect is written to the action ledger *before* it happens, and
every launch is preceded by asking the backend what is actually running. Those
two rules are what make "zero duplicate active jobs" hold across controller
crashes rather than merely in the happy path.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aegis import metrics
from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.checkpoint.store import CheckpointStore, StorageFault
from aegis.checkpoint.validator import (
    CheckpointValidator,
    Compatibility,
    ValidationResult,
)
from aegis.clock import Clock, SystemClock
from aegis.controller.policy import backoff_with_jitter
from aegis.controller.progress import ProgressWatcher, detect_stall
from aegis.events.base import HealthEventSource
from aegis.models import (
    ActionKind,
    ActionState,
    HealthEvent,
    JobSpec,
    JobState,
    Outcome,
    ProgressSample,
    RecoveryPolicy,
    RunHandle,
    RunPhase,
    RunStatus,
    SignalKind,
    deterministic_action_id,
)
from aegis.store.db import JobRow
from aegis.store.repository import LeaseLost, Repository, handle_from_json

log = logging.getLogger(__name__)

# Signals that describe a condition to watch rather than a failure to act on.
ADVISORY_SIGNALS = frozenset({SignalKind.NODE_DRAINING})


class LaunchFailed(RuntimeError):
    """An attempt was spent without producing a running job."""


class LaunchBlocked(RuntimeError):
    """Launching now would be unsafe. Costs no attempt; retry on a later tick.

    Raised when a previous run could not be confirmed dead. Starting a
    replacement in that state is exactly the duplicate-active-job outcome the
    controller exists to prevent, so it waits instead.
    """


@dataclass(slots=True)
class ReconcilerConfig:
    holder_id: str = field(default_factory=lambda: f"aegis-{uuid.uuid4().hex[:8]}")
    lease_ttl_seconds: float = 60.0
    terminate_grace_seconds: float = 20.0
    # Grace before a just-launched run is expected to show a heartbeat.
    startup_grace_seconds: float = 90.0


@dataclass(slots=True)
class TickResult:
    job_id: str
    state: JobState
    changed: bool = False
    note: str = ""


class Reconciler:
    def __init__(
        self,
        repo: Repository,
        backend: ExecutionBackend,
        checkpoint_store_factory,
        *,
        event_source: HealthEventSource | None = None,
        config: ReconcilerConfig | None = None,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.repo = repo
        self.backend = backend
        self.checkpoint_store_factory = checkpoint_store_factory
        self.event_source = event_source
        self.config = config or ReconcilerConfig()
        self.clock = clock or SystemClock()
        self.rng = rng or random.Random()
        self._excluded_nodes: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ tick

    def tick(self, job_id: str) -> TickResult:
        row = self.repo.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        state = JobState(row.state)
        if state in (JobState.SUCCEEDED, JobState.EXHAUSTED, JobState.STOPPED):
            return TickResult(job_id, state, note="terminal")

        spec = self.repo.spec_of(row)
        policy = self.repo.policy_of(row)

        try:
            fence = self.repo.acquire_lease(
                job_id, self.config.holder_id, self.config.lease_ttl_seconds
            )
        except LeaseLost as exc:
            log.debug("skipping job %s: %s", job_id, exc)
            return TickResult(job_id, state, note=f"lease held elsewhere: {exc}")

        handle = self._current_handle(row)
        status = self._status(handle)
        self._reap(handle, status)
        progress = self._progress(spec, row, handle)
        events = self._collect_events(row, spec, policy, status, progress)

        result = self._dispatch(row, spec, policy, fence, handle, status, progress, events)
        self._publish_metrics(job_id)
        return result

    def _dispatch(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        handle: RunHandle | None,
        status: RunStatus,
        progress: ProgressSample | None,
        events: list[HealthEvent],
    ) -> TickResult:
        state = JobState(row.state)
        match state:
            case JobState.PENDING:
                return self._on_pending(row, spec, policy, fence)
            case JobState.RUNNING:
                return self._on_running(row, spec, policy, fence, status, progress, events)
            case JobState.SUSPECT:
                return self._on_suspect(row, spec, policy, fence, status, progress, events)
            case JobState.RECOVERING:
                return self._on_recovering(row, spec, policy, fence, handle)
            case JobState.VERIFYING:
                return self._on_verifying(row, spec, policy, fence, status, progress, events)
            case _:
                return TickResult(row.job_id, state, note="no handler")

    # ---------------------------------------------------------------- states

    def _on_pending(
        self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, fence: int
    ) -> TickResult:
        try:
            self._launch(row, spec, policy, fence, attempt=0, resume_step=None)
        except LaunchBlocked as exc:
            log.warning("job %s launch blocked: %s", row.job_id, exc)
            return TickResult(row.job_id, JobState.PENDING, note=f"blocked: {exc}")
        except LaunchFailed as exc:
            # Failing to start is a failure like any other; route it through
            # the recovery budget rather than retrying unboundedly here.
            self._open_incident(row, spec, policy, [])
            self._set_state(
                row.job_id,
                JobState.SUSPECT,
                suspect_since=self.clock.now(),
                message=f"initial launch failed: {exc}",
            )
            return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="launch failed")
        self._set_state(row.job_id, JobState.RUNNING, message="initial launch")
        return TickResult(row.job_id, JobState.RUNNING, changed=True, note="launched")

    def _on_running(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        status: RunStatus,
        progress: ProgressSample | None,
        events: list[HealthEvent],
    ) -> TickResult:
        self._record_progress(row, progress)

        if status.phase is RunPhase.SUCCEEDED:
            self._set_state(
                row.job_id,
                JobState.SUCCEEDED,
                finished_at=self.clock.now(),
                message="training completed",
            )
            return TickResult(row.job_id, JobState.SUCCEEDED, changed=True, note="completed")

        actionable = [e for e in events if e.kind not in ADVISORY_SIGNALS]
        if not actionable:
            return TickResult(row.job_id, JobState.RUNNING, note="healthy")

        incident_id = self._open_incident(row, spec, policy, events)
        self._set_state(
            row.job_id,
            JobState.SUSPECT,
            suspect_since=self.clock.now(),
            message=f"correlating {len(actionable)} signal(s): "
            + ", ".join(sorted({str(e.kind) for e in actionable})),
        )
        log.warning(
            "job %s entering SUSPECT (incident %s): %s",
            row.job_id,
            incident_id,
            [str(e.kind) for e in actionable],
        )
        return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="signals observed")

    def _on_suspect(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        status: RunStatus,
        progress: ProgressSample | None,
        events: list[HealthEvent],
    ) -> TickResult:
        """Hold briefly so related signals land in one incident.

        A single bad node produces a node condition plus one worker exit per
        rank. Acting on the first one would start several recoveries for what
        is one failure.
        """
        self._record_progress(row, progress)
        now = self.clock.now()
        since = row.suspect_since or now
        waited = (now - since).total_seconds()

        if waited < policy.correlation_window_seconds:
            return TickResult(
                row.job_id,
                JobState.SUSPECT,
                note=f"correlating ({waited:.1f}/{policy.correlation_window_seconds:.1f}s)",
            )

        # The run recovered on its own while we were correlating.
        if status.is_active and self._progressed_since(row, progress):
            incident_id = self.repo.open_incident_id(row.job_id)
            if incident_id is not None:
                self.repo.close_incident(
                    incident_id,
                    outcome=Outcome.ABANDONED,
                    detail="job resumed progress during correlation window",
                )
            self._set_state(row.job_id, JobState.RUNNING, suspect_since=None, message="recovered without intervention")
            return TickResult(row.job_id, JobState.RUNNING, changed=True, note="self-healed")

        next_attempt = row.attempt + 1
        if next_attempt > policy.max_attempts:
            return self._exhaust(
                row,
                spec,
                policy,
                fence,
                reason=f"attempt limit reached ({policy.max_attempts})",
            )

        delay = backoff_with_jitter(policy, next_attempt, self.rng)
        self._set_state(
            row.job_id,
            JobState.RECOVERING,
            recovery_started_at=row.recovery_started_at or since,
            next_attempt_at=now + timedelta(seconds=delay),
            suspect_since=None,
            message=f"recovery attempt {next_attempt} in {delay:.1f}s",
        )
        log.warning(
            "job %s -> RECOVERING attempt %s after %.1fs backoff",
            row.job_id,
            next_attempt,
            delay,
        )
        return TickResult(row.job_id, JobState.RECOVERING, changed=True, note="recovery scheduled")

    def _on_recovering(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        handle: RunHandle | None,
    ) -> TickResult:
        now = self.clock.now()
        attempt = row.attempt + 1
        next_action_id = deterministic_action_id(row.job_id, attempt, ActionKind.LAUNCH)

        # Stop the old run first, always, even if we are still in backoff. The
        # exception is a run already belonging to the attempt we are about to
        # make: that is an orphan from a crashed controller, and killing it
        # would throw away work we are allowed to adopt.
        self._terminate_strays(row, spec, fence, except_action=next_action_id)

        if row.recovery_started_at is not None:
            elapsed = (now - row.recovery_started_at).total_seconds()
            if elapsed > policy.recovery_deadline_seconds:
                return self._exhaust(
                    row,
                    spec,
                    policy,
                    fence,
                    reason=f"recovery deadline exceeded ({elapsed:.0f}s > {policy.recovery_deadline_seconds:.0f}s)",
                )

        if row.next_attempt_at is not None and now < row.next_attempt_at:
            remaining = (row.next_attempt_at - now).total_seconds()
            return TickResult(row.job_id, JobState.RECOVERING, note=f"backoff {remaining:.1f}s")

        incident_id = self.repo.open_incident_id(row.job_id)

        try:
            chosen, rejected = self._select_checkpoint(spec, policy)
        except StorageFault as exc:
            # Storage being briefly unavailable is not evidence that the
            # checkpoint is bad, so this must not consume a recovery attempt.
            retry_at = now + timedelta(seconds=max(5.0, policy.backoff_base_seconds))
            self._set_state(
                row.job_id,
                JobState.RECOVERING,
                next_attempt_at=retry_at,
                message=f"checkpoint storage unavailable, retrying: {exc}",
            )
            log.warning("job %s: storage unavailable during selection: %s", row.job_id, exc)
            return TickResult(row.job_id, JobState.RECOVERING, note="storage retry")

        if incident_id is not None and rejected:
            self.repo.update_incident(
                incident_id,
                rejected_checkpoints=[
                    {"step": r.step, "reasons": [str(x) for x in r.reasons], "detail": r.detail}
                    for r in rejected
                ],
            )
        for result in rejected:
            for reason in result.reasons:
                metrics.CHECKPOINTS_REJECTED.labels(spec.name, str(reason)).inc()

        store = self.checkpoint_store_factory(spec)
        if chosen is None and store.list_steps():
            # Checkpoints exist but none survived validation. Restarting from
            # scratch here would silently discard real progress, so this is a
            # bounded failure instead.
            return self._exhaust(
                row,
                spec,
                policy,
                fence,
                reason=f"no valid checkpoint among {len(rejected)} candidate(s)",
            )

        resume_step = chosen.step if chosen else None
        try:
            self._launch(row, spec, policy, fence, attempt=attempt, resume_step=resume_step)
        except LaunchBlocked as exc:
            # No attempt consumed: the budget measures recovery tries, not
            # ticks spent waiting for a stuck process to die.
            log.warning("job %s cannot relaunch yet: %s", row.job_id, exc)
            self._set_state(row.job_id, JobState.RECOVERING, message=f"blocked: {exc}")
            return TickResult(row.job_id, JobState.RECOVERING, note=f"blocked: {exc}")
        except LaunchFailed as exc:
            # The attempt is spent. Record it so the budget shrinks and the
            # next tick schedules the following attempt with a longer backoff.
            self._set_state(
                row.job_id,
                JobState.SUSPECT,
                attempt=attempt,
                suspect_since=now,
                message=f"attempt {attempt} failed to launch: {exc}",
            )
            return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="launch failed")

        if incident_id is not None:
            self.repo.update_incident(incident_id, chosen_step=resume_step, attempt=attempt)

        metrics.RECOVERY_ATTEMPTS.labels(spec.name, policy.policy_id).inc()
        self._set_state(
            row.job_id,
            JobState.VERIFYING,
            attempt=attempt,
            verify_baseline_step=resume_step or 0,
            verify_started_at=now,
            next_attempt_at=None,
            message=f"attempt {attempt} launched from step {resume_step if resume_step is not None else 'scratch'}",
        )
        log.warning(
            "job %s -> VERIFYING attempt %s from step %s",
            row.job_id,
            attempt,
            resume_step,
        )
        return TickResult(row.job_id, JobState.VERIFYING, changed=True, note="relaunched")

    def _on_verifying(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        status: RunStatus,
        progress: ProgressSample | None,
        events: list[HealthEvent],
    ) -> TickResult:
        """Recovery is not successful because a process started.

        It is successful once the job has taken `required_progress_steps` new
        steps past the restore point and written a fresh complete checkpoint.
        """
        self._record_progress(row, progress)
        now = self.clock.now()
        baseline = row.verify_baseline_step or 0
        target = baseline + policy.required_progress_steps

        if status.phase is RunPhase.FAILED or any(
            e.kind not in ADVISORY_SIGNALS for e in events
        ):
            detail = status.message or "signals during verification"
            log.warning("job %s failed during verification: %s", row.job_id, detail)
            self._open_incident(row, spec, policy, events)
            self._set_state(
                row.job_id,
                JobState.SUSPECT,
                suspect_since=now,
                message=f"attempt {row.attempt} failed verification: {detail}",
            )
            return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="verification failed")

        steps_ok = progress is not None and progress.step >= target
        checkpoint_ok = not policy.require_new_checkpoint or self._has_new_checkpoint(
            spec, policy, baseline
        )

        if status.phase is RunPhase.SUCCEEDED and steps_ok:
            self._finish_recovery(row, spec, policy, note="job completed after recovery")
            self._set_state(
                row.job_id,
                JobState.SUCCEEDED,
                finished_at=now,
                message="training completed after recovery",
            )
            return TickResult(row.job_id, JobState.SUCCEEDED, changed=True, note="completed")

        if steps_ok and checkpoint_ok:
            self._finish_recovery(row, spec, policy, note="verified progress")
            self._set_state(
                row.job_id,
                JobState.RUNNING,
                verify_baseline_step=None,
                verify_started_at=None,
                recovery_started_at=None,
                message=f"recovery verified at step {progress.step if progress else target}",
            )
            return TickResult(row.job_id, JobState.RUNNING, changed=True, note="recovery verified")

        if row.verify_started_at is not None:
            waited = (now - row.verify_started_at).total_seconds()
            if waited > policy.verification_timeout_seconds:
                self._open_incident(row, spec, policy, [])
                self._set_state(
                    row.job_id,
                    JobState.SUSPECT,
                    suspect_since=now,
                    message=f"verification timed out after {waited:.0f}s",
                )
                return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="verify timeout")

        current = progress.step if progress else baseline
        return TickResult(
            row.job_id,
            JobState.VERIFYING,
            note=f"step {current}/{target}, new checkpoint={'yes' if checkpoint_ok else 'no'}",
        )

    # --------------------------------------------------------------- actions

    def _launch(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        *,
        attempt: int,
        resume_step: int | None,
    ) -> RunHandle | None:
        """Launch attempt `attempt`, at most once, ever.

        The ledger entry is written first. If it already exists we are either
        resuming after a crash or racing a replica, and the only safe move is
        to look at what is actually running before deciding.
        """
        self.repo.assert_lease(row.job_id, self.config.holder_id, fence)
        action, created = self.repo.begin_action(
            row.job_id, attempt, ActionKind.LAUNCH, fence, holder=self.config.holder_id
        )

        if not created:
            live = [h for h in self.backend.discover(spec) if h.action_id == action.action_id]
            if live:
                metrics.DUPLICATE_LAUNCHES_PREVENTED.labels(spec.name).inc()
                log.info(
                    "job %s attempt %s already running (action %s); adopting",
                    row.job_id,
                    attempt,
                    action.action_id,
                )
                self.repo.mark_action(action.action_id, ActionState.STARTED, handle=live[0])
                return live[0]
            if action.state in (ActionState.COMPLETED, ActionState.FAILED):
                raise LaunchFailed(
                    f"attempt {attempt} already resolved as {action.state} with nothing running"
                )
            # Recorded but never actually started: we crashed in the window
            # between ledger write and launch. Nothing is running, so retrying
            # under the same action id is safe and is not a duplicate.
            log.info(
                "job %s attempt %s recorded but not running; completing the launch",
                row.job_id,
                attempt,
            )

        # Anything else still alive for this job must go before we add a run.
        if survivors := self._terminate_strays(row, spec, fence, except_action=action.action_id):
            raise LaunchBlocked(
                f"{len(survivors)} run(s) still alive: "
                + ", ".join(h.reference for h in survivors)
            )

        run_id = f"{row.job_id}-a{attempt}-{uuid.uuid4().hex[:6]}"
        request = LaunchRequest(
            spec=spec,
            action_id=action.action_id,
            attempt=attempt,
            run_id=run_id,
            resume_step=resume_step,
            excluded_nodes=tuple(sorted(self._excluded_nodes.get(row.job_id, set()))),
        )
        try:
            handle = self.backend.launch(request)
        except Exception as exc:
            self.repo.mark_action(action.action_id, ActionState.FAILED, detail=str(exc))
            log.exception("job %s attempt %s launch failed", row.job_id, attempt)
            raise LaunchFailed(str(exc)) from exc

        self.repo.mark_action(action.action_id, ActionState.STARTED, handle=handle)
        self.repo.update_job(row.job_id, run_id=run_id, last_progress_at=self.clock.now())
        if (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
            self.repo.append_incident_action(
                incident_id,
                {
                    "kind": str(ActionKind.LAUNCH),
                    "attempt": attempt,
                    "resume_step": resume_step,
                    "run_id": run_id,
                    "excluded_nodes": list(request.excluded_nodes),
                },
            )
        return handle

    def _terminate_strays(
        self, row: JobRow, spec: JobSpec, fence: int, *, except_action: str | None = None
    ) -> list[RunHandle]:
        """Stop every run for this job except the one we intend to keep.

        Returns the handles that could not be confirmed dead.
        """
        survivors: list[RunHandle] = []
        for handle in self.backend.discover(spec):
            if except_action and handle.action_id == except_action:
                continue
            log.info("terminating stray run %s for job %s", handle.reference, row.job_id)
            try:
                self.backend.terminate(handle, grace_seconds=self.config.terminate_grace_seconds)
            except Exception as exc:
                log.error("failed to terminate %s: %s", handle.reference, exc)
                survivors.append(handle)
                continue
            self.repo.mark_action(
                handle.action_id, ActionState.COMPLETED, detail="terminated by controller"
            )
        return survivors

    def _exhaust(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        fence: int,
        *,
        reason: str,
    ) -> TickResult:
        log.error("job %s exhausted: %s", row.job_id, reason)
        if survivors := self._terminate_strays(row, spec, fence):
            log.error(
                "job %s: %s run(s) could not be stopped while escalating: %s",
                row.job_id,
                len(survivors),
                ", ".join(h.reference for h in survivors),
            )

        action, created = self.repo.begin_action(
            row.job_id, row.attempt, ActionKind.ESCALATE, fence
        )
        if created:
            self.repo.mark_action(action.action_id, ActionState.COMPLETED, detail=reason)

        if (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
            incident = self.repo.get_incident(incident_id)
            duration = (
                (self.clock.now() - incident.opened_at).total_seconds() if incident else None
            )
            self.repo.close_incident(
                incident_id, outcome=Outcome.EXHAUSTED, recovery_seconds=duration, detail=reason
            )
        metrics.RECOVERY_OUTCOMES.labels(spec.name, policy.policy_id, str(Outcome.EXHAUSTED)).inc()
        self._set_state(
            row.job_id,
            JobState.EXHAUSTED,
            finished_at=self.clock.now(),
            message=reason,
        )
        return TickResult(row.job_id, JobState.EXHAUSTED, changed=True, note=reason)

    def _finish_recovery(
        self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, *, note: str
    ) -> None:
        incident_id = self.repo.open_incident_id(row.job_id)
        if incident_id is None:
            return
        incident = self.repo.get_incident(incident_id)
        if incident is None:
            return
        duration = (self.clock.now() - incident.opened_at).total_seconds()
        lost = None
        if incident.step_at_failure is not None and incident.chosen_step is not None:
            lost = max(0, incident.step_at_failure - incident.chosen_step)
        self.repo.close_incident(
            incident_id,
            outcome=Outcome.RECOVERED,
            recovery_seconds=duration,
            lost_steps=lost,
            detail=note,
        )
        metrics.RECOVERY_OUTCOMES.labels(spec.name, policy.policy_id, str(Outcome.RECOVERED)).inc()
        metrics.RECOVERY_SECONDS.labels(spec.name, policy.policy_id).observe(duration)
        if lost is not None:
            metrics.LOST_STEPS.labels(spec.name).observe(lost)
        log.warning(
            "job %s recovery verified in %.1fs (lost %s steps)", row.job_id, duration, lost
        )

    # ------------------------------------------------------------ observation

    def _current_handle(self, row: JobRow) -> RunHandle | None:
        launches = [a for a in self.repo.live_launches(row.job_id) if a.handle]
        if not launches:
            return None
        return handle_from_json(max(launches, key=lambda a: a.attempt).handle)

    def _reap(self, handle: RunHandle | None, status: RunStatus) -> None:
        """Close the ledger entry for a run that has finished.

        Without this the ledger keeps reporting old attempts as outstanding,
        and a controller restart would reason about a run that ended long ago.
        """
        if handle is None or status.is_active or status.phase is RunPhase.UNKNOWN:
            return
        self.repo.mark_action(
            handle.action_id,
            ActionState.COMPLETED,
            detail=f"run ended: {status.phase}"
            + (f" ({status.message})" if status.message else ""),
        )

    def _status(self, handle: RunHandle | None) -> RunStatus:
        if handle is None:
            return RunStatus(RunPhase.GONE)
        try:
            return self.backend.status(handle)
        except Exception as exc:
            log.exception("status check failed for %s", handle.reference)
            return RunStatus(RunPhase.UNKNOWN, message=str(exc))

    def _progress(
        self, spec: JobSpec, row: JobRow, handle: RunHandle | None
    ) -> ProgressSample | None:
        watcher = ProgressWatcher(self.backend.progress_path(spec))
        return watcher.read(expected_run_id=row.run_id or None)

    def _collect_events(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        status: RunStatus,
        progress: ProgressSample | None,
    ) -> list[HealthEvent]:
        """Gather external signals and derive internal ones, then deduplicate."""
        now = self.clock.now()
        events: list[HealthEvent] = []

        if self.event_source is not None:
            events.extend(self.event_source.poll(row.job_id))

        supervised = JobState(row.state) in (JobState.RUNNING, JobState.VERIFYING)
        # A run that vanished is as much a failure as one that exited
        # non-zero, and is what a tick lost to a controller crash looks like.
        if status.phase is RunPhase.FAILED or (supervised and status.phase is RunPhase.GONE):
            events.append(
                HealthEvent(
                    kind=SignalKind.WORKER_EXIT,
                    source="backend",
                    observed_at=now,
                    job_id=row.job_id,
                    detail={
                        "exit_code": status.exit_code,
                        "message": status.message or str(status.phase),
                    },
                )
            )

        if supervised and status.is_active:
            # Before the current run has produced any heartbeat we are timing
            # startup, not a stall, and startup is legitimately much slower.
            # `last_progress_at` is set at launch, so it anchors both cases.
            timeout = (
                policy.stall_timeout_seconds
                if progress is not None
                else self.config.startup_grace_seconds
            )
            verdict = detect_stall(
                now=now,
                last_progress_at=row.last_progress_at,
                timeout_seconds=timeout,
            )
            if verdict.stalled:
                events.append(
                    HealthEvent(
                        kind=SignalKind.PROGRESS_STALLED,
                        source="progress-watcher",
                        observed_at=now,
                        job_id=row.job_id,
                        detail={
                            "seconds_since_progress": verdict.seconds_since_progress,
                            "last_step": row.last_observed_step,
                        },
                    )
                )

        for event in events:
            if event.node and event.kind in (
                SignalKind.NODE_UNHEALTHY,
                SignalKind.NODE_DRAINING,
            ):
                self._excluded_nodes.setdefault(row.job_id, set()).add(event.node)

        return self._deduplicate(row, spec, events)

    def _deduplicate(
        self, row: JobRow, spec: JobSpec, events: list[HealthEvent]
    ) -> list[HealthEvent]:
        """Collapse repeats within the currently open incident.

        Signals seen before any incident exists are still returned (they are
        what opens one) but are persisted once the incident is created.
        """
        if not events:
            return []
        incident_id = self.repo.open_incident_id(row.job_id)
        if incident_id is None:
            seen: dict[str, HealthEvent] = {}
            for event in events:
                seen.setdefault(event.dedup_key, event)
            return list(seen.values())

        fresh: list[HealthEvent] = []
        for event in events:
            scoped = HealthEvent(
                kind=event.kind,
                source=event.source,
                observed_at=event.observed_at,
                node=event.node,
                job_id=row.job_id,
                detail=event.detail,
            )
            if self.repo.record_signal(incident_id, scoped):
                metrics.SIGNALS.labels(spec.name, str(event.kind)).inc()
                fresh.append(scoped)
        return fresh

    def _open_incident(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        events: list[HealthEvent],
    ) -> int:
        incident_id = self.repo.open_incident_id(row.job_id)
        if incident_id is None:
            incident_id = self.repo.open_incident(
                row.job_id,
                job_name=spec.name,
                attempt=row.attempt,
                policy_id=policy.policy_id,
                workload=self._workload_fingerprint(spec),
                step_at_failure=row.last_observed_step,
            )
        for event in events:
            scoped = HealthEvent(
                kind=event.kind,
                source=event.source,
                observed_at=event.observed_at,
                node=event.node,
                job_id=row.job_id,
                detail=event.detail,
            )
            if self.repo.record_signal(incident_id, scoped):
                metrics.SIGNALS.labels(spec.name, str(event.kind)).inc()
        return incident_id

    @staticmethod
    def _workload_fingerprint(spec: JobSpec) -> dict:
        """What the retrieval step in Milestone 3 will match "similar
        workloads" on."""
        import torch

        return {
            "job_name": spec.name,
            "world_size": spec.world_size,
            "total_steps": spec.total_steps,
            "image": spec.image,
            "torch_version": torch.__version__,
        }

    # ----------------------------------------------------------- checkpoints

    def _select_checkpoint(
        self, spec: JobSpec, policy: RecoveryPolicy
    ) -> tuple[ValidationResult | None, list[ValidationResult]]:
        store: CheckpointStore = self.checkpoint_store_factory(spec)
        validator = CheckpointValidator(store)
        return validator.select_latest_valid(
            Compatibility(job_name=spec.name, world_size=spec.world_size),
            max_candidates=policy.max_checkpoint_candidates,
        )

    def _has_new_checkpoint(self, spec: JobSpec, policy: RecoveryPolicy, baseline: int) -> bool:
        store: CheckpointStore = self.checkpoint_store_factory(spec)
        validator = CheckpointValidator(store)
        try:
            for step in store.list_steps():
                if step <= baseline:
                    break
                if validator.validate(
                    step, Compatibility(job_name=spec.name, world_size=spec.world_size)
                ).valid:
                    return True
        except StorageFault:
            return False
        return False

    # -------------------------------------------------------------- plumbing

    def _record_progress(self, row: JobRow, progress: ProgressSample | None) -> None:
        if progress is None or progress.step <= row.last_observed_step:
            return
        self.repo.update_job(
            row.job_id,
            last_observed_step=progress.step,
            last_progress_at=self.clock.now(),
        )
        row.last_observed_step = progress.step
        row.last_progress_at = self.clock.now()

    def _progressed_since(self, row: JobRow, progress: ProgressSample | None) -> bool:
        if progress is None or row.suspect_since is None or row.last_progress_at is None:
            return False
        return row.last_progress_at > row.suspect_since

    def _set_state(self, job_id: str, state: JobState, **fields) -> None:
        self.repo.update_job(job_id, state=state, **fields)

    def _publish_metrics(self, job_id: str) -> None:
        row = self.repo.get_job(job_id)
        if row is None:
            return
        for state in JobState:
            metrics.JOB_STATE.labels(row.name, row.job_id, str(state)).set(
                1 if row.state == state else 0
            )
        metrics.CURRENT_STEP.labels(row.name, row.job_id).set(row.last_observed_step)
