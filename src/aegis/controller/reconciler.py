"""The recovery state machine.

One `tick()` is a complete, crash-safe step for one job:

    observe   -> backend status, progress heartbeat, health signals, node health
    correlate -> fold signals into at most one open incident
    decide    -> advance the state machine under the recovery lease
    act       -> record the action, then perform it

A recovery is:

    survey    which complete checkpoints are reachable on surviving storage
    plan      cheapest (checkpoint, placement, replica-per-rank) combination
    launch    ranks stage their shards from the planned replicas and restore
    verify    restore report proves the state matches what was saved, then the
              job must make real progress and commit a new checkpoint

Every side effect is written to the action ledger *before* it happens, and
every launch is preceded by asking the backend what is actually running.
Everything the next tick needs -- state, attempt, placement, node health,
checkpoints proven bad -- is in the database, so a controller that crashes at
any line resumes where it left off without repeating a launch.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from aegis import metrics
from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.checkpoint.catalog import CheckpointCatalog, Compatibility
from aegis.checkpoint.layout import Layout
from aegis.checkpoint.restore import RestoreReport
from aegis.checkpoint.tiers import StorageProfiles, get_profiles, read_conditions
from aegis.clock import Clock, SystemClock
from aegis.controller.planner import RecoveryPlan, RecoveryPlanner
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
ADVISORY_SIGNALS = frozenset({SignalKind.NODE_DRAINING, SignalKind.NODE_HEALTHY})
NODE_DOWN_SIGNALS = frozenset({SignalKind.NODE_UNHEALTHY, SignalKind.NODE_DRAINING})


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
    # Nodes the job may be placed on. Empty means world_size + 2 synthetic
    # nodes, enough to survive losing two.
    node_pool: tuple[str, ...] = ()
    storage_profile: str | None = None


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
        *,
        event_source: HealthEventSource | None = None,
        config: ReconcilerConfig | None = None,
        clock: Clock | None = None,
        rng: random.Random | None = None,
        profiles: StorageProfiles | None = None,
    ) -> None:
        self.repo = repo
        self.backend = backend
        self.event_source = event_source
        self.config = config or ReconcilerConfig()
        self.clock = clock or SystemClock()
        self.rng = rng or random.Random()
        self.profiles = profiles or get_profiles(self.config.storage_profile)
        self._catalogs: dict[str, CheckpointCatalog] = {}

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
            fence = self.repo.acquire_lease(job_id, self.config.holder_id, self.config.lease_ttl_seconds)
        except LeaseLost as exc:
            log.debug("skipping job %s: %s", job_id, exc)
            return TickResult(job_id, state, note=f"lease held elsewhere: {exc}")

        handle = self._current_handle(row)
        status = self._status(handle)
        self._reap(handle, status)
        progress = self._progress(spec, row, handle)
        # Before stall detection: the first heartbeat after a slow start must
        # count as progress, not be judged against the launch timestamp.
        if JobState(row.state) is not JobState.RECOVERING:
            self._record_progress(row, progress)
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

    def _on_pending(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, fence: int) -> TickResult:
        healthy = self._healthy_nodes(spec)
        if len(healthy) < spec.world_size:
            return TickResult(
                row.job_id,
                JobState.PENDING,
                note=f"waiting for capacity: {len(healthy)}/{spec.world_size} healthy nodes",
            )
        placement = list(row.placement or healthy[: spec.world_size])
        try:
            self._launch(row, spec, policy, fence, attempt=0, resume_step=None, placement=placement)
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
        self._set_state(row.job_id, JobState.RUNNING, message=f"initial launch on {','.join(placement)}")
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
            self._set_state(row.job_id, JobState.SUCCEEDED, finished_at=self.clock.now(), message="training completed")
            return TickResult(row.job_id, JobState.SUCCEEDED, changed=True, note="completed")

        actionable = [e for e in events if e.kind not in ADVISORY_SIGNALS]
        if not actionable:
            return TickResult(row.job_id, JobState.RUNNING, note="healthy")

        incident_id = self._open_incident(row, spec, policy, events)
        self.repo.mark_timeline(
            row.job_id, "detected", attempt=row.attempt, step=row.last_observed_step,
            signals=sorted({str(e.kind) for e in actionable}),
        )
        self._set_state(
            row.job_id,
            JobState.SUSPECT,
            suspect_since=self.clock.now(),
            message=f"correlating {len(actionable)} signal(s): " + ", ".join(sorted({str(e.kind) for e in actionable})),
        )
        log.warning(
            "job %s entering SUSPECT (incident %s): %s", row.job_id, incident_id, [str(e.kind) for e in actionable]
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

        # The run recovered on its own while we were correlating -- unless it
        # is sitting on a node known to be bad, in which case it is living on
        # borrowed time and we move it anyway.
        on_bad_node = bool(set(row.placement or []) & self.repo.unhealthy_nodes())
        if status.is_active and self._progressed_since(row, progress) and not on_bad_node:
            incident_id = self.repo.open_incident_id(row.job_id)
            if incident_id is not None:
                self.repo.close_incident(
                    incident_id, outcome=Outcome.ABANDONED, detail="job resumed progress during correlation window"
                )
            self._set_state(row.job_id, JobState.RUNNING, suspect_since=None, message="recovered without intervention")
            return TickResult(row.job_id, JobState.RUNNING, changed=True, note="self-healed")

        next_attempt = row.attempt + 1
        if next_attempt > policy.max_attempts:
            return self._exhaust(row, spec, policy, fence, reason=f"attempt limit reached ({policy.max_attempts})")

        delay = backoff_with_jitter(policy, next_attempt, self.rng)
        self._set_state(
            row.job_id,
            JobState.RECOVERING,
            recovery_started_at=row.recovery_started_at or since,
            next_attempt_at=now + timedelta(seconds=delay),
            suspect_since=None,
            message=f"recovery attempt {next_attempt} in {delay:.1f}s",
        )
        self.repo.mark_timeline(row.job_id, "recovering", attempt=next_attempt, backoff_s=round(delay, 3))
        log.warning("job %s -> RECOVERING attempt %s after %.1fs backoff", row.job_id, next_attempt, delay)
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
                    row, spec, policy, fence,
                    reason=f"recovery deadline exceeded ({elapsed:.0f}s > {policy.recovery_deadline_seconds:.0f}s)",
                )

        if row.next_attempt_at is not None and now < row.next_attempt_at:
            remaining = (row.next_attempt_at - now).total_seconds()
            return TickResult(row.job_id, JobState.RECOVERING, note=f"backoff {remaining:.1f}s")

        # An orphan of this exact attempt is already running: a previous
        # controller planned and launched it, then died. Adopt it and its
        # plan rather than planning again.
        adopted = next((h for h in self.backend.discover(spec) if h.action_id == next_action_id), None)
        if adopted is not None:
            return self._adopt(row, spec, policy, fence, attempt, adopted)

        incident_id = self.repo.open_incident_id(row.job_id)
        try:
            plan = self._plan(row, spec, policy, incident_id)
        except OSError as exc:
            # Storage being briefly unreadable is not evidence that any
            # checkpoint is bad, so this must not consume a recovery attempt.
            retry_at = now + timedelta(seconds=max(5.0, policy.backoff_base_seconds))
            self._set_state(
                row.job_id, JobState.RECOVERING, next_attempt_at=retry_at,
                message=f"checkpoint storage unavailable, retrying: {exc}",
            )
            log.warning("job %s: storage unavailable during survey: %s", row.job_id, exc)
            return TickResult(row.job_id, JobState.RECOVERING, note="storage retry")

        if incident_id is not None:
            self.repo.update_incident(incident_id, rejected_checkpoints=plan.rejected)
            self.repo.append_incident(incident_id, "plans", {"attempt": attempt, **plan.to_record()})
        for record in plan.rejected:
            for reason in record.get("reasons", []):
                metrics.CHECKPOINTS_REJECTED.labels(spec.name, str(reason)).inc()

        if plan.blocked:
            self._set_state(row.job_id, JobState.RECOVERING, message=f"waiting: {plan.reason}")
            return TickResult(row.job_id, JobState.RECOVERING, note=plan.reason)
        if plan.chosen is None:
            # Checkpoints exist but none is recoverable. Restarting from
            # scratch here would silently discard real progress, so this is
            # a bounded failure instead.
            return self._exhaust(row, spec, policy, fence, reason=plan.reason)

        chosen = plan.chosen
        restore = plan.restore_plan()
        self.repo.mark_timeline(
            row.job_id, "planned", attempt=attempt, step=chosen.step, placement=chosen.placement,
            estimate_s=round(chosen.cost.total_s, 3), summary=plan.reason,
        )
        try:
            self._launch(
                row, spec, policy, fence,
                attempt=attempt,
                resume_step=chosen.step,
                placement=chosen.placement,
                restore_plan=restore.to_json() if restore and restore.step is not None else None,
            )
        except LaunchBlocked as exc:
            # No attempt consumed: the budget measures recovery tries, not
            # ticks spent waiting for a stuck process to die.
            log.warning("job %s cannot relaunch yet: %s", row.job_id, exc)
            self._set_state(row.job_id, JobState.RECOVERING, message=f"blocked: {exc}")
            return TickResult(row.job_id, JobState.RECOVERING, note=f"blocked: {exc}")
        except LaunchFailed as exc:
            self._set_state(
                row.job_id, JobState.SUSPECT, attempt=attempt, suspect_since=now,
                message=f"attempt {attempt} failed to launch: {exc}",
            )
            return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="launch failed")

        return self._enter_verifying(row, spec, policy, attempt, chosen.step, chosen.placement, incident_id)

    def _adopt(
        self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, fence: int, attempt: int, handle: RunHandle
    ) -> TickResult:
        metrics.DUPLICATE_LAUNCHES_PREVENTED.labels(spec.name).inc()
        log.info("job %s attempt %s already running (%s); adopting", row.job_id, attempt, handle.reference)
        self.repo.begin_action(row.job_id, attempt, ActionKind.LAUNCH, fence, holder=self.config.holder_id)
        self.repo.mark_action(handle.action_id, ActionState.STARTED, handle=handle)
        run_id = handle.extra.get("run_id") or row.run_id
        self.repo.update_job(row.job_id, run_id=run_id, last_progress_at=self.clock.now())
        placement = handle.extra.get("placement") or row.placement or []
        resume_step = handle.extra.get("resume_step")
        return self._enter_verifying(
            row, spec, policy, attempt, resume_step, placement, self.repo.open_incident_id(row.job_id)
        )

    def _enter_verifying(
        self,
        row: JobRow,
        spec: JobSpec,
        policy: RecoveryPolicy,
        attempt: int,
        resume_step: int | None,
        placement: list[str],
        incident_id: int | None,
    ) -> TickResult:
        now = self.clock.now()
        if incident_id is not None:
            self.repo.update_incident(incident_id, chosen_step=resume_step, attempt=attempt)
        metrics.RECOVERY_ATTEMPTS.labels(spec.name, policy.policy_id).inc()
        self._set_state(
            row.job_id,
            JobState.VERIFYING,
            attempt=attempt,
            placement=list(placement),
            resume_step=resume_step,
            verify_baseline_step=resume_step or 0,
            verify_started_at=now,
            next_attempt_at=None,
            # Progress restarts from the restore point. Without this, replayed
            # steps look like no progress and the stall detector fires.
            last_observed_step=resume_step or 0,
            last_progress_at=now,
            message=(
                f"attempt {attempt} launched from "
                f"{'step ' + str(resume_step) if resume_step is not None else 'scratch'} on {','.join(placement)}"
            ),
        )
        log.warning("job %s -> VERIFYING attempt %s from step %s on %s", row.job_id, attempt, resume_step, placement)
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

        It is successful once the restore report shows the state matches what
        was saved, the job has taken `required_progress_steps` new steps past
        the restore point, and a fresh checkpoint has been committed.
        """
        self._record_progress(row, progress)
        now = self.clock.now()
        baseline = row.verify_baseline_step or 0
        target = baseline + policy.required_progress_steps
        report = self._restore_report(spec, row)

        if report is not None and not report.ok:
            return self._restore_failed(row, spec, policy, report)
        if report is not None and self.repo.mark_timeline(
            row.job_id, "restored", attempt=row.attempt, step=report.step,
            stage_s=round(report.stage_seconds, 3), load_s=round(report.restore_seconds, 3),
            sources=[r.staged.get("source") if r.staged else None for r in report.ranks],
        ):
            if (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
                self.repo.append_incident(
                    incident_id, "restore_reports", {"attempt": row.attempt, **_report_record(report)}
                )

        if status.phase is RunPhase.FAILED or any(e.kind not in ADVISORY_SIGNALS for e in events):
            detail = status.message or "signals during verification"
            log.warning("job %s failed during verification: %s", row.job_id, detail)
            self._open_incident(row, spec, policy, events)
            self.repo.mark_timeline(row.job_id, "failed_during_verification", attempt=row.attempt, detail=detail)
            self._set_state(
                row.job_id, JobState.SUSPECT, suspect_since=now,
                message=f"attempt {row.attempt} failed verification: {detail}",
            )
            return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="verification failed")

        if progress is not None and progress.step > baseline:
            self.repo.mark_timeline(row.job_id, "training_resumed", attempt=row.attempt, step=progress.step)
            incident = self._incident(row.job_id)
            if incident is not None and incident.step_at_failure is not None and progress.step >= incident.step_at_failure:
                self.repo.mark_timeline(row.job_id, "caught_up", attempt=row.attempt, step=progress.step)

        steps_ok = progress is not None and progress.step >= target
        checkpoint_ok = not policy.require_new_checkpoint or self._catalog(spec).has_committed_since(
            baseline, lost_nodes=self.repo.unhealthy_nodes()
        )

        # The training script exits 0 only after reaching total_steps, which
        # is stronger evidence than any step count: a recovery from the final
        # checkpoint has no further steps to take.
        if status.phase is RunPhase.SUCCEEDED:
            self._finish_recovery(row, spec, policy, note="job completed after recovery")
            self._set_state(row.job_id, JobState.SUCCEEDED, finished_at=now, message="training completed after recovery")
            return TickResult(row.job_id, JobState.SUCCEEDED, changed=True, note="completed")

        if steps_ok and checkpoint_ok:
            self._finish_recovery(row, spec, policy, note="verified progress")
            self._set_state(
                row.job_id, JobState.RUNNING,
                verify_baseline_step=None, verify_started_at=None, recovery_started_at=None,
                message=f"recovery verified at step {progress.step if progress else target}",
            )
            return TickResult(row.job_id, JobState.RUNNING, changed=True, note="recovery verified")

        if row.verify_started_at is not None:
            waited = (now - row.verify_started_at).total_seconds()
            if waited > policy.verification_timeout_seconds:
                self._open_incident(row, spec, policy, [])
                self._set_state(
                    row.job_id, JobState.SUSPECT, suspect_since=now,
                    message=f"verification timed out after {waited:.0f}s",
                )
                return TickResult(row.job_id, JobState.SUSPECT, changed=True, note="verify timeout")

        current = progress.step if progress else baseline
        return TickResult(
            row.job_id,
            JobState.VERIFYING,
            note=f"step {current}/{target}, new checkpoint={'yes' if checkpoint_ok else 'no'}",
        )

    def _restore_failed(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, report: RestoreReport) -> TickResult:
        """The run could not restore. Staging failures are storage trouble;
        verification failures mean the checkpoint itself cannot be trusted
        and is excluded from every future plan."""
        now = self.clock.now()
        failed = report.failed_checks()
        kind = SignalKind.STORAGE_UNAVAILABLE if report.outcome == "staging_failed" else SignalKind.CHECKPOINT_INVALID
        bad_steps = list(row.bad_steps or [])
        if report.outcome == "verification_failed" and report.step is not None and report.step not in bad_steps:
            bad_steps.append(report.step)
        event = HealthEvent(
            kind=kind, source="restore-report", observed_at=now, job_id=row.job_id,
            detail={"step": report.step, "outcome": report.outcome, "failed": failed[:10]},
        )
        self._open_incident(row, spec, policy, [event])
        if self.repo.mark_timeline(
            row.job_id, report.outcome, attempt=row.attempt, step=report.step, failed=failed[:10]
        ) and (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
            self.repo.append_incident(incident_id, "restore_reports", {"attempt": row.attempt, **_report_record(report)})
        log.warning("job %s attempt %s %s: %s", row.job_id, row.attempt, report.outcome, failed)
        self._set_state(
            row.job_id, JobState.SUSPECT, suspect_since=now, bad_steps=bad_steps,
            message=f"attempt {row.attempt} {report.outcome.replace('_', ' ')} at step {report.step}: "
            + "; ".join(failed[:3]),
        )
        return TickResult(row.job_id, JobState.SUSPECT, changed=True, note=report.outcome)

    # -------------------------------------------------------------- planning

    def _plan(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, incident_id: int | None) -> RecoveryPlan:
        lost = self.repo.unhealthy_nodes()
        layout = self._layout(spec)
        conditions = read_conditions(layout)
        views = self._catalog(spec).survey(
            lost_nodes=lost,
            compatibility=Compatibility(job_name=spec.name, world_size=spec.world_size),
            exclude_steps=set(row.bad_steps or []),
            max_candidates=policy.max_checkpoint_candidates,
            conditions=conditions,
        )
        incident = self.repo.get_incident(incident_id) if incident_id is not None else None
        step_at_failure = (
            incident.step_at_failure if incident is not None and incident.step_at_failure is not None
            else row.last_observed_step
        )
        return RecoveryPlanner(self.profiles, strategy=policy.planner).plan(
            views=views,
            world_size=spec.world_size,
            healthy_nodes=self._healthy_nodes(spec),
            previous_placement=list(row.placement or []),
            step_at_failure=step_at_failure,
            step_seconds=row.step_seconds,
            conditions=conditions,
        )

    def _node_pool(self, spec: JobSpec) -> list[str]:
        if self.config.node_pool:
            return list(self.config.node_pool)
        return [f"node-{i}" for i in range(spec.world_size + 2)]

    def _healthy_nodes(self, spec: JobSpec) -> list[str]:
        bad = self.repo.unhealthy_nodes()
        return [n for n in self._node_pool(spec) if n not in bad]

    def _layout(self, spec: JobSpec) -> Layout:
        return Layout(spec.checkpoint_dir)

    def _catalog(self, spec: JobSpec) -> CheckpointCatalog:
        if (catalog := self._catalogs.get(spec.job_id)) is None:
            catalog = CheckpointCatalog(self._layout(spec))
            self._catalogs[spec.job_id] = catalog
        return catalog

    def _restore_report(self, spec: JobSpec, row: JobRow) -> RestoreReport | None:
        path = self.backend.restore_report_path(spec, row.attempt)
        if not path:
            return None
        report = RestoreReport.load(Path(path))
        if report is None or report.run_id != row.run_id or report.outcome == "fresh":
            return None
        return report

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
        placement: list[str],
        restore_plan: str | None = None,
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
                log.info("job %s attempt %s already running (action %s); adopting", row.job_id, attempt, action.action_id)
                self.repo.mark_action(action.action_id, ActionState.STARTED, handle=live[0])
                return live[0]
            if action.state in (ActionState.COMPLETED, ActionState.FAILED):
                raise LaunchFailed(f"attempt {attempt} already resolved as {action.state} with nothing running")
            # Recorded but never actually started: we crashed in the window
            # between ledger write and launch. Nothing is running, so retrying
            # under the same action id is safe and is not a duplicate.
            log.info("job %s attempt %s recorded but not running; completing the launch", row.job_id, attempt)

        # Anything else still alive for this job must go before we add a run.
        if survivors := self._terminate_strays(row, spec, fence, except_action=action.action_id):
            raise LaunchBlocked(f"{len(survivors)} run(s) still alive: " + ", ".join(h.reference for h in survivors))

        run_id = f"{row.job_id}-a{attempt}-{uuid.uuid4().hex[:6]}"
        bad = self.repo.unhealthy_nodes()
        request = LaunchRequest(
            spec=spec,
            action_id=action.action_id,
            attempt=attempt,
            run_id=run_id,
            resume_step=resume_step,
            excluded_nodes=tuple(sorted(bad)),
            placement=tuple(placement),
            restore_plan=restore_plan,
        )
        try:
            handle = self.backend.launch(request)
        except Exception as exc:
            self.repo.mark_action(action.action_id, ActionState.FAILED, detail=str(exc))
            log.exception("job %s attempt %s launch failed", row.job_id, attempt)
            raise LaunchFailed(str(exc)) from exc

        self.repo.mark_action(action.action_id, ActionState.STARTED, handle=handle)
        self.repo.update_job(row.job_id, run_id=run_id, placement=list(placement), last_progress_at=self.clock.now())
        row.run_id = run_id
        if (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
            self.repo.append_incident_action(
                incident_id,
                {
                    "kind": str(ActionKind.LAUNCH),
                    "attempt": attempt,
                    "resume_step": resume_step,
                    "run_id": run_id,
                    "placement": list(placement),
                    "excluded_nodes": list(request.excluded_nodes),
                },
            )
            self.repo.mark_timeline(row.job_id, "launched", attempt=attempt, run_id=run_id)
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
            self.repo.mark_action(handle.action_id, ActionState.COMPLETED, detail="terminated by controller")
        return survivors

    def _exhaust(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, fence: int, *, reason: str) -> TickResult:
        log.error("job %s exhausted: %s", row.job_id, reason)
        if survivors := self._terminate_strays(row, spec, fence):
            log.error(
                "job %s: %s run(s) could not be stopped while escalating: %s",
                row.job_id, len(survivors), ", ".join(h.reference for h in survivors),
            )

        action, created = self.repo.begin_action(row.job_id, row.attempt, ActionKind.ESCALATE, fence)
        if created:
            self.repo.mark_action(action.action_id, ActionState.COMPLETED, detail=reason)

        if (incident_id := self.repo.open_incident_id(row.job_id)) is not None:
            incident = self.repo.get_incident(incident_id)
            duration = (self.clock.now() - incident.opened_at).total_seconds() if incident else None
            self.repo.mark_timeline(row.job_id, "exhausted", attempt=row.attempt, reason=reason)
            self.repo.close_incident(incident_id, outcome=Outcome.EXHAUSTED, recovery_seconds=duration, detail=reason)
        metrics.RECOVERY_OUTCOMES.labels(spec.name, policy.policy_id, str(Outcome.EXHAUSTED)).inc()
        self._set_state(row.job_id, JobState.EXHAUSTED, finished_at=self.clock.now(), message=reason)
        return TickResult(row.job_id, JobState.EXHAUSTED, changed=True, note=reason)

    def _finish_recovery(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, *, note: str) -> None:
        incident_id = self.repo.open_incident_id(row.job_id)
        if incident_id is None:
            return
        incident = self.repo.get_incident(incident_id)
        if incident is None:
            return
        duration = (self.clock.now() - incident.opened_at).total_seconds()
        lost = None
        if incident.step_at_failure is not None:
            lost = max(0, incident.step_at_failure - (incident.chosen_step or 0))
        self.repo.mark_timeline(row.job_id, "verified", attempt=row.attempt)
        self.repo.close_incident(
            incident_id, outcome=Outcome.RECOVERED, recovery_seconds=duration, lost_steps=lost, detail=note
        )
        metrics.RECOVERY_OUTCOMES.labels(spec.name, policy.policy_id, str(Outcome.RECOVERED)).inc()
        metrics.RECOVERY_SECONDS.labels(spec.name, policy.policy_id).observe(duration)
        if lost is not None:
            metrics.LOST_STEPS.labels(spec.name).observe(lost)
        log.warning("job %s recovery verified in %.1fs (replayed %s steps)", row.job_id, duration, lost)

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
            detail=f"run ended: {status.phase}" + (f" ({status.message})" if status.message else ""),
        )

    def _status(self, handle: RunHandle | None) -> RunStatus:
        if handle is None:
            return RunStatus(RunPhase.GONE)
        try:
            return self.backend.status(handle)
        except Exception as exc:
            log.exception("status check failed for %s", handle.reference)
            return RunStatus(RunPhase.UNKNOWN, message=str(exc))

    def _progress(self, spec: JobSpec, row: JobRow, handle: RunHandle | None) -> ProgressSample | None:
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
        placement = set(row.placement or [])
        events: list[HealthEvent] = []

        if self.event_source is not None:
            watermark = row.events_watermark
            newest = watermark
            for event in self.event_source.poll(row.job_id):
                if watermark is not None and event.observed_at <= watermark:
                    continue
                newest = event.observed_at if newest is None else max(newest, event.observed_at)
                self._apply_node_event(event)
                # A node problem matters to this job only if the job is on it.
                if event.node and event.job_id in (None, "") and event.node not in placement:
                    continue
                events.append(event)
            if newest is not None and newest != watermark:
                self.repo.update_job(row.job_id, events_watermark=newest)
                row.events_watermark = newest

        supervised = JobState(row.state) in (JobState.RUNNING, JobState.VERIFYING)

        # Level-triggered: a node recorded as unhealthy is a reason to act for
        # as long as the job sits on it, whether or not the event that said so
        # was seen by this controller process.
        if supervised:
            for node in sorted(placement & self.repo.unhealthy_nodes()):
                events.append(
                    HealthEvent(
                        kind=SignalKind.NODE_UNHEALTHY, source="node-health", observed_at=now,
                        node=node, job_id=row.job_id, detail={"reason": "node recorded unhealthy"},
                    )
                )

        # A run that vanished is as much a failure as one that exited
        # non-zero, and is what a tick lost to a controller crash looks like.
        if status.phase is RunPhase.FAILED or (supervised and status.phase is RunPhase.GONE):
            events.append(
                HealthEvent(
                    kind=SignalKind.WORKER_EXIT, source="backend", observed_at=now, job_id=row.job_id,
                    detail={"exit_code": status.exit_code, "message": status.message or str(status.phase)},
                )
            )

        if supervised and status.is_active and not (progress is not None and progress.finished):
            # Before the current run has produced any heartbeat we are timing
            # startup (and staging), not a stall. `last_progress_at` is set at
            # launch, so it anchors both cases.
            timeout = policy.stall_timeout_seconds if progress is not None else self.config.startup_grace_seconds
            verdict = detect_stall(now=now, last_progress_at=row.last_progress_at, timeout_seconds=timeout)
            if verdict.stalled:
                events.append(
                    HealthEvent(
                        kind=SignalKind.PROGRESS_STALLED, source="progress-watcher", observed_at=now,
                        job_id=row.job_id,
                        detail={"seconds_since_progress": verdict.seconds_since_progress, "last_step": row.last_observed_step},
                    )
                )

        return self._deduplicate(row, spec, events)

    def _apply_node_event(self, event: HealthEvent) -> None:
        if not event.node:
            return
        if event.kind in NODE_DOWN_SIGNALS:
            reason = str(event.detail.get("reason") or event.kind)
            if self.repo.set_node_health(event.node, False, reason):
                log.warning("node %s marked unhealthy: %s", event.node, reason)
        elif event.kind is SignalKind.NODE_HEALTHY:
            if self.repo.set_node_health(event.node, True, "reported healthy"):
                log.info("node %s healthy again", event.node)

    def _deduplicate(self, row: JobRow, spec: JobSpec, events: list[HealthEvent]) -> list[HealthEvent]:
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
            scoped = _scoped(event, row.job_id)
            if self.repo.record_signal(incident_id, scoped):
                metrics.SIGNALS.labels(spec.name, str(event.kind)).inc()
                fresh.append(scoped)
        return fresh

    def _open_incident(self, row: JobRow, spec: JobSpec, policy: RecoveryPolicy, events: list[HealthEvent]) -> int:
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
            if self.repo.record_signal(incident_id, _scoped(event, row.job_id)):
                metrics.SIGNALS.labels(spec.name, str(event.kind)).inc()
        return incident_id

    def _incident(self, job_id: str):
        incident_id = self.repo.open_incident_id(job_id)
        return self.repo.get_incident(incident_id) if incident_id is not None else None

    @staticmethod
    def _workload_fingerprint(spec: JobSpec) -> dict:
        import torch

        return {
            "job_name": spec.name,
            "world_size": spec.world_size,
            "total_steps": spec.total_steps,
            "image": spec.image,
            "torch_version": torch.__version__,
        }

    # -------------------------------------------------------------- plumbing

    def _record_progress(self, row: JobRow, progress: ProgressSample | None) -> None:
        if progress is None or progress.step <= row.last_observed_step:
            return
        fields: dict = {"last_observed_step": progress.step, "last_progress_at": self.clock.now()}
        if progress.step_seconds:
            fields["step_seconds"] = progress.step_seconds
        self.repo.update_job(row.job_id, **fields)
        row.last_observed_step = progress.step
        row.last_progress_at = fields["last_progress_at"]

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
            metrics.JOB_STATE.labels(row.name, row.job_id, str(state)).set(1 if row.state == state else 0)
        metrics.CURRENT_STEP.labels(row.name, row.job_id).set(row.last_observed_step)


def _scoped(event: HealthEvent, job_id: str) -> HealthEvent:
    return HealthEvent(
        kind=event.kind, source=event.source, observed_at=event.observed_at,
        node=event.node, job_id=job_id, detail=event.detail,
    )


def _report_record(report: RestoreReport) -> dict:
    return {
        "run_id": report.run_id,
        "step": report.step,
        "outcome": report.outcome,
        "stage_seconds": report.stage_seconds,
        "restore_seconds": report.restore_seconds,
        "consistent_across_ranks": report.consistent_across_ranks,
        "ranks": [
            {"rank": r.rank, "node": r.node, "staged": r.staged, "checks": r.checks, "error": r.error}
            for r in report.ranks
        ],
    }
