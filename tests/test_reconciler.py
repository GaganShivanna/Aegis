"""Recovery behaviour, covering the acceptance cases in the PRD."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from aegis.checkpoint import CheckpointCatalog
from aegis.checkpoint.tiers import UNTHROTTLED
from aegis.controller.reconciler import Reconciler, ReconcilerConfig
from aegis.events.base import InMemoryEventSource
from aegis.models import HealthEvent, JobState, Outcome, SignalKind

POOL = ("node-0", "node-1", "node-2", "node-3")


def make_reconciler(repo, backend, clock, *, events=None, holder="ctl-a", pool=POOL):
    return Reconciler(
        repo,
        backend,
        event_source=events,
        config=ReconcilerConfig(holder_id=holder, startup_grace_seconds=30.0, node_pool=pool),
        clock=clock,
        rng=random.Random(0),
        profiles=UNTHROTTLED,
    )


def state(repo, spec) -> JobState:
    return JobState(repo.get_job(spec.job_id).state)


def drive_to_recovery(repo, spec, backend, clock, reconciler, policy, *, fail_at_step=50):
    """Launch, make progress, lose a worker, and land in RECOVERING."""
    reconciler.tick(spec.job_id)  # PENDING -> RUNNING
    backend.report_progress(fail_at_step)
    reconciler.tick(spec.job_id)

    backend.fail_current()
    reconciler.tick(spec.job_id)  # RUNNING -> SUSPECT
    clock.advance(policy.correlation_window_seconds + 1)
    reconciler.tick(spec.job_id)  # SUSPECT -> RECOVERING


# --------------------------------------------------------------- worker loss


def test_worker_loss_is_recovered_and_verified(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    drive_to_recovery(repo, spec, backend, clock, r, policy)
    assert state(repo, spec) is JobState.RECOVERING

    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING
    assert backend.launch_log[-1].resume_step == 40
    assert backend.active_count == 1

    # Progress alone is not yet success.
    backend.report_progress(55)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING

    backend.report_progress(60)
    checkpoints.write(50)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING

    incident = repo.list_incidents(spec.job_id)[0]
    assert incident.outcome == Outcome.RECOVERED
    assert incident.chosen_step == 40
    assert incident.lost_steps == 10


def test_new_checkpoint_is_required_even_when_steps_are_met(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    backend.report_progress(200)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING, "steps alone must not verify recovery"

    checkpoints.write(60)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING


def test_exactly_twenty_new_steps_are_required(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)
    checkpoints.write(50)

    backend.report_progress(59)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING

    backend.report_progress(60)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING


# ------------------------------------------------------- checkpoint fallback


def test_incomplete_checkpoint_is_skipped_for_an_older_one(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(30)
    checkpoints.write(40)
    checkpoints.write(50, skip_manifest=True)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=55)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    assert backend.launch_log[-1].resume_step == 40
    incident = repo.list_incidents(spec.job_id)[0]
    assert [c["step"] for c in incident.rejected_checkpoints] == [50]
    assert "no_manifest" in incident.rejected_checkpoints[0]["reasons"]


def test_all_checkpoints_invalid_escalates_instead_of_restarting_from_scratch(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40, skip_manifest=True)
    checkpoints.write(50, peer=False, drop_rank=1)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.EXHAUSTED
    assert "no recoverable checkpoint" in repo.get_job(spec.job_id).message
    assert backend.active_count == 0


def test_storage_failure_retries_without_consuming_an_attempt(
    repo, spec, backend, clock, checkpoints, policy, monkeypatch
):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    real_survey = CheckpointCatalog.survey
    failures = iter([OSError("object store timed out")])

    def flaky_survey(self, **kwargs):
        if (exc := next(failures, None)) is not None:
            raise exc
        return real_survey(self, **kwargs)

    monkeypatch.setattr(CheckpointCatalog, "survey", flaky_survey)

    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)

    r.tick(spec.job_id)  # storage is down
    assert state(repo, spec) is JobState.RECOVERING
    assert repo.get_job(spec.job_id).attempt == 0, "a storage outage is not a failed attempt"
    assert backend.recovery_launches == []

    clock.advance(policy.backoff_base_seconds + 1)
    r.tick(spec.job_id)  # storage is back
    assert state(repo, spec) is JobState.VERIFYING
    assert repo.get_job(spec.job_id).attempt == 1


# ------------------------------------------------------------ attempt budget


def test_repeated_failures_stop_at_the_attempt_limit(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)

    for attempt in range(1, policy.max_attempts + 1):
        backend.fail_current()
        r.tick(spec.job_id)
        clock.advance(policy.correlation_window_seconds + 1)
        r.tick(spec.job_id)
        clock.advance(policy.backoff_for(attempt) + 1)
        r.tick(spec.job_id)
        assert repo.get_job(spec.job_id).attempt == attempt

    backend.fail_current()
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.EXHAUSTED
    assert "attempt limit" in repo.get_job(spec.job_id).message
    assert backend.active_count == 0

    incidents = repo.list_incidents(spec.job_id)
    assert incidents[0].outcome == Outcome.EXHAUSTED


def test_backoff_grows_between_attempts(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)

    scheduled = repo.get_job(spec.job_id).next_attempt_at
    assert scheduled is not None
    wait = (scheduled - clock.now()).total_seconds()
    assert wait == pytest.approx(policy.backoff_for(1), abs=0.01)

    r.tick(spec.job_id)
    assert backend.recovery_launches == [], "must not launch during backoff"


def test_recovery_deadline_stops_the_job(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)

    clock.advance(policy.recovery_deadline_seconds + 10)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.EXHAUSTED
    assert "deadline" in repo.get_job(spec.job_id).message


# ------------------------------------------------------------- stalled worker


def test_stalled_worker_is_detected(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING

    # The process is alive and healthy by every signal except the one that
    # matters: it has stopped taking steps.
    clock.advance(policy.stall_timeout_seconds + 1)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.SUSPECT
    incident = repo.list_incidents(spec.job_id)[0]
    assert any(s["kind"] == SignalKind.PROGRESS_STALLED for s in incident.signals)


# -------------------------------------------------------- signal correlation


def test_related_signals_collapse_into_one_incident(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)

    # One bad node, reported repeatedly, plus the worker exits it caused.
    now = datetime.now(UTC)
    for _ in range(3):
        events.push(
            HealthEvent(SignalKind.NODE_UNHEALTHY, "nvsentinel", now, node="gpu-01", job_id=spec.job_id)
        )
    backend.fail_current()

    r.tick(spec.job_id)
    clock.advance(1)
    for _ in range(2):
        events.push(
            HealthEvent(SignalKind.NODE_UNHEALTHY, "nvsentinel", now, node="gpu-01", job_id=spec.job_id)
        )
    r.tick(spec.job_id)

    incidents = repo.list_incidents(spec.job_id)
    assert len(incidents) == 1, "one node failure must not open several incidents"
    kinds = [s["kind"] for s in incidents[0].signals]
    assert kinds.count(SignalKind.NODE_UNHEALTHY) == 1, "repeats must be deduplicated"


def test_unhealthy_node_is_excluded_from_the_relaunch(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)

    events.push(
        HealthEvent(
            SignalKind.NODE_UNHEALTHY,
            "nvsentinel",
            datetime.now(UTC),
            node="gpu-07",
            job_id=spec.job_id,
        )
    )
    backend.fail_current()
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    assert backend.launch_log[-1].excluded_nodes == ("gpu-07",)


def test_transient_signal_during_healthy_progress_is_abandoned(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)

    events.push(
        HealthEvent(
            SignalKind.STORAGE_UNAVAILABLE,
            "probe",
            datetime.now(UTC),
            job_id=spec.job_id,
        )
    )
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.SUSPECT

    # The job keeps making progress while we correlate, so there is nothing
    # to recover from.
    clock.advance(policy.correlation_window_seconds + 1)
    backend.report_progress(80)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.RUNNING
    assert backend.recovery_launches == [], "nothing was broken, so nothing was relaunched"
    assert repo.list_incidents(spec.job_id)[0].outcome == Outcome.ABANDONED


# ------------------------------------------------------------- completion


def test_successful_job_reaches_succeeded(repo, spec, backend, clock, policy):
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    r.tick(spec.job_id)
    backend.report_progress(500)
    r.tick(spec.job_id)
    backend.succeed_current()
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.SUCCEEDED
