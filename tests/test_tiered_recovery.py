"""Recovery across storage tiers and node loss, restore verification, and the
state the controller must keep in the database to survive its own crashes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aegis.controller.reconciler import LaunchFailed
from aegis.events.base import InMemoryEventSource
from aegis.models import HealthEvent, JobState, Outcome, SignalKind
from tests.test_reconciler import drive_to_recovery, make_reconciler, state


def node_down(node: str, *, at: datetime | None = None) -> HealthEvent:
    return HealthEvent(SignalKind.NODE_UNHEALTHY, "nvsentinel", at or datetime.now(UTC), node=node)


def relaunch(repo, spec, clock, r, policy, attempt: int = 1) -> None:
    clock.advance(policy.backoff_for(attempt) + 1)
    r.tick(spec.job_id)


# -------------------------------------------------------------- node loss


def test_node_loss_moves_the_rank_and_restores_from_its_peer_replica(
    repo, spec, backend, clock, checkpoints, layout, policy
):
    checkpoints.write(40)
    checkpoints.write(50)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)

    r.tick(spec.job_id)
    assert backend.launch_log[0].placement == ("node-0", "node-1")
    backend.report_progress(55)
    r.tick(spec.job_id)

    layout.lose_node("node-1")
    events.push(node_down("node-1"))
    backend.fail_current()
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    relaunch(repo, spec, clock, r, policy)

    request = backend.launch_log[-1]
    assert state(repo, spec) is JobState.VERIFYING
    assert request.resume_step == 50, "the newest checkpoint survived through its peer copy"
    assert "node-1" not in request.placement
    assert request.excluded_nodes == ("node-1",)
    assert '"location": "peer@node-0"' in request.restore_plan
    assert repo.unhealthy_nodes() == {"node-1"}

    incident = repo.list_incidents(spec.job_id)[0]
    plan = incident.plans[0]
    assert plan["chosen"]["step"] == 50 and plan["strategy"] == "cost"


def test_node_events_for_nodes_the_job_is_not_on_are_ignored(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)
    r.tick(spec.job_id)
    backend.report_progress(45)
    r.tick(spec.job_id)

    events.push(node_down("node-3"))
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING
    assert repo.unhealthy_nodes() == {"node-3"}, "but the node is still recorded as bad"


def test_node_health_survives_a_controller_restart(repo, spec, backend, clock, checkpoints, policy):
    """Level-triggered: a node recorded unhealthy keeps the job off it even
    if the controller that saw the event is gone."""
    checkpoints.write(40)
    repo.register_job(spec)
    first = make_reconciler(repo, backend, clock, holder="ctl-a")
    first.tick(spec.job_id)
    backend.report_progress(45)
    first.tick(spec.job_id)

    # The event was consumed and recorded, then the controller died before
    # acting on it.
    repo.set_node_health("node-1", False, "xid 79")

    clock.advance(61)
    second = make_reconciler(repo, backend, clock, holder="ctl-b")
    second.tick(spec.job_id)
    assert state(repo, spec) is JobState.SUSPECT
    incident = repo.list_incidents(spec.job_id)[0]
    assert any(s["kind"] == SignalKind.NODE_UNHEALTHY and s["node"] == "node-1" for s in incident.signals)


def test_a_job_on_a_bad_node_is_moved_even_if_it_keeps_making_progress(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)
    r.tick(spec.job_id)
    backend.report_progress(45)
    r.tick(spec.job_id)

    events.push(node_down("node-0"))
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    backend.report_progress(50)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RECOVERING, "progress on a known-bad node is not self-healing"


def test_healthy_report_returns_a_node_to_the_pool(repo, spec, backend, clock, checkpoints, policy):
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)
    repo.register_job(spec)
    events.push(node_down("node-2"))
    r.tick(spec.job_id)
    assert repo.unhealthy_nodes() == {"node-2"}
    events.push(HealthEvent(SignalKind.NODE_HEALTHY, "operator", datetime.now(UTC) + timedelta(seconds=1), node="node-2"))
    r.tick(spec.job_id)
    assert repo.unhealthy_nodes() == set()


def test_old_events_are_not_reapplied_after_a_restart(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    seen_at = datetime.now(UTC)
    old = HealthEvent(SignalKind.STORAGE_UNAVAILABLE, "probe", seen_at, job_id=spec.job_id)

    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events)
    r.tick(spec.job_id)
    backend.report_progress(45)
    events.push(old)
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    backend.report_progress(60)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING  # transient, abandoned

    # A restarted controller re-reads its event source from the beginning.
    replay = InMemoryEventSource()
    replay.push(old)
    clock.advance(61)
    backend.report_progress(70)
    make_reconciler(repo, backend, clock, events=replay, holder="ctl-b").tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING
    assert len(repo.list_incidents(spec.job_id)) == 1


def test_recovery_waits_for_capacity_without_spending_attempts(repo, spec, backend, clock, checkpoints, layout, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    r = make_reconciler(repo, backend, clock, events=events, pool=("node-0", "node-1"))
    r.tick(spec.job_id)
    backend.report_progress(45)
    r.tick(spec.job_id)

    events.push(node_down("node-1"))
    backend.fail_current()
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    relaunch(repo, spec, clock, r, policy)

    assert state(repo, spec) is JobState.RECOVERING
    assert "insufficient capacity" in repo.get_job(spec.job_id).message
    assert backend.recovery_launches == []
    assert repo.get_job(spec.job_id).attempt == 0

    events.push(HealthEvent(SignalKind.NODE_HEALTHY, "operator", datetime.now(UTC) + timedelta(seconds=1), node="node-1"))
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING
    assert repo.get_job(spec.job_id).attempt == 1


# ------------------------------------------------------ restore verification


def test_restore_that_fails_verification_excludes_the_checkpoint(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(30)
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=45)
    relaunch(repo, spec, clock, r, policy)
    assert backend.launch_log[-1].resume_step == 40

    backend.write_restore_report(outcome="verification_failed", failed_check="rank.data_position")
    backend.fail_current()
    r.tick(spec.job_id)

    row = repo.get_job(spec.job_id)
    assert state(repo, spec) is JobState.SUSPECT
    assert row.bad_steps == [40]
    assert "verification failed" in row.message

    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    relaunch(repo, spec, clock, r, policy, attempt=2)
    assert backend.launch_log[-1].resume_step == 30, "a checkpoint that restored wrong is never used again"

    incident = repo.list_incidents(spec.job_id)[0]
    assert incident.restore_reports[0]["outcome"] == "verification_failed"
    assert any(s["kind"] == SignalKind.CHECKPOINT_INVALID for s in incident.signals)
    assert any(r["step"] == 40 and "restore_verification_failed" in r["reasons"] for r in incident.plans[1]["rejected"])


def test_staging_failure_is_storage_trouble_not_a_bad_checkpoint(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=45)
    relaunch(repo, spec, clock, r, policy)

    backend.write_restore_report(outcome="staging_failed", error="source vanished")
    backend.fail_current()
    r.tick(spec.job_id)

    assert repo.get_job(spec.job_id).bad_steps in (None, [])
    incident = repo.list_incidents(spec.job_id)[0]
    assert any(s["kind"] == SignalKind.STORAGE_UNAVAILABLE for s in incident.signals)


def test_verified_restore_is_recorded_with_its_timings(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=45)
    relaunch(repo, spec, clock, r, policy)

    backend.write_restore_report()
    backend.report_progress(42)
    r.tick(spec.job_id)
    backend.report_progress(46)
    r.tick(spec.job_id)
    checkpoints.write(60)
    backend.report_progress(60)
    r.tick(spec.job_id)

    incident = repo.list_incidents(spec.job_id)[0]
    assert incident.outcome == Outcome.RECOVERED
    phases = [e["phase"] for e in incident.timeline]
    assert phases == [
        "detected", "recovering", "planned", "launched", "restored", "training_resumed", "caught_up", "verified",
    ]
    assert len(incident.restore_reports) == 1


# ----------------------------------------------------------- regressions


def test_progress_restarts_from_the_restore_point_after_relaunch(repo, spec, backend, clock, checkpoints, policy):
    """Replayed steps are progress. Without resetting the observed step,
    a replay longer than the stall timeout looks like a hang."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=90)
    relaunch(repo, spec, clock, r, policy)
    assert repo.get_job(spec.job_id).last_observed_step == 40

    for step in (50, 60, 70, 80):
        clock.advance(policy.stall_timeout_seconds - 5)
        backend.report_progress(step)
        r.tick(spec.job_id)
        assert state(repo, spec) is JobState.VERIFYING, f"replaying step {step} is not a stall"


def test_first_heartbeat_after_a_slow_start_is_not_a_stall(repo, spec, backend, clock, policy):
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    r.tick(spec.job_id)
    clock.advance(policy.stall_timeout_seconds + 10)  # slow start, still inside startup grace
    backend.report_progress(3)
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.RUNNING


def test_recovery_from_the_final_checkpoint_completes(repo, spec, backend, clock, checkpoints, policy):
    """Nothing is left to train after restoring the last step; a clean exit
    has to count as success rather than an unmet progress target."""
    checkpoints.write(500)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy, fail_at_step=500)
    relaunch(repo, spec, clock, r, policy)
    assert backend.launch_log[-1].resume_step == 500

    backend.succeed_current()
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.SUCCEEDED
    assert repo.list_incidents(spec.job_id)[0].outcome == Outcome.RECOVERED


def test_measured_step_time_prices_replay(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    r.tick(spec.job_id)
    backend.report_progress(50, step_seconds=0.5)
    r.tick(spec.job_id)
    assert repo.get_job(spec.job_id).step_seconds == 0.5

    backend.fail_current()
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    relaunch(repo, spec, clock, r, policy)
    cost = repo.list_incidents(spec.job_id)[0].plans[0]["chosen"]["cost"]
    assert cost["replay_steps"] == 10
    assert cost["replay_s"] == 5.0


def test_adopted_orphan_keeps_the_placement_it_was_launched_with(
    repo, spec, backend, clock, checkpoints, layout, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    events = InMemoryEventSource()
    first = make_reconciler(repo, backend, clock, events=events, holder="ctl-a")
    first.tick(spec.job_id)
    backend.report_progress(45)
    first.tick(spec.job_id)
    layout.lose_node("node-1")
    events.push(node_down("node-1"))
    backend.fail_current()
    first.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    first.tick(spec.job_id)

    # The controller dies inside the launch: the run exists, the job row
    # still says RECOVERING and knows nothing about the plan it carried out.
    backend.crash_after_launch = True
    clock.advance(policy.backoff_for(1) + 1)
    with pytest.raises(LaunchFailed):
        first._launch(
            repo.get_job(spec.job_id), spec, policy, repo.acquire_lease(spec.job_id, "ctl-a", 60),
            attempt=1, resume_step=40, placement=["node-0", "node-2"],
        )

    clock.advance(61)
    make_reconciler(repo, backend, clock, holder="ctl-b").tick(spec.job_id)
    row = repo.get_job(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING
    assert len(backend.recovery_launches) == 1, "adopted, not relaunched"
    assert row.placement == ["node-0", "node-2"]
    assert row.resume_step == 40
