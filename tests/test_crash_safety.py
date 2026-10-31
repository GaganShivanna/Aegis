"""Controller restart and duplicate prevention.

The acceptance criterion is "zero duplicate active jobs". Each test here kills
the controller at a different point and checks that exactly one run survives.
"""

from __future__ import annotations

import pytest

from aegis.controller.reconciler import LaunchFailed
from aegis.models import ActionKind, ActionState, JobState
from tests.test_reconciler import drive_to_recovery, make_reconciler, state


def test_restart_after_launch_adopts_instead_of_relaunching(
    repo, spec, backend, clock, checkpoints, policy
):
    checkpoints.write(40)
    repo.register_job(spec)
    first = make_reconciler(repo, backend, clock, holder="ctl-a")

    drive_to_recovery(repo, spec, backend, clock, first, policy)
    clock.advance(policy.backoff_for(1) + 1)
    first.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING
    launches_before = len(backend.launch_log)

    # Controller dies; its lease expires; a replacement takes over. The
    # training run is still alive throughout.
    clock.advance(policy.verification_timeout_seconds / 2)
    second = make_reconciler(repo, backend, clock, holder="ctl-b")
    second.tick(spec.job_id)

    assert len(backend.launch_log) == launches_before, "a live run must not be relaunched"
    assert backend.active_count == 1


def test_crash_between_ledger_write_and_launch_still_launches_once(
    repo, spec, backend, clock, checkpoints, policy
):
    """The narrowest window: intent recorded, nothing started."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)

    # Stand in for the crash: write the ledger entry, then stop.
    token = repo.acquire_lease(spec.job_id, "ctl-a", 60)
    action, created = repo.begin_action(spec.job_id, 1, ActionKind.LAUNCH, token)
    assert created and action.state == ActionState.PENDING
    assert backend.recovery_launches == []

    # A fresh controller finds the orphaned intent, confirms nothing is
    # running, and completes it.
    successor = make_reconciler(repo, backend, clock, holder="ctl-b")
    clock.advance(61)
    successor.tick(spec.job_id)

    assert len(backend.recovery_launches) == 1
    assert state(repo, spec) is JobState.VERIFYING
    assert repo.get_action(spec.job_id, 1, ActionKind.LAUNCH).state == ActionState.STARTED


def test_crash_after_launch_but_before_recording_adopts_the_live_run(
    repo, spec, backend, clock, checkpoints, policy
):
    """The dangerous window: the cluster has a run, the ledger does not know."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock, holder="ctl-a")
    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)

    backend.crash_after_launch = True
    with pytest.raises(LaunchFailed):
        r._launch(
            repo.get_job(spec.job_id),
            spec,
            policy,
            repo.acquire_lease(spec.job_id, "ctl-a", 60),
            attempt=1,
            resume_step=40,
        )
    assert backend.active_count == 1, "the run really did start"
    assert repo.get_action(spec.job_id, 1, ActionKind.LAUNCH).state == ActionState.FAILED

    clock.advance(61)
    successor = make_reconciler(repo, backend, clock, holder="ctl-b")
    successor.tick(spec.job_id)

    assert backend.active_count == 1, "the orphan must be adopted, not duplicated"
    assert len(backend.recovery_launches) == 1
    assert repo.get_action(spec.job_id, 1, ActionKind.LAUNCH).state == ActionState.STARTED


def test_a_hung_run_is_terminated_before_a_new_one_starts(
    repo, spec, backend, clock, checkpoints, policy
):
    """A stalled job is still "running"; recovery must stop it explicitly."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)
    hung = backend.current

    clock.advance(policy.stall_timeout_seconds + 1)
    r.tick(spec.job_id)  # RUNNING -> SUSPECT via stall detection
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)  # -> RECOVERING
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)  # terminate the hung run, then relaunch

    assert hung.action_id in backend.terminated
    assert backend.active_count == 1
    assert len(backend.recovery_launches) == 1


def test_relaunch_waits_when_the_old_run_cannot_be_killed(
    repo, spec, backend, clock, checkpoints, policy
):
    """Better to stall a recovery than to run two copies of the same job."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)

    r.tick(spec.job_id)
    backend.report_progress(50)
    r.tick(spec.job_id)

    backend.unkillable = True
    clock.advance(policy.stall_timeout_seconds + 1)
    r.tick(spec.job_id)
    clock.advance(policy.correlation_window_seconds + 1)
    r.tick(spec.job_id)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    assert state(repo, spec) is JobState.RECOVERING
    assert backend.recovery_launches == [], "must not add a second run"
    assert backend.active_count == 1
    assert repo.get_job(spec.job_id).attempt == 0, "waiting is not a spent attempt"

    # Once the stray finally dies, recovery proceeds normally.
    backend.unkillable = False
    r.tick(spec.job_id)
    assert state(repo, spec) is JobState.VERIFYING
    assert backend.active_count == 1


def test_two_controllers_produce_one_recovery(repo, spec, backend, clock, checkpoints, policy):
    checkpoints.write(40)
    repo.register_job(spec)
    a = make_reconciler(repo, backend, clock, holder="ctl-a")
    b = make_reconciler(repo, backend, clock, holder="ctl-b")

    a.tick(spec.job_id)
    backend.report_progress(50)
    a.tick(spec.job_id)
    backend.fail_current()

    # Both controllers see the failure and race through the whole sequence.
    for _ in range(3):
        a.tick(spec.job_id)
        b.tick(spec.job_id)
        clock.advance(policy.correlation_window_seconds + 1)
    clock.advance(policy.backoff_for(1) + 1)
    a.tick(spec.job_id)
    b.tick(spec.job_id)

    assert len(backend.recovery_launches) == 1
    assert backend.active_count == 1
    assert len(repo.list_incidents(spec.job_id)) == 1


def test_state_survives_a_full_repository_reopen(repo, spec, backend, clock, checkpoints, policy, tmp_path):
    """Everything needed to resume is on disk, not in controller memory."""
    checkpoints.write(40)
    repo.register_job(spec)
    r = make_reconciler(repo, backend, clock)
    drive_to_recovery(repo, spec, backend, clock, r, policy)
    clock.advance(policy.backoff_for(1) + 1)
    r.tick(spec.job_id)

    from aegis.store.repository import Repository

    reopened = Repository.from_url(f"sqlite:///{tmp_path / 'aegis.sqlite'}", clock=clock)
    row = reopened.get_job(spec.job_id)

    assert JobState(row.state) is JobState.VERIFYING
    assert row.attempt == 1
    assert row.verify_baseline_step == 40
    assert len(reopened.live_launches(spec.job_id)) == 1
