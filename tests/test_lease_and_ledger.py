"""Lease and action ledger: the machinery behind "zero duplicate active jobs"."""

from __future__ import annotations

import pytest

from aegis.models import ActionKind, ActionState, deterministic_action_id
from aegis.store.repository import LeaseLost


def test_lease_is_exclusive(repo, spec):
    repo.register_job(spec)
    token = repo.acquire_lease(spec.job_id, "controller-a", 60)
    assert token == 1

    with pytest.raises(LeaseLost):
        repo.acquire_lease(spec.job_id, "controller-b", 60)


def test_renewing_keeps_the_same_fence_token(repo, spec, clock):
    repo.register_job(spec)
    first = repo.acquire_lease(spec.job_id, "controller-a", 60)
    clock.advance(10)
    second = repo.acquire_lease(spec.job_id, "controller-a", 60)
    assert first == second == 1


def test_expired_lease_is_taken_over_with_a_higher_token(repo, spec, clock):
    repo.register_job(spec)
    repo.acquire_lease(spec.job_id, "controller-a", 30)
    clock.advance(31)

    token = repo.acquire_lease(spec.job_id, "controller-b", 30)
    assert token == 2
    assert repo.lease_token(spec.job_id, "controller-a") is None


def test_stale_fence_token_is_refused(repo, spec, clock):
    """A controller that stalls past its lease cannot act on an old plan."""
    repo.register_job(spec)
    stale = repo.acquire_lease(spec.job_id, "controller-a", 30)
    clock.advance(31)
    repo.acquire_lease(spec.job_id, "controller-b", 30)

    with pytest.raises(LeaseLost):
        repo.assert_lease(spec.job_id, "controller-a", stale)


def test_action_id_is_stable_across_processes(spec):
    a = deterministic_action_id(spec.job_id, 2, ActionKind.LAUNCH)
    b = deterministic_action_id(spec.job_id, 2, ActionKind.LAUNCH)
    assert a == b
    assert a != deterministic_action_id(spec.job_id, 3, ActionKind.LAUNCH)
    assert a != deterministic_action_id(spec.job_id, 2, ActionKind.TERMINATE)


def test_second_begin_action_does_not_create(repo, spec):
    repo.register_job(spec)
    token = repo.acquire_lease(spec.job_id, "controller-a", 60)

    first, created_first = repo.begin_action(spec.job_id, 1, ActionKind.LAUNCH, token)
    second, created_second = repo.begin_action(spec.job_id, 1, ActionKind.LAUNCH, token)

    assert created_first is True
    assert created_second is False
    assert first.action_id == second.action_id


def test_two_controllers_cannot_both_record_a_launch(repo, spec, clock):
    """The exact race the lease exists to stop."""
    repo.register_job(spec)
    token_a = repo.acquire_lease(spec.job_id, "controller-a", 30)
    clock.advance(31)
    token_b = repo.acquire_lease(spec.job_id, "controller-b", 30)

    _, created_b = repo.begin_action(
        spec.job_id, 1, ActionKind.LAUNCH, token_b, holder="controller-b"
    )
    assert created_b is True

    with pytest.raises(LeaseLost):
        repo.begin_action(spec.job_id, 1, ActionKind.LAUNCH, token_a, holder="controller-a")


def test_live_launches_reports_outstanding_work(repo, spec):
    repo.register_job(spec)
    token = repo.acquire_lease(spec.job_id, "controller-a", 60)

    action, _ = repo.begin_action(spec.job_id, 1, ActionKind.LAUNCH, token)
    assert len(repo.live_launches(spec.job_id)) == 1

    repo.mark_action(action.action_id, ActionState.COMPLETED)
    assert repo.live_launches(spec.job_id) == []


def test_signals_are_deduplicated_within_an_incident(repo, spec):
    from datetime import UTC, datetime

    from aegis.models import HealthEvent, SignalKind

    repo.register_job(spec)
    incident = repo.open_incident(
        spec.job_id,
        job_name=spec.name,
        attempt=0,
        policy_id="test@v1",
        workload={},
        step_at_failure=100,
    )
    event = HealthEvent(
        kind=SignalKind.NODE_UNHEALTHY,
        source="nvsentinel",
        observed_at=datetime.now(UTC),
        node="gpu-01",
        job_id=spec.job_id,
    )

    assert repo.record_signal(incident, event) is True
    assert repo.record_signal(incident, event) is False
    assert len(repo.signals_for_incident(incident)) == 1
