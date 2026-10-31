"""Persistence operations for the controller.

Two invariants are enforced here rather than in the reconciler, because the
reconciler can die at any line and the database cannot:

  * at most one controller holds a job's recovery lease at a time, and every
    action is stamped with the fence token it was planned under;
  * an action for a given (job, attempt, kind) can only be created once, so a
    relaunch that already happened is never repeated after a restart.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any, Iterator

from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from aegis.clock import Clock, SystemClock
from aegis.models import (
    ActionKind,
    ActionState,
    HealthEvent,
    JobSpec,
    JobState,
    Outcome,
    RecoveryPolicy,
    RunHandle,
    deterministic_action_id,
)
from aegis.store.db import (
    ActionRow,
    IncidentRow,
    JobRow,
    LeaseRow,
    SignalRow,
    create_engine_for,
    create_session_factory,
)


class LeaseLost(RuntimeError):
    """Raised when an action is attempted without a currently valid lease."""


class Repository:
    def __init__(
        self,
        engine: Engine,
        *,
        clock: Clock | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = engine
        self.clock = clock or SystemClock()
        self._sessions = session_factory or create_session_factory(engine)

    @classmethod
    def from_url(cls, url: str, *, clock: Clock | None = None) -> Repository:
        return cls(create_engine_for(url), clock=clock)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------ jobs

    def register_job(self, spec: JobSpec) -> JobRow:
        now = self.clock.now()
        with self.session() as s:
            existing = s.get(JobRow, spec.job_id)
            if existing is not None:
                return existing
            row = JobRow(
                job_id=spec.job_id,
                name=spec.name,
                state=JobState.PENDING,
                spec=_spec_to_json(spec),
                policy=asdict(spec.policy),
                policy_id=spec.policy.policy_id,
                attempt=0,
                created_at=now,
                updated_at=now,
            )
            s.add(row)
            return row

    def get_job(self, job_id: str) -> JobRow | None:
        with self.session() as s:
            return s.get(JobRow, job_id)

    def list_jobs(self, *, active_only: bool = False) -> list[JobRow]:
        stmt = select(JobRow)
        if active_only:
            stmt = stmt.where(
                JobRow.state.notin_(
                    [JobState.SUCCEEDED, JobState.EXHAUSTED, JobState.STOPPED]
                )
            )
        with self.session() as s:
            return list(s.scalars(stmt.order_by(JobRow.created_at)))

    def update_job(self, job_id: str, **fields: Any) -> JobRow:
        fields["updated_at"] = self.clock.now()
        with self.session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                raise KeyError(job_id)
            for key, value in fields.items():
                setattr(row, key, value)
            s.add(row)
            return row

    @staticmethod
    def spec_of(row: JobRow) -> JobSpec:
        return _spec_from_json({**row.spec, "policy": row.policy})

    @staticmethod
    def policy_of(row: JobRow) -> RecoveryPolicy:
        return RecoveryPolicy(**row.policy)

    # ---------------------------------------------------------------- leases

    def acquire_lease(self, job_id: str, holder: str, ttl_seconds: float) -> int:
        """Take or renew the recovery lease. Returns the fence token.

        Renewing an already-held lease keeps the token; taking over an expired
        one bumps it, which invalidates any action the previous holder planned
        but never completed.
        """
        now = self.clock.now()
        expires = now + timedelta(seconds=ttl_seconds)

        with self.session() as s:
            current = s.get(LeaseRow, job_id)

            if current is None:
                try:
                    with s.begin_nested():
                        s.add(
                            LeaseRow(
                                job_id=job_id,
                                holder=holder,
                                fence_token=1,
                                acquired_at=now,
                                expires_at=expires,
                            )
                        )
                    return 1
                except IntegrityError:
                    current = s.get(LeaseRow, job_id)
                    if current is None:
                        raise LeaseLost(f"lease race on job {job_id}") from None

            held_by_us = current.holder == holder
            expired = current.expires_at <= now
            if not (held_by_us or expired):
                raise LeaseLost(
                    f"job {job_id} lease held by {current.holder} until {current.expires_at.isoformat()}"
                )

            token = current.fence_token if held_by_us else current.fence_token + 1
            result = s.execute(
                update(LeaseRow)
                .where(
                    LeaseRow.job_id == job_id,
                    LeaseRow.fence_token == current.fence_token,
                )
                .values(
                    holder=holder,
                    fence_token=token,
                    acquired_at=now if not held_by_us else current.acquired_at,
                    expires_at=expires,
                )
            )
            if result.rowcount != 1:
                raise LeaseLost(f"lost lease race on job {job_id}")
            return token

    def lease_token(self, job_id: str, holder: str) -> int | None:
        with self.session() as s:
            row = s.get(LeaseRow, job_id)
            if row is None or row.holder != holder or row.expires_at <= self.clock.now():
                return None
            return row.fence_token

    def release_lease(self, job_id: str, holder: str) -> None:
        with self.session() as s:
            row = s.get(LeaseRow, job_id)
            if row is not None and row.holder == holder:
                row.expires_at = self.clock.now()
                s.add(row)

    def assert_lease(self, job_id: str, holder: str, fence_token: int) -> None:
        token = self.lease_token(job_id, holder)
        if token is None:
            raise LeaseLost(f"no valid lease on job {job_id} for {holder}")
        if token != fence_token:
            raise LeaseLost(
                f"stale fence token for job {job_id}: have {fence_token}, current {token}"
            )

    # --------------------------------------------------------------- actions

    def begin_action(
        self,
        job_id: str,
        attempt: int,
        kind: ActionKind,
        fence_token: int,
        *,
        holder: str | None = None,
    ) -> tuple[ActionRow, bool]:
        """Record intent to act. Returns (row, created).

        `created=False` means this exact action was already recorded, by this
        controller before it crashed or by another replica. The caller must
        reconcile against the backend instead of launching again.
        """
        if holder is not None:
            self.assert_lease(job_id, holder, fence_token)

        action_id = deterministic_action_id(job_id, attempt, kind)
        now = self.clock.now()
        with self.session() as s:
            existing = s.get(ActionRow, action_id)
            if existing is not None:
                return existing, False
            row = ActionRow(
                action_id=action_id,
                job_id=job_id,
                attempt=attempt,
                kind=kind,
                state=ActionState.PENDING,
                fence_token=fence_token,
                created_at=now,
                updated_at=now,
            )
            try:
                with s.begin_nested():
                    s.add(row)
            except IntegrityError:
                existing = s.get(ActionRow, action_id)
                if existing is None:
                    raise
                return existing, False
            return row, True

    def mark_action(
        self,
        action_id: str,
        state: ActionState,
        *,
        handle: RunHandle | None = None,
        detail: str = "",
    ) -> ActionRow:
        with self.session() as s:
            row = s.get(ActionRow, action_id)
            if row is None:
                raise KeyError(action_id)
            row.state = state
            row.updated_at = self.clock.now()
            if handle is not None:
                row.handle = _handle_to_json(handle)
            if detail:
                row.detail = detail
            s.add(row)
            return row

    def get_action(self, job_id: str, attempt: int, kind: ActionKind) -> ActionRow | None:
        with self.session() as s:
            return s.get(ActionRow, deterministic_action_id(job_id, attempt, kind))

    def actions_for(self, job_id: str, states: list[ActionState] | None = None) -> list[ActionRow]:
        stmt = select(ActionRow).where(ActionRow.job_id == job_id)
        if states:
            stmt = stmt.where(ActionRow.state.in_(states))
        with self.session() as s:
            return list(s.scalars(stmt.order_by(ActionRow.attempt, ActionRow.created_at)))

    def live_launches(self, job_id: str) -> list[ActionRow]:
        """Launch actions the ledger believes are still outstanding."""
        return [
            a
            for a in self.actions_for(job_id, [ActionState.PENDING, ActionState.STARTED])
            if a.kind == ActionKind.LAUNCH
        ]

    # ------------------------------------------------------------- incidents

    def open_incident(
        self,
        job_id: str,
        *,
        job_name: str,
        attempt: int,
        policy_id: str,
        workload: dict[str, Any],
        step_at_failure: int | None,
    ) -> int:
        with self.session() as s:
            row = IncidentRow(
                job_id=job_id,
                job_name=job_name,
                attempt=attempt,
                opened_at=self.clock.now(),
                policy_id=policy_id,
                workload=workload,
                signals=[],
                actions=[],
                rejected_checkpoints=[],
                step_at_failure=step_at_failure,
                outcome=Outcome.IN_PROGRESS,
            )
            s.add(row)
            s.flush()
            return row.id

    def open_incident_id(self, job_id: str) -> int | None:
        stmt = (
            select(IncidentRow.id)
            .where(IncidentRow.job_id == job_id, IncidentRow.outcome == Outcome.IN_PROGRESS)
            .order_by(IncidentRow.opened_at.desc())
            .limit(1)
        )
        with self.session() as s:
            return s.scalar(stmt)

    def get_incident(self, incident_id: int) -> IncidentRow | None:
        with self.session() as s:
            return s.get(IncidentRow, incident_id)

    def update_incident(self, incident_id: int, **fields: Any) -> IncidentRow:
        with self.session() as s:
            row = s.get(IncidentRow, incident_id)
            if row is None:
                raise KeyError(incident_id)
            for key, value in fields.items():
                setattr(row, key, value)
            s.add(row)
            return row

    def append_incident_action(self, incident_id: int, entry: dict[str, Any]) -> None:
        with self.session() as s:
            row = s.get(IncidentRow, incident_id)
            if row is None:
                return
            # Reassign: JSON columns are not change-tracked in place.
            row.actions = [*(row.actions or []), {**entry, "at": self.clock.now().isoformat()}]
            s.add(row)

    def close_incident(
        self,
        incident_id: int,
        *,
        outcome: Outcome,
        recovery_seconds: float | None = None,
        lost_steps: int | None = None,
        detail: str = "",
    ) -> None:
        with self.session() as s:
            row = s.get(IncidentRow, incident_id)
            if row is None:
                return
            row.outcome = outcome
            row.closed_at = self.clock.now()
            row.recovery_seconds = recovery_seconds
            row.lost_steps = lost_steps
            if detail:
                row.detail = detail
            s.add(row)

    def list_incidents(self, job_id: str | None = None, limit: int = 50) -> list[IncidentRow]:
        stmt = select(IncidentRow).order_by(IncidentRow.opened_at.desc()).limit(limit)
        if job_id:
            stmt = stmt.where(IncidentRow.job_id == job_id)
        with self.session() as s:
            return list(s.scalars(stmt))

    # ----------------------------------------------------------------- signals

    def record_signal(self, incident_id: int | None, event: HealthEvent) -> bool:
        """Persist a health event. Returns False if it was a duplicate.

        Deduplication is scoped to the incident: the same node reported bad
        twice inside one incident is one signal, but the same node going bad
        again after recovery is genuinely new.
        """
        with self.session() as s:
            row = SignalRow(
                incident_id=incident_id,
                job_id=event.job_id or "",
                dedup_key=event.dedup_key,
                kind=event.kind,
                source=event.source,
                node=event.node,
                observed_at=event.observed_at,
                detail=event.detail,
            )
            try:
                with s.begin_nested():
                    s.add(row)
            except IntegrityError:
                return False

            if incident_id is not None:
                incident = s.get(IncidentRow, incident_id)
                if incident is not None:
                    incident.signals = [
                        *(incident.signals or []),
                        {
                            "kind": str(event.kind),
                            "source": event.source,
                            "node": event.node,
                            "observed_at": event.observed_at.isoformat(),
                            "detail": event.detail,
                        },
                    ]
                    s.add(incident)
            return True

    def signals_for_incident(self, incident_id: int) -> list[SignalRow]:
        stmt = select(SignalRow).where(SignalRow.incident_id == incident_id)
        with self.session() as s:
            return list(s.scalars(stmt.order_by(SignalRow.observed_at)))


# --------------------------------------------------------------- serialisation


def _spec_to_json(spec: JobSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload.pop("policy", None)
    return payload


def _spec_from_json(payload: dict[str, Any]) -> JobSpec:
    data = dict(payload)
    policy = data.pop("policy", None)
    return JobSpec(**data, policy=RecoveryPolicy(**policy) if policy else RecoveryPolicy())


def _handle_to_json(handle: RunHandle) -> dict[str, Any]:
    payload = asdict(handle)
    payload["started_at"] = handle.started_at.isoformat()
    return payload


def handle_from_json(payload: dict[str, Any]) -> RunHandle:
    data = dict(payload)
    started = data.get("started_at")
    data["started_at"] = (
        datetime.fromisoformat(started) if isinstance(started, str) else started or datetime.now(UTC)
    )
    return RunHandle(**data)


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)
