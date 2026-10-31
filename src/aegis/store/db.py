"""Durable state.

SQLite by default so the project runs with no setup; PostgreSQL via the same
schema for anything real. Everything the controller needs to resume after a
crash lives here: job state, the recovery lease, and the action ledger.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class UtcDateTime(TypeDecorator):
    """Stores timezone-aware UTC datetimes and reads them back that way.

    SQLite silently drops tzinfo, which turns every later comparison into a
    naive/aware TypeError at the worst possible moment.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    spec: Mapped[dict] = mapped_column(JSON)
    policy: Mapped[dict] = mapped_column(JSON)
    policy_id: Mapped[str] = mapped_column(String(64))

    attempt: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Progress bookkeeping
    last_observed_step: Mapped[int] = mapped_column(Integer, default=0)
    last_progress_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    resume_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verify_baseline_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verify_started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

    # Recovery bookkeeping
    recovery_started_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    suspect_since: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")


class LeaseRow(Base):
    """One active recovery lease per job.

    `fence_token` increments on every fresh acquisition. Actions carry the
    token they were planned under, so a controller that pauses, loses its
    lease, and wakes up cannot complete a stale action.
    """

    __tablename__ = "leases"

    job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    holder: Mapped[str] = mapped_column(String(128))
    fence_token: Mapped[int] = mapped_column(Integer, default=1)
    acquired_at: Mapped[datetime] = mapped_column(UtcDateTime)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime, index=True)


class ActionRow(Base):
    """Ledger of every side effect, written before the side effect happens."""

    __tablename__ = "actions"
    __table_args__ = (
        UniqueConstraint("job_id", "attempt", "kind", name="uq_action_job_attempt_kind"),
        Index("ix_action_job_state", "job_id", "state"),
    )

    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(128), ForeignKey("jobs.job_id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(32), index=True)
    fence_token: Mapped[int] = mapped_column(Integer, default=0)
    handle: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime, default=lambda: datetime.now(UTC))


class IncidentRow(Base):
    """The learning layer's training data: what broke, what we did, what happened."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(128), index=True)
    job_name: Mapped[str] = mapped_column(String(128), index=True, default="")
    attempt: Mapped[int] = mapped_column(Integer, default=0)

    opened_at: Mapped[datetime] = mapped_column(UtcDateTime)
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)

    workload: Mapped[dict] = mapped_column(JSON, default=dict)
    signals: Mapped[list] = mapped_column(JSON, default=list)
    actions: Mapped[list] = mapped_column(JSON, default=list)
    rejected_checkpoints: Mapped[list] = mapped_column(JSON, default=list)

    chosen_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    step_at_failure: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lost_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recovery_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    policy_id: Mapped[str] = mapped_column(String(64), default="")
    outcome: Mapped[str] = mapped_column(String(32), index=True, default="in_progress")
    detail: Mapped[str] = mapped_column(Text, default="")


class SignalRow(Base):
    """Raw health events, deduplicated per incident."""

    __tablename__ = "signals"
    __table_args__ = (
        UniqueConstraint("incident_id", "dedup_key", name="uq_signal_incident_dedup"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("incidents.id"), nullable=True)
    job_id: Mapped[str] = mapped_column(String(128), index=True)
    dedup_key: Mapped[str] = mapped_column(String(256), index=True)
    kind: Mapped[str] = mapped_column(String(48))
    source: Mapped[str] = mapped_column(String(64))
    node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    observed_at: Mapped[datetime] = mapped_column(UtcDateTime)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


def create_engine_for(url: str, *, echo: bool = False) -> Engine:
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["timeout"] = 30
    engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        _enable_sqlite_wal(engine)
    Base.metadata.create_all(engine)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _enable_sqlite_wal(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        # WAL + FULL sync: the action ledger is only useful if it survives the
        # crash that it exists to protect against.
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
