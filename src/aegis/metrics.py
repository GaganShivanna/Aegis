"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

RECOVERY_ATTEMPTS = Counter(
    "aegis_recovery_attempts_total",
    "Recovery attempts started",
    ["job_name", "policy"],
)
RECOVERY_OUTCOMES = Counter(
    "aegis_recovery_outcomes_total",
    "Recovery outcomes by result",
    ["job_name", "policy", "outcome"],
)
RECOVERY_SECONDS = Histogram(
    "aegis_recovery_duration_seconds",
    "Time from incident open to verified progress",
    ["job_name", "policy"],
    buckets=(5, 15, 30, 60, 120, 300, 600, 1800, 3600),
)
LOST_STEPS = Histogram(
    "aegis_lost_steps",
    "Steps discarded by rolling back to the chosen checkpoint",
    ["job_name"],
    buckets=(0, 10, 50, 100, 500, 1000, 5000),
)
CHECKPOINTS_REJECTED = Counter(
    "aegis_checkpoints_rejected_total",
    "Checkpoints rejected during selection",
    ["job_name", "reason"],
)
SIGNALS = Counter(
    "aegis_health_signals_total",
    "Health signals accepted after deduplication",
    ["job_name", "kind"],
)
DUPLICATE_LAUNCHES_PREVENTED = Counter(
    "aegis_duplicate_launches_prevented_total",
    "Launches suppressed because the action ledger already recorded them",
    ["job_name"],
)
JOB_STATE = Gauge(
    "aegis_job_state",
    "1 for the job's current state, 0 otherwise",
    ["job_name", "job_id", "state"],
)
CURRENT_STEP = Gauge(
    "aegis_job_step",
    "Latest observed training step",
    ["job_name", "job_id"],
)


def serve(port: int) -> None:
    start_http_server(port)
