"""Recovery policies.

Milestone 1 ships a small set of hand-written, versioned policies and always
uses the baseline. Milestone 3 ranks them from incident history; the registry
and version stamping exist now so incidents recorded today remain comparable
once ranking arrives.
"""

from __future__ import annotations

import random

from aegis.models import RecoveryPolicy

BASELINE = RecoveryPolicy(
    name="baseline",
    version=1,
    max_attempts=3,
    backoff_base_seconds=10.0,
    backoff_multiplier=2.0,
    backoff_max_seconds=300.0,
    jitter_ratio=0.2,
    required_progress_steps=20,
)

# Shorter waits; appropriate where failures are usually transient and the
# cluster has spare capacity to absorb a fast retry.
FAST_RETRY = RecoveryPolicy(
    name="fast-retry",
    version=1,
    max_attempts=4,
    backoff_base_seconds=2.0,
    backoff_multiplier=1.5,
    backoff_max_seconds=60.0,
    jitter_ratio=0.3,
    required_progress_steps=20,
)

# Longer waits and a deeper checkpoint search, for jobs on flaky storage where
# retrying immediately tends to hit the same bad state.
PATIENT = RecoveryPolicy(
    name="patient",
    version=1,
    max_attempts=3,
    backoff_base_seconds=60.0,
    backoff_multiplier=2.0,
    backoff_max_seconds=900.0,
    jitter_ratio=0.1,
    required_progress_steps=20,
    max_checkpoint_candidates=20,
)

# Used by tests and the demo, where waiting out a real backoff is pointless.
DEMO = RecoveryPolicy(
    name="demo",
    version=1,
    max_attempts=3,
    backoff_base_seconds=1.0,
    backoff_multiplier=1.5,
    backoff_max_seconds=5.0,
    jitter_ratio=0.1,
    correlation_window_seconds=1.0,
    stall_timeout_seconds=25.0,
    required_progress_steps=20,
    verification_timeout_seconds=180.0,
)

POLICY_REGISTRY: dict[str, RecoveryPolicy] = {
    p.name: p for p in (BASELINE, FAST_RETRY, PATIENT, DEMO)
}


def get_policy(name: str) -> RecoveryPolicy:
    try:
        return POLICY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown policy {name!r}; known: {', '.join(sorted(POLICY_REGISTRY))}"
        ) from None


def backoff_with_jitter(
    policy: RecoveryPolicy, attempt: int, rng: random.Random | None = None
) -> float:
    """Exponential backoff with full-width symmetric jitter.

    Jitter matters when a node failure takes out several jobs at once: without
    it they all retry in lockstep and collide on the same replacement capacity.
    """
    base = policy.backoff_for(attempt)
    if policy.jitter_ratio <= 0:
        return base
    r = rng or random
    spread = base * policy.jitter_ratio
    return max(0.0, base + r.uniform(-spread, spread))
