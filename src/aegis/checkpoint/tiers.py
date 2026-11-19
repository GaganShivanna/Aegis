"""Storage tier performance and health.

Each tier has a profile (bandwidth, latency) and a live condition (available,
degraded). The planner uses both to estimate how long a recovery path will
take; the simulated tiers use them to actually take that long, so estimates
and measurements can be compared honestly.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from aegis.checkpoint.layout import Layout, Tier
from aegis.checkpoint.store import StorageFault, atomic_write_text

_CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class TierProfile:
    bandwidth_bytes_per_s: float
    latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class StorageProfiles:
    """Performance model for every stage of a recovery.

    `local` is disk-to-disk on one node, `peer` is node-to-node network, and
    `durable` is object storage. `startup_s` is the cost of bringing the
    worker group up; `fresh_node_s` is the extra cost of scheduling onto a
    node the job was not already using.
    """

    name: str
    local: TierProfile
    peer: TierProfile
    durable: TierProfile
    restore_bytes_per_s: float
    restore_fixed_s: float
    startup_s: float
    fresh_node_s: float
    throttle: bool = True

    def for_tier(self, tier: Tier) -> TierProfile:
        return {Tier.LOCAL: self.local, Tier.PEER: self.peer, Tier.DURABLE: self.durable}[tier]

    def rate(self, tier: Tier, condition: TierCondition | None = None) -> float:
        condition = condition or TierCondition()
        if not condition.available:
            return 0.0
        return self.for_tier(tier).bandwidth_bytes_per_s * condition.bandwidth_factor

    def transfer_seconds(
        self, tier: Tier, nbytes: int, condition: TierCondition | None = None
    ) -> float:
        rate = self.rate(tier, condition)
        if rate <= 0:
            return math.inf
        return self.for_tier(tier).latency_s + nbytes / rate

    def restore_seconds(self, nbytes: int) -> float:
        return self.restore_fixed_s + nbytes / self.restore_bytes_per_s


# The toy model's shards are ~320 KB, so real-world bandwidths would make
# every transfer free and every plan trivially "take the newest checkpoint".
# These are scaled so the ratio of transfer time to replay time resembles a
# large model: reading a checkpoint from object storage costs roughly as much
# as replaying a few dozen steps, a peer copy about a tenth of that.
SCALED = StorageProfiles(
    name="scaled",
    local=TierProfile(20e6, 0.0),
    peer=TierProfile(1e6, 0.05),
    durable=TierProfile(120e3, 0.25),
    restore_bytes_per_s=5e6,
    restore_fixed_s=0.2,
    startup_s=4.0,
    fresh_node_s=1.0,
)

UNTHROTTLED = replace(
    SCALED,
    name="unthrottled",
    local=TierProfile(math.inf),
    peer=TierProfile(math.inf),
    durable=TierProfile(math.inf),
    restore_bytes_per_s=math.inf,
    restore_fixed_s=0.0,
    throttle=False,
)

PROFILES = {p.name: p for p in (SCALED, UNTHROTTLED)}


def get_profiles(name: str | None = None) -> StorageProfiles:
    key = name or os.environ.get("AEGIS_STORAGE_PROFILE", "scaled")
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(f"unknown storage profile {key!r}; known: {', '.join(PROFILES)}") from None


# ------------------------------------------------------------------ conditions


@dataclass(frozen=True, slots=True)
class TierCondition:
    available: bool = True
    bandwidth_factor: float = 1.0

    @property
    def degraded(self) -> bool:
        return not self.available or self.bandwidth_factor < 1.0


def read_conditions(layout: Layout) -> dict[Tier, TierCondition]:
    conditions = {tier: TierCondition() for tier in Tier}
    try:
        payload = json.loads(layout.tier_state_file.read_text())
    except (OSError, ValueError):
        return conditions
    for name, raw in payload.items():
        try:
            conditions[Tier(name)] = TierCondition(**raw)
        except (ValueError, TypeError):
            continue
    return conditions


def set_condition(
    layout: Layout,
    tier: Tier,
    *,
    available: bool | None = None,
    bandwidth_factor: float | None = None,
) -> TierCondition:
    conditions = read_conditions(layout)
    current = conditions[tier]
    updated = TierCondition(
        available=current.available if available is None else available,
        bandwidth_factor=current.bandwidth_factor if bandwidth_factor is None else bandwidth_factor,
    )
    conditions[tier] = updated
    layout.root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        layout.tier_state_file,
        json.dumps({str(t): asdict(c) for t, c in conditions.items()}, indent=2),
    )
    return updated


def clear_conditions(layout: Layout) -> None:
    layout.tier_state_file.unlink(missing_ok=True)


# ------------------------------------------------------------------------ copy


def throttled_copy(
    src: Path,
    dst: Path,
    *,
    tier: Tier,
    profiles: StorageProfiles,
    condition: TierCondition | None = None,
) -> float:
    """Copy `src` to `dst` atomically at the speed `tier` currently allows.

    Returns elapsed seconds. Raises StorageFault if the tier is unavailable or
    the source disappears mid-copy (its node died), so callers can treat every
    storage problem the same way.
    """
    condition = condition or TierCondition()
    if not condition.available:
        raise StorageFault(f"{tier} tier unavailable")

    start = time.monotonic()
    rate = profiles.rate(tier, condition)
    latency = profiles.for_tier(tier).latency_s
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")

    try:
        if profiles.throttle and latency > 0:
            time.sleep(latency)
        copied = 0
        t0 = time.monotonic()
        with src.open("rb") as fin, tmp.open("wb") as fout:
            while chunk := fin.read(_CHUNK):
                fout.write(chunk)
                copied += len(chunk)
                if profiles.throttle and math.isfinite(rate) and rate > 0:
                    ahead = copied / rate - (time.monotonic() - t0)
                    if ahead > 0:
                        time.sleep(ahead)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, dst)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        if isinstance(exc, StorageFault):
            raise
        raise StorageFault(f"copy {src} -> {dst} failed: {exc}") from exc
    return time.monotonic() - start
