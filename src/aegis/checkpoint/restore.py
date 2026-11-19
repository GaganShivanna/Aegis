"""Restore plans, staging, and the restore report.

The controller decides *what* to restore and *where each shard comes from*;
it writes that down as a RestorePlan. Each rank then stages its own shard
onto its node's local disk in parallel -- the way an init container or a
per-node fetcher would -- verifies it against the manifest, and loads it.

Every rank reports what it did, and the training job writes one restore
report per attempt. The controller reads it to tell a storage failure during
staging apart from state that loaded but did not match what was saved.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aegis.checkpoint.layout import Layout, Location, Tier
from aegis.checkpoint.manifest import MANIFEST_NAME, CheckpointManifest, sha256_file
from aegis.checkpoint.store import StorageFault, atomic_write_text
from aegis.checkpoint.tiers import StorageProfiles, read_conditions, throttled_copy

PLAN_NAME = "restore_plan.json"
REPORT_NAME = "restore_report.json"


class StagingError(StorageFault):
    """A shard could not be brought to its rank's node intact."""


@dataclass(frozen=True, slots=True)
class ShardSource:
    rank: int
    location: str
    path: str
    size_bytes: int


@dataclass(slots=True)
class RestorePlan:
    step: int | None
    placement: list[str]
    sources: list[ShardSource] = field(default_factory=list)
    manifest_json: str | None = None

    def source_for(self, rank: int) -> ShardSource | None:
        return next((s for s in self.sources if s.rank == rank), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> RestorePlan:
        payload = json.loads(raw)
        sources = [ShardSource(**s) for s in payload.pop("sources", [])]
        return cls(sources=sources, **payload)

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: Path) -> RestorePlan:
        return cls.from_json(path.read_text())


def transfer_tier(source: Location, target_node: str) -> Tier:
    """Which link a copy from `source` to `target_node` actually crosses."""
    if source.tier is Tier.DURABLE:
        return Tier.DURABLE
    return Tier.LOCAL if source.node == target_node else Tier.PEER


@dataclass(frozen=True, slots=True)
class StageResult:
    rank: int
    source: str
    tier: str
    bytes: int
    seconds: float
    copied: bool


def stage_rank(
    layout: Layout, plan: RestorePlan, rank: int, profiles: StorageProfiles
) -> tuple[Path, StageResult]:
    """Bring `rank`'s shard to its node's local disk and verify it.

    Returns the local checkpoint directory. Raises StagingError if the source
    is gone or the copy does not match the manifest.
    """
    if plan.step is None or plan.manifest_json is None:
        raise ValueError("plan has nothing to stage")
    manifest = CheckpointManifest.from_json(plan.manifest_json)
    entry = manifest.shard_for(rank)
    source = plan.source_for(rank)
    if entry is None or source is None:
        raise StagingError(f"plan has no source for rank {rank}")

    node = plan.placement[rank]
    target_dir = layout.dir_for(Location(Tier.LOCAL, node), plan.step)
    target = target_dir / entry.name
    origin = Location.parse(source.location)
    started = time.monotonic()

    copied = False
    if Path(source.path).resolve() != target.resolve():
        tier = transfer_tier(origin, node)
        try:
            throttled_copy(
                Path(source.path),
                target,
                tier=tier,
                profiles=profiles,
                condition=read_conditions(layout)[tier],
            )
        except StorageFault as exc:
            raise StagingError(f"rank {rank}: staging from {origin} failed: {exc}") from exc
        copied = True
    else:
        tier = Tier.LOCAL

    try:
        actual = sha256_file(target)
    except OSError as exc:
        raise StagingError(f"rank {rank}: staged shard unreadable: {exc}") from exc
    if actual != entry.sha256:
        raise StagingError(f"rank {rank}: shard from {origin} does not match the manifest")
    if not (target_dir / MANIFEST_NAME).is_file():
        atomic_write_text(target_dir / MANIFEST_NAME, plan.manifest_json)

    return target_dir, StageResult(
        rank=rank,
        source=str(origin),
        tier=str(tier),
        bytes=entry.size_bytes,
        seconds=time.monotonic() - started,
        copied=copied,
    )


# --------------------------------------------------------------------- report


@dataclass(slots=True)
class RankRestoreReport:
    rank: int
    node: str
    staged: dict[str, Any] | None = None
    load_seconds: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and all(self.checks.values())


@dataclass(slots=True)
class RestoreReport:
    run_id: str
    attempt: int
    step: int | None
    ranks: list[RankRestoreReport] = field(default_factory=list)
    # Do all ranks hold identical replicated state after loading?
    consistent_across_ranks: bool = True
    outcome: str = "restored"  # restored | staging_failed | verification_failed | fresh
    stage_seconds: float = 0.0
    restore_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.outcome in ("restored", "fresh")

    def failed_checks(self) -> list[str]:
        failed = []
        for rank in self.ranks:
            failed.extend(f"rank{rank.rank}.{name}" for name, good in rank.checks.items() if not good)
            if rank.error:
                failed.append(f"rank{rank.rank}: {rank.error}")
        if not self.consistent_across_ranks:
            failed.append("replicated state differs across ranks")
        return failed

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> RestoreReport:
        payload = json.loads(raw)
        ranks = [RankRestoreReport(**r) for r in payload.pop("ranks", [])]
        return cls(ranks=ranks, **payload)

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.to_json())

    @classmethod
    def load(cls, path: Path) -> RestoreReport | None:
        try:
            return cls.from_json(path.read_text())
        except (OSError, ValueError, TypeError):
            return None
