"""Which checkpoints can still be restored, and from where.

A checkpoint is *recoverable* when, using only storage that is still
reachable:

  * at least one copy of its manifest exists, and every reachable copy agrees
    on the committed content (`commit_key`);
  * the manifest is a supported format and matches the job being launched;
  * every rank has at least one replica whose size and SHA-256 match the
    manifest.

Replicas are pooled across tiers: rank 0 may come from its own node's disk
while rank 1 comes from a peer and rank 2 from object storage. The manifest
is the integrity reference for all of them, so a shard is trusted no matter
where it was found, as long as it hashes correctly.

Silent numerical corruption inside a well-formed tensor still passes all of
this; restore-time digest verification in the training job is what catches
state that was saved wrong.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from aegis.checkpoint.layout import Layout, Location, Tier
from aegis.checkpoint.manifest import (
    MANIFEST_NAME,
    SUPPORTED_FORMAT_VERSIONS,
    CheckpointManifest,
    ShardEntry,
    sha256_file,
)
from aegis.checkpoint.tiers import TierCondition, read_conditions


class RejectionReason(StrEnum):
    NO_MANIFEST = "no_manifest"
    MANIFEST_UNPARSEABLE = "manifest_unparseable"
    MANIFEST_CONFLICT = "manifest_conflict"
    UNSUPPORTED_FORMAT = "unsupported_format_version"
    INCOMPATIBLE = "metadata_incompatible"
    UNEXPECTED_SHARD_COUNT = "unexpected_shard_count"
    MISSING_SHARD = "missing_shard"
    SIZE_MISMATCH = "size_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    RESTORE_VERIFICATION_FAILED = "restore_verification_failed"


class ReplicaState(StrEnum):
    OK = "ok"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass(frozen=True, slots=True)
class Replica:
    rank: int
    location: Location
    path: Path
    size_bytes: int
    state: ReplicaState

    @property
    def usable(self) -> bool:
        return self.state is ReplicaState.OK


@dataclass(frozen=True, slots=True)
class Compatibility:
    job_name: str | None = None
    world_size: int | None = None
    model_signature: str | None = None
    max_step: int | None = None


@dataclass(slots=True)
class CheckpointView:
    step: int
    manifest: CheckpointManifest | None = None
    manifest_locations: list[Location] = field(default_factory=list)
    replicas: dict[int, list[Replica]] = field(default_factory=dict)
    bad_replicas: list[Replica] = field(default_factory=list)
    reasons: list[RejectionReason] = field(default_factory=list)
    detail: str = ""

    @property
    def missing_ranks(self) -> list[int]:
        if self.manifest is None:
            return []
        return sorted(r for r in self.manifest.expected_ranks if not self.replicas.get(r))

    @property
    def recoverable(self) -> bool:
        return self.manifest is not None and not self.reasons and not self.missing_ranks

    @property
    def tiers(self) -> dict[int, list[str]]:
        order = {Tier.LOCAL: 0, Tier.PEER: 1, Tier.DURABLE: 2}
        return {
            rank: [str(r.location) for r in sorted(reps, key=lambda r: (order[r.location.tier], r.location.node))]
            for rank, reps in sorted(self.replicas.items())
        }

    def summary(self) -> str:
        if self.recoverable:
            spread = "; ".join(f"r{rank}: {', '.join(locs)}" for rank, locs in self.tiers.items())
            return f"step {self.step}: recoverable ({spread})"
        reasons = ", ".join(self.reasons) or "unknown"
        return f"step {self.step}: not recoverable ({reasons}){f' - {self.detail}' if self.detail else ''}"

    def to_record(self) -> dict:
        return {
            "step": self.step,
            "recoverable": self.recoverable,
            "reasons": [str(r) for r in self.reasons],
            "detail": self.detail,
            "replicas": self.tiers,
            "bad_replicas": [
                {"rank": r.rank, "location": str(r.location), "state": str(r.state)}
                for r in self.bad_replicas
            ],
        }


class CheckpointCatalog:
    def __init__(self, layout: Layout, *, verify_checksums: bool = True) -> None:
        self.layout = layout
        self.verify_checksums = verify_checksums
        # (path, size, mtime_ns) -> sha256. Replicas are immutable once
        # written, so a cache hit is safe and keeps repeated surveys cheap.
        self._hash_cache: dict[tuple[str, int, int], str] = {}

    def survey(
        self,
        *,
        lost_nodes: set[str] | frozenset[str] = frozenset(),
        compatibility: Compatibility | None = None,
        exclude_steps: set[int] | frozenset[int] = frozenset(),
        max_candidates: int = 10,
        conditions: dict[Tier, TierCondition] | None = None,
    ) -> list[CheckpointView]:
        """Newest-first views of the candidate checkpoints."""
        conditions = conditions or read_conditions(self.layout)
        reachable = [
            loc
            for loc in self.layout.locations(exclude_nodes=lost_nodes)
            if conditions[loc.tier].available
        ]

        where: dict[int, list[Location]] = defaultdict(list)
        for location in reachable:
            for step in self.layout.steps_in(location):
                where[step].append(location)

        views = []
        for step in sorted(where, reverse=True)[:max_candidates]:
            view = self._view(step, where[step], compatibility or Compatibility())
            if step in exclude_steps:
                view.reasons.append(RejectionReason.RESTORE_VERIFICATION_FAILED)
                view.detail = "a previous restore from this checkpoint failed verification"
            views.append(view)
        return views

    def latest_recoverable(self, **kwargs) -> tuple[CheckpointView | None, list[CheckpointView]]:
        rejected = []
        for view in self.survey(**kwargs):
            if view.recoverable:
                return view, rejected
            rejected.append(view)
        return None, rejected

    def has_committed_since(self, step: int, *, lost_nodes: set[str] | frozenset[str] = frozenset()) -> bool:
        """Cheap check used during verification: is there a committed
        checkpoint newer than `step` on any reachable location?"""
        for location in self.layout.locations(exclude_nodes=lost_nodes):
            for candidate in self.layout.steps_in(location):
                if candidate <= step:
                    break
                if (self.layout.dir_for(location, candidate) / MANIFEST_NAME).is_file():
                    return True
        return False

    # -------------------------------------------------------------- internals

    def _view(self, step: int, locations: list[Location], compat: Compatibility) -> CheckpointView:
        view = CheckpointView(step=step)

        manifests: dict[str, CheckpointManifest] = {}
        unparseable = []
        for location in locations:
            path = self.layout.dir_for(location, step) / MANIFEST_NAME
            if not path.is_file():
                continue
            try:
                manifest = CheckpointManifest.from_json(path.read_text())
            except (OSError, ValueError, TypeError, KeyError) as exc:
                unparseable.append(f"{location}: {exc}")
                continue
            manifests.setdefault(manifest.commit_key, manifest)
            view.manifest_locations.append(location)

        if not manifests:
            view.reasons.append(
                RejectionReason.MANIFEST_UNPARSEABLE if unparseable else RejectionReason.NO_MANIFEST
            )
            view.detail = "; ".join(unparseable) or "shards present but never committed"
            return view
        if len(manifests) > 1:
            view.reasons.append(RejectionReason.MANIFEST_CONFLICT)
            view.detail = f"{len(manifests)} different commits claim step {step}"
            return view

        manifest = next(iter(manifests.values()))
        view.manifest = manifest

        if manifest.format_version not in SUPPORTED_FORMAT_VERSIONS:
            view.reasons.append(RejectionReason.UNSUPPORTED_FORMAT)
            view.detail = f"format_version={manifest.format_version}"
            return view
        if problems := _incompatibilities(manifest, compat):
            view.reasons.append(RejectionReason.INCOMPATIBLE)
            view.detail = "; ".join(problems)
            return view
        declared = {s.rank for s in manifest.shards}
        if declared != manifest.expected_ranks:
            view.reasons.append(RejectionReason.UNEXPECTED_SHARD_COUNT)
            view.detail = f"manifest declares ranks {sorted(declared)}, expected {sorted(manifest.expected_ranks)}"
            return view

        for entry in manifest.shards:
            for location in locations:
                replica = self._check(entry, location, step)
                if replica.state is ReplicaState.MISSING:
                    continue
                if replica.usable:
                    view.replicas.setdefault(entry.rank, []).append(replica)
                else:
                    view.bad_replicas.append(replica)

        if missing := view.missing_ranks:
            corrupt = {r.rank for r in view.bad_replicas}
            if corrupt & set(missing):
                kinds = {r.state for r in view.bad_replicas if r.rank in missing}
                view.reasons.append(
                    RejectionReason.CHECKSUM_MISMATCH
                    if ReplicaState.CHECKSUM_MISMATCH in kinds
                    else RejectionReason.SIZE_MISMATCH
                )
            else:
                view.reasons.append(RejectionReason.MISSING_SHARD)
            view.detail = f"no intact reachable copy of rank(s) {missing}"
        return view

    def _check(self, entry: ShardEntry, location: Location, step: int) -> Replica:
        path = self.layout.dir_for(location, step) / entry.name
        try:
            stat = path.stat()
        except OSError:
            return Replica(entry.rank, location, path, 0, ReplicaState.MISSING)
        if stat.st_size != entry.size_bytes:
            return Replica(entry.rank, location, path, stat.st_size, ReplicaState.SIZE_MISMATCH)
        if self.verify_checksums and self._sha(path, stat) != entry.sha256:
            return Replica(entry.rank, location, path, stat.st_size, ReplicaState.CHECKSUM_MISMATCH)
        return Replica(entry.rank, location, path, stat.st_size, ReplicaState.OK)

    def _sha(self, path: Path, stat: os.stat_result) -> str:
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if (cached := self._hash_cache.get(key)) is None:
            cached = sha256_file(path)
            self._hash_cache[key] = cached
        return cached


def _incompatibilities(manifest: CheckpointManifest, compat: Compatibility) -> list[str]:
    problems = []
    meta = manifest.metadata
    if compat.job_name is not None and meta.job_name != compat.job_name:
        problems.append(f"job_name {meta.job_name!r} != {compat.job_name!r}")
    if compat.world_size is not None and meta.world_size != compat.world_size:
        problems.append(f"world_size {meta.world_size} != {compat.world_size}")
    if compat.model_signature is not None and meta.model_signature != compat.model_signature:
        problems.append("model structure differs from the job being launched")
    if compat.max_step is not None and manifest.step > compat.max_step:
        problems.append(f"step {manifest.step} is ahead of required {compat.max_step}")
    return problems
