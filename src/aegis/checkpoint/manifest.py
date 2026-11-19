"""Checkpoint commit record.

The manifest is written once, after every rank's shard is durable on its
primary (local) location, and copies of it travel with every replica. Its
existence anywhere is what makes a checkpoint *committed*; a set of shards
with no reachable manifest is an incomplete write and is never used.

The manifest is also the integrity reference for every replica: a shard
fetched from a peer node or from object storage is trusted only if its size
and SHA-256 match the entry recorded here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
SUPPORTED_FORMAT_VERSIONS = frozenset({2})

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def shard_name(rank: int) -> str:
    return f"shard_rank{rank:05d}.pt"


def model_signature(state_dict: dict[str, Any]) -> str:
    """Fingerprint of parameter names, shapes and dtypes, used to reject a
    checkpoint written by a structurally different model."""
    parts = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        shape = tuple(getattr(tensor, "shape", ()))
        dtype = str(getattr(tensor, "dtype", type(tensor).__name__))
        parts.append(f"{name}|{shape}|{dtype}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ShardEntry:
    rank: int
    name: str
    size_bytes: int
    sha256: str
    # Node holding the primary copy at write time.
    node: str = ""

    @classmethod
    def from_file(cls, rank: int, path: Path, node: str = "") -> ShardEntry:
        return cls(
            rank=rank,
            name=path.name,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            node=node,
        )


@dataclass(frozen=True, slots=True)
class RankState:
    rank: int
    rng: str
    data_position: str
    samples_consumed: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    job_name: str
    world_size: int
    model_signature: str
    optimizer_class: str
    torch_version: str
    scheduler_class: str = ""
    dataset_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    format_version: int
    step: int
    created_at: str
    shards: list[ShardEntry]
    metadata: CheckpointMetadata
    # Digests of replicated state: model, optimizer, scheduler, step.
    state_digests: dict[str, str] = field(default_factory=dict)
    # Per-rank digests: RNG and data position.
    rank_state: list[RankState] = field(default_factory=list)
    attempt: int = 0
    run_id: str = ""

    @property
    def expected_ranks(self) -> set[int]:
        return set(range(self.metadata.world_size))

    def shard_for(self, rank: int) -> ShardEntry | None:
        return next((s for s in self.shards if s.rank == rank), None)

    def rank_state_for(self, rank: int) -> RankState | None:
        return next((r for r in self.rank_state if r.rank == rank), None)

    @property
    def total_bytes(self) -> int:
        return sum(s.size_bytes for s in self.shards)

    @property
    def commit_key(self) -> str:
        """Identity of the committed content. Copies of the same checkpoint
        must agree on it; copies that disagree mean two different writes
        claimed the same step, and neither can be trusted."""
        parts = [str(self.step)] + [
            f"{s.rank}:{s.size_bytes}:{s.sha256}" for s in sorted(self.shards, key=lambda s: s.rank)
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> CheckpointManifest:
        payload = json.loads(raw)
        shards = [ShardEntry(**s) for s in payload.pop("shards", [])]
        rank_state = [RankState(**r) for r in payload.pop("rank_state", [])]
        metadata = CheckpointMetadata(**payload.pop("metadata"))
        return cls(shards=shards, rank_state=rank_state, metadata=metadata, **payload)

    @classmethod
    def load(cls, directory: Path) -> CheckpointManifest:
        return cls.from_json((directory / MANIFEST_NAME).read_text())
