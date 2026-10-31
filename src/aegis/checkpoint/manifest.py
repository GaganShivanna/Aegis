"""Checkpoint completion manifest.

The manifest is the commit point of a checkpoint. Shards are written and
fsynced first; the manifest is published last via an atomic rename. A
checkpoint directory without a manifest is by definition incomplete, which is
what makes a crash mid-write indistinguishable from "never happened".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
SUPPORTED_FORMAT_VERSIONS = frozenset({1})

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
    """Stable fingerprint of parameter names, shapes and dtypes.

    Used to reject a checkpoint written by a structurally different model
    before we attempt to load it into the current one.
    """
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

    @classmethod
    def from_file(cls, rank: int, path: Path) -> ShardEntry:
        return cls(
            rank=rank,
            name=path.name,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Everything needed to decide whether this checkpoint fits the job we are
    about to restart."""

    job_name: str
    world_size: int
    model_signature: str
    optimizer_class: str
    torch_version: str
    dataset_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    format_version: int
    step: int
    created_at: str
    shards: list[ShardEntry]
    metadata: CheckpointMetadata
    # Monotonic run counter; lets us tell a checkpoint from this attempt apart
    # from one left behind by a previous attempt at the same step.
    attempt: int = 0
    run_id: str = ""

    @property
    def expected_ranks(self) -> set[int]:
        return set(range(self.metadata.world_size))

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> CheckpointManifest:
        payload = json.loads(raw)
        shards = [ShardEntry(**s) for s in payload.pop("shards", [])]
        metadata = CheckpointMetadata(**payload.pop("metadata"))
        return cls(shards=shards, metadata=metadata, **payload)

    @classmethod
    def load(cls, directory: Path) -> CheckpointManifest:
        return cls.from_json((directory / MANIFEST_NAME).read_text())
