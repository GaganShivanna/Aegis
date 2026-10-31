"""Writes checkpoints that are either complete or absent, never half-written.

Ordering matters and is the whole point of this module:
  1. every rank writes and fsyncs its own shard
  2. all ranks rendezvous (so rank 0 knows the shards are durable)
  3. rank 0 hashes the shards and publishes the manifest by atomic rename

A crash at any point before step 3 leaves a directory the validator rejects.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from aegis.checkpoint.manifest import (
    FORMAT_VERSION,
    CheckpointManifest,
    CheckpointMetadata,
    ShardEntry,
    shard_name,
)
from aegis.checkpoint.store import CheckpointStore

Barrier = Callable[[], None]


def _noop_barrier() -> None:
    return None


class CheckpointWriter:
    def __init__(
        self,
        store: CheckpointStore,
        *,
        rank: int,
        world_size: int,
        job_name: str,
        run_id: str = "",
        attempt: int = 0,
        barrier: Barrier = _noop_barrier,
    ) -> None:
        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.job_name = job_name
        self.run_id = run_id
        self.attempt = attempt
        self.barrier = barrier

    def begin(self, step: int) -> Path:
        """Rank 0 clears any remains of a previous write at this step."""
        if self.rank == 0:
            return self.store.begin(step)
        return self.store.path_for(step)

    def write_shard(self, step: int, payload: dict[str, Any]) -> Path:
        path = self.store.path_for(step) / shard_name(self.rank)
        _atomic_torch_save(payload, path)
        return path

    def publish(self, step: int, metadata: CheckpointMetadata) -> Path:
        """Rank 0 commits the checkpoint by publishing the manifest."""
        directory = self.store.path_for(step)
        shards = [
            ShardEntry.from_file(rank, directory / shard_name(rank))
            for rank in range(self.world_size)
            if (directory / shard_name(rank)).is_file()
        ]
        manifest = CheckpointManifest(
            format_version=FORMAT_VERSION,
            step=step,
            created_at=datetime.now(UTC).isoformat(),
            shards=shards,
            metadata=metadata,
            attempt=self.attempt,
            run_id=self.run_id,
        )
        self.store.publish(step, manifest.to_json())
        return directory

    def save(
        self,
        *,
        step: int,
        shard_payload: dict[str, Any],
        metadata: CheckpointMetadata,
        drop_shard: bool = False,
        skip_manifest: bool = False,
    ) -> Path:
        """Collective save. Every rank calls this; the barriers order it.

        `drop_shard` and `skip_manifest` are fault-injection switches used to
        manufacture the incomplete checkpoints the validator must reject.
        """
        self.begin(step)
        self.barrier()

        if not drop_shard:
            self.write_shard(step, shard_payload)

        # Every shard is durable on the other side of this barrier, so rank 0
        # can safely hash them and declare the checkpoint complete.
        self.barrier()

        if self.rank == 0 and not skip_manifest:
            return self.publish(step, metadata)
        return self.store.path_for(step)

    def prune(self, keep: int) -> None:
        """Retain the newest `keep` complete checkpoints plus any incomplete
        ones newer than them (those are someone else's in-flight work)."""
        if self.rank != 0 or keep <= 0:
            return
        complete = [s for s in self.store.list_steps() if self.store.has_manifest(s)]
        for step in complete[keep:]:
            self.store.delete(step)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
