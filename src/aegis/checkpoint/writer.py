"""Collective, tiered checkpoint writer.

The blocking part of a save is deliberately small:

  1. every rank writes and fsyncs its shard to its own node's local disk
  2. ranks exchange shard checksums and per-rank state digests
  3. rank 0 builds the manifest; every rank stores a copy beside its shard

At that point the checkpoint is committed and training resumes. Copies to a
peer node and to durable storage happen afterwards on a background thread, so
a failure shortly after a save can find the newest checkpoint only on local
disks, an older one on peers, and an older one still in object storage. That
spread is real, and it is what the recovery planner reasons about.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from aegis.checkpoint.layout import Layout, Location, Tier, peer_of
from aegis.checkpoint.manifest import (
    FORMAT_VERSION,
    MANIFEST_NAME,
    CheckpointManifest,
    CheckpointMetadata,
    RankState,
    ShardEntry,
    shard_name,
)
from aegis.checkpoint.store import StorageFault, atomic_write_text
from aegis.checkpoint.tiers import StorageProfiles, read_conditions, throttled_copy
from aegis.telemetry import MetricsLog

UPLOAD_MARKER = ".uploaded.rank{rank:05d}"


@dataclass(slots=True)
class Collective:
    """The three collective operations a save needs, injectable for tests."""

    rank: int
    world_size: int
    barrier: Callable[[], None]
    all_gather: Callable[[Any], list[Any]]
    broadcast: Callable[[Any], Any]

    @classmethod
    def from_torch(cls) -> Collective:
        import torch.distributed as dist

        def all_gather(obj: Any) -> list[Any]:
            out: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(out, obj)
            return out

        def broadcast(obj: Any) -> Any:
            box = [obj]
            dist.broadcast_object_list(box, src=0)
            return box[0]

        return cls(dist.get_rank(), dist.get_world_size(), dist.barrier, all_gather, broadcast)

    @classmethod
    def single(cls) -> Collective:
        return cls(0, 1, lambda: None, lambda obj: [obj], lambda obj: obj)


@dataclass(frozen=True, slots=True)
class ReplicationPolicy:
    peer: bool = True
    # Object storage receives every checkpoint whose step is a multiple of
    # this. Durable uploads are slow; uploading every checkpoint would build
    # an ever-growing backlog. 0 disables durable replication.
    durable_interval_steps: int = 20
    keep_local: int = 3
    keep_peer: int = 3
    keep_durable: int = 4


@dataclass(frozen=True, slots=True)
class SaveResult:
    step: int
    blocking_seconds: float
    shard_bytes: int
    committed: bool
    durable: bool = False


@dataclass(slots=True)
class _Job:
    step: int
    manifest_json: str
    shard_path: Path
    durable: bool
    submitted_at: float


class _Lane:
    """One background worker draining a queue of copy jobs in order."""

    def __init__(self, name: str, handler: Callable[[_Job], None]) -> None:
        self.handler = handler
        self.queue: deque[_Job] = deque()
        self.cv = threading.Condition()
        self.in_flight: int | None = None
        self.done: set[int] = set()
        self.closing = False
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, job: _Job) -> None:
        with self.cv:
            self.queue.append(job)
            self.cv.notify_all()

    def pending(self) -> set[int]:
        with self.cv:
            steps = {job.step for job in self.queue}
            if self.in_flight is not None:
                steps.add(self.in_flight)
            return steps

    def newer_queued(self, step: int) -> bool:
        with self.cv:
            return any(job.step > step for job in self.queue)

    def wait(self, step: int, deadline: float) -> bool:
        with self.cv:
            while step not in self.done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.cv.wait(remaining)
            return True

    def close(self) -> None:
        with self.cv:
            self.closing = True
            self.cv.notify_all()

    def join(self, deadline: float) -> None:
        self.thread.join(max(0.0, deadline - time.monotonic()))

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.queue and not self.closing:
                    self.cv.wait()
                if not self.queue:
                    return
                job = self.queue.popleft()
                self.in_flight = job.step
            try:
                self.handler(job)
            finally:
                with self.cv:
                    self.in_flight = None
                    self.done.add(job.step)
                    self.cv.notify_all()


class Replicator:
    """Background copies of one rank's shards to its peer node and to durable
    storage.

    A durable checkpoint is committed by whichever rank finishes uploading
    last: every rank writes a marker after its upload and then checks for
    everyone else's. Because each rank checks *after* writing its own marker,
    the last one always sees the full set, and the commit never waits behind
    anyone's next upload. Two ranks may both commit; the manifest is identical
    and the write is atomic, so that is harmless.

    Peer and durable copies run on separate lanes. Object storage is an order
    of magnitude slower; sharing one queue would leave peer replicas several
    checkpoints stale every time an upload was in flight, which is exactly
    when a node loss would need them.
    """

    def __init__(
        self,
        layout: Layout,
        *,
        rank: int,
        world_size: int,
        peer_node: str | None,
        policy: ReplicationPolicy,
        profiles: StorageProfiles,
        metrics: MetricsLog,
    ) -> None:
        self.layout = layout
        self.rank = rank
        self.world_size = world_size
        self.peer_node = peer_node
        self.policy = policy
        self.profiles = profiles
        self.metrics = metrics

        self._peer = _Lane(f"aegis-peer-{rank}", self._replicate_peer) if policy.peer and peer_node else None
        self._durable = (
            _Lane(f"aegis-durable-{rank}", self._replicate_durable) if policy.durable_interval_steps > 0 else None
        )

    @property
    def _lanes(self) -> list[_Lane]:
        return [lane for lane in (self._peer, self._durable) if lane is not None]

    # -------------------------------------------------------------- control

    def submit(self, job: _Job) -> None:
        if self._peer is not None:
            self._peer.submit(job)
        if job.durable and self._durable is not None:
            self._durable.submit(job)

    def pending_steps(self) -> set[int]:
        steps: set[int] = set()
        for lane in self._lanes:
            steps |= lane.pending()
        return steps

    def wait(self, step: int, timeout: float = 60.0) -> bool:
        """Wait until the peer copy of `step` has been attempted."""
        if self._peer is None:
            return True
        return self._peer.wait(step, time.monotonic() + timeout)

    def close(self, timeout: float = 30.0) -> None:
        """Drain outstanding copies, bounded by `timeout`."""
        deadline = time.monotonic() + timeout
        for lane in self._lanes:
            lane.close()
        for lane in self._lanes:
            lane.join(deadline)

    # ------------------------------------------------------------- handlers

    def _replicate_peer(self, job: _Job) -> None:
        location = Location(Tier.PEER, self.peer_node)
        self._copy(job, location)
        self._prune(location, self.policy.keep_peer, protect=set())

    def _replicate_durable(self, job: _Job) -> None:
        # A newer upload is already queued, so this one would only delay it.
        # Skipping keeps object storage as fresh as bandwidth allows.
        if self._durable is not None and self._durable.newer_queued(job.step):
            self.metrics.emit("replication_skipped", rank=self.rank, tier="durable", step=job.step)
            return
        if not self._copy(job, Location(Tier.DURABLE)):
            return
        directory = self.layout.dir_for(Location(Tier.DURABLE), job.step)
        try:
            (directory / UPLOAD_MARKER.format(rank=self.rank)).touch()
            markers = [directory / UPLOAD_MARKER.format(rank=r) for r in range(self.world_size)]
            if not all(m.exists() for m in markers) or (directory / MANIFEST_NAME).exists():
                return
            atomic_write_text(directory / MANIFEST_NAME, job.manifest_json)
        except OSError as exc:
            self.metrics.emit("replication_failed", rank=self.rank, tier="durable", step=job.step, error=str(exc))
            return
        self.metrics.emit("durable_committed", rank=self.rank, step=job.step, lag_s=time.time() - job.submitted_at)
        self._prune(Location(Tier.DURABLE), self.policy.keep_durable, protect=self.pending_steps())

    def _copy(self, job: _Job, location: Location) -> bool:
        target_dir = self.layout.dir_for(location, job.step)
        condition = read_conditions(self.layout)[location.tier]
        try:
            seconds = throttled_copy(
                job.shard_path,
                target_dir / job.shard_path.name,
                tier=location.tier,
                profiles=self.profiles,
                condition=condition,
            )
            # Durable gets its manifest only once every rank has uploaded;
            # a peer directory holds one complete replica and can have it now.
            if location.tier is Tier.PEER:
                atomic_write_text(target_dir / MANIFEST_NAME, job.manifest_json)
        except (StorageFault, OSError) as exc:
            self.metrics.emit(
                "replication_failed", rank=self.rank, tier=str(location.tier), step=job.step, error=str(exc)
            )
            return False
        self.metrics.emit(
            "replicated",
            rank=self.rank,
            tier=str(location.tier),
            step=job.step,
            seconds=seconds,
            bytes=(target_dir / job.shard_path.name).stat().st_size,
            lag_s=time.time() - job.submitted_at,
        )
        return True

    def _prune(self, location: Location, keep: int, *, protect: set[int]) -> None:
        if keep <= 0:
            return
        steps = self.layout.steps_in(location)
        committed = [s for s in steps if (self.layout.dir_for(location, s) / MANIFEST_NAME).exists()]
        if len(committed) <= keep:
            return
        floor = committed[keep - 1]
        for step in steps:
            if step < floor and step not in protect:
                shutil.rmtree(self.layout.dir_for(location, step), ignore_errors=True)


class TieredCheckpointWriter:
    def __init__(
        self,
        layout: Layout,
        collective: Collective,
        *,
        placement: list[str],
        job_name: str,
        run_id: str = "",
        attempt: int = 0,
        replication: ReplicationPolicy | None = None,
        profiles: StorageProfiles,
        metrics: MetricsLog | None = None,
    ) -> None:
        if len(placement) != collective.world_size:
            raise ValueError(f"placement {placement} does not cover {collective.world_size} ranks")
        self.layout = layout
        self.collective = collective
        self.rank = collective.rank
        self.world_size = collective.world_size
        self.placement = list(placement)
        self.node = placement[self.rank]
        self.job_name = job_name
        self.run_id = run_id
        self.attempt = attempt
        self.replication = replication or ReplicationPolicy()
        self.profiles = profiles
        self.metrics = metrics or MetricsLog(None)
        self.replicator = Replicator(
            layout,
            rank=self.rank,
            world_size=self.world_size,
            peer_node=peer_of(self.rank, placement),
            policy=self.replication,
            profiles=profiles,
            metrics=self.metrics,
        )

    @property
    def local(self) -> Location:
        return Location(Tier.LOCAL, self.node)

    def save(
        self,
        *,
        step: int,
        payload: dict[str, Any],
        metadata: CheckpointMetadata,
        global_digests: dict[str, str],
        rank_digests: dict[str, str],
        samples_consumed: int,
        skip_manifest: bool = False,
    ) -> SaveResult:
        """Collective save; every rank must call it.

        `skip_manifest` is fault injection: shards land, the commit record
        never does, which is what a crash mid-checkpoint leaves behind.
        """
        started = time.monotonic()
        directory = self.layout.dir_for(self.local, step)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

        shard = directory / shard_name(self.rank)
        _atomic_torch_save(payload, shard)
        entry = ShardEntry.from_file(self.rank, shard, self.node)
        state = RankState(
            rank=self.rank,
            rng=rank_digests["rng"],
            data_position=rank_digests["data_position"],
            samples_consumed=samples_consumed,
        )

        # Doubles as the barrier: nobody gets past here until every shard is
        # durable on its primary location.
        gathered = self.collective.all_gather((asdict(entry), asdict(state)))
        if skip_manifest:
            return SaveResult(step, time.monotonic() - started, entry.size_bytes, committed=False)

        manifest_json = None
        if self.rank == 0:
            manifest_json = CheckpointManifest(
                format_version=FORMAT_VERSION,
                step=step,
                created_at=datetime.now(UTC).isoformat(),
                shards=[ShardEntry(**e) for e, _ in gathered],
                metadata=metadata,
                state_digests=dict(global_digests),
                rank_state=[RankState(**s) for _, s in gathered],
                attempt=self.attempt,
                run_id=self.run_id,
            ).to_json()
        manifest_json = self.collective.broadcast(manifest_json)
        atomic_write_text(directory / MANIFEST_NAME, manifest_json)

        blocking = time.monotonic() - started
        interval = self.replication.durable_interval_steps
        durable = interval > 0 and step % interval == 0
        self.replicator.submit(_Job(step, manifest_json, shard, durable, time.time()))
        self._prune_local()

        self.metrics.emit(
            "checkpoint",
            rank=self.rank,
            step=step,
            blocking_s=blocking,
            bytes=entry.size_bytes,
            durable=durable,
        )
        return SaveResult(step, blocking, entry.size_bytes, committed=True, durable=durable)

    def corrupt_local_shard(self, step: int) -> Path:
        """Fault injection: flip bytes in this rank's primary copy, after its
        replicas were taken, as bit rot on a local disk would."""
        self.replicator.wait(step, timeout=60)
        path = self.layout.dir_for(self.local, step) / shard_name(self.rank)
        data = bytearray(path.read_bytes())
        for offset in range(len(data) // 2, min(len(data), len(data) // 2 + 64)):
            data[offset] ^= 0xFF
        path.write_bytes(bytes(data))
        return path

    def close(self, timeout: float = 30.0) -> None:
        self.replicator.close(timeout)

    def _prune_local(self) -> None:
        keep = self.replication.keep_local
        if keep <= 0:
            return
        protect = self.replicator.pending_steps()
        steps = self.layout.steps_in(self.local)
        for step in steps[keep:]:
            if step not in protect:
                shutil.rmtree(self.layout.dir_for(self.local, step), ignore_errors=True)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
