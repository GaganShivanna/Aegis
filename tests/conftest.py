from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from aegis.checkpoint import (
    FORMAT_VERSION,
    MANIFEST_NAME,
    CheckpointManifest,
    CheckpointMetadata,
    Layout,
    Location,
    RankState,
    ShardEntry,
    Tier,
    model_signature,
    peer_of,
)
from aegis.checkpoint.manifest import shard_name
from aegis.checkpoint.state import global_digests, rank_digests
from aegis.clock import ManualClock
from aegis.models import JobSpec, RecoveryPolicy
from aegis.store.repository import Repository
from tests.fakes import FakeBackend


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def repo(tmp_path: Path, clock: ManualClock) -> Repository:
    return Repository.from_url(f"sqlite:///{tmp_path / 'aegis.sqlite'}", clock=clock)


@pytest.fixture
def policy() -> RecoveryPolicy:
    return RecoveryPolicy(
        name="test",
        version=1,
        max_attempts=3,
        backoff_base_seconds=10.0,
        backoff_multiplier=2.0,
        jitter_ratio=0.0,
        correlation_window_seconds=5.0,
        stall_timeout_seconds=60.0,
        required_progress_steps=20,
        verification_timeout_seconds=300.0,
    )


@pytest.fixture
def spec(tmp_path: Path, policy: RecoveryPolicy) -> JobSpec:
    return JobSpec(
        job_id="job-test-0001",
        name="unit",
        world_size=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        workdir=str(tmp_path / "work"),
        command=["true"],
        total_steps=500,
        policy=policy,
    )


@pytest.fixture
def layout(spec: JobSpec) -> Layout:
    return Layout(spec.checkpoint_dir)


@pytest.fixture
def backend(spec: JobSpec) -> FakeBackend:
    return FakeBackend(progress_file=str(Path(spec.workdir) / "progress.json"))


class CheckpointFactory:
    """Writes tiered checkpoints laid out exactly as the training job does:
    each rank's primary on its own node, a ring replica on the next node,
    and optionally a committed copy in durable storage."""

    def __init__(self, layout: Layout, world_size: int, job_name: str) -> None:
        self.layout = layout
        self.world_size = world_size
        self.job_name = job_name
        self.model = torch.nn.Linear(4, 2)
        self.placement = [f"node-{r}" for r in range(world_size)]

    def metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            job_name=self.job_name,
            world_size=self.world_size,
            model_signature=model_signature(self.model.state_dict()),
            optimizer_class="Adam",
            torch_version=torch.__version__,
        )

    def payload(self, step: int, rank: int) -> dict:
        return {
            "step": step,
            "samples_consumed": step * 32,
            "rng_state": torch.get_rng_state(),
            "model": self.model.state_dict(),
            "optimizer": {"state": {}, "param_groups": [{"lr": 0.001}]},
            "scheduler": {},
            "rank": rank,
        }

    def write(
        self,
        step: int,
        *,
        peer: bool = True,
        durable: bool = False,
        skip_manifest: bool = False,
        drop_rank: int | None = None,
        placement: list[str] | None = None,
    ) -> CheckpointManifest:
        placement = placement or self.placement
        shards, states = [], []
        for rank in range(self.world_size):
            directory = self.layout.dir_for(Location(Tier.LOCAL, placement[rank]), step)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / shard_name(rank)
            torch.save(self.payload(step, rank), path)
            shards.append(ShardEntry.from_file(rank, path, placement[rank]))
            r = rank_digests(rng_state=torch.get_rng_state(), samples_consumed=step * 32)
            states.append(RankState(rank, r["rng"], r["data_position"], step * 32))
        manifest = CheckpointManifest(
            format_version=FORMAT_VERSION,
            step=step,
            created_at=datetime.now(UTC).isoformat(),
            shards=shards,
            metadata=self.metadata(),
            state_digests=global_digests(
                model_state=self.model.state_dict(),
                optimizer_state={"state": {}, "param_groups": [{"lr": 0.001}]},
                scheduler_state={},
                step=step,
            ),
            rank_state=states,
        )
        for rank in range(self.world_size):
            local = self.layout.dir_for(Location(Tier.LOCAL, placement[rank]), step)
            if drop_rank == rank:
                (local / shard_name(rank)).unlink()
            if not skip_manifest:
                (local / MANIFEST_NAME).write_text(manifest.to_json())
            if peer and drop_rank != rank and (node := peer_of(rank, placement)):
                self._copy(rank, step, local, Location(Tier.PEER, node), manifest, skip_manifest)
            if durable and drop_rank != rank:
                self._copy(rank, step, local, Location(Tier.DURABLE), manifest, skip_manifest)
        return manifest

    def _copy(self, rank, step, source_dir, location, manifest, skip_manifest) -> None:
        target = self.layout.dir_for(location, step)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / shard_name(rank), target / shard_name(rank))
        if not skip_manifest:
            (target / MANIFEST_NAME).write_text(manifest.to_json())

    def corrupt(self, step: int, rank: int, location: Location) -> Path:
        path = self.layout.dir_for(location, step) / shard_name(rank)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        return path

    def remove(self, step: int, rank: int, location: Location) -> None:
        (self.layout.dir_for(location, step) / shard_name(rank)).unlink()


@pytest.fixture
def checkpoints(layout: Layout, spec: JobSpec) -> CheckpointFactory:
    return CheckpointFactory(layout, spec.world_size, spec.name)
