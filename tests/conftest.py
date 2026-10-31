from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aegis.checkpoint import CheckpointWriter, LocalCheckpointStore, model_signature
from aegis.checkpoint.manifest import CheckpointMetadata
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
def store(spec: JobSpec) -> LocalCheckpointStore:
    return LocalCheckpointStore(spec.checkpoint_dir)


@pytest.fixture
def backend(spec: JobSpec) -> FakeBackend:
    return FakeBackend(progress_file=str(Path(spec.workdir) / "progress.json"))


class CheckpointFactory:
    """Writes checkpoints shaped exactly like the training script's."""

    def __init__(self, store: LocalCheckpointStore, world_size: int, job_name: str) -> None:
        self.store = store
        self.world_size = world_size
        self.job_name = job_name
        self.model = torch.nn.Linear(4, 2)

    def metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            job_name=self.job_name,
            world_size=self.world_size,
            model_signature=model_signature(self.model.state_dict()),
            optimizer_class="Adam",
            torch_version=torch.__version__,
        )

    def _writer(self, rank: int, step: int) -> CheckpointWriter:
        return CheckpointWriter(
            self.store,
            rank=rank,
            world_size=self.world_size,
            job_name=self.job_name,
            run_id=f"run-{step}",
        )

    def _payload(self, step: int) -> dict:
        return {
            "step": step,
            "samples_consumed": step * 32,
            "rng_state": torch.get_rng_state(),
            "model": self.model.state_dict(),
            "optimizer": {"state": {}, "param_groups": [{"lr": 0.001}]},
        }

    def write(
        self, step: int, *, skip_manifest: bool = False, drop_rank: int | None = None
    ) -> None:
        """Drive the collective save serially, in the order the barriers impose."""
        self._writer(0, step).begin(step)
        for rank in range(self.world_size):
            if rank != drop_rank:
                self._writer(rank, step).write_shard(step, self._payload(step))
        if not skip_manifest:
            self._writer(0, step).publish(step, self.metadata())


@pytest.fixture
def checkpoints(store: LocalCheckpointStore, spec: JobSpec) -> CheckpointFactory:
    return CheckpointFactory(store, spec.world_size, spec.name)
