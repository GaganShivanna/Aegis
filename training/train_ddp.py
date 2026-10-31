"""Fixed-world-size DDP training job with resumable state.

Resumability is the contract Aegis depends on: a restart must continue from
the checkpoint rather than merely start again. That means model, optimizer,
step counter, RNG state and data position all round-trip.

Launched by torchrun; configured entirely through AEGIS_* environment
variables so the controller is the only thing that decides where to resume.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.checkpoint import CheckpointWriter, LocalCheckpointStore, model_signature  # noqa: E402
from aegis.checkpoint.manifest import CheckpointMetadata, shard_name  # noqa: E402
from aegis.checkpoint.validator import CheckpointValidator, Compatibility  # noqa: E402
from aegis.controller.progress import write_progress  # noqa: E402

log = logging.getLogger("train")

SEED = 1234
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 8
BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class Fault:
    """Fault injection, driven by AEGIS_FAULT=kind:step[:rank].

    Used by the acceptance tests and the rehearsal harness to produce the
    failure modes Aegis claims to handle.

    `attempts` decides whether the failure is transient or permanent. It
    defaults to attempt 0 only, modelling a hardware fault that NVSentinel
    repairs before the retry; AEGIS_FAULT_ATTEMPTS=* makes it permanent, which
    is how the unrecoverable cases are exercised.
    """

    kind: str
    step: int
    rank: int | None = None
    attempts: frozenset[int] | None = frozenset({0})  # None means every attempt

    @classmethod
    def parse(cls, raw: str, attempts_raw: str = "0") -> Fault | None:
        if not raw:
            return None
        parts = raw.split(":")
        if len(parts) < 2:
            return None
        rank = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
        attempts: frozenset[int] | None
        if attempts_raw.strip() in ("*", "all"):
            attempts = None
        else:
            attempts = frozenset(
                int(a) for a in attempts_raw.split(",") if a.strip() != ""
            )
        return cls(kind=parts[0], step=int(parts[1]), rank=rank, attempts=attempts)

    def active_on(self, attempt: int) -> bool:
        return self.attempts is None or attempt in self.attempts

    def applies_to(self, rank: int, step: int) -> bool:
        return self.step == step and (self.rank is None or self.rank == rank)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_batch(samples_consumed: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for a given data position.

    Generating from (seed, position, rank) rather than iterating a loader is
    what makes "data position" a single integer that can be checkpointed and
    restored exactly.
    """
    gen = torch.Generator().manual_seed(SEED * 1_000_003 + samples_consumed * 31 + rank)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen)
    y = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), generator=gen)
    return x, y


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    job_name = os.environ.get("AEGIS_JOB_NAME", "train")
    run_id = os.environ.get("AEGIS_RUN_ID", "local")
    attempt = int(os.environ.get("AEGIS_ATTEMPT", "0"))
    ckpt_dir = os.environ.get("AEGIS_CHECKPOINT_DIR", "./checkpoints")
    progress_file = os.environ.get("AEGIS_PROGRESS_FILE", "./progress.json")
    total_steps = int(os.environ.get("AEGIS_TOTAL_STEPS", "500"))
    ckpt_interval = int(os.environ.get("AEGIS_CHECKPOINT_INTERVAL", "10"))
    step_delay = float(os.environ.get("AEGIS_STEP_DELAY", "0.05"))
    resume_raw = os.environ.get("AEGIS_RESUME_STEP", "").strip()
    fault = Fault.parse(
        os.environ.get("AEGIS_FAULT", ""), os.environ.get("AEGIS_FAULT_ATTEMPTS", "0")
    )
    if fault is not None and not fault.active_on(attempt):
        print(f"[rank {rank}] fault {fault.kind} not armed for attempt {attempt}", flush=True)
        fault = None

    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    try:
        return _train(
            rank=rank,
            world_size=world_size,
            job_name=job_name,
            run_id=run_id,
            attempt=attempt,
            ckpt_dir=ckpt_dir,
            progress_file=progress_file,
            total_steps=total_steps,
            ckpt_interval=ckpt_interval,
            step_delay=step_delay,
            resume_step=int(resume_raw) if resume_raw else None,
            fault=fault,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(
    *,
    rank: int,
    world_size: int,
    job_name: str,
    run_id: str,
    attempt: int,
    ckpt_dir: str,
    progress_file: str,
    total_steps: int,
    ckpt_interval: int,
    step_delay: float,
    resume_step: int | None,
    fault: Fault | None,
) -> int:
    torch.manual_seed(SEED + rank)
    store = LocalCheckpointStore(ckpt_dir)

    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_step = 0
    samples_consumed = 0

    if resume_step is not None:
        start_step, samples_consumed = _restore(
            store=store,
            step=resume_step,
            rank=rank,
            world_size=world_size,
            job_name=job_name,
            model=model,
            optimizer=optimizer,
        )
        print(
            f"[rank {rank}] resumed from step {start_step} "
            f"(samples_consumed={samples_consumed})",
            flush=True,
        )
    else:
        print(f"[rank {rank}] starting from scratch", flush=True)

    # Every rank must enter DDP with identical weights; after a restore they
    # all loaded rank 0's shard, so this only guards the from-scratch path.
    ddp = DistributedDataParallel(model)
    for param in ddp.module.parameters():
        dist.broadcast(param.data, src=0)

    writer = CheckpointWriter(
        store,
        rank=rank,
        world_size=world_size,
        job_name=job_name,
        run_id=run_id,
        attempt=attempt,
        barrier=dist.barrier,
    )
    metadata = CheckpointMetadata(
        job_name=job_name,
        world_size=world_size,
        model_signature=model_signature(ddp.module.state_dict()),
        optimizer_class=type(optimizer).__name__,
        torch_version=torch.__version__,
        dataset_fingerprint=f"synthetic-{INPUT_DIM}x{OUTPUT_DIM}-seed{SEED}",
    )

    for step in range(start_step + 1, total_steps + 1):
        _maybe_fault(fault, rank, step, phase="pre_step")

        x, y = make_batch(samples_consumed, rank)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(ddp(x), y)
        loss.backward()
        optimizer.step()
        samples_consumed += BATCH_SIZE

        if rank == 0:
            write_progress(
                progress_file,
                step=step,
                run_id=run_id,
                attempt=attempt,
                loss=float(loss.detach()),
            )

        if step % ckpt_interval == 0:
            _save(
                writer=writer,
                metadata=metadata,
                step=step,
                rank=rank,
                model=ddp.module,
                optimizer=optimizer,
                samples_consumed=samples_consumed,
                fault=fault,
            )
            if rank == 0:
                writer.prune(keep=5)

        _maybe_fault(fault, rank, step, phase="post_step")

        if step_delay:
            time.sleep(step_delay)

    dist.barrier()
    if rank == 0:
        write_progress(
            progress_file,
            step=total_steps,
            run_id=run_id,
            attempt=attempt,
            finished=True,
        )
        print(f"[rank 0] training complete at step {total_steps}", flush=True)
    return 0


def _save(
    *,
    writer: CheckpointWriter,
    metadata: CheckpointMetadata,
    step: int,
    rank: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    samples_consumed: int,
    fault: Fault | None,
) -> None:
    payload = {
        "step": step,
        "samples_consumed": samples_consumed,
        "rng_state": torch.get_rng_state(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    hit = fault is not None and fault.step == step
    # Crash-mid-write: shards land but the manifest never does.
    skip_manifest = hit and fault.kind == "incomplete_checkpoint"
    # Manifest published while one rank's shard never arrived.
    drop_shard = hit and fault.kind == "missing_shard" and rank == writer.world_size - 1

    writer.save(
        step=step,
        shard_payload=payload,
        metadata=metadata,
        drop_shard=drop_shard,
        skip_manifest=skip_manifest,
    )

    if drop_shard:
        print(f"[rank {rank}] FAULT: withheld shard for step {step}", flush=True)

    # Both faults end the run: an incomplete checkpoint is only interesting
    # because something died around the time it was written.
    if skip_manifest or (hit and fault.kind == "missing_shard"):
        print(f"[rank {rank}] FAULT: dying at step {step}", flush=True)
        os._exit(1)

    if fault is not None and fault.kind == "corrupt_checkpoint" and fault.step == step:
        dist.barrier()
        if rank == 0:
            target = writer.store.path_for(step) / shard_name(0)
            with target.open("r+b") as fh:
                fh.truncate(max(0, target.stat().st_size // 2))
            print(f"[rank 0] FAULT: truncated shard for step {step}", flush=True)


def _restore(
    *,
    store: LocalCheckpointStore,
    step: int,
    rank: int,
    world_size: int,
    job_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Load a checkpoint the controller already validated, re-validating here.

    The second check is not redundant: the controller validated with a generic
    compatibility profile, while this process knows the exact model it is about
    to load into.
    """
    validator = CheckpointValidator(store)
    result = validator.validate(
        step,
        Compatibility(
            job_name=job_name,
            world_size=world_size,
            model_signature=model_signature(model.state_dict()),
            optimizer_class=type(optimizer).__name__,
        ),
    )
    if not result.valid:
        raise RuntimeError(f"refusing to resume from invalid checkpoint: {result.summary}")

    directory = store.path_for(step)
    # Model and optimizer come from rank 0's shard so every rank starts
    # identical; RNG and data position are per-rank.
    shared = torch.load(directory / shard_name(0), map_location="cpu", weights_only=True)
    model.load_state_dict(shared["model"])
    optimizer.load_state_dict(shared["optimizer"])

    own = torch.load(directory / shard_name(rank), map_location="cpu", weights_only=True)
    torch.set_rng_state(own["rng_state"].to(torch.uint8))
    return int(own["step"]), int(own["samples_consumed"])


def _maybe_fault(fault: Fault | None, rank: int, step: int, *, phase: str) -> None:
    if fault is None or not fault.applies_to(rank, step):
        return
    if fault.kind == "crash" and phase == "post_step":
        print(f"[rank {rank}] FAULT: hard exit at step {step}", flush=True)
        os._exit(1)
    if fault.kind == "hang" and phase == "post_step":
        print(f"[rank {rank}] FAULT: hanging at step {step}", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
