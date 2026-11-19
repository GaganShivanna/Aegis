"""Fixed-world-size DDP training job with verifiable, tiered checkpoints.

Resumability is the contract Aegis depends on: a restart must continue the
same training run, not merely start another one. So model, optimizer, LR
schedule, step counter, RNG state and data position all round-trip, and the
restore proves it: digests recorded at save time are recomputed after load
and must match, and replicated state must agree across every rank.

The task is a teacher-student regression onto a fixed random network, so loss
actually falls and "time to reach a training target" means something.

Launched by torchrun; configured entirely through AEGIS_* environment
variables so the controller alone decides where to resume and from which
copies.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aegis.checkpoint import (  # noqa: E402
    CheckpointCatalog,
    CheckpointManifest,
    CheckpointMetadata,
    Collective,
    Layout,
    Location,
    ReplicationPolicy,
    RestorePlan,
    RestoreReport,
    ShardSource,
    StagingError,
    Tier,
    TieredCheckpointWriter,
    get_profiles,
    model_signature,
)
from aegis.checkpoint.manifest import shard_name  # noqa: E402
from aegis.checkpoint.restore import REPORT_NAME, RankRestoreReport, stage_rank  # noqa: E402
from aegis.checkpoint.state import compare, global_digests, rank_digests  # noqa: E402
from aegis.controller.progress import write_progress  # noqa: E402
from aegis.events.synthetic import emit_event  # noqa: E402
from aegis.models import SignalKind  # noqa: E402
from aegis.telemetry import MetricsLog  # noqa: E402

SEED = 1234
TEACHER_SEED = 99
EVAL_SEED = 7
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 8
BATCH_SIZE = 32
EVAL_SAMPLES = 2048


@dataclass(frozen=True, slots=True)
class Fault:
    """Fault injection, driven by AEGIS_FAULT=kind:step[:rank].

    kinds:
      crash               hard exit after `step`
      hang                stop making progress after `step` (process stays up)
      incomplete_checkpoint  shards for `step` land, the manifest never does, then exit
      corrupt_shard       flip bytes in the primary copy of the shard for `step`, then exit;
                          replicas stay intact
      corrupt_everywhere  corrupt every copy of the shard for `step`, then exit
      node_loss           the rank's node and its disks disappear at `step`
      skew_restore        perturb restored data position (step ignored) so
                          restore verification must catch it

    Each fault is armed on a set of attempts: "0" (default, transient),
    "0,1" (fails again during recovery), "*" (permanent). Several faults can
    be combined with ";", each optionally suffixed "@attempts", e.g.
    "crash:45:1@0;skew_restore:0@1". AEGIS_FAULT_ATTEMPTS is the default for
    faults without a suffix.
    """

    kind: str
    step: int
    rank: int | None = None
    attempts: frozenset[int] | None = frozenset({0})

    @classmethod
    def parse_all(cls, raw: str, default_attempts: str = "0") -> list[Fault]:
        faults = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            spec, _, attempts = item.partition("@")
            if (fault := cls.parse(spec, attempts or default_attempts)) is not None:
                faults.append(fault)
        return faults

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
            attempts = frozenset(int(a) for a in attempts_raw.split(",") if a.strip() != "")
        return cls(kind=parts[0], step=int(parts[1]), rank=rank, attempts=attempts)

    def active_on(self, attempt: int) -> bool:
        return self.attempts is None or attempt in self.attempts

    def applies_to(self, rank: int, step: int) -> bool:
        return self.step == step and (self.rank is None or self.rank == rank)

    def targets(self, rank: int, world_size: int) -> bool:
        target = self.rank if self.rank is not None else world_size - 1
        return rank == target


class ToyModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_teacher() -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(TEACHER_SEED)
        teacher = ToyModel(hidden=32)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval()


def make_batch(teacher: nn.Module, samples_consumed: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for a data position.

    Generated from (seed, position, rank) rather than by iterating a loader,
    which is what makes "data position" a single integer that can be
    checkpointed and restored exactly.
    """
    gen = torch.Generator().manual_seed(SEED * 1_000_003 + samples_consumed * 31 + rank)
    x = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen)
    with torch.no_grad():
        y = teacher(x)
    return x, y


def make_eval_set(teacher: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(EVAL_SEED)
    x = torch.randn(EVAL_SAMPLES, INPUT_DIM, generator=gen)
    with torch.no_grad():
        return x, teacher(x)


def lr_lambda_factory(total_steps: int, warmup: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


@dataclass(slots=True)
class Config:
    rank: int
    world_size: int
    job_name: str
    run_id: str
    attempt: int
    layout: Layout
    placement: list[str]
    progress_file: str
    run_dir: Path
    total_steps: int
    ckpt_interval: int
    eval_interval: int
    step_delay: float
    lr: float
    warmup: int
    resume_step: int | None
    plan_path: str
    events_file: str
    replication: ReplicationPolicy
    faults: list[Fault]

    @property
    def node(self) -> str:
        return self.placement[self.rank]

    def fault(self, *kinds: str, step: int | None = None) -> Fault | None:
        return next(
            (f for f in self.faults if f.kind in kinds and (step is None or f.step == step)),
            None,
        )


def load_config() -> Config:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    attempt = int(os.environ.get("AEGIS_ATTEMPT", "0"))
    placement_raw = os.environ.get("AEGIS_PLACEMENT", "").strip()
    placement = placement_raw.split(",") if placement_raw else [f"node-{r}" for r in range(world_size)]
    resume_raw = os.environ.get("AEGIS_RESUME_STEP", "").strip()
    faults = []
    for fault in Fault.parse_all(os.environ.get("AEGIS_FAULT", ""), os.environ.get("AEGIS_FAULT_ATTEMPTS", "0")):
        if fault.active_on(attempt):
            faults.append(fault)
        elif rank == 0:
            print(f"[rank {rank}] fault {fault.kind} not armed for attempt {attempt}", flush=True)
    progress_file = os.environ.get("AEGIS_PROGRESS_FILE", "./progress.json")
    return Config(
        rank=rank,
        world_size=world_size,
        job_name=os.environ.get("AEGIS_JOB_NAME", "train"),
        run_id=os.environ.get("AEGIS_RUN_ID", "local"),
        attempt=attempt,
        layout=Layout(os.environ.get("AEGIS_CHECKPOINT_DIR", "./checkpoints")),
        placement=placement,
        progress_file=progress_file,
        run_dir=Path(os.environ.get("AEGIS_RUN_DIR", str(Path(progress_file).parent))),
        total_steps=int(os.environ.get("AEGIS_TOTAL_STEPS", "500")),
        ckpt_interval=int(os.environ.get("AEGIS_CHECKPOINT_INTERVAL", "10")),
        eval_interval=int(os.environ.get("AEGIS_EVAL_INTERVAL", "25")),
        step_delay=float(os.environ.get("AEGIS_STEP_DELAY", "0.05")),
        lr=float(os.environ.get("AEGIS_LR", "3e-3")),
        warmup=int(os.environ.get("AEGIS_WARMUP_STEPS", "20")),
        resume_step=int(resume_raw) if resume_raw else None,
        plan_path=os.environ.get("AEGIS_RESTORE_PLAN", ""),
        events_file=os.environ.get("AEGIS_EVENTS_FILE", ""),
        replication=ReplicationPolicy(
            peer=os.environ.get("AEGIS_PEER_REPLICATION", "1") != "0",
            durable_interval_steps=int(os.environ.get("AEGIS_DURABLE_INTERVAL", "20")),
        ),
        faults=faults,
    )


def main() -> int:
    cfg = load_config()
    if len(cfg.placement) != cfg.world_size:
        print(f"placement {cfg.placement} does not match world size {cfg.world_size}", flush=True)
        return 2
    dist.init_process_group(backend="gloo", world_size=cfg.world_size, rank=cfg.rank)
    metrics = MetricsLog(os.environ.get("AEGIS_METRICS_FILE") or None)
    try:
        return Trainer(cfg, metrics).run()
    finally:
        metrics.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class Trainer:
    def __init__(self, cfg: Config, metrics: MetricsLog) -> None:
        self.cfg = cfg
        self.metrics = metrics
        self.profiles = get_profiles()
        self.collective = Collective.from_torch()

        torch.manual_seed(SEED)
        self.teacher = make_teacher()
        self.model = ToyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda_factory(cfg.total_steps, cfg.warmup)
        )
        # Per-rank RNG stream; restored exactly on resume.
        torch.manual_seed(SEED + 1 + cfg.rank)
        self.criterion = nn.MSELoss()
        self.step = 0
        self.samples_consumed = 0

    # ------------------------------------------------------------------ run

    def run(self) -> int:
        cfg = self.cfg
        started = time.time()
        if cfg.rank == 0:
            self.metrics.emit(
                "start", run_id=cfg.run_id, attempt=cfg.attempt, resume_step=cfg.resume_step,
                placement=cfg.placement,
            )

        if cfg.resume_step is not None:
            if not self._restore():
                return 1
        else:
            self._write_report(RestoreReport(cfg.run_id, cfg.attempt, None, outcome="fresh"))
            print(f"[rank {cfg.rank}] starting from scratch on {cfg.node}", flush=True)

        ddp = DistributedDataParallel(self.model)
        for param in ddp.module.parameters():
            dist.broadcast(param.data, src=0)

        writer = TieredCheckpointWriter(
            cfg.layout,
            self.collective,
            placement=cfg.placement,
            job_name=cfg.job_name,
            run_id=cfg.run_id,
            attempt=cfg.attempt,
            replication=cfg.replication,
            profiles=self.profiles,
            metrics=self.metrics,
        )
        metadata = CheckpointMetadata(
            job_name=cfg.job_name,
            world_size=cfg.world_size,
            model_signature=model_signature(self.model.state_dict()),
            optimizer_class=type(self.optimizer).__name__,
            torch_version=torch.__version__,
            scheduler_class=type(self.scheduler).__name__,
            dataset_fingerprint=f"teacher{TEACHER_SEED}-{INPUT_DIM}x{OUTPUT_DIM}-seed{SEED}",
        )
        eval_x, eval_y = make_eval_set(self.teacher)
        ema: float | None = None

        if cfg.rank == 0:
            self.metrics.emit("resumed", step=self.step, startup_s=time.time() - started)

        for step in range(self.step + 1, cfg.total_steps + 1):
            t0 = time.monotonic()
            x, y = make_batch(self.teacher, self.samples_consumed, cfg.rank)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(ddp(x), y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.samples_consumed += BATCH_SIZE
            self.step = step

            if cfg.step_delay:
                time.sleep(cfg.step_delay)

            if step % cfg.ckpt_interval == 0:
                self._save(writer, metadata)

            if cfg.rank == 0:
                dt = time.monotonic() - t0
                ema = dt if ema is None else 0.8 * ema + 0.2 * dt
                lr = self.scheduler.get_last_lr()[0]
                self.metrics.emit("step", step=step, loss=float(loss.detach()), lr=lr, attempt=cfg.attempt)
                if step % cfg.eval_interval == 0 or step == cfg.total_steps:
                    self._evaluate(eval_x, eval_y, step)
                write_progress(
                    cfg.progress_file, step=step, run_id=cfg.run_id, attempt=cfg.attempt,
                    loss=float(loss.detach()), step_seconds=ema,
                )

            for fault in cfg.faults:
                self._maybe_fault(fault, step)

        dist.barrier()
        writer.close(timeout=15)
        if cfg.rank == 0:
            digests = global_digests(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                step=self.step,
            )
            self.metrics.emit("final", step=self.step, digests=digests, attempt=cfg.attempt)
            write_progress(
                cfg.progress_file, step=cfg.total_steps, run_id=cfg.run_id, attempt=cfg.attempt, finished=True
            )
            print(f"[rank 0] training complete at step {cfg.total_steps}", flush=True)
        return 0

    def _evaluate(self, x: torch.Tensor, y: torch.Tensor, step: int) -> None:
        with torch.no_grad():
            out = self.model(x)
            loss = float(self.criterion(out, y))
            agreement = float((out.argmax(1) == y.argmax(1)).float().mean())
        self.metrics.emit("eval", step=step, eval_loss=loss, agreement=agreement, attempt=self.cfg.attempt)

    # ----------------------------------------------------------- checkpoint

    def _state_payload(self) -> dict:
        return {
            "step": self.step,
            "samples_consumed": self.samples_consumed,
            "rng_state": torch.get_rng_state(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def _digests(self) -> tuple[dict[str, str], dict[str, str]]:
        g = global_digests(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            step=self.step,
        )
        r = rank_digests(rng_state=torch.get_rng_state(), samples_consumed=self.samples_consumed)
        return g, r

    def _save(self, writer: TieredCheckpointWriter, metadata: CheckpointMetadata) -> None:
        cfg = self.cfg
        skip_manifest = cfg.fault("incomplete_checkpoint", step=self.step) is not None
        fault = cfg.fault("corrupt_shard", "corrupt_everywhere", step=self.step)
        hit = fault is not None
        g, r = self._digests()
        writer.save(
            step=self.step,
            payload=self._state_payload(),
            metadata=metadata,
            global_digests=g,
            rank_digests=r,
            samples_consumed=self.samples_consumed,
            skip_manifest=skip_manifest,
        )
        if skip_manifest:
            print(f"[rank {cfg.rank}] FAULT: dying mid-checkpoint at step {self.step}", flush=True)
            os._exit(1)
        if hit and fault.targets(cfg.rank, cfg.world_size):
            path = writer.corrupt_local_shard(self.step)
            print(f"[rank {cfg.rank}] FAULT: corrupted {path}", flush=True)
            if fault.kind == "corrupt_everywhere":
                writer.replicator.close(timeout=60)
                for location in cfg.layout.locations():
                    replica = cfg.layout.dir_for(location, self.step) / shard_name(cfg.rank)
                    if replica.is_file() and replica != path:
                        _flip(replica)
                        print(f"[rank {cfg.rank}] FAULT: corrupted {replica}", flush=True)
        if hit:
            dist.barrier()
            os._exit(1)

    # -------------------------------------------------------------- restore

    def _restore(self) -> bool:
        """Stage, load and verify. Every rank reaches the gather below, even
        when it failed, so one bad rank cannot leave the others hanging."""
        cfg = self.cfg
        report = RankRestoreReport(rank=cfg.rank, node=cfg.node)
        failure = ""
        replicated = ""
        t_stage = time.monotonic()
        stage_s = 0.0
        try:
            plan = self._plan()
            directory, staged = stage_rank(cfg.layout, plan, cfg.rank, self.profiles)
            report.staged = asdict(staged)
            stage_s = time.monotonic() - t_stage
            replicated = self._load_and_verify(directory, report)
        except StagingError as exc:
            failure = "staging_failed"
            report.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - anything here means "cannot restore"
            failure = "verification_failed"
            report.error = f"{type(exc).__name__}: {exc}"
        if not failure and not report.ok:
            failure = "verification_failed"

        gathered = self.collective.all_gather((asdict(report), replicated, failure, stage_s))
        ranks = [RankRestoreReport(**g[0]) for g in gathered]
        digests = {g[1] for g in gathered}
        failures = [g[2] for g in gathered if g[2]]
        consistent = len(digests) == 1 and "" not in digests
        outcome = (
            "staging_failed" if "staging_failed" in failures
            else "verification_failed" if failures or not consistent
            else "restored"
        )
        full = RestoreReport(
            run_id=cfg.run_id,
            attempt=cfg.attempt,
            step=cfg.resume_step,
            ranks=ranks,
            consistent_across_ranks=consistent,
            outcome=outcome,
            stage_seconds=max(g[3] for g in gathered),
            restore_seconds=max(r.load_seconds for r in ranks),
        )
        self._write_report(full)
        if cfg.rank == 0:
            self.metrics.emit(
                "restore", step=cfg.resume_step, outcome=outcome, attempt=cfg.attempt,
                stage_s=full.stage_seconds, load_s=full.restore_seconds,
                sources=[r.staged["source"] if r.staged else None for r in ranks],
                failed=full.failed_checks(),
            )
        if outcome != "restored":
            print(f"[rank {cfg.rank}] restore {outcome}: {full.failed_checks()}", flush=True)
            return False
        print(
            f"[rank {cfg.rank}] restored step {self.step} on {cfg.node} from "
            f"{report.staged['source'] if report.staged else '?'} (samples_consumed={self.samples_consumed})",
            flush=True,
        )
        return True

    def _plan(self) -> RestorePlan:
        cfg = self.cfg
        inline = os.environ.get("AEGIS_RESTORE_PLAN_JSON", "").strip()
        if inline or (cfg.plan_path and Path(cfg.plan_path).is_file()):
            plan = RestorePlan.from_json(inline) if inline else RestorePlan.load(Path(cfg.plan_path))
            if plan.step != cfg.resume_step:
                raise RuntimeError(f"plan is for step {plan.step}, asked to resume {cfg.resume_step}")
            if plan.placement != cfg.placement:
                raise RuntimeError(f"plan placement {plan.placement} != launch placement {cfg.placement}")
            return plan
        return self._self_plan()

    def _self_plan(self) -> RestorePlan:
        """No controller plan (manual resume): take any intact replica,
        preferring this node's own disks."""
        cfg = self.cfg
        view = next(
            (v for v in CheckpointCatalog(cfg.layout).survey(max_candidates=50) if v.step == cfg.resume_step),
            None,
        )
        if view is None or not view.recoverable:
            raise StagingError(f"step {cfg.resume_step} is not recoverable: {view.summary() if view else 'absent'}")
        sources = []
        for rank in range(cfg.world_size):
            node = cfg.placement[rank]
            best = min(
                view.replicas[rank],
                key=lambda r: (r.location.node != node, r.location.tier is Tier.DURABLE),
            )
            sources.append(ShardSource(rank, str(best.location), str(best.path), best.size_bytes))
        return RestorePlan(cfg.resume_step, cfg.placement, sources, view.manifest.to_json())

    def _load_and_verify(self, directory: Path, report: RankRestoreReport) -> str:
        cfg = self.cfg
        manifest = CheckpointManifest.load(directory)
        meta = manifest.metadata
        if meta.world_size != cfg.world_size:
            raise RuntimeError(f"checkpoint world size {meta.world_size} != {cfg.world_size}")
        if meta.model_signature != model_signature(self.model.state_dict()):
            raise RuntimeError("checkpoint model structure differs from this job")

        t0 = time.monotonic()
        payload = torch.load(directory / shard_name(cfg.rank), map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["rng_state"].to(torch.uint8))
        self.step = int(payload["step"])
        self.samples_consumed = int(payload["samples_consumed"])
        if self.profiles.throttle:
            # Deserialization of this toy model is instant; the storage profile
            # models what a production-sized shard costs, as it does for copies.
            modelled = self.profiles.restore_seconds((directory / shard_name(cfg.rank)).stat().st_size)
            time.sleep(max(0.0, modelled - (time.monotonic() - t0)))
        report.load_seconds = time.monotonic() - t0

        skew = cfg.fault("skew_restore")
        if skew is not None and skew.targets(cfg.rank, cfg.world_size):
            self.samples_consumed += BATCH_SIZE
            print(f"[rank {cfg.rank}] FAULT: skewed restored data position", flush=True)

        actual_global, actual_rank = self._digests()
        expected_rank = manifest.rank_state_for(cfg.rank)
        checks = {f"global.{k}": v for k, v in compare(manifest.state_digests, actual_global).items()}
        checks.update(
            {
                f"rank.{k}": v
                for k, v in compare(
                    {"rng": expected_rank.rng, "data_position": expected_rank.data_position}
                    if expected_rank else {},
                    actual_rank,
                ).items()
            }
        )
        checks["step"] = self.step == manifest.step == cfg.resume_step
        report.checks = checks
        return "|".join(f"{k}={actual_global[k]}" for k in sorted(actual_global))

    def _write_report(self, report: RestoreReport) -> None:
        if self.cfg.rank == 0:
            self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
            report.write(self.cfg.run_dir / REPORT_NAME)

    # ---------------------------------------------------------------- faults

    def _maybe_fault(self, fault: Fault, step: int) -> None:
        cfg = self.cfg
        if fault.step != step:
            return
        if fault.kind == "node_loss":
            if fault.targets(cfg.rank, cfg.world_size):
                cfg.layout.lose_node(cfg.node)
                if cfg.events_file:
                    emit_event(
                        cfg.events_file,
                        kind=SignalKind.NODE_UNHEALTHY,
                        source="fault-injector",
                        node=cfg.node,
                        detail={"reason": "node lost with its local disks", "step": step},
                    )
                print(f"[rank {cfg.rank}] FAULT: node {cfg.node} lost at step {step}", flush=True)
                os._exit(1)
            return
        if not fault.applies_to(cfg.rank, step):
            return
        if fault.kind == "crash":
            print(f"[rank {cfg.rank}] FAULT: hard exit at step {step}", flush=True)
            os._exit(1)
        if fault.kind == "hang":
            print(f"[rank {cfg.rank}] FAULT: hanging at step {step}", flush=True)
            while True:
                time.sleep(3600)


def _flip(path: Path) -> None:
    data = bytearray(path.read_bytes())
    for offset in range(len(data) // 2, min(len(data), len(data) // 2 + 64)):
        data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


if __name__ == "__main__":
    raise SystemExit(main())
