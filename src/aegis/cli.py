"""Aegis command line."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from aegis import metrics
from aegis.backends import get_backend
from aegis.checkpoint import CheckpointValidator, LocalCheckpointStore
from aegis.checkpoint.validator import Compatibility
from aegis.config import Settings, load_settings
from aegis.controller.loop import ControlLoop
from aegis.controller.policy import POLICY_REGISTRY, get_policy
from aegis.controller.reconciler import Reconciler, ReconcilerConfig, TickResult
from aegis.events.synthetic import FileEventSource, emit_event
from aegis.models import JobSpec, JobState, SignalKind
from aegis.store.repository import Repository

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Recovery controller for distributed PyTorch training.",
)
console = Console()

WorkdirOpt = Annotated[Optional[Path], typer.Option("--workdir", help="Aegis state directory")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("aegis").setLevel(logging.DEBUG if verbose else logging.INFO)


def _context(workdir: Path | None, backend_name: str | None = None) -> tuple[Settings, Repository, Reconciler]:
    settings = load_settings(workdir=workdir, backend=backend_name)
    repo = Repository.from_url(settings.resolved_database_url())
    backend = _build_backend(settings)
    reconciler = Reconciler(
        repo,
        backend,
        lambda spec: LocalCheckpointStore(spec.checkpoint_dir),
        event_source=FileEventSource(settings.events_path),
        config=ReconcilerConfig(),
    )
    return settings, repo, reconciler


def _build_backend(settings: Settings):
    if settings.backend == "local":
        return get_backend("local")
    return get_backend(
        settings.backend,
        namespace=settings.namespace,
        image=settings.image,
        kubeconfig=settings.kubeconfig or None,
    )


# --------------------------------------------------------------------- submit


@app.command()
def submit(
    name: Annotated[str, typer.Option("--name", help="Job name")] = "demo",
    world_size: Annotated[int, typer.Option("--world-size")] = 2,
    total_steps: Annotated[int, typer.Option("--total-steps")] = 400,
    policy: Annotated[str, typer.Option("--policy", help=f"One of: {', '.join(POLICY_REGISTRY)}")] = "baseline",
    fault: Annotated[str, typer.Option("--fault", help="kind:step[:rank] passed to the training script")] = "",
    fault_attempts: Annotated[str, typer.Option("--fault-attempts", help="Attempts the fault fires on; '*' makes it permanent")] = "0",
    checkpoint_interval: Annotated[int, typer.Option("--checkpoint-interval")] = 10,
    step_delay: Annotated[float, typer.Option("--step-delay", help="Artificial per-step delay, seconds")] = 0.05,
    workdir: WorkdirOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Register a training job with the controller."""
    _setup_logging(verbose)
    settings, repo, _ = _context(workdir)

    spec = _build_spec(
        settings,
        name=name,
        world_size=world_size,
        total_steps=total_steps,
        policy=policy,
        fault=fault,
        fault_attempts=fault_attempts,
        checkpoint_interval=checkpoint_interval,
        step_delay=step_delay,
    )
    repo.register_job(spec)
    console.print(f"[green]registered[/green] job [bold]{spec.job_id}[/bold] (policy {spec.policy.policy_id})")
    console.print(f"  checkpoints: {spec.checkpoint_dir}")
    console.print(f"  run `aegis controller --workdir {settings.workdir}` to start reconciling")


def _build_spec(
    settings: Settings,
    *,
    name: str,
    world_size: int,
    total_steps: int,
    policy: str,
    fault: str,
    fault_attempts: str,
    checkpoint_interval: int,
    step_delay: float,
) -> JobSpec:
    job_id = JobSpec.new_id(name)
    env = {
        "AEGIS_CHECKPOINT_INTERVAL": str(checkpoint_interval),
        "AEGIS_STEP_DELAY": str(step_delay),
    }
    if fault:
        env["AEGIS_FAULT"] = fault
        env["AEGIS_FAULT_ATTEMPTS"] = fault_attempts
    return JobSpec(
        job_id=job_id,
        name=name,
        world_size=world_size,
        checkpoint_dir=str(settings.checkpoint_dir(name).resolve()),
        workdir=str(Path.cwd()),
        command=[],
        env=env,
        total_steps=total_steps,
        namespace=settings.namespace,
        image=settings.image,
        policy=get_policy(policy),
    )


# ----------------------------------------------------------------- controller


@app.command()
def controller(
    job: Annotated[Optional[str], typer.Option("--job", help="Reconcile only this job id")] = None,
    interval: Annotated[float, typer.Option("--interval")] = 2.0,
    until_terminal: Annotated[bool, typer.Option("--until-terminal", help="Exit once all jobs reach a terminal state")] = False,
    metrics_port: Annotated[int, typer.Option("--metrics-port", help="0 disables the exporter")] = 0,
    workdir: WorkdirOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the reconcile loop."""
    _setup_logging(verbose)
    settings, repo, reconciler = _context(workdir)
    if metrics_port:
        metrics.serve(metrics_port)
        console.print(f"metrics on :{metrics_port}/metrics")

    loop = ControlLoop(repo, reconciler, interval_seconds=interval, observer=_print_tick)
    loop.install_signal_handlers()
    console.print("[bold]aegis controller[/bold] started; Ctrl-C to stop")
    stats = loop.run(job_ids=[job] if job else None, until_terminal=until_terminal)
    console.print(f"stopped after {stats.ticks} ticks, {stats.transitions} transitions")


_last_note: dict[str, str] = {}


def _print_tick(result: TickResult) -> None:
    key = result.job_id
    line = f"{result.state}:{result.note}"
    if result.changed:
        console.print(f"[bold cyan]{result.job_id}[/bold cyan] -> [bold]{result.state}[/bold] {result.note}")
    elif _last_note.get(key) != line:
        console.print(f"[dim]{result.job_id} {result.state} {result.note}[/dim]")
    _last_note[key] = line


# --------------------------------------------------------------------- inject


@app.command()
def inject(
    kind: Annotated[str, typer.Argument(help=f"One of: {', '.join(k.value for k in SignalKind)}")],
    job: Annotated[Optional[str], typer.Option("--job")] = None,
    node: Annotated[Optional[str], typer.Option("--node")] = None,
    detail: Annotated[str, typer.Option("--detail", help="JSON object")] = "{}",
    workdir: WorkdirOpt = None,
) -> None:
    """Inject a synthetic health event, standing in for NVSentinel."""
    settings = load_settings(workdir=workdir)
    event = emit_event(
        settings.events_path,
        kind=SignalKind(kind),
        job_id=job,
        node=node,
        detail=json.loads(detail),
    )
    console.print(f"[yellow]injected[/yellow] {event.kind} node={event.node} job={event.job_id or 'all'}")


# --------------------------------------------------------------------- status


@app.command()
def status(
    workdir: WorkdirOpt = None,
    all_jobs: Annotated[bool, typer.Option("--all", help="Include finished jobs")] = False,
) -> None:
    """Show job states."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    rows = repo.list_jobs(active_only=not all_jobs)
    if not rows:
        console.print("[dim]no jobs[/dim]")
        return

    table = Table("job", "state", "attempt", "step", "policy", "message")
    for row in rows:
        colour = {
            JobState.RUNNING: "green",
            JobState.SUCCEEDED: "green",
            JobState.SUSPECT: "yellow",
            JobState.RECOVERING: "yellow",
            JobState.VERIFYING: "cyan",
            JobState.EXHAUSTED: "red",
        }.get(JobState(row.state), "white")
        table.add_row(
            row.job_id,
            f"[{colour}]{row.state}[/{colour}]",
            str(row.attempt),
            str(row.last_observed_step),
            row.policy_id,
            (row.message or "")[:60],
        )
    console.print(table)


@app.command()
def incidents(
    job: Annotated[Optional[str], typer.Option("--job")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    workdir: WorkdirOpt = None,
) -> None:
    """Show recorded incidents, the input to policy ranking."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    rows = repo.list_incidents(job, limit)
    if not rows:
        console.print("[dim]no incidents[/dim]")
        return

    table = Table("job", "attempt", "signals", "chosen", "lost", "seconds", "outcome", "detail")
    for row in rows:
        kinds = sorted({s.get("kind", "?") for s in (row.signals or [])})
        rejected = len(row.rejected_checkpoints or [])
        chosen = str(row.chosen_step) if row.chosen_step is not None else "-"
        if rejected:
            chosen += f" (+{rejected} rejected)"
        table.add_row(
            row.job_id,
            str(row.attempt),
            ",".join(kinds) or "-",
            chosen,
            str(row.lost_steps if row.lost_steps is not None else "-"),
            f"{row.recovery_seconds:.1f}" if row.recovery_seconds else "-",
            row.outcome,
            (row.detail or "")[:40],
        )
    console.print(table)


# ------------------------------------------------------------------ validate


@app.command()
def validate(
    checkpoint_dir: Annotated[Path, typer.Argument(help="Directory holding step_* checkpoints")],
    job_name: Annotated[Optional[str], typer.Option("--job-name")] = None,
    world_size: Annotated[Optional[int], typer.Option("--world-size")] = None,
) -> None:
    """Validate every checkpoint in a directory and show which one Aegis would pick."""
    store = LocalCheckpointStore(checkpoint_dir)
    validator = CheckpointValidator(store)
    compat = Compatibility(job_name=job_name, world_size=world_size)

    steps = store.list_steps()
    if not steps:
        console.print(f"[dim]no checkpoints in {checkpoint_dir}[/dim]")
        return

    table = Table("step", "verdict", "reasons", "detail")
    chosen = None
    for step in steps:
        result = validator.validate(step, compat)
        if result.valid and chosen is None:
            chosen = step
        table.add_row(
            str(step),
            "[green]valid[/green]" if result.valid else "[red]rejected[/red]",
            ", ".join(str(r) for r in result.reasons) or "-",
            result.detail[:60] or "-",
        )
    console.print(table)
    if chosen is None:
        console.print("[red]no valid checkpoint; Aegis would escalate rather than restart from scratch[/red]")
    else:
        console.print(f"[green]would resume from step {chosen}[/green]")


SCENARIOS: dict[str, dict] = {
    "worker-loss": {
        "fault": "crash:45:1",
        "fault_attempts": "0",
        "blurb": "rank 1 dies at step 45; Aegis resumes from the last complete checkpoint",
    },
    "incomplete-checkpoint": {
        "fault": "incomplete_checkpoint:50",
        "fault_attempts": "0",
        "blurb": "the job dies mid-checkpoint at step 50; Aegis rejects it and falls back to step 40",
    },
    "missing-shard": {
        "fault": "missing_shard:50",
        "fault_attempts": "0",
        "blurb": "a shard never lands at step 50; the manifest no longer matches and the checkpoint is rejected",
    },
    "stalled-worker": {
        "fault": "hang:45:1",
        "fault_attempts": "0",
        "blurb": "rank 1 hangs at step 45; nothing crashes, so only the progress watcher can see it",
    },
    "repeated-fault": {
        "fault": "crash:45:1",
        "fault_attempts": "*",
        "blurb": "the fault returns every attempt; Aegis must stop at the budget rather than loop forever",
    },
}


@app.command()
def demo(
    scenario: Annotated[str, typer.Argument(help=f"One of: {', '.join(SCENARIOS)}")] = "worker-loss",
    world_size: Annotated[int, typer.Option("--world-size")] = 2,
    total_steps: Annotated[int, typer.Option("--total-steps")] = 220,
    step_delay: Annotated[float, typer.Option("--step-delay")] = 0.15,
    workdir: WorkdirOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Submit a job with an injected fault and reconcile it to completion."""
    if scenario not in SCENARIOS:
        raise typer.BadParameter(f"unknown scenario; choose from {', '.join(SCENARIOS)}")
    config = SCENARIOS[scenario]

    _setup_logging(verbose)
    settings, repo, reconciler = _context(workdir)

    console.rule(f"[bold]{scenario}[/bold]")
    console.print(config["blurb"])
    console.print()

    spec = _build_spec(
        settings,
        name=scenario,
        world_size=world_size,
        total_steps=total_steps,
        policy="demo",
        fault=config["fault"],
        fault_attempts=config["fault_attempts"],
        checkpoint_interval=10,
        step_delay=step_delay,
    )
    repo.register_job(spec)

    loop = ControlLoop(repo, reconciler, interval_seconds=2.0, observer=_print_tick)
    loop.install_signal_handlers()
    loop.run(job_ids=[spec.job_id], until_terminal=True)

    console.print()
    row = repo.get_job(spec.job_id)
    outcome = "green" if JobState(row.state) is JobState.SUCCEEDED else "red"
    console.print(f"final state: [{outcome}]{row.state}[/{outcome}] - {row.message}")
    incidents(job=spec.job_id, limit=10, workdir=settings.workdir)


if __name__ == "__main__":
    app()
