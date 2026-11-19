"""Aegis command line."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from aegis import metrics
from aegis.backends import get_backend
from aegis.checkpoint import CheckpointCatalog, Compatibility, Layout, Tier
from aegis.checkpoint.tiers import read_conditions
from aegis.config import Settings, load_settings
from aegis.controller.loop import ControlLoop
from aegis.controller.planner import STRATEGIES
from aegis.controller.policy import POLICY_REGISTRY, get_policy
from aegis.controller.reconciler import Reconciler, ReconcilerConfig, TickResult
from aegis.events.synthetic import FileEventSource, emit_event
from aegis.faults import degrade_tier, heal_node, lose_node, restore_tiers
from aegis.models import JobSpec, JobState, SignalKind
from aegis.store.repository import Repository

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Recovery controller for distributed PyTorch training.")
node_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Node health and node-loss injection.")
tier_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Storage tier degradation.")
app.add_typer(node_app, name="node")
app.add_typer(tier_app, name="tier")
console = Console()

WorkdirOpt = Annotated[Optional[Path], typer.Option("--workdir", help="Aegis state directory")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("aegis").setLevel(logging.DEBUG if verbose else logging.INFO)


def build_reconciler(settings: Settings, repo: Repository, **config) -> Reconciler:
    return Reconciler(
        repo,
        _build_backend(settings),
        event_source=FileEventSource(settings.events_path),
        config=ReconcilerConfig(node_pool=settings.node_pool, storage_profile=settings.storage_profile, **config),
    )


def _context(workdir: Path | None, backend_name: str | None = None) -> tuple[Settings, Repository, Reconciler]:
    settings = load_settings(workdir=workdir, backend=backend_name)
    repo = Repository.from_url(settings.resolved_database_url())
    return settings, repo, build_reconciler(settings, repo)


def _build_backend(settings: Settings):
    if settings.backend == "local":
        return get_backend("local")
    return get_backend(
        settings.backend, namespace=settings.namespace, image=settings.image, kubeconfig=settings.kubeconfig or None
    )


def _job_spec(repo: Repository, job_id: str) -> JobSpec:
    row = repo.get_job(job_id)
    if row is None:
        raise typer.BadParameter(f"no job {job_id}")
    return repo.spec_of(row)


# --------------------------------------------------------------------- submit


@app.command()
def submit(
    name: Annotated[str, typer.Option("--name", help="Job name")] = "demo",
    world_size: Annotated[int, typer.Option("--world-size")] = 2,
    total_steps: Annotated[int, typer.Option("--total-steps")] = 400,
    policy: Annotated[str, typer.Option("--policy", help=f"One of: {', '.join(POLICY_REGISTRY)}")] = "baseline",
    planner: Annotated[str, typer.Option("--planner", help=f"One of: {', '.join(STRATEGIES)}")] = "cost",
    fault: Annotated[str, typer.Option("--fault", help="kind:step[:rank][@attempts], ';'-separated")] = "",
    fault_attempts: Annotated[str, typer.Option("--fault-attempts", help="Default attempts a fault fires on; '*' = permanent")] = "0",
    checkpoint_interval: Annotated[int, typer.Option("--checkpoint-interval")] = 10,
    durable_interval: Annotated[int, typer.Option("--durable-interval", help="Upload every Nth step to durable storage; 0 disables")] = 20,
    peer_replication: Annotated[bool, typer.Option("--peer/--no-peer", help="Replicate each shard to a neighbouring node")] = True,
    step_delay: Annotated[float, typer.Option("--step-delay", help="Artificial per-step delay, seconds")] = 0.05,
    workdir: WorkdirOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Register a training job with the controller."""
    _setup_logging(verbose)
    settings, repo, _ = _context(workdir)
    spec = build_spec(
        settings, name=name, world_size=world_size, total_steps=total_steps, policy=policy, planner=planner,
        fault=fault, fault_attempts=fault_attempts, checkpoint_interval=checkpoint_interval,
        durable_interval=durable_interval, peer_replication=peer_replication, step_delay=step_delay,
    )
    repo.register_job(spec)
    console.print(f"[green]registered[/green] job [bold]{spec.job_id}[/bold] (policy {spec.policy.policy_id})")
    console.print(f"  checkpoints: {spec.checkpoint_dir}")
    console.print(f"  run `aegis controller --workdir {settings.workdir}` to start reconciling")


def build_spec(
    settings: Settings,
    *,
    name: str,
    world_size: int,
    total_steps: int,
    policy: str,
    planner: str = "cost",
    fault: str = "",
    fault_attempts: str = "0",
    checkpoint_interval: int = 10,
    durable_interval: int = 20,
    peer_replication: bool = True,
    step_delay: float = 0.05,
    extra_env: dict[str, str] | None = None,
) -> JobSpec:
    if planner not in STRATEGIES:
        raise typer.BadParameter(f"unknown planner {planner!r}; choose from {', '.join(STRATEGIES)}")
    job_id = JobSpec.new_id(name)
    env = {
        "AEGIS_CHECKPOINT_INTERVAL": str(checkpoint_interval),
        "AEGIS_DURABLE_INTERVAL": str(durable_interval),
        "AEGIS_PEER_REPLICATION": "1" if peer_replication else "0",
        "AEGIS_STEP_DELAY": str(step_delay),
        "AEGIS_STORAGE_PROFILE": settings.storage_profile,
        "AEGIS_EVENTS_FILE": str(settings.events_path.resolve()),
        **(extra_env or {}),
    }
    if fault:
        env["AEGIS_FAULT"] = fault
        env["AEGIS_FAULT_ATTEMPTS"] = fault_attempts
    base = get_policy(policy)
    chosen = base if planner == "cost" else replace(base, name=f"{base.name}-{planner}", planner=planner)
    return JobSpec(
        job_id=job_id,
        name=name,
        world_size=world_size,
        checkpoint_dir=str(settings.checkpoint_dir(job_id).resolve()),
        workdir=str(Path.cwd()),
        command=[],
        env=env,
        total_steps=total_steps,
        namespace=settings.namespace,
        image=settings.image,
        policy=chosen,
    )


# ----------------------------------------------------------------- controller


@app.command()
def controller(
    job: Annotated[Optional[str], typer.Option("--job", help="Reconcile only this job id")] = None,
    interval: Annotated[float, typer.Option("--interval")] = 2.0,
    until_terminal: Annotated[bool, typer.Option("--until-terminal", help="Exit once all jobs reach a terminal state")] = False,
    lease_ttl: Annotated[float, typer.Option("--lease-ttl", help="Recovery lease TTL, seconds")] = 60.0,
    metrics_port: Annotated[int, typer.Option("--metrics-port", help="0 disables the exporter")] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", help="Only print state transitions")] = False,
    workdir: WorkdirOpt = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the reconcile loop."""
    _setup_logging(verbose)
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    reconciler = build_reconciler(settings, repo, lease_ttl_seconds=lease_ttl)
    if metrics_port:
        metrics.serve(metrics_port)
        console.print(f"metrics on :{metrics_port}/metrics")

    loop = ControlLoop(repo, reconciler, interval_seconds=interval, observer=_print_transitions if quiet else _print_tick)
    loop.install_signal_handlers()
    console.print(f"[bold]aegis controller[/bold] {reconciler.config.holder_id} started; Ctrl-C to stop")
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


def _print_transitions(result: TickResult) -> None:
    if result.changed:
        console.print(f"[bold cyan]{result.job_id}[/bold cyan] -> [bold]{result.state}[/bold] {result.note}")


# --------------------------------------------------------------------- inject


@app.command()
def inject(
    kind: Annotated[str, typer.Argument(help=f"One of: {', '.join(k.value for k in SignalKind)}")],
    job: Annotated[Optional[str], typer.Option("--job")] = None,
    node: Annotated[Optional[str], typer.Option("--node")] = None,
    detail: Annotated[str, typer.Option("--detail", help="JSON object")] = "{}",
    workdir: WorkdirOpt = None,
) -> None:
    """Inject a raw synthetic health event, standing in for NVSentinel."""
    settings = load_settings(workdir=workdir)
    event = emit_event(settings.events_path, kind=SignalKind(kind), job_id=job, node=node, detail=json.loads(detail))
    console.print(f"[yellow]injected[/yellow] {event.kind} node={event.node} job={event.job_id or 'all'}")


@node_app.command("list")
def node_list(workdir: WorkdirOpt = None) -> None:
    """Node health as the controller has recorded it."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    recorded = {n.name: n for n in repo.list_nodes()}
    table = Table("node", "health", "reason", "since")
    for name in sorted(set(settings.node_pool) | set(recorded)):
        row = recorded.get(name)
        healthy = row is None or bool(row.healthy)
        table.add_row(
            name,
            "[green]healthy[/green]" if healthy else "[red]unhealthy[/red]",
            row.reason if row and not healthy else "-",
            row.changed_at.strftime("%H:%M:%S") if row else "-",
        )
    console.print(table)


@node_app.command("lose")
def node_lose(
    node: Annotated[str, typer.Argument()],
    job: Annotated[str, typer.Option("--job", help="Job whose checkpoint storage lives on the node")],
    workdir: WorkdirOpt = None,
) -> None:
    """Fault: the node dies with its local disks, and its workers with it."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    spec = _job_spec(repo, job)
    moved = lose_node(repo, _build_backend(settings), spec, node, events_path=settings.events_path)
    console.print(f"[red]lost[/red] {node}" + (f" (disks moved to {moved.name})" if moved else " (it held no data)"))


@node_app.command("heal")
def node_heal(node: Annotated[str, typer.Argument()], workdir: WorkdirOpt = None) -> None:
    """Report a node healthy again so it is eligible for placement."""
    settings = load_settings(workdir=workdir)
    heal_node(node, events_path=settings.events_path)
    console.print(f"[green]reported healthy[/green] {node}")


@tier_app.command("degrade")
def tier_degrade(
    tier: Annotated[str, typer.Argument(help="local | peer | durable")],
    job: Annotated[str, typer.Option("--job")],
    factor: Annotated[Optional[float], typer.Option("--factor", help="Bandwidth multiplier, e.g. 0.1")] = None,
    unavailable: Annotated[bool, typer.Option("--unavailable", help="Take the tier offline entirely")] = False,
    workdir: WorkdirOpt = None,
) -> None:
    """Fault: slow a storage tier down or take it offline."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    condition = degrade_tier(
        _job_spec(repo, job), Tier(tier), bandwidth_factor=factor, available=False if unavailable else None
    )
    console.print(f"[yellow]{tier}[/yellow] now available={condition.available} bandwidth x{condition.bandwidth_factor}")


@tier_app.command("restore")
def tier_restore(job: Annotated[str, typer.Option("--job")], workdir: WorkdirOpt = None) -> None:
    """Return every storage tier to normal."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    restore_tiers(_job_spec(repo, job))
    console.print("[green]all tiers restored[/green]")


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

    table = Table("job", "state", "attempt", "step", "placement", "policy", "message")
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
            ",".join(row.placement or []) or "-",
            row.policy_id,
            (row.message or "")[:70],
        )
    console.print(table)


@app.command()
def incidents(
    job: Annotated[Optional[str], typer.Option("--job")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    workdir: WorkdirOpt = None,
) -> None:
    """Show recorded incidents."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    rows = repo.list_incidents(job, limit)
    if not rows:
        console.print("[dim]no incidents[/dim]")
        return

    table = Table("id", "job", "attempts", "signals", "failed at", "resumed from", "replayed", "seconds", "outcome")
    for row in rows:
        kinds = sorted({s.get("kind", "?") for s in (row.signals or [])})
        table.add_row(
            str(row.id),
            row.job_id,
            str(row.attempt),
            ",".join(kinds) or "-",
            str(row.step_at_failure if row.step_at_failure is not None else "-"),
            str(row.chosen_step) if row.chosen_step is not None else "-",
            str(row.lost_steps if row.lost_steps is not None else "-"),
            f"{row.recovery_seconds:.1f}" if row.recovery_seconds else "-",
            row.outcome,
        )
    console.print(table)
    console.print("[dim]`aegis incident <id>` shows the plan, restore report and timeline[/dim]")


@app.command()
def incident(incident_id: Annotated[int, typer.Argument()], workdir: WorkdirOpt = None) -> None:
    """One incident in full: timeline, plans considered, restore verification."""
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    row = repo.get_incident(incident_id)
    if row is None:
        raise typer.BadParameter(f"no incident {incident_id}")
    print_incident(row)


def print_incident(row) -> None:
    console.rule(f"incident {row.id} - {row.job_id} - {row.outcome}")
    console.print(f"failed at step {row.step_at_failure}; {row.detail}")

    timeline = row.timeline or []
    if timeline:
        from datetime import datetime

        start = datetime.fromisoformat(timeline[0]["at"])
        table = Table("t+s", "phase", "attempt", "detail", title="timeline")
        for entry in timeline:
            extra = {k: v for k, v in entry.items() if k not in ("phase", "at", "attempt")}
            table.add_row(
                f"{(datetime.fromisoformat(entry['at']) - start).total_seconds():6.1f}",
                entry["phase"],
                str(entry.get("attempt", "")),
                _short(extra),
            )
        console.print(table)

    for plan in row.plans or []:
        console.print(f"\n[bold]plan for attempt {plan.get('attempt')}[/bold] ({plan.get('strategy')}): {plan.get('reason')}")
        options = ([plan["chosen"]] if plan.get("chosen") else []) + plan.get("alternatives", [])
        if options:
            table = Table("", "step", "sources", "placement", "sched", "transfer", "restore", "replay", "total")
            for i, option in enumerate(options):
                cost = option["cost"]
                table.add_row(
                    "[green]chosen[/green]" if i == 0 and plan.get("chosen") else "",
                    str(option["step"] if option["step"] is not None else "scratch"),
                    ", ".join(f"r{s['rank']}<-{s['location']}" for s in option["sources"]) or "-",
                    ",".join(option["placement"]),
                    f"{cost['scheduling_s']:.1f}",
                    f"{cost['transfer_s']:.1f}",
                    f"{cost['restore_s']:.1f}",
                    f"{cost['replay_s']:.1f} ({cost['replay_steps']})",
                    f"{cost['total_s']:.1f}",
                )
            console.print(table)
        for rejected in plan.get("rejected", []):
            console.print(f"  [red]rejected[/red] step {rejected['step']}: {', '.join(rejected.get('reasons') or [])} {rejected.get('detail', '')}")

    for report in row.restore_reports or []:
        verdict = "[green]verified[/green]" if report["outcome"] == "restored" else f"[red]{report['outcome']}[/red]"
        console.print(
            f"\n[bold]restore, attempt {report.get('attempt')}[/bold]: step {report['step']} {verdict}; "
            f"staged in {report['stage_seconds']:.2f}s, loaded in {report['restore_seconds']:.2f}s, "
            f"ranks consistent={report['consistent_across_ranks']}"
        )
        for rank in report["ranks"]:
            failed = [k for k, ok in rank["checks"].items() if not ok]
            src = rank["staged"]["source"] if rank.get("staged") else "?"
            console.print(
                f"  rank {rank['rank']} on {rank['node']} from {src}: "
                + (f"[red]failed {', '.join(failed)}[/red]" if failed else f"{len(rank['checks'])} checks ok")
                + (f" [red]{rank['error']}[/red]" if rank.get("error") else "")
            )


def _short(values: dict) -> str:
    text = ", ".join(f"{k}={v}" for k, v in values.items() if k != "summary")
    return text if len(text) < 110 else text[:107] + "..."


# ---------------------------------------------------------------- checkpoints


@app.command()
def checkpoints(
    target: Annotated[str, typer.Argument(help="Job id, or a checkpoint root directory")],
    lost: Annotated[Optional[str], typer.Option("--lost", help="Comma-separated nodes to treat as lost")] = None,
    workdir: WorkdirOpt = None,
) -> None:
    """Show every checkpoint, where its copies are, and whether it is recoverable."""
    settings = load_settings(workdir=workdir)
    root = Path(target)
    lost_nodes: set[str] = set(lost.split(",")) if lost else set()
    if not root.is_dir():
        repo = Repository.from_url(settings.resolved_database_url())
        spec = _job_spec(repo, target)
        root = Path(spec.checkpoint_dir)
        if lost is None:
            lost_nodes = repo.unhealthy_nodes()
    layout = Layout(root)
    views = CheckpointCatalog(layout).survey(lost_nodes=lost_nodes, max_candidates=100)
    if not views:
        console.print(f"[dim]no checkpoints under {root}[/dim]")
        return
    conditions = read_conditions(layout)
    degraded = {str(t): c for t, c in conditions.items() if c.degraded}
    if degraded:
        console.print(f"[yellow]degraded tiers:[/yellow] {degraded}")
    if lost_nodes:
        console.print(f"[yellow]excluding lost nodes:[/yellow] {', '.join(sorted(lost_nodes))}")

    table = Table("step", "verdict", "copies per rank", "problems")
    for view in views:
        copies = "; ".join(f"r{rank}: {', '.join(locs)}" for rank, locs in view.tiers.items()) or "-"
        problems = ", ".join(str(r) for r in view.reasons)
        if view.bad_replicas:
            problems += (" | " if problems else "") + ", ".join(
                f"r{r.rank}@{r.location}:{r.state}" for r in view.bad_replicas
            )
        table.add_row(
            str(view.step),
            "[green]recoverable[/green]" if view.recoverable else "[red]not recoverable[/red]",
            copies,
            problems or "-",
        )
    console.print(table)


# ----------------------------------------------------------------------- demo

SCENARIOS: dict[str, dict] = {
    "worker-loss": {
        "fault": "crash:45:1",
        "blurb": "rank 1 dies at step 45; Aegis resumes from step 40, read straight off the surviving nodes' disks",
    },
    "incomplete-checkpoint": {
        "fault": "incomplete_checkpoint:50",
        "blurb": "the job dies mid-checkpoint at step 50; no manifest was committed, so Aegis falls back to step 40",
    },
    "corrupt-shard": {
        "fault": "corrupt_shard:50:1",
        "blurb": "rank 1's primary copy of step 50 rots on disk; Aegis restores step 50 anyway, from the peer replica",
    },
    "corrupt-everywhere": {
        "fault": "corrupt_everywhere:50:1",
        "blurb": "every copy of rank 1's step-50 shard is corrupt; step 50 is unrecoverable and Aegis uses step 40",
    },
    "node-loss": {
        "fault": "node_loss:55:1",
        "blurb": "node-1 dies with its disks at step 55; rank 1's shard survives on its peer, and rank 1 moves to a fresh node",
    },
    "node-loss-no-peers": {
        "fault": "node_loss:55:1",
        "peer": False,
        "blurb": "as node-loss but without peer replicas: step 50 is gone with node-1, so recovery reaches into durable storage",
    },
    "durable-outage": {
        "fault": "node_loss:55:1",
        "tiers": {"durable": {"available": False}},
        "blurb": "object storage is down for the whole run, then a node dies; peer replicas carry the recovery alone",
    },
    "stalled-worker": {
        "fault": "hang:45:1",
        "blurb": "rank 1 hangs at step 45; nothing crashes, so only the progress watcher can see it",
    },
    "failure-during-recovery": {
        "fault": "crash:45:1@0,1",
        "blurb": "the crash recurs on the first recovery attempt too; Aegis backs off and the second attempt holds",
    },
    "restore-verification": {
        "fault": "crash:45:1@0;skew_restore:0@1",
        "blurb": "attempt 1 restores a data position that does not match the manifest; verification rejects it, "
        "the checkpoint is excluded, and attempt 2 uses an older one",
    },
    "repeated-fault": {
        "fault": "crash:45:1@*",
        "blurb": "the fault returns every attempt; Aegis must stop at the budget rather than loop forever",
    },
}


@app.command()
def demo(
    scenario: Annotated[str, typer.Argument(help=f"One of: {', '.join(SCENARIOS)}")] = "worker-loss",
    world_size: Annotated[int, typer.Option("--world-size")] = 2,
    total_steps: Annotated[int, typer.Option("--total-steps")] = 160,
    step_delay: Annotated[float, typer.Option("--step-delay")] = 0.1,
    planner: Annotated[str, typer.Option("--planner", help=f"One of: {', '.join(STRATEGIES)}")] = "cost",
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

    spec = build_spec(
        settings, name=scenario, world_size=world_size, total_steps=total_steps, policy="demo", planner=planner,
        fault=config["fault"], checkpoint_interval=10, peer_replication=config.get("peer", True),
        step_delay=step_delay,
    )
    for tier, condition in config.get("tiers", {}).items():
        degrade_tier(spec, Tier(tier), **condition)
    repo.register_job(spec)

    loop = ControlLoop(repo, reconciler, interval_seconds=1.0, observer=_print_tick)
    loop.install_signal_handlers()
    loop.run(job_ids=[spec.job_id], until_terminal=True)

    console.print()
    row = repo.get_job(spec.job_id)
    outcome = "green" if JobState(row.state) is JobState.SUCCEEDED else "red"
    console.print(f"final state: [{outcome}]{row.state}[/{outcome}] - {row.message}")
    for incident_row in reversed(repo.list_incidents(spec.job_id, 10)):
        print_incident(incident_row)
    console.print(f"\njob id: {spec.job_id}  (try `aegis checkpoints {spec.job_id}`)")


# ----------------------------------------------------------------- experiment


@app.command()
def experiment(
    scenarios: Annotated[str, typer.Option("--scenarios", help="Comma-separated, or 'all'")] = "all",
    planners: Annotated[str, typer.Option("--planners", help="Comma-separated strategies")] = "cost,newest,durable-only",
    trials: Annotated[int, typer.Option("--trials")] = 2,
    total_steps: Annotated[int, typer.Option("--total-steps")] = 160,
    step_delay: Annotated[float, typer.Option("--step-delay")] = 0.1,
    out: Annotated[Path, typer.Option("--out", help="Results directory")] = Path("experiments"),
    timeout: Annotated[float, typer.Option("--timeout", help="Per-trial limit, seconds")] = 300.0,
) -> None:
    """Run fault-injection trials and write a measured report."""
    from aegis.experiments.harness import EXPERIMENT_SCENARIOS, run_experiment

    names = list(EXPERIMENT_SCENARIOS) if scenarios == "all" else [s.strip() for s in scenarios.split(",")]
    unknown = [n for n in names if n not in EXPERIMENT_SCENARIOS]
    if unknown:
        raise typer.BadParameter(f"unknown scenario(s) {unknown}; choose from {', '.join(EXPERIMENT_SCENARIOS)}")
    report = run_experiment(
        scenarios=names,
        planners=[p.strip() for p in planners.split(",") if p.strip()],
        trials=trials,
        total_steps=total_steps,
        step_delay=step_delay,
        out=out,
        timeout=timeout,
        console=console,
    )
    console.print(f"\n[green]report[/green] {report}")


if __name__ == "__main__":
    app()
