"""Fault-injection experiment runner.

Each trial gets its own Aegis workdir (database, events, checkpoints, runs) so
trials cannot see each other's state, and runs the controller as a separate
process exactly as an operator would -- which is what lets the
controller-crash scenario SIGKILL it mid-recovery and start a fresh one.

Trials run one at a time. The fault scenarios measure latency; running them
concurrently on one machine would mostly measure CPU contention.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from aegis.backends.local import LocalBackend
from aegis.checkpoint import Tier
from aegis.config import load_settings
from aegis.experiments.analysis import Reference, TrialMeasure, measure_trial, summarize
from aegis.faults import degrade_tier
from aegis.models import TERMINAL_STATES, JobState
from aegis.store.repository import Repository
from aegis.telemetry import read_events

TERMINAL = {s.value for s in TERMINAL_STATES}


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: str
    blurb: str
    peer: bool = True
    tiers: dict = field(default_factory=dict)
    # Tier conditions applied once the controller has detected the failure:
    # storage that degrades during recovery, after the checkpoints were written.
    tiers_on_failure: dict = field(default_factory=dict)
    # SIGKILL the controller when the job first reaches one of these states,
    # then start a new one.
    kill_controller_in: tuple[str, ...] = ()
    # Whether the recovery planner has a real choice to make here. Scenarios
    # where every strategy picks the same checkpoint run once, not per planner.
    planner_sensitive: bool = False


EXPERIMENT_SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("baseline", "", "no fault: the reference every other run is compared against"),
        Scenario("worker-loss", "crash:45:1", "rank 1 crashes at step 45"),
        Scenario("incomplete-checkpoint", "incomplete_checkpoint:50", "job dies mid-save at step 50; no manifest"),
        Scenario("corrupt-shard", "corrupt_shard:50:1", "rank 1's local copy of step 50 is corrupted"),
        Scenario("node-loss", "node_loss:55:1", "node-1 dies with its disks at step 55", planner_sensitive=True),
        Scenario(
            "node-loss-no-peers", "node_loss:55:1", "node loss with no peer replicas; only durable copies survive",
            peer=False, planner_sensitive=True,
        ),
        Scenario(
            "storage-degraded", "node_loss:55:1",
            "node loss, then the peer network drops to 2% bandwidth during recovery: the newest checkpoint "
            "is now the slowest to fetch",
            tiers_on_failure={"peer": {"bandwidth_factor": 0.02}}, planner_sensitive=True,
        ),
        Scenario(
            "durable-outage", "node_loss:55:1", "object storage down for the whole run, then a node dies",
            tiers={"durable": {"available": False}}, planner_sensitive=True,
        ),
        Scenario("failure-during-recovery", "crash:45:1@0,1", "the crash recurs on the first recovery attempt"),
        Scenario(
            "restore-verification", "crash:45:1@0;skew_restore:0@1",
            "attempt 1 restores a wrong data position; verification must reject it",
        ),
        Scenario(
            "controller-crash", "crash:45:1", "the controller is SIGKILLed mid-recovery and replaced",
            kill_controller_in=(JobState.RECOVERING.value, JobState.VERIFYING.value),
        ),
        Scenario("repeated-fault", "crash:45:1@*", "the fault returns every attempt; the budget must stop it"),
    ]
}


@dataclass
class TrialConfig:
    total_steps: int
    step_delay: float
    timeout: float
    world_size: int = 2
    checkpoint_interval: int = 10
    durable_interval: int = 20
    controller_interval: float = 0.5
    lease_ttl: float = 5.0


class _Controller:
    def __init__(self, workdir: Path, job_id: str, cfg: TrialConfig, log: Path) -> None:
        self.workdir, self.job_id, self.cfg, self.log = workdir, job_id, cfg, log
        self.proc: subprocess.Popen | None = None
        self.restarts = 0

    def start(self) -> None:
        argv = [
            sys.executable, "-m", "aegis.cli", "controller",
            "--workdir", str(self.workdir), "--job", self.job_id, "--until-terminal",
            "--interval", str(self.cfg.controller_interval), "--lease-ttl", str(self.cfg.lease_ttl), "--quiet",
        ]
        out = self.log.open("a")
        out.write(f"\n=== controller start {datetime.now(UTC).isoformat()} ===\n")
        out.flush()
        self.proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _incident_records(repo: Repository, job_id: str) -> list[dict]:
    return [
        {
            "id": row.id,
            "outcome": row.outcome,
            "attempt": row.attempt,
            "step_at_failure": row.step_at_failure,
            "chosen_step": row.chosen_step,
            "timeline": row.timeline or [],
            "plans": row.plans or [],
            "restore_reports": row.restore_reports or [],
            "signals": row.signals or [],
            "detail": row.detail,
        }
        for row in reversed(repo.list_incidents(job_id, 50))
    ]


def run_trial(
    scenario: Scenario,
    planner: str,
    trial: int,
    cfg: TrialConfig,
    root: Path,
    reference: Reference | None,
    console: Console,
) -> tuple[TrialMeasure, list[dict]]:
    from aegis.cli import build_spec

    workdir = (root / "trials" / f"{scenario.name}--{planner}--{trial}").resolve()
    settings = load_settings(workdir=workdir)
    repo = Repository.from_url(settings.resolved_database_url())
    spec = build_spec(
        settings, name=scenario.name, world_size=cfg.world_size, total_steps=cfg.total_steps, policy="demo",
        planner=planner, fault=scenario.fault, checkpoint_interval=cfg.checkpoint_interval,
        durable_interval=cfg.durable_interval, peer_replication=scenario.peer, step_delay=cfg.step_delay,
    )
    spec = replace(spec, workdir=str(workdir))
    for tier, condition in scenario.tiers.items():
        degrade_tier(spec, Tier(tier), **condition)

    submitted = time.time()
    repo.register_job(spec)
    controller = _Controller(workdir, spec.job_id, cfg, workdir / "controller.log")
    controller.start()

    deadline = submitted + cfg.timeout
    killed = degraded = False
    state = JobState.PENDING.value
    while time.time() < deadline:
        row = repo.get_job(spec.job_id)
        state = row.state
        if state in TERMINAL:
            break
        if scenario.tiers_on_failure and not degraded and state == JobState.SUSPECT.value:
            for tier, condition in scenario.tiers_on_failure.items():
                degrade_tier(spec, Tier(tier), **condition)
            degraded = True
        if scenario.kill_controller_in and not killed and state in scenario.kill_controller_in:
            controller.kill()
            killed = True
            console.print(f"    [yellow]SIGKILL controller in {state}; restarting[/yellow]")
            time.sleep(1.0)
            controller.restarts += 1
            controller.start()
        if not controller.alive:
            controller.restarts += 1
            controller.start()
        time.sleep(0.2)
    finished = time.time()

    timed_out = state not in TERMINAL
    controller.stop()
    if timed_out:
        backend = LocalBackend()
        for handle in backend.discover(spec):
            backend.terminate(handle, grace_seconds=5)

    row = repo.get_job(spec.job_id)
    events = read_events(LocalBackend().metrics_path(spec))
    incidents = _incident_records(repo, spec.job_id)
    if reference is None:
        # The first baseline is its own reference.
        reference = Reference.from_events(events, finished - submitted)
    measure = measure_trial(
        scenario=scenario.name, planner=planner, trial=trial, job_id=spec.job_id,
        final_state="timeout" if timed_out else row.state, message=row.message or "",
        submitted_at=submitted, finished_at=finished, world_size=cfg.world_size,
        events=events, incidents=incidents, reference=reference, controller_restarts=controller.restarts,
    )
    return measure, events


def run_experiment(
    *,
    scenarios: list[str],
    planners: list[str],
    trials: int,
    total_steps: int,
    step_delay: float,
    out: Path,
    timeout: float,
    console: Console,
) -> Path:
    cfg = TrialConfig(total_steps=total_steps, step_delay=step_delay, timeout=timeout)
    root = (out / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fault_scenarios = [s for s in scenarios if s != "baseline"]

    console.rule(f"[bold]experiment[/bold] -> {root}")
    results: list[TrialMeasure] = []
    references: list[Reference] = []

    reference: Reference | None = None
    for trial in range(trials):
        console.print(f"[bold]trial {trial}[/bold] baseline")
        measure, events = run_trial(EXPERIMENT_SCENARIOS["baseline"], "cost", trial, cfg, root, reference, console)
        if reference is None and measure.final_state == "succeeded":
            reference = Reference.from_events(events, measure.wall_s)
        results.append(measure)
        _print_line(console, measure)

    if reference is None or reference.digests is None:
        console.print("[red]baseline did not finish; fault runs will have nothing to be compared against[/red]")

    for trial in range(trials):
        for name in fault_scenarios:
            scenario = EXPERIMENT_SCENARIOS[name]
            for planner in planners if scenario.planner_sensitive else planners[:1]:
                console.print(f"[bold]trial {trial}[/bold] {name} [dim]({planner})[/dim]")
                measure, _ = run_trial(scenario, planner, trial, cfg, root, reference, console)
                results.append(measure)
                _print_line(console, measure)
                _write(root, cfg, results, reference, scenarios, planners, trials)

    return _write(root, cfg, results, reference, scenarios, planners, trials)


def _print_line(console: Console, m: TrialMeasure) -> None:
    color = "green" if m.final_state == "succeeded" else "red"
    first = m.incidents[0] if m.incidents else None
    bits = [f"[{color}]{m.final_state}[/{color}]", f"wall {m.wall_s:.1f}s", f"launches {m.launches}"]
    if first:
        bits.append(f"resume {first.time_to_resume_s}s from step {first.chosen_step} via {','.join(first.sources)}")
    bits.append(f"replayed {m.replayed_steps}")
    if m.digests_match is not None:
        bits.append("state identical" if m.digests_match else "[yellow]state differs[/yellow]")
    console.print("    " + "  ".join(bits))


def _write(
    root: Path,
    cfg: TrialConfig,
    results: list[TrialMeasure],
    reference: Reference | None,
    scenarios: list[str],
    planners: list[str],
    trials: int,
) -> Path:
    baseline_ttt = [r.time_to_target_s or r.wall_s for r in results if r.scenario == "baseline"]
    reference_wall = sum(baseline_ttt) / len(baseline_ttt) if baseline_ttt else None
    summary = summarize(results, reference_wall)
    payload = {
        "config": {**cfg.__dict__, "scenarios": scenarios, "planners": planners, "trials": trials},
        "reference": {
            "target_eval_loss": reference.target_loss if reference else None,
            "wall_s": reference_wall,
        },
        "summary": summary,
        "trials": [r.to_record() for r in results],
    }
    (root / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    report = root / "report.md"
    report.write_text(_markdown(payload))
    return report


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1e-2 and value else f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "-"
    return str(value)


def _markdown(payload: dict) -> str:
    cfg = payload["config"]
    ref = payload["reference"]
    lines = [
        "# Aegis fault-injection experiment",
        "",
        f"- world size {cfg['world_size']}, {cfg['total_steps']} steps, step delay {cfg['step_delay']}s, "
        f"checkpoint every {cfg['checkpoint_interval']} steps, durable every {cfg['durable_interval']}",
        f"- trials per scenario: {cfg['trials']}; planners: {', '.join(cfg['planners'])}",
        f"- reference (uninterrupted) run: {_fmt(ref['wall_s'])}s to target; target eval loss {_fmt(ref['target_eval_loss'])}",
        "",
        "Times are seconds measured from the moment the controller detected the failure.",
        "`resume` = first training step after the restore; `caught up` = back at the step where the fault hit;",
        "`verified` = recovery declared good. `worker-s` is CPU worker-seconds (busy time x world size).",
        "`state identical` compares final model/optimizer/LR-schedule/RNG digests with the reference run.",
        "",
        "## Recovery",
        "",
        "| scenario | planner | ok | recovered | detect | resume | caught up | verified | replayed steps | restore est / measured | sources |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["summary"]:
        lines.append(
            f"| {r['scenario']} | {r['planner']} | {r['succeeded']}/{r['trials']} | {r['recovered']}/{r['incidents']} | "
            f"{_fmt(r['detection_s'])} | {_fmt(r['resume_s'])} | {_fmt(r['caught_up_s'])} | {_fmt(r['verified_s'])} | "
            f"{_fmt(r['replayed_steps'])} | {_fmt(r['est_restore_s'])} / {_fmt(r['measured_restore_s'])} | {_fmt(r['sources'])} |"
        )
    lines += [
        "",
        "## Cost and training behaviour",
        "",
        "| scenario | planner | time to target | vs reference | worker-s | checkpoint overhead | state identical | max loss diff |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["summary"]:
        overhead = f"{r['ckpt_overhead'] * 100:.1f}%" if r["ckpt_overhead"] is not None else "-"
        slowdown = f"+{r['target_slowdown_s']:.1f}s" if r["target_slowdown_s"] is not None else "-"
        worker_s = f"{r['worker_hours'] * 3600:.0f}" if r["worker_hours"] is not None else "-"
        lines.append(
            f"| {r['scenario']} | {r['planner']} | {_fmt(r['time_to_target_s'])} | {slowdown} | "
            f"{worker_s} | {overhead} | {r['digests_match']} | {_fmt(r['max_loss_diff'])} |"
        )
    lines += ["", "## Trials", ""]
    for t in payload["trials"]:
        lines.append(
            f"- `{t['scenario']}` / {t['planner']} #{t['trial']}: **{t['final_state']}** in {t['wall_s']:.1f}s, "
            f"{t['launches']} launches, {t['controller_restarts']} controller restarts -- {t['message']}"
        )
        for i in t["incidents"]:
            lines.append(
                f"  - incident: {i['outcome']}, failed at step {i['step_at_failure']}, restored step "
                f"{i['chosen_step']} from {_fmt(i['sources'])} after {i['attempts']} attempt(s)"
            )
    return "\n".join(lines) + "\n"
