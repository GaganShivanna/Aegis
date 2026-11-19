"""Turn one trial's raw record -- the job's metrics log and its incidents --
into measurements.

Kept free of process management so it can be tested on synthetic logs.

The metrics log holds every attempt of a job back to back. Rank 0 writes a
`start` event when an attempt's processes come up, so the log splits into
one segment per attempt at those events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any


@dataclass
class Segment:
    attempt: int
    resume_step: int | None
    started: float
    ended: float
    steps: dict[int, float] = field(default_factory=dict)  # step -> loss
    step_times: dict[int, float] = field(default_factory=dict)  # step -> wall time
    evals: list[tuple[float, int, float]] = field(default_factory=list)  # (t, step, eval_loss)
    checkpoint_blocking_s: float = 0.0
    checkpoints: int = 0
    final_digests: dict | None = None

    @property
    def last_step(self) -> int:
        return max(self.steps, default=self.resume_step or 0)

    @property
    def seconds(self) -> float:
        return max(0.0, self.ended - self.started)


def segments(events: list[dict[str, Any]]) -> list[Segment]:
    out: list[Segment] = []
    for e in events:
        kind, t = e.get("event"), float(e.get("t", 0.0))
        if kind == "start":
            out.append(Segment(int(e.get("attempt") or 0), e.get("resume_step"), t, t))
            continue
        if not out:
            continue
        seg = out[-1]
        seg.ended = max(seg.ended, t)
        if kind == "step":
            seg.steps[int(e["step"])] = float(e["loss"])
            seg.step_times[int(e["step"])] = t
        elif kind == "eval":
            seg.evals.append((t, int(e["step"]), float(e["eval_loss"])))
        elif kind == "checkpoint" and int(e.get("rank", 0)) == 0:
            seg.checkpoint_blocking_s += float(e.get("blocking_s", 0.0))
            seg.checkpoints += 1
        elif kind == "final":
            seg.final_digests = e.get("digests")
    return out


def _ts(value: str | None) -> float | None:
    return datetime.fromisoformat(value).timestamp() if value else None


def _phase(timeline: list[dict], phase: str) -> float | None:
    return next((_ts(e["at"]) for e in timeline if e.get("phase") == phase), None)


def _between(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(b - a, 3)


@dataclass
class IncidentMeasure:
    outcome: str
    attempts: int
    step_at_failure: int | None
    chosen_step: int | None
    detection_s: float | None  # last sign of life -> detected
    time_to_resume_s: float | None  # detected -> first training step after restore
    time_to_caught_up_s: float | None  # detected -> back at the step where it failed
    time_to_verified_s: float | None  # detected -> recovery declared good
    sources: list[str]
    estimated_restore_s: float | None  # planner: transfer + restore for the chosen candidate
    measured_restore_s: float | None  # job: staging + load, from the restore report
    estimated_total_s: float | None
    replay_steps_planned: int | None


def measure_incident(incident: dict, segs: list[Segment]) -> IncidentMeasure:
    timeline = incident.get("timeline") or []
    detected = _phase(timeline, "detected")
    previous = [s for s in segs if detected is not None and s.started <= detected]
    last_alive = max((s.ended for s in previous), default=None)
    plans = [p for p in (incident.get("plans") or []) if p.get("chosen")]
    chosen = plans[-1]["chosen"] if plans else None
    cost = chosen["cost"] if chosen else None
    reports = [r for r in (incident.get("restore_reports") or []) if r.get("outcome") == "restored"]
    report = reports[-1] if reports else None
    return IncidentMeasure(
        outcome=incident.get("outcome", "?"),
        attempts=int(incident.get("attempt") or 0),
        step_at_failure=incident.get("step_at_failure"),
        chosen_step=incident.get("chosen_step"),
        detection_s=_between(last_alive, detected),
        time_to_resume_s=_between(detected, _phase(timeline, "training_resumed")),
        time_to_caught_up_s=_between(detected, _phase(timeline, "caught_up")),
        time_to_verified_s=_between(detected, _phase(timeline, "verified")),
        sources=[s["location"] for s in chosen["sources"]] if chosen else [],
        estimated_restore_s=round(cost["transfer_s"] + cost["restore_s"], 3) if cost else None,
        measured_restore_s=round(report["stage_seconds"] + report["restore_seconds"], 3) if report else None,
        estimated_total_s=round(sum(cost[k] for k in ("scheduling_s", "transfer_s", "restore_s", "replay_s")), 3)
        if cost
        else None,
        replay_steps_planned=cost["replay_steps"] if cost else None,
    )


@dataclass
class TrialMeasure:
    scenario: str
    planner: str
    trial: int
    job_id: str
    final_state: str
    message: str
    wall_s: float
    attempts: int
    launches: int
    controller_restarts: int
    worker_hours: float
    checkpoint_overhead: float  # fraction of busy time rank 0 spent blocked in saves
    checkpoints: int
    replayed_steps: int
    time_to_target_s: float | None
    reached_target: bool
    digests_match: bool | None
    max_loss_diff: float | None
    final_eval_loss: float | None
    incidents: list[IncidentMeasure]

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Reference:
    """What an uninterrupted run did: the bar every faulted run is held to."""

    digests: dict | None
    losses: dict[int, float]
    target_loss: float | None
    wall_s: float

    @classmethod
    def from_events(cls, events: list[dict], wall_s: float) -> Reference:
        segs = segments(events)
        if not segs:
            return cls(None, {}, None, wall_s)
        seg = segs[-1]
        target = seg.evals[-1][2] if seg.evals else None
        return cls(seg.final_digests, dict(seg.steps), target, wall_s)


def measure_trial(
    *,
    scenario: str,
    planner: str,
    trial: int,
    job_id: str,
    final_state: str,
    message: str,
    submitted_at: float,
    finished_at: float,
    world_size: int,
    events: list[dict],
    incidents: list[dict],
    reference: Reference | None,
    controller_restarts: int = 0,
    target_tolerance: float = 1e-9,
) -> TrialMeasure:
    segs = segments(events)
    busy = sum(s.seconds for s in segs)
    blocking = sum(s.checkpoint_blocking_s for s in segs)

    replayed = 0
    for prev, seg in zip(segs, segs[1:]):
        if seg.resume_step is not None:
            replayed += max(0, prev.last_step - seg.resume_step)
        else:
            replayed += prev.last_step

    # The trajectory the model actually followed: for each step, the attempt
    # that ran it last.
    path: dict[int, float] = {}
    for seg in segs:
        path.update(seg.steps)
    max_diff = None
    if reference and reference.losses:
        common = [s for s in path if s in reference.losses]
        if common:
            max_diff = max(abs(path[s] - reference.losses[s]) for s in common)

    final = next((s.final_digests for s in reversed(segs) if s.final_digests), None)
    digests_match = None
    if reference and reference.digests is not None:
        digests_match = final == reference.digests if final is not None else False

    evals = sorted(e for s in segs for e in s.evals)
    final_eval = evals[-1][2] if evals else None
    time_to_target = None
    if reference and reference.target_loss is not None:
        hit = next((t for t, _, loss in evals if loss <= reference.target_loss * (1 + target_tolerance)), None)
        time_to_target = round(hit - submitted_at, 3) if hit is not None else None

    return TrialMeasure(
        scenario=scenario,
        planner=planner,
        trial=trial,
        job_id=job_id,
        final_state=final_state,
        message=message,
        wall_s=round(finished_at - submitted_at, 3),
        attempts=max((s.attempt for s in segs), default=0),
        launches=len(segs),
        controller_restarts=controller_restarts,
        worker_hours=round(busy * world_size / 3600, 6),
        checkpoint_overhead=round(blocking / busy, 4) if busy else 0.0,
        checkpoints=sum(s.checkpoints for s in segs),
        replayed_steps=replayed,
        time_to_target_s=time_to_target,
        reached_target=time_to_target is not None,
        digests_match=digests_match,
        max_loss_diff=max_diff,
        final_eval_loss=final_eval,
        incidents=[measure_incident(i, segs) for i in incidents],
    )


# --------------------------------------------------------------- aggregation


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(mean(present), 3) if present else None


def summarize(trials: list[TrialMeasure], reference_wall_s: float | None) -> list[dict]:
    groups: dict[tuple[str, str], list[TrialMeasure]] = {}
    for t in trials:
        groups.setdefault((t.scenario, t.planner), []).append(t)
    rows = []
    for (scenario, planner), group in groups.items():
        incidents = [i for t in group for i in t.incidents]
        ttt = _avg([t.time_to_target_s for t in group])
        rows.append(
            {
                "scenario": scenario,
                "planner": planner,
                "trials": len(group),
                "succeeded": sum(t.final_state == "succeeded" for t in group),
                "recovered": sum(i.outcome == "recovered" for i in incidents),
                "incidents": len(incidents),
                "detection_s": _avg([i.detection_s for i in incidents]),
                "resume_s": _avg([i.time_to_resume_s for i in incidents]),
                "caught_up_s": _avg([i.time_to_caught_up_s for i in incidents]),
                "verified_s": _avg([i.time_to_verified_s for i in incidents]),
                "replayed_steps": _avg([t.replayed_steps for t in group]),
                "worker_hours": _avg([t.worker_hours for t in group]),
                "ckpt_overhead": _avg([t.checkpoint_overhead for t in group]),
                "time_to_target_s": ttt,
                "target_slowdown_s": round(ttt - reference_wall_s, 3) if ttt is not None and reference_wall_s else None,
                "digests_match": f"{sum(bool(t.digests_match) for t in group)}/{len(group)}",
                "max_loss_diff": max((t.max_loss_diff for t in group if t.max_loss_diff is not None), default=None),
                "est_restore_s": _avg([i.estimated_restore_s for i in incidents]),
                "measured_restore_s": _avg([i.measured_restore_s for i in incidents]),
                "sources": sorted({src for i in incidents for src in i.sources}),
            }
        )
    return rows
