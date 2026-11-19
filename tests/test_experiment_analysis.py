from __future__ import annotations

from datetime import UTC, datetime

from aegis.experiments.analysis import Reference, measure_trial, segments, summarize


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat()


def run_events(start: float, attempt: int, resume: int | None, steps: range, *, final=None, loss=lambda s: 1.0 / s):
    events = [{"t": start, "event": "start", "attempt": attempt, "resume_step": resume}]
    t = start + 1.0
    for step in steps:
        events.append({"t": t, "event": "step", "step": step, "loss": loss(step)})
        if step % 10 == 0:
            events.append({"t": t, "event": "checkpoint", "rank": 0, "step": step, "blocking_s": 0.01})
            events.append({"t": t, "event": "checkpoint", "rank": 1, "step": step, "blocking_s": 0.5})
        if step % 25 == 0:
            events.append({"t": t, "event": "eval", "step": step, "eval_loss": 1.0 / step})
        t += 0.1
    if final is not None:
        events.append({"t": t, "event": "final", "digests": final})
    return events


DIGESTS = {"global": {"model": "abc"}}


def reference() -> Reference:
    return Reference.from_events(run_events(0.0, 0, None, range(1, 101), final=DIGESTS), 12.0)


def test_log_splits_into_one_segment_per_attempt():
    events = run_events(0.0, 0, None, range(1, 46)) + run_events(20.0, 1, 40, range(41, 101), final=DIGESTS)
    first, second = segments(events)
    assert (first.attempt, first.last_step) == (0, 45)
    assert (second.attempt, second.resume_step, second.last_step) == (1, 40, 100)
    assert first.checkpoints == 4, "only rank 0's saves count; ranks save in lockstep"
    assert second.final_digests == DIGESTS


def test_a_resumed_run_is_measured_against_the_reference():
    events = run_events(0.0, 0, None, range(1, 46)) + run_events(20.0, 1, 40, range(41, 101), final=DIGESTS)
    incident = {
        "outcome": "recovered", "attempt": 1, "step_at_failure": 45, "chosen_step": 40,
        "timeline": [
            {"phase": "detected", "at": iso(7.0)},
            {"phase": "training_resumed", "at": iso(21.5)},
            {"phase": "verified", "at": iso(25.0)},
        ],
        "plans": [{"chosen": {
            "sources": [{"location": "local@node-0"}, {"location": "peer@node-0"}],
            "cost": {"scheduling_s": 4.0, "transfer_s": 0.5, "restore_s": 0.3, "replay_s": 0.5, "replay_steps": 5},
        }}],
        "restore_reports": [{"outcome": "restored", "stage_seconds": 0.6, "restore_seconds": 0.25}],
    }
    m = measure_trial(
        scenario="worker-loss", planner="cost", trial=0, job_id="j", final_state="succeeded", message="",
        submitted_at=-1.0, finished_at=30.0, world_size=2, events=events, incidents=[incident], reference=reference(),
    )
    assert m.launches == 2 and m.attempts == 1
    assert m.replayed_steps == 5
    assert m.digests_match is True and m.max_loss_diff == 0.0
    assert m.reached_target and m.time_to_target_s > 20
    (i,) = m.incidents
    assert i.time_to_resume_s == 14.5 and i.time_to_verified_s == 18.0
    assert i.detection_s == round(7.0 - (1.0 + 44 * 0.1), 3)
    assert i.sources == ["local@node-0", "peer@node-0"]
    assert (i.estimated_restore_s, i.measured_restore_s, i.estimated_total_s) == (0.8, 0.85, 5.3)
    assert 0 < m.checkpoint_overhead < 0.02


def test_a_run_that_restarted_from_scratch_replays_everything_and_differs():
    events = run_events(0.0, 0, None, range(1, 46)) + run_events(
        20.0, 1, None, range(1, 101), final={"global": {"model": "other"}}, loss=lambda s: 2.0 / s
    )
    m = measure_trial(
        scenario="x", planner="cost", trial=0, job_id="j", final_state="succeeded", message="",
        submitted_at=0.0, finished_at=40.0, world_size=2, events=events, incidents=[], reference=reference(),
    )
    assert m.replayed_steps == 45
    assert m.digests_match is False
    assert m.max_loss_diff > 0


def test_summary_groups_by_scenario_and_planner():
    ref = reference()
    events = run_events(0.0, 0, None, range(1, 101), final=DIGESTS)
    trials = [
        measure_trial(
            scenario="baseline", planner="cost", trial=n, job_id=f"j{n}", final_state="succeeded", message="",
            submitted_at=0.0, finished_at=12.0, world_size=2, events=events, incidents=[], reference=ref,
        )
        for n in range(2)
    ]
    (row,) = summarize(trials, reference_wall_s=11.0)
    assert row["trials"] == 2 and row["succeeded"] == 2 and row["digests_match"] == "2/2"
    assert row["replayed_steps"] == 0
