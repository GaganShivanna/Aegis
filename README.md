# Aegis

A recovery controller for distributed PyTorch training. Aegis watches a training job, correlates failure signals into a single incident, picks a checkpoint it can prove is complete, relaunches under a bounded attempt budget, and refuses to call the recovery successful until the job demonstrably makes progress again.

It sits between two things that already exist and does the part neither of them does:

| Layer | Owns |
| --- | --- |
| NVSentinel | hardware detection, quarantine, draining, remediation |
| **Aegis** | **correlating signals, choosing a checkpoint, job-level restart budget, verifying recovery** |
| PyTorch Elastic | executing the worker restart |

Aegis never cordons a node, uncordons a node, or resets a GPU. It consumes node conditions and excludes unhealthy nodes from placement; repairing them is NVSentinel's job.

**Status:** Milestone 1 (Recovery MVP) is built and runs end to end on a laptop with CPU workers. Milestones 2 and 3 are not built; see [Not built yet](#not-built-yet).

## Quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). No cluster, no database, no GPU.

```bash
uv sync
uv run aegis demo worker-loss
```

That submits a 2-worker DDP job that loses a rank at step 45 and reconciles it to completion. You should see Aegis detect the exit, fall back to the last complete checkpoint, relaunch, and verify:

```
-> running    launched
-> suspect    signals observed          # worker_exit correlated into incident 1
-> recovering recovery scheduled        # attempt 1 after 1.0s backoff
-> verifying  relaunched                # resumed from step 40
   verifying  step 50/60, new checkpoint=yes
-> running    recovery verified         # 10.2s, 5 steps lost
-> succeeded  completed
```

### Scenarios

```bash
uv run aegis demo <scenario>
```

| Scenario | What it exercises |
| --- | --- |
| `worker-loss` | a rank exits non-zero; resume from the last complete checkpoint |
| `incomplete-checkpoint` | the job dies mid-write; the partial checkpoint is rejected and Aegis falls back |
| `missing-shard` | a shard never lands; the manifest no longer matches the expected ranks |
| `stalled-worker` | a rank hangs. Nothing crashes, so only the progress watcher can see it |
| `repeated-fault` | the fault returns every attempt; Aegis must stop at the budget, not loop |

### Driving it manually

```bash
# register a job
uv run aegis submit --name myjob --world-size 2 --total-steps 400

# reconcile it
uv run aegis controller --metrics-port 9090

# inject a signal, standing in for NVSentinel
uv run aegis inject node_unhealthy --node gpu-07

# inspect
uv run aegis status
uv run aegis incidents
uv run aegis validate .aegis/checkpoints/myjob --job-name myjob --world-size 2
```

`aegis validate` is useful on its own: point it at any checkpoint directory and it reports, per checkpoint, whether Aegis would accept it and exactly why not.

## How it works

### The state machine

One `tick()` is a complete, crash-safe step for one job.

```
PENDING ──launch──> RUNNING ──signals──> SUSPECT ──correlation window──> RECOVERING
                       ^                    │                                │
                       │                    └──progress resumed──> RUNNING   │ backoff,
                       │                                                     │ pick checkpoint
                       └──20 new steps + new checkpoint── VERIFYING <────────┘
                                                              │
                                     attempt limit / deadline │
                                                              v
                                                          EXHAUSTED
```

`SUSPECT` exists because a single bad node produces a node condition *plus* one worker exit per rank. Acting on the first signal would start several recoveries for one failure, so Aegis holds briefly and folds them into one incident.

`VERIFYING` exists because a process starting is not evidence that recovery worked. A job is only recovered once it has taken 20 new steps past the restore point *and* written a fresh complete checkpoint.

### The checkpoint contract

A checkpoint is complete when, and only when, its manifest exists. Shards are written and fsynced first; the manifest is published last via atomic rename. A crash at any earlier point leaves a directory the validator rejects, which makes "died mid-write" indistinguishable from "never happened".

Validation gates, in order: manifest present and parseable → supported format version → every expected rank declared and present → sizes match → SHA-256 checksums match → metadata compatible with the job being launched → every shard deserializes → trial restore succeeds. Selection walks backwards from the newest checkpoint and takes the first one that clears all of them, recording the rejected candidates on the incident.

This catches truncation, missing shards, bit rot and structural mismatch. **It cannot catch silent numerical corruption** in otherwise well-formed tensors.

A storage read error is deliberately *not* treated as "no checkpoint here". It propagates, and the controller retries without consuming a recovery attempt.

### Why there is never a second copy of your job

"Zero duplicate active jobs" is an acceptance criterion, so it is enforced structurally rather than by being careful:

- **One recovery lease per job**, with a fence token that increments whenever the lease changes hands. A controller that stalls past its lease cannot complete an action it planned earlier.
- **Deterministic action IDs.** The ID for (job, attempt, `launch`) is derived by hash, so two replicas, or one controller before and after a crash, compute the same ID. The ledger's unique constraint does the rest.
- **The ledger is written before the side effect.** On restart, an action recorded but not started is reconciled against what is actually running, not assumed.
- **The backend is asked what exists before every launch**, and strays are terminated first. On Kubernetes the Job name is derived from the action ID, so a duplicate create fails with 409 rather than producing a second set of workers.
- **If a stray cannot be confirmed dead, Aegis does not launch.** It stalls the recovery and retries, because a stalled recovery is recoverable and two concurrent jobs writing the same checkpoints are not.

### Resumability

The training script round-trips model, optimizer, step counter, per-rank RNG state, and data position. `tests/test_training_resume.py` asserts this the only way that really counts: a run interrupted at step 10 and resumed to step 20 produces bit-identical weights to an uninterrupted 20-step run.

## Testing

```bash
uv run pytest                  # 47 tests, ~18s
uv run pytest -m "not slow"    # skip the tests that spawn real torchrun
```

The suite covers the PRD's acceptance cases: worker loss, stalled workers, incomplete checkpoints, missing shards, checksum mismatch, temporary storage failure, repeated faults, attempt exhaustion, recovery deadline, signal deduplication, and four distinct controller-crash windows.

## Layout

```
src/aegis/
  checkpoint/     manifest, atomic writer, validator, storage
  controller/     reconciler (state machine), policies, progress watcher, loop
  backends/       local subprocess, kubernetes
  store/          durable job state, lease, action ledger, incidents
  events/         health event sources
training/
  train_ddp.py    resumable DDP job with fault injection
```

## Configuration

Any setting is overridable by an `AEGIS_`-prefixed environment variable. Recovery policy is versioned and selected by name (`baseline`, `fast-retry`, `patient`, `demo`); the baseline is three attempts, exponential backoff from 10s with 20% jitter, and 20 steps required to verify.

Storage defaults to SQLite under `.aegis/`. Point `AEGIS_DATABASE_URL` at PostgreSQL for anything real — the schema is identical.

## Kubernetes

`src/aegis/backends/k8s.py` creates one Job per attempt with `backoffLimit: 0`, labels carrying the job and action IDs, and node-affinity exclusion for nodes reported unhealthy. It is written but **not yet exercised against a live cluster** — there is no cluster on this machine. Install `kind` and run with `AEGIS_BACKEND=k8s` to try it.

Aegis owns retries, so Elastic is configured with `max_restarts=0` and the Job controller with `backoffLimit: 0`. Two retry policies underneath an attempt budget would make the budget meaningless.

## Not built yet

Milestone 2 (NVSentinel integration): consuming real node conditions, drain/remediation ordering, topology-aware placement scoring.

Milestone 3 (adaptive recovery): incident retrieval over similar workloads, policy ranking, LLM scenario generation with a validator, the Chaos Mesh rehearsal runner, and the promotion gate. The incident schema, policy versioning and workload fingerprints are already recorded so that incidents captured today remain usable once ranking exists.

## Known limits

- CPU-only here. Gloo on two local ranks validates orchestration and recovery logic, not GPU fault handling. Simulated GPU faults exercise control flow, not physical hardware recovery.
- Checkpoint validation cannot detect silent numerical corruption.
- Fixed world size. Elastic resharding is out of scope.
- The local backend detects liveness by process group, which is theoretically vulnerable to PID reuse; the recorded exit code is checked first to make that unlikely.
- No uninterrupted-training guarantee. The goal is less downtime and less lost progress.
