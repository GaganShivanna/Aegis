# Aegis

A recovery controller for distributed PyTorch training. When a worker or a whole node fails, Aegis works out which checkpoints are still reachable on the infrastructure that survived, picks the cheapest way to resume, places the job on healthy nodes, restores it, verifies that the restored state is exactly what was saved, and only then declares the recovery successful. Every decision is persisted, so the controller itself can be killed at any point and a fresh one finishes the job without launching a duplicate.

It sits between two things that already exist and does the part neither of them does:

| Layer | Owns |
| --- | --- |
| NVSentinel | hardware detection, quarantine, draining, remediation |
| **Aegis** | **checkpoint reachability, recovery planning, placement, restore verification, job-level restart budget** |
| PyTorch Elastic | executing the worker restart |

Aegis never cordons a node, uncordons a node, or resets a GPU. It consumes node conditions and keeps jobs off unhealthy nodes; repairing them is NVSentinel's job.

## Quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). No cluster, no database, no GPU.

```bash
uv sync
uv run aegis demo node-loss
```

That runs a 2-worker DDP job on a simulated 4-node pool. At step 55, node-1 dies *with its disks*. Aegis sees that step 50 is still complete (rank 0's copy on node-0, rank 1's copy replicated to node-0 as a peer), moves rank 1 to a fresh node, restores both ranks, checks every restored value against the checkpoint, and finishes training:

```
 t+s  phase             detail
 0.0  detected          step=55, signals=['node_unhealthy', 'worker_exit']
 1.0  recovering        backoff_s=1.004
 2.0  planned           step=50, placement=['node-0', 'node-2'], estimate_s=6.17
 2.1  launched
 5.1  restored          step=50, stage_s=0.387, load_s=0.27, sources=['local@node-0', 'peer@node-0']
 5.1  training_resumed  step=52
 6.1  caught_up         step=61
 7.1  verified

plan (cost): step 50 on node-0,node-2: est 6.2s (sched 5.0 + transfer 0.4 + restore 0.3 + replay 5 steps 0.5)
  alternatives: step 40 7.2s, step 30 8.3s, step 20 (durable) 11.9s
  rank 0 on node-0 from local@node-0: 7 checks ok
  rank 1 on node-2 from peer@node-0: 7 checks ok
```

The demo prints the incident at the end. `uv run aegis incident <id>` shows it again later, with the plan, every alternative and why it lost, and the restore report.

### Scenarios

```bash
uv run aegis demo <scenario> [--planner cost|newest|durable-only]
```

| Scenario | What it exercises |
| --- | --- |
| `worker-loss` | a rank crashes; resume from the last checkpoint, read straight off local disks |
| `incomplete-checkpoint` | the job dies mid-save; no manifest was committed, so that step is invisible |
| `corrupt-shard` | rank 1's local copy rots; the same step is restored from its peer replica |
| `corrupt-everywhere` | every copy of a shard is corrupt; Aegis falls back a checkpoint |
| `node-loss` | a node dies with its disks; the rank moves, its shard comes from a peer |
| `node-loss-no-peers` | as above without peer replicas: recovery reaches into durable storage |
| `durable-outage` | object storage is down the whole run, then a node dies; peers carry the recovery |
| `stalled-worker` | a rank hangs; nothing crashes, so only the progress watcher sees it |
| `failure-during-recovery` | the fault recurs on the first recovery attempt |
| `restore-verification` | a restore comes back with the wrong data position; it is rejected and that checkpoint is never used again |
| `repeated-fault` | the fault returns every attempt; Aegis stops at the budget |

### Experiments

```bash
uv run aegis experiment                                   # all scenarios, 3 planners, 2 trials (~25 min)
uv run aegis experiment --scenarios baseline,node-loss,storage-degraded --trials 3
```

Each trial runs in its own state directory with the controller as a separate process, so the `controller-crash` scenario can SIGKILL it mid-recovery and start a new one. Results go to `experiments/<timestamp>/results.json` and a readable `report.md`. See [Measured results](#measured-results).

### Driving it by hand

```bash
uv run aegis submit --name myjob --total-steps 400 --fault "crash:45:1"
uv run aegis controller                     # Ctrl-C to stop; restart it any time

uv run aegis node lose node-1 --job <id>    # take a node and its disks away
uv run aegis tier degrade peer --job <id> --factor 0.05   # or --unavailable
uv run aegis status
uv run aegis checkpoints <job-id> --lost node-1   # what is recoverable if node-1 is gone?
uv run aegis incident 1
```

## How it works

### Checkpoints live in three tiers

| Tier | Where | Speed (scaled profile) | Survives |
| --- | --- | --- | --- |
| local | the node the rank runs on | 20 MB/s | process crashes |
| peer | the next node in the ring (`rank r -> node (r+1) % n`) | 1 MB/s + 50 ms | loss of one node |
| durable | object storage | 120 KB/s + 250 ms | anything short of losing the bucket |

A save is collective. Every rank writes its shard locally and fsyncs it, the ranks exchange SHA-256 digests, rank 0 builds one manifest and broadcasts it, and every rank publishes it next to its shard with an atomic rename. That blocking part takes about 1% of training time. Replication to peers and to durable storage happens in background lanes, so a slow upload never stalls training. Durable uploads happen every Nth checkpoint and skip a step if a newer one is already queued. The rank that finishes the last durable upload commits the durable manifest.

The manifest is the integrity reference for every copy. It records, per rank, the shard's size and checksum and digests of model, optimizer, LR schedule, RNG and data position.

The bandwidths are scaled down so that a 330 KB toy shard behaves like a multi-GB production shard relative to step time. `AEGIS_STORAGE_PROFILE=unthrottled` turns the model off.

### Which checkpoints are still reachable

After a failure the catalog surveys every tier on every surviving node, plus durable storage. For each step, it pools all copies of each rank's shard and verifies each one against the manifest. A step is *recoverable* only if every rank has at least one intact copy that is reachable. Copies on lost nodes, on unavailable tiers, or that fail size or checksum checks don't count. Manifests that disagree between tiers are treated as a conflict, not resolved by guessing. Rejections carry a reason (`missing_shard`, `checksum_mismatch`, `restore_verification_failed`, ...) that ends up on the incident.

### Choosing how to recover

The planner prices each candidate as

```
scheduling (startup + fresh node) + max over ranks of transfer time + restore time + replay time
```

It evaluates the shard-source/node placement for each candidate step. It brute-forces small pools and uses a greedy search for large ones. It knows a shard already on the target node costs nothing to move, and it uses each tier's current degradation. Replay is priced with the job's measured step time.

The `cost` strategy minimises that total. `newest` and `durable-only` are there as baselines to compare against. If there are not enough healthy nodes, the plan is *blocked*: the job waits without spending a recovery attempt. Restarting from scratch only happens when no checkpoint exists at all.

The plan names, per rank, which copy to read and which node to run on. The controller records it on the incident, along with the alternatives and rejected steps, before launching anything.

### Restore verification

Each rank stages its own shard from the planned copy, re-checks its SHA-256, loads it, and then recomputes digests of what is now in memory. Those are compared with the manifest: global model, optimizer, LR scheduler and step, plus per-rank RNG state and data position. The ranks also cross-check that replicated state is identical across them. The result is a `restore_report.json`.

Only after a `restored` report, new training steps, and a freshly committed checkpoint does the controller call the recovery verified. A `verification_failed` restore puts that step on the job's exclusion list for good; a `staging_failed` restore is treated as storage trouble and retried.

### The state machine survives its own controller

```
PENDING ──launch──> RUNNING ──signals──> SUSPECT ──correlation window──> RECOVERING
                       ^                    │                                │ survey, plan,
                       │                    └──progress resumed──> RUNNING   │ place, launch
                       └──restored + steps + new checkpoint── VERIFYING <────┘
                                                              │
                                     attempt limit / deadline │
                                                              v
                                                          EXHAUSTED
```

One `tick()` is a complete, idempotent step for one job, and everything it depends on is in the database:

- job state, placement, excluded checkpoints and measured step time;
- node health, which is level-triggered: a node recorded as unhealthy keeps jobs off it even if the controller that saw the event is gone;
- an event watermark, so a restarted controller does not re-apply old events;
- the incident timeline (`detected -> recovering -> planned -> launched -> restored -> training_resumed -> caught_up -> verified`).

Duplicate launches are prevented structurally:

- a recovery lease with a fence token;
- deterministic action IDs;
- a ledger written before the side effect;
- the backend asked what actually exists before every launch.

A controller that dies mid-launch is replaced by one that *adopts* the orphaned run, keeping its plan and placement, rather than starting a second copy. `controller-crash` in the experiment checks this end to end.

## Measured results

From `aegis experiment --trials 1 --total-steps 120` on an M-series laptop (CPU, 2 workers, checkpoint every 10 steps, durable every 20). The full report is in [`experiments/`](experiments/).

The uninterrupted reference run reaches its final eval loss in 20.7 s and uses 32 worker-seconds. Times below are measured from the moment the controller detected the failure.

| scenario | planner | outcome | restored from | resume after | verified after | replayed steps | restore est / measured | time to target | state identical |
|---|---|---|---|---|---|---|---|---|---|
| worker-loss | cost | recovered | step 40, local | 5.7 s | 7.8 s | 5 | 0.27 / 0.27 s | +7.5 s | yes |
| incomplete-checkpoint | cost | recovered | step 40, local | 6.3 s | 8.9 s | 9 | 0.27 / 0.27 s | +5.5 s | yes |
| corrupt-shard | cost | recovered | step 50, peer | 5.7 s | 7.8 s | 0 | 0.28 / 0.29 s | +2.8 s | yes |
| node-loss | cost | recovered | step 50, local + peer | 6.2 s | 8.3 s | 5 | 0.64 / 0.65 s | +2.8 s | yes |
| node-loss | durable-only | recovered | step 20, durable | 8.8 s | 10.8 s | 35 | 3.21 / 3.23 s | +8.9 s | yes |
| node-loss-no-peers | cost | recovered | step 20, durable | 8.3 s | 10.4 s | 35 | 3.21 / 3.23 s | +9.4 s | yes |
| storage-degraded | **cost** | recovered | step 20, durable | **8.8 s** | 10.8 s | 35 | 3.21 / 3.23 s | **+8.6 s** | yes |
| storage-degraded | newest | recovered | step 50, slow peer | 22.4 s | 25.1 s | 5 | 16.47 / 16.50 s | +22.0 s | yes |
| durable-outage | cost | recovered | step 50, local + peer | 8.5 s | 10.6 s | 5 | 0.64 / 0.66 s | +5.7 s | yes |
| durable-outage | durable-only | **exhausted** | nothing reachable | - | - | - | - | - | - |
| failure-during-recovery | cost | recovered on attempt 2 | step 40, local | 5.6 s | 13.9 s | 10 | 0.27 / 0.27 s | +7.9 s | yes |
| restore-verification | cost | recovered on attempt 2 | step 30 (40 rejected) | 10.3 s | 12.3 s | 15 | 0.27 / 0.27 s | +7.7 s | yes |
| controller-crash | cost | recovered, 1 launch | step 40, local | 9.3 s | 11.3 s | 5 | 0.27 / 0.27 s | +5.6 s | yes |
| repeated-fault | cost | exhausted at budget | - | - | - | 15 | - | - | - |

These are single trials. Run-to-run noise in the latencies is a few seconds, mostly from backoff jitter and process startup, so compare columns with `--trials 3` or more before drawing fine conclusions.

What the numbers say:

- **Recovered training is bit-identical.** In every scenario that recovered, the final model, optimizer, LR schedule and RNG digests equal those of the uninterrupted run, and the per-step loss curve matches exactly. Recovery costs time, not correctness.
- **Most of the recovery latency is fixed cost:** detection (about 1 s), the correlation window, backoff and worker startup (about 4 s). Which tier the shards came from adds 0–3 s on top.
- **The planner earns its keep when storage misbehaves.** In `storage-degraded`, the peer network drops to 2% during recovery. `newest` still fetches step 50 over it and takes 22 s to resume. `cost` sees that, takes the older durable checkpoint and replays 35 steps instead, resuming in 9 s. When nothing is degraded, `cost` and `newest` agree. `durable-only` cannot recover at all when object storage is down.
- **The cost model is accurate here**, with estimated and measured restore times within 2%. That is partly by construction: the planner and the simulated storage share one model. What it does show is that staging, placement and per-rank parallel fetches behave the way the planner assumes.
- **Peer replicas are what make node loss cheap.** With peers, node loss resumes from the last checkpoint. Without them it falls back to the last *committed durable* checkpoint, 35 steps earlier.
- **Checkpoint overhead** (time training is blocked in saves) is about 1% with asynchronous replication.

## Testing

```bash
uv run pytest                  # 96 tests, ~20 s
uv run pytest -m "not slow"    # skip the tests that spawn real torchrun
```

The suite covers:

- **Checkpoints:** layout, ring replication, throttling, the collective writer, catalog verdicts for every rejection reason, and staging.
- **Planner:** its choices, placement, capacity blocking and degraded tiers.
- **Reconciler and crash safety:** correlation, budgets, deadlines, node health across restarts, restore-verification exclusion, adoption after a controller crash, and several regressions found while building it.
- **End to end with real torchrun:** crash and resume, and node loss restored from a peer, both bit-identical to an uninterrupted run, plus a skewed restore being rejected.

## Layout

```
src/aegis/
  checkpoint/   layout (tiers, nodes), tiers (storage model), writer (collective save + replication),
                catalog (reachability), restore (plans, staging, reports), manifest, state digests
  controller/   reconciler (state machine), planner, policies, progress watcher, loop
  experiments/  harness (trials, controller subprocess, fault timing), analysis (measurements, report)
  backends/     local subprocess, kubernetes
  store/        job state, node health, lease, action ledger, incidents
  faults.py     node loss, tier degradation
training/
  train_ddp.py  resumable DDP job: teacher-student regression, warmup+cosine LR, restore verification,
                fault injection, metrics log
```

## Known limits

- **CPU only, simulated cluster.** The "nodes" are directories on one machine and workers are local Gloo processes. Losing a node means losing its directories, and durable storage is a directory too. What this validates is the orchestration and recovery logic, not GPU fault handling or real network behaviour.
- **"GPU-hours" are CPU worker-hours** (busy time x world size) in the reports.
- **Storage speeds come from a scaled model**, not measurements of real storage. The planner's estimates match the simulation because both use the same model; against real storage, the profile would need to be calibrated.
- **Kubernetes:** `backends/k8s.py` passes the restore plan and placement to the job and excludes unhealthy nodes. It does not yet pin individual ranks to the planned nodes, and it has not been run against a live cluster.
- Restore verification proves the restored state equals what was saved. It cannot detect state that was already numerically wrong when it was saved.
- Fixed world size: a job needs as many healthy nodes as ranks. Elastic resharding is out of scope.
