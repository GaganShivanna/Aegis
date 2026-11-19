# Aegis fault-injection experiment

- world size 2, 120 steps, step delay 0.1s, checkpoint every 10 steps, durable every 20
- trials per scenario: 1; planners: cost, newest, durable-only
- reference (uninterrupted) run: 20.72s to target; target eval loss 0.00259

Times are seconds measured from the moment the controller detected the failure.
`resume` = first training step after the restore; `caught up` = back at the step where the fault hit;
`verified` = recovery declared good. `worker-s` is CPU worker-seconds (busy time x world size).
`state identical` compares final model/optimizer/LR-schedule/RNG digests with the reference run.

## Recovery

| scenario | planner | ok | recovered | detect | resume | caught up | verified | replayed steps | restore est / measured | sources |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | cost | 1/1 | 0/0 | - | - | - | - | 0 | - / - | - |
| worker-loss | cost | 1/1 | 1/1 | 0.74 | 5.71 | 6.22 | 7.79 | 5 | 0.27 / 0.27 | local@node-0, local@node-1 |
| incomplete-checkpoint | cost | 1/1 | 1/1 | 0.88 | 6.32 | 7.35 | 8.88 | 9 | 0.27 / 0.27 | local@node-0, local@node-1 |
| corrupt-shard | cost | 1/1 | 1/1 | 1.09 | 5.74 | 5.74 | 7.80 | 0 | 0.28 / 0.29 | peer@node-0, peer@node-1 |
| node-loss | cost | 1/1 | 1/1 | 0.07 | 6.19 | 6.75 | 8.29 | 5 | 0.64 / 0.65 | local@node-0, peer@node-0 |
| node-loss | newest | 1/1 | 1/1 | 0.26 | 6.73 | 7.25 | 8.79 | 5 | 0.64 / 0.65 | local@node-0, peer@node-0 |
| node-loss | durable-only | 1/1 | 1/1 | 0.19 | 8.77 | - | 10.82 | 35 | 3.21 / 3.23 | durable |
| node-loss-no-peers | cost | 1/1 | 1/1 | 0.33 | 8.28 | - | 10.35 | 35 | 3.21 / 3.23 | durable |
| node-loss-no-peers | newest | 1/1 | 1/1 | 0.26 | 8.88 | - | 10.94 | 35 | 3.21 / 3.23 | durable |
| node-loss-no-peers | durable-only | 1/1 | 1/1 | 0.30 | 10.68 | - | 12.74 | 35 | 3.21 / 3.23 | durable |
| storage-degraded | cost | 1/1 | 1/1 | 0.10 | 8.75 | - | 10.83 | 35 | 3.21 / 3.23 | durable |
| storage-degraded | newest | 1/1 | 1/1 | 0.44 | 22.43 | 22.96 | 25.07 | 5 | 16.47 / 16.50 | local@node-0, peer@node-0 |
| storage-degraded | durable-only | 1/1 | 1/1 | 0.25 | 8.87 | - | 10.95 | 35 | 3.21 / 3.23 | durable |
| durable-outage | cost | 1/1 | 1/1 | 0.55 | 8.52 | 9.03 | 10.58 | 5 | 0.64 / 0.66 | local@node-0, peer@node-0 |
| durable-outage | newest | 1/1 | 1/1 | 0.30 | 5.15 | 5.67 | 7.72 | 5 | 0.64 / 0.65 | local@node-0, peer@node-0 |
| durable-outage | durable-only | 0/1 | 0/1 | 0.04 | - | - | - | 0 | - / - | - |
| failure-during-recovery | cost | 1/1 | 1/1 | 0.62 | 5.64 | 5.64 | 13.87 | 10 | 0.27 / 0.27 | local@node-0, local@node-1 |
| restore-verification | cost | 1/1 | 1/1 | 0.67 | 10.27 | 11.82 | 12.33 | 15 | 0.27 / 0.27 | local@node-0, local@node-1 |
| controller-crash | cost | 1/1 | 1/1 | 0.60 | 9.26 | 9.78 | 11.33 | 5 | 0.27 / 0.27 | local@node-0, local@node-1 |
| repeated-fault | cost | 0/1 | 0/1 | 0.47 | 5.20 | - | - | 15 | 0.27 / 0.27 | local@node-0, local@node-1 |

## Cost and training behaviour

| scenario | planner | time to target | vs reference | worker-s | checkpoint overhead | state identical | max loss diff |
|---|---|---|---|---|---|---|---|
| baseline | cost | 20.72 | +0.0s | 32 | 0.8% | 1/1 | 0.00 |
| worker-loss | cost | 28.26 | +7.5s | 43 | 0.9% | 1/1 | 0.00 |
| incomplete-checkpoint | cost | 26.18 | +5.5s | 40 | 0.7% | 1/1 | 0.00 |
| corrupt-shard | cost | 23.48 | +2.8s | 40 | 0.7% | 1/1 | 0.00 |
| node-loss | cost | 23.51 | +2.8s | 40 | 0.7% | 1/1 | 0.00 |
| node-loss | newest | 24.49 | +3.8s | 40 | 0.8% | 1/1 | 0.00 |
| node-loss | durable-only | 29.66 | +8.9s | 54 | 0.6% | 1/1 | 0.00 |
| node-loss-no-peers | cost | 30.07 | +9.4s | 54 | 0.5% | 1/1 | 0.00 |
| node-loss-no-peers | newest | 29.59 | +8.9s | 54 | 0.7% | 1/1 | 0.00 |
| node-loss-no-peers | durable-only | 31.53 | +10.8s | 54 | 0.6% | 1/1 | 0.00 |
| storage-degraded | cost | 29.29 | +8.6s | 72 | 0.5% | 1/1 | 0.00 |
| storage-degraded | newest | 42.75 | +22.0s | 94 | 1.2% | 1/1 | 0.00 |
| storage-degraded | durable-only | 31.43 | +10.7s | 72 | 0.4% | 1/1 | 0.00 |
| durable-outage | cost | 26.44 | +5.7s | 29 | 1.3% | 1/1 | 0.00 |
| durable-outage | newest | 22.93 | +2.2s | 29 | 0.7% | 1/1 | 0.00 |
| durable-outage | durable-only | - | - | 11 | 0.9% | 0/1 | 0.00 |
| failure-during-recovery | cost | 28.59 | +7.9s | 40 | 0.6% | 1/1 | 0.00 |
| restore-verification | cost | 28.41 | +7.7s | 36 | 0.8% | 1/1 | 0.00 |
| controller-crash | cost | 26.32 | +5.6s | 40 | 0.5% | 1/1 | 0.00 |
| repeated-fault | cost | - | - | 14 | 0.6% | 0/1 | 0.00 |

## Trials

- `baseline` / cost #0: **succeeded** in 25.6s, 1 launches, 0 controller restarts -- training completed
- `worker-loss` / cost #0: **succeeded** in 35.8s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 45, restored step 40 from local@node-0, local@node-1 after 1 attempt(s)
- `incomplete-checkpoint` / cost #0: **succeeded** in 33.0s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 49, restored step 40 from local@node-0, local@node-1 after 1 attempt(s)
- `corrupt-shard` / cost #0: **succeeded** in 30.3s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 49, restored step 50 from peer@node-1, peer@node-0 after 1 attempt(s)
- `node-loss` / cost #0: **succeeded** in 30.2s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 50 from local@node-0, peer@node-0 after 1 attempt(s)
- `node-loss` / newest #0: **succeeded** in 31.0s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 50 from local@node-0, peer@node-0 after 1 attempt(s)
- `node-loss` / durable-only #0: **succeeded** in 37.4s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `node-loss-no-peers` / cost #0: **succeeded** in 36.9s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `node-loss-no-peers` / newest #0: **succeeded** in 37.4s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `node-loss-no-peers` / durable-only #0: **succeeded** in 38.7s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `storage-degraded` / cost #0: **succeeded** in 46.6s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `storage-degraded` / newest #0: **succeeded** in 59.8s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 50 from local@node-0, peer@node-0 after 1 attempt(s)
- `storage-degraded` / durable-only #0: **succeeded** in 48.4s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 20 from durable, durable after 1 attempt(s)
- `durable-outage` / cost #0: **succeeded** in 27.8s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 50 from local@node-0, peer@node-0 after 1 attempt(s)
- `durable-outage` / newest #0: **succeeded** in 24.7s, 2 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 55, restored step 50 from local@node-0, peer@node-0 after 1 attempt(s)
- `durable-outage` / durable-only #0: **exhausted** in 11.3s, 1 launches, 0 controller restarts -- no recoverable checkpoint among 3 candidate(s)
  - incident: exhausted, failed at step 55, restored step None from - after 0 attempt(s)
- `failure-during-recovery` / cost #0: **succeeded** in 34.9s, 3 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 45, restored step 40 from local@node-0, local@node-1 after 2 attempt(s)
- `restore-verification` / cost #0: **succeeded** in 32.8s, 3 launches, 0 controller restarts -- training completed
  - incident: recovered, failed at step 45, restored step 30 from local@node-0, local@node-1 after 2 attempt(s)
- `controller-crash` / cost #0: **succeeded** in 32.7s, 2 launches, 1 controller restarts -- training completed
  - incident: recovered, failed at step 45, restored step 40 from local@node-0, local@node-1 after 1 attempt(s)
- `repeated-fault` / cost #0: **exhausted** in 28.5s, 4 launches, 0 controller restarts -- attempt limit reached (3)
  - incident: exhausted, failed at step 45, restored step 40 from local@node-0, local@node-1 after 3 attempt(s)
