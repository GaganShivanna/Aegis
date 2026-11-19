"""Cost-based recovery planning.

For every checkpoint that is still recoverable, and every way of placing the
ranks on healthy nodes, estimate how long it takes until the job is back
where it was when it failed:

    scheduling  bring the worker group up, plus extra if a node is new to the job
    transfer    stage each rank's shard onto its node (ranks stage in parallel)
    restore     deserialize and load state
    replay      recompute the steps between the checkpoint and the failure

and pick the cheapest. "Newest valid checkpoint" is often right, but not when
its only intact copy is in slow object storage and a slightly older one sits
on a surviving node's disk, or when the newest copy would force a rank onto a
fresh node. The planner makes that trade explicit and records it.

Two baseline strategies exist so the experiments can measure what the cost
model is worth: `newest` always takes the newest recoverable checkpoint (with
the best placement for it), `durable-only` behaves as if node-local tiers did
not exist.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

from aegis.checkpoint.catalog import CheckpointView, Replica
from aegis.checkpoint.layout import Tier
from aegis.checkpoint.restore import RestorePlan, ShardSource, transfer_tier
from aegis.checkpoint.tiers import StorageProfiles, TierCondition

STRATEGIES = ("cost", "newest", "durable-only")
DEFAULT_STEP_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class CostEstimate:
    scheduling_s: float
    transfer_s: float
    restore_s: float
    replay_s: float
    replay_steps: int

    @property
    def total_s(self) -> float:
        return self.scheduling_s + self.transfer_s + self.restore_s + self.replay_s


@dataclass(slots=True)
class Candidate:
    step: int | None
    placement: list[str]
    sources: list[ShardSource]
    cost: CostEstimate
    view: CheckpointView | None = None

    @property
    def label(self) -> str:
        return "scratch" if self.step is None else f"step {self.step}"

    def describe(self) -> str:
        c = self.cost
        spread = ", ".join(f"r{s.rank}<-{s.location}" for s in self.sources) or "no checkpoint"
        return (
            f"{self.label} [{spread}] on {','.join(self.placement)}: est {c.total_s:.1f}s "
            f"(sched {c.scheduling_s:.1f} + transfer {c.transfer_s:.1f} + restore {c.restore_s:.1f}"
            f" + replay {c.replay_steps} steps {c.replay_s:.1f})"
        )

    def to_record(self) -> dict:
        return {
            "step": self.step,
            "placement": self.placement,
            "sources": [asdict(s) for s in self.sources],
            "cost": {**asdict(self.cost), "total_s": self.cost.total_s},
        }


@dataclass(slots=True)
class RecoveryPlan:
    strategy: str
    chosen: Candidate | None
    alternatives: list[Candidate] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    reason: str = ""
    # Set when the plan cannot run right now for a reason that may clear
    # (not enough healthy nodes); retrying later costs no attempt.
    blocked: bool = False

    def restore_plan(self) -> RestorePlan | None:
        if self.chosen is None:
            return None
        manifest = self.chosen.view.manifest if self.chosen.view else None
        return RestorePlan(
            step=self.chosen.step,
            placement=list(self.chosen.placement),
            sources=list(self.chosen.sources),
            manifest_json=manifest.to_json() if manifest else None,
        )

    def to_record(self) -> dict:
        return {
            "strategy": self.strategy,
            "chosen": self.chosen.to_record() if self.chosen else None,
            "alternatives": [c.to_record() for c in self.alternatives[:5]],
            "rejected": self.rejected,
            "reason": self.reason,
            "blocked": self.blocked,
        }


class RecoveryPlanner:
    def __init__(self, profiles: StorageProfiles, *, strategy: str = "cost", max_placements: int = 5000) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown planner strategy {strategy!r}; known: {', '.join(STRATEGIES)}")
        self.profiles = profiles
        self.strategy = strategy
        self.max_placements = max_placements

    def plan(
        self,
        *,
        views: list[CheckpointView],
        world_size: int,
        healthy_nodes: list[str],
        previous_placement: list[str],
        step_at_failure: int,
        step_seconds: float | None,
        conditions: dict[Tier, TierCondition],
    ) -> RecoveryPlan:
        step_seconds = step_seconds if step_seconds and step_seconds > 0 else DEFAULT_STEP_SECONDS
        rejected = [v.to_record() for v in views if not v.recoverable]

        if len(healthy_nodes) < world_size:
            return RecoveryPlan(
                self.strategy,
                None,
                rejected=rejected,
                reason=f"insufficient capacity: need {world_size} healthy nodes, have {len(healthy_nodes)}",
                blocked=True,
            )

        candidates: list[Candidate] = []
        for view in views:
            if not view.recoverable:
                continue
            replicas = view.replicas
            if self.strategy == "durable-only":
                replicas = {
                    rank: [r for r in reps if r.location.tier is Tier.DURABLE]
                    for rank, reps in replicas.items()
                }
                if any(not reps for reps in replicas.values()):
                    rejected.append({**view.to_record(), "detail": "not in durable storage"})
                    continue
            candidate = self._best_placement(
                view, replicas, world_size, healthy_nodes, previous_placement,
                step_at_failure, step_seconds, conditions,
            )
            if candidate is None:
                rejected.append({**view.to_record(), "detail": "no placement can reach every shard"})
                continue
            candidates.append(candidate)

        if not views:
            # Nothing was ever committed anywhere reachable. Starting over is
            # the only option and loses nothing that still exists.
            candidates.append(self._scratch(world_size, healthy_nodes, previous_placement, step_at_failure, step_seconds))

        if not candidates:
            return RecoveryPlan(
                self.strategy,
                None,
                rejected=rejected,
                reason=f"no recoverable checkpoint among {len(views)} candidate(s)",
            )

        if self.strategy == "cost":
            ordered = sorted(candidates, key=lambda c: (round(c.cost.total_s, 3), -(c.step or -1)))
        else:
            ordered = sorted(candidates, key=lambda c: -(c.step or -1))
        chosen = ordered[0]
        return RecoveryPlan(
            self.strategy,
            chosen,
            alternatives=ordered[1:],
            rejected=rejected,
            reason=chosen.describe(),
        )

    # -------------------------------------------------------------- internals

    def _rank_cost(
        self, replica: Replica, node: str, conditions: dict[Tier, TierCondition]
    ) -> float:
        if replica.location.tier is Tier.LOCAL and replica.location.node == node:
            return 0.0
        tier = transfer_tier(replica.location, node)
        return self.profiles.transfer_seconds(tier, replica.size_bytes, conditions[tier])

    def _best_placement(
        self,
        view: CheckpointView,
        replicas: dict[int, list[Replica]],
        world_size: int,
        healthy: list[str],
        previous: list[str],
        step_at_failure: int,
        step_seconds: float,
        conditions: dict[Tier, TierCondition],
    ) -> Candidate | None:
        # options[rank][node] = (seconds, replica)
        options: list[dict[str, tuple[float, Replica]]] = []
        for rank in range(world_size):
            per_node: dict[str, tuple[float, Replica]] = {}
            for node in healthy:
                best = min(
                    ((self._rank_cost(r, node, conditions), r) for r in replicas.get(rank, [])),
                    key=lambda pair: pair[0],
                    default=None,
                )
                if best is not None and math.isfinite(best[0]):
                    per_node[node] = best
            if not per_node:
                return None
            options.append(per_node)

        previous_set = set(previous)

        def score(assignment: tuple[str, ...]) -> tuple[float, float, int]:
            transfer = max(options[r][n][0] for r, n in enumerate(assignment))
            fresh = any(n not in previous_set for n in assignment)
            moved = sum(1 for r, n in enumerate(assignment) if r >= len(previous) or previous[r] != n)
            total = transfer + (self.profiles.fresh_node_s if fresh else 0.0)
            return (round(total, 6), sum(options[r][n][0] for r, n in enumerate(assignment)), moved)

        best_assignment: tuple[str, ...] | None = None
        if math.perm(len(healthy), world_size) <= self.max_placements:
            for assignment in itertools.permutations(healthy, world_size):
                if any(n not in options[r] for r, n in enumerate(assignment)):
                    continue
                if best_assignment is None or score(assignment) < score(best_assignment):
                    best_assignment = assignment
        else:
            best_assignment = self._greedy(options, healthy)
        if best_assignment is None:
            return None

        placement = list(best_assignment)
        transfer = max(options[r][n][0] for r, n in enumerate(placement))
        sources = [
            ShardSource(
                rank=r,
                location=str(options[r][n][1].location),
                path=str(options[r][n][1].path),
                size_bytes=options[r][n][1].size_bytes,
            )
            for r, n in enumerate(placement)
        ]
        replay_steps = max(0, step_at_failure - view.step)
        largest = max(s.size_bytes for s in sources)
        return Candidate(
            step=view.step,
            placement=placement,
            sources=sources,
            view=view,
            cost=CostEstimate(
                scheduling_s=self._scheduling(placement, previous_set),
                transfer_s=transfer,
                restore_s=self.profiles.restore_seconds(largest),
                replay_s=replay_steps * step_seconds,
                replay_steps=replay_steps,
            ),
        )

    @staticmethod
    def _greedy(options: list[dict[str, tuple[float, Replica]]], healthy: list[str]) -> tuple[str, ...] | None:
        assignment: dict[int, str] = {}
        used: set[str] = set()
        for rank in sorted(range(len(options)), key=lambda r: len(options[r])):
            choices = sorted((cost, node) for node, (cost, _) in options[rank].items() if node not in used)
            if not choices:
                return None
            assignment[rank] = choices[0][1]
            used.add(choices[0][1])
        return tuple(assignment[r] for r in range(len(options)))

    def _scheduling(self, placement: list[str], previous: set[str]) -> float:
        fresh = any(n not in previous for n in placement)
        return self.profiles.startup_s + (self.profiles.fresh_node_s if fresh else 0.0)

    def _scratch(
        self, world_size: int, healthy: list[str], previous: list[str], step_at_failure: int, step_seconds: float
    ) -> Candidate:
        keep = [n for n in previous if n in healthy][:world_size]
        placement = keep + [n for n in healthy if n not in keep][: world_size - len(keep)]
        return Candidate(
            step=None,
            placement=placement,
            sources=[],
            cost=CostEstimate(
                scheduling_s=self._scheduling(placement, set(previous)),
                transfer_s=0.0,
                restore_s=0.0,
                replay_s=step_at_failure * step_seconds,
                replay_steps=step_at_failure,
            ),
        )
