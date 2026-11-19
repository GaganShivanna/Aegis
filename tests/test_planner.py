"""Cost-based recovery planning."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aegis.checkpoint import CheckpointCatalog, Location, Tier
from aegis.checkpoint.tiers import SCALED, TierCondition, TierProfile
from aegis.controller.planner import RecoveryPlanner

NODES = ["node-0", "node-1", "node-2", "node-3"]
NORMAL = {tier: TierCondition() for tier in Tier}

# Shards in these tests are a few KB, so bandwidths are scaled down to make
# transfer times comparable to replay times, as they would be for a large model.
PROFILES = replace(
    SCALED,
    local=TierProfile(1e9, 0.0),
    peer=TierProfile(20_000, 0.0),
    durable=TierProfile(1_000, 0.0),
    restore_bytes_per_s=1e9,
    restore_fixed_s=0.0,
    startup_s=4.0,
    fresh_node_s=1.0,
)


def plan(layout, *, strategy="cost", lost=frozenset(), step_at_failure=55, step_seconds=0.1,
         previous=("node-0", "node-1"), healthy=None, conditions=None):
    healthy = healthy if healthy is not None else [n for n in NODES if n not in lost]
    views = CheckpointCatalog(layout).survey(lost_nodes=set(lost), conditions=conditions or NORMAL)
    return RecoveryPlanner(PROFILES, strategy=strategy).plan(
        views=views,
        world_size=2,
        healthy_nodes=healthy,
        previous_placement=list(previous),
        step_at_failure=step_at_failure,
        step_seconds=step_seconds,
        conditions=conditions or NORMAL,
    )


def test_all_local_picks_the_newest_checkpoint_in_place(layout, checkpoints):
    checkpoints.write(40)
    checkpoints.write(50)
    result = plan(layout)
    assert result.chosen.step == 50
    assert result.chosen.placement == ["node-0", "node-1"]
    assert result.chosen.cost.transfer_s == 0.0
    assert result.chosen.cost.replay_steps == 5
    assert [c.step for c in result.alternatives] == [40]


def test_node_loss_reads_the_peer_replica_onto_a_fresh_node(layout, checkpoints):
    checkpoints.write(50)
    layout.lose_node("node-1")
    result = plan(layout, lost={"node-1"})
    chosen = result.chosen
    assert chosen.step == 50
    assert "node-1" not in chosen.placement
    assert chosen.placement[0] == "node-0", "rank 0 stays where its shard is"
    sources = {s.rank: s.location for s in chosen.sources}
    assert sources == {0: "local@node-0", 1: "peer@node-0"}
    assert chosen.cost.scheduling_s == PROFILES.startup_s + PROFILES.fresh_node_s


def test_rank_is_placed_where_its_replica_already_is(layout, checkpoints):
    """Rank 1's only surviving copy is the peer replica on node-0. Rank 0's
    is also on node-0. One must cross the network; the planner should put
    the move on whichever rank makes the total cheapest, never both."""
    checkpoints.write(50)
    layout.lose_node("node-1")
    result = plan(layout, lost={"node-1"})
    assert sum(1 for s in result.chosen.sources if not s.location.endswith(result.chosen.placement[s.rank])) <= 1


def test_cheap_replay_beats_a_slow_download(layout, checkpoints):
    """Step 50 for rank 1 survives only in object storage; step 40 is on
    disks that survived. Replaying 15 cheap steps beats a slow download."""
    checkpoints.write(40, durable=False)
    checkpoints.write(50, peer=False, durable=True)
    # Make step 40 survive node-1's loss via its peer replica.
    layout.lose_node("node-1")

    result = plan(layout, lost={"node-1"}, step_seconds=0.1)
    assert result.chosen.step == 40
    durable_option = next(c for c in result.alternatives if c.step == 50)
    assert durable_option.cost.transfer_s > result.chosen.cost.total_s - result.chosen.cost.scheduling_s
    assert durable_option.cost.total_s > result.chosen.cost.total_s


def test_expensive_replay_justifies_the_slow_download(layout, checkpoints):
    checkpoints.write(40, durable=False)
    checkpoints.write(50, peer=False, durable=True)
    layout.lose_node("node-1")

    result = plan(layout, lost={"node-1"}, step_seconds=5.0)
    assert result.chosen.step == 50
    assert {s.rank: s.location for s in result.chosen.sources}[1] == "durable"


def test_newest_strategy_ignores_cost(layout, checkpoints):
    checkpoints.write(40, durable=False)
    checkpoints.write(50, peer=False, durable=True)
    layout.lose_node("node-1")
    result = plan(layout, strategy="newest", lost={"node-1"}, step_seconds=0.1)
    assert result.chosen.step == 50


def test_durable_only_strategy_cannot_see_node_local_copies(layout, checkpoints):
    checkpoints.write(40, durable=True)
    checkpoints.write(50)
    result = plan(layout, strategy="durable-only")
    assert result.chosen.step == 40
    assert all(s.location == "durable" for s in result.chosen.sources)
    assert any(r["step"] == 50 for r in result.rejected)


def test_degraded_durable_storage_shifts_the_choice(layout, checkpoints):
    checkpoints.write(40, durable=False)
    checkpoints.write(50, peer=False, durable=True)
    layout.lose_node("node-1")
    healthy = plan(layout, lost={"node-1"}, step_seconds=1.0)
    degraded = plan(
        layout, lost={"node-1"}, step_seconds=1.0,
        conditions={**NORMAL, Tier.DURABLE: TierCondition(bandwidth_factor=0.05)},
    )
    assert healthy.chosen.step == 50
    assert degraded.chosen.step == 40


def test_not_enough_healthy_nodes_blocks_without_failing(layout, checkpoints):
    checkpoints.write(50)
    result = plan(layout, healthy=["node-0"])
    assert result.blocked and result.chosen is None
    assert "capacity" in result.reason


def test_nothing_ever_committed_restarts_from_scratch(layout):
    result = plan(layout, step_at_failure=7)
    assert result.chosen.step is None
    assert result.chosen.cost.replay_steps == 7
    assert result.restore_plan().step is None


def test_checkpoints_present_but_unrecoverable_produce_no_plan(layout, checkpoints):
    checkpoints.write(50, peer=False)
    layout.lose_node("node-1")
    result = plan(layout, lost={"node-1"})
    assert result.chosen is None and not result.blocked
    assert result.rejected[0]["step"] == 50


def test_restore_plan_carries_the_manifest_and_sources(layout, checkpoints):
    manifest = checkpoints.write(50)
    restore = plan(layout).restore_plan()
    assert restore.step == 50
    assert restore.placement == ["node-0", "node-1"]
    assert '"step": 50' in restore.manifest_json
    assert [s.rank for s in restore.sources] == [0, 1]
    assert restore.sources[0].size_bytes == manifest.shards[0].size_bytes


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError):
        RecoveryPlanner(PROFILES, strategy="yolo")


def test_large_pools_fall_back_to_greedy_placement(layout, checkpoints):
    checkpoints.write(50)
    views = CheckpointCatalog(layout).survey()
    result = RecoveryPlanner(PROFILES, max_placements=1).plan(
        views=views, world_size=2, healthy_nodes=NODES, previous_placement=["node-0", "node-1"],
        step_at_failure=55, step_seconds=0.1, conditions=NORMAL,
    )
    assert result.chosen.step == 50
    assert len(set(result.chosen.placement)) == 2
    assert Location.parse(result.chosen.sources[0].location).tier is Tier.LOCAL
