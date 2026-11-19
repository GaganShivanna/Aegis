"""Tiered checkpoints: writing, replication, recoverability, staging."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest
import torch

from aegis.checkpoint import (
    CheckpointCatalog,
    CheckpointManifest,
    Collective,
    Compatibility,
    Layout,
    Location,
    RejectionReason,
    ReplicaState,
    ReplicationPolicy,
    RestorePlan,
    ShardSource,
    StagingError,
    StorageFault,
    Tier,
    TieredCheckpointWriter,
    peer_of,
    stage_rank,
)
from aegis.checkpoint.manifest import MANIFEST_NAME, shard_name
from aegis.checkpoint.state import compare, digest, global_digests, rank_digests
from aegis.checkpoint.tiers import SCALED, UNTHROTTLED, TierProfile, set_condition, throttled_copy
from aegis.telemetry import MetricsLog, read_events

LOCAL0 = Location(Tier.LOCAL, "node-0")
LOCAL1 = Location(Tier.LOCAL, "node-1")
PEER0 = Location(Tier.PEER, "node-0")  # holds rank 1's replica
PEER1 = Location(Tier.PEER, "node-1")  # holds rank 0's replica
DURABLE = Location(Tier.DURABLE)


# -------------------------------------------------------------- digests


def test_digest_is_stable_and_sensitive():
    model = torch.nn.Linear(4, 2)
    first = digest(model.state_dict())
    assert digest(model.state_dict()) == first
    with torch.no_grad():
        model.weight[0, 0] += 1e-6
    assert digest(model.state_dict()) != first


def test_digest_distinguishes_structure_from_content():
    assert digest([1, 2]) != digest((1, 2))
    assert digest({"a": 1}) != digest({"a": 1.0})
    assert digest(torch.zeros(2, dtype=torch.float32)) != digest(torch.zeros(2, dtype=torch.float64))


def test_compare_treats_missing_expected_as_mismatch():
    actual = rank_digests(rng_state=torch.get_rng_state(), samples_consumed=64)
    assert compare(actual, actual) == {"rng": True, "data_position": True}
    assert compare({"rng": actual["rng"]}, actual) == {"rng": True, "data_position": False}


def test_optimizer_and_scheduler_digests_survive_a_save_load_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (s + 1))
    model(torch.randn(3, 4)).sum().backward()
    opt.step()
    sched.step()
    before = global_digests(
        model_state=model.state_dict(), optimizer_state=opt.state_dict(), scheduler_state=sched.state_dict(), step=1
    )
    torch.save({"m": model.state_dict(), "o": opt.state_dict(), "s": sched.state_dict()}, tmp_path / "x.pt")

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0 / (s + 1))
    loaded = torch.load(tmp_path / "x.pt", weights_only=True)
    model2.load_state_dict(loaded["m"])
    opt2.load_state_dict(loaded["o"])
    sched2.load_state_dict(loaded["s"])
    after = global_digests(
        model_state=model2.state_dict(), optimizer_state=opt2.state_dict(), scheduler_state=sched2.state_dict(), step=1
    )
    assert after == before


# --------------------------------------------------------------- layout


def test_layout_paths_and_node_loss(layout, checkpoints):
    checkpoints.write(10)
    assert layout.known_nodes() == ["node-0", "node-1"]
    assert layout.steps_in(LOCAL0) == [10]
    assert (layout.dir_for(PEER0, 10) / shard_name(1)).is_file(), "rank 1 replicates to node-0"

    moved = layout.lose_node("node-1")
    assert moved is not None and moved.exists()
    assert layout.known_nodes() == ["node-0"]
    assert layout.lose_node("node-1") is None


def test_ring_replication_places_each_shard_on_another_node():
    placement = ["a", "b", "c"]
    assert [peer_of(r, placement) for r in range(3)] == ["b", "c", "a"]
    assert peer_of(0, ["a"]) is None


def test_location_round_trips_through_its_string_form():
    for location in (LOCAL0, PEER1, DURABLE):
        assert Location.parse(str(location)) == location


# ----------------------------------------------------------------- tiers


def test_throttled_copy_takes_as_long_as_the_tier_allows(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 200_000)
    profiles = replace(SCALED, peer=TierProfile(1_000_000, 0.0))
    seconds = throttled_copy(src, tmp_path / "out" / "dst.bin", tier=Tier.PEER, profiles=profiles)
    assert seconds >= 0.18
    assert (tmp_path / "out" / "dst.bin").read_bytes() == src.read_bytes()


def test_unavailable_tier_and_vanished_source_are_storage_faults(tmp_path, layout):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    set_condition(layout, Tier.DURABLE, available=False)
    from aegis.checkpoint.tiers import read_conditions

    with pytest.raises(StorageFault):
        throttled_copy(src, tmp_path / "d", tier=Tier.DURABLE, profiles=UNTHROTTLED,
                       condition=read_conditions(layout)[Tier.DURABLE])
    with pytest.raises(StorageFault):
        throttled_copy(tmp_path / "gone.bin", tmp_path / "d2", tier=Tier.PEER, profiles=UNTHROTTLED)
    assert not list(tmp_path.glob(".*.tmp")), "failed copies leave no partial files"


def test_degraded_tier_is_priced_accordingly():
    from aegis.checkpoint.tiers import TierCondition

    normal = SCALED.transfer_seconds(Tier.DURABLE, 120_000)
    slow = SCALED.transfer_seconds(Tier.DURABLE, 120_000, TierCondition(bandwidth_factor=0.1))
    assert slow > normal * 5
    assert SCALED.transfer_seconds(Tier.DURABLE, 1, TierCondition(available=False)) == float("inf")


# ---------------------------------------------------------------- writer


class ThreadCollective:
    """Collective operations for ranks running as threads in one process."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots: list = [None] * world_size

    def for_rank(self, rank: int) -> Collective:
        def all_gather(obj):
            self.slots[rank] = obj
            self.barrier.wait()
            out = list(self.slots)
            self.barrier.wait()
            return out

        def broadcast(obj):
            if rank == 0:
                self.slots[0] = obj
            self.barrier.wait()
            value = self.slots[0]
            self.barrier.wait()
            return value

        return Collective(rank, self.world_size, self.barrier.wait, all_gather, broadcast)


def save_from_threads(layout, checkpoints, steps, *, replication=None, metrics=None, skip_manifest_at=None):
    group = ThreadCollective(2)
    writers = [
        TieredCheckpointWriter(
            layout, group.for_rank(r), placement=["node-0", "node-1"], job_name="unit",
            replication=replication or ReplicationPolicy(durable_interval_steps=20),
            profiles=UNTHROTTLED, metrics=metrics,
        )
        for r in range(2)
    ]
    errors = []

    def run(rank):
        try:
            for step in steps:
                writers[rank].save(
                    step=step,
                    payload=checkpoints.payload(step, rank),
                    metadata=checkpoints.metadata(),
                    global_digests={"model": "m", "optimizer": "o", "scheduler": "s", "step": str(step)},
                    rank_digests={"rng": f"rng{rank}", "data_position": f"pos{rank}"},
                    samples_consumed=step * 32,
                    skip_manifest=step == skip_manifest_at,
                )
            writers[rank].close(timeout=10)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors


def test_writer_commits_locally_and_replicates_to_peer_and_durable(layout, checkpoints, tmp_path):
    metrics = MetricsLog(tmp_path / "m.jsonl")
    save_from_threads(layout, checkpoints, [10, 20], metrics=metrics)

    for location, rank in ((LOCAL0, 0), (LOCAL1, 1), (PEER1, 0), (PEER0, 1)):
        assert (layout.dir_for(location, 10) / shard_name(rank)).is_file(), location
        assert (layout.dir_for(location, 10) / MANIFEST_NAME).is_file(), location
    assert layout.steps_in(DURABLE) == [20], "only steps on the durable interval are uploaded"
    assert (layout.dir_for(DURABLE, 20) / MANIFEST_NAME).is_file(), "durable copy committed once both ranks uploaded"

    manifest = CheckpointManifest.load(layout.dir_for(LOCAL0, 20))
    assert [s.node for s in manifest.shards] == ["node-0", "node-1"]
    assert manifest.rank_state_for(1).data_position == "pos1"
    events = {e["event"] for e in read_events(tmp_path / "m.jsonl")}
    assert {"checkpoint", "replicated", "durable_committed"} <= events


def test_crash_mid_checkpoint_leaves_shards_but_no_commit(layout, checkpoints):
    save_from_threads(layout, checkpoints, [10, 20], skip_manifest_at=20)
    assert (layout.dir_for(LOCAL0, 20) / shard_name(0)).is_file()
    assert not (layout.dir_for(LOCAL0, 20) / MANIFEST_NAME).exists()
    view = CheckpointCatalog(layout).survey()[0]
    assert view.step == 20 and RejectionReason.NO_MANIFEST in view.reasons


def test_durable_outage_does_not_block_training_or_peers(layout, checkpoints, tmp_path):
    set_condition(layout, Tier.DURABLE, available=False)
    metrics = MetricsLog(tmp_path / "m.jsonl")
    save_from_threads(layout, checkpoints, [20], metrics=metrics)
    assert (layout.dir_for(PEER0, 20) / shard_name(1)).is_file()
    assert not (layout.dir_for(DURABLE, 20) / MANIFEST_NAME).exists()
    failed = [e for e in read_events(tmp_path / "m.jsonl") if e["event"] == "replication_failed"]
    assert {e["tier"] for e in failed} == {"durable"}


def test_local_retention_keeps_the_newest_checkpoints(layout, checkpoints):
    save_from_threads(layout, checkpoints, [10, 20, 30, 40, 50], replication=ReplicationPolicy(keep_local=3, keep_peer=2, durable_interval_steps=0))
    assert layout.steps_in(LOCAL0) == [50, 40, 30]
    assert layout.steps_in(PEER0) == [50, 40]


# --------------------------------------------------------------- catalog


def test_newest_checkpoint_is_recoverable_with_every_copy_listed(layout, checkpoints):
    checkpoints.write(10, durable=True)
    view = CheckpointCatalog(layout).survey()[0]
    assert view.recoverable
    assert view.tiers == {0: ["local@node-0", "peer@node-1", "durable"], 1: ["local@node-1", "peer@node-0", "durable"]}


def test_node_loss_is_survived_through_the_peer_replica(layout, checkpoints):
    checkpoints.write(10)
    view = CheckpointCatalog(layout).survey(lost_nodes={"node-1"})[0]
    assert view.recoverable
    assert view.tiers == {0: ["local@node-0"], 1: ["peer@node-0"]}


def test_without_replicas_a_node_loss_makes_the_checkpoint_unrecoverable(layout, checkpoints):
    checkpoints.write(10, peer=False)
    view = CheckpointCatalog(layout).survey(lost_nodes={"node-1"})[0]
    assert not view.recoverable
    assert view.missing_ranks == [1]
    assert RejectionReason.MISSING_SHARD in view.reasons


def test_corrupt_primary_falls_back_to_an_intact_replica(layout, checkpoints):
    checkpoints.write(10)
    checkpoints.corrupt(10, 1, LOCAL1)
    view = CheckpointCatalog(layout).survey()[0]
    assert view.recoverable
    assert [str(r.location) for r in view.replicas[1]] == ["peer@node-0"]
    assert [(r.rank, r.state) for r in view.bad_replicas] == [(1, ReplicaState.CHECKSUM_MISMATCH)]


def test_every_copy_corrupt_rejects_the_checkpoint_for_an_older_one(layout, checkpoints):
    checkpoints.write(10)
    checkpoints.write(20)
    checkpoints.corrupt(20, 1, LOCAL1)
    checkpoints.corrupt(20, 1, PEER0)
    chosen, rejected = CheckpointCatalog(layout).latest_recoverable()
    assert chosen.step == 10
    assert rejected[0].step == 20 and RejectionReason.CHECKSUM_MISMATCH in rejected[0].reasons


def test_truncated_replica_is_a_size_mismatch(layout, checkpoints):
    checkpoints.write(10, peer=False)
    path = layout.dir_for(LOCAL1, 10) / shard_name(1)
    path.write_bytes(path.read_bytes()[:100])
    view = CheckpointCatalog(layout).survey()[0]
    assert RejectionReason.SIZE_MISMATCH in view.reasons


def test_a_manifest_on_any_tier_vouches_for_shards_elsewhere(layout, checkpoints):
    """Durable shards uploaded before the durable commit are still usable,
    because the local manifest is the integrity reference for every copy."""
    checkpoints.write(10, durable=True)
    (layout.dir_for(DURABLE, 10) / MANIFEST_NAME).unlink()
    view = CheckpointCatalog(layout).survey(lost_nodes={"node-1"})[0]
    assert view.recoverable
    assert "durable" in view.tiers[1]


def test_conflicting_manifests_for_one_step_are_not_trusted(layout, checkpoints):
    checkpoints.write(10)
    other = CheckpointManifest.load(layout.dir_for(PEER1, 10))
    tampered = replace(other, shards=[replace(other.shards[0], sha256="0" * 64), other.shards[1]])
    (layout.dir_for(PEER1, 10) / MANIFEST_NAME).write_text(tampered.to_json())
    view = CheckpointCatalog(layout).survey()[0]
    assert RejectionReason.MANIFEST_CONFLICT in view.reasons


def test_incompatible_checkpoint_is_rejected(layout, checkpoints):
    checkpoints.write(10)
    view = CheckpointCatalog(layout).survey(compatibility=Compatibility(world_size=4))[0]
    assert RejectionReason.INCOMPATIBLE in view.reasons


def test_unavailable_tier_is_left_out_of_the_survey(layout, checkpoints):
    checkpoints.write(10, peer=False, durable=True)
    set_condition(layout, Tier.DURABLE, available=False)
    view = CheckpointCatalog(layout).survey(lost_nodes={"node-1"})[0]
    assert not view.recoverable


def test_checkpoint_that_failed_restore_verification_is_excluded(layout, checkpoints):
    checkpoints.write(10)
    checkpoints.write(20)
    chosen, rejected = CheckpointCatalog(layout).latest_recoverable(exclude_steps={20})
    assert chosen.step == 10
    assert RejectionReason.RESTORE_VERIFICATION_FAILED in rejected[0].reasons


def test_has_committed_since_ignores_uncommitted_shards(layout, checkpoints):
    checkpoints.write(10)
    checkpoints.write(20, skip_manifest=True)
    catalog = CheckpointCatalog(layout)
    assert not catalog.has_committed_since(10)
    checkpoints.write(30)
    assert catalog.has_committed_since(10)


# ---------------------------------------------------------------- staging


def plan_for(layout, step, sources, placement):
    manifest = CheckpointManifest.load(layout.dir_for(sources[0][1], step))
    return RestorePlan(
        step=step,
        placement=placement,
        sources=[
            ShardSource(rank, str(loc), str(layout.dir_for(loc, step) / shard_name(rank)), 0)
            for rank, loc in sources
        ],
        manifest_json=manifest.to_json(),
    )


def test_staging_copies_a_replica_to_a_fresh_node_and_verifies_it(layout, checkpoints):
    checkpoints.write(10)
    layout.lose_node("node-1")
    plan = plan_for(layout, 10, [(0, LOCAL0), (1, PEER0)], ["node-0", "node-2"])

    directory, result = stage_rank(layout, plan, 1, UNTHROTTLED)
    assert directory == layout.dir_for(Location(Tier.LOCAL, "node-2"), 10)
    assert (directory / shard_name(1)).is_file() and (directory / MANIFEST_NAME).is_file()
    assert result.copied and result.tier == "peer"

    _, local = stage_rank(layout, plan, 0, UNTHROTTLED)
    assert not local.copied, "a shard already on its node is verified in place"


def test_staging_rejects_a_source_that_no_longer_matches(layout, checkpoints):
    checkpoints.write(10)
    plan = plan_for(layout, 10, [(0, LOCAL0), (1, PEER0)], ["node-0", "node-2"])
    checkpoints.corrupt(10, 1, PEER0)
    with pytest.raises(StagingError):
        stage_rank(layout, plan, 1, UNTHROTTLED)


def test_staging_from_a_lost_node_is_a_storage_fault(layout, checkpoints):
    checkpoints.write(10)
    plan = plan_for(layout, 10, [(0, LOCAL0), (1, PEER0)], ["node-1", "node-2"])
    layout.lose_node("node-0")
    with pytest.raises(StorageFault):
        stage_rank(layout, plan, 1, UNTHROTTLED)


def test_staging_is_throttled_by_the_tier_it_crosses(layout, checkpoints):
    checkpoints.write(10, durable=True)
    plan = plan_for(layout, 10, [(0, LOCAL0), (1, DURABLE)], ["node-0", "node-2"])
    size = (layout.dir_for(DURABLE, 10) / shard_name(1)).stat().st_size
    profiles = replace(SCALED, durable=TierProfile(size / 0.3, 0.0))
    started = time.monotonic()
    _, result = stage_rank(layout, plan, 1, profiles)
    assert result.tier == "durable"
    assert time.monotonic() - started >= 0.25
