"""Checkpoint validation: every incomplete checkpoint must be rejected."""

from __future__ import annotations

import json

import pytest

from aegis.checkpoint import CheckpointValidator, LocalCheckpointStore
from aegis.checkpoint.manifest import MANIFEST_NAME, shard_name
from aegis.checkpoint.store import FaultInjectingStore, StorageFault
from aegis.checkpoint.validator import Compatibility, RejectionReason


def test_complete_checkpoint_is_valid(store, checkpoints, spec):
    checkpoints.write(100)
    result = CheckpointValidator(store).validate(
        100, Compatibility(job_name=spec.name, world_size=spec.world_size)
    )
    assert result.valid, result.summary


def test_missing_manifest_is_rejected(store, checkpoints):
    checkpoints.write(100, skip_manifest=True)
    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.NO_MANIFEST in result.reasons


def test_missing_shard_is_rejected(store, checkpoints):
    checkpoints.write(100, drop_rank=1)
    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.UNEXPECTED_SHARD_COUNT in result.reasons


def test_shard_deleted_after_publish_is_rejected(store, checkpoints):
    checkpoints.write(100)
    (store.path_for(100) / shard_name(1)).unlink()
    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.MISSING_SHARD in result.reasons


def test_truncated_shard_is_rejected(store, checkpoints):
    checkpoints.write(100)
    target = store.path_for(100) / shard_name(0)
    with target.open("r+b") as fh:
        fh.truncate(target.stat().st_size // 2)
    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.SIZE_MISMATCH in result.reasons


def test_bit_flip_is_caught_by_checksum(store, checkpoints):
    checkpoints.write(100)
    target = store.path_for(100) / shard_name(0)
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))

    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.CHECKSUM_MISMATCH in result.reasons


def test_world_size_mismatch_is_rejected(store, checkpoints, spec):
    checkpoints.write(100)
    result = CheckpointValidator(store).validate(100, Compatibility(world_size=8))
    assert not result.valid
    assert RejectionReason.METADATA_INCOMPATIBLE in result.reasons


def test_unparseable_manifest_is_rejected(store, checkpoints):
    checkpoints.write(100)
    (store.path_for(100) / MANIFEST_NAME).write_text("{not json")
    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.MANIFEST_UNPARSEABLE in result.reasons


def test_unsupported_format_version_is_rejected(store, checkpoints):
    checkpoints.write(100)
    path = store.path_for(100) / MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload["format_version"] = 99
    path.write_text(json.dumps(payload))

    result = CheckpointValidator(store).validate(100)
    assert not result.valid
    assert RejectionReason.UNSUPPORTED_FORMAT in result.reasons


def test_failing_trial_restore_is_rejected(store, checkpoints):
    checkpoints.write(100)

    def refuse(_payload):
        raise RuntimeError("model does not accept these weights")

    result = CheckpointValidator(store, trial_restore=refuse).validate(100)
    assert not result.valid
    assert RejectionReason.RESTORE_FAILED in result.reasons
    assert "does not accept" in result.detail


def test_selection_falls_back_past_incomplete_checkpoints(store, checkpoints, spec):
    checkpoints.write(100)
    checkpoints.write(110)
    checkpoints.write(120, skip_manifest=True)
    checkpoints.write(130, drop_rank=1)

    chosen, rejected = CheckpointValidator(store).select_latest_valid(
        Compatibility(job_name=spec.name, world_size=spec.world_size)
    )
    assert chosen is not None
    assert chosen.step == 110
    assert [r.step for r in rejected] == [130, 120]


def test_selection_returns_nothing_when_all_are_bad(store, checkpoints):
    checkpoints.write(100, skip_manifest=True)
    checkpoints.write(110, skip_manifest=True)

    chosen, rejected = CheckpointValidator(store).select_latest_valid()
    assert chosen is None
    assert len(rejected) == 2


def test_storage_fault_propagates_instead_of_looking_like_absence(store, checkpoints):
    """A read error must not be mistaken for "no checkpoint here"."""
    checkpoints.write(100)
    flaky = FaultInjectingStore(store, fail_times=1)
    with pytest.raises(StorageFault):
        CheckpointValidator(flaky).select_latest_valid()


def test_manifest_is_the_commit_point(store, checkpoints, tmp_path):
    """Shards present without a manifest are invisible to selection."""
    checkpoints.write(100)
    checkpoints.write(200, skip_manifest=True)

    assert store.list_steps() == [200, 100]
    chosen, _ = CheckpointValidator(store).select_latest_valid()
    assert chosen is not None and chosen.step == 100


def test_round_trip_preserves_training_state(store, checkpoints):
    import torch

    checkpoints.write(100)
    payload = torch.load(
        store.path_for(100) / shard_name(1), map_location="cpu", weights_only=True
    )
    assert payload["step"] == 100
    assert payload["samples_consumed"] == 3200
    assert "rng_state" in payload
    assert "model" in payload
