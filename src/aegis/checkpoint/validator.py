"""Checkpoint validation and selection.

A checkpoint is usable only if it clears every gate below. Anything short of
that is rejected with a recorded reason and the selector falls back to an
older checkpoint.

This catches truncation, missing shards, bit rot and structural mismatch. It
does not and cannot catch silent numerical corruption in otherwise
well-formed tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

import torch

from aegis.checkpoint.manifest import (
    MANIFEST_NAME,
    SUPPORTED_FORMAT_VERSIONS,
    CheckpointManifest,
    sha256_file,
    shard_name,
)
from aegis.checkpoint.store import CheckpointStore, StorageFault

log = logging.getLogger(__name__)

# Given a fully deserialised rank-0 shard, raise if it cannot be restored.
TrialRestore = Callable[[dict[str, Any]], None]


class RejectionReason(StrEnum):
    NO_MANIFEST = "no_manifest"
    MANIFEST_UNPARSEABLE = "manifest_unparseable"
    UNSUPPORTED_FORMAT = "unsupported_format_version"
    MISSING_SHARD = "missing_shard"
    UNEXPECTED_SHARD_COUNT = "unexpected_shard_count"
    SIZE_MISMATCH = "size_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    METADATA_INCOMPATIBLE = "metadata_incompatible"
    DESERIALIZE_FAILED = "deserialize_failed"
    RESTORE_FAILED = "restore_failed"
    STEP_MISMATCH = "step_mismatch"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    step: int
    valid: bool
    reasons: list[RejectionReason] = field(default_factory=list)
    detail: str = ""
    manifest: CheckpointManifest | None = None

    @property
    def summary(self) -> str:
        if self.valid:
            return f"step {self.step}: valid"
        reasons = ", ".join(self.reasons) or "unknown"
        return f"step {self.step}: rejected ({reasons}){f' - {self.detail}' if self.detail else ''}"


@dataclass(frozen=True, slots=True)
class Compatibility:
    """What the job we are about to launch requires of a checkpoint."""

    job_name: str | None = None
    world_size: int | None = None
    model_signature: str | None = None
    optimizer_class: str | None = None
    max_step: int | None = None


class CheckpointValidator:
    def __init__(
        self,
        store: CheckpointStore,
        *,
        verify_checksums: bool = True,
        trial_restore: TrialRestore | None = None,
    ) -> None:
        self.store = store
        self.verify_checksums = verify_checksums
        self.trial_restore = trial_restore

    def validate(
        self, step: int, compatibility: Compatibility | None = None
    ) -> ValidationResult:
        directory = self.store.path_for(step)
        compat = compatibility or Compatibility()

        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            return ValidationResult(step, False, [RejectionReason.NO_MANIFEST])

        try:
            manifest = CheckpointManifest.from_json(manifest_path.read_text())
        except StorageFault:
            raise
        except Exception as exc:
            return ValidationResult(
                step, False, [RejectionReason.MANIFEST_UNPARSEABLE], str(exc)
            )

        reasons: list[RejectionReason] = []
        detail = ""

        if manifest.format_version not in SUPPORTED_FORMAT_VERSIONS:
            return ValidationResult(
                step,
                False,
                [RejectionReason.UNSUPPORTED_FORMAT],
                f"format_version={manifest.format_version}",
                manifest,
            )

        if manifest.step != step:
            reasons.append(RejectionReason.STEP_MISMATCH)
            detail = f"manifest step {manifest.step} != directory step {step}"

        # Every expected shard must be declared and present.
        declared = {entry.rank for entry in manifest.shards}
        expected = manifest.expected_ranks
        if declared != expected:
            missing = sorted(expected - declared)
            return ValidationResult(
                step,
                False,
                [RejectionReason.UNEXPECTED_SHARD_COUNT],
                f"manifest declares ranks {sorted(declared)}, expected {sorted(expected)}"
                + (f"; missing {missing}" if missing else ""),
                manifest,
            )

        for entry in manifest.shards:
            path = directory / entry.name
            if not path.is_file():
                reasons.append(RejectionReason.MISSING_SHARD)
                detail = f"shard {entry.name} absent"
                continue
            if path.stat().st_size != entry.size_bytes:
                reasons.append(RejectionReason.SIZE_MISMATCH)
                detail = (
                    f"shard {entry.name}: {path.stat().st_size} bytes on disk, "
                    f"manifest says {entry.size_bytes}"
                )
                continue
            if self.verify_checksums and sha256_file(path) != entry.sha256:
                reasons.append(RejectionReason.CHECKSUM_MISMATCH)
                detail = f"shard {entry.name} checksum mismatch"

        if reasons:
            return ValidationResult(step, False, reasons, detail, manifest)

        if incompat := _incompatibilities(manifest, compat):
            return ValidationResult(
                step,
                False,
                [RejectionReason.METADATA_INCOMPATIBLE],
                "; ".join(incompat),
                manifest,
            )

        # Deserialize every shard. This is what catches a file that is the
        # right length and hashes correctly but was never valid torch data,
        # and more usefully, confirms the payload still loads under the
        # current torch version.
        rank0_payload: dict[str, Any] | None = None
        for entry in manifest.shards:
            path = directory / entry.name
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except StorageFault:
                raise
            except Exception as exc:
                return ValidationResult(
                    step,
                    False,
                    [RejectionReason.DESERIALIZE_FAILED],
                    f"{entry.name}: {exc}",
                    manifest,
                )
            if entry.rank == 0:
                rank0_payload = payload

        if self.trial_restore is not None and rank0_payload is not None:
            try:
                self.trial_restore(rank0_payload)
            except StorageFault:
                raise
            except Exception as exc:
                return ValidationResult(
                    step,
                    False,
                    [RejectionReason.RESTORE_FAILED],
                    str(exc),
                    manifest,
                )

        return ValidationResult(step, True, [], "", manifest)

    def select_latest_valid(
        self,
        compatibility: Compatibility | None = None,
        *,
        max_candidates: int = 10,
    ) -> tuple[ValidationResult | None, list[ValidationResult]]:
        """Newest valid checkpoint, plus the rejected candidates above it.

        The rejected list is what gets recorded on the incident, so an
        operator can see exactly why Aegis skipped back.
        """
        rejected: list[ValidationResult] = []
        for step in self.store.list_steps()[:max_candidates]:
            result = self.validate(step, compatibility)
            if result.valid:
                return result, rejected
            log.info("checkpoint %s", result.summary)
            rejected.append(result)
        return None, rejected


def _incompatibilities(
    manifest: CheckpointManifest, compat: Compatibility
) -> list[str]:
    problems = []
    meta = manifest.metadata
    if compat.job_name is not None and meta.job_name != compat.job_name:
        problems.append(f"job_name {meta.job_name!r} != {compat.job_name!r}")
    if compat.world_size is not None and meta.world_size != compat.world_size:
        problems.append(f"world_size {meta.world_size} != {compat.world_size}")
    if (
        compat.model_signature is not None
        and meta.model_signature != compat.model_signature
    ):
        problems.append("model structure differs from the job being launched")
    if (
        compat.optimizer_class is not None
        and meta.optimizer_class != compat.optimizer_class
    ):
        problems.append(
            f"optimizer {meta.optimizer_class!r} != {compat.optimizer_class!r}"
        )
    if compat.max_step is not None and manifest.step > compat.max_step:
        problems.append(f"step {manifest.step} is ahead of required {compat.max_step}")
    return problems


def load_shard(directory: Path, rank: int) -> dict[str, Any]:
    return torch.load(
        directory / shard_name(rank), map_location="cpu", weights_only=True
    )
