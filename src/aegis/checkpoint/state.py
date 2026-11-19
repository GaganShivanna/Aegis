"""Training-state digests.

A digest is a stable fingerprint of a piece of training state. Digests are
recorded in the manifest at save time and recomputed after restore; if any of
them differ, the restored job is not the job that was saved, whatever the
checksums on the files say.

Global state (model, optimizer, LR schedule, step) is replicated across DDP
ranks and must agree everywhere. Per-rank state (RNG, data position) differs
by design and is checked rank by rank.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import torch

GLOBAL_COMPONENTS = ("model", "optimizer", "scheduler", "step")
RANK_COMPONENTS = ("rng", "data_position")


def digest(obj: Any) -> str:
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def _feed(h: "hashlib._Hash", obj: Any) -> None:
    # Every value is tagged and length-prefixed so that structurally
    # different inputs can never serialise to the same byte stream.
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(b"T")
        _feed_str(h, str(t.dtype))
        _feed_str(h, str(tuple(t.shape)))
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    elif isinstance(obj, dict):
        h.update(b"D" + len(obj).to_bytes(8, "little"))
        for key in sorted(obj, key=lambda k: (type(k).__name__, str(k))):
            _feed(h, key)
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update((b"L" if isinstance(obj, list) else b"U") + len(obj).to_bytes(8, "little"))
        for item in obj:
            _feed(h, item)
    elif isinstance(obj, bool):
        h.update(b"B1" if obj else b"B0")
    elif isinstance(obj, int):
        _feed_str(h, str(obj), tag=b"I")
    elif isinstance(obj, float):
        h.update(b"F" + struct.pack("<d", obj))
    elif isinstance(obj, str):
        _feed_str(h, obj)
    elif obj is None:
        h.update(b"N")
    else:
        _feed_str(h, repr(obj), tag=b"R")


def _feed_str(h: "hashlib._Hash", value: str, tag: bytes = b"S") -> None:
    raw = value.encode()
    h.update(tag + len(raw).to_bytes(8, "little") + raw)


def global_digests(
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    step: int,
) -> dict[str, str]:
    return {
        "model": digest(model_state),
        "optimizer": digest(optimizer_state),
        "scheduler": digest(scheduler_state or {}),
        "step": digest(step),
    }


def rank_digests(*, rng_state: torch.Tensor, samples_consumed: int) -> dict[str, str]:
    return {"rng": digest(rng_state), "data_position": digest(samples_consumed)}


def compare(expected: dict[str, str], actual: dict[str, str]) -> dict[str, bool]:
    """Component-by-component agreement. A component missing from `expected`
    counts as a mismatch: an unverifiable restore is not a verified one."""
    return {key: expected.get(key) == value for key, value in actual.items()}
