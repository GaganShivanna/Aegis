"""Fault injection from outside the training job.

Faults the job can produce itself (crashes, hangs, torn or corrupt
checkpoints) are armed through AEGIS_FAULT. These are the ones that happen
*to* a job: a node and its disks vanishing, a storage tier slowing down or
going away.
"""

from __future__ import annotations

import os
from pathlib import Path

from aegis.backends.base import ExecutionBackend
from aegis.checkpoint.layout import Layout, Tier
from aegis.checkpoint.tiers import TierCondition, clear_conditions, set_condition
from aegis.events.synthetic import emit_event
from aegis.models import JobSpec, SignalKind
from aegis.store.repository import Repository, handle_from_json


def lose_node(
    repo: Repository,
    backend: ExecutionBackend,
    spec: JobSpec,
    node: str,
    *,
    events_path: str | os.PathLike[str],
) -> Path | None:
    """The node dies: its disks go, the health system reports it, and the
    worker group collapses because one member is gone.

    Locally every rank shares one process group, so the whole group is
    killed; on a real cluster the survivors would fail at their next
    collective, which is the same outcome a moment later.
    """
    moved = Layout(spec.checkpoint_dir).lose_node(node)
    emit_event(
        events_path,
        kind=SignalKind.NODE_UNHEALTHY,
        source="fault-injector",
        node=node,
        detail={"reason": "node lost with its local disks"},
    )
    for action in repo.live_launches(spec.job_id):
        if not action.handle:
            continue
        handle = handle_from_json(action.handle)
        if node in (handle.extra.get("placement") or []):
            try:
                backend.terminate(handle, grace_seconds=0.0)
            except Exception:
                pass
    return moved


def heal_node(node: str, *, events_path: str | os.PathLike[str]) -> None:
    emit_event(events_path, kind=SignalKind.NODE_HEALTHY, source="operator", node=node)


def degrade_tier(
    spec_or_layout: JobSpec | Layout,
    tier: Tier,
    *,
    bandwidth_factor: float | None = None,
    available: bool | None = None,
) -> TierCondition:
    layout = spec_or_layout if isinstance(spec_or_layout, Layout) else Layout(spec_or_layout.checkpoint_dir)
    return set_condition(layout, tier, available=available, bandwidth_factor=bandwidth_factor)


def restore_tiers(spec_or_layout: JobSpec | Layout) -> None:
    layout = spec_or_layout if isinstance(spec_or_layout, Layout) else Layout(spec_or_layout.checkpoint_dir)
    clear_conditions(layout)
