"""End-to-end checks that a restart continues training rather than restarting it.

Runs real torchrun processes. If resuming is exact, a run interrupted and
resumed lands on bit-identical model, optimizer, LR-schedule and RNG state to
an uninterrupted run. Anything less -- a missed RNG state, a reset data
position -- shows up here as a digest mismatch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.checkpoint import Layout, RestoreReport
from aegis.telemetry import read_events

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 2
TOTAL = 30

pytestmark = pytest.mark.slow


def run_training(
    root: Path,
    name: str,
    *,
    resume_step: int | None = None,
    attempt: int = 0,
    placement: str = "node-0,node-1",
    fault: str = "",
    expect_ok: bool = True,
    timeout: int = 240,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "AEGIS_CHECKPOINT_DIR": str(root / "ckpt"),
        "AEGIS_PROGRESS_FILE": str(run_dir / "progress.json"),
        "AEGIS_METRICS_FILE": str(run_dir / "metrics.jsonl"),
        "AEGIS_RUN_DIR": str(run_dir),
        "AEGIS_JOB_NAME": "resume-test",
        "AEGIS_RUN_ID": name,
        "AEGIS_ATTEMPT": str(attempt),
        "AEGIS_PLACEMENT": placement,
        "AEGIS_TOTAL_STEPS": str(TOTAL),
        "AEGIS_CHECKPOINT_INTERVAL": "10",
        "AEGIS_DURABLE_INTERVAL": "10",
        "AEGIS_STEP_DELAY": "0",
        "AEGIS_STORAGE_PROFILE": "unthrottled",
        "AEGIS_RESUME_STEP": "" if resume_step is None else str(resume_step),
        "AEGIS_FAULT": fault,
        "MASTER_ADDR": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
            f"--nproc_per_node={WORLD_SIZE}", "--max_restarts=0",
            "--master_addr=127.0.0.1", f"--master_port={_free_port()}",
            "training/train_ddp.py",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    if (result.returncode == 0) != expect_ok:
        pytest.fail(f"training exit {result.returncode} (expected ok={expect_ok}):\n{result.stdout}\n{result.stderr}")
    return run_dir


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def final_digests(run_dir: Path) -> dict:
    finals = [e for e in read_events(run_dir / "metrics.jsonl") if e["event"] == "final"]
    assert finals, "run did not finish"
    return finals[-1]["digests"]


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("reference")
    return final_digests(run_training(root, "reference"))


def test_resume_after_a_crash_is_bit_exact(tmp_path, reference):
    run_training(tmp_path, "crashed", fault="crash:15:1", expect_ok=False)
    resumed = run_training(tmp_path, "resumed", resume_step=10, attempt=1)

    report = RestoreReport.load(resumed / "restore_report.json")
    assert report.outcome == "restored" and report.consistent_across_ranks
    assert final_digests(resumed) == reference


def test_node_loss_restores_from_the_peer_replica(tmp_path, reference):
    run_training(tmp_path, "crashed", fault="crash:25:0", expect_ok=False)
    Layout(tmp_path / "ckpt").lose_node("node-1")

    resumed = run_training(tmp_path, "moved", resume_step=20, attempt=1, placement="node-0,node-2")
    report = RestoreReport.load(resumed / "restore_report.json")
    assert report.outcome == "restored"
    sources = {r.rank: r.staged["source"] for r in report.ranks}
    assert sources[1] == "peer@node-0", "rank 1's own disk is gone; its shard comes from its ring peer"
    assert final_digests(resumed) == reference


def test_a_restore_that_does_not_match_is_rejected(tmp_path):
    run_training(tmp_path, "crashed", fault="crash:15:1", expect_ok=False)
    resumed = run_training(tmp_path, "skewed", resume_step=10, attempt=1, fault="skew_restore:0@1", expect_ok=False)

    report = RestoreReport.load(resumed / "restore_report.json")
    assert report.outcome == "verification_failed"
    assert report.failed_checks()
