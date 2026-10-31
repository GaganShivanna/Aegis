"""End-to-end check that a restart continues training rather than restarting it.

Runs real torchrun processes. If resuming is exact, a run interrupted at step
10 and resumed to step 20 lands on bit-identical weights to an uninterrupted
run of 20 steps. Anything less -- a missed RNG state, a reset data position --
shows up here as a numeric difference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aegis.checkpoint import LocalCheckpointStore
from aegis.checkpoint.manifest import shard_name

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 2


def run_training(
    checkpoint_dir: Path,
    *,
    total_steps: int,
    resume_step: int | None = None,
    timeout: int = 240,
) -> None:
    env = {
        **os.environ,
        "AEGIS_CHECKPOINT_DIR": str(checkpoint_dir),
        "AEGIS_PROGRESS_FILE": str(checkpoint_dir.parent / "progress.json"),
        "AEGIS_JOB_NAME": "resume-test",
        "AEGIS_RUN_ID": f"run-{total_steps}-{resume_step}",
        "AEGIS_TOTAL_STEPS": str(total_steps),
        "AEGIS_CHECKPOINT_INTERVAL": "10",
        "AEGIS_STEP_DELAY": "0",
        "AEGIS_RESUME_STEP": "" if resume_step is None else str(resume_step),
        "MASTER_ADDR": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={WORLD_SIZE}",
            "--max_restarts=0",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            "training/train_ddp.py",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(f"training failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_model(checkpoint_dir: Path, step: int) -> dict[str, torch.Tensor]:
    path = LocalCheckpointStore(checkpoint_dir).path_for(step) / shard_name(0)
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


@pytest.mark.slow
def test_resumed_run_matches_uninterrupted_run(tmp_path):
    straight = tmp_path / "straight" / "ckpt"
    run_training(straight, total_steps=20)
    expected = load_model(straight, 20)

    interrupted = tmp_path / "interrupted" / "ckpt"
    run_training(interrupted, total_steps=10)
    run_training(interrupted, total_steps=20, resume_step=10)
    actual = load_model(interrupted, 20)

    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), (
            f"{name} differs after resume; training restarted instead of continuing"
        )


@pytest.mark.slow
def test_resume_restores_step_and_data_position(tmp_path):
    ckpt = tmp_path / "ckpt"
    run_training(ckpt, total_steps=10)
    run_training(ckpt, total_steps=20, resume_step=10)

    for rank in range(WORLD_SIZE):
        payload = torch.load(
            LocalCheckpointStore(ckpt).path_for(20) / shard_name(rank),
            map_location="cpu",
            weights_only=True,
        )
        assert payload["step"] == 20
        # 20 steps of 32 samples, counted continuously across the restart.
        assert payload["samples_consumed"] == 20 * 32
