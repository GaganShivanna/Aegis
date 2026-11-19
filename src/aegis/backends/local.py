"""Local process backend.

Runs torchrun in its own session so the whole worker group can be signalled as
a unit. Exit codes are written to disk by the launching shell rather than read
from the Popen object, so a run's outcome survives the controller that started
it dying.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.checkpoint.restore import PLAN_NAME, REPORT_NAME
from aegis.models import JobSpec, RunHandle, RunPhase, RunStatus

log = logging.getLogger(__name__)

RUN_META = "run.json"
EXIT_CODE_FILE = "exit_code"
PROGRESS_FILE = "progress.json"
METRICS_FILE = "metrics.jsonl"
TRAINING_SCRIPT = Path(__file__).resolve().parents[3] / "training" / "train_ddp.py"


class LocalBackend(ExecutionBackend):
    name = "local"

    def __init__(self, *, python_executable: str | None = None) -> None:
        self.python = python_executable or sys.executable
        # Launching shells we still need to reap. Only populated for runs this
        # process started; adopted runs are tracked purely through the
        # filesystem.
        self._procs: dict[str, subprocess.Popen] = {}

    # ------------------------------------------------------------- layout

    def job_dir(self, spec: JobSpec) -> Path:
        return Path(spec.workdir) / "runs" / spec.job_id

    def run_dir(self, spec: JobSpec, attempt: int) -> Path:
        return self.job_dir(spec) / f"attempt-{attempt:04d}"

    def progress_path(self, spec: JobSpec) -> str:
        # One file per job, not per attempt: the controller watches a single
        # path across restarts and uses the embedded run_id to ignore
        # heartbeats left behind by a previous attempt.
        return str(self.job_dir(spec) / PROGRESS_FILE)

    def metrics_path(self, spec: JobSpec) -> str:
        return str(self.job_dir(spec) / METRICS_FILE)

    def restore_report_path(self, spec: JobSpec, attempt: int) -> str | None:
        return str(self.run_dir(spec, attempt) / REPORT_NAME)

    # ------------------------------------------------------------- launch

    def launch(self, request: LaunchRequest) -> RunHandle:
        spec = request.spec
        run_dir = self.run_dir(spec, request.attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / EXIT_CODE_FILE).unlink(missing_ok=True)
        (run_dir / REPORT_NAME).unlink(missing_ok=True)
        plan_path = run_dir / PLAN_NAME
        if request.restore_plan:
            _write_json_text(plan_path, request.restore_plan)
        else:
            plan_path.unlink(missing_ok=True)

        env = {
            **os.environ,
            **spec.env,
            **request.extra_env,
            "AEGIS_METRICS_FILE": self.metrics_path(spec),
            "AEGIS_JOB_ID": spec.job_id,
            "AEGIS_JOB_NAME": spec.name,
            "AEGIS_RUN_ID": request.run_id,
            "AEGIS_ATTEMPT": str(request.attempt),
            "AEGIS_ACTION_ID": request.action_id,
            "AEGIS_CHECKPOINT_DIR": spec.checkpoint_dir,
            "AEGIS_PROGRESS_FILE": self.progress_path(spec),
            "AEGIS_WORLD_SIZE": str(spec.world_size),
            "AEGIS_TOTAL_STEPS": str(spec.total_steps),
            "AEGIS_RESUME_STEP": "" if request.resume_step is None else str(request.resume_step),
            "AEGIS_EXCLUDED_NODES": ",".join(request.excluded_nodes),
            "AEGIS_PLACEMENT": ",".join(request.placement),
            "AEGIS_RESTORE_PLAN": str(plan_path) if request.restore_plan else "",
            "AEGIS_RUN_DIR": str(run_dir),
            # Aegis owns job-level retries; Elastic must not run its own.
            "TORCHELASTIC_MAX_RESTARTS": "0",
            "OMP_NUM_THREADS": env_default("OMP_NUM_THREADS", "1"),
            "MASTER_ADDR": "127.0.0.1",
            "GLOO_SOCKET_IFNAME": env_default(
                "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
            ),
        }

        argv = self._build_argv(spec, request)
        logfile = run_dir / "run.log"
        # The trailing redirect records the exit status durably; without it a
        # controller crash would lose the outcome of the run it was watching.
        script = f"{shlex.join(argv)} ; printf %s $? > {shlex.quote(str(run_dir / EXIT_CODE_FILE))}"

        with logfile.open("ab") as out:
            out.write(f"\n=== attempt {request.attempt} run {request.run_id} ===\n".encode())
            out.flush()
            proc = subprocess.Popen(
                ["/bin/sh", "-c", script],
                cwd=spec.workdir,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        handle = RunHandle(
            job_id=spec.job_id,
            action_id=request.action_id,
            attempt=request.attempt,
            backend=self.name,
            reference=str(proc.pid),
            started_at=datetime.now(UTC),
            run_dir=str(run_dir),
            extra={
                "pgid": proc.pid,
                "run_id": request.run_id,
                "log": str(logfile),
                "placement": list(request.placement),
                "resume_step": request.resume_step,
            },
        )
        self._procs[handle.action_id] = proc
        _write_json(run_dir / RUN_META, _handle_payload(handle))
        log.info(
            "launched job=%s attempt=%s pid=%s resume_step=%s",
            spec.job_id,
            request.attempt,
            proc.pid,
            request.resume_step,
        )
        return handle

    def _build_argv(self, spec: JobSpec, request: LaunchRequest) -> list[str]:
        if spec.command:
            return list(spec.command)
        # Static rendezvous pinned to loopback. The c10d backend resolves the
        # host name, which on macOS yields an IPv6 address with no reverse
        # record and stalls the rendezvous for minutes.
        return [
            self.python,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={spec.world_size}",
            "--max_restarts=0",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            str(TRAINING_SCRIPT),
        ]

    # ------------------------------------------------------------- status

    def status(self, handle: RunHandle) -> RunStatus:
        run_dir = Path(handle.run_dir)
        exit_file = run_dir / EXIT_CODE_FILE

        if (proc := self._procs.get(handle.action_id)) is not None:
            proc.poll()

        if exit_file.is_file():
            raw = exit_file.read_text().strip()
            try:
                code = int(raw)
            except ValueError:
                return RunStatus(RunPhase.FAILED, None, f"unreadable exit code {raw!r}")
            if code == 0:
                return RunStatus(RunPhase.SUCCEEDED, 0)
            return RunStatus(RunPhase.FAILED, code, f"exited with code {code}")

        pgid = int(handle.extra.get("pgid", handle.reference))
        if _pgid_alive(pgid):
            return RunStatus(RunPhase.RUNNING)

        # No exit code and no process: the run died in a way that skipped the
        # shell's status write, e.g. SIGKILL of the whole group.
        return RunStatus(RunPhase.FAILED, None, "process group vanished without recording an exit code")

    def terminate(self, handle: RunHandle, *, grace_seconds: float = 20.0) -> None:
        pgid = int(handle.extra.get("pgid", handle.reference))
        if self._has_ended(handle):
            return

        log.info("terminating job=%s attempt=%s pgid=%s", handle.job_id, handle.attempt, pgid)
        _signal_group(pgid, signal.SIGTERM)

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if self._has_ended(handle):
                return

        _signal_group(pgid, signal.SIGKILL)
        for _ in range(20):
            time.sleep(0.1)
            if self._has_ended(handle):
                return
        raise RuntimeError(f"process group {pgid} survived SIGKILL")

    def _has_ended(self, handle: RunHandle) -> bool:
        # Reap the launching shell before inspecting the group. A signalled
        # shell that has not been waited on stays a zombie, and a group whose
        # only remaining member is a zombie answers killpg with EPERM rather
        # than ESRCH -- which reads as "alive and refusing to die".
        self._reap(handle)
        if (Path(handle.run_dir) / EXIT_CODE_FILE).is_file():
            return True
        return not _pgid_alive(int(handle.extra.get("pgid", handle.reference)))

    def _reap(self, handle: RunHandle) -> None:
        """Collect the launching shell if it has exited. Never blocks."""
        proc = self._procs.get(handle.action_id)
        if proc is not None and proc.poll() is not None:
            self._procs.pop(handle.action_id, None)

    def discover(self, spec: JobSpec) -> list[RunHandle]:
        job_dir = self.job_dir(spec)
        if not job_dir.is_dir():
            return []
        alive: list[RunHandle] = []
        for run_dir in sorted(job_dir.glob("attempt-*")):
            meta = run_dir / RUN_META
            if not meta.is_file() or (run_dir / EXIT_CODE_FILE).is_file():
                continue
            try:
                handle = _handle_from_payload(json.loads(meta.read_text()))
            except (ValueError, TypeError, KeyError):
                continue
            if _pgid_alive(int(handle.extra.get("pgid", handle.reference))):
                alive.append(handle)
        return alive


def env_default(key: str, fallback: str) -> str:
    return os.environ.get(key, fallback)


def _pgid_alive(pgid: int) -> bool:
    # Signal 0 only checks existence. A recycled pid could in principle alias a
    # dead group; the exit-code file is checked first to make that unlikely.
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, payload: dict) -> None:
    _write_json_text(path, json.dumps(payload, indent=2, default=str))


def _write_json_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _handle_payload(handle: RunHandle) -> dict:
    return {
        "job_id": handle.job_id,
        "action_id": handle.action_id,
        "attempt": handle.attempt,
        "backend": handle.backend,
        "reference": handle.reference,
        "started_at": handle.started_at.isoformat(),
        "run_dir": handle.run_dir,
        "extra": handle.extra,
    }


def _handle_from_payload(payload: dict) -> RunHandle:
    return RunHandle(
        job_id=payload["job_id"],
        action_id=payload["action_id"],
        attempt=payload["attempt"],
        backend=payload["backend"],
        reference=payload["reference"],
        started_at=datetime.fromisoformat(payload["started_at"]),
        run_dir=payload.get("run_dir", ""),
        extra=payload.get("extra", {}),
    )
