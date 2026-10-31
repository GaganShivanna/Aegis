"""Kubernetes backend.

One Job per attempt, named deterministically from the action id. That name is
the cluster-side half of the duplicate-prevention guarantee: a second create
for the same action fails with 409 Conflict instead of producing a second set
of workers, no matter how many controller replicas try.

Aegis only excludes nodes and submits work. It never cordons, uncordons or
resets a GPU; that is NVSentinel's job.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.models import JobSpec, RunHandle, RunPhase, RunStatus

log = logging.getLogger(__name__)

MANAGED_BY = "aegis"
LABEL_JOB_ID = "aegis.io/job-id"
LABEL_ACTION_ID = "aegis.io/action-id"
LABEL_ATTEMPT = "aegis.io/attempt"
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"


class KubernetesBackend(ExecutionBackend):
    name = "k8s"

    def __init__(
        self,
        *,
        namespace: str = "default",
        image: str = "",
        kubeconfig: str | None = None,
        service_account: str = "",
        shared_volume_claim: str = "",
    ) -> None:
        from kubernetes import client, config

        try:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                config.load_incluster_config()
        except Exception:
            config.load_kube_config()

        self.namespace = namespace
        self.image = image
        self.service_account = service_account
        self.shared_volume_claim = shared_volume_claim
        self.batch = client.BatchV1Api()
        self.core = client.CoreV1Api()
        self._client = client

    # -------------------------------------------------------------- naming

    @staticmethod
    def job_name(request_or_handle: LaunchRequest | RunHandle) -> str:
        action_id = request_or_handle.action_id
        job_id = (
            request_or_handle.spec.job_id
            if isinstance(request_or_handle, LaunchRequest)
            else request_or_handle.job_id
        )
        return f"aegis-{job_id[:28]}-{action_id[:12]}".lower().replace("_", "-")

    def progress_path(self, spec: JobSpec) -> str:
        return f"{spec.workdir}/runs/{spec.job_id}/progress.json"

    # -------------------------------------------------------------- launch

    def launch(self, request: LaunchRequest) -> RunHandle:
        name = self.job_name(request)
        body = self._build_job(request, name)
        try:
            self.batch.create_namespaced_job(namespace=self._ns(request.spec), body=body)
        except self._client.exceptions.ApiException as exc:
            if exc.status != 409:
                raise
            # Already created for this action: adopt rather than duplicate.
            log.info("job %s already exists; adopting", name)
        return RunHandle(
            job_id=request.spec.job_id,
            action_id=request.action_id,
            attempt=request.attempt,
            backend=self.name,
            reference=f"{self._ns(request.spec)}/{name}",
            started_at=datetime.now(UTC),
            extra={"run_id": request.run_id, "name": name, "namespace": self._ns(request.spec)},
        )

    def _build_job(self, request: LaunchRequest, name: str) -> dict[str, Any]:
        spec = request.spec
        env = {
            **spec.env,
            **request.extra_env,
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
            "TORCHELASTIC_MAX_RESTARTS": "0",
        }
        labels = {
            LABEL_JOB_ID: spec.job_id,
            LABEL_ACTION_ID: request.action_id,
            LABEL_ATTEMPT: str(request.attempt),
            LABEL_MANAGED_BY: MANAGED_BY,
        }

        command = spec.command or [
            "torchrun",
            "--nnodes=1",
            f"--nproc_per_node={spec.world_size}",
            "--max_restarts=0",
            "training/train_ddp.py",
        ]

        container: dict[str, Any] = {
            "name": "trainer",
            "image": spec.image or self.image,
            "command": command,
            "env": [{"name": k, "value": v} for k, v in env.items()],
            "resources": {"limits": {"nvidia.com/gpu": str(spec.world_size)}},
        }
        volumes: list[dict[str, Any]] = []
        if self.shared_volume_claim:
            container["volumeMounts"] = [{"name": "state", "mountPath": spec.workdir}]
            volumes.append(
                {
                    "name": "state",
                    "persistentVolumeClaim": {"claimName": self.shared_volume_claim},
                }
            )

        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
            "volumes": volumes,
        }
        if self.service_account:
            pod_spec["serviceAccountName"] = self.service_account
        if spec.node_selector:
            pod_spec["nodeSelector"] = spec.node_selector
        if affinity := self._exclusion_affinity(request.excluded_nodes):
            pod_spec["affinity"] = affinity

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "labels": labels},
            "spec": {
                # Aegis owns retries. Letting the Job controller retry too
                # would put a second restart policy underneath the attempt
                # budget and make the budget meaningless.
                "backoffLimit": 0,
                "completions": 1,
                "parallelism": 1,
                "ttlSecondsAfterFinished": 86400,
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        }

    @staticmethod
    def _exclusion_affinity(excluded_nodes: tuple[str, ...]) -> dict[str, Any] | None:
        """Keep the job off nodes NVSentinel has flagged.

        This is placement *exclusion* only; the scheduler still chooses where
        the pod lands. Topology-aware scoring comes later.
        """
        if not excluded_nodes:
            return None
        return {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "NotIn",
                                    "values": list(excluded_nodes),
                                }
                            ]
                        }
                    ]
                }
            }
        }

    # -------------------------------------------------------------- status

    def status(self, handle: RunHandle) -> RunStatus:
        namespace = handle.extra.get("namespace", self.namespace)
        name = handle.extra.get("name") or handle.reference.split("/")[-1]
        try:
            job = self.batch.read_namespaced_job_status(name=name, namespace=namespace)
        except self._client.exceptions.ApiException as exc:
            if exc.status == 404:
                return RunStatus(RunPhase.GONE, message="job not found")
            raise

        status = job.status
        if status.succeeded:
            return RunStatus(RunPhase.SUCCEEDED, 0)
        if status.failed:
            return RunStatus(
                RunPhase.FAILED,
                None,
                _failure_message(status),
                self._pod_nodes(namespace, handle.action_id),
            )
        if status.active:
            return RunStatus(RunPhase.RUNNING)
        return RunStatus(RunPhase.PENDING)

    def _pod_nodes(self, namespace: str, action_id: str) -> tuple[str, ...]:
        try:
            pods = self.core.list_namespaced_pod(
                namespace=namespace, label_selector=f"{LABEL_ACTION_ID}={action_id}"
            )
        except Exception:
            return ()
        return tuple(p.spec.node_name for p in pods.items if p.spec.node_name)

    def terminate(self, handle: RunHandle, *, grace_seconds: float = 20.0) -> None:
        namespace = handle.extra.get("namespace", self.namespace)
        name = handle.extra.get("name") or handle.reference.split("/")[-1]
        try:
            self.batch.delete_namespaced_job(
                name=name,
                namespace=namespace,
                body=self._client.V1DeleteOptions(
                    propagation_policy="Foreground",
                    grace_period_seconds=int(grace_seconds),
                ),
            )
        except self._client.exceptions.ApiException as exc:
            if exc.status != 404:
                raise

    def discover(self, spec: JobSpec) -> list[RunHandle]:
        namespace = self._ns(spec)
        try:
            jobs = self.batch.list_namespaced_job(
                namespace=namespace, label_selector=f"{LABEL_JOB_ID}={spec.job_id}"
            )
        except Exception:
            log.exception("discover failed for job %s", spec.job_id)
            return []

        handles = []
        for job in jobs.items:
            if job.status.succeeded or job.status.failed:
                continue
            labels = job.metadata.labels or {}
            handles.append(
                RunHandle(
                    job_id=spec.job_id,
                    action_id=labels.get(LABEL_ACTION_ID, ""),
                    attempt=int(labels.get(LABEL_ATTEMPT, "0")),
                    backend=self.name,
                    reference=f"{namespace}/{job.metadata.name}",
                    started_at=job.metadata.creation_timestamp or datetime.now(UTC),
                    extra={"name": job.metadata.name, "namespace": namespace},
                )
            )
        return handles

    def _ns(self, spec: JobSpec) -> str:
        return spec.namespace or self.namespace


def _failure_message(status: Any) -> str:
    for condition in status.conditions or []:
        if condition.type == "Failed":
            return condition.message or condition.reason or "job failed"
    return "job failed"
