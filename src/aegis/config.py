"""Runtime configuration.

Every setting is overridable by an AEGIS_-prefixed environment variable so the
same image runs locally and in a cluster without a config file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", extra="ignore")

    workdir: Path = Path(".aegis")
    database_url: str = ""
    backend: str = "local"
    policy: str = "baseline"

    tick_interval_seconds: float = 2.0
    metrics_port: int = 0  # 0 disables the exporter

    # Nodes jobs may be placed on. Locally these are simulated: each has its
    # own checkpoint directories, which disappear when the node is lost.
    nodes: str = "node-0,node-1,node-2,node-3"
    # Storage performance model; see aegis.checkpoint.tiers.
    storage_profile: str = "scaled"

    # Kubernetes
    namespace: str = "default"
    kubeconfig: str = ""
    image: str = ""

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.workdir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.workdir / 'aegis.sqlite').resolve()}"

    @property
    def node_pool(self) -> tuple[str, ...]:
        return tuple(n.strip() for n in self.nodes.split(",") if n.strip())

    @property
    def events_path(self) -> Path:
        return self.workdir / "events.jsonl"

    def checkpoint_dir(self, job_id: str) -> Path:
        return self.workdir / "checkpoints" / job_id

    def ensure_dirs(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "runs").mkdir(parents=True, exist_ok=True)
        (self.workdir / "checkpoints").mkdir(parents=True, exist_ok=True)


def load_settings(**overrides) -> Settings:
    settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
    settings.ensure_dirs()
    return settings
