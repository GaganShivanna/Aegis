from aegis.backends.base import ExecutionBackend, LaunchRequest
from aegis.backends.local import LocalBackend

__all__ = ["ExecutionBackend", "LaunchRequest", "LocalBackend", "get_backend"]


def get_backend(name: str, **kwargs):
    if name == "local":
        return LocalBackend(**kwargs)
    if name in ("k8s", "kubernetes"):
        from aegis.backends.k8s import KubernetesBackend

        return KubernetesBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
