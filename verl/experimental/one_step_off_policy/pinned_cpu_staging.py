"""Pinned CPU staging service for one-step-off dynamic resize.

This service keeps staged pages in a long-lived Ray actor so the resize path
can move state through host memory instead of spilling the whole handoff to
local disk files.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import ray
import torch


PINNED_CPU_STAGING_NAMESPACE = "verl_pinned_cpu_staging"


def _pin_object(value: Any) -> Any:
    if torch.is_tensor(value):
        cpu_tensor = value.detach().cpu()
        return cpu_tensor.pin_memory() if not cpu_tensor.is_pinned() else cpu_tensor
    if isinstance(value, dict):
        return {key: _pin_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pin_object(item) for item in value)
    return deepcopy(value)


def _count_pinned_tensors(value: Any) -> int:
    if torch.is_tensor(value):
        return 1 if value.is_pinned() else 0
    if isinstance(value, dict):
        return sum(_count_pinned_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_pinned_tensors(item) for item in value)
    return 0


def _estimate_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_estimate_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_bytes(item) for item in value)
    return 1


def _probe_pin_memory_capability() -> tuple[bool, str]:
    """Probe pinned-memory support inside the service worker process.

    The pinned CPU backend only makes sense if the actor process can actually
    call ``pin_memory()``. In practice that depends on the CUDA-visible runtime
    environment of the Ray worker process, not just the trainer driver.
    """

    try:
        sample = torch.empty(1, dtype=torch.uint8).pin_memory()
        if not sample.is_pinned():
            return False, "pin_memory() returned a non-pinned tensor"
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        return False, repr(exc)


def _build_pinned_cpu_service_runtime_env() -> dict[str, dict[str, str]]:
    """Preserve CUDA visibility for zero-GPU actors that only pin host memory.

    The service does not need Ray GPU resources, but it still needs CUDA-visible
    runtime state so ``torch.pin_memory()`` can succeed in the worker process.
    """

    env_vars = {"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0"}
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return {"env_vars": env_vars}


@ray.remote(num_cpus=1, max_concurrency=64)
class PinnedCPUStagingService:
    """A tiny host-memory object store for staged resize pages.

    The service stores model/optimizer pages as pinned CPU tensors so workers can
    fetch them later during the short resize window.
    """

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._pin_memory_available, self._pin_memory_error = _probe_pin_memory_capability()

    def create_session(self, session_id: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self._sessions.setdefault(
            session_id,
            {
                "metadata": deepcopy(metadata or {}),
                "objects": {},
                "stats": {"object_count": 0, "pinned_tensor_count": 0, "total_bytes": 0},
            },
        )
        return self.describe_session(session_id)

    def put_object(self, session_id: str, object_key: str, value: Any, *, pin_tensors: bool = False) -> dict[str, Any]:
        session = self._sessions.setdefault(
            session_id,
            {"metadata": {}, "objects": {}, "stats": {"object_count": 0, "pinned_tensor_count": 0, "total_bytes": 0}},
        )
        if pin_tensors and not self._pin_memory_available:
            raise RuntimeError(
                "PinnedCPUStagingService cannot pin tensors in this Ray actor process; "
                f"CUDA_VISIBLE_DEVICES={self._cuda_visible_devices!r}, error={self._pin_memory_error}"
            )
        stored = _pin_object(value) if pin_tensors else deepcopy(value)
        session["objects"][object_key] = stored
        session["stats"]["object_count"] = len(session["objects"])
        session["stats"]["pinned_tensor_count"] = sum(_count_pinned_tensors(item) for item in session["objects"].values())
        session["stats"]["total_bytes"] = sum(_estimate_bytes(item) for item in session["objects"].values())
        return self.describe_session(session_id)

    def get_object(self, session_id: str, object_key: str) -> Any:
        return deepcopy(self._sessions[session_id]["objects"][object_key])

    def pop_object(self, session_id: str, object_key: str) -> Any:
        session = self._sessions[session_id]
        value = deepcopy(session["objects"].pop(object_key))
        session["stats"]["object_count"] = len(session["objects"])
        session["stats"]["pinned_tensor_count"] = sum(_count_pinned_tensors(item) for item in session["objects"].values())
        session["stats"]["total_bytes"] = sum(_estimate_bytes(item) for item in session["objects"].values())
        return value

    def describe_session(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            return {"exists": False, "session_id": session_id}
        return {
            "exists": True,
            "session_id": session_id,
            "metadata": deepcopy(session.get("metadata", {})),
            "pin_memory_available": self._pin_memory_available,
            "pin_memory_error": self._pin_memory_error,
            "cuda_visible_devices": self._cuda_visible_devices,
            "object_keys": sorted(session["objects"].keys()),
            "object_count": int(session["stats"].get("object_count", 0)),
            "pinned_tensor_count": int(session["stats"].get("pinned_tensor_count", 0)),
            "total_bytes": int(session["stats"].get("total_bytes", 0)),
        }

    def release_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


def get_or_create_pinned_cpu_staging_service(service_name: str):
    """Return a named staging service actor, creating it on first use."""

    try:
        return ray.get_actor(service_name, namespace=PINNED_CPU_STAGING_NAMESPACE)
    except ValueError:
        return PinnedCPUStagingService.options(
            name=service_name,
            namespace=PINNED_CPU_STAGING_NAMESPACE,
            lifetime="detached",
            runtime_env=_build_pinned_cpu_service_runtime_env(),
        ).remote()
