"""Host-side staging metadata helpers for one-step-off dynamic resize."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


HOST_STAGING_MANIFEST = "host_staging_manifest.json"


@dataclass(slots=True)
class HostStagingConfig:
    enable: bool = True
    backend: str = "disk_fallback"
    chunk_mb: int = 256
    stage_optimizer: bool = True
    progressive_swap: bool = True
    cleanup_after_load: bool = True
    preclear_rollout_kv_cache: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "HostStagingConfig":
        cfg = cfg or {}
        return cls(
            enable=bool(cfg.get("enable", True)),
            backend=str(cfg.get("backend", "disk_fallback")),
            chunk_mb=max(int(cfg.get("chunk_mb", 256)), 1),
            stage_optimizer=bool(cfg.get("stage_optimizer", True)),
            progressive_swap=bool(cfg.get("progressive_swap", True)),
            cleanup_after_load=bool(cfg.get("cleanup_after_load", True)),
            preclear_rollout_kv_cache=bool(cfg.get("preclear_rollout_kv_cache", True)),
        )

    def effective_backend(self) -> str:
        # Round two keeps the abstraction flexible, but the implementation is
        # intentionally conservative: unsupported host-memory backends fall back
        # to the disk-backed staging path instead of failing mid-resize.
        if self.backend == "pinned_cpu":
            return "disk_fallback"
        return self.backend

    def to_manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requested_backend"] = self.backend
        data["backend"] = self.effective_backend()
        data["manifest_version"] = 1
        return data


def manifest_path(stage_dir: str) -> str:
    return os.path.join(stage_dir, HOST_STAGING_MANIFEST)


def write_host_staging_manifest(stage_dir: str, config: HostStagingConfig) -> str:
    os.makedirs(stage_dir, exist_ok=True)
    path = manifest_path(stage_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_manifest_dict(), f, indent=2, sort_keys=True)
    return path


def read_host_staging_manifest(stage_dir: str) -> dict[str, Any]:
    path = manifest_path(stage_dir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)