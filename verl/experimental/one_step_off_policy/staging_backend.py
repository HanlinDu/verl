"""Host-side staging metadata helpers for one-step-off dynamic resize."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch


HOST_STAGING_MANIFEST = "host_staging_manifest.json"
_PAGED_STATE_SUFFIX = "_pages.json"
RESTORE_SESSION_MANIFEST = "restore_session_manifest.json"


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


@dataclass(slots=True)
class RestoreSessionManifest:
    session_id: str
    backend: str
    status: str = "staged"
    model_page_count: int = 0
    optimizer_page_count: int = 0
    staged_model_bytes: int = 0
    staged_optimizer_bytes: int = 0
    applied_model_pages: int = 0
    applied_optimizer_pages: int = 0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest_version"] = 1
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RestoreSessionManifest":
        data = data or {}
        return cls(
            session_id=str(data.get("session_id") or uuid.uuid4().hex),
            backend=str(data.get("backend", "disk_fallback")),
            status=str(data.get("status", "staged")),
            model_page_count=max(int(data.get("model_page_count", 0)), 0),
            optimizer_page_count=max(int(data.get("optimizer_page_count", 0)), 0),
            staged_model_bytes=max(int(data.get("staged_model_bytes", 0)), 0),
            staged_optimizer_bytes=max(int(data.get("staged_optimizer_bytes", 0)), 0),
            applied_model_pages=max(int(data.get("applied_model_pages", 0)), 0),
            applied_optimizer_pages=max(int(data.get("applied_optimizer_pages", 0)), 0),
            last_error=str(data.get("last_error", "")),
        )


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


def restore_session_manifest_path(stage_dir: str) -> str:
    return os.path.join(stage_dir, RESTORE_SESSION_MANIFEST)


def has_restore_session_manifest(stage_dir: str) -> bool:
    return os.path.exists(restore_session_manifest_path(stage_dir))


def read_restore_session_manifest(stage_dir: str) -> dict[str, Any]:
    with open(restore_session_manifest_path(stage_dir), encoding="utf-8") as f:
        return json.load(f)


def write_restore_session_manifest(stage_dir: str, manifest: RestoreSessionManifest) -> dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    data = manifest.to_dict()
    with open(restore_session_manifest_path(stage_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data


def create_restore_session_manifest(
    stage_dir: str,
    *,
    backend: str,
    session_id: str | None = None,
    model_manifest: dict[str, Any] | None = None,
    optimizer_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = RestoreSessionManifest(
        session_id=session_id or uuid.uuid4().hex,
        backend=backend,
        model_page_count=_manifest_page_count(model_manifest),
        optimizer_page_count=_manifest_page_count(optimizer_manifest),
        staged_model_bytes=_manifest_total_bytes(model_manifest),
        staged_optimizer_bytes=_manifest_total_bytes(optimizer_manifest),
    )
    return write_restore_session_manifest(stage_dir, manifest)


def update_restore_session_manifest(stage_dir: str, **updates: Any) -> dict[str, Any]:
    manifest = RestoreSessionManifest.from_dict(read_restore_session_manifest(stage_dir))
    for key, value in updates.items():
        if hasattr(manifest, key):
            setattr(manifest, key, value)
    return write_restore_session_manifest(stage_dir, manifest)


def record_restore_progress(
    stage_dir: str,
    *,
    applied_model_pages: int | None = None,
    applied_optimizer_pages: int | None = None,
    status: str | None = None,
    last_error: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if applied_model_pages is not None:
        updates["applied_model_pages"] = max(int(applied_model_pages), 0)
    if applied_optimizer_pages is not None:
        updates["applied_optimizer_pages"] = max(int(applied_optimizer_pages), 0)
    if status is not None:
        updates["status"] = status
    if last_error is not None:
        updates["last_error"] = last_error
    return update_restore_session_manifest(stage_dir, **updates)


def finalize_restore_session_manifest(stage_dir: str, *, status: str = "completed") -> dict[str, Any]:
    manifest = RestoreSessionManifest.from_dict(read_restore_session_manifest(stage_dir))
    return update_restore_session_manifest(
        stage_dir,
        status=status,
        applied_model_pages=manifest.model_page_count,
        applied_optimizer_pages=manifest.optimizer_page_count,
        last_error="",
    )


def paged_state_manifest_path(stage_dir: str, prefix: str) -> str:
    return os.path.join(stage_dir, f"{prefix}{_PAGED_STATE_SUFFIX}")


def has_paged_state_dict(stage_dir: str, prefix: str) -> bool:
    return os.path.exists(paged_state_manifest_path(stage_dir, prefix))


def read_paged_state_manifest(stage_dir: str, prefix: str) -> dict[str, Any]:
    with open(paged_state_manifest_path(stage_dir, prefix), encoding="utf-8") as f:
        return json.load(f)


def save_paged_state_dict(stage_dir: str, prefix: str, state_dict: dict[str, Any], page_bytes: int) -> dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    target_bytes = max(int(page_bytes), 1)
    pages = _split_state_dict_into_pages(state_dict, target_bytes)
    files: list[str] = []

    page_metas: list[dict[str, Any]] = []
    for idx, page in enumerate(pages):
        file_name = f"{prefix}.page_{idx:04d}.pt"
        torch.save(page, os.path.join(stage_dir, file_name))
        files.append(file_name)
        page_metas.append(
            {
                "page_id": idx,
                "file_name": file_name,
                "entry_count": len(page),
                "tensor_keys": list(page.keys()),
                "estimated_bytes": _estimate_object_bytes(page),
            }
        )

    manifest = {
        "manifest_version": 1,
        "prefix": prefix,
        "page_bytes": target_bytes,
        "page_count": len(files),
        "files": files,
        "pages": page_metas,
        "total_bytes": sum(int(page_meta["estimated_bytes"]) for page_meta in page_metas),
    }
    with open(paged_state_manifest_path(stage_dir, prefix), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def save_paged_optimizer_state_dict(
    stage_dir: str,
    prefix: str,
    optim_state_dict: dict[str, Any],
    page_bytes: int,
) -> dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    target_bytes = max(int(page_bytes), 1)
    state = dict(optim_state_dict.get("state", {}))
    param_groups = list(optim_state_dict.get("param_groups", []))
    pages = _split_optimizer_state_into_pages(state, param_groups, target_bytes)
    files: list[str] = []

    page_metas: list[dict[str, Any]] = []
    for idx, page in enumerate(pages):
        file_name = f"{prefix}.page_{idx:04d}.pt"
        torch.save(page, os.path.join(stage_dir, file_name))
        files.append(file_name)
        page_metas.append(
            {
                "page_id": idx,
                "file_name": file_name,
                "entry_count": len(page.get("state", {})),
                "tensor_keys": list(page.get("state", {}).keys()),
                "estimated_bytes": _estimate_object_bytes(page),
            }
        )

    manifest = {
        "manifest_version": 1,
        "prefix": prefix,
        "page_bytes": target_bytes,
        "page_count": len(files),
        "files": files,
        "pages": page_metas,
        "state_entries": len(state),
        "total_bytes": sum(int(page_meta["estimated_bytes"]) for page_meta in page_metas),
    }
    with open(paged_state_manifest_path(stage_dir, prefix), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def iter_paged_state_dict(stage_dir: str, prefix: str) -> Iterator[dict[str, Any]]:
    manifest = read_paged_state_manifest(stage_dir, prefix)
    for file_name in manifest.get("files", []):
        yield torch.load(os.path.join(stage_dir, file_name), weights_only=False)


def load_paged_optimizer_state_dict(stage_dir: str, prefix: str) -> dict[str, Any]:
    merged_state: dict[str, Any] = {}
    merged_param_groups: list[dict[str, Any]] | None = None

    for page in iter_paged_state_dict(stage_dir, prefix):
        merged_state.update(page.get("state", {}))
        page_param_groups = list(page.get("param_groups", []))
        if merged_param_groups is None:
            merged_param_groups = page_param_groups

    return {
        "state": merged_state,
        "param_groups": merged_param_groups or [],
    }


def _split_state_dict_into_pages(state_dict: dict[str, Any], page_bytes: int) -> list[dict[str, Any]]:
    if not state_dict:
        return [{}]

    pages: list[dict[str, Any]] = []
    current_page: dict[str, Any] = {}
    current_bytes = 0

    for key, value in state_dict.items():
        value_bytes = _estimate_object_bytes(value)
        if current_page and current_bytes + value_bytes > page_bytes:
            pages.append(current_page)
            current_page = {}
            current_bytes = 0
        current_page[key] = value
        current_bytes += value_bytes

    if current_page:
        pages.append(current_page)
    return pages


def _split_optimizer_state_into_pages(
    state: dict[str, Any],
    param_groups: list[dict[str, Any]],
    page_bytes: int,
) -> list[dict[str, Any]]:
    if not state:
        return [{"state": {}, "param_groups": param_groups}]

    pages: list[dict[str, Any]] = []
    current_state: dict[str, Any] = {}
    current_bytes = 0

    for key, value in state.items():
        value_bytes = _estimate_object_bytes(value)
        if current_state and current_bytes + value_bytes > page_bytes:
            pages.append({"state": current_state, "param_groups": param_groups})
            current_state = {}
            current_bytes = 0
        current_state[key] = value
        current_bytes += value_bytes

    if current_state:
        pages.append({"state": current_state, "param_groups": param_groups})
    return pages


def _estimate_object_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_estimate_object_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_object_bytes(item) for item in value)
    return 1


def _manifest_page_count(manifest: dict[str, Any] | None) -> int:
    if not manifest:
        return 0
    return max(int(manifest.get("page_count", 0)), 0)


def _manifest_total_bytes(manifest: dict[str, Any] | None) -> int:
    if not manifest:
        return 0
    if "total_bytes" in manifest:
        return max(int(manifest.get("total_bytes", 0)), 0)
    return sum(int(page.get("estimated_bytes", 0)) for page in manifest.get("pages", []))