"""Host-side staging metadata helpers for one-step-off dynamic resize."""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
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
    service_name: str | None = None
    chunk_mb: int = 256
    stage_optimizer: bool = True
    optimizer_restore_policy: str = "deferred"
    progressive_swap: bool = True
    async_optimizer_preload: bool = True
    preload_queue_depth: int = 2
    device_preload_threshold: float = 0.9
    cleanup_after_load: bool = True
    preclear_rollout_kv_cache: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "HostStagingConfig":
        cfg = cfg or {}
        return cls(
            enable=bool(cfg.get("enable", True)),
            backend=str(cfg.get("backend", "disk_fallback")),
            service_name=_normalize_optional_string(cfg.get("service_name")),
            chunk_mb=max(int(cfg.get("chunk_mb", 256)), 1),
            stage_optimizer=bool(cfg.get("stage_optimizer", True)),
            optimizer_restore_policy=_normalize_optimizer_restore_policy(cfg.get("optimizer_restore_policy", "deferred")),
            progressive_swap=bool(cfg.get("progressive_swap", True)),
            async_optimizer_preload=bool(cfg.get("async_optimizer_preload", True)),
            preload_queue_depth=max(int(cfg.get("preload_queue_depth", 2)), 1),
            device_preload_threshold=min(max(float(cfg.get("device_preload_threshold", 0.9)), 0.0), 1.0),
            cleanup_after_load=bool(cfg.get("cleanup_after_load", True)),
            preclear_rollout_kv_cache=bool(cfg.get("preclear_rollout_kv_cache", True)),
        )

    def should_restore_optimizer_on_load(self) -> bool:
        return self.stage_optimizer and self.optimizer_restore_policy == "immediate"

    def should_defer_optimizer_restore(self) -> bool:
        return self.stage_optimizer and self.optimizer_restore_policy == "deferred"

    def effective_backend(self) -> str:
        # The backend is now a real runtime choice. Budget protection may still
        # downgrade `pinned_cpu` to `disk_fallback`, but the config layer no
        # longer hides that decision by auto-falling back here.
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
    service_name: str | None = None
    status: str = "staged"
    optimizer_restore_policy: str = "deferred"
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
            service_name=_normalize_optional_string(data.get("service_name")),
            status=str(data.get("status", "staged")),
            optimizer_restore_policy=_normalize_optimizer_restore_policy(data.get("optimizer_restore_policy", "deferred")),
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
    optimizer_restore_policy: str = "deferred",
    model_manifest: dict[str, Any] | None = None,
    optimizer_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = RestoreSessionManifest(
        session_id=session_id or uuid.uuid4().hex,
        backend=backend,
        optimizer_restore_policy=_normalize_optimizer_restore_policy(optimizer_restore_policy),
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


def probe_local_pin_memory_capability() -> tuple[bool, str]:
    try:
        sample = torch.empty(1, dtype=torch.uint8).pin_memory()
        if not sample.is_pinned():
            return False, "pin_memory() returned a non-pinned tensor"
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        return False, repr(exc)


def pin_memory_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        cpu_tensor = value.detach().cpu()
        return cpu_tensor if cpu_tensor.is_pinned() else cpu_tensor.pin_memory()
    if isinstance(value, dict):
        return {key: pin_memory_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pin_memory_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(pin_memory_tree(item) for item in value)
    return value


def load_paged_state_page(stage_dir: str, file_name: str, *, pin_memory: bool = False) -> dict[str, Any]:
    page = torch.load(os.path.join(stage_dir, file_name), weights_only=False)
    return pin_memory_tree(page) if pin_memory else page


def save_paged_state_dict(stage_dir: str, prefix: str, state_dict: dict[str, Any], page_bytes: int) -> dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    target_bytes = max(int(page_bytes), 1)
    manifest, pages = build_paged_state_pages(prefix=prefix, state_dict=state_dict, page_bytes=target_bytes)
    for page_meta, page in zip(manifest["pages"], pages, strict=True):
        torch.save(page, os.path.join(stage_dir, page_meta["file_name"]))
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
    manifest, pages = build_paged_optimizer_state_pages(
        prefix=prefix,
        optim_state_dict=optim_state_dict,
        page_bytes=target_bytes,
    )
    for page_meta, page in zip(manifest["pages"], pages, strict=True):
        torch.save(page, os.path.join(stage_dir, page_meta["file_name"]))
    with open(paged_state_manifest_path(stage_dir, prefix), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def build_paged_state_pages(
    *,
    prefix: str,
    state_dict: dict[str, Any],
    page_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_bytes = max(int(page_bytes), 1)
    pages = _split_state_dict_into_pages(state_dict, target_bytes)
    page_metas: list[dict[str, Any]] = []

    for idx, page in enumerate(pages):
        file_name = f"{prefix}.page_{idx:04d}.pt"
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
        "page_count": len(page_metas),
        "files": [page_meta["file_name"] for page_meta in page_metas],
        "pages": page_metas,
        "total_bytes": sum(int(page_meta["estimated_bytes"]) for page_meta in page_metas),
    }
    return manifest, pages


def build_paged_optimizer_state_pages(
    *,
    prefix: str,
    optim_state_dict: dict[str, Any],
    page_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_bytes = max(int(page_bytes), 1)
    state = dict(optim_state_dict.get("state", {}))
    param_groups = list(optim_state_dict.get("param_groups", []))
    pages = _split_optimizer_state_into_pages(state, param_groups, target_bytes)
    page_metas: list[dict[str, Any]] = []

    for idx, page in enumerate(pages):
        file_name = f"{prefix}.page_{idx:04d}.pt"
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
        "page_count": len(page_metas),
        "files": [page_meta["file_name"] for page_meta in page_metas],
        "pages": page_metas,
        "state_entries": len(state),
        "total_bytes": sum(int(page_meta["estimated_bytes"]) for page_meta in page_metas),
    }
    return manifest, pages


def iter_paged_state_dict(
    stage_dir: str,
    prefix: str,
    *,
    pin_memory: bool = False,
    prefetch: bool = False,
) -> Iterator[dict[str, Any]]:
    manifest = read_paged_state_manifest(stage_dir, prefix)
    file_names = list(manifest.get("files", []))
    if not file_names:
        return

    if pin_memory and prefetch:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{prefix}_page_prefetch") as executor:
            next_future = executor.submit(load_paged_state_page, stage_dir, file_names[0], pin_memory=True)
            for idx in range(len(file_names)):
                page = next_future.result()
                if idx + 1 < len(file_names):
                    next_future = executor.submit(load_paged_state_page, stage_dir, file_names[idx + 1], pin_memory=True)
                yield page
        return

    for file_name in file_names:
        yield load_paged_state_page(stage_dir, file_name, pin_memory=pin_memory)


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


def _normalize_optimizer_restore_policy(value: Any) -> str:
    policy = str(value or "deferred").strip().lower()
    if policy not in {"immediate", "deferred"}:
        return "deferred"
    return policy


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None