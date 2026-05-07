"""Simplified memory budget protection for dynamic resize."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResizeBudgetConfig:
    enable: bool = False
    memory_budget_ratio: float = 0.85

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "ResizeBudgetConfig":
        cfg = cfg or {}
        ratio = float(cfg.get("memory_budget_ratio", 0.85))
        ratio = min(max(ratio, 0.1), 0.99)
        return cls(enable=bool(cfg.get("enable", False)), memory_budget_ratio=ratio)


@dataclass(slots=True)
class ResizeBudgetSnapshot:
    host_free_bytes: int | None = None
    disk_free_bytes: int | None = None
    gpu_free_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "ResizeBudgetSnapshot":
        data = data or {}
        return cls(
            host_free_bytes=_to_optional_int(data.get("host_free_bytes")),
            disk_free_bytes=_to_optional_int(data.get("disk_free_bytes")),
            gpu_free_bytes=_to_optional_int(data.get("gpu_free_bytes")),
        )


@dataclass(slots=True)
class ResizeBudgetDecision:
    allow_resize: bool
    effective_backend: str
    blocked: bool
    reason: str


def _to_optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


class ResizeBudgetController:
    """A conservative budget gate driven by one global memory budget ratio.

    The controller does not expose many knobs. It only answers two questions:
    1. Is this resize allowed under the current host/GPU budget?
    2. If host-memory staging is too expensive, should we fall back to disk?
    """

    def __init__(self, config: ResizeBudgetConfig):
        self.config = config

    def evaluate_export(
        self,
        *,
        requested_backend: str,
        snapshot: ResizeBudgetSnapshot,
        estimated_host_peak_bytes: int,
        estimated_stage_bytes: int,
    ) -> ResizeBudgetDecision:
        # Export-side protection is intentionally simple: if host staging would
        # exceed the current budget, prefer a disk-backed staging path; if even
        # that is unsafe, block the resize before any heavy transfer starts.
        if not self.config.enable:
            return ResizeBudgetDecision(True, requested_backend, False, "disabled")

        host_budget = self._budget_limit(snapshot.host_free_bytes)
        disk_budget = self._budget_limit(snapshot.disk_free_bytes)
        effective_backend = requested_backend

        if requested_backend == "pinned_cpu" and host_budget is not None and estimated_stage_bytes > host_budget:
            effective_backend = "disk_fallback"

        if host_budget is not None and estimated_host_peak_bytes > host_budget:
            return ResizeBudgetDecision(False, effective_backend, True, "host_budget")

        if effective_backend == "disk_fallback" and disk_budget is not None and estimated_stage_bytes > disk_budget:
            return ResizeBudgetDecision(False, effective_backend, True, "disk_budget")

        return ResizeBudgetDecision(True, effective_backend, False, "ok")

    def evaluate_restore(self, *, snapshot: ResizeBudgetSnapshot, estimated_gpu_peak_bytes: int) -> ResizeBudgetDecision:
        # Restore-side protection never invents a new strategy. It only answers
        # whether the current staged restore is safe enough to proceed.
        if not self.config.enable:
            return ResizeBudgetDecision(True, "unchanged", False, "disabled")

        gpu_budget = self._budget_limit(snapshot.gpu_free_bytes)
        if gpu_budget is not None and estimated_gpu_peak_bytes > gpu_budget:
            return ResizeBudgetDecision(False, "unchanged", True, "gpu_budget")

        return ResizeBudgetDecision(True, "unchanged", False, "ok")

    def _budget_limit(self, free_bytes: int | None) -> int | None:
        if free_bytes is None:
            return None
        return int(free_bytes * self.config.memory_budget_ratio)