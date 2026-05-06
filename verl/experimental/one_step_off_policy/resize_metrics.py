"""Helpers for extracting resize-control observations from one-step-off steps."""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_resize_observation(*, timing_raw: dict[str, Any], batch=None) -> dict[str, float]:
    """Build a compact observation for resize gating.

    The first-round controller focuses on scheduled topology changes. It uses
    recent rollout-heavy latency versus train-heavy latency as the primary
    signal, while keeping the extraction intentionally lightweight.
    """

    gen_s = _safe_float(timing_raw.get("gen"))
    sync_s = _safe_float(timing_raw.get("sync_rollout_weights"))
    update_actor_s = _safe_float(timing_raw.get("update_actor"))
    update_critic_s = _safe_float(timing_raw.get("update_critic"))

    rollout_time_s = gen_s + sync_s
    train_time_s = update_actor_s + update_critic_s

    token_count = 0.0
    batch_size = 0.0
    if batch is not None:
        try:
            batch_size = float(len(batch))
        except Exception:
            batch_size = 0.0

        global_token_num = getattr(batch, "meta_info", {}).get("global_token_num")
        if isinstance(global_token_num, list):
            token_count = float(sum(global_token_num))

    return {
        "gen_s": gen_s,
        "sync_rollout_weights_s": sync_s,
        "update_actor_s": update_actor_s,
        "update_critic_s": update_critic_s,
        "rollout_time_s": rollout_time_s,
        "train_time_s": train_time_s,
        "token_count": token_count,
        "batch_size": batch_size,
    }