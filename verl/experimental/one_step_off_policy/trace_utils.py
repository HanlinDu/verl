from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ResizeTraceConfig:
    enabled: bool = False
    path: str = ""
    run_name: str = ""
    flush: bool = False


def build_resize_trace_config(config: Any) -> ResizeTraceConfig:
    trace_cfg = None
    try:
        trace_cfg = OmegaConf.select(config, "trainer.dynamic_resize.trace")
    except Exception:
        trace_cfg = None
    if trace_cfg is None:
        try:
            trace_cfg = OmegaConf.select(config, "dynamic_resize.trace")
        except Exception:
            trace_cfg = None
    if trace_cfg is None:
        return ResizeTraceConfig()
    try:
        data = OmegaConf.to_container(trace_cfg, resolve=True) or {}
    except Exception:
        data = dict(trace_cfg) if isinstance(trace_cfg, dict) else {}
    path = os.path.abspath(os.path.expanduser(str(data.get("path", "")).strip())) if data.get("path") else ""
    enabled = bool(data.get("enable", False) and path)
    return ResizeTraceConfig(
        enabled=enabled,
        path=path,
        run_name=str(data.get("run_name", "")).strip(),
        flush=bool(data.get("flush", False)),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if dataclass_is_instance(value):
        return _json_safe(asdict(value))
    return repr(value)


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def append_resize_trace(config: ResizeTraceConfig, payload: dict[str, Any]) -> None:
    if not config.enabled or not config.path:
        return
    os.makedirs(os.path.dirname(config.path), exist_ok=True)
    record = {
        "kind": "span",
        "run_name": config.run_name,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        **{str(k): _json_safe(v) for k, v in payload.items()},
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(config.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        if config.flush:
            os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def resize_trace_span(
    config: ResizeTraceConfig,
    event: str,
    *,
    step: int | None = None,
    lane: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    if not config.enabled:
        yield
        return

    started_wall_ns = time.time_ns()
    started_mono_ns = time.monotonic_ns()
    exc_type = None
    try:
        yield
    except Exception as exc:  # pragma: no cover - passthrough with trace side effect
        exc_type = type(exc).__name__
        raise
    finally:
        ended_wall_ns = time.time_ns()
        ended_mono_ns = time.monotonic_ns()
        append_resize_trace(
            config,
            {
                "event": event,
                "step": step,
                "lane": lane or "",
                "ts_start_ns": started_wall_ns,
                "ts_end_ns": ended_wall_ns,
                "dur_ms": round((ended_mono_ns - started_mono_ns) / 1_000_000.0, 3),
                "status": "error" if exc_type else "ok",
                "error_type": exc_type or "",
                "metadata": metadata or {},
            },
        )
