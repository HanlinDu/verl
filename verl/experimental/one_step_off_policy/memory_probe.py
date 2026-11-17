# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-wide counters and truthful completeness for optional hybrid probes."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any


def _error_text(exc: Exception) -> str:
    try:
        return str(exc)[:2000]
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


def capture_memory_snapshot(*, physical_rank: int, node_id: str, accelerator_id_raw: str, device_module: Any) -> dict:
    """Read the current device without synchronization; never fabricate missing counters.

    Driver values include every process on the card. process_torch belongs to
    the hybrid worker only, not the separately hosted vLLM allocator.
    """
    result = {
        "physical_rank": physical_rank,
        "node_id": node_id,
        "accelerator_id_raw": accelerator_id_raw,
        "process_scope": "hybrid_worker",
        "sampled_at_ns": time.time_ns(),
        "pid": os.getpid(),
        "driver": {"free_bytes": None, "total_bytes": None, "used_bytes": None},
        "process_torch": {"allocated_bytes": None, "reserved_bytes": None},
        "status": "error",
        "error_type": "",
        "error_message": "",
    }
    try:
        if not device_module.is_available():
            raise RuntimeError("device memory counters unavailable")
        if not node_id or not accelerator_id_raw:
            raise RuntimeError("physical accelerator identity unavailable")
        free, total = device_module.mem_get_info()
        result["driver"] = {"free_bytes": free, "total_bytes": total, "used_bytes": total - free}
        result["process_torch"]["allocated_bytes"] = device_module.memory_allocated()
        result["process_torch"]["reserved_bytes"] = device_module.memory_reserved()
        if not _valid_counters(result):
            raise ValueError("invalid device memory counters")
        result["status"] = "ok"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = _error_text(exc)
    return result


def _identity(row: Any) -> tuple | None:
    if not isinstance(row, dict):
        return None
    rank, node, accelerator = (row.get(key) for key in ("physical_rank", "node_id", "accelerator_id_raw"))
    if type(rank) is not int or rank < 0 or not isinstance(node, str) or not node:
        return None
    if not isinstance(accelerator, str) or not accelerator:
        return None
    return rank, node, accelerator


def _valid_counters(row: dict) -> bool:
    driver, process = row.get("driver"), row.get("process_torch")
    if not isinstance(driver, dict) or not isinstance(process, dict):
        return False
    values = [driver.get(key) for key in ("free_bytes", "total_bytes", "used_bytes")]
    values += [process.get(key) for key in ("allocated_bytes", "reserved_bytes")]
    if any(type(value) is not int or value < 0 for value in values):
        return False
    free, total, used, allocated, reserved = values
    return total > 0 and free <= total and used == total - free and allocated <= reserved


def summarize_memory_snapshot(expected: list[dict], observed: list[dict], errors: list[dict]) -> dict:
    """Validate against an independent allocation map, retaining all raw evidence."""
    failures = []

    def add(kind: str, row: Any, message: str) -> None:
        identity = _identity(row)
        failure = {"kind": kind, "error_type": "InvalidMemoryEvidence", "error_message": message}
        if identity is not None:
            failure.update(zip(("physical_rank", "node_id", "accelerator_id_raw"), identity, strict=True))
        failures.append(failure)

    for error in errors:
        if isinstance(error, dict):
            failures.append({**error, "kind": "worker"})
        else:
            add("worker", error, "malformed sampling error")
    wanted = Counter(_identity(row) for row in expected)
    actual = Counter(_identity(row) for row in observed)
    if not expected:
        add("expected", {}, "physical allocation map unavailable")
    expected_ranks = Counter(identity[0] for identity in wanted if identity is not None)
    expected_cards = Counter(identity[1:] for identity in wanted if identity is not None)
    if None in wanted or any(count != 1 for count in wanted.values()):
        add("expected", {}, "malformed or duplicate expected identity")
    if set(expected_ranks) != set(range(len(expected))) or any(count != 1 for count in expected_ranks.values()):
        add("expected", {}, "expected physical ranks are incomplete or repeated")
    if any(count != 1 for count in expected_cards.values()):
        add("expected", {}, "one physical card assigned to multiple ranks")
    for row in expected:
        identity = _identity(row)
        if identity is not None and actual[identity] == 0:
            add("missing", row, "expected card did not return a sample")
    for identity, count in actual.items():
        if count > 1 and identity is not None:
            add(
                "duplicate",
                dict(zip(("physical_rank", "node_id", "accelerator_id_raw"), identity, strict=True)),
                "multiple samples for one physical rank/card",
            )
    for row in observed:
        identity = _identity(row)
        if identity is None or identity not in wanted:
            add("unexpected", row, "sample does not match an expected physical card")
        if not isinstance(row, dict):
            continue
        if row.get("status") != "ok":
            failures.append(
                {
                    **{
                        key: row.get(key)
                        for key in ("physical_rank", "node_id", "accelerator_id_raw", "error_type", "error_message")
                    },
                    "kind": "worker",
                }
            )
        elif not _valid_counters(row) or row.get("process_scope") != "hybrid_worker":
            add("counter", row, "invalid counters or process scope")
    return {
        "expected": expected,
        "observed": observed,
        "errors": failures,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "complete": bool(expected) and not failures and len(observed) == len(expected),
    }
