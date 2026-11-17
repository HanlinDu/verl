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

import importlib.util
import os
import sys
from pathlib import Path

MEMORY_PROBE_PATH = Path(__file__).resolve().parents[3] / "verl/experimental/one_step_off_policy/memory_probe.py"


def _load_memory_probe():
    # memory_probe is a stdlib-only helper. Load it by path so these behavior
    # tests do not import the full verl/trainer dependency stack.
    spec = importlib.util.spec_from_file_location("memory_probe_under_test", MEMORY_PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


memory_probe = _load_memory_probe()

EMPTY_DRIVER = {"free_bytes": None, "total_bytes": None, "used_bytes": None}
EMPTY_PROCESS = {"allocated_bytes": None, "reserved_bytes": None}


class _FakeDeviceModule:
    def __init__(
        self,
        *,
        available=True,
        free=60,
        total=100,
        allocated=7,
        reserved=11,
        mem_get_info_error=None,
        memory_allocated_error=None,
        memory_reserved_error=None,
    ):
        self.available = available
        self.free = free
        self.total = total
        self.allocated = allocated
        self.reserved = reserved
        self.mem_get_info_error = mem_get_info_error
        self.memory_allocated_error = memory_allocated_error
        self.memory_reserved_error = memory_reserved_error

    def is_available(self):
        return self.available

    def mem_get_info(self):
        if self.mem_get_info_error is not None:
            raise self.mem_get_info_error
        return self.free, self.total

    def memory_allocated(self):
        if self.memory_allocated_error is not None:
            raise self.memory_allocated_error
        return self.allocated

    def memory_reserved(self):
        if self.memory_reserved_error is not None:
            raise self.memory_reserved_error
        return self.reserved


def _expected(physical_rank, node_id, accelerator_id_raw, membership=None):
    return {
        "physical_rank": physical_rank,
        "node_id": node_id,
        "accelerator_id_raw": accelerator_id_raw,
        "membership": membership if membership is not None else {"actor": True, "rollout_dormant": False},
    }


def _sample(
    physical_rank,
    node_id,
    accelerator_id_raw,
    *,
    free=60,
    total=100,
    used=40,
    allocated=7,
    reserved=11,
    status="ok",
):
    return {
        "physical_rank": physical_rank,
        "node_id": node_id,
        "accelerator_id_raw": accelerator_id_raw,
        "process_scope": "hybrid_worker",
        "sampled_at_ns": 1_700_000_000_000_000_000,
        "pid": 4242,
        "driver": {"free_bytes": free, "total_bytes": total, "used_bytes": used},
        "process_torch": {"allocated_bytes": allocated, "reserved_bytes": reserved},
        "status": status,
        "error_type": "",
        "error_message": "",
    }


def _identity(entry):
    return (entry.get("physical_rank"), entry.get("node_id"), entry.get("accelerator_id_raw"))


def _errors_of_kind(summary, kind):
    return [error for error in summary["errors"] if error.get("kind") == kind]


def test_capture_reports_identity_driver_and_hybrid_process_counters():
    snapshot = memory_probe.capture_memory_snapshot(
        physical_rank=3,
        node_id="node-a",
        accelerator_id_raw="0",
        device_module=_FakeDeviceModule(),
    )

    assert snapshot["physical_rank"] == 3
    assert snapshot["node_id"] == "node-a"
    assert snapshot["accelerator_id_raw"] == "0"
    assert isinstance(snapshot["accelerator_id_raw"], str)
    assert snapshot["process_scope"] == "hybrid_worker"
    assert isinstance(snapshot["sampled_at_ns"], int)
    assert snapshot["sampled_at_ns"] > 0
    assert snapshot["pid"] == os.getpid()
    assert snapshot["driver"] == {"free_bytes": 60, "total_bytes": 100, "used_bytes": 40}
    assert snapshot["process_torch"] == {"allocated_bytes": 7, "reserved_bytes": 11}
    assert snapshot["status"] == "ok"
    assert snapshot["error_type"] == ""
    assert snapshot["error_message"] == ""


def test_capture_without_device_reports_error_and_no_zero_counters():
    snapshot = memory_probe.capture_memory_snapshot(
        physical_rank=0,
        node_id="node-cpu",
        accelerator_id_raw="",
        device_module=_FakeDeviceModule(
            available=False,
            mem_get_info_error=RuntimeError("no accelerator is available"),
        ),
    )

    assert snapshot["physical_rank"] == 0
    assert snapshot["node_id"] == "node-cpu"
    assert snapshot["accelerator_id_raw"] == ""
    assert snapshot["status"] == "error"
    assert snapshot["error_type"] == "RuntimeError"
    assert snapshot["error_message"]
    assert snapshot["driver"] == EMPTY_DRIVER
    assert snapshot["process_torch"] == EMPTY_PROCESS


def test_capture_partial_read_failure_stays_incomplete_and_keeps_read_values():
    snapshot = memory_probe.capture_memory_snapshot(
        physical_rank=2,
        node_id="node-a",
        accelerator_id_raw="1",
        device_module=_FakeDeviceModule(
            memory_reserved_error=RuntimeError("reserved counter unavailable"),
        ),
    )

    assert snapshot["status"] == "error"
    assert snapshot["error_type"] == "RuntimeError"
    assert snapshot["error_message"] == "reserved counter unavailable"
    assert snapshot["driver"] == {"free_bytes": 60, "total_bytes": 100, "used_bytes": 40}
    assert snapshot["process_torch"]["allocated_bytes"] == 7
    assert snapshot["process_torch"]["reserved_bytes"] is None


def test_summary_complete_for_two_nodes_with_identical_local_accelerator_id():
    expected = [
        _expected(0, "node-a", "0", {"actor": True, "rollout_dormant": False}),
        _expected(1, "node-b", "0", {"actor": True, "rollout_dormant": True}),
    ]
    observed = [
        _sample(1, "node-b", "0", free=20, total=100, used=80, allocated=9, reserved=13),
        _sample(0, "node-a", "0"),
    ]

    summary = memory_probe.summarize_memory_snapshot(expected, observed, [])

    assert summary["complete"] is True
    assert summary["expected_count"] == 2
    assert summary["observed_count"] == 2
    assert summary["errors"] == []
    assert summary["expected"] == expected
    assert sorted(_identity(row) for row in summary["observed"]) == sorted(_identity(row) for row in observed)


def test_summary_missing_sample_is_incomplete():
    expected = [_expected(0, "node-a", "0"), _expected(1, "node-b", "1")]
    observed = [_sample(0, "node-a", "0")]

    summary = memory_probe.summarize_memory_snapshot(expected, observed, [])

    assert summary["complete"] is False
    assert summary["expected_count"] == 2
    assert summary["observed_count"] == 1
    missing = _errors_of_kind(summary, "missing")
    assert len(missing) == 1
    assert _identity(missing[0]) == (1, "node-b", "1")


def test_summary_duplicate_sample_keeps_row_count_and_is_incomplete():
    expected = [_expected(0, "node-a", "0"), _expected(1, "node-b", "1")]
    observed = [_sample(0, "node-a", "0"), _sample(0, "node-a", "0")]

    summary = memory_probe.summarize_memory_snapshot(expected, observed, [])

    assert summary["complete"] is False
    assert summary["expected_count"] == 2
    assert summary["observed_count"] == 2
    assert len(summary["observed"]) == 2
    duplicate = _errors_of_kind(summary, "duplicate")
    assert len(duplicate) == 1
    assert _identity(duplicate[0]) == (0, "node-a", "0")


def test_summary_wrong_card_at_expected_rank_is_incomplete():
    expected = [_expected(0, "node-a", "0")]
    observed = [_sample(0, "node-a", "1")]

    summary = memory_probe.summarize_memory_snapshot(expected, observed, [])

    assert summary["complete"] is False
    assert summary["expected_count"] == 1
    assert summary["observed_count"] == 1
    missing = _errors_of_kind(summary, "missing")
    unexpected = _errors_of_kind(summary, "unexpected")
    assert [_identity(error) for error in missing] == [(0, "node-a", "0")]
    assert [_identity(error) for error in unexpected] == [(0, "node-a", "1")]


def test_summary_worker_error_is_incomplete_and_preserved():
    expected = [_expected(0, "node-a", "0")]
    worker_error = {
        "physical_rank": 0,
        "node_id": "node-a",
        "accelerator_id_raw": "0",
        "error_type": "TimeoutError",
        "error_message": "probe timed out",
    }

    summary = memory_probe.summarize_memory_snapshot(expected, [], [worker_error])

    assert summary["complete"] is False
    assert summary["expected_count"] == 1
    assert summary["observed_count"] == 0
    worker_errors = _errors_of_kind(summary, "worker")
    assert len(worker_errors) == 1
    assert worker_errors[0]["error_type"] == "TimeoutError"
    assert worker_errors[0]["error_message"] == "probe timed out"


def test_summary_invalid_counters_are_incomplete_without_raising():
    expected = [_expected(0, "node-a", "0"), _expected(1, "node-b", "1")]
    observed = [
        _sample(0, "node-a", "0", free=101, total=100, used=0),
        _sample(1, "node-b", "1", reserved="11"),
    ]

    summary = memory_probe.summarize_memory_snapshot(expected, observed, [])

    assert summary["complete"] is False
    assert summary["expected_count"] == 2
    assert summary["observed_count"] == 2
    assert len(summary["observed"]) == 2
    counter_errors = _errors_of_kind(summary, "counter")
    assert {error["physical_rank"] for error in counter_errors} == {0, 1}
