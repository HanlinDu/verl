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
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

TRACE_UTILS_PATH = Path(__file__).resolve().parents[3] / "verl/experimental/one_step_off_policy/trace_utils.py"


def _load_trace_utils():
    # trace_utils is stdlib-only apart from a configuration-time OmegaConf import.
    # Keep this test independent of the full verl/trainer import stack.
    stub = None
    if importlib.util.find_spec("omegaconf") is None:
        stub = ModuleType("omegaconf")
        stub.OmegaConf = type("OmegaConf", (), {})
        sys.modules["omegaconf"] = stub
    spec = importlib.util.spec_from_file_location("trace_utils_under_test", TRACE_UTILS_PATH)
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
        if stub is not None:
            sys.modules.pop("omegaconf", None)
    return module


trace_utils = _load_trace_utils()


def _enabled_config(path):
    return trace_utils.ResizeTraceConfig(enabled=True, path=str(path), run_name="unit")


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_warned(caplog):
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


class UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("error text is not printable")


def test_successful_span_records_empty_error_message(tmp_path):
    path = tmp_path / "spans.jsonl"
    with trace_utils.resize_trace_span(
        _enabled_config(path),
        "resize_hard_switch_rebuild",
        step=3,
        lane="actor",
        metadata={"target_actor_size": 4},
    ):
        pass

    [record] = _read_records(path)
    assert record["status"] == "ok"
    assert record["error_type"] == ""
    assert record["error_message"] == ""
    assert record["event"] == "resize_hard_switch_rebuild"
    assert record["metadata"] == {"target_actor_size": 4}


def test_failed_span_keeps_original_exception_and_records_message(tmp_path):
    path = tmp_path / "spans.jsonl"
    error = RuntimeError("original failure")

    with pytest.raises(RuntimeError) as excinfo:
        with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch_destroy", step=3):
            raise error

    assert excinfo.value is error
    [record] = _read_records(path)
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "original failure"


def test_unprintable_exception_text_does_not_replace_original_exception(tmp_path, caplog):
    path = tmp_path / "spans.jsonl"
    error = UnprintableError()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(UnprintableError) as excinfo:
            with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch"):
                raise error

    assert excinfo.value is error
    [record] = _read_records(path)
    assert record["status"] == "error"
    assert record["error_type"] == "UnprintableError"
    assert isinstance(record["error_message"], str)
    assert "UnprintableError" in record["error_message"]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interruptions_are_recorded_as_errors_and_reraised_unchanged(tmp_path, error_type):
    path = tmp_path / "spans.jsonl"
    error = error_type("stop now")

    with pytest.raises(error_type) as excinfo:
        with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch_update_weights"):
            raise error

    assert excinfo.value is error
    [record] = _read_records(path)
    assert record["status"] == "error"
    assert record["error_type"] == error_type.__name__
    assert record["error_message"] == "stop now"


def test_long_error_message_keeps_2000_character_prefix_and_truncation_marker(tmp_path):
    path = tmp_path / "spans.jsonl"
    prefix = "x" * 2000
    error = RuntimeError(prefix + "TAIL" * 500)

    with pytest.raises(RuntimeError):
        with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch"):
            raise error

    [record] = _read_records(path)
    message = record["error_message"]
    assert message.startswith(prefix)
    assert len(message) > len(prefix)
    assert "TAIL" not in message
    assert "truncat" in message.lower()


def test_trace_write_failure_does_not_break_successful_operation(tmp_path, caplog):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    path = blocker / "spans.jsonl"
    body_ran = False

    with caplog.at_level(logging.WARNING):
        with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch_rebuild", step=3):
            body_ran = True

    assert body_ran
    assert not path.exists()
    _assert_warned(caplog)


def test_trace_write_failure_does_not_replace_failed_operation(tmp_path, caplog):
    path = tmp_path / "spans.jsonl"
    path.mkdir()
    error = RuntimeError("business failure")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as excinfo:
            with trace_utils.resize_trace_span(_enabled_config(path), "resize_hard_switch_rebuild", step=3):
                raise error

    assert excinfo.value is error
    _assert_warned(caplog)


def test_append_resize_trace_write_failure_is_best_effort(tmp_path, caplog):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        trace_utils.append_resize_trace(
            _enabled_config(blocker / "schedule.jsonl"),
            {"event": "resize_schedule", "step": 3},
        )

    _assert_warned(caplog)


def test_disabled_trace_does_nothing_and_preserves_failure(tmp_path):
    path = tmp_path / "missing" / "spans.jsonl"
    config = trace_utils.ResizeTraceConfig(enabled=False, path=str(path), run_name="unit")
    error = RuntimeError("business failure")

    with trace_utils.resize_trace_span(config, "resize_hard_switch"):
        pass
    with pytest.raises(RuntimeError) as excinfo:
        with trace_utils.resize_trace_span(config, "resize_hard_switch"):
            raise error

    assert excinfo.value is error
    assert not path.exists()
    assert not path.parent.exists()
