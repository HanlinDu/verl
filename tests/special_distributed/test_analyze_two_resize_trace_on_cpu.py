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

"""Offline analyzer checks for the resize -> sync -> resize trace verdict."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_two_resize_trace as analyzer  # noqa: E402


def _resize(step, actor, rollout, *, status="restored", success=1.0, applied=1.0, fallback=""):
    return {
        "event": "resize_schedule",
        "step": step,
        "metrics": {
            "resize/schedule_applied": applied,
            "resize/hard_switch_success": success,
            "resize/active_actor_size": float(actor),
            "resize/active_rollout_size": float(rollout),
            "resize/handoff_model_transfer_status": status,
            "resize/handoff_model_transfer_success": success,
            "resize/handoff_model_transfer_fallback_reason": fallback,
        },
    }


def _successful_trace():
    return [
        _resize(2, 4, 4),
        {"event": "resize_weight_sync", "step": 2, "actor_size": 4, "rollout_size": 4},
        {"event": "training_step_end", "step": 3, "actor_size": 4, "rollout_size": 4},
        _resize(6, 6, 2),
        {"event": "resize_weight_sync", "step": 6, "actor_size": 6, "rollout_size": 2},
        {"event": "training_step_end", "step": 7, "actor_size": 6, "rollout_size": 2},
        {"event": "training_step_end", "step": 12, "actor_size": 6, "rollout_size": 2},
        {"event": "training_run_end", "step": 12},
    ]


def _analyze(events, **kwargs):
    options = dict(expected_resizes=2, final_actor=6, final_rollout=2, transports=["P2P/CUMEM"])
    options.update(kwargs)
    return analyzer.analyze(events, **options)


def test_two_restored_resizes_pass():
    summary = _analyze(_successful_trace())
    assert summary["status"] == "PASS"
    assert summary["applied_resize_count"] == 2
    assert [r["actor_size"] for r in summary["resizes"]] == [4.0, 6.0]


def test_single_resize_fails_when_two_expected():
    events = [_resize(2, 4, 4)]
    summary = analyzer.analyze(events, expected_resizes=2, final_actor=6, final_rollout=2)
    assert summary["status"] == "FAIL"
    assert any("applied 1 resizes" in reason for reason in summary["reasons"])


def test_disk_fallback_fails():
    events = [_resize(2, 4, 4), _resize(4, 6, 2, status="fallback", success=0.0, fallback="boom")]
    summary = analyzer.analyze(events, expected_resizes=2, final_actor=6, final_rollout=2)
    assert summary["status"] == "FAIL"
    assert any("fallback reason" in reason for reason in summary["reasons"])


def test_wrong_final_split_fails():
    events = [_resize(2, 4, 4), _resize(4, 6, 2)]
    summary = analyzer.analyze(events, expected_resizes=2, final_actor=6, final_rollout=1)
    assert summary["status"] == "FAIL"
    assert any("final rollout size" in reason for reason in summary["reasons"])


def test_unapplied_schedule_is_not_counted():
    events = [_resize(2, 4, 4, applied=0.0), _resize(4, 6, 2)]
    summary = analyzer.analyze(events, expected_resizes=2, final_actor=6, final_rollout=2)
    assert summary["status"] == "FAIL"
    assert summary["applied_resize_count"] == 1


def test_malformed_trace_reports_unknown(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        analyzer.load_trace(path)


def test_cli_writes_summary_and_returns_zero(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(json.dumps(event) for event in _successful_trace()) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    nccl_log = tmp_path / "nccl.log"
    nccl_log.write_text("Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM\n")
    code = analyzer.main([str(trace), "--output", str(output), "--nccl-log", str(nccl_log)])
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_resize_events_without_training_or_sync_fail():
    summary = _analyze([_resize(2, 4, 4), _resize(6, 6, 2)])
    assert summary["status"] == "FAIL"
    assert any("synchronization" in r for r in summary["reasons"])
    assert any("post-resize training" in r for r in summary["reasons"])


def test_missing_terminal_event_fails():
    assert _analyze(_successful_trace()[:-1])["status"] == "FAIL"


def test_unknown_transport_cannot_pass():
    summary = _analyze(_successful_trace(), transports=[])
    assert summary["status"] == "FAIL"
    assert summary["route_status"] == "UNKNOWN"


def test_cross_node_route_without_actor_payload_cannot_pass():
    summary = _analyze(_successful_trace(), transports=["NET/IB/GDRDMA"], cross_node=True)
    assert summary["status"] == "FAIL"
    assert any("different hosts" in r for r in summary["reasons"])


def test_cross_node_requires_completed_shard_payload_evidence():
    events = _successful_trace()
    events[3]["metrics"]["resize/handoff_model_transfer_host_count"] = 2
    events[3]["metrics"]["resize/handoff_model_transfer_inter_node_bytes"] = 4096
    assert _analyze(events, transports=["NET/IB/GDRDMA"], cross_node=True)["status"] == "PASS"
