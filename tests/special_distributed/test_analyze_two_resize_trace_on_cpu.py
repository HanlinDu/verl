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


def test_two_restored_resizes_pass():
    events = [
        {"event": "resize_observation", "metrics": {}},
        _resize(2, 4, 4),
        _resize(4, 6, 2),
    ]
    summary = analyzer.analyze(events, expected_resizes=2, final_actor=6, final_rollout=2)
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
        "\n".join(json.dumps(event) for event in [_resize(2, 4, 4), _resize(4, 6, 2)]) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    code = analyzer.main([str(trace), "--output", str(output)])
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
