#!/usr/bin/env python3
"""Offline verdict for a resize -> sync -> resize run from ``resize_trace.jsonl``.

The 4d branch emits ``resize_schedule`` events carrying the applied metrics; it
does not emit the later ``resize_hard_switch`` span. A run passes only when the
expected number of resizes were actually applied, each reported a restored
GPU-direct model handoff, and the final logical split matches the schedule.

Usage:
    python tests/special_distributed/analyze_two_resize_trace.py <trace.jsonl> \
        --expected-resizes 2 --final-actor 6 --final-rollout 2

Pure stdlib: no torch, no Ray, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


def _applied_resizes(events: list[dict]) -> list[dict]:
    applied = []
    for event in events:
        if event.get("event") != "resize_schedule":
            continue
        metrics = event.get("metrics") or {}
        if float(metrics.get("resize/schedule_applied", 0.0)) == 1.0:
            applied.append(event)
    return applied


def analyze(events: list[dict], *, expected_resizes: int, final_actor: int, final_rollout: int) -> dict:
    applied = _applied_resizes(events)
    reasons: list[str] = []
    resizes = []
    for event in applied:
        metrics = event.get("metrics") or {}
        record = {
            "step": event.get("step"),
            "actor_size": metrics.get("resize/active_actor_size"),
            "rollout_size": metrics.get("resize/active_rollout_size"),
            "hard_switch_success": metrics.get("resize/hard_switch_success"),
            "model_transfer_status": metrics.get("resize/handoff_model_transfer_status"),
            "model_transfer_success": metrics.get("resize/handoff_model_transfer_success"),
            "fallback_reason": metrics.get("resize/handoff_model_transfer_fallback_reason", ""),
        }
        resizes.append(record)
        if float(record["hard_switch_success"] or 0.0) != 1.0:
            reasons.append(f"step {record['step']}: hard switch did not succeed")
        if record["model_transfer_status"] != "restored":
            reasons.append(
                f"step {record['step']}: model transfer status is "
                f"{record['model_transfer_status']!r}, expected 'restored'"
            )
        if float(record["model_transfer_success"] or 0.0) != 1.0:
            reasons.append(f"step {record['step']}: model transfer success is not 1.0")
        if record["fallback_reason"]:
            reasons.append(f"step {record['step']}: fallback reason {record['fallback_reason']!r}")

    if len(applied) != expected_resizes:
        reasons.append(f"applied {len(applied)} resizes, expected {expected_resizes}")

    final = resizes[-1] if resizes else None
    if final is None:
        reasons.append("no applied resize found")
    else:
        if final["actor_size"] != float(final_actor):
            reasons.append(f"final actor size {final['actor_size']} != {final_actor}")
        if final["rollout_size"] != float(final_rollout):
            reasons.append(f"final rollout size {final['rollout_size']} != {final_rollout}")

    status = PASS if not reasons else FAIL
    return {
        "status": status,
        "reasons": reasons,
        "applied_resize_count": len(applied),
        "resizes": resizes,
        "final_actor_size": final["actor_size"] if final else None,
        "final_rollout_size": final["rollout_size"] if final else None,
        "scope": (
            "Trace only. Weight-sync success after each resize is inferred from the run "
            "reaching the next step, not from a per-sync trace event."
        ),
    }


def load_trace(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-resizes", type=int, default=2)
    parser.add_argument("--final-actor", type=int, default=6)
    parser.add_argument("--final-rollout", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        events = load_trace(args.trace)
    except (OSError, ValueError) as exc:
        summary = {"status": UNKNOWN, "reasons": [str(exc)], "applied_resize_count": 0, "resizes": []}
    else:
        summary = analyze(
            events,
            expected_resizes=args.expected_resizes,
            final_actor=args.final_actor,
            final_rollout=args.final_rollout,
        )

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
