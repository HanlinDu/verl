#!/usr/bin/env python3
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

"""Offline verdict for a resize -> sync -> resize run from ``resize_trace.jsonl``.

A run passes only when both direct model handoffs succeed, each resize has an
explicit weight-sync event and subsequent training, the final step completes,
and NCCL channel logs confirm the expected transport class.

Usage:
    python tests/special_distributed/analyze_two_resize_trace.py <trace.jsonl> \
        --expected-resizes 2 --final-actor 6 --final-rollout 2 --nccl-log <nccl.log>

Pure stdlib: no torch, no Ray, no GPU.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

# Load the pure helper without importing verl's training dependencies.
_route = runpy.run_path(str(Path(__file__).resolve().parents[2] / "verl/utils/resize_route.py"))


def _applied_resizes(events: list[dict]) -> list[dict]:
    applied = []
    for event in events:
        if event.get("event") != "resize_schedule":
            continue
        metrics = event.get("metrics") or {}
        if float(metrics.get("resize/schedule_applied", 0.0)) == 1.0:
            applied.append(event)
    return applied


def analyze(
    events: list[dict],
    *,
    expected_resizes: int,
    final_actor: int,
    final_rollout: int,
    expected_final_step: int = 12,
    transports: list[str] | None = None,
    cross_node: bool = False,
) -> dict:
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
            "inter_node_bytes": float(metrics.get("resize/handoff_model_transfer_inter_node_bytes", 0)),
            "host_count": int(metrics.get("resize/handoff_model_transfer_host_count", 1)),
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

        if not any(
            e.get("event") == "resize_weight_sync"
            and e.get("step") == record["step"]
            and e.get("actor_size") == record["actor_size"]
            and e.get("rollout_size") == record["rollout_size"]
            for e in events
        ):
            reasons.append(f"step {record['step']}: no successful resize weight synchronization event")
        if not any(
            e.get("event") == "training_step_end"
            and int(e.get("step", -1)) > int(record["step"])
            and e.get("actor_size") == record["actor_size"]
            and e.get("rollout_size") == record["rollout_size"]
            for e in events
        ):
            reasons.append(f"step {record['step']}: no post-resize training completion event")

    if len(applied) != expected_resizes:
        reasons.append(f"applied {len(applied)} resizes, expected {expected_resizes}")
    steps = [int(r["step"]) for r in resizes]
    if steps != sorted(set(steps)):
        reasons.append("resize steps must be unique and increasing")
    if not any(
        e.get("event") == "training_step_end"
        and e.get("step") == expected_final_step
        and e.get("actor_size") == final_actor
        and e.get("rollout_size") == final_rollout
        for e in events
    ):
        reasons.append(f"training step {expected_final_step} did not complete")
    if not any(e.get("event") == "training_run_end" and e.get("step") == expected_final_step for e in events):
        reasons.append(f"no successful run end at step {expected_final_step}")

    route_status, route_reason = _route["route_verdict"](transports or [], cross_node=cross_node)
    if route_status != PASS:
        reasons.append(f"NCCL route not confirmed: {route_status}: {route_reason}")
    if cross_node and not any(r["host_count"] >= 2 and r["inter_node_bytes"] > 0 for r in resizes):
        reasons.append("no completed actor shard transfer between different hosts")

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
        "route_status": route_status,
        "route_reason": route_reason,
        "scope": (
            "Explicit resize sync, post-resize training, run completion, and NCCL route evidence. "
            "Route tokens do not certify GPUDirect RDMA or full training-state equality."
        ),
    }


def load_trace(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: expected an event object")
            events.append(event)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-resizes", type=int, default=2)
    parser.add_argument("--final-actor", type=int, default=6)
    parser.add_argument("--final-rollout", type=int, default=2)
    parser.add_argument("--expected-final-step", type=int, default=12)
    parser.add_argument("--nccl-log", type=Path, nargs="+", default=[])
    parser.add_argument("--cross-node", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        events = load_trace(args.trace)
        transports = []
        for path in args.nccl_log:
            for line in path.read_text(encoding="utf-8").splitlines():
                match = _route["CHANNEL_RE"].search(line)
                if match:
                    transports.append(match.group(1))
        summary = analyze(
            events,
            expected_resizes=args.expected_resizes,
            final_actor=args.final_actor,
            final_rollout=args.final_rollout,
            expected_final_step=args.expected_final_step,
            transports=transports,
            cross_node=args.cross_node,
        )
    except (OSError, ValueError, TypeError) as exc:
        summary = {"status": UNKNOWN, "reasons": [str(exc)], "applied_resize_count": 0, "resizes": []}

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
