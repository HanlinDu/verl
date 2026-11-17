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

import asyncio
import json
from types import SimpleNamespace

import pytest

from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.one_step_off_policy.trace_utils import ResizeTraceConfig


def _scheduled_trainer(tmp_path):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 3
    trainer._dynamic_resize_enabled = True
    trainer._resize_controller = None
    trainer._latest_resize_control_metrics = {}
    trainer._active_dynamic_resize_topology = {"actor_size": 6, "rollout_size": 2}
    trainer.resource_pool_manager = SimpleNamespace(dynamic_resize_topology=SimpleNamespace(total_size=8))
    trainer._normalize_dynamic_resize_schedule = lambda: [{"step": 3, "actor_pool": {"world_size": 4}}]
    trainer._dynamic_resize_hard_switch_enabled = lambda: True
    trainer._resize_trace_config = ResizeTraceConfig(enabled=True, path=str(tmp_path / "trace.jsonl"))
    return trainer


def test_runtime_target_validation_failure_is_recorded_before_switch(tmp_path):
    trainer = _scheduled_trainer(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(trainer._maybe_dynamic_resize())
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert records[-1]["status"] == "error"
    assert records[-1]["error_type"] == "ValueError"
    assert "shared pool size" in records[-1]["error_message"]
    assert not any(record.get("status") == "ok" and record["event"] == "resize_hard_switch" for record in records)
    assert trainer._active_dynamic_resize_topology == {"actor_size": 6, "rollout_size": 2}
