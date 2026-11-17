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

import pytest
from omegaconf import OmegaConf

from verl.experimental.one_step_off_policy.worker_init_plan import WorkerCommInitPlan, build_worker_comm_init_plan


def _make_config(*, fsdp_size: int = 4, ulysses_sequence_parallel_size: int = 1, nccl_timeout: int = 321):
    return OmegaConf.create(
        {
            "nccl_timeout": nccl_timeout,
            "actor": {
                "fsdp_config": {"fsdp_size": fsdp_size},
                "ulysses_sequence_parallel_size": ulysses_sequence_parallel_size,
            },
        }
    )


def test_build_worker_comm_init_plan_records_metadata_only_inputs():
    config = _make_config(fsdp_size=8, ulysses_sequence_parallel_size=1, nccl_timeout=123)

    plan = build_worker_comm_init_plan(
        config,
        "actor",
        rank=2,
        world_size=8,
        init_method="tcp://127.0.0.1:12345",
        device_type="cuda",
        nccl_backend="nccl",
    )

    assert plan.role == "actor"
    assert plan.rank == 2
    assert plan.world_size == 8
    assert plan.init_method == "tcp://127.0.0.1:12345"
    assert plan.timeout_seconds == 123
    assert plan.backend == "cpu:gloo,cuda:nccl"
    assert plan.fsdp_size == 8
    assert plan.ulysses_sequence_parallel_size == 1
    assert plan.ulysses_mesh_shape is None


def test_build_worker_comm_init_plan_creates_ulysses_mesh_shape_when_enabled():
    config = _make_config(fsdp_size=4, ulysses_sequence_parallel_size=2)

    plan = build_worker_comm_init_plan(
        config,
        "rollout",
        rank=0,
        world_size=8,
        device_type="cuda",
        nccl_backend="nccl",
    )

    assert plan.ulysses_sequence_parallel_size == 2
    assert plan.ulysses_mesh_shape == (4, 2)


def test_worker_comm_init_plan_roundtrip_preserves_mesh_metadata():
    plan = WorkerCommInitPlan(
        role="actor",
        rank=1,
        world_size=4,
        init_method="env://",
        timeout_seconds=600,
        backend="cpu:gloo,cuda:nccl",
        fsdp_size=4,
        ulysses_sequence_parallel_size=2,
        ulysses_mesh_shape=(2, 2),
    )

    restored = WorkerCommInitPlan.from_dict(plan.to_dict())

    assert restored == plan


def test_build_worker_comm_init_plan_rejects_invalid_ulysses_partition():
    config = _make_config(fsdp_size=4, ulysses_sequence_parallel_size=3)

    with pytest.raises(ValueError, match="world_size must be divisible"):
        build_worker_comm_init_plan(
            config,
            "actor",
            rank=0,
            world_size=8,
            device_type="cuda",
            nccl_backend="nccl",
        )
