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

from verl.utils.fsdp_resize import FSDPResizeOverlapPlanner, ShardOverlap, compute_1d_shard_range


def test_stage_model_for_resize_does_not_reject_multiple_hostnames(monkeypatch):
    import ast
    import types
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    path = Path(__file__).resolve().parents[2] / "verl/workers/engine/fsdp/transformer_impl.py"
    source = ast.parse(path.read_text())
    engine_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "FSDPEngine")
    engine_class = ast.ClassDef(
        name=engine_class.name,
        bases=[],
        keywords=[],
        body=[node for node in engine_class.body if getattr(node, "name", None) == "stage_model_for_resize"],
        decorator_list=[],
    )
    module = types.ModuleType("_multi_node_resize_staging")
    module.__dict__.update(
        torch=torch,
        device_name="cuda",
        get_device_id=lambda: 0,
        stage_fsdp2_model_state=lambda *args, **kwargs: ({}, {"tensor_count": 0, "staged_bytes": 0}),
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[engine_class], type_ignores=[])), str(path), "exec"),
        module.__dict__,
    )
    engine = object.__new__(module.FSDPEngine)
    engine.engine_config = SimpleNamespace(strategy="fsdp2", ulysses_sequence_parallel_size=1, fsdp_size=-1)
    engine.model_config = SimpleNamespace(lora={"rank": 0}, lora_adapter_path=None)
    engine.module = object()
    engine._is_lora = False

    def gather_hosts(values, value):
        values[:] = ["node-a", "node-b"]

    calls = []
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "new_group", lambda **kwargs: calls.append(kwargs) or "transfer-group")
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_hosts)
    result = engine.stage_model_for_resize([0], [0, 1])

    assert result["status"] == "staged"
    assert calls == [{"ranks": [0, 1]}]
    assert engine._resize_model_state == {}


def test_staging_regression_restores_distributed_api_spies():
    import torch

    with pytest.raises((RuntimeError, ValueError), match="Default process group has not been initialized"):
        torch.distributed.get_world_size()


@pytest.mark.parametrize(
    ("dim_size", "world_size", "expected"),
    [
        (12, 4, [(0, 3), (3, 3), (6, 3), (9, 3)]),
        (12, 6, [(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)]),
        (3, 5, [(0, 1), (1, 1), (2, 1), (3, 0), (3, 0)]),
    ],
)
def test_compute_1d_shard_range_matches_dtensor_chunking(dim_size, world_size, expected):
    assert [compute_1d_shard_range(dim_size, world_size, rank) for rank in range(world_size)] == expected


@pytest.mark.parametrize(
    ("dim_size", "world_size", "rank"),
    [(-1, 4, 0), (8, 0, 0), (8, 4, -1), (8, 4, 4)],
)
def test_compute_1d_shard_range_rejects_invalid_inputs(dim_size, world_size, rank):
    with pytest.raises(ValueError):
        compute_1d_shard_range(dim_size, world_size, rank)


def test_overlap_planner_computes_exact_4_to_6_reshard_plan():
    planner = FSDPResizeOverlapPlanner(dim_size=12, old_world_size=4, new_world_size=6)

    assert planner.send_row_splits(0) == [2, 1, 0, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 1, 2, 0, 0, 0]
    assert planner.send_row_splits(2) == [0, 0, 0, 2, 1, 0]
    assert planner.send_row_splits(3) == [0, 0, 0, 0, 1, 2]

    assert planner.recv_row_splits(0) == [2, 0, 0, 0, 0, 0]
    assert planner.recv_row_splits(1) == [1, 1, 0, 0, 0, 0]
    assert planner.recv_row_splits(2) == [0, 2, 0, 0, 0, 0]
    assert planner.recv_row_splits(3) == [0, 0, 2, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 1, 1, 0, 0]
    assert planner.recv_row_splits(5) == [0, 0, 0, 2, 0, 0]

    assert planner.overlap(0, 1) == ShardOverlap(
        source_rank=0,
        target_rank=1,
        global_offset=2,
        source_offset=2,
        target_offset=0,
        size=1,
    )
    assert planner.overlap(1, 1) == ShardOverlap(
        source_rank=1,
        target_rank=1,
        global_offset=3,
        source_offset=0,
        target_offset=1,
        size=1,
    )


@pytest.mark.parametrize(
    ("dim_size", "old_world_size", "new_world_size"),
    [
        (12, 4, 6),
        (13, 4, 6),
        (3, 2, 5),
        (0, 4, 6),
    ],
)
def test_overlap_planner_conserves_source_and_target_rows(dim_size, old_world_size, new_world_size):
    planner = FSDPResizeOverlapPlanner(dim_size, old_world_size, new_world_size)

    for source_rank, source_interval in enumerate(planner.source_intervals):
        assert sum(planner.send_row_splits(source_rank)) == source_interval.size

    for target_rank, target_interval in enumerate(planner.target_intervals):
        assert sum(planner.recv_row_splits(target_rank)) == target_interval.size

    assert sum(sum(planner.send_row_splits(rank)) for rank in range(old_world_size)) == dim_size
    assert sum(sum(planner.recv_row_splits(rank)) for rank in range(new_world_size)) == dim_size


def test_overlap_planner_handles_uneven_and_empty_target_shards():
    planner = FSDPResizeOverlapPlanner(dim_size=3, old_world_size=2, new_world_size=5)

    assert planner.send_row_splits(0) == [1, 1, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 0, 1, 0, 0]
    assert planner.recv_row_splits(3) == [0, 0, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 0, 0, 0]
