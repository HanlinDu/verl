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
