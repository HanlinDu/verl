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
import torch
import torch.distributed as dist

import verl.utils.fsdp_resize as fsdp_resize
from verl.utils.distributed import clear_dtensor_dispatch_caches
from verl.utils.fsdp_resize import FSDPResizePlanner, ResizeRuntime, compute_1d_shard_range, transfer_fsdp2_runtime


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


def test_expand_4_to_6_plan_is_exact_and_non_redundant():
    planner = FSDPResizePlanner(12, old_ranks=[0, 1, 2, 3], new_ranks=[0, 1, 2, 3, 4, 5])

    assert planner.group_ranks == [0, 1, 2, 3, 4, 5]
    assert planner.send_row_splits(0) == [2, 1, 0, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 1, 2, 0, 0, 0]
    assert planner.send_row_splits(2) == [0, 0, 0, 2, 1, 0]
    assert planner.send_row_splits(3) == [0, 0, 0, 0, 1, 2]

    assert planner.recv_row_splits(0) == [2, 0, 0, 0, 0, 0]
    assert planner.recv_row_splits(1) == [1, 1, 0, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 1, 1, 0, 0]
    assert planner.recv_row_splits(5) == [0, 0, 0, 2, 0, 0]

    # rank 0 keeps everything it needs; new members need their whole shard
    assert planner.network_recv_rows(0) == 0
    assert planner.network_recv_rows(1) == 1
    assert planner.network_recv_rows(4) == 2
    assert sum(planner.network_recv_rows(r) for r in planner.new_ranks) == 9  # 3/4 of 12 rows


def test_shrink_6_to_2_plan_uses_the_union_group():
    planner = FSDPResizePlanner(12, old_ranks=[0, 1, 2, 3, 4, 5], new_ranks=[0, 1])

    assert planner.group_ranks == [0, 1, 2, 3, 4, 5]
    # old rank 2 holds rows [4, 6) which belong to new rank 0
    assert planner.send_row_splits(2) == [2, 0, 0, 0, 0, 0]
    # old rank 5 holds rows [10, 12) which belong to new rank 1
    assert planner.send_row_splits(5) == [0, 2, 0, 0, 0, 0]
    # new rank 1 needs rows [6, 12) from old ranks 3, 4, 5
    assert planner.recv_row_splits(1) == [0, 0, 0, 2, 2, 2]
    assert planner.recv_row_splits(0) == [2, 2, 2, 0, 0, 0]
    assert planner.network_recv_rows(0) == 4  # keeps its own [0, 2)
    assert planner.network_recv_rows(1) == 6  # its old rows [2, 4) are not part of its new shard


def test_non_prefix_rank_sets_are_indexed_by_mesh_position():
    planner = FSDPResizePlanner(8, old_ranks=[2, 3], new_ranks=[1, 2, 3, 4])

    assert planner.group_ranks == [1, 2, 3, 4]
    # old rank 2 is mesh position 0 -> rows [0, 4); new rank 1 is position 0 -> [0, 2), new rank 2 -> [2, 4)
    assert planner.send_row_splits(2) == [2, 2, 0, 0]
    assert planner.send_row_splits(3) == [0, 0, 2, 2]
    assert planner.recv_row_splits(1) == [0, 2, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 2, 0]
    # ranks 2 and 3 keep rows they already hold; ranks 1 and 4 are new members
    assert planner.network_recv_rows(2) == 0
    assert planner.network_recv_rows(3) == 0
    assert planner.network_recv_rows(1) == 2
    assert planner.network_recv_rows(4) == 2


@pytest.mark.parametrize(
    ("dim_size", "old_ranks", "new_ranks"),
    [
        (12, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (13, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (13, [0, 1, 2, 3, 4, 5], [0, 1]),
        (3, [0, 1], [0, 1, 2, 3, 4]),
        (0, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (7, [4, 5], [0, 1, 2]),
    ],
)
def test_planner_conserves_rows(dim_size, old_ranks, new_ranks):
    planner = FSDPResizePlanner(dim_size, old_ranks, new_ranks)

    for index, rank in enumerate(old_ranks):
        assert sum(planner.send_row_splits(rank)) == planner.source_intervals[index].size
    for index, rank in enumerate(new_ranks):
        assert sum(planner.recv_row_splits(rank)) == planner.target_intervals[index].size
    assert sum(sum(planner.send_row_splits(r)) for r in old_ranks) == dim_size
    assert sum(sum(planner.recv_row_splits(r)) for r in new_ranks) == dim_size
    # what one side sends is exactly what the other side receives, pairwise
    for s, src in enumerate(planner.group_ranks):
        for t, dst in enumerate(planner.group_ranks):
            sent = planner.send_row_splits(src)[t] if src in old_ranks else 0
            received = planner.recv_row_splits(dst)[s] if dst in new_ranks else 0
            assert sent == received


def test_planner_handles_uneven_and_empty_target_shards():
    planner = FSDPResizePlanner(3, old_ranks=[0, 1], new_ranks=[0, 1, 2, 3, 4])

    assert planner.send_row_splits(0) == [1, 1, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 0, 1, 0, 0]
    assert planner.recv_row_splits(3) == [0, 0, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("old_ranks,new_ranks", [([], [0]), ([0], []), ([1, 0], [0, 1]), ([0, 1], [0, 0, 1])])
def test_planner_rejects_empty_or_unsorted_rank_lists(old_ranks, new_ranks):
    with pytest.raises(ValueError):
        FSDPResizePlanner(8, old_ranks, new_ranks)


@pytest.fixture
def one_rank_world():
    """A one-rank Gloo world on CPU; nothing is left behind for other tests."""
    if dist.is_initialized():
        pytest.skip("needs to own the default process group")
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        clear_dtensor_dispatch_caches()


def _plain_runtime(steps: int) -> ResizeRuntime:
    """An unsharded Linear with AdamW and an LR schedule, after ``steps`` optimizer steps."""
    torch.manual_seed(0)
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + 0.1 * step))
    for _ in range(steps):
        module(torch.ones(4, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return ResizeRuntime(module=module, optimizer=optimizer, lr_scheduler=scheduler)


@pytest.fixture
def one_rank_transfer(one_rank_world, monkeypatch):
    """``transfer_fsdp2_runtime`` from rank 0 to rank 0 of a one-rank Gloo world.

    Unsharded tensors move by broadcast, which Gloo supports, so the whole transfer runs without an
    accelerator; only the final device synchronisation has to be pointed at the CPU.
    """
    monkeypatch.setattr(fsdp_resize, "get_torch_device", lambda: torch.cpu)

    def transfer(old, new, **kwargs):
        return transfer_fsdp2_runtime(
            old=old, new=new, old_ranks=[0], new_ranks=[0], group=dist.group.WORLD, device=torch.device("cpu"), **kwargs
        )

    return transfer


def test_scalar_state_is_the_training_state_that_does_not_travel_as_a_tensor():
    state = fsdp_resize._scalar_state(_plain_runtime(steps=2))

    lr = repr(1e-2 / 1.2)
    assert state["optimizer.state[weight].step"] == "2.0 (float32 tensor on cpu)"
    assert state["optimizer.state[bias].step"] == "2.0 (float32 tensor on cpu)"
    assert state["optimizer.param_groups[0].lr"] == f"{lr} (float)"
    assert state["optimizer.param_groups[0].betas[1]"] == "0.999 (float)"
    assert state["optimizer.param_groups[0].amsgrad"] == "False (bool)"
    assert state["lr_scheduler.last_epoch"] == "2 (int)"
    assert state["lr_scheduler._last_lr[0]"] == f"{lr} (float)"
    assert state["lr_scheduler.lr_lambdas[0]"] == "None (NoneType)"
    # the moments have rows: they move as tensors and are covered by the checksums
    assert not any("exp_avg" in path for path in state)
    # the parameters of a group are not hyper-parameters
    assert not any(path.startswith("optimizer.param_groups[0].params") for path in state)


def test_verified_transfer_compares_the_scalar_state_of_both_sides(one_rank_transfer):
    old, new = _plain_runtime(steps=2), _plain_runtime(steps=0)
    expected = fsdp_resize._scalar_state(old)

    stats = one_rank_transfer(old, new, verify=True)

    assert fsdp_resize._scalar_state(new) == expected
    assert stats["verified"] == 1
    assert stats["verified_scalar_count"] == len(expected) > 0


def _wrong_step(manifest):
    manifest["optimizer"]["state"]["bias"]["scalars"]["step"]["value"] += 1.0


def _wrong_lr(manifest):
    manifest["optimizer"]["param_groups"][0]["lr"] *= 2.0


def _wrong_scheduler(manifest):
    manifest["lr_scheduler"]["last_epoch"] += 1


def _missing_scalar(manifest):
    del manifest["optimizer"]["state"]["bias"]["scalars"]["step"]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_wrong_step, r"optimizer\.state\[bias\]\.step is 3\.0 \(float32 tensor on cpu\) on new rank 0, 2\.0 "),
        (_wrong_lr, r"optimizer\.param_groups\[0\]\.lr is 0\.0166\d+ \(float\) on new rank 0, 0\.0083\d+ "),
        (_wrong_scheduler, r"lr_scheduler\.last_epoch is 3 \(int\) on new rank 0, 2 \(int\) on old rank 0"),
        (_missing_scalar, r"optimizer\.state\[bias\]\.step is missing on new rank 0, 2\.0 "),
    ],
)
def test_verified_transfer_names_scalar_state_that_arrived_wrong(one_rank_transfer, monkeypatch, damage, message):
    """The checksums cover tensors with rows. ``step``, the hyper-parameters and the scheduler state travel
    in the manifest, so a manifest that is wrong about one of them has to be caught by reading both sides."""
    build_manifest = fsdp_resize._build_manifest

    def lying_manifest(runtime, **kwargs):
        manifest = build_manifest(runtime, **kwargs)
        damage(manifest)
        return manifest

    monkeypatch.setattr(fsdp_resize, "_build_manifest", lying_manifest)

    with pytest.raises(RuntimeError, match=message):
        one_rank_transfer(_plain_runtime(steps=2), _plain_runtime(steps=0), verify=True)


def test_unverified_transfer_does_not_read_the_scalar_state(one_rank_transfer, monkeypatch):
    def unexpected(runtime):
        raise AssertionError("scalar state collected although verification is off")

    monkeypatch.setattr(fsdp_resize, "_scalar_state", unexpected)
    old, new = _plain_runtime(steps=2), _plain_runtime(steps=0)

    stats = one_rank_transfer(old, new, verify=False)

    assert stats["verified"] == 0
    assert stats["verified_scalar_count"] == 0
    assert float(new.optimizer.state[new.module.bias]["step"]) == 2.0
    assert new.lr_scheduler.last_epoch == 2
