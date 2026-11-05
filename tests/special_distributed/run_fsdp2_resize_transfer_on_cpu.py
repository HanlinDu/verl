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

"""Verify two real Gloo/DTensor transfers with a CPU synchronization adapter.

Run from the repository root with PYTHONPATH=. and:
    python -m torch.distributed.run --standalone --nproc-per-node=4 \
        tests/special_distributed/run_fsdp2_resize_transfer_on_cpu.py

This exercises the transfer helper, not the CUDA FSDP runtime or cross-host NCCL.
"""

from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.utils.dtensor_cache import clear_dtensor_dispatch_caches
from verl.utils.fsdp_resize import restore_fsdp2_model_state, stage_fsdp2_model_state


def _build_model(group, world_size, *, zero=False):
    mesh = DeviceMesh.from_group(group, device_type="cpu", mesh=torch.arange(world_size), mesh_dim_names=["fsdp"])
    module = nn.Module()
    weight = torch.zeros(7, 4) if zero else torch.arange(28, dtype=torch.float32).reshape(7, 4)
    module.register_parameter("weight", nn.Parameter(distribute_tensor(weight, mesh, [Shard(0)])))
    module.register_buffer("running", torch.zeros(2) if zero else torch.tensor([9.25, 1.5]))
    return module


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("this verification requires four local ranks")
    old_size = 2
    old_group = dist.new_group(ranks=list(range(old_size)))
    model = _build_model(old_group, old_size) if rank < old_size else None
    for new_size in (3, 4):
        new_group = dist.new_group(ranks=list(range(new_size)))
        if rank < new_size:
            staged, stats = stage_fsdp2_model_state(
                model,
                old_ranks=list(range(old_size)),
                new_ranks=list(range(new_size)),
                transfer_group=new_group,
                device=torch.device("cpu"),
            )
            assert stats["transfer_host_count"] == 1
            assert stats["inter_node_sent_bytes"] == 0
        dist.barrier()
        model = None
        if rank < old_size:
            dist.destroy_process_group(old_group)
        clear_dtensor_dispatch_caches()
        if rank < new_size:
            model = _build_model(new_group, new_size, zero=True)
            restored = restore_fsdp2_model_state(model, staged)
            torch.testing.assert_close(model.weight.full_tensor(), torch.arange(28, dtype=torch.float32).reshape(7, 4))
            torch.testing.assert_close(model.running, torch.tensor([9.25, 1.5]))
            assert restored["tensor_count"] == 2
        dist.barrier()
        if rank == 0:
            print(f"PASS: local Gloo DTensor transfer {old_size}->{new_size}; uneven shards and replicated buffer")
        old_size, old_group = new_size, new_group
    dist.destroy_process_group()


if __name__ == "__main__":
    # Replace only accelerator synchronization; tensors and collectives are real.
    with patch("verl.utils.fsdp_resize.get_torch_device", return_value=torch.cpu):
        main()
