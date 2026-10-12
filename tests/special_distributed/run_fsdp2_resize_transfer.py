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

"""Focused 4-rank to 6-rank FSDP2 model-transfer smoke test.

Run with:
    torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from verl.utils.fsdp_resize import restore_fsdp2_model_state, stage_fsdp2_model_state


def _build_model() -> nn.Module:
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(4, 12), nn.GELU(), nn.Linear(12, 3, bias=False))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) + index * 1000)
    return model.cuda()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    old_ranks = list(range(4))
    new_ranks = list(range(6))

    old_group = dist.new_group(ranks=old_ranks)
    old_model = None
    if rank in old_ranks:
        old_mesh = DeviceMesh.from_group(
            old_group,
            device_type="cuda",
            mesh=torch.arange(len(old_ranks)),
            mesh_dim_names=["fsdp"],
        )
        old_model = _build_model()
        fully_shard(old_model, mesh=old_mesh)

    staged_state, _ = stage_fsdp2_model_state(
        old_model,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        transfer_group=dist.group.WORLD,
        device=torch.device("cuda", local_rank),
    )
    assert staged_state is not None

    del old_model
    if rank in old_ranks:
        dist.destroy_process_group(old_group)
    torch.cuda.empty_cache()
    dist.barrier()

    new_mesh = DeviceMesh.from_group(
        dist.group.WORLD,
        device_type="cuda",
        mesh=torch.arange(len(new_ranks)),
        mesh_dim_names=["fsdp"],
    )
    new_model = _build_model()
    with torch.no_grad():
        for parameter in new_model.parameters():
            parameter.zero_()
    fully_shard(new_model, mesh=new_mesh)
    restore_fsdp2_model_state(new_model, staged_state)

    expected_model = _build_model()
    expected_state = expected_model.state_dict()
    actual_state = new_model.state_dict()
    for name, value in actual_state.items():
        actual = value.full_tensor() if hasattr(value, "full_tensor") else value
        torch.testing.assert_close(actual, expected_state[name])

    dist.barrier()
    if rank == 0:
        print("FSDP2 GPU-direct resize transfer 4->6 passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
