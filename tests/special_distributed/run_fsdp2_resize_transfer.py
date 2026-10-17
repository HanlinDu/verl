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

"""FSDP2 GPU-direct resize transfer test: 4 -> 6 -> 2 -> 6 ranks with AdamW state.

For every resize the new model is built on the new mesh *without* weights (meta -> fully_shard ->
to_empty) while the old one is still alive, ``transfer_fsdp2_runtime`` fills it in place, then the
old model and groups are destroyed.  Checks:

* parameters, optimizer moments and ``step`` are bitwise identical to the old runtime;
* the tied weight stays tied and the non-persistent buffer is restored;
* training continues on the new mesh (real FSDP all-gather / reduce-scatter on the new group) and
  matches a single-device golden model;
* a rank that keeps rows receives nothing over the network for them (only an on-device copy);
* verification notices scalar state that arrived wrong: with a source manifest that lies about ``step``,
  the learning rate or the scheduler position, every rank raises and names the entry.

Run with:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

import verl.utils.fsdp_resize as fsdp_resize
from verl.utils.distributed import clear_dtensor_dispatch_caches
from verl.utils.fsdp_resize import FSDPResizePlanner, ResizeRuntime, transfer_fsdp2_runtime

VOCAB, D, HID, B, T, LR = 37, 24, 40, 12, 5, 1e-2


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.l1 = nn.Linear(D, HID)
        self.l2 = nn.Linear(HID, D, bias=False)
        self.norm = nn.LayerNorm(D)
        self.scale = nn.Parameter(torch.ones(5))  # dim0=5 -> empty shards at world size 6
        self.register_buffer("inv_freq", (torch.arange(D, dtype=torch.float32) + 1) / D, persistent=False)
        self.head = nn.Linear(D, VOCAB, bias=False)
        self.head.weight = self.embed.weight  # tied, like Qwen3-0.6B

    def forward(self, x):
        h = self.embed(x) * self.inv_freq
        h = torch.cat([h[..., :5] * self.scale, h[..., 5:]], dim=-1)
        h = self.l2(F.gelu(self.l1(h)))
        return self.head(self.norm(h))


def shard_model(model, mesh):
    fully_shard(model.l1, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def make_runtime(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + 0.1 * step))
    return ResizeRuntime(module=model, optimizer=optimizer, lr_scheduler=scheduler)


def make_group_and_mesh(ranks, rank):
    group = dist.new_group(ranks=ranks)
    mesh = None
    if rank in ranks:
        mesh = DeviceMesh.from_group(group, "cuda", mesh=torch.tensor(ranks), mesh_dim_names=("fsdp",))
    return group, mesh


def batch(step, device):
    gen = torch.Generator().manual_seed(1000 + step)
    return torch.randint(0, VOCAB, (B, T), generator=gen).to(device)


def train_step(runtime, ranks, step, device, rank):
    tokens = batch(step, device)
    if ranks is None:
        x = tokens
    else:
        per = B // len(ranks)
        x = tokens[ranks.index(rank) * per : (ranks.index(rank) + 1) * per]
    logits = runtime.module(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
    loss.backward()
    runtime.optimizer.step()
    runtime.lr_scheduler.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def full_state(runtime):
    out = {}
    for name, param in runtime.module.named_parameters():
        out[name] = param.full_tensor().clone()
        for key, value in runtime.optimizer.state.get(param, {}).items():
            out[f"{name}.{key}"] = (value.full_tensor() if hasattr(value, "full_tensor") else value).clone()
    out["lr"] = torch.tensor(runtime.optimizer.param_groups[0]["lr"])
    out["sched_last_epoch"] = torch.tensor(runtime.lr_scheduler.last_epoch)
    return out


def wrong_step(manifest):
    manifest["optimizer"]["state"]["l1.weight"]["scalars"]["step"]["value"] += 1.0


def wrong_lr(manifest):
    manifest["optimizer"]["param_groups"][0]["lr"] *= 2.0


def wrong_scheduler(manifest):
    manifest["lr_scheduler"]["last_epoch"] += 1


# What the source's manifest lies about -> the entry verification has to name.
SCALAR_FAULTS = {
    "optimizer.state[l1.weight].step": wrong_step,
    "optimizer.param_groups[0].lr": wrong_lr,
    "lr_scheduler.last_epoch": wrong_scheduler,
}


def expect_scalar_faults_caught(cur, cur_ranks, rank, device):
    """Scalar state travels in the manifest, not as tensors, so the checksums cannot vouch for it.

    Let the source's manifest lie about one entry at a time while moving ``cur`` to two ranks into a
    throwaway target (the transfer leaves the old runtime intact): every rank has to raise and name the
    entry.
    """
    target_ranks = [0, 1]
    target_group, target_mesh = make_group_and_mesh(target_ranks, rank)
    build_manifest = fsdp_resize._build_manifest
    for path, damage in SCALAR_FAULTS.items():
        transfer_group = dist.new_group(ranks=cur_ranks)
        if rank in cur_ranks:
            target = None
            if rank in target_ranks:
                with torch.device("meta"):
                    model = Toy()
                target = make_runtime(shard_model(model, target_mesh).to_empty(device=device))

            def lying_manifest(runtime, _damage=damage, **kwargs):
                manifest = build_manifest(runtime, **kwargs)
                _damage(manifest)
                return manifest

            fsdp_resize._build_manifest = lying_manifest
            try:
                transfer_fsdp2_runtime(
                    old=cur,
                    new=target,
                    old_ranks=cur_ranks,
                    new_ranks=target_ranks,
                    group=transfer_group,
                    device=device,
                    verify=True,
                )
            except RuntimeError as exc:
                assert path in str(exc), f"rank {rank}: verification raised without naming {path}: {exc}"
            else:
                raise AssertionError(f"rank {rank}: verification did not notice a wrong {path}")
            finally:
                fsdp_resize._build_manifest = build_manifest
            del target
            dist.destroy_process_group(transfer_group)
        clear_dtensor_dispatch_caches()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank in target_ranks:
        dist.destroy_process_group(target_group)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(0)
    golden = make_runtime(Toy().to(device))

    cur_ranks = [0, 1, 2, 3]
    cur_group, cur_mesh = make_group_and_mesh(cur_ranks, rank)
    cur = None
    if rank in cur_ranks:
        torch.manual_seed(0)
        cur = make_runtime(shard_model(Toy().to(device), cur_mesh))

    step = 0

    def train(k):
        nonlocal step
        for _ in range(k):
            train_step(golden, None, step, device, rank)
            if rank in cur_ranks:
                train_step(cur, cur_ranks, step, device, rank)
            step += 1
        if rank in cur_ranks:
            golden_params = dict(golden.module.named_parameters())
            for name, param in cur.module.named_parameters():
                torch.testing.assert_close(param.full_tensor(), golden_params[name], rtol=2e-5, atol=2e-5)
            assert cur.optimizer.param_groups[0]["lr"] == golden.optimizer.param_groups[0]["lr"]
        dist.barrier()

    train(2)

    for new_ranks in ([0, 1, 2, 3, 4, 5], [0, 1], [0, 1, 2, 3, 4, 5]):
        old_ranks, old, old_group = cur_ranks, cur, cur_group
        union = sorted(set(old_ranks) | set(new_ranks))

        # new runtime on the new mesh, no weights, while the old one is alive
        new_group, new_mesh = make_group_and_mesh(new_ranks, rank)
        new = None
        if rank in new_ranks:
            with torch.device("meta"):
                model = Toy()
            shard_model(model, new_mesh)
            model.to_empty(device=device)
            new = make_runtime(model)

        transfer_group = dist.new_group(ranks=union)
        old_full = full_state(old) if rank in old_ranks else None
        stats = None
        if rank in union:
            stats = transfer_fsdp2_runtime(
                old=old,
                new=new,
                old_ranks=old_ranks,
                new_ranks=new_ranks,
                group=transfer_group,
                device=device,
                verify=True,
            )
            assert stats["verified"] == 1

        if rank in new_ranks:
            new_full = full_state(new)
            if rank in old_ranks:
                assert new_full.keys() == old_full.keys()
                for key in new_full:
                    assert torch.equal(new_full[key].cpu(), old_full[key].cpu()), f"{key} differs after transfer"
            assert new.module.head.weight is new.module.embed.weight, "tied weight broken"
            assert torch.equal(new.module.inv_freq, golden.module.inv_freq), "buffer not restored"
            # a rank that keeps rows must not receive them over the network
            planner = FSDPResizePlanner(VOCAB, old_ranks, new_ranks)
            if rank in old_ranks and planner.network_recv_rows(rank) == 0:
                embed_rows_bytes = planner.target_intervals[new_ranks.index(rank)].size * D * 4
                assert stats["self_copy_bytes"] >= embed_rows_bytes
        dist.barrier()
        if rank in new_ranks:
            print(
                f"[{old_ranks}->{new_ranks}] rank {rank}: net_recv={stats['net_recv_bytes']}B "
                f"self_copy={stats['self_copy_bytes']}B tensors={stats['tensor_count']} "
                f"opt_tensors={stats['optimizer_tensor_count']} buffers={stats['buffer_count']}"
                + ("  bitwise==old" if rank in old_ranks else "  new member")
            )

        # tear down the old runtime and groups, then keep training on the new mesh
        if rank in old_ranks:
            del old
            dist.destroy_process_group(old_group)
        if rank in union:
            dist.destroy_process_group(transfer_group)
        clear_dtensor_dispatch_caches()
        torch.cuda.empty_cache()
        cur_ranks, cur, cur_group = new_ranks, new, new_group
        train(3)

    expect_scalar_faults_caught(cur, cur_ranks, rank, device)

    dist.barrier()
    if rank == 0:
        print(f"FSDP2 GPU-direct resize transfer 4->6->2->6 passed ({len(SCALAR_FAULTS)} scalar-state faults caught)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
