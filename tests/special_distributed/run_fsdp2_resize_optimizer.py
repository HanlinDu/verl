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

"""Optimizer check of the FSDP2 GPU-direct resize on a real model (fp32 AdamW with an LR schedule).

``run_fsdp2_resize_transfer.py`` proves the mechanics on a toy model.  This script answers "did the
optimizer arrive, and does training go on as if nothing had happened?" on the model in ``MODEL_PATH``,
for the mesh sizes in ``RESIZE_SEQUENCE`` (default ``4,6,2``, the actor sizes ``run_dynamic_resize.sh``
goes through by default):

* **next update** -- the old runtime stays alive next to the new one.  After every transfer all
  parameters, ``exp_avg``, ``exp_avg_sq``, ``step``, ``param_groups`` and the scheduler state are compared
  bitwise.  Then the same gradient is written into both runtimes, both take one optimizer and scheduler
  step, and everything is compared bitwise again.
* **faults**, around the first transfer, into throwaway targets --
  F1: one optimizer tensor is not moved; verification has to raise and name it.
  F2: the manifest reports ``step + 1``; verification has to raise and name ``step``.  The same lie
  without verification shows what it would have cost: nothing but ``step`` differs after the transfer,
  and the next update is off by what Adam's bias correction predicts.
* **trajectories** -- language-model training on fixed token batches: ``baseline`` stays on the first
  mesh, ``exact`` is transferred to the second one after two steps, ``fresh`` is transferred and then
  loses its optimizer state, which is what a weights-only resize leaves behind.  ``exact`` has to stay on
  the baseline, ``fresh`` has to leave it.

Rank 0 prints every expectation as PASS or FAIL.  Any FAIL makes the run exit non-zero; otherwise the
last line is ``FSDP2 GPU-direct resize optimizer check 4->6->2 passed``.  With ``OUT_DIR`` set, everything
that was measured is also written to ``$OUT_DIR/optimizer_check.json``.

A size that repeats the one before it is no transfer, as in the trainer, which keeps its topology at such
a stage: ``4,4,2`` checks ``4->2``.  The model is trained in fp32 on a few thousand tokens; Qwen3-0.6B
takes about a minute and 6.7 GiB per GPU for the default sizes, 12.5 GiB on a mesh of two ranks that
receives the first transfer.

Run with as many processes as the largest mesh:
    MODEL_PATH=<causal LM directory> PYTHONPATH=. torchrun --standalone --nproc-per-node=6 \
        tests/special_distributed/run_fsdp2_resize_optimizer.py
"""

import gc
import itertools
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

import verl.utils.fsdp_resize as fsdp_resize
from verl.utils.distributed import clear_dtensor_dispatch_caches
from verl.utils.fsdp_resize import ResizeRuntime, compute_1d_shard_range, reshard_fsdp2_modules, transfer_fsdp2_runtime
from verl.utils.fsdp_utils import apply_fsdp2

MODEL = os.environ["MODEL_PATH"]
# A stage that keeps the actor size is no switch for the trainer, so a repeated size is no transfer here.
SEQUENCE = [size for size, _ in itertools.groupby(map(int, os.environ.get("RESIZE_SEQUENCE", "4,6,2").split(",")))]
OUT_DIR = os.environ.get("OUT_DIR")
LABEL = "->".join(str(size) for size in SEQUENCE)
LR, WD, BETAS, EPS = 1e-4, 0.01, (0.9, 0.999), 1e-8
BATCH = math.lcm(*SEQUENCE)  # every mesh splits the batch evenly: the same objective on all of them
SEQ = max(16, min(256, 3072 // BATCH))  # 12 x 256 tokens for 4,6,2; shorter sequences when the batch is larger
GRAD_SCALE = 1e-3
WARMUP_STEPS, STEPS_AFTER_SWITCH = 2, 6
FAILED = []


def log(message):
    if dist.get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def check(ok, text):
    """Rank 0 sits in every mesh and is sent what the other ranks found, so it gives the verdicts."""
    if dist.get_rank() == 0:
        print(f"{'PASS' if ok else 'FAIL'}  {text}", flush=True)
        if not ok:
            FAILED.append(text)


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


# ---- runtimes ------------------------------------------------------------------------------------------
def group_mesh(size, rank):
    ranks = list(range(size))
    group = dist.new_group(ranks=ranks)
    mesh = None
    if rank in ranks:
        mesh = DeviceMesh.from_group(group, "cuda", mesh=torch.tensor(ranks), mesh_dim_names=("fsdp",))
    return ranks, group, mesh


def make_runtime(model):
    model.eval()  # no dropout: every mesh has to see the same objective
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=BETAS, eps=EPS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + 0.1 * step))
    return ResizeRuntime(module=model, optimizer=optimizer, lr_scheduler=scheduler)


def build_live(mesh, device):
    """The pretrained model, sharded on ``mesh`` with the trainer's wrap policy; nothing on other ranks."""
    if mesh is None:
        return None
    model = AutoModelForCausalLM.from_pretrained(MODEL).float().to(device)
    apply_fsdp2(model, {"mesh": mesh}, {})
    return make_runtime(model)


def build_empty(config, mesh, device):
    """What the trainer prepares as a resize target: the structure on ``mesh``, storage not initialised."""
    if mesh is None:
        return None
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config).float()
    apply_fsdp2(model, {"mesh": mesh}, {})
    model.to_empty(device=device)
    return make_runtime(model)


def release(group=None, member=False):
    """Call after dropping the last reference to a runtime; ``group`` is its mesh group, if that goes too."""
    if member:
        dist.destroy_process_group(group)
    clear_dtensor_dispatch_caches()
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def transfer(old, new, old_ranks, new_ranks, rank, device, verify=True):
    """``transfer_fsdp2_runtime`` over a group of its own; returns the statistics of this rank."""
    union = sorted(set(old_ranks) | set(new_ranks))
    group = dist.new_group(ranks=union)
    try:
        if rank in union:
            return transfer_fsdp2_runtime(
                old=old, new=new, old_ranks=old_ranks, new_ranks=new_ranks, group=group, device=device, verify=verify
            )
    finally:
        release(group, rank in union)


def aborted(old, new, old_ranks, new_ranks, rank, device):
    """A verified transfer that is expected to fail: what every rank of the transfer raised with."""
    message = None
    try:
        transfer(old, new, old_ranks, new_ranks, rank, device)
    except RuntimeError as exc:
        message = str(exc)
    del new
    release()
    return gather(message)[: max(len(old_ranks), len(new_ranks))]


# ---- data and gradients that do not depend on the mesh -----------------------------------------------------
def local(tensor):
    return (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach()


def mesh_group(runtime):
    return next(runtime.module.parameters()).device_mesh.get_group()


def train_step(runtime, ranks, step, vocab, device, rank):
    """One language-model step on this rank's share of a fixed batch; global loss, gradient norm and lr."""
    generator = torch.Generator().manual_seed(1000 + step)
    tokens = torch.randint(0, vocab, (BATCH, SEQ), generator=generator).to(device)
    share = BATCH // len(ranks)
    x = tokens[ranks.index(rank) * share : (ranks.index(rank) + 1) * share]
    loss = runtime.module(input_ids=x, labels=x).loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.module.parameters(), max_norm=1e9)  # measurement only
    runtime.optimizer.step()
    runtime.lr_scheduler.step()
    runtime.optimizer.zero_grad(set_to_none=True)
    loss = loss.detach().clone()
    dist.all_reduce(loss, group=mesh_group(runtime))
    loss /= len(ranks)
    grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
    return float(loss), float(grad_norm), float(runtime.optimizer.param_groups[0]["lr"])


def hashed_grad_like(param):
    """A pseudo-random gradient that is a pure function of the *global* element index.

    Integer arithmetic is exact on every device, and the only float operations are one correctly rounded
    int -> float32 conversion and elementwise operations.  So every rank produces bit-identical values for
    the rows it holds, however the tensor is sharded.
    """
    mesh = param.device_mesh
    row_offset, _ = compute_1d_shard_range(param.shape[0], mesh.size(), mesh.get_local_rank())
    shard = local(param)
    start = row_offset * math.prod(param.shape[1:])
    index = torch.arange(start, start + shard.numel(), device=shard.device, dtype=torch.int64)
    hashed = (index * 2654435761 + 12345) & 0xFFFFFFFF
    return ((hashed.to(torch.float32) / 4294967296.0 - 0.5) * GRAD_SCALE).view(shard.shape)


def squared_norm(tensor):
    """Sum of squares in float64, without a float64 copy of the whole tensor."""
    return sum(chunk.double().square().sum() for chunk in tensor.flatten().split(1 << 24))


@torch.no_grad()
def hashed_step(runtime, measure=False):
    """Write the hashed gradient into every parameter and take one optimizer and scheduler step.

    With ``measure`` the norm of the parameter update is returned, and how far a *fresh* AdamW (no
    moments, first step) would have moved for the same gradient and learning rate.  Its update is
    ``-lr * wd * p - lr * g / (|g| + eps)``: the bias corrections cancel the ``1 - beta`` factors.
    """
    params = list(runtime.module.parameters())
    lr = runtime.optimizer.param_groups[0]["lr"]
    for param in params:
        grad = hashed_grad_like(param)
        param.grad = DTensor.from_local(
            grad, param.device_mesh, param.placements, shape=param.shape, stride=param.stride()
        )
    before = [local(param).clone() for param in params] if measure else []
    runtime.optimizer.step()
    runtime.lr_scheduler.step()
    sums = torch.zeros(3, dtype=torch.float64, device=local(params[0]).device)
    for param, start in zip(params, before, strict=False):
        grad = local(param.grad)
        exact = local(param) - start
        fresh = -(lr * WD) * start - lr * grad / (grad.abs() + EPS)
        sums += torch.stack([squared_norm(exact), squared_norm(fresh), squared_norm(exact - fresh)])
    runtime.optimizer.zero_grad(set_to_none=True)
    if not measure:
        return None
    dist.all_reduce(sums, group=mesh_group(runtime))
    exact, fresh, difference = sums.sqrt().tolist()
    return {"lr": lr, "update_norm": exact, "fresh_update_norm": fresh, "fresh_vs_exact": difference / exact}


# ---- comparison --------------------------------------------------------------------------------------------
@torch.no_grad()
def compare(old, new, in_old, in_new):
    """Bitwise comparison of two runtimes: parameters, optimizer state, ``param_groups`` and scheduler state.

    ``full_tensor()`` is collective on a mesh, so every member of either mesh has to call this.  A rank in
    both meshes returns what it found; a rank in one of them only takes part in the gathers.
    """
    if in_old:
        reshard_fsdp2_modules(old.module)
    if in_new:
        reshard_fsdp2_modules(new.module)
    old_params = dict(old.module.named_parameters()) if in_old else {}
    new_params = dict(new.module.named_parameters()) if in_new else {}
    kinds, differing = {}, []
    for name in sorted(new_params or old_params):
        old_state = old.optimizer.state.get(old_params[name], {}) if in_old else {}
        new_state = new.optimizer.state.get(new_params[name], {}) if in_new else {}
        pairs = [("param", old_params.get(name), new_params.get(name))]
        pairs += [(key, old_state.get(key), new_state.get(key)) for key in sorted(new_state or old_state)]
        for kind, a, b in pairs:
            a = a.full_tensor() if isinstance(a, DTensor) else a
            b = b.full_tensor() if isinstance(b, DTensor) else b
            if not (in_old and in_new):
                continue
            found = kinds.setdefault(kind, {"compared": 0, "differing": 0, "squared_difference": 0.0})
            found["compared"] += 1
            if a is None or b is None or not torch.equal(a, b):
                found["differing"] += 1
                differing.append(f"{name}.{kind}")
                if a is not None and b is not None:
                    found["squared_difference"] += float(squared_norm(a - b))
    if not (in_old and in_new):
        return None

    def hyper(runtime):
        return [{k: v for k, v in group.items() if k != "params"} for group in runtime.optimizer.param_groups]

    return {
        "compared": sum(found["compared"] for found in kinds.values()),
        "differing": len(differing),
        "examples": differing[:3],
        "kinds": kinds,
        "param_groups_and_scheduler_equal": hyper(old) == hyper(new)
        and old.lr_scheduler.state_dict() == new.lr_scheduler.state_dict(),
    }


def all_equal(findings):
    """Every rank that sits in both meshes compared everything itself and found no difference."""
    findings = [found for found in findings if found is not None]
    return bool(findings) and all(
        found["differing"] == 0 and found["param_groups_and_scheduler_equal"] for found in findings
    )


def describe(found):
    kinds = ", ".join(f"{kind['compared']} {name}" for name, kind in found["kinds"].items())
    return (
        f"{found['compared']} tensors compared ({kinds}), {found['differing']} differ, "
        f"param_groups and scheduler equal={found['param_groups_and_scheduler_equal']}"
    )


def bias_correction(step):
    """Adam scales its update by this factor at ``step``."""
    return math.sqrt(1 - BETAS[1] ** step) / (1 - BETAS[0] ** step)


def largest_parameter(config):
    """A parameter name that every rank can work out by itself, whichever meshes it is in."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    sizes = {name: param.numel() for name, param in model.named_parameters()}
    return max(sorted(sizes), key=sizes.get)


# ---- next update and faults --------------------------------------------------------------------------------
def run_next_update(rank, device, config):
    report = []
    move, build_manifest = fsdp_resize._Transfer._move, fsdp_resize._build_manifest
    victim = f"{largest_parameter(config)}.exp_avg_sq"

    def skipping_victim(self, item, old_t, new_t):
        if item["name"] != victim:
            move(self, item, old_t, new_t)

    def lying_about_step(runtime, **kwargs):
        manifest = build_manifest(runtime, **kwargs)
        for state in manifest["optimizer"]["state"].values():
            state["scalars"]["step"]["value"] += 1.0
        return manifest

    cur_ranks, cur_group, cur_mesh = group_mesh(SEQUENCE[0], rank)
    cur = build_live(cur_mesh, device)
    for step in range(WARMUP_STEPS):  # real AdamW steps: the moments are filled, ``step`` and lr have moved
        if rank in cur_ranks:
            loss, grad_norm, lr = train_step(cur, cur_ranks, step, config.vocab_size, device, rank)
            log(f"warm-up step {step} on {len(cur_ranks)} ranks: loss={loss:.6f} grad_norm={grad_norm:.4f} lr={lr:.3e}")
    dist.barrier()

    for index, size in enumerate(SEQUENCE[1:]):
        old_ranks, old, old_group = cur_ranks, cur, cur_group
        new_ranks, new_group, new_mesh = group_mesh(size, rank)
        in_old, in_new = rank in old_ranks, rank in new_ranks
        label = f"{len(old_ranks)}->{len(new_ranks)}"
        entry = {"transfer": label}

        if index == 0:
            # F1: one optimizer tensor is not moved
            fsdp_resize._Transfer._move = skipping_victim
            try:
                messages = aborted(old, build_empty(config, new_mesh, device), old_ranks, new_ranks, rank, device)
            finally:
                fsdp_resize._Transfer._move = move
            entry["F1"] = {"not_moved": victim, "errors": messages}
            check(
                all(message and victim in message for message in messages),
                f"F1 ({label}, {victim} not moved): verification raised on every rank and names the tensor",
            )
            log(f"   {str(messages[0]).split(' | ')[0]}")

            # F2: the manifest reports step + 1 for every parameter
            fsdp_resize._build_manifest = lying_about_step
            try:
                messages = aborted(old, build_empty(config, new_mesh, device), old_ranks, new_ranks, rank, device)
            finally:
                fsdp_resize._build_manifest = build_manifest
            entry["F2"] = {"errors": messages}
            if rank == 0:
                stepped = sum(1 for param in old.module.parameters() if "step" in old.optimizer.state.get(param, {}))
                wrong = f"{stepped * len(new_ranks)} scalar state entries differ"
                check(
                    all(message and wrong in message and "].step is " in message for message in messages),
                    f"F2 ({label}, manifest reports step + 1): verification raised on every rank and names step, "
                    f"wrong for {stepped} parameters on each of the {len(new_ranks)} new ranks",
                )
                log(f"   {str(messages[0]).split(' | ')[0]}")

        # the transfer itself, then the state and the next update of both runtimes
        new = build_empty(config, new_mesh, device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stats = transfer(old, new, old_ranks, new_ranks, rank, device)
        entry["seconds"] = time.perf_counter() - start
        entry["stats"] = stats
        state = gather(compare(old, new, in_old, in_new))
        if in_new:
            entry["arrived"] = {
                "step": float(new.optimizer.state[next(new.module.parameters())]["step"]),
                "lr": new.optimizer.param_groups[0]["lr"],
                "scheduler": {k: v for k, v in new.lr_scheduler.state_dict().items() if k != "lr_lambdas"},
            }
            hashed_step(new)
        if in_old:
            entry["update"] = hashed_step(old, measure=True)
        update = gather(compare(old, new, in_old, in_new))
        entry["state"], entry["next_update"] = state, update
        if rank == 0:
            log(f"{label}: moved in {entry['seconds']:.2f}s, arrived with {entry['arrived']}")
            check(
                stats["verified"] == 1 and stats["verified_scalar_count"] > 0,
                f"{label}: the transfer verified itself: checksums of parameters={stats['tensor_count']} "
                f"optimizer_tensors={stats['optimizer_tensor_count']} buffers={stats['buffer_count']}, "
                f"scalar state entries={stats['verified_scalar_count']}",
            )
            both = sorted(set(old_ranks) & set(new_ranks))
            check(all_equal(state), f"{label}: state after the transfer, ranks {both}: {describe(state[0])}")
            check(
                all_equal(update),
                f"{label}: after one more update with the same gradient, ranks {both}: {describe(update[0])}",
            )
            log(
                f"   that update has norm {entry['update']['update_norm']:.4f} at lr={entry['update']['lr']:.4e}; "
                f"a fresh AdamW would have moved {entry['update']['fresh_update_norm']:.4f}, relative difference "
                f"{entry['update']['fresh_vs_exact']:.4f}"
            )

        if index == 0:
            # F2 without verification: what the wrong step costs. The old runtime is the reference; nothing
            # needs it afterwards, so it may take one more update.
            wrong = build_empty(config, new_mesh, device)
            fsdp_resize._build_manifest = lying_about_step
            try:
                transfer(old, wrong, old_ranks, new_ranks, rank, device, verify=False)
            finally:
                fsdp_resize._build_manifest = build_manifest
            arrived = compare(old, wrong, in_old, in_new)
            if in_old:
                number = float(old.optimizer.state[next(old.module.parameters())]["step"]) + 1
                truth = hashed_step(old, measure=True)
            if in_new:
                hashed_step(wrong)
            cost = compare(old, wrong, in_old, in_new)
            del wrong
            release()
            if rank == 0:
                kinds = cost["kinds"]
                measured = math.sqrt(kinds["param"]["squared_difference"]) / truth["update_norm"]
                predicted = 1 - bias_correction(number + 1) / bias_correction(number)
                entry["F2"].update(arrived=arrived, cost=cost, update=number, measured=measured, predicted=predicted)
                check(
                    arrived["differing"] == arrived["kinds"]["step"]["differing"] > 0,
                    f"F2 without verification: the transfer goes through, and nothing but step differs "
                    f"afterwards: {describe(arrived)}, e.g. {arrived['examples']}",
                )
                check(
                    kinds["exp_avg"]["differing"] == kinds["exp_avg_sq"]["differing"] == 0
                    and kinds["param"]["differing"] > 0
                    and abs(measured - predicted) < 0.01,
                    f"F2 without verification: update {number:.0f} then moves {kinds['param']['differing']} of "
                    f"{kinds['param']['compared']} parameters differently while both moments stay equal; the "
                    f"difference is {measured:.4f} of the update, and Adam's bias correction for step "
                    f"{number + 1:.0f} instead of {number:.0f} predicts {predicted:.4f}",
                )

        report.append(entry)
        cur_ranks, cur, cur_group = new_ranks, new, new_group
        del old, new
        release(old_group, in_old)
    del cur
    release(cur_group, rank in cur_ranks)
    return report


# ---- trajectories --------------------------------------------------------------------------------------------
def run_trajectory(variant, rank, device, config):
    """Loss, gradient norm and lr of ``WARMUP_STEPS + STEPS_AFTER_SWITCH`` steps.

    ``variant`` says what happens in between: nothing (``baseline``), the first transfer (``exact``), or
    that transfer followed by the loss of the optimizer state (``fresh``).
    """
    cur_ranks, cur_group, cur_mesh = group_mesh(SEQUENCE[0], rank)
    cur = build_live(cur_mesh, device)
    trace = []
    for step in range(WARMUP_STEPS + STEPS_AFTER_SWITCH):
        if step == WARMUP_STEPS and variant != "baseline":
            new_ranks, new_group, new_mesh = group_mesh(SEQUENCE[1], rank)
            new = build_empty(config, new_mesh, device)
            transfer(cur, new, cur_ranks, new_ranks, rank, device)
            if variant == "fresh" and rank in new_ranks:
                new.optimizer.state.clear()  # moments and step are gone; hyper-parameters and scheduler stay
            old_group, was_member = cur_group, rank in cur_ranks
            cur_ranks, cur, cur_group = new_ranks, new, new_group
            del new
            release(old_group, was_member)
        if rank in cur_ranks:
            loss, grad_norm, lr = train_step(cur, cur_ranks, step, config.vocab_size, device, rank)
            trace.append({"step": step, "ranks": len(cur_ranks), "loss": loss, "grad_norm": grad_norm, "lr": lr})
        dist.barrier()
    del cur
    release(cur_group, rank in cur_ranks)
    return trace


def judge_trajectories(traces):
    baseline, exact, fresh = traces["baseline"], traces["exact"], traces["fresh"]
    exact_off = [abs(e["loss"] - b["loss"]) / abs(b["loss"]) for b, e in zip(baseline, exact, strict=True)]
    fresh_off = [abs(f["loss"] - b["loss"]) / abs(b["loss"]) for b, f in zip(baseline, fresh, strict=True)]
    print("trajectories  step  ranks     loss baseline / exact / fresh         exact off  fresh off  lr")
    for b, e, f, e_off, f_off in zip(baseline, exact, fresh, exact_off, fresh_off, strict=True):
        print(
            f"              {b['step']:>4}  {b['ranks']} {e['ranks']} {f['ranks']}     "
            f"{b['loss']:.7f} / {e['loss']:.7f} / {f['loss']:.7f}   {e_off:.1e}    {f_off:.1e}    {e['lr']:.4e}",
            flush=True,
        )
    # The first update on the new mesh is step WARMUP_STEPS; the loss of the step after it shows what it did.
    affected = WARMUP_STEPS + 1
    check(
        [e["ranks"] for e in exact] == [SEQUENCE[0]] * WARMUP_STEPS + [SEQUENCE[1]] * STEPS_AFTER_SWITCH
        and [e["lr"] for e in exact] == [b["lr"] for b in baseline],
        f"trajectory: exact takes {WARMUP_STEPS} steps on {SEQUENCE[0]} ranks and {STEPS_AFTER_SWITCH} on "
        f"{SEQUENCE[1]}, with the learning rates of the baseline",
    )
    check(
        max(exact_off) < 1e-5 and fresh_off[affected] > 100 * max(exact_off),
        f"trajectory: exact stays on the {SEQUENCE[0]}-rank baseline, its loss is never off by more than "
        f"{max(exact_off):.1e} (limit 1e-05, the two meshes add the same numbers up in a different order); "
        f"without the optimizer state step {affected} is off by {fresh_off[affected]:.1e} "
        f"(has to be over 100 times more)",
    )


def main():
    if len(SEQUENCE) < 2:
        if os.environ.get("RANK", "0") == "0":
            print(f"FSDP2 GPU-direct resize optimizer check: the mesh size stays {LABEL}, no transfer to check")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    assert dist.get_world_size() >= max(SEQUENCE), f"{LABEL} needs {max(SEQUENCE)} processes"
    config = AutoConfig.from_pretrained(MODEL)
    log(f"optimizer check {LABEL}: model={MODEL} batch={BATCH}x{SEQ} torch={torch.__version__}")

    results = {"model": MODEL, "sequence": SEQUENCE, "transfers": run_next_update(rank, device, config)}
    results["trajectories"] = {
        variant: run_trajectory(variant, rank, device, config) for variant in ("baseline", "exact", "fresh")
    }
    results["peak_memory_gib"] = gather(round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
    if rank == 0:
        judge_trajectories(results["trajectories"])
        log(f"peak GPU memory per rank: {results['peak_memory_gib']} GiB")
        results["failed"] = FAILED
        if OUT_DIR:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, "optimizer_check.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)
    dist.barrier()
    dist.destroy_process_group()
    if FAILED:
        raise SystemExit(f"FSDP2 GPU-direct resize optimizer check {LABEL} FAILED: {len(FAILED)} expectations")
    if rank == 0:
        print(f"FSDP2 GPU-direct resize optimizer check {LABEL} passed")


if __name__ == "__main__":
    main()
