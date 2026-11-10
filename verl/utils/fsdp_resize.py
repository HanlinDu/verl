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

"""GPU-direct FSDP2 runtime transfer for dynamic resize.

Moves the *training state* of an FSDP2 model that lives on an ``old`` set of
ranks to a freshly built FSDP2 model on a ``new`` set of ranks: fp32 sharded
parameters, per-parameter optimizer state, replicated buffers, optimizer
hyper-parameters and the LR-scheduler state.  Nothing is staged through host
memory and no row is sent twice.

How it works
------------
* FSDP2 shards every parameter (hence every optimizer moment) along dim 0 with
  ``torch.chunk`` semantics, so each rank's local shard is a contiguous row
  interval.  With ``old_ranks``/``new_ranks`` in increasing order, the rows a
  source has to send are consecutive sub-intervals of its own local shard,
  ordered by target, and the rows a target receives arrive in source order,
  i.e. already in row order.  One ``all_to_all_single`` per tensor therefore
  uses the **old local shard as input and the new local shard as output** with
  no intermediate buffer: zero copies on both ends, every row crosses the wire
  exactly once and only to ranks that do not already hold it.  Rows a rank
  keeps are moved by NCCL's self chunk (an on-device memcpy).
* Old and new models coexist during the transfer.  The caller builds the new
  model on the new mesh without loading weights (meta -> ``fully_shard`` ->
  ``to_empty``); this module fills it in place.  Expansion, shrink and
  repeated resizes are all the same code path.
* Replicated tensors (buffers such as rotary ``inv_freq``, optimizer scalars)
  are broadcast from the first old rank.  Metadata (tensor manifest, optimizer
  ``step`` values, ``param_groups``, scheduler ``state_dict``) travels with
  ``broadcast_object_list``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import prod
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Replicate, Shard

from verl.utils.device import get_torch_device


def compute_1d_shard_range(dim_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Return the offset and size used by a one-dimensional DTensor ``Shard(0)``."""
    if dim_size < 0:
        raise ValueError(f"dim_size must be non-negative, got {dim_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    shard_size, offset = Shard._local_shard_size_and_offset(dim_size, world_size, rank)
    return int(offset), int(shard_size)


@dataclass(frozen=True, slots=True)
class ShardInterval:
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


class FSDPResizePlanner:
    """Row-interval plan for one dim-0 sharded tensor moving from ``old_ranks`` to ``new_ranks``.

    Split vectors are indexed by position in the transfer group, i.e. ``sorted(set(old) | set(new))``.
    ``old_ranks``/``new_ranks`` must be strictly increasing so that the rows a rank sends (receives)
    are consecutive, in-order sub-intervals of its local shard; this is what makes the transfer
    zero-copy.
    """

    def __init__(self, dim_size: int, old_ranks: Sequence[int], new_ranks: Sequence[int]):
        old_ranks, new_ranks = list(old_ranks), list(new_ranks)
        if not old_ranks or not new_ranks:
            raise ValueError("old_ranks and new_ranks must be non-empty")
        for name, ranks in (("old_ranks", old_ranks), ("new_ranks", new_ranks)):
            if any(b <= a for a, b in zip(ranks, ranks[1:], strict=False)):
                raise ValueError(f"{name} must be strictly increasing, got {ranks}")
        self.dim_size = dim_size
        self.old_ranks = old_ranks
        self.new_ranks = new_ranks
        self.group_ranks = sorted(set(old_ranks) | set(new_ranks))
        self.source_intervals = [
            ShardInterval(*compute_1d_shard_range(dim_size, len(old_ranks), i)) for i in range(len(old_ranks))
        ]
        self.target_intervals = [
            ShardInterval(*compute_1d_shard_range(dim_size, len(new_ranks), i)) for i in range(len(new_ranks))
        ]

    def overlap_rows(self, source_index: int, target_index: int) -> int:
        source = self.source_intervals[source_index]
        target = self.target_intervals[target_index]
        return max(min(source.end, target.end) - max(source.offset, target.offset), 0)

    def send_row_splits(self, source_rank: int) -> list[int]:
        """Rows ``source_rank`` sends to each member of the transfer group (group order)."""
        s = self.old_ranks.index(source_rank)
        return [self.overlap_rows(s, self.new_ranks.index(r)) if r in self.new_ranks else 0 for r in self.group_ranks]

    def recv_row_splits(self, target_rank: int) -> list[int]:
        """Rows ``target_rank`` receives from each member of the transfer group (group order)."""
        t = self.new_ranks.index(target_rank)
        return [self.overlap_rows(self.old_ranks.index(r), t) if r in self.old_ranks else 0 for r in self.group_ranks]

    def network_recv_rows(self, target_rank: int) -> int:
        """Rows ``target_rank`` must actually receive from other ranks (excludes its own overlap)."""
        splits = self.recv_row_splits(target_rank)
        own = splits[self.group_ranks.index(target_rank)] if target_rank in self.old_ranks else 0
        return sum(splits) - own


@dataclass
class ResizeRuntime:
    """The pieces of a training runtime that move during a resize."""

    module: torch.nn.Module
    optimizer: torch.optim.Optimizer | None = None
    lr_scheduler: Any | None = None


def raise_if_any_failed(error: str | None, group: dist.ProcessGroup, device: torch.device) -> None:
    """Exchange per-rank error strings over ``group`` and raise on every rank if any rank failed."""
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    failed = [message for message in errors if message]
    if failed:
        raise RuntimeError("GPU-direct resize aborted: " + " | ".join(failed[:4]))


_DIGEST_MODULUS = 1 << 20
_DIGEST_CHUNK = 1 << 25


def _shard_digest(local: torch.Tensor, global_offset_elems: int, acc: torch.Tensor) -> None:
    """Accumulate an exact, position-sensitive checksum of ``local`` into ``acc`` (int64 ``[2]``).

    ``acc[0]`` sums the raw 32-bit words, ``acc[1]`` sums them weighted by their *global* flat index,
    so shards of the same full tensor combine with a plain sum regardless of how the tensor is split.
    """
    flat = local.detach().reshape(-1).contiguous()
    if flat.numel() == 0:
        return
    if flat.element_size() == 4:
        words, word_offset = flat.view(torch.int32), global_offset_elems
    else:
        words, word_offset = flat.view(torch.uint8), global_offset_elems * flat.element_size()
    for start in range(0, words.numel(), _DIGEST_CHUNK):
        chunk = words[start : start + _DIGEST_CHUNK].to(torch.int64)
        weights = (
            torch.arange(
                word_offset + start, word_offset + start + chunk.numel(), device=chunk.device, dtype=torch.int64
            )
            % _DIGEST_MODULUS
            + 1
        )
        acc[0] += chunk.sum()
        acc[1] += (chunk * weights).sum()


def reshard_fsdp2_modules(module: torch.nn.Module) -> None:
    """Make sure every FSDP2 module registers its *sharded* DTensor parameters."""
    for sub in module.modules():
        if isinstance(sub, FSDPModule):
            sub.reshard()


# ---- low-memory pipeline: allocate / move / release one FSDP2 unit at a time ---------------------------
#
# The default transfer allocates the whole new runtime before moving anything and frees the whole old
# runtime afterwards, so every member rank peaks at old_shard + new_shard. Moving one FSDP2 unit (a
# fully_shard'ed module, i.e. one decoder layer or the root group) at a time -- materialise its new
# shards, move its tensors, drop its old shards -- bounds the peak at max(old, new) + one unit instead.
# Communication is unchanged (still one all_to_all_single per tensor); what changes is the storage
# lifecycle, which has to reach into FSDP2's per-parameter padded buffers (``FSDPParam``).


@dataclass
class ResizeUnit:
    name: str
    param_names: list[str]
    fsdp_params: list[Any]


def fsdp2_units(module: torch.nn.Module) -> list[ResizeUnit]:
    """Group the parameters of ``module`` by the FSDP2 module that shards them (model order)."""
    name_by_param = {id(param): name for name, param in module.named_parameters()}
    units: list[ResizeUnit] = []
    seen: set[str] = set()
    for module_name, sub in module.named_modules():
        if not isinstance(sub, FSDPModule):
            continue
        group = sub._get_fsdp_state()._fsdp_param_group
        if group is None:
            continue
        names, fsdp_params = [], []
        for fsdp_param in group.fsdp_params:
            name = name_by_param.get(id(fsdp_param.sharded_param))
            if name is None:
                raise RuntimeError(f"FSDP2 parameter of {module_name or '<root>'} is not a registered parameter")
            names.append(name)
            fsdp_params.append(fsdp_param)
            seen.add(name)
        units.append(ResizeUnit(module_name or "<root>", names, fsdp_params))
    unmanaged = [name for name in name_by_param.values() if name not in seen]
    if unmanaged:
        units.append(ResizeUnit("<unmanaged>", unmanaged, []))
    return units


@torch.no_grad()
def materialize_buffers(module: torch.nn.Module, device: torch.device | int) -> None:
    """Give meta buffers real (uninitialised) storage; parameters are left for the per-unit pipeline."""
    for sub in module.modules():
        for key, buf in list(sub._buffers.items()):
            if buf is not None and buf.is_meta:
                sub._buffers[key] = torch.empty_like(buf, device=device)


def _swap_storage(param: torch.nn.Parameter, device: torch.device | str) -> None:
    """Give ``param`` fresh (uninitialised) storage on ``device`` while keeping the Parameter object.

    Same mechanism as ``nn.Module.to_empty`` for tensor subclasses: ``swap_tensors`` keeps ties and the
    references FSDP2 holds to the Parameter object intact (``.data =`` refuses meta <-> device swaps).
    """
    fresh = torch.nn.Parameter(torch.empty_like(param, device=device), requires_grad=param.requires_grad)
    torch.utils.swap_tensors(param, fresh)


@torch.no_grad()
def materialize_unit(unit: ResizeUnit, module: torch.nn.Module, device: torch.device) -> None:
    """Allocate the local shards of one unit of a meta-initialised FSDP2 model."""
    params = dict(module.named_parameters())
    for fsdp_param in unit.fsdp_params:
        param = fsdp_param.sharded_param
        if param.is_meta:
            _swap_storage(param, device)
            fsdp_param.reset_sharded_param()  # re-pads and re-points FSDP2's sharded storage
    for name in unit.param_names[len(unit.fsdp_params) :]:  # unmanaged (plain) parameters
        if params[name].is_meta:
            _swap_storage(params[name], device)


@torch.no_grad()
def release_unit(unit: ResizeUnit, module: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
    """Drop the local shards (and optimizer state) of one unit of the old model after it has been moved."""
    params = dict(module.named_parameters())
    for name in unit.param_names:
        if optimizer is not None:
            optimizer.state.pop(params[name], None)
    for fsdp_param in unit.fsdp_params:
        fsdp_param._sharded_param_data = None  # FSDP2's padded storage; the DTensor local tensor views it
        _swap_storage(fsdp_param.sharded_param, "meta")
    for name in unit.param_names[len(unit.fsdp_params) :]:
        _swap_storage(params[name], "meta")


def order_units(units: Sequence[ResizeUnit], sizes: dict[str, int], *, expanding: bool) -> list[ResizeUnit]:
    """Order that minimises the pipeline's peak: memory falls per unit when expanding (smallest unit
    first keeps the first, highest step small) and rises when shrinking (largest unit first keeps the
    last, highest step small)."""
    return sorted(units, key=lambda unit: sizes[unit.name], reverse=not expanding)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in resize manifest: {name}")
    return dtype


def _tensor_kind(name: str, value: torch.Tensor) -> str:
    if isinstance(value, DTensor):
        placements = value.placements
        if len(placements) != 1:
            raise NotImplementedError(f"GPU-direct resize needs a 1-D mesh, got placements {placements} for {name}")
        placement = placements[0]
        if isinstance(placement, Shard):
            if int(placement.dim) != 0:
                raise NotImplementedError(f"GPU-direct resize only supports Shard(0), got {placement} for {name}")
            return "shard0"
        if isinstance(placement, Replicate):
            return "replicate"
        raise NotImplementedError(f"GPU-direct resize does not support placement {placement} for {name}")
    return "replicate"


def _describe(name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": tuple(int(v) for v in value.shape),
        "dtype": _dtype_name(value.dtype),
        "kind": _tensor_kind(name, value),
    }


def _local(value: torch.Tensor) -> torch.Tensor:
    # Under no_grad, DTensor.to_local() returns the real local storage (a view into FSDP2's padded buffer).
    return value.to_local() if isinstance(value, DTensor) else value


def _build_manifest(runtime: ResizeRuntime, *, with_units: bool = False) -> dict[str, Any]:
    params = dict(runtime.module.named_parameters())  # dedups tied weights
    manifest: dict[str, Any] = {
        "params": [_describe(name, params[name]) for name in sorted(params)],
        "buffers": [_describe(name, buf) for name, buf in sorted(runtime.module.named_buffers(), key=lambda x: x[0])],
        "optimizer": None,
        "lr_scheduler": None,
        "units": None,
    }
    if with_units:
        manifest["units"] = [
            {
                "name": unit.name,
                "params": sorted(unit.param_names),
                "size": sum(params[name].numel() for name in unit.param_names),
            }
            for unit in fsdp2_units(runtime.module)
        ]
    optimizer = runtime.optimizer
    if optimizer is not None:
        state: dict[str, dict[str, Any]] = {}
        for name in sorted(params):
            param_state = optimizer.state.get(params[name])
            if not param_state:
                continue
            tensors: dict[str, dict[str, Any]] = {}
            scalars: dict[str, dict[str, Any]] = {}
            for key in sorted(param_state):
                value = param_state[key]
                if not isinstance(value, torch.Tensor):
                    raise NotImplementedError(f"optimizer state {key} of {name} is not a tensor: {type(value)}")
                if value.ndim == 0:
                    scalars[key] = {
                        "value": value.item(),
                        "dtype": _dtype_name(value.dtype),
                        "on_accelerator": value.is_cuda,
                    }
                else:
                    tensors[key] = _describe(f"{name}.{key}", value)
            state[name] = {"tensors": tensors, "scalars": scalars}
        manifest["optimizer"] = {
            "state": state,
            "param_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
        }
    if runtime.lr_scheduler is not None:
        manifest["lr_scheduler"] = runtime.lr_scheduler.state_dict()
    return manifest


def _flatten_scalars(path: str, value: Any, out: dict[str, str]) -> None:
    """Record ``value`` in ``out`` as exact text, one entry per leaf of nested dicts, lists and tuples."""
    if isinstance(value, dict) and value:
        for key, item in value.items():
            _flatten_scalars(f"{path}.{key}", item, out)
    elif isinstance(value, list | tuple) and value:
        for index, item in enumerate(value):
            _flatten_scalars(f"{path}[{index}]", item, out)
    elif isinstance(value, torch.Tensor):
        out[path] = f"{value.tolist()!r} ({_dtype_name(value.dtype)} tensor on {value.device.type})"
    else:
        out[path] = f"{value!r} ({type(value).__name__})"  # repr is exact for float, int, bool, None and str


def _scalar_state(runtime: ResizeRuntime) -> dict[str, str]:
    """The training state that travels in the manifest instead of as a tensor, read from the live objects.

    Optimizer scalars (``step``), ``param_groups`` hyper-parameters and the LR-scheduler state, flattened
    to ``{path: exact text}`` so that two runtimes can be compared entry by entry.
    """
    out: dict[str, str] = {}
    optimizer = runtime.optimizer
    if optimizer is not None:
        for name, param in runtime.module.named_parameters():
            for key, value in optimizer.state.get(param, {}).items():
                if not isinstance(value, torch.Tensor) or value.ndim == 0:  # the others move as tensors
                    _flatten_scalars(f"optimizer.state[{name}].{key}", value, out)
        for index, group in enumerate(optimizer.param_groups):
            hyper = {key: value for key, value in group.items() if key != "params"}
            _flatten_scalars(f"optimizer.param_groups[{index}]", hyper, out)
    if runtime.lr_scheduler is not None:
        _flatten_scalars("lr_scheduler", runtime.lr_scheduler.state_dict(), out)
    return out


def _check_units(expected: list[dict[str, Any]], actual: Sequence[ResizeUnit]) -> None:
    """Both sides must shard the same parameters in the same FSDP2 units (the pipeline moves unit by unit)."""
    got = {unit.name: sorted(unit.param_names) for unit in actual}
    want = {unit["name"]: list(unit["params"]) for unit in expected}
    if got != want:
        only_want = sorted(set(want) - set(got))[:4]
        only_got = sorted(set(got) - set(want))[:4]
        differing = [name for name in want if name in got and got[name] != want[name]][:4]
        raise RuntimeError(
            f"GPU-direct resize FSDP units differ from source: missing={only_want}, unexpected={only_got}, "
            f"membership differs={differing}"
        )


def _check_structure(expected: list[dict[str, Any]], actual: dict[str, torch.Tensor], what: str) -> None:
    mismatches = []
    actual_names = set(actual)
    expected_names = {item["name"] for item in expected}
    if actual_names != expected_names:
        raise RuntimeError(
            f"GPU-direct resize {what} names differ from source: missing={sorted(expected_names - actual_names)[:8]}, "
            f"unexpected={sorted(actual_names - expected_names)[:8]}"
        )
    for item in expected:
        local = _describe(item["name"], actual[item["name"]])
        if local != item:
            mismatches.append((item, local))
    if mismatches:
        raise RuntimeError(f"GPU-direct resize {what} structure mismatch (source vs local): {mismatches[:4]}")


class _Transfer:
    def __init__(
        self,
        *,
        old: ResizeRuntime | None,
        new: ResizeRuntime | None,
        old_ranks: Sequence[int],
        new_ranks: Sequence[int],
        group: dist.ProcessGroup,
        device: torch.device,
        verify: bool = False,
        pipeline: bool = False,
    ):
        self.old_ranks, self.new_ranks = list(old_ranks), list(new_ranks)
        self.group, self.device = group, device
        self.pipeline = pipeline
        self.rank = dist.get_rank()
        self.in_old, self.in_new = self.rank in self.old_ranks, self.rank in self.new_ranks
        if not self.in_old and not self.in_new:
            raise ValueError(f"rank {self.rank} is neither in old_ranks {old_ranks} nor new_ranks {new_ranks}")
        if self.in_old and old is None:
            raise RuntimeError(f"rank {self.rank} is an old rank but has no live runtime to send from")
        if self.in_new and new is None:
            raise RuntimeError(f"rank {self.rank} is a new rank but has no prepared runtime to receive into")
        self.old, self.new = old if self.in_old else None, new if self.in_new else None
        self.group_ranks = sorted(set(self.old_ranks) | set(self.new_ranks))
        if dist.get_world_size(group) != len(self.group_ranks):
            raise ValueError(
                f"transfer group has {dist.get_world_size(group)} ranks, expected {len(self.group_ranks)} (old ∪ new)"
            )
        self.my_pos = self.group_ranks.index(self.rank)
        self.src = self.old_ranks[0]
        self.verify = verify
        self.stats = {
            "tensor_count": 0,
            "optimizer_tensor_count": 0,
            "buffer_count": 0,
            "net_recv_bytes": 0,
            "net_send_bytes": 0,
            "self_copy_bytes": 0,
            "verified": 0,
            "verified_scalar_count": 0,
        }

    # -- collectives -----------------------------------------------------------------------------------
    def _all_to_all_rows(self, item: dict[str, Any], old_t: torch.Tensor | None, new_t: torch.Tensor | None):
        shape = tuple(item["shape"])
        dtype = _resolve_dtype(item["dtype"])
        if not shape:
            raise RuntimeError(f"cannot Shard(0) a scalar tensor {item['name']}")
        planner = FSDPResizePlanner(shape[0], self.old_ranks, self.new_ranks)
        row_width = prod(shape[1:])
        group_size = len(self.group_ranks)

        if self.in_old:
            local = _local(old_t)
            in_splits = [rows * row_width for rows in planner.send_row_splits(self.rank)]
            if local.dtype != dtype or not local.is_contiguous() or local.numel() != sum(in_splits):
                raise RuntimeError(
                    f"source local shard of {item['name']} on rank {self.rank} does not match the plan: "
                    f"shape={tuple(local.shape)} dtype={local.dtype} contiguous={local.is_contiguous()} "
                    f"expected_numel={sum(in_splits)}"
                )
            send = local.view(-1)
        else:
            in_splits = [0] * group_size
            send = torch.empty(0, dtype=dtype, device=self.device)

        if self.in_new:
            local = _local(new_t)
            out_splits = [rows * row_width for rows in planner.recv_row_splits(self.rank)]
            if local.dtype != dtype or not local.is_contiguous() or local.numel() != sum(out_splits):
                raise RuntimeError(
                    f"target local shard of {item['name']} on rank {self.rank} does not match the plan: "
                    f"shape={tuple(local.shape)} dtype={local.dtype} contiguous={local.is_contiguous()} "
                    f"expected_numel={sum(out_splits)}"
                )
            recv = local.view(-1)
        else:
            out_splits = [0] * group_size
            recv = torch.empty(0, dtype=dtype, device=self.device)

        dist.all_to_all_single(recv, send, output_split_sizes=out_splits, input_split_sizes=in_splits, group=self.group)

        elem = torch.empty(0, dtype=dtype).element_size()
        self.stats["net_recv_bytes"] += (sum(out_splits) - out_splits[self.my_pos]) * elem
        self.stats["net_send_bytes"] += (sum(in_splits) - in_splits[self.my_pos]) * elem
        self.stats["self_copy_bytes"] += out_splits[self.my_pos] * elem

    def _broadcast_replicated(self, item: dict[str, Any], old_t: torch.Tensor | None, new_t: torch.Tensor | None):
        tensor = _local(new_t) if self.in_new else _local(old_t)
        if self.in_new and self.in_old:
            tensor.copy_(_local(old_t))
        staging = tensor if tensor.device == self.device else tensor.to(self.device)
        dist.broadcast(staging, src=self.src, group=self.group)
        if staging is not tensor:
            tensor.copy_(staging)
        if self.in_new and self.rank != self.src:
            self.stats["net_recv_bytes"] += tensor.numel() * tensor.element_size()

    def _move(self, item: dict[str, Any], old_t: torch.Tensor | None, new_t: torch.Tensor | None):
        if item["kind"] == "shard0":
            self._all_to_all_rows(item, old_t, new_t)
        else:
            self._broadcast_replicated(item, old_t, new_t)

    # -- phases ------------------------------------------------------------------------------------------
    def run(self) -> dict[str, int | float]:
        start = time.perf_counter()
        if self.in_old:
            reshard_fsdp2_modules(self.old.module)
        if self.in_new:
            reshard_fsdp2_modules(self.new.module)

        old_params = dict(self.old.module.named_parameters()) if self.in_old else {}
        new_params = dict(self.new.module.named_parameters()) if self.in_new else {}
        old_bufs = dict(self.old.module.named_buffers()) if self.in_old else {}
        new_bufs = dict(self.new.module.named_buffers()) if self.in_new else {}

        obj = [_build_manifest(self.old, with_units=self.pipeline) if self.rank == self.src else None]
        dist.broadcast_object_list(obj, src=self.src, group=self.group, device=self.device)
        manifest = obj[0]
        self.manifest = manifest
        self.old_params, self.new_params = old_params, new_params
        # Fail fast if the freshly built (or the live) model does not have exactly the source's structure.
        # The verdict is exchanged so that every rank raises instead of one rank raising and the others
        # waiting in the first all_to_all until the NCCL timeout.
        checks = []
        if self.in_old:
            checks.append((old_params, old_bufs, self.old))
        if self.in_new:
            checks.append((new_params, new_bufs, self.new))
        error = None
        try:
            for params, bufs, runtime in checks:
                _check_structure(manifest["params"], params, "parameter")
                _check_structure(manifest["buffers"], bufs, "buffer")
                if self.pipeline:
                    _check_units(manifest["units"], fsdp2_units(runtime.module))
        except Exception as exc:  # noqa: BLE001
            error = f"rank {self.rank}: {exc}"
        raise_if_any_failed(error, self.group, self.device)

        opt_manifest = manifest["optimizer"]
        if opt_manifest is not None:
            if self.in_old and self.old.optimizer is None:
                raise RuntimeError("source rank has optimizer state in the manifest but no optimizer")
            if self.in_new and self.new.optimizer is None:
                raise RuntimeError("target runtime has no optimizer to receive optimizer state")
        items_by_name = {item["name"]: item for item in manifest["params"]}
        self._digest_index = self._build_digest_index(manifest)
        old_acc = self._new_digest_acc() if (self.verify and self.in_old) else None
        # Read before anything moves: the low-memory pipeline releases the old optimizer state unit by unit.
        old_scalars = _scalar_state(self.old) if (self.verify and self.in_old) else None
        self._old_mesh_group = None

        if self.pipeline:
            # 1+2. one FSDP2 unit at a time: allocate its new shards, move params + optimizer state, drop
            #      the old shards. Peak = max(old, new) + one unit instead of old + new.
            old_units = {unit.name: unit for unit in fsdp2_units(self.old.module)} if self.in_old else {}
            new_units = {unit.name: unit for unit in fsdp2_units(self.new.module)} if self.in_new else {}
            sizes = {unit["name"]: int(unit["size"]) for unit in manifest["units"]}
            expanding = len(self.new_ranks) > len(self.old_ranks)
            ordered = order_units(
                [ResizeUnit(u["name"], list(u["params"]), []) for u in manifest["units"]], sizes, expanding=expanding
            )
            for unit in ordered:
                if self.in_new:
                    materialize_unit(new_units[unit.name], self.new.module, self.device)
                for name in unit.param_names:
                    self._move_param_and_state(items_by_name[name])
                if old_acc is not None:
                    for name in unit.param_names:
                        self._digest_param_and_state(old_acc, name, self.old, old_params, self.old_ranks)
                if self.in_old:
                    release_unit(old_units[unit.name], self.old.module, self.old.optimizer)
        else:
            # 1+2. all parameters, then their optimizer state, in manifest order
            if old_acc is not None:
                for item in manifest["params"]:
                    self._digest_param_and_state(old_acc, item["name"], self.old, old_params, self.old_ranks)
            for item in manifest["params"]:
                self._move_param_and_state(item)

        # hyper-parameters and scheduler
        if opt_manifest is not None and self.in_new:
            groups = self.new.optimizer.param_groups
            if len(groups) != len(opt_manifest["param_groups"]):
                raise RuntimeError(
                    f"optimizer param_groups differ: source {len(opt_manifest['param_groups'])}, target {len(groups)}"
                )
            for group, hyper in zip(groups, opt_manifest["param_groups"], strict=False):
                group.update(hyper)
        if self.in_new and manifest["lr_scheduler"] is not None:
            if self.new.lr_scheduler is None:
                raise RuntimeError("source has an LR scheduler but the target runtime does not")
            self.new.lr_scheduler.load_state_dict(manifest["lr_scheduler"])

        # 3. buffers (not sharded by FSDP2; includes non-persistent ones such as rotary inv_freq)
        if old_acc is not None:
            for item in manifest["buffers"]:
                self._digest_one(
                    old_acc, self._digest_index[("buf", item["name"])], old_bufs[item["name"]], item, self.old_ranks
                )
        for item in manifest["buffers"]:
            self._broadcast_replicated(item, old_bufs.get(item["name"]), new_bufs.get(item["name"]))
            self.stats["buffer_count"] += 1

        if self.verify:
            if old_acc is not None and self._old_mesh_group is not None:
                dist.all_reduce(old_acc, group=self._old_mesh_group)
            self._verify(new_params, new_bufs, old_acc)
            self._verify_scalar_state(old_scalars)
            self.stats["verified"] = 1

        get_torch_device().synchronize(self.device)
        dist.barrier(group=self.group)
        self.stats["transfer_seconds"] = time.perf_counter() - start
        return dict(self.stats)

    def _move_param_and_state(self, item: dict[str, Any]) -> None:
        name = item["name"]
        self._move(item, self.old_params.get(name), self.new_params.get(name))
        self.stats["tensor_count"] += 1
        opt_manifest = self.manifest["optimizer"]
        state_meta = opt_manifest["state"].get(name) if opt_manifest is not None else None
        if state_meta is None:
            return
        if self.in_new:
            self._ensure_optimizer_state(self.new_params[name], item, state_meta)
        for key, tensor_item in state_meta["tensors"].items():
            old_s = self.old.optimizer.state[self.old_params[name]][key] if self.in_old else None
            new_s = self.new.optimizer.state[self.new_params[name]][key] if self.in_new else None
            self._move(tensor_item, old_s, new_s)
            self.stats["optimizer_tensor_count"] += 1

    # -- optional end-to-end verification -----------------------------------------------------------------
    def _build_digest_index(self, manifest) -> dict[tuple[str, str], int]:
        """Slot index of every moved tensor: parameters, their optimizer tensors, then buffers."""
        index: dict[tuple[str, str], int] = {}
        for item in manifest["params"]:
            index[("param", item["name"])] = len(index)
        opt_manifest = manifest["optimizer"]
        if opt_manifest is not None:
            for item in manifest["params"]:
                state_meta = opt_manifest["state"].get(item["name"])
                for key in state_meta["tensors"] if state_meta else ():
                    index[("opt", f"{item['name']}.{key}")] = len(index)
        for item in manifest["buffers"]:
            index[("buf", item["name"])] = len(index)
        return index

    def _new_digest_acc(self) -> torch.Tensor:
        return torch.zeros(2 * len(self._digest_index), dtype=torch.int64, device=self.device)

    def _digest_one(
        self, acc, slot: int, tensor: torch.Tensor, item: dict[str, Any], side_ranks: list[int], mesh=None
    ) -> None:
        view = acc[2 * slot : 2 * slot + 2]
        if item["kind"] == "shard0" and isinstance(tensor, DTensor):
            mesh = tensor.device_mesh if mesh is None else mesh
            row_offset, _ = compute_1d_shard_range(item["shape"][0], mesh.size(), mesh.get_local_rank())
            _shard_digest(_local(tensor), row_offset * prod(item["shape"][1:]), view)
            if side_ranks is self.old_ranks:
                self._old_mesh_group = mesh.get_group()
            else:
                self._new_mesh_group = mesh.get_group()
        elif self.rank == side_ranks[0]:  # replicated: count once per side
            _shard_digest(_local(tensor), 0, view)

    def _digest_param_and_state(self, acc, name: str, runtime, params, side_ranks: list[int]) -> None:
        item = self._item(name)
        param = params[name]
        # Optimizer state is laid out like its parameter, but only the parameter is guaranteed to live on
        # the mesh the runtime was built on: state restored from a checkpoint is an unpickled DTensor whose
        # own mesh names a process group of the process that wrote the checkpoint.
        mesh = param.device_mesh if isinstance(param, DTensor) else None
        self._digest_one(acc, self._digest_index[("param", name)], param, item, side_ranks, mesh)
        opt_manifest = self.manifest["optimizer"]
        state_meta = opt_manifest["state"].get(name) if opt_manifest is not None else None
        if state_meta:
            state = runtime.optimizer.state[param]
            for key, tensor_item in state_meta["tensors"].items():
                slot = self._digest_index[("opt", f"{name}.{key}")]
                self._digest_one(acc, slot, state[key], tensor_item, side_ranks, mesh)

    def _item(self, name: str) -> dict[str, Any]:
        if not hasattr(self, "_items_by_name"):
            self._items_by_name = {item["name"]: item for item in self.manifest["params"]}
        return self._items_by_name[name]

    def _verify(self, new_params, new_bufs, old_acc: torch.Tensor | None) -> None:
        """Compare checksums of every moved tensor on the new side with the old side's."""
        if self.in_new:
            self._new_mesh_group = None
            new_acc = self._new_digest_acc()
            for item in self.manifest["params"]:
                self._digest_param_and_state(new_acc, item["name"], self.new, new_params, self.new_ranks)
            for item in self.manifest["buffers"]:
                self._digest_one(
                    new_acc, self._digest_index[("buf", item["name"])], new_bufs[item["name"]], item, self.new_ranks
                )
            if self._new_mesh_group is not None:
                dist.all_reduce(new_acc, group=self._new_mesh_group)
            reference = old_acc.clone() if self.rank == self.src else torch.empty_like(new_acc)
        else:
            new_acc = None
            reference = old_acc.clone()  # old-only ranks just relay the broadcast
        dist.broadcast(reference, src=self.src, group=self.group)
        error = None
        if self.in_new:
            bad = (reference.view(-1, 2) != new_acc.view(-1, 2)).any(dim=1).nonzero().flatten().tolist()
            if bad:
                slot_names = {slot: key for key, slot in self._digest_index.items()}
                examples = [slot_names[i] for i in bad[:5]]
                error = f"rank {self.rank}: {len(bad)} tensors differ after transfer, e.g. {examples}"
        raise_if_any_failed(error, self.group, self.device)

    def _verify_scalar_state(self, old_state: dict[str, str] | None) -> None:
        """Compare the state that travels in the manifest, which the checksums do not see.

        Every old rank reports what its live runtime held before the transfer and every new rank what its
        live runtime holds now.  Each report has to equal the first old rank's, entry by entry.  All ranks
        see the same reports, so all of them raise on a difference.
        """
        mine = {"old": old_state, "new": _scalar_state(self.new) if self.in_new else None}
        reports: list[Any] = [None] * len(self.group_ranks)
        dist.all_gather_object(reports, mine, group=self.group)
        reference = reports[self.group_ranks.index(self.src)]["old"]
        differences = []
        for rank, report in zip(self.group_ranks, reports, strict=True):
            for side, state in report.items():
                for path in sorted(reference.keys() | state.keys()) if state is not None else ():
                    if reference.get(path) != state.get(path):
                        differences.append(
                            f"{path} is {state.get(path, 'missing')} on {side} rank {rank}, "
                            f"{reference.get(path, 'missing')} on old rank {self.src}"
                        )
        if differences:
            raise RuntimeError(
                f"GPU-direct resize aborted: {len(differences)} scalar state entries differ after transfer, e.g. "
                + " | ".join(differences[:4])
            )
        self.stats["verified_scalar_count"] = len(reference)

    def _ensure_optimizer_state(self, param: torch.Tensor, item: dict[str, Any], state_meta: dict[str, Any]):
        state = self.new.optimizer.state[param]  # defaultdict(dict): creates the entry
        if state:
            return
        for key, tensor_item in state_meta["tensors"].items():
            if tensor_item["kind"] == "shard0":
                if tuple(tensor_item["shape"]) != tuple(item["shape"]):
                    raise NotImplementedError(
                        f"optimizer state {key} of {item['name']} is sharded but not parameter-shaped: "
                        f"{tensor_item['shape']} vs {item['shape']}"
                    )
                state[key] = torch.zeros_like(param, dtype=_resolve_dtype(tensor_item["dtype"]))
            else:
                state[key] = torch.empty(
                    tuple(tensor_item["shape"]), dtype=_resolve_dtype(tensor_item["dtype"]), device=self.device
                )
        for key, scalar in state_meta["scalars"].items():
            state[key] = torch.tensor(
                scalar["value"],
                dtype=_resolve_dtype(scalar["dtype"]),
                device=self.device if scalar["on_accelerator"] else "cpu",
            )


@torch.no_grad()
def transfer_fsdp2_runtime(
    *,
    old: ResizeRuntime | None,
    new: ResizeRuntime | None,
    old_ranks: Sequence[int],
    new_ranks: Sequence[int],
    group: dist.ProcessGroup,
    device: torch.device,
    verify: bool = False,
    pipeline: bool = False,
) -> dict[str, int | float]:
    """Move the training state of ``old`` (on ``old_ranks``) into ``new`` (on ``new_ranks``) in place.

    Must be called by every rank in ``sorted(set(old_ranks) | set(new_ranks))`` with ``group`` spanning
    exactly those ranks.  Ranks only in ``old_ranks`` pass ``new=None`` (they only send); ranks only in
    ``new_ranks`` pass ``old=None`` (they only receive).  Returns per-rank transfer statistics.

    With ``verify=True`` an exact, position-sensitive checksum of every moved tensor (reduced over each
    side's mesh, no all-gather) is compared between the two sides, and so is the state that travels in the
    manifest instead (optimizer scalars such as ``step``, ``param_groups`` hyper-parameters, LR-scheduler
    state), read from the live runtimes entry by entry.  A mismatch raises on every rank.

    With ``pipeline=True`` the new model must still have meta parameters (buffers materialised, see
    ``materialize_buffers``); shards are allocated, moved and released one FSDP2 unit at a time so the
    peak device memory is max(old, new) + one unit instead of old + new. The old runtime is hollow
    afterwards and must be discarded.
    """
    return _Transfer(
        old=old,
        new=new,
        old_ranks=old_ranks,
        new_ranks=new_ranks,
        group=group,
        device=device,
        verify=verify,
        pipeline=pipeline,
    ).run()
