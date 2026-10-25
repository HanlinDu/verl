"""Stale DTensor dispatch cache regression across a destroyed resize subgroup.

``DeviceMesh.__eq__`` compares mesh content and ignores the underlying process
group, while the group names differ per ``new_group`` call. DTensor's sharding
propagator caches a spec that embeds the *first* mesh, so after a resize
destroys that group and a later resize recreates an equal-content mesh, a
cache-hit result carries the dead mesh and its group can no longer be resolved.

Run directly:
    python tests/special_distributed/test_dtensor_cache_invalidation_on_cpu.py

This is CPU/Gloo evidence for cache lifetime only. It is not CUDA/NCCL proof.
"""

import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ``verl`` and ``verl.utils`` pull Ray and tokenizer deps at import time. Stub
# them as namespace packages so the cache helper loads in a CPU-only process.
for name, path in (("verl", ROOT / "verl"), ("verl.utils", ROOT / "verl/utils")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

from verl.utils.dtensor_cache import clear_dtensor_dispatch_caches  # noqa: E402


def worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)

    # Cycle 1: populate the sharding-propagator cache with a spec on m1.
    group1 = dist.new_group([0, 1, 2, 3])
    mesh1 = DeviceMesh.from_group(group1, "cpu", mesh=torch.arange(4))
    local = torch.ones(4, 4)
    tensor1 = DTensor.from_local(local, mesh1, [Shard(0)], run_check=False, shape=(16, 4), stride=(4, 1))
    _ = tensor1.sum()

    # A resize destroys the old subgroup. Without cache invalidation the next
    # cycle's cache hit still points at this dead mesh.
    dist.destroy_process_group(group1)
    clear_dtensor_dispatch_caches()
    dist.barrier()

    # Cycle 2: equal content, fresh group and mesh.
    group2 = dist.new_group([0, 1, 2, 3])
    mesh2 = DeviceMesh.from_group(group2, "cpu", mesh=torch.arange(4))
    tensor2 = DTensor.from_local(local, mesh2, [Shard(0)], run_check=False, shape=(16, 4), stride=(4, 1))
    result = tensor2.sum()

    assert result.device_mesh is mesh2, (
        f"rank {rank}: cache hit reused mesh {result.device_mesh._dim_group_names} "
        f"instead of live mesh {mesh2._dim_group_names}"
    )
    # Resolving the group is what failed with the stale spec.
    resolved = result.device_mesh.get_group()
    assert dist.get_backend(resolved) == "gloo"

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("stale DTensor cache regression PASS; CUDA/NCCL NOT tested", flush=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(f"file://{directory}/rendezvous",), nprocs=4, join=True)


if __name__ == "__main__":
    main()
