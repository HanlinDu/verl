# Copyright 2026 Individual Contributor: HanlinDu
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
"""NCCL payload/placement smoke; launch in each container with torchrun."""

import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from verify import PROFILES, identity


def main():
    """Check all-reduce, broadcast, all-gather, and ring payloads on actual GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    model, nodes = PROFILES[args.profile]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Expose exactly eight GPUs per container")
    if args.output.exists():
        raise RuntimeError("Use a fresh probe output path on both nodes")
    torch.cuda.set_device(local_rank)
    before = identity()
    if not before["clean"]:
        raise RuntimeError("Probe requires a clean candidate checkout on every node")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != nodes * 8:
            raise RuntimeError("World size does not match the requested profile")
        gpu = torch.cuda.get_device_name(local_rank)
        if model.lower() not in gpu.lower():
            raise RuntimeError("Wrong GPU model for this profile: " + gpu)
        info = {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "gpu": gpu,
            "capability": list(torch.cuda.get_device_capability(local_rank)),
            **before,
        }
        ranks = [None] * world
        dist.all_gather_object(ranks, info)
        if len({r["host"] for r in ranks}) != nodes:
            raise RuntimeError("Expected distinct node hostnames; give each SSH container a unique hostname")
        if any(r["sha"] != before["sha"] or r["tree"] != before["tree"] for r in ranks):
            raise RuntimeError("Containers are running different code")
        values = torch.full((1024 * 1024,), float(rank + 1), device="cuda")
        dist.all_reduce(values)
        torch.testing.assert_close(values, torch.full_like(values, world * (world + 1) / 2), rtol=0, atol=0)
        values.fill_(float(rank))
        dist.broadcast(values, src=0)
        torch.testing.assert_close(values, torch.zeros_like(values), rtol=0, atol=0)
        scalar = torch.tensor([rank], device="cuda", dtype=torch.int64)
        gathered = [torch.empty_like(scalar) for _ in range(world)]
        dist.all_gather(gathered, scalar)
        if [x.item() for x in gathered] != list(range(world)):
            raise RuntimeError("All-gather payload mismatch")
        outgoing = torch.full((1024,), rank, device="cuda", dtype=torch.int64)
        incoming = torch.empty_like(outgoing)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, outgoing, (rank + 1) % world),
                dist.P2POp(dist.irecv, incoming, (rank - 1) % world),
            ]
        )
        for request in requests:
            request.wait()
        torch.testing.assert_close(incoming, torch.full_like(incoming, (rank - 1) % world), rtol=0, atol=0)
        torch.cuda.synchronize()
        if identity() != before:
            raise RuntimeError("Checkout changed during probe")
        dist.barrier()
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "schema": 1,
                "kind": "nccl-probe",
                "status": "passed",
                "profile": args.profile,
                **before,
                "ranks": ranks,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": list(torch.cuda.nccl.version()),
                "checks": ["all_reduce", "broadcast", "all_gather", "ring_p2p"],
            }
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2)
            print(json.dumps({"status": "passed", "output": str(args.output), "world_size": world}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
