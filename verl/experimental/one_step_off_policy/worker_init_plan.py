from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from verl.utils.device import get_device_name, get_nccl_backend


@dataclass(slots=True)
class WorkerCommInitPlan:
    """Pure metadata needed to bind worker-side communicators at commit time."""

    role: str
    rank: int
    world_size: int
    init_method: str
    timeout_seconds: int
    backend: str
    fsdp_size: int
    ulysses_sequence_parallel_size: int
    ulysses_mesh_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "rank": self.rank,
            "world_size": self.world_size,
            "init_method": self.init_method,
            "timeout_seconds": self.timeout_seconds,
            "backend": self.backend,
            "fsdp_size": self.fsdp_size,
            "ulysses_sequence_parallel_size": self.ulysses_sequence_parallel_size,
            "ulysses_mesh_shape": list(self.ulysses_mesh_shape) if self.ulysses_mesh_shape is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerCommInitPlan":
        ulysses_mesh_shape = data.get("ulysses_mesh_shape")
        return cls(
            role=str(data["role"]),
            rank=int(data["rank"]),
            world_size=int(data["world_size"]),
            init_method=str(data.get("init_method") or "env://"),
            timeout_seconds=int(data.get("timeout_seconds", 600)),
            backend=str(data["backend"]),
            fsdp_size=int(data["fsdp_size"]),
            ulysses_sequence_parallel_size=int(data.get("ulysses_sequence_parallel_size", 1)),
            ulysses_mesh_shape=tuple(int(v) for v in ulysses_mesh_shape) if ulysses_mesh_shape is not None else None,
        )


def build_worker_comm_init_plan(
    config,
    role: str,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    init_method: str | None = None,
    device_type: str | None = None,
    nccl_backend: str | None = None,
) -> WorkerCommInitPlan:
    """Build the communicator portion of worker init as metadata only.

    The plan is safe to prepare before the final switch window because it does
    not allocate process groups or device meshes; it only records how commit
    should do so later.
    """

    resolved_rank = int(rank if rank is not None else os.environ.get("RANK", 0))
    resolved_world_size = int(world_size if world_size is not None else os.environ.get("WORLD_SIZE", 1))
    resolved_init_method = str(init_method or os.environ.get("DIST_INIT_METHOD") or "env://")
    resolved_device_type = str(device_type or get_device_name())
    resolved_nccl_backend = str(nccl_backend or get_nccl_backend())
    timeout_seconds = int(config.get("nccl_timeout", 600))
    fsdp_size = int(config.actor.fsdp_config.fsdp_size)
    ulysses_sequence_parallel_size = int(config.actor.get("ulysses_sequence_parallel_size", 1))
    if ulysses_sequence_parallel_size < 1:
        raise ValueError(f"ulysses_sequence_parallel_size must be >= 1, got {ulysses_sequence_parallel_size}")
    if resolved_world_size % ulysses_sequence_parallel_size != 0:
        raise ValueError(
            "world_size must be divisible by ulysses_sequence_parallel_size, "
            f"got world_size={resolved_world_size}, ulysses_sequence_parallel_size={ulysses_sequence_parallel_size}"
        )

    ulysses_mesh_shape = None
    if ulysses_sequence_parallel_size > 1:
        ulysses_mesh_shape = (resolved_world_size // ulysses_sequence_parallel_size, ulysses_sequence_parallel_size)

    return WorkerCommInitPlan(
        role=role,
        rank=resolved_rank,
        world_size=resolved_world_size,
        init_method=resolved_init_method,
        timeout_seconds=timeout_seconds,
        backend=f"cpu:gloo,{resolved_device_type}:{resolved_nccl_backend}",
        fsdp_size=fsdp_size,
        ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
        ulysses_mesh_shape=ulysses_mesh_shape,
    )