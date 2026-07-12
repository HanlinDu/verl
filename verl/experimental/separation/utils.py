# Copyright 2025 Meituan Ltd. and/or its affiliates
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


from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy


SHARED_DYNAMIC_RESIZE_POOL = "dynamic_resize_shared_pool"


@dataclass(frozen=True, slots=True)
class DynamicResizePoolTopology:
    shared_pool_name: str
    shared_pool_spec: list[int]
    initial_actor_size: int
    initial_rollout_size: int
    schedule_splits: list[dict[str, int]]

    @property
    def total_size(self) -> int:
        return sum(self.shared_pool_spec)


def _cfg_to_plain_dict(cfg) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        return OmegaConf.to_container(cfg, resolve=True) or {}
    except Exception:
        return dict(cfg) if hasattr(cfg, "items") else {}


def _pool_store(*, nnodes: int, n_gpus_per_node: int) -> list[int]:
    return [int(n_gpus_per_node)] * int(nnodes)


def _combine_pool_stores(left: list[int], right: list[int]) -> list[int]:
    node_count = max(len(left), len(right))
    return [
        (left[node_idx] if node_idx < len(left) else 0) + (right[node_idx] if node_idx < len(right) else 0)
        for node_idx in range(node_count)
    ]


def _pool_target_size(spec: Any) -> int | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    if not isinstance(spec, dict):
        return None
    if "world_size" in spec:
        return int(spec["world_size"])
    if "n_gpus" in spec:
        return int(spec["n_gpus"])
    if "size" not in spec:
        return None
    size = spec["size"]
    if isinstance(size, int):
        return size
    if isinstance(size, (list, tuple)):
        index = int(spec.get("index", 0))
        if index < 0 or index >= len(size):
            raise ValueError(f"dynamic resize pool index {index} out of range for size={size}")
        return int(size[index])
    return None


def dynamic_resize_shared_pool_enabled(config) -> bool:
    resize_cfg = _cfg_to_plain_dict(OmegaConf.select(config, "trainer.dynamic_resize"))
    return bool(resize_cfg.get("enable", False) and resize_cfg.get("shared_pool", False))


def build_dynamic_resize_pool_topology(config) -> DynamicResizePoolTopology | None:
    if not dynamic_resize_shared_pool_enabled(config):
        return None

    resize_cfg = _cfg_to_plain_dict(OmegaConf.select(config, "trainer.dynamic_resize"))
    trainer_store = _pool_store(
        nnodes=int(config.trainer.nnodes),
        n_gpus_per_node=int(config.trainer.n_gpus_per_node),
    )
    rollout_store = _pool_store(
        nnodes=int(config.rollout.nnodes),
        n_gpus_per_node=int(config.rollout.n_gpus_per_node),
    )
    shared_pool_spec = _combine_pool_stores(trainer_store, rollout_store)
    initial_actor_size = sum(trainer_store)
    initial_rollout_size = sum(rollout_store)
    total_size = sum(shared_pool_spec)

    schedule = resize_cfg.get("schedule", []) or []
    if isinstance(schedule, dict):
        schedule = list(schedule.values())
    if not isinstance(schedule, list):
        raise TypeError("trainer.dynamic_resize.schedule must be a list or dict")

    schedule_splits = []
    for item in schedule:
        if not isinstance(item, dict):
            raise TypeError(f"Each trainer.dynamic_resize.schedule item must be a dict, got {type(item).__name__}")
        actor_size = _pool_target_size(item.get("actor_pool"))
        rollout_size = _pool_target_size(item.get("rollout_pool"))
        if actor_size is None and rollout_size is None:
            continue
        actor_size = initial_actor_size if actor_size is None else actor_size
        rollout_size = initial_rollout_size if rollout_size is None else rollout_size
        if actor_size + rollout_size != total_size:
            raise ValueError(
                "dynamic resize shared_pool schedule must keep actor_pool + rollout_pool equal to "
                f"the shared pool size {total_size}, got actor={actor_size}, rollout={rollout_size}"
            )
        schedule_splits.append(
            {
                "step": int(item.get("step", -1)),
                "actor_size": int(actor_size),
                "rollout_size": int(rollout_size),
            }
        )

    return DynamicResizePoolTopology(
        shared_pool_name=SHARED_DYNAMIC_RESIZE_POOL,
        shared_pool_spec=shared_pool_spec,
        initial_actor_size=initial_actor_size,
        initial_rollout_size=initial_rollout_size,
        schedule_splits=schedule_splits,
    )


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}
    dynamic_resize_topology = build_dynamic_resize_pool_topology(config)

    # Actor/Critic resource pool
    training_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]
    if any(role in roles for role in training_roles):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        if dynamic_resize_topology is not None:
            resource_pool_spec[dynamic_resize_topology.shared_pool_name] = dynamic_resize_topology.shared_pool_spec
            trainer_pool_name = dynamic_resize_topology.shared_pool_name
        else:
            trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
            resource_pool_spec["trainer_pool"] = trainer_pool
            trainer_pool_name = "trainer_pool"

        for role in training_roles:
            if role in roles:
                mapping[role] = trainer_pool_name

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"
        if dynamic_resize_topology is not None:
            mapping[Role.Rollout] = dynamic_resize_topology.shared_pool_name

    if Role.RewardModel in roles:
        rm_cfg = config.reward.reward_model
        assert rm_cfg.n_gpus_per_node > 0, "config.reward.reward_model.n_gpus_per_node must be greater than 0"
        assert rm_cfg.nnodes > 0, "config.reward.reward_model.nnodes must be greater than 0"

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.dynamic_resize_topology = dynamic_resize_topology
    return resource_pool_manager


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Always use the unified model engine worker implementation.
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.engine_workers import TrainingWorker

    ray_worker_group_cls = RayWorkerGroup

    train_role = Role.Actor
    if config.get("async_training", {}).get("use_trainer_do_validate", False):
        train_role = Role.ActorRollout

    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Critic: ray.remote(TrainingWorker),
    }

    # Add reference policy (if KL loss or reward is required)
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls
