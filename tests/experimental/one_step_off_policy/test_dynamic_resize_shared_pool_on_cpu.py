from types import SimpleNamespace

from omegaconf import OmegaConf

import verl.experimental.one_step_off_policy.ray_trainer as ray_trainer_module
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.separation.utils import DynamicResizePoolTopology
from verl.trainer.ppo.utils import Role


class _DummyPool:
    def __init__(self, world_size: int, name: str):
        self.world_size = world_size
        self.name = name


def test_init_resource_pools_splits_initial_dynamic_resize_shared_pool(monkeypatch):
    shared_pool = _DummyPool(world_size=8, name="dynamic_resize_shared_pool")
    actor_pool = _DummyPool(world_size=6, name="dynamic_resize_actor_pool")
    rollout_pool = _DummyPool(world_size=2, name="dynamic_resize_rollout_pool")

    def _fake_split_resource_pool(resource_pool, split_size):
        assert resource_pool is shared_pool
        assert split_size == [6, 2]
        return [actor_pool, rollout_pool]

    monkeypatch.setattr(ray_trainer_module, "split_resource_pool", _fake_split_resource_pool)

    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 6,
                "dynamic_resize": {"enable": True, "shared_pool": True},
            },
            "rollout": {"nnodes": 1, "n_gpus_per_node": 2},
        }
    )
    trainer.resource_pool_manager = SimpleNamespace(
        dynamic_resize_topology=DynamicResizePoolTopology(
            shared_pool_name="dynamic_resize_shared_pool",
            shared_pool_spec=[8],
            initial_actor_size=6,
            initial_rollout_size=2,
            schedule_splits=[],
        ),
        resource_pool_dict={"dynamic_resize_shared_pool": shared_pool},
        mapping={
            Role.Actor: "dynamic_resize_shared_pool",
            Role.Critic: "dynamic_resize_shared_pool",
            Role.RefPolicy: "dynamic_resize_shared_pool",
        },
        create_resource_pool=lambda: None,
    )

    trainer._init_resource_pools()

    assert trainer.resource_pool_manager.resource_pool_dict["dynamic_resize_actor_pool"] is actor_pool
    assert trainer.resource_pool_manager.resource_pool_dict["dynamic_resize_rollout_pool"] is rollout_pool
    assert trainer.resource_pool_manager.mapping[Role.Actor] == "dynamic_resize_actor_pool"
    assert trainer.resource_pool_manager.mapping[Role.Critic] == "dynamic_resize_actor_pool"
    assert trainer.resource_pool_manager.mapping[Role.RefPolicy] == "dynamic_resize_actor_pool"
    assert trainer._dynamic_resize_rollout_resource_pool is rollout_pool
    assert trainer._active_dynamic_resize_topology == {"actor_size": 6, "rollout_size": 2}
    assert trainer.resource_pool_to_cls == {actor_pool: {}}
