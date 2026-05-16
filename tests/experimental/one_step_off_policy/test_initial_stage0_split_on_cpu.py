from types import SimpleNamespace

from omegaconf import OmegaConf

import verl.experimental.one_step_off_policy.ray_trainer as ray_trainer_module
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer


class _DummyPool:
    def __init__(self, world_size: int, name: str):
        self.world_size = world_size
        self.name = name


def test_init_worker_groups_respects_negative_step_stage0_split(monkeypatch):
    shared_pool = _DummyPool(world_size=8, name="shared_pool")
    actor_pool = _DummyPool(world_size=6, name="shared_pool_split_actor")
    rollout_pool = _DummyPool(world_size=2, name="shared_pool_split_rollout")

    def _fake_split_resource_pool(resource_pool, split_size):
        assert resource_pool is shared_pool
        assert split_size == [6, 2]
        return [actor_pool, rollout_pool]

    class _FakeRayWorkerGroup:
        def __init__(self, *, resource_pool, ray_cls_with_init, **kwargs):
            self.resource_pool = resource_pool
            self.ray_cls_with_init = ray_cls_with_init

        def spawn(self, prefix_set):
            return {
                prefix: SimpleNamespace(resource_pool=self.resource_pool, workers=[object()] * self.resource_pool.world_size)
                for prefix in prefix_set
            }

    monkeypatch.setattr(ray_trainer_module, "split_resource_pool", _fake_split_resource_pool)
    monkeypatch.setattr(ray_trainer_module, "create_colocated_worker_cls", lambda class_dict: object())

    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "dynamic_resize": {
                    "enable": True,
                    "schedule": {
                        "stage0": {
                            "step": -1,
                            "actor_pool": {"mode": "split", "from_pool": "shared_pool", "index": 0, "size": [6, 2]},
                            "rollout_pool": {"mode": "split", "from_pool": "shared_pool", "index": 1, "size": [6, 2]},
                        }
                    },
                }
            }
        }
    )
    trainer.resource_pool_manager = SimpleNamespace(resource_pool_dict={"shared_pool": shared_pool}, get_resource_pool=lambda role: shared_pool)
    trainer.resource_pool_to_cls = {shared_pool: {"actor": object(), "rollout": object()}}
    trainer.ray_worker_group_cls = _FakeRayWorkerGroup
    trainer._get_detached_workers_cfg = lambda: None
    trainer._get_worker_group_kwargs = lambda: {}

    trainer._init_worker_groups()

    assert trainer.all_wg["actor"].resource_pool is actor_pool
    assert trainer.all_wg["rollout"].resource_pool is rollout_pool
    assert len(trainer.all_wg["actor"].workers) == 6
    assert len(trainer.all_wg["rollout"].workers) == 2