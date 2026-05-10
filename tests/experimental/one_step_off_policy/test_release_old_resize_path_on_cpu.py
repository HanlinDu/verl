import asyncio

from omegaconf import OmegaConf

from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer


def test_release_old_shared_pool_resize_is_normalized_to_staged_path():
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 2
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "dynamic_resize": {
                    "enable": True,
                    "schedule": {
                        "stage1": {
                            "step": 2,
                            "actor_pool": {"mode": "split", "from_pool": "shared_pool", "index": 0, "size": [6, 2]},
                            "rollout_pool": {
                                "mode": "split",
                                "from_pool": "shared_pool",
                                "index": 1,
                                "size": [6, 2],
                            },
                            "release_old": True,
                        }
                    },
                }
            }
        }
    )
    trainer._resize_controller = None
    trainer._latest_resize_control_metrics = {}

    staged_calls = []

    def _gate(_item):
        return True, 0, {}

    async def _staged(item, *, actor_resume_path=None):
        staged_calls.append((item, actor_resume_path))
        return True

    async def _unexpected_async(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("generic resize path should not be used for shared-pool split resize")

    def _unexpected_sync(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("generic resize path should not be used for shared-pool split resize")

    trainer._gate_dynamic_resize_schedule_item = _gate
    trainer._should_use_staged_shared_pool_resize = lambda item: True
    trainer._staged_resize_shared_pool = _staged
    trainer.add_role_group_async = _unexpected_async
    trainer._switch_actor_rollout_groups = _unexpected_async
    trainer._resolve_role_pool = _unexpected_sync

    asyncio.run(trainer._maybe_dynamic_resize())

    assert len(staged_calls) == 1
    item, actor_resume_path = staged_calls[0]
    assert item["release_old"] is True
    assert actor_resume_path is None
