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

"""Exercise migration boundaries without importing optional GPU engines."""

import ast
import asyncio
import os
from enum import Enum
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
TRAINER = "verl/experimental/one_step_off_policy/ray_trainer.py"


def _load_methods(path, owner, names, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    cls.bases = []
    cls.keywords = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    for node in cls.body:
        node.decorator_list = []
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[owner]


class _Role(Enum):
    Actor = "actor"
    ActorRollout = "actor_rollout"
    ActorRolloutRef = "actor_rollout_ref"
    Critic = "critic"

    def __str__(self):
        return self.value


def test_ordinary_recipes_keep_separate_nccl_runtime():
    folder = ROOT / "verl/experimental/one_step_off_policy/config"
    for name in ("one_step_off_ppo_trainer.yaml", "one_step_off_ppo_megatron_trainer.yaml"):
        cfg = OmegaConf.load(folder / name)
        assert not OmegaConf.select(cfg, "trainer.dynamic_resize.enable")
        assert not OmegaConf.select(cfg, "actor_rollout_ref.hybrid_engine", default=False)
        assert cfg.actor_rollout_ref.rollout.checkpoint_engine.backend == "nccl"


@pytest.mark.parametrize("hybrid", [False, True])
def test_worker_role_matches_selected_runtime(hybrid):
    cls = _load_methods(
        TRAINER,
        "OneStepOffRayTrainer",
        {"_create_actor_rollout_classes"},
        Role=_Role,
        RayClassWithInitArgs=lambda **kwargs: kwargs,
    )
    trainer = cls()
    trainer.hybrid_engine = hybrid
    role = _Role.ActorRolloutRef if hybrid else _Role.Actor
    trainer.role_worker_mapping = {role: object()}
    trainer.resource_pool_manager = SimpleNamespace(get_resource_pool=lambda _: "pool")
    trainer.resource_pool_to_cls = {"pool": {}}
    trainer.config = SimpleNamespace(actor_rollout_ref={})
    trainer._create_actor_rollout_classes()
    assert trainer.actor_role == role
    assert set(trainer.resource_pool_to_cls["pool"]) == {str(role)}


def test_separate_rollout_manager_receives_no_actor_worker_group():
    create = Mock(return_value=object())
    cls = _load_methods(
        TRAINER,
        "OneStepOffRayTrainer",
        {"_init_async_rollout_manager"},
        LLMServerManager=SimpleNamespace(create=create),
    )
    trainer = cls()
    trainer.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"mode": "async"}}})
    trainer.hybrid_engine = False
    trainer.actor_rollout_wg = object()
    trainer._init_agent_loop_manager = Mock()
    trainer._init_async_rollout_manager()
    assert create.call_args.kwargs["worker_group"] is None


def test_direct_restore_still_loads_scheduler_and_rng():
    cls = _load_methods(
        TRAINER,
        "OneStepOffRayTrainer",
        {"_load_dynamic_resize_checkpoint"},
        os=os,
        Role=_Role,
    )
    trainer = cls()
    trainer.actor_rollout_wg = SimpleNamespace(load_checkpoint=Mock(), load_replicated_state_for_resize=Mock())
    trainer.critic_wg = SimpleNamespace(load_checkpoint=Mock())
    trainer.use_critic = True
    trainer._load_dynamic_resize_checkpoint("checkpoint", load_actor=False)
    trainer.actor_rollout_wg.load_checkpoint.assert_not_called()
    trainer.actor_rollout_wg.load_replicated_state_for_resize.assert_called_once_with("checkpoint/actor")
    trainer.critic_wg.load_checkpoint.assert_called_once_with("checkpoint/critic", del_local_after_load=False)


def test_replicated_restore_uses_checkpoint_extra_without_touching_optimizer(tmp_path, monkeypatch):
    rng = Mock()
    scheduler = Mock()
    optimizer = Mock()
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    torch.save(
        {"rng": {"seed": 123}, "lr_scheduler": {"last_epoch": 7}}, tmp_path / "extra_state_world_size_2_rank_0.pt"
    )
    cls = _load_methods(
        "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
        "FSDPCheckpointManager",
        {"_load_extra_state", "load_replicated_state"},
        torch=torch,
        os=os,
        glob=glob,
        copy_to_local=lambda path: path,
        log_with_rank=Mock(),
        logger=None,
    )
    manager = cls()
    manager.should_load_extra = True
    manager.world_size = 4
    manager.rank = 3
    manager.process_group = object()
    manager.lr_scheduler = scheduler
    manager.load_rng_state = rng
    manager.optimizer = optimizer
    manager.load_replicated_state(str(tmp_path))
    rng.assert_called_once_with({"seed": 123})
    scheduler.load_state_dict.assert_called_once_with({"last_epoch": 7})
    optimizer.load_state_dict.assert_not_called()
    barrier.assert_called_once_with(group=manager.process_group)


def test_quiesce_failure_stops_before_workers_are_destroyed():
    cls = _load_methods(TRAINER, "OneStepOffRayTrainer", {"_destroy_dynamic_resize_runtime"})
    trainer = cls()
    trainer.checkpoint_manager = SimpleNamespace(
        abort_replicas=AsyncMock(side_effect=RuntimeError("inflight rollout")), sleep_replicas=AsyncMock()
    )
    trainer._kill_ray_actors = Mock()
    with pytest.raises(RuntimeError, match="quiescent"):
        asyncio.run(trainer._destroy_dynamic_resize_runtime())
    trainer._kill_ray_actors.assert_not_called()
