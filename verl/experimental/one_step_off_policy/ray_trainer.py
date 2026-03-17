# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import contextlib
import os
import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.collective import collective
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.experimental.one_step_off_policy.utils import need_critic
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls, split_resource_pool
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils import omega_conf_to_dataclass
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger


class OneStepOffRayTrainer(RayPPOTrainer):
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger()
        # Track active worker groups per role for dynamic switching.
        self.role_groups: dict[Role, dict[str, RayWorkerGroup]] = {
            Role.Actor: {},
            Role.Rollout: {},
        }

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = lora_rank > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _maybe_patch_reward_metadata(self, non_tensor_batch) -> None:
        # Keep this fallback intentionally narrow. We only synthesize reward
        # metadata for samples that already look like GSM8K-style supervised
        # examples (i.e. they contain `answer`). This avoids silently forcing a
        # GSM8K router onto unrelated datasets while still unblocking the
        # one-step-off path that currently depends on the shared reward-loop
        # contract.
        answers = non_tensor_batch.get("answer")
        if answers is None:
            return

        if "data_source" not in non_tensor_batch:
            non_tensor_batch["data_source"] = np.array(["openai/gsm8k"] * len(answers), dtype=object)

        if "reward_model" not in non_tensor_batch:
            non_tensor_batch["reward_model"] = np.array(
                [{"ground_truth": answer, "style": "rule"} for answer in answers],
                dtype=object,
            )

    def _validate(self):
        @contextlib.contextmanager
        def _patched_val_dataloader():
            original_val_dataloader = self.val_dataloader

            def _iter_with_required_reward_fields():
                for test_data in original_val_dataloader:
                    # One-step-off validation still delegates to the shared PPO
                    # validation pipeline. That shared path assumes each sample
                    # already carries reward-routing metadata (`data_source`) and
                    # rule/model reward metadata (`reward_model`). Some validation
                    # datasets used by this experimental trainer only contain the
                    # textual fields (for example `prompt` and `answer`), so we
                    # patch the missing metadata here instead of changing the base
                    # trainer's contract for all PPO variants.
                    self._maybe_patch_reward_metadata(test_data)

                    yield test_data

            self.val_dataloader = _iter_with_required_reward_fields()
            try:
                yield
            finally:
                self.val_dataloader = original_val_dataloader

        self.actor_rollout_wg = self.rollout_wg
        try:
            with _patched_val_dataloader():
                return super()._validate()
        finally:
            self.actor_rollout_wg = self.actor_wg

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_async_rollout_manager()

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_critic_class()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor, Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_critic_class(self):
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _get_detached_workers_cfg(self) -> dict | None:
        cfg = OmegaConf.select(self.config.trainer, "detached_workers")
        if cfg is None:
            return None
        return OmegaConf.to_container(cfg, resolve=True)

    def _get_worker_group_kwargs(self) -> dict:
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name
        return wg_kwargs

    def _init_worker_groups(self):
        detached_cfg = self._get_detached_workers_cfg()

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = self._get_worker_group_kwargs()

        # detached ray workers
        if detached_cfg and detached_cfg.get("enable"):
            name_prefix = detached_cfg.get("name_prefix", "verl_detached_")
            attach_only = detached_cfg.get("attach_only", False)
            worker_names_map = detached_cfg.get("worker_names", {}) or {}

            pool_name_map = {pool: name for name, pool in self.resource_pool_manager.resource_pool_dict.items()}

            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                pool_name = pool_name_map.get(resource_pool, "default")
                pool_worker_names = worker_names_map.get(pool_name)

                if pool_worker_names:
                    wg_dict = self.ray_worker_group_cls.from_detached(
                        name_prefix=name_prefix,
                        worker_names=pool_worker_names,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs,
                    )
                else:
                    if attach_only:
                        raise ValueError(
                            f"detached_workers.attach_only is True but worker_names missing for pool '{pool_name}'"
                        )
                    wg_dict = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        name_prefix=name_prefix,
                        detached=True,
                        **wg_kwargs,
                    )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)
        # normal ray workers
        else:
            # IMPORTANT:
            # We intentionally avoid using `spawn(prefix_set=...)` here.
            # `spawn` creates multiple WorkerGroups that share the same underlying Ray Actors
            # (via from_detached), which will make actor/rollout overlap on the same ActorID.
            # That overlap will deadlock weight sync (broadcast) in detached / dynamic resize mode.
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                for role_name, role_cls in class_dict.items():
                    worker_dict_cls = create_colocated_worker_cls(class_dict={role_name: role_cls})
                    wg = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs,
                    )
                    # The returned dict is keyed by the role name.
                    spawn_wg = wg.spawn(prefix_set=[role_name])
                    all_wg.update(spawn_wg)
        self.all_wg = all_wg

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg
        # Register the initial groups as the default targets.
        self.role_groups[Role.Actor] = {"primary": self.actor_wg}
        self.role_groups[Role.Rollout] = {"primary": self.rollout_wg}
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)
        self._create_weight_sync_group()

    def _build_role_group(
        self,
        role: Role,
        *,
        resource_pool=None,
        name_prefix: str | None = None,
        detached: bool = False,
    ) -> RayWorkerGroup:
        # Build a single-role group to keep dynamic add/switch logic reusable.
        resource_pool = resource_pool or self.resource_pool_manager.get_resource_pool(role)
        role_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[role],
            config=self.config.actor_rollout_ref,
            role=str(role),
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict={str(role): role_cls})
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            name_prefix=name_prefix,
            detached=detached,
            **self._get_worker_group_kwargs(),
        )
        # Build and spawn new worker group.
        return wg_dict.spawn(prefix_set=[str(role)])[str(role)]

    def add_role_group(
        self,
        role: Role,
        *,
        name: str | None = None,
        resource_pool=None,
        detached: bool = False,
        name_prefix: str | None = None,
    ) -> RayWorkerGroup:
        role_wg = self._build_role_group(
            role,
            resource_pool=resource_pool,
            name_prefix=name_prefix,
            detached=detached,
        )
        role_wg.init_model()
        group_map = self.role_groups.setdefault(role, {})
        group_name = name or f"{role.value.lower()}_group_{len(group_map)}"
        group_map[group_name] = role_wg
        return role_wg

    def switch_role_group(
        self,
        role: Role,
        role_wg: RayWorkerGroup,
        *,
        resume_from_path: str | None = None,
        release_old: bool = False,
    ) -> None:
        # Switch the active group and sync weights/manager as needed.
        if role == Role.Actor:
            old_wg = self.actor_wg
            if resume_from_path is not None:
                role_wg.load_checkpoint(
                    resume_from_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
            else:
                weights_info = old_wg.get_actor_weights_info()[0]
                role_wg.set_actor_weights_info(weights_info)

            self.actor_wg = role_wg
            self.actor_rollout_wg = role_wg
            weights_info = role_wg.get_actor_weights_info()[0]
            self.rollout_wg.set_actor_weights_info(weights_info)

        elif role == Role.Rollout:
            old_wg = self.rollout_wg
            if resume_from_path is not None:
                print("Warning: rollout group does not support checkpoint restore; ignoring resume_from_path")
            weights_info = self.actor_wg.get_actor_weights_info()[0]
            role_wg.set_actor_weights_info(weights_info)
            self.rollout_wg = role_wg
            # Rollout manager must be rebuilt to bind the new rollout workers.
            self._init_async_rollout_manager()

        else:
            raise ValueError(f"Unsupported role for switch: {role}")

        self._create_weight_sync_group()

        if release_old:
            self.remove_role_group(role, old_wg)

    def remove_role_group(self, role: Role, role_wg: RayWorkerGroup) -> None:
        for group_name, group in list(self.role_groups.get(role, {}).items()):
            if group is role_wg:
                self.role_groups[role].pop(group_name)
                break
        for worker in role_wg.workers:
            try:
                ray.kill(worker)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                print(f"Warning: failed to kill {role} worker {worker}: {exc}")

    def add_actor_group(self, **kwargs) -> RayWorkerGroup:
        return self.add_role_group(Role.Actor, **kwargs)

    def switch_actor_group(self, actor_wg: RayWorkerGroup, **kwargs) -> None:
        return self.switch_role_group(Role.Actor, actor_wg, **kwargs)

    def remove_actor_group(self, actor_wg: RayWorkerGroup) -> None:
        return self.remove_role_group(Role.Actor, actor_wg)

    def add_rollout_group(self, **kwargs) -> RayWorkerGroup:
        return self.add_role_group(Role.Rollout, **kwargs)

    def switch_rollout_group(self, rollout_wg: RayWorkerGroup, **kwargs) -> None:
        return self.switch_role_group(Role.Rollout, rollout_wg, **kwargs)

    def remove_rollout_group(self, rollout_wg: RayWorkerGroup) -> None:
        return self.remove_role_group(Role.Rollout, rollout_wg)

    def _create_weight_sync_group(self):
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)

        if self.device_name == "npu":
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                n_workers,
            )
            ray.get(
                self.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                )
            )
        else:
            # Create Ray collective group for fallback communication
            collective.create_collective_group(
                actor_rollout_workers,
                n_workers,
                list(range(0, n_workers)),
                backend=get_nccl_backend(),
                group_name="actor_rollout",
            )
            # NOTE(HanlinDu): collective init not finished before broadcast, so we init here to avoid potential issues
            # may not be necessary for all cases, but safer to have it
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                n_workers,
            )
            ray.get(
                self.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                )
            )

    def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        self.async_rollout_mode = True

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = OneStepOffAgentLoopManager(
            config=self.config, worker_group=self.rollout_wg, rm_resource_pool=rm_resource_pool
        )

    def _resolve_role_pool(self, role: Role, pool_spec: dict | None):
        if not pool_spec:
            return None

        mode = pool_spec.get("mode", "pool")
        if mode == "pool":
            # Directly pick a named pool (or the role default if name is omitted).
            pool_name = pool_spec.get("name")
            if pool_name is None:
                return self.resource_pool_manager.get_resource_pool(role)
            return self.resource_pool_manager.resource_pool_dict[pool_name]

        if mode == "split":
            # Split a base pool into sub-pools and select one by index.
            split_size = pool_spec.get("size")
            split_index = pool_spec.get("index", 0)
            # NOTE: split only partitions the base pool's existing world_size.
            # To switch from 4/4 -> 6/2, base_pool must be an 8-GPU pool
            # (e.g., shared_pool). Splitting a 4-GPU pool can never yield 6.
            base_pool_name = pool_spec.get("from_pool")
            if base_pool_name is not None:
                base_pool = self.resource_pool_manager.resource_pool_dict[base_pool_name]
            else:
                base_role = pool_spec.get("from_role", role)
                if isinstance(base_role, str):
                    base_role = Role[base_role]
                base_pool = self.resource_pool_manager.get_resource_pool(base_role)
            assert not (base_pool_name == "shared_pool" and isinstance(split_size, list)) or sum(split_size) == base_pool.world_size, (
                f"shared_pool world_size {base_pool.world_size} != sum(split_plan) {sum(split_size)}"
            )
            sub_pools = split_resource_pool(base_pool, split_size)
            if split_index >= len(sub_pools):
                raise IndexError(f"split index {split_index} out of range for {len(sub_pools)} sub-pools")
            return sub_pools[split_index]

        raise ValueError(f"Unsupported pool spec mode: {mode}")

    def _maybe_dynamic_resize(self):
        # Execute scheduled resize steps driven by config.trainer.dynamic_resize.
        # Accept both of the config shapes currently seen in experiments:
        #   1) a list of schedule items
        #   2) a named mapping such as {stage0: {...}, stage1: {...}}
        # The latter is convenient in YAML, but iterating over it directly
        # yields string keys, so normalize it before processing.
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", True):
            return
        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())
        elif not isinstance(schedule, list):
            raise TypeError(f"trainer.dynamic_resize.schedule must be a list or dict, got {type(schedule).__name__}")

        # FIXME(HanlinDu): this looped check seems heavy and unnecessary to do at every step
        for item in schedule:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each trainer.dynamic_resize.schedule item must be a dict, "
                    f"got {type(item).__name__}: {item!r}"
                )
            if item.get("step") != self.global_steps:
                continue

            actor_spec = item.get("actor_pool")
            rollout_spec = item.get("rollout_pool")
            release_old = item.get("release_old", True)
            detached = item.get("detached", False)
            actor_resume_path = item.get("actor_resume_from_path")

            actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
            rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

            new_actor = self.add_role_group(
                Role.Actor,
                name=item.get("actor_group_name"),
                resource_pool=actor_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
            )
            new_rollout = self.add_role_group(
                Role.Rollout,
                name=item.get("rollout_group_name"),
                resource_pool=rollout_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
            )
            self._switch_actor_rollout_groups(
                new_actor,
                new_rollout,
                resume_from_path=actor_resume_path,
                release_old=release_old,
            )

    def _switch_actor_rollout_groups(
        self,
        new_actor_wg: RayWorkerGroup,
        new_rollout_wg: RayWorkerGroup,
        *,
        resume_from_path: str | None = None,
        release_old: bool = False,
    ) -> None:
        # Dynamic resize must switch actor and rollout as one pair. Updating them
        # one by one temporarily creates a mixed topology (new actor + old rollout
        # or the reverse), and that intermediate state can hang weight sync /
        # collective group setup. Stage both groups first, then publish them
        # together and rebuild the dependent managers once.
        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg

        if resume_from_path is not None:
            new_actor_wg.load_checkpoint(
                resume_from_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        else:
            weights_info = old_actor_wg.get_actor_weights_info()[0]
            new_actor_wg.set_actor_weights_info(weights_info)

        weights_info = new_actor_wg.get_actor_weights_info()[0]
        new_rollout_wg.set_actor_weights_info(weights_info)

        self.actor_wg = new_actor_wg
        self.rollout_wg = new_rollout_wg
        self.actor_rollout_wg = new_actor_wg

        # Rollout manager and sync group both depend on the final actor/rollout pair,
        # so rebuild them only after both references are updated.
        self._init_async_rollout_manager()
        self._create_weight_sync_group()

        if release_old:
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)


    def _post_load_checkpoint_for_switch(self) -> None:
        self.sync_rollout_weights()
        if self.async_rollout_manager is not None:
            asyncio.run(self.async_rollout_manager.clear_kv_cache())

    def sync_rollout_weights(self):
        actor_workers = getattr(self.actor_wg, "workers", [])
        rollout_workers = getattr(self.rollout_wg, "workers", [])

        # Safety guard: actor and rollout must NOT share the same underlying Ray Actor.
        # Otherwise, broadcast will deadlock (rollout enters first and blocks the actor call).
        try:
            actor_ids = {getattr(w, "_actor_id", None) for w in actor_workers}
            rollout_ids = {getattr(w, "_actor_id", None) for w in rollout_workers}
            actor_ids.discard(None)
            rollout_ids.discard(None)
            overlap = actor_ids.intersection(rollout_ids)
        except Exception:  # pragma: no cover - best effort
            overlap = set()

        if overlap:
            raise RuntimeError(
                "Actor/Rollout worker groups overlap on the same Ray ActorID(s): "
                f"{list(overlap)[:6]} (showing up to 6). "
                "This will deadlock sync_rollout_weights. "
                "Please create actor/rollout as separate Ray Actors (no shared spawn/from_detached)."
            )

        rollout_refs = self.rollout_wg.sync_rollout_weights()
        actor_refs = self.actor_wg.sync_rollout_weights()
        if rollout_refs is None and actor_refs is None:
            return
        refs = []
        if rollout_refs is not None:
            refs.extend(rollout_refs if isinstance(rollout_refs, list) else [rollout_refs])
        if actor_refs is not None:
            refs.extend(actor_refs if isinstance(actor_refs, list) else [actor_refs])
        if refs:
            ray.get(refs)

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # The async agent/reward loop reuses non-tensor fields from the original
        # training batch. Some one-step-off datasets only provide prompt/answer
        # style fields and omit the reward-routing metadata expected by the
        # downstream reward loop worker. Patch the minimal fields here so the
        # compatibility fix stays local to the experimental trainer.
        self._maybe_patch_reward_metadata(batch.non_tensor_batch)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences_async(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Launch individual reward computations as each generation completes
        future_reward = None
        if self.config.reward_model.launch_reward_fn_async:
            # Store the object reference and set up callback
            future_reward = self._launch_individual_rewards.remote(batch, self.config, self.tokenizer)

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        # Get generation results
        gen_batch_result = batch
        original_non_tensor_batch = batch.non_tensor_batch

        # Repeat non_tensor_batch to match the number of responses
        n = config.actor_rollout_ref.rollout.n
        repeated_non_tensor_batch = {}
        for key, value in original_non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(value, n, axis=0)

        # Split into individual responses with preserved non_tensor_batch
        responses_split = []
        for i in range(len(gen_batch_result)):
            response_data = gen_batch_result[i : i + 1]  # Get single response
            # Add repeated non_tensor_batch values
            for key in repeated_non_tensor_batch:
                response_data.non_tensor_batch[key] = repeated_non_tensor_batch[key][i : i + 1]
            responses_split.append(response_data)

        # Launch async reward computation
        reward_futures = [
            compute_reward_async.remote(response_data, config, tokenizer) for response_data in responses_split
        ]

        # Wait for results and combine
        results = ray.get(reward_futures)
        rewards_list = [r[0] for r in results]
        extras_list = [r[1] for r in results]

        combined_reward_tensor = torch.cat(rewards_list, dim=0)
        combined_extras_dict = {}
        if extras_list and extras_list[0]:
            for key in extras_list[0].keys():
                combined_extras_dict[key] = [d[key] for d in extras_list if key in d]

        return combined_reward_tensor, combined_extras_dict

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # after load checkpoint sync rollout weights
        self.sync_rollout_weights()
        await self.async_rollout_manager.clear_kv_cache()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()

        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))

        while batch_data_future is not None:
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            if do_profile:
                self.actor_wg.start_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            with marked_timer("step", timing_raw):
                # wait for the previous batch
                with marked_timer("gen", timing_raw, color="red"):
                    _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
                    timing_raw.update(batch.meta_info["timing"])
                    timing_raw.update(_timing_raw)
                    metrics.update(_metrics)
                    batch.meta_info.pop("timing", None)

                # sync weights from actor to rollout
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    self.sync_rollout_weights()
                    await self.async_rollout_manager.clear_kv_cache()

                # async next generation
                if not is_last_step:
                    batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                    await asyncio.sleep(0)

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(
                            data=batch, config=self.config, tokenizer=self.tokenizer
                        )
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # await asyncio.sleep(0) ensures:
                # Asynchronous tasks can start executing immediately
                # The event loop can handle other pending coroutines
                # Prevents computations in a certain phase from blocking the entire asynchronous workflow
                #
                # The purpose here is to ensure that after triggering
                # `self.async_rollout_manager.generate_sequences_async(gen_batch_output)`,
                # the subsequent relevant logic can proceed in a timely manner
                await asyncio.sleep(0)

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:  # Recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                await asyncio.sleep(0)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                await asyncio.sleep(0)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                await asyncio.sleep(0)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Compute rollout correction: IS weights, rejection sampling, and metrics
                    # Only runs in decoupled mode (computes once per batch using stable π_old)
                    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs  # Only in decoupled mode
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        # Compute IS weights, apply rejection sampling, compute metrics
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                await asyncio.sleep(0)

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                await asyncio.sleep(0)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        rollout_config = self.config.actor_rollout_ref.rollout
                        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                        # TODO: Make "temperature" single source of truth from generation.
                        batch.meta_info["temperature"] = rollout_config.temperature
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                await asyncio.sleep(0)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            await asyncio.sleep(0)
            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
            await asyncio.sleep(0)

            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            # Check if the conditions for saving a checkpoint are met.
            # The conditions include a mandatory condition (1) and
            # one of the following optional conditions (2/3/4):
            # 1. The save frequency is set to a positive value.
            # 2. It's the last training step.
            # 3. The current step number is a multiple of the save frequency.
            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
            ):
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

            # dynamic resize of actor/rollout pool
            # NOTE(HanlinDu): this may be executed asynchronously with _save_checkpoint()
            self._maybe_dynamic_resize()

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.actor_rollout_ref.actor, "profiler")
                and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_rollout_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            if hasattr(self.train_dataset, "on_batch_end"):
                # The dataset may be changed after each training batch
                self.train_dataset.on_batch_end(batch=batch)
