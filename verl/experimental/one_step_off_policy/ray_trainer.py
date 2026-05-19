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
import inspect
import logging
import math
import os
import shutil
import socket
import time
import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.experimental.state.api import get_actor as ray_get_actor_state
from ray.util.collective import collective
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.one_step_off_policy.communicator_cache import (
    CommunicatorCacheConfig,
    WeightSyncCommunicatorCache,
    build_topology_key,
)
from verl.experimental.one_step_off_policy.resize_budget import (
    ResizeBudgetConfig,
    ResizeBudgetController,
    ResizeBudgetSnapshot,
)
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.experimental.one_step_off_policy.resize_controller import (
    ACTION_EXPAND_ROLLOUT,
    ACTION_EXPAND_TRAIN,
    ACTION_HOLD,
    ACTION_TO_CODE,
    ResizeController,
    ResizeControllerConfig,
)
from verl.experimental.one_step_off_policy.resize_metrics import build_resize_observation
from verl.experimental.one_step_off_policy.staging_backend import HostStagingConfig, has_restore_session_manifest, read_restore_session_manifest
from verl.experimental.one_step_off_policy.trace_utils import build_resize_trace_config, resize_trace_span
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
    calculate_workload,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils import omega_conf_to_dataclass
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffRayTrainer(RayPPOTrainer):
    # Staged resize execution model for shared-pool in-place rebalance:
    #
    # Layer 1: static prepare
    #   - parse resize spec / split pools / build plans
    #   - do not consume new shared-pool worker slots yet
    #
    # Layer 2: resource transition
    #   - release the shrinking side first (e.g. old rollout 4 -> 2 releases 2 slots)
    #   - create/rebuild the new actor on the freed slots

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for one-step-off worker groups.

        Keep rollout profiling separate from actor profiling and avoid passing the
        PPO trainer's `role` kwarg into torch profiler implementations.
        """
        if do_profile:
            self.actor_wg.start_profile(profile_step=self.global_steps)
            if not self.hybrid_engine:
                self.rollout_wg.start_profile(profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for one-step-off worker groups."""
        if do_profile:
            self.actor_wg.stop_profile()
            if not self.hybrid_engine:
                self.rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()
    #   - release the other old side if needed, then create the new rollout
    #
    # Layer 3: final commit/publish
    #   - commit new worker/model init
    #   - restore/copy weights
    #   - publish new actor/rollout pair and rebuild dependent managers/groups
    #
    # For shared-pool in-place rebalance such as 4+4 -> 6+2 or 6+2 -> 2+6 on 8 GPUs,
    # trying to build the full new topology before shrinking the old one can
    # over-allocate the pool and leave new workers stuck in PENDING_CREATION.
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def _trace_step(self) -> int:
        return int(getattr(self, "global_steps", 0))

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
        self._active_role_world_sizes: dict[Role, int] = {}
        self._weight_sync_group_version = 0
        self._active_weight_sync_group_name = "actor_rollout"
        self._pending_weight_sync_group_cleanup: list[tuple[str, RayWorkerGroup, RayWorkerGroup]] = []
        self._last_runtime_batch_plan: dict[str, int] | None = None
        self._resize_padding_history: list[DataProto] = []
        self._resize_padding_history_size = 8
        self._active_topology_specs: dict[Role, dict | None] = {Role.Actor: None, Role.Rollout: None}
        # Round one: schedule is still the source of candidate topologies.
        # The controller only decides whether a scheduled switch should fire.
        self._dynamic_resize_mode = self._resolve_dynamic_resize_mode()
        self._resize_controller = self._build_resize_controller()
        self._latest_resize_control_metrics: dict[str, float | str | int] = self._default_resize_control_metrics()
        self._resize_trace_config = build_resize_trace_config(self.config)
        self._dynamic_resize_enabled = self._build_dynamic_resize_flag("enable", default=False)
        self._dynamic_resize_phased_init_enabled = self._build_dynamic_resize_flag("phased_init.enable", default=True)
        self._dynamic_resize_async_comm_prewarm_enabled = self._build_dynamic_resize_flag(
            "async_comm_prewarm.enable", default=True
        )
        # Round two: handoff is configured through a staging backend abstraction
        # so the staged resize path can evolve from disk-backed transfer toward
        # richer host-memory implementations without changing the trainer flow.
        self._handoff_staging_config = self._build_handoff_staging_config()
        self._latest_resize_execution_metrics: dict[str, float | str | int] = self._default_resize_execution_metrics()
        self._communicator_cache_config = self._build_communicator_cache_config()
        self._weight_sync_communicator_cache = WeightSyncCommunicatorCache(self._communicator_cache_config)
        self._latest_communicator_cache_metrics: dict[str, float | str | int] = self._default_communicator_cache_metrics()
        self._resize_budget_config = self._build_resize_budget_config()
        self._resize_budget_controller = ResizeBudgetController(self._resize_budget_config)
        self._latest_resize_budget_metrics: dict[str, float | str | int] = self._default_resize_budget_metrics()

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

    def _resolve_dynamic_resize_mode(self) -> str:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return "schedule"
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return str(cfg.get("mode", "schedule"))

    def _build_resize_controller(self) -> ResizeController | None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return None
        cfg = OmegaConf.to_container(cfg, resolve=True)
        mode = str(cfg.get("mode", "schedule"))
        if mode != "schedule_with_hysteresis":
            return None
        controller_cfg = ResizeControllerConfig.from_dict(cfg.get("hysteresis", {}))
        if not controller_cfg.enable:
            return None
        return ResizeController(controller_cfg)

    def _build_handoff_staging_config(self) -> HostStagingConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return HostStagingConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return HostStagingConfig.from_dict(cfg.get("handoff", {}))

    def _build_communicator_cache_config(self) -> CommunicatorCacheConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return CommunicatorCacheConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return CommunicatorCacheConfig.from_dict(cfg.get("communicator_cache", {}))

    def _build_dynamic_resize_flag(self, key: str, *, default: bool) -> bool:
        value = OmegaConf.select(self.config.trainer, f"dynamic_resize.{key}")
        if value is None:
            return default
        return bool(value)

    def _build_resize_budget_config(self) -> ResizeBudgetConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return ResizeBudgetConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return ResizeBudgetConfig.from_dict(cfg.get("budget_protection", {}))

    def _build_step_path_debug_metrics(self, timing_raw: dict[str, float]) -> dict[str, float | int]:
        rollout_background_s = float(timing_raw.get("generate_async", 0.0) or 0.0)
        rollout_wait_exposed_s = float(timing_raw.get("gen", 0.0) or 0.0)
        sync_s = float(timing_raw.get("sync_rollout_weights", 0.0) or 0.0)
        reward_s = float(timing_raw.get("reward", 0.0) or 0.0)
        adv_s = float(timing_raw.get("adv", 0.0) or 0.0)
        old_log_prob_s = float(timing_raw.get("old_log_prob", 0.0) or 0.0)
        ref_s = float(timing_raw.get(str(Role.RefPolicy), 0.0) or 0.0)
        values_s = float(timing_raw.get("values", 0.0) or 0.0)
        update_actor_s = float(timing_raw.get("update_actor", 0.0) or 0.0)
        update_critic_s = float(timing_raw.get("update_critic", 0.0) or 0.0)
        step_s = float(timing_raw.get("step", 0.0) or 0.0)

        train_update_s = update_actor_s + update_critic_s
        train_post_wait_s = sync_s + reward_s + adv_s + old_log_prob_s + ref_s + values_s + train_update_s
        rollout_hidden_by_overlap_s = max(rollout_background_s - rollout_wait_exposed_s, 0.0)

        # For async off-policy execution, rollout and train happen on different ranks and
        # overlap in wall-clock time. A per-step system time should therefore be compared
        # against the longest rank path rather than a train+rollout sum.
        step_driver_path_s = rollout_wait_exposed_s + train_post_wait_s
        step_reconstructed_s = max(rollout_background_s, train_post_wait_s)

        actor_world_size = len(getattr(getattr(self, "actor_wg", None), "workers", []) or [])
        rollout_world_size = len(getattr(getattr(self, "rollout_wg", None), "workers", []) or [])
        agent_loop_workers = len(getattr(getattr(self, "async_rollout_manager", None), "agent_loop_workers", []) or [])

        return {
            "debug/actor_world_size": actor_world_size,
            "debug/rollout_world_size": rollout_world_size,
            "debug/agent_loop_worker_count": agent_loop_workers,
            "debug/timing_rollout_background_total_s": rollout_background_s,
            "debug/timing_rollout_wait_exposed_s": rollout_wait_exposed_s,
            "debug/timing_rollout_hidden_by_overlap_s": rollout_hidden_by_overlap_s,
            "debug/timing_train_update_total_s": train_update_s,
            "debug/timing_train_post_wait_total_s": train_post_wait_s,
            "debug/timing_step_driver_path_s": step_driver_path_s,
            "debug/timing_step_reconstructed_s": step_reconstructed_s,
            "debug/timing_step_reconstruction_error_s": step_s - step_reconstructed_s,
        }

    def _default_resize_control_metrics(self) -> dict[str, float | str | int]:
        enabled = 1.0 if self._resize_controller is not None else 0.0
        return {
            "resize/mode": self._dynamic_resize_mode,
            "resize/controller_enabled": enabled,
            "resize/rollout_train_ratio": 0.0,
            "resize/hysteresis_signal": ACTION_HOLD,
            "resize/hysteresis_signal_code": 0.0,
            "resize/hysteresis_decision": ACTION_HOLD,
            "resize/hysteresis_decision_code": 0.0,
            "resize/required_action": ACTION_HOLD,
            "resize/required_action_code": 0.0,
            "resize/window_fill": 0,
            "resize/dwell_remaining": 0,
            "resize/cooldown_remaining": 0,
            "resize/gate_pass": -1.0,
        }

    def _default_resize_execution_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/host_stage_enabled": 1.0 if self._handoff_staging_config.enable else 0.0,
            "resize/phased_init_enabled": 1.0 if self._dynamic_resize_phased_init_enabled else 0.0,
            "resize/host_stage_backend": self._handoff_staging_config.effective_backend(),
            "resize/host_stage_requested_backend": self._handoff_staging_config.backend,
            "resize/host_stage_export_s": 0.0,
            "resize/host_stage_export_initial_barrier_s": 0.0,
            "resize/host_stage_export_load_model_to_gpu_s": 0.0,
            "resize/host_stage_export_load_optimizer_to_gpu_s": 0.0,
            "resize/host_stage_export_sanitize_optimizer_s": 0.0,
            "resize/host_stage_export_model_state_dict_s": 0.0,
            "resize/host_stage_export_optimizer_state_dict_s": 0.0,
            "resize/host_stage_export_build_model_pages_s": 0.0,
            "resize/host_stage_export_build_optimizer_pages_s": 0.0,
            "resize/host_stage_export_create_session_s": 0.0,
            "resize/host_stage_export_stage_model_pages_s": 0.0,
            "resize/host_stage_export_stage_optimizer_pages_s": 0.0,
            "resize/host_stage_export_host_put_s": 0.0,
            "resize/host_stage_export_host_file_write_s": 0.0,
            "resize/host_stage_export_disk_spill_s": 0.0,
            "resize/host_stage_export_extra_state_s": 0.0,
            "resize/host_stage_export_manifest_update_s": 0.0,
            "resize/host_stage_export_final_barrier_s": 0.0,
            "resize/host_stage_export_offload_s": 0.0,
            "resize/host_stage_export_model_page_count": 0.0,
            "resize/host_stage_export_optimizer_page_count": 0.0,
            "resize/host_stage_export_model_state_host_pages": 0.0,
            "resize/host_stage_export_model_state_disk_pages": 0.0,
            "resize/host_stage_export_model_state_host_bytes": 0.0,
            "resize/host_stage_export_model_state_disk_bytes": 0.0,
            "resize/host_stage_export_optim_state_host_pages": 0.0,
            "resize/host_stage_export_optim_state_disk_pages": 0.0,
            "resize/host_stage_export_optim_state_host_bytes": 0.0,
            "resize/host_stage_export_optim_state_disk_bytes": 0.0,
            "resize/host_stage_import_s": 0.0,
            "resize/host_stage_import_wait_s": 0.0,
            "resize/post_switch_weight_sync_s": 0.0,
            "resize/post_switch_kv_clear_s": 0.0,
            "resize/progressive_swap_s": 0.0,
            "resize/kv_cache_preclear_s": 0.0,
            "resize/host_stage_cleanup": 0.0,
            "resize/optimizer_deferred_restore": 0.0,
            "resize/optimizer_pending_pages": 0.0,
            "resize/optimizer_materialize_s": 0.0,
            "resize/optimizer_load_pages_s": 0.0,
            "resize/optimizer_set_state_dict_s": 0.0,
            "resize/optimizer_streaming_restore": 0.0,
            "resize/optimizer_streamed_pages": 0.0,
            "resize/optimizer_full_state_restore": 0.0,
            "resize/optimizer_materialize_count": 0.0,
            "resize/model_restore_device_sync_s": 0.0,
            "resize/model_device_preload_pages": 0.0,
            "resize/model_device_preload_bytes": 0.0,
            "resize/model_device_preload_max_pending_pages": 0.0,
            "resize/model_device_preload_max_pending_bytes": 0.0,
            "resize/optimizer_device_preload_pages": 0.0,
            "resize/optimizer_device_preload_bytes": 0.0,
            "resize/optimizer_device_preload_max_pending_pages": 0.0,
            "resize/optimizer_device_preload_max_pending_bytes": 0.0,
            "resize/actor_prepare_worker_init_s": 0.0,
            "resize/actor_commit_worker_init_s": 0.0,
            "resize/actor_prepare_model_init_s": 0.0,
            "resize/actor_commit_model_init_s": 0.0,
            "resize/rollout_prepare_worker_init_s": 0.0,
            "resize/rollout_commit_worker_init_s": 0.0,
            "resize/rollout_prepare_model_init_s": 0.0,
            "resize/rollout_commit_model_init_s": 0.0,
            "resize/restore_failed": 0.0,
            "resize/partial_restore_cleanup_count": 0.0,
        }

    def _default_communicator_cache_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/comm_cache_enabled": 1.0 if self._communicator_cache_config.enable else 0.0,
            "resize/async_comm_prewarm_enabled": 1.0 if self._dynamic_resize_async_comm_prewarm_enabled else 0.0,
            "resize/comm_cache_hit": 0.0,
            "resize/comm_cache_miss": 0.0,
            "resize/comm_live_cache_hit": 0.0,
            "resize/comm_registry_hit": 0.0,
            "resize/comm_registry_miss": 0.0,
            "resize/comm_prewarm_ready": 0.0,
            "resize/comm_prewarm_create_s": 0.0,
            "resize/comm_prewarm_warmup_s": 0.0,
            "resize/comm_prewarm_warmup_broadcast_s_max": 0.0,
            "resize/comm_prewarm_warmup_broadcast_s_mean": 0.0,
            "resize/comm_prewarm_warmup_total_s_max": 0.0,
            "resize/comm_prewarm_warmup_total_s_mean": 0.0,
            "resize/comm_activate_s": 0.0,
            "resize/comm_full_build_s": 0.0,
            "resize/comm_prepare_path": "",
            "resize/comm_cache_reused_group": "",
        }

    def _default_resize_budget_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/budget_enabled": 1.0 if self._resize_budget_config.enable else 0.0,
            "resize/budget_ratio": self._resize_budget_config.memory_budget_ratio,
            "resize/budget_blocked": 0.0,
            "resize/budget_reason": "",
            "resize/budget_effective_backend": self._handoff_staging_config.effective_backend(),
        }

    def _reset_resize_execution_metrics(self) -> dict[str, float | str | int]:
        self._latest_resize_execution_metrics = self._default_resize_execution_metrics()
        return dict(self._latest_resize_execution_metrics)

    def _reset_communicator_cache_metrics(self) -> dict[str, float | str | int]:
        self._latest_communicator_cache_metrics = self._default_communicator_cache_metrics()
        return dict(self._latest_communicator_cache_metrics)

    def _reset_resize_budget_metrics(self) -> dict[str, float | str | int]:
        self._latest_resize_budget_metrics = self._default_resize_budget_metrics()
        return dict(self._latest_resize_budget_metrics)

    def _update_resize_execution_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_resize_execution_metrics.update(kwargs)
        return dict(self._latest_resize_execution_metrics)

    def _update_communicator_cache_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_communicator_cache_metrics.update(kwargs)
        return dict(self._latest_communicator_cache_metrics)

    def _update_resize_budget_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_resize_budget_metrics.update(kwargs)
        return dict(self._latest_resize_budget_metrics)

    def _start_deferred_optimizer_materialize(self, actor_wg: RayWorkerGroup) -> None:
        if getattr(self, "_pending_optimizer_materialize_refs", None):
            return
        try:
            self._pending_optimizer_materialize_refs = actor_wg.execute_all_async(f"{str(Role.Actor)}_materialize_pending_optimizer_restore")
            logger.info("[one-step-off][resize][host-stage] started async optimizer materialize on new actor group")
        except Exception as exc:  # pragma: no cover - runtime safety
            self._pending_optimizer_materialize_refs = None
            logger.warning("[one-step-off][resize][host-stage] failed to start async optimizer materialize: %r", exc)

    def _collect_deferred_optimizer_materialize_metrics(self, *, block: bool = False) -> dict[str, float | int]:
        refs = getattr(self, "_pending_optimizer_materialize_refs", None)
        if not refs:
            return {}
        timeout = None if block else 0
        ready, remaining = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        if remaining:
            return {}
        self._pending_optimizer_materialize_refs = None
        results = ray.get(ready)
        merged: dict[str, float | int] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    merged[key] = max(float(merged.get(key, 0.0)), float(value))
        return merged

    @staticmethod
    def _merge_resize_numeric_metrics(results) -> dict[str, float]:
        merged: dict[str, float] = {}
        for item in results or []:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key.startswith("resize/") and isinstance(value, (int, float)):
                    merged[key] = max(float(merged.get(key, 0.0)), float(value))
        return merged

    def _start_async_actor_host_restore(self, actor_wg: RayWorkerGroup, actor_resume_path: str, runtime_staging_cfg: dict):
        refs = actor_wg.execute_all_async(
            f"{str(Role.Actor)}_load_actor_handoff_state_from_host",
            actor_resume_path,
            staging_config=runtime_staging_cfg,
        )
        return {"refs": refs, "started_at": time.monotonic()}

    def _start_async_actor_optimizer_host_export(
        self, actor_wg: RayWorkerGroup, actor_resume_path: str, runtime_staging_cfg: dict
    ):
        refs = actor_wg.execute_all_async(
            f"{str(Role.Actor)}_stage_actor_optimizer_state_to_host",
            actor_resume_path,
            staging_config=runtime_staging_cfg,
        )
        return {"refs": refs, "started_at": time.monotonic()}

    def _finish_async_actor_optimizer_host_export(self, export_task) -> dict[str, float]:
        if export_task is None:
            return {}
        wait_started_at = time.monotonic()
        results = ray.get(export_task["refs"])
        wait_duration = time.monotonic() - wait_started_at
        total_duration = time.monotonic() - float(export_task["started_at"])
        metrics = self._merge_resize_numeric_metrics(results)
        metrics["resize/host_stage_async_optimizer_export_wait_s"] = float(wait_duration)
        metrics["resize/host_stage_async_optimizer_export_total_s"] = float(total_duration)
        return metrics

    def _finish_async_actor_host_restore(
        self,
        restore_task,
        *,
        actor_wg: RayWorkerGroup,
        actor_resume_path: str,
        runtime_staging_cfg: dict,
    ) -> dict[str, float]:
        if restore_task is None:
            return {}
        wait_started_at = time.monotonic()
        try:
            restore_results = ray.get(restore_task["refs"])
        except Exception as exc:
            self._update_resize_execution_metrics(
                **{
                    "resize/restore_failed": 1.0,
                    "resize/partial_restore_cleanup_count": 1.0,
                }
            )
            logger.exception(
                "[one-step-off][resize][host-stage] async restore failed: path=%s "
                "resize/restore_failed=1.0 resize/partial_restore_cleanup_count=1.0 error=%r",
                actor_resume_path,
                exc,
            )
            try:
                actor_wg.cleanup_actor_handoff_restore_session(actor_resume_path)
            finally:
                raise
        wait_duration = time.monotonic() - wait_started_at
        import_duration = time.monotonic() - float(restore_task["started_at"])
        restore_preload_metrics = self._merge_resize_numeric_metrics(restore_results)
        metrics = {
            "resize/host_stage_import_s": import_duration,
            "resize/host_stage_import_wait_s": wait_duration,
            "resize/progressive_swap_s": import_duration if runtime_staging_cfg.get("progressive_swap", False) else 0.0,
            **restore_preload_metrics,
        }
        self._update_resize_execution_metrics(**metrics)
        return metrics

    @staticmethod
    def _extract_max_metric_value(results, key: str) -> float:
        values: list[float] = []

        def _collect(item) -> None:
            if item is None:
                return
            if isinstance(item, list):
                for sub in item:
                    _collect(sub)
                return
            if isinstance(item, dict) and key in item:
                try:
                    values.append(float(item[key]))
                except (TypeError, ValueError):
                    return

        _collect(results)
        return max(values) if values else 0.0

    @staticmethod
    def _estimate_weights_info_bytes(weights_info) -> int:
        total_bytes = 0
        for _, shape, dtype in weights_info:
            element_size = torch.empty((), dtype=dtype).element_size()
            total_bytes += math.prod(shape) * element_size
        return int(total_bytes)

    @staticmethod
    def _estimate_host_export_peak_bytes(model_bytes: int, *, stage_optimizer: bool) -> int:
        # Conservative heuristic: full model state plus CPU-side optimizer state.
        multiplier = 3.0 if stage_optimizer else 1.25
        return int(model_bytes * multiplier)

    @staticmethod
    def _estimate_gpu_restore_peak_bytes(
        model_bytes: int,
        *,
        progressive_swap: bool = False,
        chunk_mb: int | None = None,
    ) -> int:
        # When paged restore is enabled, the extra transient GPU pressure is
        # dominated by the current page rather than a second full model copy.
        working_bytes = model_bytes
        if progressive_swap and chunk_mb is not None and chunk_mb > 0:
            working_bytes = min(model_bytes, int(chunk_mb) * 1024 * 1024)
        return int(working_bytes * 1.2)

    def _make_runtime_handoff_staging_dict(self, *, effective_backend: str) -> dict[str, float | str | bool | int]:
        runtime_cfg = self._handoff_staging_config.to_manifest_dict()
        runtime_cfg["backend"] = effective_backend
        return runtime_cfg

    def _get_role_group_resource_snapshot(self, role: Role, role_wg: RayWorkerGroup, staging_path: str | None = None) -> ResizeBudgetSnapshot:
        # Query one representative worker before resize so the trainer can make
        # a budget decision without pushing large tensors through the switch path.
        snapshot = role_wg.execute_rank_zero_sync(f"{str(role)}_get_resize_resource_snapshot", staging_path)
        return ResizeBudgetSnapshot.from_dict(snapshot)

    def _current_topology_key(self) -> str:
        return build_topology_key(
            actor_spec=self._active_topology_specs.get(Role.Actor),
            rollout_spec=self._active_topology_specs.get(Role.Rollout),
        )

    def _candidate_topology_key(self, *, actor_spec: dict | None, rollout_spec: dict | None) -> str:
        return build_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)

    def _warmup_communicator_cache_keys(self) -> None:
        if not self._communicator_cache_config.enable or not self._communicator_cache_config.reserve_schedule_topologies:
            return

        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        schedule = []
        if cfg is not None:
            cfg = OmegaConf.to_container(cfg, resolve=True)
            schedule = cfg.get("schedule", []) or []
            if isinstance(schedule, dict):
                schedule = list(schedule.values())

        for item in schedule:
            if not isinstance(item, dict):
                continue
            topology_key = self._candidate_topology_key(
                actor_spec=item.get("actor_pool"),
                rollout_spec=item.get("rollout_pool"),
            )
            self._weight_sync_communicator_cache.register_topology(
                topology_key,
                actor_spec=item.get("actor_pool"),
                rollout_spec=item.get("rollout_pool"),
                actor_world_size=self._pool_world_size_from_spec(Role.Actor, item.get("actor_pool")),
                rollout_world_size=self._pool_world_size_from_spec(Role.Rollout, item.get("rollout_pool")),
            )
            self._weight_sync_communicator_cache.reserve(topology_key)

    def _build_resize_control_metrics(self, snapshot: dict[str, float | str | int] | None = None) -> dict[str, float | str | int]:
        snapshot = snapshot or {}
        base = self._default_resize_control_metrics()
        for key, value in snapshot.items():
            base[f"resize/{key}"] = value
        return base

    def _record_resize_control_observation(self, *, timing_raw: dict, batch: DataProto | None) -> dict[str, float | str | int]:
        if self._resize_controller is None:
            self._latest_resize_control_metrics = self._default_resize_control_metrics()
            return dict(self._latest_resize_control_metrics)

        observation = build_resize_observation(timing_raw=timing_raw, batch=batch)
        snapshot = self._resize_controller.observe(step=self.global_steps, observation=observation)
        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
        return dict(self._latest_resize_control_metrics)

    def _resolve_scheduled_resize_action(self, item: dict) -> str:
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        if actor_spec is None or rollout_spec is None:
            return ACTION_HOLD

        active_actor = self._active_role_world_sizes.get(Role.Actor)
        active_rollout = self._active_role_world_sizes.get(Role.Rollout)
        if active_actor is None or active_rollout is None:
            return ACTION_HOLD

        actor_target = self._pool_world_size_from_spec(Role.Actor, actor_spec)
        rollout_target = self._pool_world_size_from_spec(Role.Rollout, rollout_spec)
        if self._resize_controller is None:
            actor_delta = actor_target - active_actor
            rollout_delta = rollout_target - active_rollout
            if rollout_delta > 0 and actor_delta < 0:
                return ACTION_EXPAND_ROLLOUT
            if actor_delta > 0 and rollout_delta < 0:
                return ACTION_EXPAND_TRAIN
            return ACTION_HOLD

        return self._resize_controller.infer_required_action(
            active_actor=active_actor,
            active_rollout=active_rollout,
            actor_target=actor_target,
            rollout_target=rollout_target,
        )

    def _gate_dynamic_resize_schedule_item(self, item: dict) -> tuple[bool, str, dict[str, float | str | int]]:
        required_action = self._resolve_scheduled_resize_action(item)
        if self._dynamic_resize_mode != "schedule_with_hysteresis" or self._resize_controller is None:
            snapshot = {
                "required_action": required_action,
                "required_action_code": ACTION_TO_CODE[required_action],
                "gate_pass": 1.0,
            }
            metrics = self._build_resize_control_metrics(snapshot)
            self._latest_resize_control_metrics = metrics
            return True, required_action, metrics

        allow, snapshot = self._resize_controller.gate(step=self.global_steps, required_action=required_action)
        metrics = self._build_resize_control_metrics(snapshot)
        self._latest_resize_control_metrics = metrics
        return allow, required_action, metrics

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

        self.actor_rollout_wg = self.actor_wg
        try:
            with _patched_val_dataloader():
                return super()._validate()
        finally:
            self.actor_rollout_wg = self.actor_wg

    def _build_runtime_batch_plan(self, batch: DataProto) -> dict[str, int]:
        total_samples = len(batch)
        existing_plan = batch.meta_info.get("runtime_batch_plan") or {}
        base_total_samples = int(existing_plan.get("base_total_samples", total_samples))
        actor_world_size = max(self._active_role_world_sizes.get(Role.Actor, 1), 1)
        critic_world_size = max(self._active_role_world_sizes.get(Role.Critic, actor_world_size), 1)

        actor_micro = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", None)
        critic_micro = self.config.critic.get("ppo_micro_batch_size_per_gpu", None) if self.use_critic else None
        actor_micro = actor_micro if actor_micro is not None else max(base_total_samples // actor_world_size, 1)
        critic_micro = critic_micro if critic_micro is not None else max(base_total_samples // critic_world_size, 1)

        actor_alignment = actor_world_size * actor_micro
        critic_alignment = critic_world_size * critic_micro if self.use_critic else actor_alignment
        target_alignment = math.lcm(actor_alignment, critic_alignment)

        padded_total_samples = base_total_samples
        pad_samples = 0
        if base_total_samples % target_alignment != 0:
            padded_total_samples = math.ceil(base_total_samples / target_alignment) * target_alignment
            pad_samples = padded_total_samples - base_total_samples

        actor_runtime_mini = max(actor_alignment, padded_total_samples // actor_world_size)
        if actor_runtime_mini % actor_micro != 0:
            actor_runtime_mini = math.ceil(actor_runtime_mini / actor_micro) * actor_micro

        critic_runtime_mini = 0
        if self.use_critic:
            critic_runtime_mini = max(critic_alignment, padded_total_samples // critic_world_size)
            if critic_runtime_mini % critic_micro != 0:
                critic_runtime_mini = math.ceil(critic_runtime_mini / critic_micro) * critic_micro

        return {
            "base_total_samples": base_total_samples,
            "original_total_samples": total_samples,
            "padded_total_samples": padded_total_samples,
            "pad_samples": pad_samples,
            "net_pad_delta": padded_total_samples - total_samples,
            "target_alignment": target_alignment,
            "actor_world_size": actor_world_size,
            "actor_micro_batch_size_per_gpu": actor_micro,
            "actor_runtime_mini_batch_size": actor_runtime_mini,
            "critic_world_size": critic_world_size,
            "critic_micro_batch_size_per_gpu": critic_micro if self.use_critic else 0,
            "critic_runtime_mini_batch_size": critic_runtime_mini,
            "padding_source": "none",
            "global_step": self.global_steps,
        }

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each DP rank gets similar total tokens.

        This mirrors the shared PPO trainer implementation, except the actor DP
        size is queried from `actor_wg` directly because one-step-off keeps actor
        and rollout as separate groups instead of a composite `actor_rollout_wg`.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size(self.actor_wg, "actor")

        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _strip_resize_history_training_state(self, batch: DataProto) -> DataProto:
        tensor_exclude_keys = {
            "old_log_probs",
            "ref_log_prob",
            "values",
            "advantages",
            "returns",
            "token_level_rewards",
            "token_level_scores",
            "acc",
            "verifier_scores",
            "reward_baselines",
            "kl",
            "is_weights",
            "rollout_keep_mask",
        }
        tensor_keys = [key for key in batch.batch.keys() if key not in tensor_exclude_keys]
        return batch.select(batch_keys=tensor_keys, non_tensor_batch_keys=list(batch.non_tensor_batch.keys()), meta_info_keys=[])

    def _maybe_record_resize_history_batch(self, batch: DataProto) -> None:
        batch = self._strip_resize_history_training_state(batch)
        batch_size = len(batch)
        if batch_size <= 0:
            return
        snapshot_size = min(batch_size, 64)
        if batch_size == snapshot_size:
            snapshot = batch[:snapshot_size]
        else:
            sample_indices = np.random.choice(batch_size, size=snapshot_size, replace=False)
            snapshot = batch[sample_indices]
        snapshot = snapshot.select(meta_info_keys=[])

        self._resize_padding_history.append(snapshot)
        if len(self._resize_padding_history) > self._resize_padding_history_size:
            self._resize_padding_history = self._resize_padding_history[-self._resize_padding_history_size :]

    def _sample_padding_from_history(self, pad_samples: int) -> DataProto | None:
        if pad_samples <= 0 or not self._resize_padding_history:
            return None
        history_pool = DataProto.concat(self._resize_padding_history)
        if len(history_pool) <= 0:
            return None
        replace = len(history_pool) < pad_samples
        indices = np.random.choice(len(history_pool), size=pad_samples, replace=replace)
        padding_batch = history_pool[indices]
        if "uid" in padding_batch.non_tensor_batch:
            historical_uids = padding_batch.non_tensor_batch["uid"]
            padding_batch.non_tensor_batch["uid"] = np.array(
                [f"{uid}::histpad::step{self.global_steps}::{i}" for i, uid in enumerate(historical_uids)], dtype=object
            )
        return padding_batch

    def _apply_runtime_batch_plan(self, batch: DataProto) -> DataProto:
        existing_plan = batch.meta_info.get("runtime_batch_plan")
        if existing_plan is not None:
            self._last_runtime_batch_plan = existing_plan
            return batch

        plan = self._build_runtime_batch_plan(batch)
        self._last_runtime_batch_plan = plan

        if plan["net_pad_delta"] < 0:
            batch = batch[: plan["padded_total_samples"]]

            if "uid" in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"][: plan["padded_total_samples"]]

        if plan["net_pad_delta"] > 0:
            padding_batch = self._sample_padding_from_history(plan["net_pad_delta"])
            repeat_indices = None
            if padding_batch is None:
                repeat_indices = np.arange(plan["net_pad_delta"]) % plan["original_total_samples"]
                padding_batch = batch[repeat_indices.tolist()]
                plan["padding_source"] = "current"
            else:
                plan["padding_source"] = "history"
            print(
                "[one-step-off][padding] step=%s base=%s current=%s target=%s net_delta=%s source=%s history_snapshots=%s",
                self.global_steps,
                plan["base_total_samples"],
                plan["original_total_samples"],
                plan["padded_total_samples"],
                plan["net_pad_delta"],
                plan["padding_source"],
                len(self._resize_padding_history),
            )
            batch = DataProto.concat([batch, padding_batch])

            repeated_from = batch.non_tensor_batch.get("uid", None)
            if repeated_from is not None and repeat_indices is not None:
                original_uids = repeated_from[: plan["original_total_samples"]]
                padded_uids = original_uids[repeat_indices]
                padded_uids = np.array(
                    [f"{uid}::currpad::step{self.global_steps}::{i}" for i, uid in enumerate(padded_uids)], dtype=object
                )
                batch.non_tensor_batch["uid"] = np.concatenate([original_uids, padded_uids])

        plan["pad_samples"] = max(plan["net_pad_delta"], 0)

        batch.meta_info["runtime_batch_plan"] = plan
        batch.meta_info["mini_batch_size"] = plan["actor_runtime_mini_batch_size"]
        batch.meta_info["critic_mini_batch_size"] = plan["critic_runtime_mini_batch_size"]

        return batch

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

    def _get_resize_worker_group_kwargs(self) -> dict:
        wg_kwargs = self._get_worker_group_kwargs()

        # For SubRayResourcePool overlap-resize, scheduling an extra helper task
        # (`get_master_addr_port`) onto the reused placement group can block
        # because the old topology may still occupy the bundles. Reusing the old
        # worker group's rendezvous metadata is also unsafe because it couples the
        # new group to the old process group's endpoint. Instead, allocate a fresh
        # driver-side rendezvous endpoint and pass it explicitly to the new group.
        master_addr = None
        try:
            if getattr(self.actor_wg, "workers", None):
                master_addr = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
        except Exception as exc:
            logger.warning("[one-step-off][resize] failed to query current actor node ip for new rendezvous: %s", exc)

        if master_addr is None:
            master_addr = ray.util.get_node_ip_address().strip("[]")

        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        wg_kwargs["master_addr"] = str(master_addr)
        wg_kwargs["master_port"] = str(master_port)
        return wg_kwargs

    def _init_worker_groups(self):
        detached_cfg = self._get_detached_workers_cfg()
        initial_pool_overrides = {
            Role.Actor: self._resolve_initial_role_pool_override(Role.Actor),
            Role.Rollout: self._resolve_initial_role_pool_override(Role.Rollout),
        }

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
                    role = Role.from_string(role_name)
                    role_resource_pool = initial_pool_overrides.get(role) or resource_pool
                    worker_dict_cls = create_colocated_worker_cls(class_dict={role_name: role_cls})
                    wg = self.ray_worker_group_cls(
                        resource_pool=role_resource_pool,
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
        if self._dynamic_resize_enabled and self._dynamic_resize_phased_init_enabled:
            self._prepare_role_group_init(self.actor_wg, role=Role.Actor)
            self._prepare_role_group_init(self.rollout_wg, role=Role.Rollout)
            self._commit_role_group_init(self.actor_wg, role=Role.Actor)
            self._commit_role_group_init(self.rollout_wg, role=Role.Rollout)
        else:
            self.actor_wg.init_model()
            self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg
        # Register the initial groups as the default targets.
        self.role_groups[Role.Actor] = {"primary": self.actor_wg}
        self.role_groups[Role.Rollout] = {"primary": self.rollout_wg}
        self._initialize_active_topology_state()
        self._warmup_communicator_cache_keys()
        weights_info = self._get_actor_weights_info(self.actor_wg)[0]
        self._set_actor_weights_info(self.rollout_wg, weights_info)
        self._create_weight_sync_group()

    def _pool_world_size_from_spec(self, role: Role, pool_spec: dict | None) -> int:
        pool = self._resolve_role_pool(role, pool_spec)
        if pool is None:
            pool = self.resource_pool_manager.get_resource_pool(role)
        world_size = getattr(pool, "world_size", None)
        if world_size is None:
            raise ValueError(f"Cannot determine world_size for role={role}, pool_spec={pool_spec}, pool={pool}")
        return int(world_size)

    def _initialize_active_topology_state(self) -> None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is not None:
            cfg = OmegaConf.to_container(cfg, resolve=True)
            if not cfg.get("enable", False):
                self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, None)
                self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, None)
                return

        initial_item = self._get_initial_dynamic_resize_schedule_item()

        if initial_item is not None:
            actor_spec = initial_item.get("actor_pool")
            rollout_spec = initial_item.get("rollout_pool")
            self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)
            return

        self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, None)
        self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, None)

    def _publish_active_topology_state(self, *, actor_spec: dict | None, rollout_spec: dict | None) -> None:
        self._active_topology_specs[Role.Actor] = actor_spec
        self._active_topology_specs[Role.Rollout] = rollout_spec
        self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, actor_spec)
        self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, rollout_spec)

    def _get_initial_dynamic_resize_schedule_item(self) -> dict | None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return None
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", False):
            return None

        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())

        for item in schedule:
            if not isinstance(item, dict):
                continue
            if item.get("step", 0) < 0 and item.get("actor_pool") and item.get("rollout_pool"):
                return item
        return None

    def _resolve_initial_role_pool_override(self, role: Role):
        if role not in {Role.Actor, Role.Rollout}:
            return None

        initial_item = self._get_initial_dynamic_resize_schedule_item()
        if initial_item is None:
            return None

        pool_spec = initial_item.get("actor_pool") if role == Role.Actor else initial_item.get("rollout_pool")
        if not pool_spec:
            return None
        return self._resolve_role_pool(role, pool_spec)

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
            **self._get_resize_worker_group_kwargs(),
        )
        spawn_wg = wg_dict.spawn(prefix_set=[str(role)])
        role_wg = spawn_wg[str(role)]
        return role_wg

    def _prepare_role_group_init(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """尽量提前执行不依赖最终切换发布的初始化准备。

        对 one-step-off 实验 worker，优先调用显式 prepare 生命周期；
        对未实现该接口的旧 worker，静默回退，保持兼容。
        """
        if not self._dynamic_resize_phased_init_enabled:
            return
        role_prefix = str(role)
        for method_name in ("prepare_worker_init", "prepare_model_init"):
            remote_method_name = f"{role_prefix}_{method_name}"
            with resize_trace_span(
                self._resize_trace_config,
                f"{role_prefix}_{method_name}_group",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"role": role_prefix, "phase": "prepare"},
            ):
                rank0_result = role_wg.execute_rank_zero_sync(remote_method_name)
                all_results = role_wg.execute_all_sync(remote_method_name)
            duration = self._extract_max_metric_value([rank0_result, all_results], f"resize/{method_name}_s")
            if duration > 0:
                self._update_resize_execution_metrics(**{f"resize/{role_prefix}_{method_name}_s": duration})

    async def _prepare_role_group_init_async(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """异步 prepare 接口骨架。

        当前实现先复用同步版本，统一 trainer 侧接口，便于后续把 prepare
        真正下沉到后台任务/并发编排中。
        """
        result = self._prepare_role_group_init(role_wg, role=role)
        if inspect.isawaitable(result):
            await result

    def _commit_role_group_init(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """执行依赖最终 worker/runtime 状态的初始化提交。

        优先走新的显式 commit 生命周期；如果 worker 还未实现，回退到原始
        `init_model()` 入口，确保非实验路径与已有逻辑不受影响。
        """
        role_prefix = str(role)
        commit_worker_method = f"{role_prefix}_commit_worker_init"
        commit_model_method = f"{role_prefix}_commit_model_init"
        with resize_trace_span(
            self._resize_trace_config,
            f"{role_prefix}_commit_worker_init_group",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"role": role_prefix, "phase": "commit_worker"},
        ):
            worker_results = role_wg.execute_all_sync(commit_worker_method)
        with resize_trace_span(
            self._resize_trace_config,
            f"{role_prefix}_commit_model_init_group",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"role": role_prefix, "phase": "commit_model"},
        ):
            model_results = role_wg.execute_all_sync(commit_model_method)

        self._update_resize_execution_metrics(
            **{
                f"resize/{role_prefix}_prepare_worker_init_s": self._extract_max_metric_value(
                    worker_results, "resize/prepare_worker_init_s"
                ),
                f"resize/{role_prefix}_commit_worker_init_s": self._extract_max_metric_value(
                    worker_results, "resize/commit_worker_init_s"
                ),
                f"resize/{role_prefix}_prepare_model_init_s": self._extract_max_metric_value(
                    model_results, "resize/prepare_model_init_s"
                ),
                f"resize/{role_prefix}_commit_model_init_s": self._extract_max_metric_value(
                    model_results, "resize/commit_model_init_s"
                ),
            }
        )

    async def _commit_role_group_init_async(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """异步 commit 接口骨架。

        语义上该接口预留给“旧 worker 释放后再提交”的场景；当前阶段先同步执
        行，保证调用链已成型，后续可以把真正的异步调度填进来。
        """
        result = self._commit_role_group_init(role_wg, role=role)
        if inspect.isawaitable(result):
            await result

    def _get_actor_weights_info(self, actor_wg: RayWorkerGroup):
        return actor_wg.execute_all_sync(f"{str(Role.Actor)}_get_actor_weights_info")

    def _set_actor_weights_info(self, rollout_wg: RayWorkerGroup, weights_info) -> None:
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_set_actor_weights_info", weights_info)

    def _create_actor_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return actor_wg.execute_all_sync(
            f"{str(Role.Actor)}_create_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _create_rollout_weight_sync_group(
        self,
        rollout_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return rollout_wg.execute_all_sync(
            f"{str(Role.Rollout)}_create_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _prepare_actor_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return actor_wg.execute_all_sync(
            f"{str(Role.Actor)}_prepare_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _prepare_rollout_weight_sync_group(
        self,
        rollout_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return rollout_wg.execute_all_sync(
            f"{str(Role.Rollout)}_prepare_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _destroy_actor_weight_sync_group(self, actor_wg: RayWorkerGroup, group_name: str):
        return actor_wg.execute_all_sync(f"{str(Role.Actor)}_destroy_weight_sync_group", group_name)

    def _destroy_rollout_weight_sync_group(self, rollout_wg: RayWorkerGroup, group_name: str):
        return rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_destroy_weight_sync_group", group_name)

    def _set_weight_sync_src_rank(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str, src_rank: int) -> None:
        actor_wg.execute_all_sync(f"{str(Role.Actor)}_set_weight_sync_src_rank", group_name, int(src_rank))
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_set_weight_sync_src_rank", group_name, int(src_rank))

    def _clear_weight_sync_src_rank(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str) -> None:
        actor_wg.execute_all_sync(f"{str(Role.Actor)}_clear_weight_sync_src_rank", group_name)
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_clear_weight_sync_src_rank", group_name)

    def _warmup_weight_sync_group(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str):
        # The first NCCL collective on a freshly prepared Ray collective group can
        # pay a large lazy-initialization cost. Exercise the exact group with a
        # one-element broadcast before it becomes the full weight-sync path.
        rollout_refs = rollout_wg.execute_all_async(f"{str(Role.Rollout)}_warmup_weight_sync_group", group_name)
        actor_refs = actor_wg.execute_all_async(f"{str(Role.Actor)}_warmup_weight_sync_group", group_name)
        return ray.get(rollout_refs + actor_refs)

    @staticmethod
    def _collect_weight_sync_group_warmup_metrics(results) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {}
        if not results:
            return metrics
        for field in ("broadcast_s", "total_s"):
            values = [float(item[field]) for item in results if isinstance(item, dict) and isinstance(item.get(field), (int, float))]
            if values:
                metrics[f"resize/comm_prewarm_warmup_{field}_max"] = max(values)
                metrics[f"resize/comm_prewarm_warmup_{field}_mean"] = sum(values) / len(values)
        role_counts: dict[str, int] = {}
        for item in results:
            if isinstance(item, dict):
                role = str(item.get("role", "unknown"))
                role_counts[role] = role_counts.get(role, 0) + 1
        for role, count in role_counts.items():
            metrics[f"resize/comm_prewarm_warmup_{role}_workers"] = float(count)
        first = next((item for item in results if isinstance(item, dict)), None)
        if first is not None:
            for field in ("group_name", "collective_world_size", "collective_initialized"):
                value = first.get(field)
                if isinstance(value, (int, float, bool)):
                    metrics[f"resize/comm_prewarm_warmup_{field}"] = float(value)
                elif value is not None:
                    metrics[f"resize/comm_prewarm_warmup_{field}"] = str(value)
        return metrics

    def add_role_group(
        self,
        role: Role,
        *,
        name: str | None = None,
        resource_pool=None,
        detached: bool = False,
        name_prefix: str | None = None,
        prepare_only: bool = False,
    ) -> RayWorkerGroup:
        role_wg = self._build_role_group(
            role,
            resource_pool=resource_pool,
            name_prefix=name_prefix,
            detached=detached,
        )
        self._prepare_role_group_init(role_wg, role=role)
        if not prepare_only:
            self._commit_role_group_init(role_wg, role=role)
        group_map = self.role_groups.setdefault(role, {})
        group_name = name or f"{str(role)}_group_{len(group_map)}"
        group_map[group_name] = role_wg
        return role_wg

    async def add_role_group_async(
        self,
        role: Role,
        *,
        name: str | None = None,
        resource_pool=None,
        detached: bool = False,
        name_prefix: str | None = None,
        prepare_only: bool = False,
    ) -> RayWorkerGroup:
        """异步 role-group 创建接口骨架。

        目标语义：
        - 可以在旧 worker 生命周期未结束时，先把新 group spawn 出来并 prepare
        - commit 则视调度策略稍后执行

        当前阶段底层 spawn 仍是同步封装，因此这里先统一 async API 形态。
        """
        role_wg = self._build_role_group(
            role,
            resource_pool=resource_pool,
            name_prefix=name_prefix,
            detached=detached,
        )
        await self._prepare_role_group_init_async(role_wg, role=role)
        if not prepare_only:
            await self._commit_role_group_init_async(role_wg, role=role)
        group_map = self.role_groups.setdefault(role, {})
        group_name = name or f"{str(role)}_group_{len(group_map)}"
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
                weights_info = self._get_actor_weights_info(old_wg)[0]
                self._set_actor_weights_info(role_wg, weights_info)

            self.actor_wg = role_wg
            self.actor_rollout_wg = role_wg
            weights_info = self._get_actor_weights_info(role_wg)[0]
            self._set_actor_weights_info(self.rollout_wg, weights_info)

        elif role == Role.Rollout:
            old_wg = self.rollout_wg
            if resume_from_path is not None:
                print("Warning: rollout group does not support checkpoint restore; ignoring resume_from_path")
            weights_info = self._get_actor_weights_info(self.actor_wg)[0]
            self._set_actor_weights_info(role_wg, weights_info)
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

    def _next_weight_sync_group_name(self) -> str:
        self._weight_sync_group_version += 1
        return f"actor_rollout_v{self._weight_sync_group_version}"

    def _is_weight_sync_group_name_busy(self, group_name: str | None) -> bool:
        if group_name is None:
            return False
        if group_name == getattr(self, "_active_weight_sync_group_name", None):
            return True
        return any(pending_group_name == group_name for pending_group_name, _, _ in self._pending_weight_sync_group_cleanup)

    def _register_weight_sync_group_for_cleanup(
        self, group_name: str, actor_wg: RayWorkerGroup | None, rollout_wg: RayWorkerGroup | None
    ) -> None:
        if actor_wg is None or rollout_wg is None:
            return
        self._pending_weight_sync_group_cleanup.append((group_name, actor_wg, rollout_wg))

    @staticmethod
    def _is_expected_weight_sync_cleanup_error(exc: Exception) -> bool:
        message = str(exc)
        return "`ray.kill`" in message or "does not exist" in message

    def _cleanup_pending_weight_sync_groups(self) -> None:
        pending = self._pending_weight_sync_group_cleanup
        self._pending_weight_sync_group_cleanup = []
        for group_name, actor_wg, rollout_wg in pending:
            self._weight_sync_communicator_cache.discard_by_group_name(group_name)
            try:
                self._destroy_actor_weight_sync_group(actor_wg, group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip actor-side weight sync cleanup for terminated worker "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy actor-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )
            try:
                self._destroy_rollout_weight_sync_group(rollout_wg, group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip rollout-side weight sync cleanup for terminated worker "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy rollout-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )
            try:
                collective.destroy_collective_group(group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip driver-side weight sync cleanup for missing group "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy driver-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )

    def _prepare_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        rollout_wg: RayWorkerGroup,
        *,
        group_name: str | None = None,
        topology_key: str | None = None,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> str:
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = actor_wg.workers + rollout_wg.workers
        n_workers = len(actor_rollout_workers)
        topology_key = topology_key or self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)
        actor_spec = actor_spec if actor_spec is not None else self._active_topology_specs.get(Role.Actor)
        rollout_spec = rollout_spec if rollout_spec is not None else self._active_topology_specs.get(Role.Rollout)
        had_registry_entry = self._weight_sync_communicator_cache.get_topology(topology_key) is not None
        self._weight_sync_communicator_cache.register_topology(
            topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(actor_wg.workers),
            rollout_world_size=len(rollout_wg.workers),
            world_size=n_workers,
        )

        cached_entry = self._weight_sync_communicator_cache.get(
            topology_key=topology_key,
            actor_wg=actor_wg,
            rollout_wg=rollout_wg,
        )
        if cached_entry is not None:
            self._update_communicator_cache_metrics(
                **{
                    "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                    "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                    "resize/comm_prewarm_ready": 1.0 if cached_entry.is_prewarmed else 0.0,
                    "resize/comm_prepare_path": "live_hit",
                    "resize/comm_cache_reused_group": cached_entry.group_name,
                }
            )
            return cached_entry.group_name

        candidate_group_name = group_name or self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(candidate_group_name):
            candidate_group_name = None
        group_name = candidate_group_name or self._next_weight_sync_group_name()

        with resize_trace_span(
            self._resize_trace_config,
            "comm_prewarm_create",
            step=self._trace_step(),
            lane="comm_prewarm",
            metadata={"topology_key": topology_key, "group_name": group_name},
        ):
            started_at = time.monotonic()
            if self.device_name == "npu":
                master_address = ray.get(actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(actor_wg.workers[0]._get_free_port.remote())
                self._prepare_actor_weight_sync_group(actor_wg, master_address, master_port, 0, n_workers, group_name)
                self._prepare_rollout_weight_sync_group(
                    rollout_wg,
                    master_address,
                    master_port,
                    len(actor_wg.workers),
                    n_workers,
                    group_name,
                )
            else:
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                master_address = ray.get(actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(actor_wg.workers[0]._get_free_port.remote())
                self._prepare_actor_weight_sync_group(actor_wg, master_address, master_port, 0, n_workers, group_name)
                self._prepare_rollout_weight_sync_group(
                    rollout_wg,
                    master_address,
                    master_port,
                    len(actor_wg.workers),
                    n_workers,
                    group_name,
                )
            warmup_started_at = time.monotonic()
            warmup_results = self._warmup_weight_sync_group(actor_wg, rollout_wg, group_name)
            warmup_duration = time.monotonic() - warmup_started_at
            warmup_metrics = self._collect_weight_sync_group_warmup_metrics(warmup_results)
            prewarm_duration = time.monotonic() - started_at

        self._weight_sync_communicator_cache.put(
            topology_key=topology_key,
            group_name=group_name,
            actor_wg=actor_wg,
            rollout_wg=rollout_wg,
            is_prewarmed=True,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(actor_wg.workers),
            rollout_world_size=len(rollout_wg.workers),
            world_size=n_workers,
        )
        self._update_communicator_cache_metrics(
            **{
                "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                "resize/comm_prewarm_ready": 1.0,
                "resize/comm_prewarm_create_s": prewarm_duration,
                "resize/comm_prewarm_warmup_s": warmup_duration,
                "resize/comm_prepare_path": "registry_prewarm" if had_registry_entry else "full_build",
                "resize/comm_cache_reused_group": group_name,
                **warmup_metrics,
            }
        )
        return group_name

    def _create_weight_sync_group(
        self,
        *,
        group_name: str | None = None,
        topology_key: str | None = None,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ):
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)
        topology_key = topology_key or self._current_topology_key()
        actor_spec = actor_spec if actor_spec is not None else self._active_topology_specs.get(Role.Actor)
        rollout_spec = rollout_spec if rollout_spec is not None else self._active_topology_specs.get(Role.Rollout)
        had_registry_entry = self._weight_sync_communicator_cache.get_topology(topology_key) is not None
        # Track topology-scoped metadata separately from live worker bindings.
        # This commit does not prewarm communicators yet; it only makes the
        # future prewarm input explicit and reusable across worker lifecycles.
        self._weight_sync_communicator_cache.register_topology(
            topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(self.actor_wg.workers),
            rollout_world_size=len(self.rollout_wg.workers),
            world_size=n_workers,
        )

        cached_entry = self._weight_sync_communicator_cache.get(
            topology_key=topology_key,
            actor_wg=self.actor_wg,
            rollout_wg=self.rollout_wg,
        )
        if cached_entry is not None:
            # Fast path: the communicator for this exact live actor/rollout pair
            # is already available, so only switch the active group binding.
            reused_group_name = cached_entry.group_name
            was_prewarmed = cached_entry.is_prewarmed
            with resize_trace_span(
                self._resize_trace_config,
                "comm_activate",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"topology_key": topology_key, "group_name": reused_group_name},
            ):
                activate_started_at = time.monotonic()
                self.actor_wg.execute_all_sync(f"{str(Role.Actor)}_activate_weight_sync_group", reused_group_name)
                self.rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_activate_weight_sync_group", reused_group_name)
                activate_duration = time.monotonic() - activate_started_at
            if was_prewarmed:
                self._weight_sync_communicator_cache.mark_activated(
                    topology_key=topology_key,
                    actor_wg=self.actor_wg,
                    rollout_wg=self.rollout_wg,
                )
            self._active_weight_sync_group_name = reused_group_name
            self._update_communicator_cache_metrics(
                **{
                    "resize/comm_cache_hit": 1.0,
                    "resize/comm_cache_miss": 0.0,
                    "resize/comm_live_cache_hit": 1.0,
                    "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                    "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                    "resize/comm_prewarm_ready": 1.0 if was_prewarmed else 0.0,
                    "resize/comm_activate_s": activate_duration,
                    "resize/comm_prepare_path": "prewarmed_activate" if was_prewarmed else "live_hit",
                    "resize/comm_cache_reused_group": reused_group_name,
                }
            )
            return

        # Slow path: either this topology has never been seen, or it is being
        # revisited with newly spawned workers. In both cases we rebuild the
        # communicator and then refresh the cache entry.
        candidate_group_name = group_name or self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(candidate_group_name):
            candidate_group_name = None
        group_name = candidate_group_name or self._next_weight_sync_group_name()
        registry_hit = had_registry_entry

        with resize_trace_span(
            self._resize_trace_config,
            "comm_full_build",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"topology_key": topology_key, "group_name": group_name},
        ):
            started_at = time.monotonic()
            if self.device_name == "npu":
                master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
                self._create_actor_weight_sync_group(
                    self.actor_wg,
                    master_address,
                    master_port,
                    0,
                    n_workers,
                    group_name,
                )
                self._create_rollout_weight_sync_group(
                    self.rollout_wg,
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                    group_name,
                )
            else:
                # Create Ray collective group for fallback communication
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                # NOTE(HanlinDu): collective init not finished before broadcast, so we init here to avoid potential issues
                # may not be necessary for all cases, but safer to have it
                master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
                self._create_actor_weight_sync_group(
                    self.actor_wg,
                    master_address,
                    master_port,
                    0,
                    n_workers,
                    group_name,
                )
                self._create_rollout_weight_sync_group(
                    self.rollout_wg,
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                    group_name,
                )
            full_build_duration = time.monotonic() - started_at
        self._active_weight_sync_group_name = group_name
        self._weight_sync_communicator_cache.put(
            topology_key=topology_key,
            group_name=group_name,
            actor_wg=self.actor_wg,
            rollout_wg=self.rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(self.actor_wg.workers),
            rollout_world_size=len(self.rollout_wg.workers),
            world_size=n_workers,
        )
        self._update_communicator_cache_metrics(
            **{
                "resize/comm_cache_hit": 0.0,
                "resize/comm_cache_miss": 1.0,
                "resize/comm_live_cache_hit": 0.0,
                "resize/comm_registry_hit": 1.0 if registry_hit else 0.0,
                "resize/comm_registry_miss": 0.0 if registry_hit else 1.0,
                "resize/comm_prewarm_ready": 0.0,
                "resize/comm_full_build_s": full_build_duration,
                "resize/comm_prepare_path": "full_build",
                "resize/comm_cache_reused_group": group_name,
            }
        )

    def _switch_weight_sync_group(
        self,
        new_actor_wg: RayWorkerGroup,
        new_rollout_wg: RayWorkerGroup,
        *,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> None:
        # Cache lookup is topology-aware, but reuse is still constrained by the
        # concrete worker-group identities for safety.
        old_group_name = getattr(self, "_active_weight_sync_group_name", None)
        old_actor_wg = getattr(self, "actor_wg", None)
        old_rollout_wg = getattr(self, "rollout_wg", None)
        topology_key = self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)
        reserved_group_name = self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(reserved_group_name):
            reserved_group_name = None
        new_group_name = reserved_group_name or self._next_weight_sync_group_name()

        self.actor_wg = new_actor_wg
        self.rollout_wg = new_rollout_wg
        self.actor_rollout_wg = new_actor_wg
        self._create_weight_sync_group(
            group_name=new_group_name,
            topology_key=topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )

        if old_group_name is not None and old_actor_wg is not None and old_rollout_wg is not None:
            self._register_weight_sync_group_for_cleanup(old_group_name, old_actor_wg, old_rollout_wg)

    def _ensure_actor_training_group_binding(self) -> None:
        """Keep shared PPO paths bound to the current actor-side group.

        In one-step-off mode, rollout generation is handled explicitly by
        `rollout_wg` / `async_rollout_manager`, while shared PPO helpers such as
        `_balance_batch`, `compute_log_prob`, `update_actor`, profiling and
        snapshot dumping still implicitly access `actor_rollout_wg` expecting an
        actor-dispatch-capable group. Rebind it proactively to avoid stale or
        rollout-only references leaking into those paths after resize.
        """
        self.actor_rollout_wg = self.actor_wg

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

    async def _init_async_rollout_manager_async(self):
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        self.async_rollout_mode = True
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = await OneStepOffAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            rm_resource_pool=rm_resource_pool,
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

    def _is_split_resize_spec(self, pool_spec: dict | None) -> bool:
        return bool(pool_spec) and pool_spec.get("mode", "pool") == "split"

    def _get_staged_shared_pool_resize_plan(self, item: dict) -> dict:
        actor_spec = item["actor_pool"]
        rollout_spec = item["rollout_pool"]

        old_actor_count = self._active_role_world_sizes.get(Role.Actor)
        old_rollout_count = self._active_role_world_sizes.get(Role.Rollout)
        if old_actor_count is None or old_rollout_count is None:
            raise ValueError(
                "staged shared-pool resize requires active topology state before resize, got "
                f"active_sizes={self._active_role_world_sizes}"
            )

        actor_target = actor_spec["size"][actor_spec.get("index", 0)]
        rollout_target = rollout_spec["size"][rollout_spec.get("index", 0)]
        actor_delta = actor_target - old_actor_count
        rollout_delta = rollout_target - old_rollout_count

        if actor_delta == 0 and rollout_delta == 0:
            raise ValueError("staged shared-pool resize got a no-op target topology")
        if actor_delta * rollout_delta >= 0:
            raise ValueError(
                "staged shared-pool resize requires one role to expand and the other to shrink, got "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )
        if abs(actor_delta) != abs(rollout_delta):
            raise ValueError(
                "staged shared-pool resize requires balanced split reallocation, got "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )

        shrinking_role = Role.Actor if actor_delta < 0 else Role.Rollout
        expanding_role = Role.Rollout if shrinking_role == Role.Actor else Role.Actor
        delta = abs(actor_delta)

        return {
            "actor_spec": actor_spec,
            "rollout_spec": rollout_spec,
            "old_actor_count": old_actor_count,
            "old_rollout_count": old_rollout_count,
            "actor_target": actor_target,
            "rollout_target": rollout_target,
            "actor_delta": actor_delta,
            "rollout_delta": rollout_delta,
            "shrinking_role": shrinking_role,
            "expanding_role": expanding_role,
            "delta": delta,
        }

    def _resolve_staged_actor_resume(self, actor_resume_path: str | None) -> tuple[str | None, bool, str | None]:
        if actor_resume_path is not None:
            logger.info("[one-step-off][resize][resume] using explicit actor checkpoint path: %s", actor_resume_path)
            return actor_resume_path, False, "checkpoint"

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        handoff_dir = os.path.join(checkpoint_folder, f"dynamic_resize_actor_handoff_step_{self.global_steps}")
        # When host staging is enabled, the same directory becomes a staging root
        # rather than a plain checkpoint handoff directory.
        if self._handoff_staging_config.enable:
            logger.info(
                "[one-step-off][resize][host-stage] using temporary actor host staging dir: %s backend=%s",
                handoff_dir,
                self._handoff_staging_config.effective_backend(),
            )
            return handoff_dir, self._handoff_staging_config.cleanup_after_load, "host_staging"
        logger.info("[one-step-off][resize][handoff] using temporary actor handoff dir: %s", handoff_dir)
        return handoff_dir, True, "handoff"

    def _normalize_actor_resume_load_path(self, actor_resume_path: str | None) -> str | None:
        if actor_resume_path is None:
            return None

        actor_subdir = os.path.join(actor_resume_path, "actor")
        if os.path.isdir(actor_subdir):
            return actor_subdir
        return actor_resume_path

    async def _shutdown_async_rollout_manager_async(self) -> None:
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            return

        shutdown_async = getattr(manager, "shutdown_async", None)
        if shutdown_async is None:
            self.async_rollout_manager = None
            return

        result = shutdown_async()
        if inspect.isawaitable(result):
            await result
        self.async_rollout_manager = None

    async def _clear_rollout_kv_cache_before_resize_async(self) -> float:
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            return 0.0

        started_at = time.monotonic()
        clear_async = getattr(manager, "clear_kv_cache_async", None)
        if clear_async is not None:
            await clear_async()
            return time.monotonic() - started_at

        clear_sync = getattr(manager, "clear_kv_cache", None)
        if clear_sync is not None:
            result = clear_sync()
            if inspect.isawaitable(result):
                await result
            return time.monotonic() - started_at

        return 0.0

    def _has_dynamic_resize_scheduled_at_step(self, step: int) -> bool:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return False
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", True):
            return False
        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())
        if not isinstance(schedule, list):
            return False
        return any(isinstance(item, dict) and item.get("step") == step for item in schedule)

    def _should_use_staged_shared_pool_resize(self, item: dict) -> bool:
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        if not (self._is_split_resize_spec(actor_spec) and self._is_split_resize_spec(rollout_spec)):
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: split spec mismatch, "
                f"actor_spec={actor_spec}, rollout_spec={rollout_spec}"
            )
            return False

        actor_size = actor_spec.get("size")
        rollout_size = rollout_spec.get("size")
        if not (isinstance(actor_size, list) and isinstance(rollout_size, list) and len(actor_size) == len(rollout_size)):
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: size list mismatch, "
                f"actor_size={actor_size}, rollout_size={rollout_size}"
            )
            return False

        same_base_pool = actor_spec.get("from_pool") == rollout_spec.get("from_pool")
        if not same_base_pool:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: actor/rollout base pool mismatch, "
                f"actor_from={actor_spec.get('from_pool')}, rollout_from={rollout_spec.get('from_pool')}"
            )
            return False

        old_actor = self._active_role_world_sizes.get(Role.Actor)
        old_rollout = self._active_role_world_sizes.get(Role.Rollout)
        if old_actor is None or old_rollout is None:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: active topology sizes missing, "
                f"active_sizes={self._active_role_world_sizes}"
            )
            return False

        actor_target = actor_size[actor_spec.get("index", 0)]
        rollout_target = rollout_size[rollout_spec.get("index", 0)]
        actor_delta = actor_target - old_actor
        rollout_delta = rollout_target - old_rollout
        enabled = same_base_pool and actor_delta * rollout_delta < 0
        if not enabled:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: staged resize needs opposite deltas, "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )
        return enabled

    async def _staged_resize_shared_pool(self, item: dict, *, actor_resume_path: str | None = None) -> bool:
        plan = self._get_staged_shared_pool_resize_plan(item)
        actor_spec = plan["actor_spec"]
        rollout_spec = plan["rollout_spec"]
        detached = item.get("detached", False)
        name_prefix = item.get("name_prefix")

        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg
        old_actor_count = plan["old_actor_count"]
        old_rollout_count = plan["old_rollout_count"]
        actor_target = plan["actor_target"]
        rollout_target = plan["rollout_target"]
        actor_delta = plan["actor_delta"]
        rollout_delta = plan["rollout_delta"]
        shrinking_role = plan["shrinking_role"]
        model_bytes = self._estimate_weights_info_bytes(self._get_actor_weights_info(old_actor_wg)[0])

        actor_resume_path, should_cleanup_actor_resume, actor_resume_kind = self._resolve_staged_actor_resume(
            actor_resume_path,
        )
        runtime_staging_cfg = self._make_runtime_handoff_staging_dict(
            effective_backend=self._handoff_staging_config.effective_backend()
        )

        if actor_resume_kind in {"host_staging", "handoff"}:
            # The same budget gate covers both host-staging export and legacy
            # handoff export. The only difference is whether we are allowed to
            # downgrade the requested backend to disk fallback.
            export_snapshot = self._get_role_group_resource_snapshot(Role.Actor, old_actor_wg, staging_path=actor_resume_path)
            export_decision = self._resize_budget_controller.evaluate_export(
                requested_backend=self._handoff_staging_config.backend if actor_resume_kind == "host_staging" else "disk_fallback",
                snapshot=export_snapshot,
                estimated_host_peak_bytes=self._estimate_host_export_peak_bytes(
                    model_bytes,
                    stage_optimizer=self._handoff_staging_config.stage_optimizer if actor_resume_kind == "host_staging" else True,
                ),
                estimated_stage_bytes=model_bytes,
            )
            self._update_resize_budget_metrics(
                **{
                    "resize/budget_blocked": 1.0 if export_decision.blocked else 0.0,
                    "resize/budget_reason": export_decision.reason,
                    "resize/budget_effective_backend": export_decision.effective_backend,
                }
            )
            if export_decision.blocked:
                logger.warning(
                    "[one-step-off][resize][budget] skip staged resize before export: step=%s reason=%s",
                    self.global_steps,
                    export_decision.reason,
                )
                return False
            if actor_resume_kind == "host_staging":
                runtime_staging_cfg = self._make_runtime_handoff_staging_dict(effective_backend=export_decision.effective_backend)

        # Phase 0: export the train-side state into the host staging backend
        # before shared-pool slots are reclaimed. For pinned + deferred full
        # optimizer restore, only model pages are on the critical path; optimizer
        # pages are appended asynchronously while new worker groups reinit.
        optimizer_export_task = None
        split_optimizer_export = (
            actor_resume_kind == "host_staging"
            and runtime_staging_cfg.get("backend") == "pinned_cpu"
            and bool(runtime_staging_cfg.get("stage_optimizer", False))
            and str(runtime_staging_cfg.get("optimizer_restore_policy", "deferred")) == "deferred"
            and bool(runtime_staging_cfg.get("optimizer_full_state_restore", True))
            and shrinking_role == Role.Rollout
        )
        if actor_resume_kind == "host_staging":
            logger.info("[one-step-off][resize][host-stage] exporting actor model state before staged switch")
            critical_staging_cfg = dict(runtime_staging_cfg)
            if split_optimizer_export:
                critical_staging_cfg["stage_optimizer"] = False
            with resize_trace_span(
                self._resize_trace_config,
                "resize_export",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={
                    "backend": runtime_staging_cfg.get("backend", "disk_fallback"),
                    "split_optimizer_export": bool(split_optimizer_export),
                },
            ):
                export_started_at = time.monotonic()
                export_results = old_actor_wg.stage_actor_handoff_state_to_host(
                    actor_resume_path,
                    staging_config=critical_staging_cfg,
                )
                export_duration = time.monotonic() - export_started_at
            export_metrics = {"resize/host_stage_export_s": export_duration}
            export_metrics.update(self._merge_resize_numeric_metrics(export_results))
            export_metrics["resize/host_stage_split_optimizer_export"] = 1.0 if split_optimizer_export else 0.0
            self._update_resize_execution_metrics(**export_metrics)
            if split_optimizer_export:
                logger.info(
                    "[one-step-off][resize][host-stage] deferring async optimizer export until after new worker reinit starts"
                )
            if self._handoff_staging_config.preclear_rollout_kv_cache:
                # Reclaim rollout-side transient memory before the switch window.
                kv_cache_preclear_s = await self._clear_rollout_kv_cache_before_resize_async()
                self._update_resize_execution_metrics(**{"resize/kv_cache_preclear_s": kv_cache_preclear_s})
        elif actor_resume_kind == "handoff":
            logger.info("[one-step-off][resize][handoff] exporting actor state before staged switch")
            with resize_trace_span(
                self._resize_trace_config,
                "resize_export",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"backend": "handoff"},
            ):
                export_started_at = time.monotonic()
                old_actor_wg.save_actor_handoff_state(actor_resume_path)
                export_duration = time.monotonic() - export_started_at
            self._update_resize_execution_metrics(**{"resize/host_stage_export_s": export_duration})

        # GPU restore is checked after KV-cache preclear/export, because the
        # effective free memory at that moment is what matters for the switch.
        gpu_role = Role.Rollout if shrinking_role == Role.Rollout else Role.Actor
        gpu_wg = old_rollout_wg if shrinking_role == Role.Rollout else old_actor_wg
        restore_snapshot = self._get_role_group_resource_snapshot(gpu_role, gpu_wg, staging_path=actor_resume_path)
        restore_decision = self._resize_budget_controller.evaluate_restore(
            snapshot=restore_snapshot,
            estimated_gpu_peak_bytes=self._estimate_gpu_restore_peak_bytes(
                model_bytes,
                progressive_swap=bool(runtime_staging_cfg.get("progressive_swap", False)),
                chunk_mb=int(runtime_staging_cfg.get("chunk_mb", 0) or 0),
            ),
        )
        self._update_resize_budget_metrics(
            **{
                "resize/budget_blocked": 1.0 if restore_decision.blocked else 0.0,
                "resize/budget_reason": restore_decision.reason,
                "resize/budget_effective_backend": runtime_staging_cfg.get("backend", "disk_fallback"),
            }
        )
        if restore_decision.blocked:
            logger.warning(
                "[one-step-off][resize][budget] skip staged resize before restore: step=%s reason=%s",
                self.global_steps,
                restore_decision.reason,
            )
            if should_cleanup_actor_resume and actor_resume_path is not None:
                shutil.rmtree(actor_resume_path, ignore_errors=True)
                self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
            return False

        logger.info(
            "[one-step-off][resize][staged] start: "
            f"old_actor={old_actor_count}, old_rollout={old_rollout_count}, "
            f"target_actor={actor_target}, target_rollout={rollout_target}, "
            f"actor_delta={actor_delta}, rollout_delta={rollout_delta}, "
            f"shrinking_role={shrinking_role}"
        )

        # Phase A: release whichever side is shrinking to free shared-pool slots.
        if shrinking_role == Role.Actor:
            self.remove_role_group(Role.Actor, old_actor_wg)
        else:
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
        rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

        actor_resume_load_path = self._normalize_actor_resume_load_path(actor_resume_path)
        defer_optimizer_restore = bool(
            runtime_staging_cfg.get("stage_optimizer", False)
            and str(runtime_staging_cfg.get("optimizer_restore_policy", "deferred")) == "deferred"
        )
        actor_restore_task = None

        async def _start_actor_restore_after_commit(actor_wg: RayWorkerGroup):
            restore_task = None
            if actor_resume_kind == "host_staging" and actor_resume_path is not None:
                logger.info("[one-step-off][resize][host-stage] starting async actor state import into new actor")
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_import_launch",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
                ):
                    restore_task = self._start_async_actor_host_restore(
                        actor_wg,
                        actor_resume_path,
                        runtime_staging_cfg,
                    )
            elif actor_resume_kind == "handoff" and actor_resume_path is not None:
                logger.info("[one-step-off][resize][handoff] importing actor state into new actor")
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_import",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": "handoff"},
                ):
                    import_started_at = time.monotonic()
                    try:
                        actor_wg.load_actor_handoff_state(actor_resume_path)
                    except Exception as exc:
                        self._update_resize_execution_metrics(
                            **{
                                "resize/restore_failed": 1.0,
                                "resize/partial_restore_cleanup_count": 1.0,
                            }
                        )
                        logger.exception(
                            "[one-step-off][resize][handoff] restore failed: path=%s "
                            "resize/restore_failed=1.0 resize/partial_restore_cleanup_count=1.0 error=%r",
                            actor_resume_path,
                            exc,
                        )
                        try:
                            actor_wg.cleanup_actor_handoff_restore_session(actor_resume_path)
                        finally:
                            raise
                    import_duration = time.monotonic() - import_started_at
                self._update_resize_execution_metrics(**{"resize/host_stage_import_s": import_duration})
                if should_cleanup_actor_resume:
                    try:
                        shutil.rmtree(actor_resume_path)
                        self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
                        logger.info("[one-step-off][resize][handoff] cleaned temporary actor handoff dir: %s", actor_resume_path)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:  # pragma: no cover - best effort cleanup
                        logger.warning(
                            "[one-step-off][resize][handoff] failed to remove temporary handoff %s: %s",
                            actor_resume_path,
                            exc,
                        )
            elif actor_resume_load_path is not None:
                logger.info("[one-step-off][resize][resume] loading actor checkpoint into new actor: %s", actor_resume_load_path)
                actor_wg.load_checkpoint(
                    actor_resume_load_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
                if should_cleanup_actor_resume:
                    try:
                        shutil.rmtree(actor_resume_path)
                        logger.info("[one-step-off][resize][resume] cleaned temporary checkpoint dir: %s", actor_resume_path)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:  # pragma: no cover - best effort cleanup
                        logger.warning(
                            "[one-step-off][resize][resume] failed to remove temporary checkpoint %s: %s",
                            actor_resume_path,
                            exc,
                        )
            else:
                logger.info(
                    "[one-step-off][resize][staged] new actor keeps its freshly initialized weights; "
                    "rollout weight metadata will be attached after rollout group commit"
                )
            return restore_task

        if shrinking_role == Role.Rollout:
            # Once the shrinking rollout has been released, the target actor and
            # rollout groups fit in the shared pool together. Build/commit them
            # concurrently; actor restore starts as soon as actor commit returns.
            async def _build_commit_actor():
                actor_wg = await asyncio.to_thread(
                    self.add_role_group,
                    Role.Actor,
                    name=item.get("actor_group_name"),
                    resource_pool=actor_pool,
                    detached=detached,
                    name_prefix=name_prefix,
                    prepare_only=True,
                )
                await asyncio.to_thread(self._commit_role_group_init, actor_wg, role=Role.Actor)
                return actor_wg

            async def _build_commit_rollout():
                rollout_wg = await asyncio.to_thread(
                    self.add_role_group,
                    Role.Rollout,
                    name=item.get("rollout_group_name"),
                    resource_pool=rollout_pool,
                    detached=detached,
                    name_prefix=name_prefix,
                    prepare_only=True,
                )
                await asyncio.to_thread(self._commit_role_group_init, rollout_wg, role=Role.Rollout)
                return rollout_wg

            actor_group_task = asyncio.create_task(_build_commit_actor())
            rollout_group_task = asyncio.create_task(_build_commit_rollout())
            new_actor_wg = await actor_group_task
            actor_restore_task = await _start_actor_restore_after_commit(new_actor_wg)
            new_rollout_wg = await rollout_group_task
            self._update_resize_execution_metrics(**{"resize/parallel_actor_rollout_reinit": 1.0})
        else:
            # Actor shrink needs old rollout lifetime/resource ordering preserved;
            # keep the original actor-first sequence for that direction.
            new_actor_wg = await self.add_role_group_async(
                Role.Actor,
                name=item.get("actor_group_name"),
                resource_pool=actor_pool,
                detached=detached,
                name_prefix=name_prefix,
                prepare_only=True,
            )
            await self._commit_role_group_init_async(new_actor_wg, role=Role.Actor)
            actor_restore_task = await _start_actor_restore_after_commit(new_actor_wg)

            self.remove_role_group(Role.Rollout, old_rollout_wg)
            new_rollout_wg = await self.add_role_group_async(
                Role.Rollout,
                name=item.get("rollout_group_name"),
                resource_pool=rollout_pool,
                detached=detached,
                name_prefix=name_prefix,
                prepare_only=True,
            )
            await self._commit_role_group_init_async(new_rollout_wg, role=Role.Rollout)
            self._update_resize_execution_metrics(**{"resize/parallel_actor_rollout_reinit": 0.0})

        if split_optimizer_export and optimizer_export_task is None:
            logger.info("[one-step-off][resize][host-stage] starting async optimizer export after worker reinit")
            optimizer_export_task = self._start_async_actor_optimizer_host_export(
                old_actor_wg,
                actor_resume_path,
                runtime_staging_cfg,
            )

        if actor_restore_task is not None:
            logger.info("[one-step-off][resize][host-stage] waiting for async actor import before reading weights")
            with resize_trace_span(
                self._resize_trace_config,
                "resize_import_wait",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
            ):
                self._finish_async_actor_host_restore(
                    actor_restore_task,
                    actor_wg=new_actor_wg,
                    actor_resume_path=actor_resume_path,
                    runtime_staging_cfg=runtime_staging_cfg,
                )
            if defer_optimizer_restore and has_restore_session_manifest(actor_resume_path):
                session_manifest = read_restore_session_manifest(actor_resume_path)
                self._update_resize_execution_metrics(
                    **{
                        "resize/optimizer_deferred_restore": 1.0,
                        "resize/optimizer_pending_pages": float(session_manifest.get("optimizer_page_count", 0)),
                    }
                )
            new_actor_wg.release_host_staging_buffer(
                actor_resume_path,
                staging_config=runtime_staging_cfg,
            )
            if should_cleanup_actor_resume and not defer_optimizer_restore:
                try:
                    shutil.rmtree(actor_resume_path)
                    self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
                    logger.info("[one-step-off][resize][host-stage] cleaned temporary actor handoff dir: %s", actor_resume_path)
                except FileNotFoundError:
                    pass
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning(
                        "[one-step-off][resize][host-stage] failed to remove temporary handoff %s: %s",
                        actor_resume_path,
                        exc,
                    )

        weights_info = self._get_actor_weights_info(new_actor_wg)[0]
        self._set_actor_weights_info(new_rollout_wg, weights_info)

        # Keep the staged shared-pool path aligned with the non-staged switch
        # flow: prewarm the inter-role communicator on the new worker lifecycle
        # before publishing the topology, so the final switch can use activate
        # instead of rebuilding the communicator in the critical window.
        if self._dynamic_resize_async_comm_prewarm_enabled:
            self._prepare_weight_sync_group(
                new_actor_wg,
                new_rollout_wg,
                topology_key=self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec),
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )

        # Publish the new pair together.
        self._switch_weight_sync_group(
            new_actor_wg,
            new_rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )
        self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)
        if defer_optimizer_restore:
            if optimizer_export_task is not None:
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_optimizer_export_wait",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
                ):
                    optimizer_export_metrics = self._finish_async_actor_optimizer_host_export(optimizer_export_task)
                self._update_resize_execution_metrics(**optimizer_export_metrics)
                if has_restore_session_manifest(actor_resume_path):
                    session_manifest = read_restore_session_manifest(actor_resume_path)
                    self._update_resize_execution_metrics(
                        **{"resize/optimizer_pending_pages": float(session_manifest.get("optimizer_page_count", 0))}
                    )
            # Default pinned optimizer restore uses the original full-state
            # distribution path. Start it asynchronously right after publishing
            # the new topology so it can overlap with post-switch rollout work.
            # The experimental chunked optimizer path remains available behind
            # optimizer_full_state_restore=false.
            use_lazy_pinned_optimizer = (
                runtime_staging_cfg.get("backend") == "pinned_cpu"
                and bool(runtime_staging_cfg.get("async_optimizer_preload", True))
                and not bool(runtime_staging_cfg.get("optimizer_full_state_restore", True))
            )
            if not use_lazy_pinned_optimizer:
                self._start_deferred_optimizer_materialize(new_actor_wg)

        await self._shutdown_async_rollout_manager_async()
        await self._init_async_rollout_manager_async()
        await self._sync_rollout_weights_after_resize_async(handoff_path=actor_resume_path)

        if shrinking_role == Role.Rollout:
            self.remove_role_group(Role.Actor, old_actor_wg)
        self._cleanup_pending_weight_sync_groups()
        logger.info("[one-step-off][resize][staged] done")
        return True

    async def _hard_resize_actor_rollout_from_checkpoint(self, item: dict, *, actor_resume_path: str | None = None) -> bool:
        """Baseline resize: destroy old actor/rollout workers, then rebuild from checkpoint.

        This path intentionally avoids the staged shared-pool handoff, pinned
        pages, direct rollout page load, phased prewarm, and communicator cache
        optimizations used by the main resize path. It keeps critic untouched so
        the experiment isolates actor/rollout topology rebuild cost.
        """
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        detached = item.get("detached", False)
        name_prefix = item.get("name_prefix")
        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg

        handoff_path = actor_resume_path or os.path.join(
            self.config.trainer.default_local_dir,
            f"hard_resize_actor_handoff_step_{self.global_steps}",
        )
        if actor_resume_path is None:
            shutil.rmtree(handoff_path, ignore_errors=True)
        hard_staging_cfg = self._make_runtime_handoff_staging_dict(effective_backend="disk_fallback")
        hard_staging_cfg.update(
            {
                "backend": "disk_fallback",
                "stage_optimizer": True,
                "optimizer_restore_policy": "immediate",
                "progressive_swap": False,
                "async_optimizer_preload": False,
                "cleanup_after_load": False,
                "preclear_rollout_kv_cache": False,
            }
        )
        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_export_handoff",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"backend": "disk_fallback"},
        ):
            export_started_at = time.monotonic()
            export_results = old_actor_wg.stage_actor_handoff_state_to_host(
                handoff_path,
                staging_config=hard_staging_cfg,
            )
            export_duration = time.monotonic() - export_started_at
        export_metrics = {"resize/hard_resize_export_handoff_s": export_duration}
        export_metrics.update(self._merge_resize_numeric_metrics(export_results))
        self._update_resize_execution_metrics(**export_metrics)

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_shutdown_rollout_manager",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            await self._shutdown_async_rollout_manager_async()

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_kill_old_workers",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
        rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

        new_actor_wg = await self.add_role_group_async(
            Role.Actor,
            name=item.get("actor_group_name"),
            resource_pool=actor_pool,
            detached=detached,
            name_prefix=name_prefix,
            prepare_only=False,
        )
        new_rollout_wg = await self.add_role_group_async(
            Role.Rollout,
            name=item.get("rollout_group_name"),
            resource_pool=rollout_pool,
            detached=detached,
            name_prefix=name_prefix,
            prepare_only=False,
        )

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_actor_load_handoff",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"path": handoff_path, "backend": "disk_fallback"},
        ):
            new_actor_wg.load_actor_handoff_state_from_host(
                handoff_path,
                staging_config=hard_staging_cfg,
            )

        self.actor_wg = new_actor_wg
        self.actor_rollout_wg = new_actor_wg
        self.rollout_wg = new_rollout_wg
        weights_info = self._get_actor_weights_info(new_actor_wg)[0]
        self._set_actor_weights_info(new_rollout_wg, weights_info)

        self._switch_weight_sync_group(
            new_actor_wg,
            new_rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )
        self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_init_rollout_manager",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            await self._init_async_rollout_manager_async()
        await self._sync_rollout_weights_after_resize_async(handoff_path=None)
        self._cleanup_pending_weight_sync_groups()
        self._update_resize_execution_metrics(
            **{
                "resize/hard_resize_destroy_all": 1.0,
                "resize/parallel_actor_rollout_reinit": 0.0,
            }
        )
        return True

    async def _maybe_dynamic_resize(self):
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

            logger.info(
                "[one-step-off][resize] trigger: "
                f"global_step={self.global_steps}, item={item}"
            )

            gate_pass, required_action, gate_metrics = self._gate_dynamic_resize_schedule_item(item)
            self._latest_resize_control_metrics = gate_metrics
            if not gate_pass:
                logger.info(
                    "[one-step-off][resize][gate] skip scheduled resize: step=%s required_action=%s signal=%s "
                    "decision=%s ratio=%.4f cooldown_remaining=%s dwell_remaining=%s",
                    self.global_steps,
                    required_action,
                    gate_metrics.get("resize/hysteresis_signal"),
                    gate_metrics.get("resize/hysteresis_decision"),
                    float(gate_metrics.get("resize/rollout_train_ratio", 0.0)),
                    gate_metrics.get("resize/cooldown_remaining"),
                    gate_metrics.get("resize/dwell_remaining"),
                )
                continue

            actor_spec = item.get("actor_pool")
            rollout_spec = item.get("rollout_pool")
            release_old = item.get("release_old", True)
            detached = item.get("detached", False)
            actor_resume_path = item.get("actor_resume_from_path")

            if bool(item.get("hard_resize_destroy_all", False)):
                logger.info("[one-step-off][resize][hard] using destroy-all hard resize baseline path")
                resize_applied = await self._hard_resize_actor_rollout_from_checkpoint(item, actor_resume_path=actor_resume_path)
                if resize_applied:
                    if self._resize_controller is not None:
                        snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
                    logger.info("[one-step-off][resize][hard] switch complete")
                continue

            staged_enabled = self._should_use_staged_shared_pool_resize(item)
            if staged_enabled:
                if release_old:
                    logger.info(
                        "[one-step-off][resize] normalize release_old=true shared-pool split resize to staged path"
                    )
                logger.info("[one-step-off][resize] using staged shared-pool resize path")
                resize_applied = await self._staged_resize_shared_pool(item, actor_resume_path=actor_resume_path)
                if resize_applied:
                    if self._resize_controller is not None:
                        snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
                    logger.info("[one-step-off][resize] staged switch complete")
                continue

            actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
            rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

            # 先只做新 group 的 prepare，给“旧 worker 仍存活时提前准备”预留时序。
            new_actor = await self.add_role_group_async(
                Role.Actor,
                name=item.get("actor_group_name"),
                resource_pool=actor_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
                prepare_only=True,
            )
            new_rollout = await self.add_role_group_async(
                Role.Rollout,
                name=item.get("rollout_group_name"),
                resource_pool=rollout_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
                prepare_only=True,
            )
            await self._switch_actor_rollout_groups(
                new_actor,
                new_rollout,
                resume_from_path=actor_resume_path,
                release_old=release_old,
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )
            if self._resize_controller is not None:
                snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
            logger.info(
                "[one-step-off][resize] switch complete: actor=%s, rollout=%s",
                len(getattr(new_actor, "workers", [])),
                len(getattr(new_rollout, "workers", [])),
            )

    async def _switch_actor_rollout_groups(
        self,
        new_actor_wg: RayWorkerGroup,
        new_rollout_wg: RayWorkerGroup,
        *,
        resume_from_path: str | None = None,
        release_old: bool = False,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> None:
        # Dynamic resize must switch actor and rollout as one pair. Updating them
        # one by one temporarily creates a mixed topology (new actor + old rollout
        # or the reverse), and that intermediate state can hang weight sync /
        # collective group setup. Stage both groups first, then publish them
        # together and rebuild the dependent managers once.
        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg

        # Phase 1: 收缩旧拓扑。
        # prepare 已经在旧 worker 存活时提前完成；如果当前策略允许释放旧 group，
        # 则优先在切换窗口开始时回收旧资源，再执行新 group 的 commit。
        if release_old:
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        # Phase 2: 提交新拓扑。
        # 这一阶段执行真正依赖最终 worker/runtime 状态的初始化。
        await self._commit_role_group_init_async(new_actor_wg, role=Role.Actor)
        await self._commit_role_group_init_async(new_rollout_wg, role=Role.Rollout)

        # Phase 3: 恢复/传递权重状态。
        # actor 先恢复可训练状态，再把最新参数视图传给 rollout。
        if resume_from_path is not None:
            logger.info("[one-step-off][resize] loading checkpoint into new actor")
            new_actor_wg.load_checkpoint(
                resume_from_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        else:
            logger.info(
                "[one-step-off][resize] new actor keeps its freshly initialized weights; "
                "rollout weight metadata will be attached afterwards"
            )

        weights_info = self._get_actor_weights_info(new_actor_wg)[0]
        self._set_actor_weights_info(new_rollout_wg, weights_info)

        # Prewarm the inter-role communicator on the new worker lifecycle before
        # publishing the new topology. The final switch can then be reduced to a
        # lightweight activation step instead of an in-window rendezvous.
        if self._dynamic_resize_async_comm_prewarm_enabled:
            self._prepare_weight_sync_group(
                new_actor_wg,
                new_rollout_wg,
                topology_key=self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec),
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )

        # Phase 4: 发布新拓扑。
        # actor/rollout 必须作为一对同时对外可见，避免混合拓扑。
        self._switch_weight_sync_group(
            new_actor_wg,
            new_rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )
        self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)

        # Phase 5: 重建依赖最终拓扑的管理器和同步链路。
        # Rollout manager 和 sync group 都依赖最终 actor/rollout 对，因此必须在
        # publish 之后统一重建。
        await self._shutdown_async_rollout_manager_async()
        await self._init_async_rollout_manager_async()
        await self._sync_rollout_weights_after_resize_async(handoff_path=resume_from_path)

        # Phase 6: 兼容旧策略的延后清理。
        # 当 release_old=False 时，保留旧语义：新拓扑发布完后再释放旧 group。
        if not release_old:
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)
            self._cleanup_pending_weight_sync_groups()



    @staticmethod
    def _collect_rollout_direct_weight_load_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = (
            "total_s",
            "prepare_inference_model_s",
            "load_pages_s",
            "h2d_s",
            "rollout_load_weights_s",
            "empty_cache_s",
            "page_count",
            "manifest_page_count",
            "empty_page_count",
            "loaded_weight_count",
            "loaded_bytes",
            "host_file_page_count",
            "disk_page_count",
            "host_page_count",
            "device_free_before_bytes",
            "device_total_before_bytes",
            "device_free_after_bytes",
            "device_total_after_bytes",
        )
        rollout_results = [item for item in results if isinstance(item, dict) and item.get("role") == "rollout"]
        if not rollout_results:
            return metrics
        metrics["rollout_direct_weight_load_worker_count"] = float(len(rollout_results))
        for field in numeric_fields:
            values = []
            for item in rollout_results:
                value = item.get(field)
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                metrics[f"rollout_direct_weight_load_{field}_max"] = max(values)
                metrics[f"rollout_direct_weight_load_{field}_mean"] = sum(values) / len(values)
        first = rollout_results[0]
        source = first.get("source")
        if source is not None:
            metrics["rollout_direct_weight_load_source"] = str(source)
        return metrics

    def _load_rollout_weights_from_handoff_model_pages(self, handoff_path: str):
        launch_started_at = time.monotonic()
        refs = self.rollout_wg.execute_all_async(
            f"{str(Role.Rollout)}_load_rollout_weights_from_handoff_model_pages",
            handoff_path,
            staging_config=self._make_runtime_handoff_staging_dict(
                effective_backend=self._handoff_staging_config.effective_backend()
            ),
        )
        wait_started_at = time.monotonic()
        results = ray.get(refs) if refs else []
        diagnostics = {
            "rollout_direct_weight_load_launch_s": float(wait_started_at - launch_started_at),
            "rollout_direct_weight_load_wait_s": float(time.monotonic() - wait_started_at),
        }
        diagnostics.update(self._collect_rollout_direct_weight_load_metrics(results))
        self._last_sync_rollout_weight_diagnostics = diagnostics
        self._update_resize_execution_metrics(**diagnostics)
        return results

    async def _sync_rollout_weights_after_resize_async(
        self,
        *,
        clear_kv_cache: bool = True,
        handoff_path: str | None = None,
    ) -> None:
        group_name = getattr(self, "_active_weight_sync_group_name", "actor_rollout")
        use_direct_staged_rollout_load = (
            handoff_path is not None
            and self._handoff_staging_config.effective_backend() == "pinned_cpu"
            and bool(getattr(self._handoff_staging_config, "stage_optimizer", False))
        )
        use_alt_src = (
            not use_direct_staged_rollout_load
            and self._handoff_staging_config.effective_backend() == "pinned_cpu"
            and not bool(getattr(self._handoff_staging_config, "stage_optimizer", False))
            and len(getattr(self.actor_wg, "workers", [])) > 1
        )
        src_rank = 1 if use_alt_src else 0
        sync_started_at = time.monotonic()
        if use_direct_staged_rollout_load:
            self._load_rollout_weights_from_handoff_model_pages(handoff_path)
        else:
            self.sync_rollout_weights(src_rank=src_rank if use_alt_src else None)
        sync_duration = time.monotonic() - sync_started_at
        if not bool(getattr(self._handoff_staging_config, "optimizer_full_state_restore", True)):
            self._start_optimizer_preload_after_weight_sync()
        kv_clear_duration = 0.0
        if clear_kv_cache and self.async_rollout_manager is not None:
            kv_started_at = time.monotonic()
            await self.async_rollout_manager.clear_kv_cache_async()
            kv_clear_duration = time.monotonic() - kv_started_at
        metrics = {
            "resize/post_switch_weight_sync_s": sync_duration,
            "resize/post_switch_kv_clear_s": kv_clear_duration,
            "resize/post_switch_weight_sync_src_rank": float(src_rank),
            "resize/post_switch_weight_sync_alt_src": 1.0 if use_alt_src else 0.0,
            "resize/post_switch_weight_sync_direct_staged_load": 1.0 if use_direct_staged_rollout_load else 0.0,
            "resize/post_switch_weight_sync_path": "handoff_model_pages" if use_direct_staged_rollout_load else "collective_broadcast",
        }
        for key, value in getattr(self, "_last_sync_rollout_weight_diagnostics", {}).items():
            metrics[f"resize/post_switch_{key}"] = value
        self._update_resize_execution_metrics(**metrics)
    @staticmethod
    def _collect_optimizer_preload_start_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = ("started", "full_state_restore", "page_count", "queue_depth", "total_s")
        for field in numeric_fields:
            values = []
            for item in results:
                if isinstance(item, dict) and isinstance(item.get(field), (int, float)):
                    values.append(float(item[field]))
            if values:
                metrics[f"resize/optimizer_preload_start_{field}_max"] = max(values)
                metrics[f"resize/optimizer_preload_start_{field}_mean"] = sum(values) / len(values)
        return metrics

    def _start_optimizer_preload_after_weight_sync(self) -> None:
        if self._handoff_staging_config.effective_backend() != "pinned_cpu":
            return
        started_at = time.monotonic()
        try:
            with resize_trace_span(
                self._resize_trace_config,
                "optimizer_preload_start",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={},
            ):
                refs = self.actor_wg.execute_all_async(f"{str(Role.Actor)}_start_pending_optimizer_preload")
                results = ray.get(refs) if refs else []
        except Exception as exc:
            self._update_resize_execution_metrics(
                **{
                    "resize/optimizer_preload_start_s": float(time.monotonic() - started_at),
                    "resize/optimizer_preload_start_failed": 1.0,
                    "resize/optimizer_preload_start_error": repr(exc),
                }
            )
            raise
        metrics = self._collect_optimizer_preload_start_metrics(results)
        metrics["resize/optimizer_preload_start_s"] = float(time.monotonic() - started_at)
        metrics["resize/optimizer_preload_start_failed"] = 0.0
        self._update_resize_execution_metrics(**metrics)


    def _post_load_checkpoint_for_switch(self) -> None:
        self.sync_rollout_weights()
        if self.async_rollout_manager is not None:
            self.async_rollout_manager.clear_kv_cache()

    async def _post_load_checkpoint_for_switch_async(self) -> None:
        self.sync_rollout_weights()
        if self.async_rollout_manager is not None:
            await self.async_rollout_manager.clear_kv_cache_async()


    @staticmethod
    def _collect_sync_rollout_weight_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = (
            "total_s",
            "load_actor_gpu_s",
            "get_params_s",
            "prepare_inference_model_s",
            "init_collective_s",
            "alloc_s",
            "actor_copy_s",
            "actor_source_direct_count",
            "actor_source_direct_bytes",
            "broadcast_s",
            "rollout_load_weights_s",
            "empty_cache_s",
            "src_rank",
            "weight_bytes",
            "weight_count",
            "device_free_before_bytes",
            "device_total_before_bytes",
            "device_free_after_bytes",
            "device_total_after_bytes",
        )
        for role in ("actor", "rollout"):
            role_results = [item for item in results if isinstance(item, dict) and item.get("role") == role]
            if not role_results:
                continue
            metrics[f"sync_rollout_weights_{role}_worker_count"] = float(len(role_results))
            for field in numeric_fields:
                values = []
                for item in role_results:
                    value = item.get(field)
                    if isinstance(value, bool):
                        value = float(value)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                if values:
                    metrics[f"sync_rollout_weights_{role}_{field}_max"] = max(values)
                    metrics[f"sync_rollout_weights_{role}_{field}_mean"] = sum(values) / len(values)
            first = role_results[0]
            for field in ("group_name", "collective_world_size", "collective_initialized", "torch_world_size"):
                value = first.get(field)
                if isinstance(value, (int, float, bool)):
                    metrics[f"sync_rollout_weights_{role}_{field}"] = float(value)
                elif value is not None:
                    metrics[f"sync_rollout_weights_{role}_{field}"] = str(value)
        return metrics

    def sync_rollout_weights(self, src_rank: int | None = None, *, actor_first: bool = False):
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

        launch_started_at = time.monotonic()
        with resize_trace_span(
            self._resize_trace_config,
            "sync_rollout_weights",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"actor_workers": len(actor_workers), "rollout_workers": len(rollout_workers)},
        ):
            rollout_refs = None
            actor_refs = None
            rollout_launch_s = 0.0
            actor_launch_s = 0.0

            def _launch_actor_sync():
                if src_rank is None:
                    return self.actor_wg.sync_rollout_weights()
                return self.actor_wg.sync_rollout_weights(int(src_rank))

            def _launch_rollout_sync():
                if src_rank is None:
                    return self.rollout_wg.sync_rollout_weights()
                return self.rollout_wg.sync_rollout_weights(int(src_rank))

            if actor_first:
                actor_launch_started_at = time.monotonic()
                actor_refs = _launch_actor_sync()
                actor_launch_s = time.monotonic() - actor_launch_started_at
                rollout_launch_started_at = time.monotonic()
                rollout_refs = _launch_rollout_sync()
                rollout_launch_s = time.monotonic() - rollout_launch_started_at
            else:
                rollout_launch_started_at = time.monotonic()
                rollout_refs = _launch_rollout_sync()
                rollout_launch_s = time.monotonic() - rollout_launch_started_at
                actor_launch_started_at = time.monotonic()
                actor_refs = _launch_actor_sync()
                actor_launch_s = time.monotonic() - actor_launch_started_at
        if rollout_refs is None and actor_refs is None:
            self._last_sync_rollout_weight_diagnostics = {
                "sync_rollout_weights_launch_s": time.monotonic() - launch_started_at,
                "sync_rollout_weights_rollout_launch_s": rollout_launch_s,
                "sync_rollout_weights_actor_launch_s": actor_launch_s,
                "sync_rollout_weights_wait_s": 0.0,
            }
            self._update_resize_execution_metrics(**self._last_sync_rollout_weight_diagnostics)
            return []
        rollout_ref_list = []
        actor_ref_list = []
        if rollout_refs is not None:
            rollout_ref_list.extend(rollout_refs if isinstance(rollout_refs, list) else [rollout_refs])
        if actor_refs is not None:
            actor_ref_list.extend(actor_refs if isinstance(actor_refs, list) else [actor_refs])

        rollout_wait_started_at = time.monotonic()
        rollout_results = ray.get(rollout_ref_list) if rollout_ref_list else []
        rollout_wait_s = time.monotonic() - rollout_wait_started_at
        actor_wait_started_at = time.monotonic()
        actor_results = ray.get(actor_ref_list) if actor_ref_list else []
        actor_wait_s = time.monotonic() - actor_wait_started_at
        diagnostics = {
            "sync_rollout_weights_launch_s": time.monotonic() - launch_started_at - rollout_wait_s - actor_wait_s,
            "sync_rollout_weights_rollout_launch_s": rollout_launch_s,
            "sync_rollout_weights_actor_launch_s": actor_launch_s,
            "sync_rollout_weights_rollout_wait_s": rollout_wait_s,
            "sync_rollout_weights_actor_wait_s": actor_wait_s,
            "sync_rollout_weights_wait_s": rollout_wait_s + actor_wait_s,
        }
        diagnostics.update(self._collect_sync_rollout_weight_metrics(rollout_results + actor_results))
        self._last_sync_rollout_weight_diagnostics = diagnostics
        self._update_resize_execution_metrics(**diagnostics)
        return rollout_results + actor_results

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
        self._ensure_actor_training_group_binding()
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
            with resize_trace_span(
                self._resize_trace_config,
                "generate_async_background",
                step=self._trace_step(),
                lane="trainer_async",
                metadata={"batch_size": len(batch.batch)},
            ):
                gen_batch_output = await self.async_rollout_manager.generate_sequences_async(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        # Dynamic-resize can change the active actor/critic topology between
        # steps. Any downstream path that reasons about the current DP shape
        # (for example `_balance_batch`) must therefore see a batch that has
        # already been aligned to the active runtime topology. Keep this local
        # to the one-step-off trainer so the shared PPO path remains unchanged.
        batch = self._apply_runtime_batch_plan(batch)

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

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._ensure_actor_training_group_binding()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # after load checkpoint sync rollout weights
        self.sync_rollout_weights()
        await self.async_rollout_manager.clear_kv_cache_async()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
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
            self._ensure_actor_training_group_binding()
            self._reset_resize_execution_metrics()
            self._reset_communicator_cache_metrics()
            self._reset_resize_budget_metrics()
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps
            resize_scheduled_this_step = self._has_dynamic_resize_scheduled_at_step(self.global_steps)

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            with marked_timer("step", timing_raw):
                # wait for the previous batch
                with marked_timer("gen", timing_raw, color="red"):
                    with resize_trace_span(
                        self._resize_trace_config,
                        "generate_async_wait",
                        step=self.global_steps,
                        lane="trainer_main",
                        metadata={},
                    ):
                        _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
                        timing_raw.update(batch.meta_info["timing"])
                        timing_raw.update(_timing_raw)
                        metrics.update(_metrics)
                        batch.meta_info.pop("timing", None)

                # sync weights from actor to rollout only when we are about to
                # launch generation on the current rollout group. On a scheduled
                # resize step the old rollout group will be replaced before the
                # next generation, so this pre-train sync is pure overhead and
                # can also contend with resize staging/preload work.
                if not resize_scheduled_this_step:
                    with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                        self.sync_rollout_weights()
                        await self.async_rollout_manager.clear_kv_cache_async()
                else:
                    timing_raw["sync_rollout_weights"] = 0.0

                # async next generation
                if not is_last_step and not resize_scheduled_this_step:
                    batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                    await asyncio.sleep(0)
                elif not is_last_step:
                    logger.info(
                        "[one-step-off][resize][prefetch] defer next async batch until resize completes: step=%s",
                        self.global_steps,
                    )
                    batch_data_future = None

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

                    self._maybe_record_resize_history_batch(batch)

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
                        with resize_trace_span(
                            self._resize_trace_config,
                            "train_update_critic",
                            step=self.global_steps,
                            lane="trainer_train",
                            metadata={},
                        ):
                            critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                await asyncio.sleep(0)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with resize_trace_span(
                            self._resize_trace_config,
                            "train_update_actor",
                            step=self.global_steps,
                            lane="trainer_train",
                            metadata={},
                        ):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    resize_execution_actor_metrics = {
                        key: value
                        for key, value in actor_output_metrics.items()
                        if key in self._latest_resize_execution_metrics
                    }
                    if resize_execution_actor_metrics:
                        self._update_resize_execution_metrics(**resize_execution_actor_metrics)
                    async_optimizer_metrics = self._collect_deferred_optimizer_materialize_metrics(block=False)
                    if async_optimizer_metrics:
                        self._update_resize_execution_metrics(**async_optimizer_metrics)
                        metrics.update(async_optimizer_metrics)
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
            metrics.update(self._record_resize_control_observation(timing_raw=timing_raw, batch=batch))
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
            await self._maybe_dynamic_resize()
            metrics.update(self._latest_resize_control_metrics)
            metrics.update(self._latest_resize_execution_metrics)
            metrics.update(self._latest_communicator_cache_metrics)
            metrics.update(self._latest_resize_budget_metrics)

            if not is_last_step and batch_data_future is None:
                logger.info(
                    "[one-step-off][resize][prefetch] start deferred async batch after resize: step=%s",
                    self.global_steps,
                )
                batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                await asyncio.sleep(0)

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
            metrics.update(self._build_step_path_debug_metrics(timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            tracking_logger.log(data=metrics, step=self.global_steps)

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
