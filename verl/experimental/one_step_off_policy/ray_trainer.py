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
import logging
import os
import uuid
from contextlib import nullcontext
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.one_step_off_policy.resize_budget import (
    ResizeBudgetConfig,
    ResizeBudgetController,
    ResizeBudgetSnapshot,
)
from verl.experimental.one_step_off_policy.resize_controller import (
    ACTION_HOLD,
    ACTION_TO_CODE,
    ResizeController,
    ResizeControllerConfig,
)
from verl.experimental.one_step_off_policy.resize_metrics import build_resize_observation
from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    build_checkpoint_artifact_manifest,
    create_restore_session_manifest,
    finalize_restore_session_manifest,
    has_restore_session_manifest,
    read_restore_session_manifest,
    write_paged_state_manifest,
    write_host_staging_manifest,
)
from verl.experimental.one_step_off_policy.trace_utils import append_resize_trace, build_resize_trace_config
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.experimental.separation.utils import dynamic_resize_shared_pool_enabled
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager


logger = logging.getLogger(__name__)


class OneStepOffRayTrainer(SeparateRayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
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
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        # Skip rollout worker mapping and let agentloop create it.
        role_worker_mapping.pop(Role.Rollout, None)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================

        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}
        self._dynamic_resize_cfg = self._get_dynamic_resize_cfg()
        self._dynamic_resize_enabled = bool(self._dynamic_resize_cfg.get("enable", False))
        self._dynamic_resize_mode = str(self._dynamic_resize_cfg.get("mode", "schedule"))
        self._resize_controller = self._build_resize_controller()
        self._resize_budget_config = ResizeBudgetConfig.from_dict(self._dynamic_resize_cfg.get("budget_protection", {}))
        self._resize_budget_controller = ResizeBudgetController(self._resize_budget_config)
        self._host_staging_config = HostStagingConfig.from_dict(
            self._dynamic_resize_cfg.get("handoff", {"enable": False})
        )
        self._resize_trace_config = build_resize_trace_config(self.config)
        self._latest_resize_control_metrics: dict[str, float | str | int] = self._default_resize_control_metrics()
        self._latest_resize_budget_metrics: dict[str, float | str | int] = self._default_resize_budget_metrics()
        self._latest_resize_handoff_metrics: dict[str, float | str | int] = self._default_resize_handoff_metrics()

    @staticmethod
    def _cfg_to_plain_dict(cfg) -> dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            return OmegaConf.to_container(cfg, resolve=True) or {}
        except Exception:
            return dict(cfg) if hasattr(cfg, "items") else {}

    def _get_dynamic_resize_cfg(self) -> dict[str, Any]:
        return self._cfg_to_plain_dict(OmegaConf.select(self.config, "trainer.dynamic_resize"))

    def _build_resize_controller(self) -> ResizeController | None:
        if not self._dynamic_resize_enabled:
            return None
        if self._dynamic_resize_mode != "schedule_with_hysteresis":
            return None
        controller_cfg = ResizeControllerConfig.from_dict(self._dynamic_resize_cfg.get("hysteresis", {}))
        if not controller_cfg.enable:
            return None
        return ResizeController(controller_cfg)

    def _default_resize_control_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/enabled": 1.0 if self._dynamic_resize_enabled else 0.0,
            "resize/mode": self._dynamic_resize_mode,
            "resize/controller_enabled": 1.0 if self._resize_controller is not None else 0.0,
            "resize/rollout_train_ratio": 0.0,
            "resize/avg_rollout_time_s": 0.0,
            "resize/avg_train_time_s": 0.0,
            "resize/hysteresis_signal": ACTION_HOLD,
            "resize/hysteresis_signal_code": ACTION_TO_CODE[ACTION_HOLD],
            "resize/hysteresis_decision": ACTION_HOLD,
            "resize/hysteresis_decision_code": ACTION_TO_CODE[ACTION_HOLD],
            "resize/required_action": ACTION_HOLD,
            "resize/required_action_code": ACTION_TO_CODE[ACTION_HOLD],
            "resize/window_fill": 0,
            "resize/dwell_remaining": 0,
            "resize/cooldown_remaining": 0,
            "resize/gate_pass": -1.0,
            "resize/schedule_triggered": 0.0,
            "resize/schedule_applied": 0.0,
            "resize/pending_resource_switch": 0.0,
            "resize/unsupported_reason": "",
            "resize/hard_switch_enabled": 1.0 if self._dynamic_resize_hard_switch_enabled() else 0.0,
            "resize/active_actor_size": 0.0,
            "resize/active_rollout_size": 0.0,
            "resize/target_actor_size": 0.0,
            "resize/target_rollout_size": 0.0,
        }

    def _default_resize_budget_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/budget_enabled": 1.0 if self._resize_budget_config.enable else 0.0,
            "resize/budget_ratio": self._resize_budget_config.memory_budget_ratio,
            "resize/budget_blocked": 0.0,
            "resize/budget_reason": "",
            "resize/budget_effective_backend": self._host_staging_config.effective_backend(),
            "resize/budget_estimated_stage_bytes": 0.0,
            "resize/budget_estimated_host_peak_bytes": 0.0,
            "resize/budget_estimated_gpu_peak_bytes": 0.0,
        }

    def _default_resize_handoff_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/handoff_enabled": 1.0 if self._host_staging_config.enable else 0.0,
            "resize/handoff_backend": self._host_staging_config.effective_backend(),
            "resize/handoff_stage_optimizer": 1.0 if self._host_staging_config.stage_optimizer else 0.0,
            "resize/handoff_optimizer_restore_policy": self._host_staging_config.optimizer_restore_policy,
            "resize/handoff_preclear_rollout_kv_cache": 1.0
            if self._host_staging_config.preclear_rollout_kv_cache
            else 0.0,
            "resize/handoff_manifest_written": 0.0,
            "resize/handoff_restore_session_status": "",
            "resize/handoff_stage_dir": "",
            "resize/handoff_session_id": "",
            "resize/handoff_error": "",
            "resize/handoff_model_artifact_count": 0.0,
            "resize/handoff_optimizer_artifact_count": 0.0,
            "resize/handoff_staged_model_bytes": 0.0,
            "resize/handoff_staged_optimizer_bytes": 0.0,
        }

    def _build_resize_control_metrics(self, snapshot: dict[str, float | str | int]) -> dict[str, float | str | int]:
        metrics = self._default_resize_control_metrics()
        for key, value in snapshot.items():
            metrics[f"resize/{key}"] = value
        metrics["resize/enabled"] = 1.0 if self._dynamic_resize_enabled else 0.0
        metrics["resize/mode"] = self._dynamic_resize_mode
        metrics["resize/controller_enabled"] = 1.0 if self._resize_controller is not None else 0.0
        return metrics

    def _dynamic_resize_hard_switch_enabled(self) -> bool:
        hard_switch_cfg = self._cfg_to_plain_dict(self._dynamic_resize_cfg.get("hard_switch"))
        return bool(
            self._dynamic_resize_enabled
            and self._dynamic_resize_cfg.get("shared_pool", False)
            and hard_switch_cfg.get("enable", False)
        )

    def _dynamic_resize_item_for_current_step(self) -> dict[str, Any] | None:
        if not self._dynamic_resize_enabled:
            return None
        for item in self._normalize_dynamic_resize_schedule():
            if int(item.get("step", -1)) == self.global_steps:
                return item
        return None

    def _should_defer_next_rollout_for_resize(self) -> bool:
        return self._dynamic_resize_hard_switch_enabled() and self._dynamic_resize_item_for_current_step() is not None

    def _record_resize_control_observation(self, *, batch: DataProto) -> dict[str, float | str | int]:
        if not self._dynamic_resize_enabled:
            self._latest_resize_control_metrics = self._default_resize_control_metrics()
            return dict(self._latest_resize_control_metrics)

        observation = build_resize_observation(timing_raw=self.timing_raw, batch=batch)
        if self._resize_controller is not None:
            snapshot = self._resize_controller.observe(step=self.global_steps, observation=observation)
            metrics = self._build_resize_control_metrics(snapshot)
        else:
            rollout_time_s = observation["rollout_time_s"]
            train_time_s = observation["train_time_s"]
            metrics = self._default_resize_control_metrics()
            metrics.update(
                {
                    "resize/rollout_train_ratio": rollout_time_s / max(train_time_s, 1e-6),
                    "resize/avg_rollout_time_s": rollout_time_s,
                    "resize/avg_train_time_s": train_time_s,
                    "resize/gate_pass": 1.0,
                }
            )
        self._latest_resize_control_metrics = metrics
        append_resize_trace(
            self._resize_trace_config,
            {
                "event": "resize_observation",
                "step": self.global_steps,
                "observation": observation,
                "metrics": metrics,
            },
        )
        return dict(metrics)

    def _normalize_dynamic_resize_schedule(self) -> list[dict[str, Any]]:
        schedule = self._dynamic_resize_cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())
        if not isinstance(schedule, list):
            raise TypeError(f"trainer.dynamic_resize.schedule must be a list or dict, got {type(schedule).__name__}")
        normalized = []
        for item in schedule:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each trainer.dynamic_resize.schedule item must be a dict, "
                    f"got {type(item).__name__}: {item!r}"
                )
            normalized.append(item)
        return normalized

    @staticmethod
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

    def _active_actor_size(self) -> int:
        actor_wg = getattr(self, "actor_wg", None) or getattr(self, "actor_rollout_wg", None)
        if actor_wg is None:
            topology = getattr(self, "_active_dynamic_resize_topology", None) or {}
            if "actor_size" in topology:
                return int(topology["actor_size"])
            return 0
        world_size = getattr(actor_wg, "world_size", None)
        if world_size is not None:
            return int(world_size)
        return len(getattr(actor_wg, "workers", []) or [])

    def _active_rollout_size(self) -> int:
        manager = getattr(self, "llm_server_manager", None)
        if manager is None:
            topology = getattr(self, "_active_dynamic_resize_topology", None) or {}
            if "rollout_size" in topology:
                return int(topology["rollout_size"])
            return 0
        replicas = getattr(manager, "rollout_replicas", []) or []
        worker_count = sum(len(getattr(replica, "workers", []) or []) for replica in replicas)
        if worker_count > 0:
            return worker_count
        return len(getattr(manager, "server_handles", []) or [])

    def _dynamic_resize_active_topology_metrics(self) -> dict[str, float]:
        active_actor = float(self._active_actor_size())
        active_rollout = float(self._active_rollout_size())
        return {
            "resize/active_actor_size": active_actor,
            "resize/active_rollout_size": active_rollout,
            "resize/target_actor_size": active_actor,
            "resize/target_rollout_size": active_rollout,
        }

    def _dynamic_resize_budget_enabled(self) -> bool:
        return bool(getattr(getattr(self, "_resize_budget_config", None), "enable", False))

    def _estimate_dynamic_resize_model_bytes(self, item: dict[str, Any]) -> int:
        for key in ("estimated_stage_bytes", "estimated_model_bytes"):
            value = item.get(key)
            if value is not None:
                try:
                    return max(int(value), 0)
                except (TypeError, ValueError):
                    pass

        model_path = OmegaConf.select(self.config, "actor_rollout_ref.model.path")
        if not model_path or not os.path.exists(str(model_path)):
            return 0
        if os.path.isfile(str(model_path)):
            return os.path.getsize(str(model_path))

        total = 0
        for root, _, files in os.walk(str(model_path)):
            for file_name in files:
                if file_name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                    try:
                        total += os.path.getsize(os.path.join(root, file_name))
                    except OSError:
                        pass
        return int(total)

    def _get_dynamic_resize_resource_snapshot(self, worker_group, staging_path: str | None = None) -> ResizeBudgetSnapshot:
        if worker_group is None:
            return ResizeBudgetSnapshot()
        try:
            snapshot = worker_group.execute_rank_zero_sync("get_resize_resource_snapshot", staging_path)
        except Exception as exc:
            logger.warning("[one-step-off][resize][budget] failed to collect resource snapshot: %s", exc)
            return ResizeBudgetSnapshot()
        return ResizeBudgetSnapshot.from_dict(snapshot)

    def _gate_dynamic_resize_budget(self, item: dict[str, Any]) -> tuple[bool, dict[str, float | str | int]]:
        metrics = self._default_resize_budget_metrics()
        if not self._dynamic_resize_budget_enabled():
            return True, metrics

        model_bytes = self._estimate_dynamic_resize_model_bytes(item)
        stage_bytes = model_bytes
        stage_optimizer = bool(getattr(self._host_staging_config, "stage_optimizer", False))
        estimated_host_peak_bytes = int(item.get("estimated_host_peak_bytes") or model_bytes * (3.0 if stage_optimizer else 1.25))
        estimated_gpu_peak_bytes = int(item.get("estimated_gpu_peak_bytes") or model_bytes)
        metrics.update(
            {
                "resize/budget_estimated_stage_bytes": float(stage_bytes),
                "resize/budget_estimated_host_peak_bytes": float(estimated_host_peak_bytes),
                "resize/budget_estimated_gpu_peak_bytes": float(estimated_gpu_peak_bytes),
            }
        )

        actor_wg = getattr(self, "actor_wg", None) or getattr(self, "actor_rollout_wg", None)
        snapshot = self._get_dynamic_resize_resource_snapshot(actor_wg, self._dynamic_resize_checkpoint_step_folder())
        export_decision = self._resize_budget_controller.evaluate_export(
            requested_backend=self._host_staging_config.effective_backend(),
            snapshot=snapshot,
            estimated_host_peak_bytes=estimated_host_peak_bytes,
            estimated_stage_bytes=stage_bytes,
        )
        metrics.update(
            {
                "resize/budget_blocked": 1.0 if export_decision.blocked else 0.0,
                "resize/budget_reason": export_decision.reason,
                "resize/budget_effective_backend": export_decision.effective_backend,
            }
        )
        if not export_decision.allow_resize:
            return False, metrics

        restore_decision = self._resize_budget_controller.evaluate_restore(
            snapshot=snapshot,
            estimated_gpu_peak_bytes=estimated_gpu_peak_bytes,
        )
        metrics.update(
            {
                "resize/budget_blocked": 1.0 if restore_decision.blocked else 0.0,
                "resize/budget_reason": restore_decision.reason,
            }
        )
        return restore_decision.allow_resize, metrics

    def _gate_dynamic_resize_schedule_item(
        self, item: dict[str, Any]
    ) -> tuple[bool, str, dict[str, float | str | int]]:
        active_actor = self._active_actor_size()
        active_rollout = self._active_rollout_size()
        actor_target, rollout_target = self._dynamic_resize_target_sizes(
            item, active_actor=active_actor, active_rollout=active_rollout
        )

        if self._resize_controller is not None:
            required_action = self._resize_controller.infer_required_action(
                active_actor=active_actor,
                active_rollout=active_rollout,
                actor_target=actor_target,
                rollout_target=rollout_target,
            )
            gate_pass, snapshot = self._resize_controller.gate(step=self.global_steps, required_action=required_action)
            metrics = self._build_resize_control_metrics(snapshot)
        else:
            required_action = ACTION_HOLD
            actor_delta = actor_target - active_actor
            rollout_delta = rollout_target - active_rollout
            if rollout_delta > 0 and actor_delta < 0:
                required_action = "expand_rollout"
            elif actor_delta > 0 and rollout_delta < 0:
                required_action = "expand_train"
            gate_pass = True
            metrics = dict(self._latest_resize_control_metrics)
            metrics.update(
                {
                    "resize/required_action": required_action,
                    "resize/required_action_code": ACTION_TO_CODE.get(required_action, 0.0),
                    "resize/gate_pass": 1.0,
                }
            )

        metrics.update(
            {
                "resize/schedule_triggered": 1.0,
                "resize/active_actor_size": float(active_actor),
                "resize/active_rollout_size": float(active_rollout),
                "resize/target_actor_size": float(actor_target),
                "resize/target_rollout_size": float(rollout_target),
            }
        )
        if gate_pass and required_action != ACTION_HOLD:
            budget_pass, budget_metrics = self._gate_dynamic_resize_budget(item)
            metrics.update(budget_metrics)
            if not budget_pass:
                gate_pass = False
                metrics["resize/gate_pass"] = 0.0
        return gate_pass, required_action, metrics

    def _dynamic_resize_target_sizes(
        self, item: dict[str, Any], *, active_actor: int | None = None, active_rollout: int | None = None
    ) -> tuple[int, int]:
        actor_target = self._pool_target_size(item.get("actor_pool"))
        rollout_target = self._pool_target_size(item.get("rollout_pool"))
        if active_actor is None:
            active_actor = self._active_actor_size()
        if active_rollout is None:
            active_rollout = self._active_rollout_size()
        actor_target = active_actor if actor_target is None else actor_target
        rollout_target = active_rollout if rollout_target is None else rollout_target

        topology = getattr(self.resource_pool_manager, "dynamic_resize_topology", None)
        if topology is not None and actor_target + rollout_target != topology.total_size:
            raise ValueError(
                "dynamic resize target actor_pool + rollout_pool must equal shared pool size "
                f"{topology.total_size}, got actor={actor_target}, rollout={rollout_target}"
            )
        return int(actor_target), int(rollout_target)

    def _dynamic_resize_checkpoint_step_folder(self) -> str:
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        return os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")

    @staticmethod
    def _dynamic_resize_handoff_stage_dir(global_step_folder: str) -> str:
        return os.path.join(global_step_folder, "dynamic_resize_handoff")

    @staticmethod
    def _checkpoint_artifact_file_names(checkpoint_dir: str, prefixes: tuple[str, ...]) -> list[str]:
        if not os.path.isdir(checkpoint_dir):
            return []
        names = []
        for file_name in os.listdir(checkpoint_dir):
            if any(file_name.startswith(prefix) for prefix in prefixes):
                names.append(file_name)
        return sorted(names)

    def _prepare_dynamic_resize_handoff_session(self, global_step_folder: str) -> dict[str, float | str | int]:
        metrics = self._default_resize_handoff_metrics()
        if not self._host_staging_config.enable:
            return metrics

        stage_dir = self._dynamic_resize_handoff_stage_dir(global_step_folder)
        actor_checkpoint_dir = os.path.join(global_step_folder, "actor")
        session_id = f"dynamic_resize_step_{self.global_steps}_{uuid.uuid4().hex}"
        try:
            write_host_staging_manifest(stage_dir, self._host_staging_config)
            model_manifest = build_checkpoint_artifact_manifest(
                actor_checkpoint_dir,
                prefix="checkpoint_model",
                file_names=self._checkpoint_artifact_file_names(
                    actor_checkpoint_dir, ("dynamic_resize_full_model.pt", "model_world_size_")
                ),
            )
            optimizer_manifest = build_checkpoint_artifact_manifest(
                actor_checkpoint_dir,
                prefix="checkpoint_optimizer",
                file_names=self._checkpoint_artifact_file_names(actor_checkpoint_dir, ("optim_world_size_",)),
            )
            write_paged_state_manifest(stage_dir, "checkpoint_model", model_manifest)
            write_paged_state_manifest(stage_dir, "checkpoint_optimizer", optimizer_manifest)
            manifest = create_restore_session_manifest(
                stage_dir,
                backend=self._host_staging_config.effective_backend(),
                service_name=self._host_staging_config.service_name,
                session_id=session_id,
                optimizer_restore_policy=self._host_staging_config.optimizer_restore_policy,
                model_manifest=model_manifest,
                optimizer_manifest=optimizer_manifest,
            )
        except Exception as exc:
            logger.warning("[one-step-off][resize][handoff] failed to write handoff manifest: %s", exc)
            metrics.update(
                {
                    "resize/handoff_manifest_written": 0.0,
                    "resize/handoff_stage_dir": stage_dir,
                    "resize/handoff_session_id": session_id,
                    "resize/handoff_error": str(exc),
                }
            )
            return metrics

        metrics.update(
            {
                "resize/handoff_manifest_written": 1.0,
                "resize/handoff_restore_session_status": manifest.get("status", ""),
                "resize/handoff_stage_dir": stage_dir,
                "resize/handoff_session_id": manifest.get("session_id", session_id),
                "resize/handoff_error": "",
                "resize/handoff_model_artifact_count": float(manifest.get("model_page_count", 0) or 0),
                "resize/handoff_optimizer_artifact_count": float(manifest.get("optimizer_page_count", 0) or 0),
                "resize/handoff_staged_model_bytes": float(manifest.get("staged_model_bytes", 0) or 0),
                "resize/handoff_staged_optimizer_bytes": float(manifest.get("staged_optimizer_bytes", 0) or 0),
            }
        )
        return metrics

    def _complete_dynamic_resize_handoff_session(self, global_step_folder: str) -> dict[str, float | str | int]:
        metrics = self._default_resize_handoff_metrics()
        if not self._host_staging_config.enable:
            return metrics

        stage_dir = self._dynamic_resize_handoff_stage_dir(global_step_folder)
        if not has_restore_session_manifest(stage_dir):
            metrics.update(
                {
                    "resize/handoff_stage_dir": stage_dir,
                    "resize/handoff_error": "restore_session_manifest_missing",
                }
            )
            return metrics

        try:
            manifest = finalize_restore_session_manifest(stage_dir)
        except Exception as exc:
            logger.warning("[one-step-off][resize][handoff] failed to finalize handoff manifest: %s", exc)
            try:
                manifest = read_restore_session_manifest(stage_dir)
            except Exception:
                manifest = {}
            metrics.update(
                {
                    "resize/handoff_manifest_written": 1.0,
                    "resize/handoff_restore_session_status": manifest.get("status", ""),
                    "resize/handoff_stage_dir": stage_dir,
                    "resize/handoff_session_id": manifest.get("session_id", ""),
                    "resize/handoff_error": str(exc),
                    "resize/handoff_model_artifact_count": float(manifest.get("model_page_count", 0) or 0),
                    "resize/handoff_optimizer_artifact_count": float(manifest.get("optimizer_page_count", 0) or 0),
                    "resize/handoff_staged_model_bytes": float(manifest.get("staged_model_bytes", 0) or 0),
                    "resize/handoff_staged_optimizer_bytes": float(manifest.get("staged_optimizer_bytes", 0) or 0),
                }
            )
            return metrics

        metrics.update(
            {
                "resize/handoff_manifest_written": 1.0,
                "resize/handoff_restore_session_status": manifest.get("status", ""),
                "resize/handoff_stage_dir": stage_dir,
                "resize/handoff_session_id": manifest.get("session_id", ""),
                "resize/handoff_error": "",
                "resize/handoff_model_artifact_count": float(manifest.get("model_page_count", 0) or 0),
                "resize/handoff_optimizer_artifact_count": float(manifest.get("optimizer_page_count", 0) or 0),
                "resize/handoff_staged_model_bytes": float(manifest.get("staged_model_bytes", 0) or 0),
                "resize/handoff_staged_optimizer_bytes": float(manifest.get("staged_optimizer_bytes", 0) or 0),
            }
        )
        return metrics

    def _load_dynamic_resize_checkpoint(self, global_step_folder: str) -> None:
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=False)
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=False)

    def _split_dynamic_resize_resource_pools(self, *, actor_size: int, rollout_size: int) -> None:
        topology = getattr(self.resource_pool_manager, "dynamic_resize_topology", None)
        if topology is None:
            raise ValueError("dynamic resize hard switch requires shared pool topology")
        if actor_size + rollout_size != topology.total_size:
            raise ValueError(
                f"dynamic resize split must sum to {topology.total_size}, got actor={actor_size}, rollout={rollout_size}"
            )

        shared_pool = self.resource_pool_manager.resource_pool_dict[topology.shared_pool_name]
        actor_pool, rollout_pool = split_resource_pool(shared_pool, [actor_size, rollout_size])
        self.resource_pool_manager.resource_pool_dict["dynamic_resize_actor_pool"] = actor_pool
        self.resource_pool_manager.resource_pool_dict["dynamic_resize_rollout_pool"] = rollout_pool
        for role in [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]:
            if role in self.resource_pool_manager.mapping:
                self.resource_pool_manager.mapping[role] = "dynamic_resize_actor_pool"
        self._dynamic_resize_rollout_resource_pool = rollout_pool
        self._active_dynamic_resize_topology = {"actor_size": actor_size, "rollout_size": rollout_size}

    @staticmethod
    def _kill_ray_actors(handles) -> None:
        for handle in list(handles or []):
            try:
                ray.kill(handle, no_restart=True)
            except Exception as exc:
                logger.debug("[one-step-off][resize] ignoring Ray actor kill failure: %s", exc)

    async def _destroy_dynamic_resize_runtime(self) -> None:
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            try:
                await checkpoint_manager.abort_replicas()
                await checkpoint_manager.sleep_replicas()
            except Exception as exc:
                logger.warning("[one-step-off][resize] failed to quiesce rollout replicas before rebuild: %s", exc)

        async_rollout_manager = getattr(self, "async_rollout_manager", None)
        if async_rollout_manager is not None:
            self._kill_ray_actors(getattr(async_rollout_manager, "agent_loop_workers", []))

        llm_server_manager = getattr(self, "llm_server_manager", None)
        if llm_server_manager is not None:
            for replica in getattr(llm_server_manager, "rollout_replicas", []) or []:
                self._kill_ray_actors(getattr(replica, "workers", []))
            self._kill_ray_actors(getattr(llm_server_manager, "server_handles", []))
            load_balancer = getattr(llm_server_manager, "global_load_balancer", None)
            if load_balancer is not None:
                self._kill_ray_actors([load_balancer])

        for worker_group in getattr(self, "all_wg", {}).values():
            self._kill_ray_actors(getattr(worker_group, "workers", []))

    def _build_dynamic_resize_checkpoint_manager(self) -> None:
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager

        self.checkpoint_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

    def _refresh_dynamic_resize_checkpoint_group_name(self) -> None:
        checkpoint_engine_config = self.config.actor_rollout_ref.rollout.checkpoint_engine
        if checkpoint_engine_config.get("backend") not in {"nccl", "hccl"}:
            return
        backend = checkpoint_engine_config.get("backend")
        engine_kwargs = checkpoint_engine_config.get("engine_kwargs")
        if engine_kwargs is None:
            engine_kwargs = {}
            checkpoint_engine_config.engine_kwargs = engine_kwargs
        with open_dict(engine_kwargs):
            backend_kwargs = engine_kwargs.get(backend)
            if backend_kwargs is None:
                backend_kwargs = {}
                engine_kwargs[backend] = backend_kwargs
            backend_kwargs_context = open_dict(backend_kwargs) if isinstance(backend_kwargs, DictConfig) else nullcontext()
            with backend_kwargs_context:
                backend_kwargs["group_name"] = f"dynamic_resize_{self.global_steps}_{uuid.uuid4().hex}"
            engine_kwargs[backend] = backend_kwargs

    def _rebuild_dynamic_resize_runtime(self, *, actor_size: int, rollout_size: int) -> None:
        self._split_dynamic_resize_resource_pools(actor_size=actor_size, rollout_size=rollout_size)
        self._refresh_dynamic_resize_checkpoint_group_name()
        mapped_pool_names = set(self.resource_pool_manager.mapping.values())
        self.resource_pool_to_cls = {
            pool: {}
            for pool_name, pool in self.resource_pool_manager.resource_pool_dict.items()
            if pool_name in mapped_pool_names
        }
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_async_rollout_manager()
        self._build_dynamic_resize_checkpoint_manager()

    async def _apply_dynamic_resize_hard_switch(
        self, *, item: dict[str, Any], required_action: str
    ) -> dict[str, float | str | int]:
        actor_target, rollout_target = self._dynamic_resize_target_sizes(item)
        metrics: dict[str, float | str | int] = {
            "resize/hard_switch_enabled": 1.0,
            "resize/hard_switch_attempted": 1.0,
            "resize/hard_switch_success": 0.0,
        }

        if not dynamic_resize_shared_pool_enabled(self.config):
            reason = "shared_pool_required_for_hard_switch"
            metrics.update({"resize/unsupported_reason": reason, "resize/pending_resource_switch": 1.0})
            return metrics

        checkpoint_step_folder = self._dynamic_resize_checkpoint_step_folder()
        with marked_timer("dynamic_resize_save_checkpoint", self.timing_raw, color="green"):
            self._save_checkpoint(save_full_model_for_dynamic_resize=True)
            metrics.update(self._prepare_dynamic_resize_handoff_session(checkpoint_step_folder))

        await self._destroy_dynamic_resize_runtime()
        with marked_timer("dynamic_resize_rebuild_runtime", self.timing_raw, color="green"):
            self._rebuild_dynamic_resize_runtime(actor_size=actor_target, rollout_size=rollout_target)
            self._load_dynamic_resize_checkpoint(checkpoint_step_folder)
            metrics.update(self._complete_dynamic_resize_handoff_session(checkpoint_step_folder))

        with marked_timer("dynamic_resize_update_weights", self.timing_raw, color="green"):
            await self.checkpoint_manager.update_weights(self.global_steps)

        metrics.update(
            {
                "resize/schedule_applied": 1.0,
                "resize/pending_resource_switch": 0.0,
                "resize/hard_switch_success": 1.0,
                "resize/active_actor_size": float(actor_target),
                "resize/active_rollout_size": float(rollout_target),
                "resize/target_actor_size": float(actor_target),
                "resize/target_rollout_size": float(rollout_target),
                "resize/unsupported_reason": "",
            }
        )
        if self._resize_controller is not None:
            snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
            metrics.update(self._build_resize_control_metrics(snapshot))
        return metrics

    async def _maybe_dynamic_resize(self) -> dict[str, float | str | int]:
        if not self._dynamic_resize_enabled:
            return {}

        metrics = {
            **getattr(self, "_latest_resize_control_metrics", {}),
            **getattr(self, "_latest_resize_budget_metrics", {}),
            **getattr(self, "_latest_resize_handoff_metrics", {}),
        }
        schedule_seen = False
        for item in self._normalize_dynamic_resize_schedule():
            if int(item.get("step", -1)) != self.global_steps:
                continue
            schedule_seen = True

            gate_pass, required_action, gate_metrics = self._gate_dynamic_resize_schedule_item(item)
            metrics.update(gate_metrics)
            if not gate_pass:
                logger.info(
                    "[one-step-off][resize][gate] skip scheduled resize: step=%s required_action=%s",
                    self.global_steps,
                    required_action,
                )
                continue

            if required_action == ACTION_HOLD:
                metrics.update({"resize/schedule_applied": 1.0, "resize/pending_resource_switch": 0.0})
                if self._resize_controller is not None:
                    snapshot = self._resize_controller.mark_resize_applied(
                        step=self.global_steps, action=required_action
                    )
                    metrics.update(self._build_resize_control_metrics(snapshot))
                continue

            if self._dynamic_resize_hard_switch_enabled():
                metrics.update(
                    await self._apply_dynamic_resize_hard_switch(item=item, required_action=required_action)
                )
                continue

            reason = "dynamic_resize_hard_switch_disabled"
            metrics.update(
                {
                    "resize/schedule_applied": 0.0,
                    "resize/pending_resource_switch": 1.0,
                    "resize/unsupported_reason": reason,
                }
            )
            logger.warning(
                "[one-step-off][resize] schedule passed gate at step=%s but resource switching is deferred: %s",
                self.global_steps,
                reason,
            )

        if not schedule_seen:
            metrics.update(self._dynamic_resize_active_topology_metrics())
        self._latest_resize_control_metrics = metrics
        append_resize_trace(
            self._resize_trace_config,
            {"event": "resize_schedule", "step": self.global_steps, "metrics": metrics},
        )
        self._latest_resize_budget_metrics = {
            key: value for key, value in metrics.items() if key.startswith("resize/budget_")
        }
        self._latest_resize_handoff_metrics = {
            key: value for key, value in metrics.items() if key.startswith("resize/handoff_")
        }
        return dict(metrics)

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()
        self._dynamic_resize_rollout_resource_pool = None
        if dynamic_resize_shared_pool_enabled(self.config):
            topology = getattr(self.resource_pool_manager, "dynamic_resize_topology", None)
            if topology is None:
                raise ValueError("trainer.dynamic_resize.shared_pool requires dynamic resize pool topology")
            shared_pool = self.resource_pool_manager.resource_pool_dict[topology.shared_pool_name]
            actor_pool, rollout_pool = split_resource_pool(
                shared_pool,
                [topology.initial_actor_size, topology.initial_rollout_size],
            )
            self.resource_pool_manager.resource_pool_dict["dynamic_resize_actor_pool"] = actor_pool
            self.resource_pool_manager.resource_pool_dict["dynamic_resize_rollout_pool"] = rollout_pool
            for role in [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]:
                if role in self.resource_pool_manager.mapping:
                    self.resource_pool_manager.mapping[role] = "dynamic_resize_actor_pool"
            self._dynamic_resize_rollout_resource_pool = rollout_pool
            self._active_dynamic_resize_topology = {
                "actor_size": topology.initial_actor_size,
                "rollout_size": topology.initial_rollout_size,
            }

        mapped_pool_names = set(self.resource_pool_manager.mapping.values())
        self.resource_pool_to_cls = {
            pool: {}
            for pool_name, pool in self.resource_pool_manager.resource_pool_dict.items()
            if pool_name in mapped_pool_names
        }

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

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
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self):
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            rollout_resource_pool=getattr(self, "_dynamic_resize_rollout_resource_pool", None),
        )
        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

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

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)

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

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        reward_tensor, reward_extra_info = extract_reward(batch)
        return reward_tensor, reward_extra_info

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return

    async def fit_step(self, batch_data_future, continuous_iterator):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_data_future: batch future
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_ref_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_advantage(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_actor(batch)
            await asyncio.sleep(0)
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self.metrics.update(self._record_resize_control_observation(batch=batch))
        self._fit_experimental(batch)
        self.metrics.update(await self._maybe_dynamic_resize())
        if batch_data_future is None and not self.is_last_step:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        # sync weights from actor to rollout
        with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
            self._fit_update_weights()

        # async next generation
        if not self.is_last_step and not self._should_defer_next_rollout_for_resize():
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future
