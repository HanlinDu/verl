# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
import os
import datetime
import math
import shutil
import gc
import queue
import threading
import time
from collections import deque
from dataclasses import fields
from typing import Any

import psutil
import ray
import torch
import torch.distributed
from codetiming import Timer
from packaging import version
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from ray.util.collective import collective
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.experimental.one_step_off_policy.distributed_utils import vllm_stateless_init_process_group
from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    _estimate_object_bytes,
    create_restore_session_manifest,
    finalize_restore_session_manifest,
    has_paged_state_dict,
    has_restore_session_manifest,
    iter_paged_state_dict,
    load_paged_optimizer_state_dict,
    probe_local_pin_memory_capability,
    record_restore_progress,
    read_host_staging_manifest,
    read_restore_session_manifest,
    save_paged_state_dict,
    save_paged_optimizer_state_dict,
    read_paged_state_manifest,
    update_restore_session_manifest,
    write_host_staging_manifest,
)
from verl.experimental.one_step_off_policy.trace_utils import build_resize_trace_config, resize_trace_span
from verl.experimental.one_step_off_policy.worker_init_plan import WorkerCommInitPlan, build_worker_comm_init_plan
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    get_fsdp_full_state_dict,
    fsdp_version,
    load_fsdp_optimizer,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.ray_utils import get_event_loop
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import create_device_mesh
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
    CriticWorker,
    FSDPCheckpointManager,
    FSDPEngineConfig,
    RewardModelWorker,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.config.actor import ActorConfig, PolicyLossConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker", "RewardModelWorker"]


_ASYNC_PRELOAD_SENTINEL = object()


def _get_reshardable_optimizer_state_dict_api():
    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )
    else:
        from verl.third_party.torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )

    return StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict


def _get_reshardable_model_state_dict_api():
    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:
        from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    return StateDictOptions, set_model_state_dict


def _to_builtin_python_container(obj):
    if isinstance(obj, DictConfig | ListConfig):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, list | tuple):
        converted = [_to_builtin_python_container(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, dict):
        return {key: _to_builtin_python_container(value) for key, value in obj.items()}
    return obj


def _sanitize_optimizer_container_values(optimizer) -> None:
    if optimizer is None:
        return

    optimizer.defaults = _to_builtin_python_container(dict(optimizer.defaults))
    for group in optimizer.param_groups:
        for key, value in list(group.items()):
            if key == "params":
                continue
            group[key] = _to_builtin_python_container(value)
    for state_key, state_value in list(optimizer.state.items()):
        optimizer.state[state_key] = _to_builtin_python_container(state_value)


def _assert_rank0_local_pin_memory_available() -> None:
    available, error = probe_local_pin_memory_capability()
    if not available:
        raise RuntimeError(
            "pinned_cpu backend requires pin_memory() support in the restore worker process; "
            f"error={error}"
        )


def _iter_local_pinned_pages(local_path: str, prefix: str):
    yield from iter_paged_state_dict(local_path, prefix, pin_memory=True, prefetch=True)


def _move_tree_to_device(value: Any, *, device: torch.device | int | str, non_blocking: bool = True) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(item, device=device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_device(item, device=device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(item, device=device, non_blocking=non_blocking) for item in value)
    return value


def _try_preload_page_to_device(
    page: dict[str, Any],
    *,
    device_preload_threshold: float,
    preload_stream,
) -> tuple[dict[str, Any], object | None]:
    if device_preload_threshold <= 0.0 or preload_stream is None:
        return page, None

    free_bytes, total_bytes = _get_available_device_memory_bytes()
    if total_bytes is None:
        return page, None

    threshold_bytes = int(total_bytes * device_preload_threshold)
    reserved_bytes = int(get_torch_device().memory_reserved())
    estimated_page_bytes = int(_estimate_object_bytes(page))
    if reserved_bytes + estimated_page_bytes > threshold_bytes:
        return page, None

    preload_stream.wait_stream(get_torch_device().current_stream())
    with get_torch_device().stream(preload_stream):
        device_page = _move_tree_to_device(page, device=get_torch_device().current_device(), non_blocking=True)
        event = get_torch_device().Event()
        preload_stream.record_event(event)
    return device_page, event


def _build_streaming_page_iterator(
    local_path: str,
    prefix: str,
    *,
    device_preload_threshold: float,
    preload_queue_depth: int,
):
    page_iter = iter_paged_state_dict(local_path, prefix, pin_memory=True, prefetch=True)
    pending_pages: deque[tuple[dict[str, Any], object | None]] = deque()
    preload_stream = get_torch_device().Stream() if device_preload_threshold > 0.0 and torch.cuda.is_available() else None

    def _fill_pending_pages() -> None:
        while len(pending_pages) < max(int(preload_queue_depth), 1):
            try:
                next_page = next(page_iter)
            except StopIteration:
                return
            pending_pages.append(
                _try_preload_page_to_device(
                    next_page,
                    device_preload_threshold=device_preload_threshold,
                    preload_stream=preload_stream,
                )
            )

    _fill_pending_pages()
    while pending_pages:
        page, ready_event = pending_pages.popleft()
        if ready_event is not None:
            get_torch_device().current_stream().wait_event(ready_event)
        yield page
        _fill_pending_pages()


def _sanitize_actor_config_for_dataclass(actor_cfg):
    actor_cfg_container = OmegaConf.to_container(actor_cfg, resolve=False)
    if not isinstance(actor_cfg_container, dict):
        return actor_cfg, None

    actor_field_names = {field.name for field in fields(ActorConfig)}
    sanitized_actor_cfg = {key: value for key, value in actor_cfg_container.items() if key in actor_field_names}
    rollout_correction_cfg = None

    policy_loss_cfg = sanitized_actor_cfg.get("policy_loss")
    if isinstance(policy_loss_cfg, dict):
        allowed_policy_loss_fields = {field.name for field in fields(PolicyLossConfig)}
        dropped_policy_loss_keys = sorted(set(policy_loss_cfg.keys()) - allowed_policy_loss_fields)
        if dropped_policy_loss_keys:
            logger.debug(
                "[one-step-off][worker] sanitize actor.policy_loss before dataclass instantiate: "
                f"dropped_keys={dropped_policy_loss_keys}"
            )
        rollout_correction_cfg = policy_loss_cfg.get("rollout_correction")
        sanitized_actor_cfg["policy_loss"] = {
            key: value for key, value in policy_loss_cfg.items() if key in allowed_policy_loss_fields
        }

    return OmegaConf.create(sanitized_actor_cfg), rollout_correction_cfg


def _get_available_host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except (AttributeError, OSError, ValueError):
        return None


def _get_available_device_memory_bytes() -> tuple[int | None, int | None]:
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(get_torch_device().current_device())
            return int(free_bytes), int(total_bytes)
    except Exception:
        pass
    return None, None


class LocalInitActorRolloutRefWorker(Worker, DistProfilerExtension):
    """One-step-off本地实验版 worker 基类。

    设计目标：
    1. 避免直接修改主线 `verl.workers.fsdp_workers.ActorRolloutRefWorker`
    2. 在 `dynamic_resize.enable` 打开时，允许 one-step-off 独立重构 `__init__` / `init_model`
    3. 在当前阶段保持行为与主线尽量一致，仅把入口本地化
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        self.role = role
        self._dynamic_resize_enabled = bool(getattr(getattr(config, "dynamic_resize", None), "enable", False))
        self._resize_trace_config = build_resize_trace_config(config)

        # 这些字段属于“安全基础成员”：
        # - 仅依赖 config/role 的静态判定
        # - 或者作为后续 prepare/commit 前的空占位
        # 不会触发 distributed / device mesh / 真正模型构建。
        self._is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = role in ["ref", "actor_rollout_ref"]
        self._lora_rank = config.model.get("lora_rank", 0)
        self._is_lora = config.model.get("lora_adapter_path") is not None or self._lora_rank > 0
        self.use_orig_params = config.actor.fsdp_config.get("use_orig_params", False)
        self._is_offload_param = False
        self._is_offload_optimizer = False

        # 以下对象会在 commit 阶段绑定；这里先显式占位，避免早期访问出现 AttributeError。
        self.device_mesh = None
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = config.actor.get("ulysses_sequence_parallel_size", 1)
        self.ulysses_sharding_manager = None
        self.processor = None
        self.tokenizer = None
        self.actor = None
        self.actor_module = None
        self.actor_module_fsdp = None
        self.actor_optimizer = None
        self.actor_lr_scheduler = None
        self.actor_model_config = None
        self.ref_module_fsdp = None
        self.ref_policy = None
        self.rollout = None
        self.rollout_device_mesh = None
        self.checkpoint_manager = None
        self.flops_counter = None
        self._pending_optimizer_restore_path: str | None = None
        self._pending_optimizer_restore_page_count = 0
        self._pending_optimizer_restore_progressive_swap = False
        self._pending_optimizer_restore_cleanup_after_load = False
        self._pending_optimizer_restore_policy = "immediate"
        self._pending_optimizer_restore_materialize_count = 0
        self._pending_optimizer_restore_backend = "disk_fallback"
        self._pending_optimizer_preload_queue_depth = 1
        self._pending_optimizer_device_preload_threshold = 0.0
        self._pending_optimizer_async_preload_enabled = False
        self._pending_optimizer_preload_session: dict[str, Any] | None = None
        self._pinned_cpu_staging_service_cache: dict[str, object] = {}

        self._worker_init_plan = None
        self._worker_init_prepared = False
        self._worker_init_committed = False
        self._model_init_plan = None
        self._model_init_prepared = False
        self._model_init_committed = False

        if self._dynamic_resize_enabled:
            logger.debug(
                f"[one-step-off][resize] dynamic_resize enabled, keep __init__ minimal and defer worker/model init "
                f"for role {role}"
            )
        else:
            logger.warning("[one-step-off][resize] dynamic_resize disabled")
            self._fallback_init_worker_state(config=config, role=role, **kwargs)

    def _fallback_init_worker_state(self, config: DictConfig, role: str, **kwargs):
        """默认仍复用主线实现，确保非 dynamic_resize 行为不变。"""
        ActorRolloutRefWorker.__init__(self, config, role, **kwargs)

    def _prepare_local_init_worker_state(self, config: DictConfig, role: str, **kwargs):
        """准备本地 worker state 初始化上下文。

        这里尽量只做：
        - 角色与静态开关判定
        - profiler 配置选择
        - 一些纯配置的预解析

        设计目标是为后续“提前执行”铺路：这些内容理论上可以先准备，
        不一定要卡在最终的资源切换窗口里。

        当前阶段为了保持行为稳定，真正依赖 distributed world / device mesh
        的内容仍放在 commit 阶段执行。
        """
        if role not in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]:
            raise ValueError(
                f"Invalid role {role}, should be one of ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )

        is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        is_ref = role in ["ref", "actor_rollout_ref"]
        logger.debug(
            "[one-step-off][worker] _prepare_local_init_worker_state roles resolved: "
            f"role={role}, is_actor={is_actor}, is_rollout={is_rollout}, is_ref={is_ref}"
        )

        if is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif is_rollout:
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})
        logger.debug(
            "[one-step-off][worker] _prepare_local_init_worker_state profiler selected: "
            f"role={role}, tool={omega_profiler_config.get('tool', None)}"
        )

        init_method = None
        master_addr = getattr(self, "_master_addr", None)
        master_port = getattr(self, "_master_port", None)
        if master_addr is not None and master_port is not None:
            normalized_master_addr = str(master_addr).replace("[", "").replace("]", "")
            init_method = f"tcp://{normalized_master_addr}:{master_port}"

        worker_comm_plan = build_worker_comm_init_plan(
            config,
            role,
            rank=getattr(self, "rank", None),
            world_size=getattr(self, "world_size", None),
            init_method=init_method,
            device_type=get_device_name(),
            nccl_backend=get_nccl_backend(),
        )

        plan = {
            "config": config,
            "role": role,
            "kwargs": kwargs,
            "is_actor": is_actor,
            "is_rollout": is_rollout,
            "is_ref": is_ref,
            "omega_profiler_config": omega_profiler_config,
            # Prepare now records the communication contract as pure metadata so
            # commit can execute against an explicit plan instead of re-reading
            # environment/config ad hoc inside the critical window.
            "worker_comm_plan": worker_comm_plan.to_dict(),
        }
        return plan

    def _commit_local_init_worker_state(self, worker_ctx):
        """提交本地 worker state 初始化。

        当前先完整镜像主线初始化语义，把 one-step-off 的初始化入口收拢到实验路径；
        后续再继续把适合提前执行的内容从 commit 搬到 prepare。

        这里保留必须按顺序执行的初始化：
        - process group 初始化
        - device mesh / ulysses mesh 构建
        - dispatch collect info 注册
        - profiler 实例绑定到 self
        - 依赖 world size 的 config normalization
        """
        import torch.distributed
        from torch.distributed.device_mesh import init_device_mesh

        config = worker_ctx["config"]
        role = worker_ctx["role"]

        self.role = role
        self._is_actor = worker_ctx["is_actor"]
        self._is_rollout = worker_ctx["is_rollout"]
        self._is_ref = worker_ctx["is_ref"]

        comm_plan = WorkerCommInitPlan.from_dict(worker_ctx["worker_comm_plan"])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=comm_plan.backend,
                rank=comm_plan.rank,
                world_size=comm_plan.world_size,
                timeout=datetime.timedelta(seconds=comm_plan.timeout_seconds),
                init_method=comm_plan.init_method,
            )

        runtime_world_size = torch.distributed.get_world_size()
        if runtime_world_size != comm_plan.world_size:
            logger.warning(
                "[one-step-off][worker] runtime world_size differs from prepared comm plan: "
                "role=%s prepared_world_size=%s runtime_world_size=%s",
                role,
                comm_plan.world_size,
                runtime_world_size,
            )
        world_size = runtime_world_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=comm_plan.fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = comm_plan.ulysses_sequence_parallel_size
        if comm_plan.ulysses_mesh_shape is not None:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=comm_plan.ulysses_mesh_shape, mesh_dim_names=["dp", "sp"]
            )

        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = config.model.get("lora_rank", 0)
        self._is_lora = config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.use_orig_params = config.actor.fsdp_config.get("use_orig_params", False)

        omega_profiler_config = worker_ctx["omega_profiler_config"]
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank,
                config=profiler_config,
                tool_config=tool_config,
            ),
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            self._is_offload_param = config.ref.fsdp_config.get("param_offload", False)

        if self._is_actor:
            config.actor.ppo_mini_batch_size *= config.rollout.n
            config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            )
            if config.actor.ppo_micro_batch_size is not None:
                config.actor.ppo_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
                config.actor.ppo_micro_batch_size_per_gpu = config.actor.ppo_micro_batch_size

            if config.actor.ppo_micro_batch_size_per_gpu is not None:
                actor_dp_size = self.device_mesh.size() // self.ulysses_sequence_parallel_size
                actor_micro_batch_size = config.actor.ppo_micro_batch_size_per_gpu
                runtime_batch_align = math.lcm(actor_dp_size, actor_micro_batch_size)
                if config.actor.ppo_mini_batch_size % actor_micro_batch_size != 0:
                    logger.debug(
                        "[one-step-off][worker] defer actor mini-batch divisibility to runtime batch alignment: "
                        f"role={role}, normalized_mini_batch_size={config.actor.ppo_mini_batch_size}, "
                        f"ppo_micro_batch_size_per_gpu={actor_micro_batch_size}, actor_dp_size={actor_dp_size}, "
                        f"runtime_batch_alignment={runtime_batch_align}"
                    )
                assert config.actor.ppo_mini_batch_size // config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {config.actor.ppo_micro_batch_size_per_gpu}"
                )

        if self._is_rollout and config.rollout.log_prob_micro_batch_size is not None:
            config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            config.rollout.log_prob_micro_batch_size_per_gpu = config.rollout.log_prob_micro_batch_size

        if self._is_ref and config.ref.log_prob_micro_batch_size is not None:
            config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            config.ref.log_prob_micro_batch_size_per_gpu = config.ref.log_prob_micro_batch_size

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._dynamic_resize_enabled:
            return self._local_init_model()
        return self._fallback_init_model()

    def _fallback_init_model(self):
        return ActorRolloutRefWorker.init_model(self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_worker_init(self):
        """显式 worker-init prepare 入口。

        注意：当前阶段该方法主要承担“显式生命周期接口”的语义，
        真正需要延后的 distributed/device-mesh 绑定仍在 commit 接口中。
        """
        if not self._dynamic_resize_enabled:
            logger.debug("[one-step-off][worker] prepare_worker_init skipped: dynamic_resize disabled, role=%s", self.role)
            return {"resize/prepare_worker_init_s": 0.0}
        if getattr(self, "_worker_init_prepared", False):
            logger.debug("[one-step-off][worker] prepare_worker_init skipped: already prepared, role=%s", self.role)
            return {"resize/prepare_worker_init_s": 0.0}
        started_at = time.monotonic()
        with resize_trace_span(
            self._resize_trace_config,
            "prepare_worker_init",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": getattr(self, "rank", None)},
        ):
            self._worker_init_plan = self._prepare_local_init_worker_state(config=self.config, role=self.role)
        self._worker_init_prepared = True
        return {"resize/prepare_worker_init_s": time.monotonic() - started_at}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def commit_worker_init(self):
        """显式 worker-init commit 入口。"""
        if not self._dynamic_resize_enabled:
            logger.debug("[one-step-off][worker] commit_worker_init skipped: dynamic_resize disabled, role=%s", self.role)
            return {"resize/prepare_worker_init_s": 0.0, "resize/commit_worker_init_s": 0.0}
        if getattr(self, "_worker_init_committed", False):
            logger.debug("[one-step-off][worker] commit_worker_init skipped: already committed, role=%s", self.role)
            return {"resize/prepare_worker_init_s": 0.0, "resize/commit_worker_init_s": 0.0}
        prepare_duration = 0.0
        if not getattr(self, "_worker_init_prepared", False):
            logger.debug("[one-step-off][worker] commit_worker_init auto prepare: role=%s", self.role)
            prepare_metrics = self.prepare_worker_init() or {}
            prepare_duration = float(prepare_metrics.get("resize/prepare_worker_init_s", 0.0))
        started_at = time.monotonic()
        with resize_trace_span(
            self._resize_trace_config,
            "commit_worker_init",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": getattr(self, "rank", None)},
        ):
            self._commit_local_init_worker_state(self._worker_init_plan)
        self._worker_init_committed = True
        return {
            "resize/prepare_worker_init_s": prepare_duration,
            "resize/commit_worker_init_s": time.monotonic() - started_at,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_model_init(self):
        """显式 model-init prepare 入口。"""
        if not self._dynamic_resize_enabled:
            return {"resize/prepare_model_init_s": 0.0}
        if getattr(self, "_model_init_prepared", False):
            return {"resize/prepare_model_init_s": 0.0}
        started_at = time.monotonic()
        with resize_trace_span(
            self._resize_trace_config,
            "prepare_model_init",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": getattr(self, "rank", None)},
        ):
            self._model_init_plan = self._prepare_local_init_model()
        self._model_init_prepared = True
        return {"resize/prepare_model_init_s": time.monotonic() - started_at}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def commit_model_init(self):
        """显式 model-init commit 入口。"""
        if not self._dynamic_resize_enabled:
            return {"resize/prepare_model_init_s": 0.0, "resize/commit_model_init_s": 0.0}
        if getattr(self, "_model_init_committed", False):
            return {"resize/prepare_model_init_s": 0.0, "resize/commit_model_init_s": 0.0}
        if not getattr(self, "_worker_init_committed", False):
            self.commit_worker_init()
        prepare_duration = 0.0
        if not getattr(self, "_model_init_prepared", False):
            prepare_metrics = self.prepare_model_init() or {}
            prepare_duration = float(prepare_metrics.get("resize/prepare_model_init_s", 0.0))
        started_at = time.monotonic()
        with resize_trace_span(
            self._resize_trace_config,
            "commit_model_init",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": getattr(self, "rank", None)},
        ):
            self._commit_local_init_model(self._model_init_plan)
        self._model_init_committed = True
        return {
            "resize/prepare_model_init_s": prepare_duration,
            "resize/commit_model_init_s": time.monotonic() - started_at,
        }

    def _local_init_model(self):
        """dynamic_resize 模式下的本地化 init_model。

        当前实现先拆成 prepare / commit 两阶段，但仍保持与原先
        `_local_init_model()` 基本一致的执行时序，避免这一步重构改变行为。

        这样后续要缩短 overlap resize 的独占窗口时，可以优先把 prepare
        阶段继续前移，而不用再改动主线或重新拆入口。
        """
        # 兼容旧入口：在未改造调用方时，仍然通过 init_model() 串起
        # 显式 prepare/commit 生命周期。
        self.prepare_model_init()
        return self.commit_model_init()

    def _prepare_local_init_model(self):
        """准备本地 init_model 的纯配置/路径阶段。

        这一阶段只做不依赖最终独占资源切换窗口的准备工作：
        - 导入 external libs，保证 HF / registry 类逻辑可用
        - 解析 override config
        - 读取开关、路径、tiled mlp 等静态配置

        设计上它应该尽量避免：
        - FSDP 包装
        - rollout engine 真正构建
        - optimizer / checkpoint manager 绑定

        当前为了保持行为稳定，真正重资源的逻辑仍在 commit 阶段中执行。
        """
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        model_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
        model_build_plan = None
        actor_local_model_path = None
        if self._is_actor or self._is_rollout:
            # 这里仍然只是做本地模型路径准备；真正的模型/FSDP/optimizer 构建会在 commit 阶段进行。
            actor_local_model_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            model_build_plan = {
                "model_path": actor_local_model_path,
                "override_model_config": override_model_config,
                "use_remove_padding": use_remove_padding,
                "use_fused_kernels": use_fused_kernels,
                "enable_gradient_checkpointing": self.config.model.get("enable_gradient_checkpointing", False),
                "trust_remote_code": self.config.model.get("trust_remote_code", False),
                "use_liger": self.config.model.get("use_liger", False),
                "role": "actor",
                "enable_activation_offload": self.config.model.get("enable_activation_offload", False),
                "use_prefix_grouper": self.config.actor.get("use_prefix_grouper", False),
                "use_tiled_mlp": model_tiled_mlp_config.get("enabled", False),
                "tiled_mlp_shards": model_tiled_mlp_config.get("num_shards", 4),
            }

        ref_model_path = None
        ref_local_model_path = None
        ref_use_prefix_grouper = False
        ref_build_plan = None
        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)
            ref_local_model_path = copy_to_local(ref_model_path, use_shm=use_shm)
            ref_use_prefix_grouper = hasattr(self.config, "actor") and self.config.actor.get("use_prefix_grouper", False)

        ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None) if self._is_ref else None
        if ref_tiled_mlp_config is None:
            ref_tiled_mlp_config = model_tiled_mlp_config
        if self._is_ref:
            ref_build_plan = {
                "model_path": ref_local_model_path,
                "override_model_config": override_model_config,
                "use_remove_padding": use_remove_padding,
                "use_fused_kernels": use_fused_kernels,
                "trust_remote_code": self.config.model.get("trust_remote_code", False),
                "use_liger": self.config.model.get("use_liger", False),
                "role": "ref",
                "use_prefix_grouper": ref_use_prefix_grouper,
                "use_tiled_mlp": ref_tiled_mlp_config.get("enabled", False),
                "tiled_mlp_shards": ref_tiled_mlp_config.get("num_shards", 4),
            }

        return {
            "override_model_config": override_model_config,
            "use_remove_padding": use_remove_padding,
            "use_shm": use_shm,
            "use_fused_kernels": use_fused_kernels,
            "actor_local_model_path": actor_local_model_path,
            "model_tiled_mlp_config": model_tiled_mlp_config,
            "model_build_plan": model_build_plan,
            "ref_model_path": ref_model_path,
            "ref_local_model_path": ref_local_model_path,
            "ref_use_prefix_grouper": ref_use_prefix_grouper,
            "ref_tiled_mlp_config": ref_tiled_mlp_config,
            "ref_build_plan": ref_build_plan,
        }

    def _commit_local_init_model(self, prepare_ctx):
        """提交本地 init_model 的重资源阶段。

        这一阶段保留当前主线等价语义，负责真正占用/绑定训练与推理资源：
        - 模型复制到本地路径后构建 FSDP/optimizer
        - rollout engine 初始化
        - ref policy 构建
        - flops counter / checkpoint manager 绑定

        后续如果要支持 overlap resize 的更短切换窗口，应优先继续把可前移的
        工作从这里挪到 prepare 阶段。
        """
        from verl.workers.actor import DataParallelPPOActor

        override_model_config = prepare_ctx["override_model_config"]
        use_remove_padding = prepare_ctx["use_remove_padding"]
        use_fused_kernels = prepare_ctx["use_fused_kernels"]

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            model_build_plan = dict(prepare_ctx["model_build_plan"])

            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=model_build_plan["model_path"],
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=model_build_plan["override_model_config"],
                use_remove_padding=model_build_plan["use_remove_padding"],
                use_fused_kernels=model_build_plan["use_fused_kernels"],
                enable_gradient_checkpointing=model_build_plan["enable_gradient_checkpointing"],
                trust_remote_code=model_build_plan["trust_remote_code"],
                use_liger=model_build_plan["use_liger"],
                role=model_build_plan["role"],
                enable_activation_offload=model_build_plan["enable_activation_offload"],
                use_prefix_grouper=model_build_plan["use_prefix_grouper"],
                use_tiled_mlp=model_build_plan["use_tiled_mlp"],
                tiled_mlp_shards=model_build_plan["tiled_mlp_shards"],
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            sanitized_actor_cfg, rollout_correction_cfg = _sanitize_actor_config_for_dataclass(self.config.actor)
            with open_dict(sanitized_actor_cfg):
                # Keep the dynamic-resize local init path aligned with the main FSDP
                # worker path: the model-level flags are the effective actor flags.
                sanitized_actor_cfg.use_remove_padding = use_remove_padding
                sanitized_actor_cfg.use_fused_kernels = use_fused_kernels
                sanitized_actor_cfg.use_prefix_grouper = model_build_plan["use_prefix_grouper"]
            actor_cfg = omega_conf_to_dataclass(sanitized_actor_cfg)
            if rollout_correction_cfg is not None:
                actor_cfg.policy_loss.rollout_correction = rollout_correction_cfg
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = prepare_ctx["ref_model_path"]

            if self.rank == 0:
                print("reference model:", ref_model_path)
            ref_build_plan = dict(prepare_ctx["ref_build_plan"])

            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=ref_build_plan["model_path"],
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=ref_build_plan["override_model_config"],
                use_remove_padding=ref_build_plan["use_remove_padding"],
                use_fused_kernels=ref_build_plan["use_fused_kernels"],
                trust_remote_code=ref_build_plan["trust_remote_code"],
                use_liger=ref_build_plan["use_liger"],
                role=ref_build_plan["role"],
                use_prefix_grouper=ref_build_plan["use_prefix_grouper"],
                use_tiled_mlp=ref_build_plan["use_tiled_mlp"],
                tiled_mlp_shards=ref_build_plan["tiled_mlp_shards"],
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if ref_build_plan["use_prefix_grouper"]:
                    self.config.ref.use_prefix_grouper = ref_build_plan["use_prefix_grouper"]
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )


class DetachSync(LocalInitActorRolloutRefWorker, AsyncActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

    def _get_active_weight_sync_group_name(self) -> str:
        return getattr(self, "_active_weight_sync_group_name", "actor_rollout")

    def _set_active_weight_sync_group_name(self, group_name: str) -> None:
        self._active_weight_sync_group_name = group_name
        if device_name == "npu":
            self._weight_sync_group = getattr(self, "_weight_sync_group_cache", {}).get(group_name)

    def _get_weight_sync_group_cache(self) -> dict[str, object]:
        cache = getattr(self, "_weight_sync_group_cache", None)
        if cache is None:
            cache = {}
            self._weight_sync_group_cache = cache
        return cache

    def _get_weight_sync_group_metadata(self) -> dict[str, tuple[int, int]]:
        metadata = getattr(self, "_weight_sync_group_metadata", None)
        if metadata is None:
            metadata = {}
            self._weight_sync_group_metadata = metadata
        return metadata

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_resize_resource_snapshot(self, staging_path: str | None = None):
        host_free_bytes = _get_available_host_memory_bytes()
        gpu_free_bytes, gpu_total_bytes = _get_available_device_memory_bytes()
        disk_free_bytes = None
        if staging_path:
            try:
                disk_free_bytes = int(shutil.disk_usage(staging_path).free)
            except Exception:
                disk_free_bytes = None
        return {
            "host_free_bytes": host_free_bytes,
            "gpu_free_bytes": gpu_free_bytes,
            "gpu_total_bytes": gpu_total_bytes,
            "disk_free_bytes": disk_free_bytes,
        }

    def _maybe_init_collective_group(self, group_name: str = "actor_rollout") -> None:
        if device_name == "npu":
            return
        rank = getattr(self, "_actor_rollout_collective_rank", None)
        world_size = getattr(self, "_actor_rollout_collective_world_size", None)
        if rank is None or world_size is None:
            return
        try:
            if hasattr(collective, "is_group_initialized") and collective.is_group_initialized(group_name=group_name):
                return
        except Exception:  # pragma: no cover - best effort
            pass

        try:
            collective.init_collective_group(
                world_size=world_size,
                rank=rank,
                backend=get_nccl_backend(),
                group_name=group_name,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to init Ray collective group '%s': %s", group_name, exc)

    def _get_collective_rank_world_size(self, group_name: str):
        try:
            if hasattr(collective, "get_rank"):
                collective_rank = collective.get_rank(group_name=group_name)
            else:  # pragma: no cover - debug-only
                collective_rank = "unavailable"
        except Exception as exc:  # pragma: no cover - debug-only
            collective_rank = f"error:{exc}"

        try:
            if hasattr(collective, "get_world_size"):
                collective_world_size = collective.get_world_size(group_name=group_name)
            elif hasattr(collective, "get_group_size"):
                collective_world_size = collective.get_group_size(group_name=group_name)
            else:  # pragma: no cover - debug-only
                collective_world_size = "unavailable"
        except Exception as exc:  # pragma: no cover - debug-only
            collective_world_size = f"error:{exc}"

        try:
            if hasattr(collective, "is_group_initialized"):
                group_initialized = collective.is_group_initialized(group_name=group_name)
            else:  # pragma: no cover - debug-only
                group_initialized = "unavailable"
        except Exception as exc:  # pragma: no cover - debug-only
            group_initialized = f"error:{exc}"

        return collective_rank, collective_world_size, group_initialized

    def _ensure_weight_sync_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        *,
        group_name: str,
        activate: bool,
    ) -> None:
        rank = torch.distributed.get_rank() + rank_offset
        self._actor_rollout_collective_rank = rank
        self._actor_rollout_collective_world_size = world_size
        metadata = self._get_weight_sync_group_metadata()
        cached_key = metadata.get(group_name)
        current_key = (rank, world_size)

        # Keep prepare idempotent: if the group for this worker already matches
        # the requested rank/world-size contract, only activate when asked.
        if cached_key == current_key:
            if activate:
                self._set_active_weight_sync_group_name(group_name)
            if device_name == "npu":
                cache = self._get_weight_sync_group_cache()
                cached_group = cache.get(group_name)
                if cached_group is not None and activate:
                    self._weight_sync_group = cached_group
                return
            self._maybe_init_collective_group(group_name=group_name)
            return

        if device_name == "npu":
            prepared_group = vllm_stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                get_torch_device().current_device(),
            )
            self._get_weight_sync_group_cache()[group_name] = prepared_group
            if activate:
                self._weight_sync_group = prepared_group
        else:
            self._maybe_init_collective_group(group_name=group_name)
        metadata[group_name] = current_key
        if activate:
            self._set_active_weight_sync_group_name(group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size, group_name="actor_rollout"):
        self._ensure_weight_sync_group(
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name=group_name,
            activate=True,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def prepare_weight_sync_group(self, master_address, master_port, rank_offset, world_size, group_name="actor_rollout"):
        # Build the communicator eagerly on the new worker lifecycle without
        # rebinding the active path yet. The trainer will activate it only after
        # the new actor/rollout pair is published together.
        self._ensure_weight_sync_group(
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name=group_name,
            activate=False,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def activate_weight_sync_group(self, group_name: str):
        # Activation is the fast path used by the trainer on communicator-cache
        # hits: switch the active handle without rebuilding the group.
        self._set_active_weight_sync_group_name(group_name)
        if device_name == "npu":
            self._weight_sync_group = self._get_weight_sync_group_cache().get(group_name)
        else:
            self._maybe_init_collective_group(group_name=group_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def destroy_weight_sync_group(self, group_name: str):
        self._get_weight_sync_group_metadata().pop(group_name, None)
        self._get_weight_sync_group_cache().pop(group_name, None)
        if device_name == "npu":
            if self._get_active_weight_sync_group_name() == group_name:
                self._weight_sync_group = None
            return
        try:
            if hasattr(collective, "is_group_initialized") and not collective.is_group_initialized(group_name=group_name):
                return
        except Exception:
            pass

        try:
            collective.destroy_collective_group(group_name)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to destroy Ray collective group '%s': %s", group_name, exc)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None
        group_name = self._get_active_weight_sync_group_name()

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        if self._is_actor:
            params = self._get_actor_params()
        else:
            params = None

        rollout_name = self.config.rollout.name
        if self._is_rollout:
            if rollout_name == "vllm":
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                inference_model = self.rollout.inference_engine.worker.model_runner.model
                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        loop = get_event_loop()

        dynamic_resize_cfg = getattr(self.config, "dynamic_resize", None)
        dynamic_resize_enabled = bool(getattr(dynamic_resize_cfg, "enable", False))
        if device_name != "npu":
            self._maybe_init_collective_group(group_name=group_name)

        collective_rank, collective_world_size, collective_initialized = self._get_collective_rank_world_size(
            group_name=group_name
        )

        for idx, (key, shape, dtype) in enumerate(self._weights_info):
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            if device_name == "npu":
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                if not collective_initialized:
                    raise RuntimeError(
                        f"Ray collective group '{group_name}' is not initialized. "
                        "Please ensure create_weight_sync_group is called on all actor/rollout workers."
                    )
                collective.broadcast(tensor, src_rank=0, group_name=group_name)

            if self._is_rollout:
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    # first_rank_in_node = self._tp_rank % tp_size_per_node == 0，
                    # Only the first rank within each node (i.e., the local rank is 0) initializes the engine;
                    # engines for other ranks are set to None.

                    if inference_model is not None:
                        loop.run_until_complete(self.update_weights(inference_model, [(key, tensor)]))

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()

    async def update_weights(self, inference_engine, params):
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        await sgl_update_weights(
            engine=inference_engine,
            params_batch=params,
            device_mesh_key="infer_tp",
            device_mesh=self.rollout_device_mesh,
        )

        if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
            await inference_engine.flush_cache()


class DetachActorWorker(DetachSync):
    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        materialize_metrics = self._ensure_optimizer_materialized_before_update()

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time, images_seqlens=images_seqlens
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
            metrics.update(materialize_metrics)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_actor_handoff_state(self, local_path: str):
        self._export_actor_handoff_state(local_path, stage_optimizer=True, chunk_mb=None)

    def _export_actor_handoff_state(
        self,
        local_path: str,
        *,
        stage_optimizer: bool,
        chunk_mb: int | None,
        backend: str = "disk_fallback",
        optimizer_restore_policy: str | None = None,
    ) -> None:
        # Export always materializes a CPU-resident full state first. Round two
        # reuses this path for both legacy checkpoint-like handoff and the new
        # host-staging abstraction.
        assert self._is_actor

        logger.info("[one-step-off][resize][handoff] rank=%s saving actor handoff state to %s", self.rank, local_path)

        os.makedirs(local_path, exist_ok=True) if self.rank == 0 else None
        torch.distributed.barrier()

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(self.actor_optimizer, device_id=get_torch_device().current_device())

        _sanitize_optimizer_container_values(self.actor_optimizer)

        logger.info("[one-step-off][resize][handoff] rank=%s collecting full model state", self.rank)
        model_state_dict = get_fsdp_full_state_dict(self.actor_module_fsdp, offload_to_cpu=True, rank0_only=True)

        optim_state_dict = None
        if stage_optimizer:
            logger.info("[one-step-off][resize][handoff] rank=%s collecting optimizer state", self.rank)
            StateDictOptions, get_optimizer_state_dict, _ = _get_reshardable_optimizer_state_dict_api()
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            optim_state_dict = get_optimizer_state_dict(self.actor_module_fsdp, self.actor_optimizer, options=options)

        model_manifest = None
        optim_manifest = None
        if self.rank == 0:
            page_bytes = max(int(chunk_mb or 0), 0) * 1024 * 1024
            if page_bytes > 0:
                model_manifest = save_paged_state_dict(local_path, "model_state", model_state_dict, page_bytes)
            else:
                torch.save(model_state_dict, os.path.join(local_path, "model_state.pt"))
                model_manifest = {
                    "page_count": 1,
                    "total_bytes": _estimate_object_bytes(model_state_dict),
                }
            if stage_optimizer:
                if page_bytes > 0:
                    optim_manifest = save_paged_optimizer_state_dict(local_path, "optim_state", optim_state_dict, page_bytes)
                else:
                    torch.save(optim_state_dict, os.path.join(local_path, "optim_state.pt"))
                    optim_manifest = {
                        "page_count": 1,
                        "total_bytes": _estimate_object_bytes(optim_state_dict),
                    }
            torch.save(
                {
                    "lr_scheduler": self.actor_lr_scheduler.state_dict() if self.actor_lr_scheduler is not None else None,
                    "rng": self.checkpoint_manager.get_rng_state() if self.checkpoint_manager is not None else None,
                    "stage_optimizer": stage_optimizer,
                    "paged_model_state": page_bytes > 0,
                    "paged_optim_state": page_bytes > 0 and stage_optimizer,
                    "backend": backend,
                },
                os.path.join(local_path, "extra_state.pt"),
            )
            create_restore_session_manifest(
                local_path,
                backend=backend,
                optimizer_restore_policy=optimizer_restore_policy or ("immediate" if stage_optimizer else "deferred"),
                model_manifest=model_manifest,
                optimizer_manifest=optim_manifest,
            )

        torch.distributed.barrier()

        logger.info("[one-step-off][resize][handoff] rank=%s finished saving actor handoff state", self.rank)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    def _export_actor_handoff_state_to_pinned_cpu(self, local_path: str, cfg: HostStagingConfig) -> None:
        """Stage handoff pages on local storage; pinned memory is applied only in the restore worker."""
        if int(cfg.chunk_mb or 0) <= 0:
            raise ValueError("pinned_cpu staging requires chunk_mb > 0 so pages can be restored through local pinned buffers")

        self._export_actor_handoff_state(
            local_path,
            stage_optimizer=cfg.stage_optimizer,
            chunk_mb=cfg.chunk_mb,
            backend="pinned_cpu",
            optimizer_restore_policy=cfg.optimizer_restore_policy,
        )
        if self.rank == 0:
            extra_state_path = os.path.join(local_path, "extra_state.pt")
            extra_state = torch.load(extra_state_path, weights_only=False)
            extra_state.update(
                {
                    "async_optimizer_preload": cfg.async_optimizer_preload,
                    "preload_queue_depth": cfg.preload_queue_depth,
                    "device_preload_threshold": cfg.device_preload_threshold,
                }
            )
            torch.save(extra_state, extra_state_path)
        torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_actor_handoff_state(self, local_path: str):
        self._restore_actor_handoff_state(
            local_path,
            load_optimizer=True,
            optimizer_restore_policy="immediate",
            progressive_swap=False,
            chunk_mb=None,
            cleanup_after_load=False,
        )

    def _restore_paged_actor_model_state(self, local_path: str, *, cpu_offload, progressive_swap: bool) -> None:
        StateDictOptions, set_model_state_dict = _get_reshardable_model_state_dict_api()

        manifest = read_paged_state_manifest(local_path, "model_state") if self.rank == 0 else None
        manifest_list = [manifest]
        torch.distributed.broadcast_object_list(manifest_list, src=0)
        manifest = manifest_list[0] or {}

        if self.rank == 0:
            model = self.actor_module_fsdp.to(device=get_torch_device().current_device(), non_blocking=True)
        else:
            model = self.actor_module_fsdp.to_empty(device=get_torch_device().current_device())

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload is not None,
            broadcast_from_rank0=True,
            strict=False,
        )

        for page_idx in range(int(manifest.get("page_count", 0))):
            if self.rank == 0:
                file_name = manifest["files"][page_idx]
                page_state = torch.load(os.path.join(local_path, file_name), weights_only=False)
            else:
                page_state = {}

            set_model_state_dict(model, page_state, options=options)

            if self.rank == 0:
                del page_state
                gc.collect()
                if has_restore_session_manifest(local_path):
                    record_restore_progress(
                        local_path,
                        applied_model_pages=page_idx + 1,
                        status="restoring_model",
                    )
            if progressive_swap:
                get_torch_device().empty_cache()

        for _, buf in model.named_buffers():
            torch.distributed.broadcast(buf, src=0)

        if cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(get_torch_device().current_device())

    def _restore_pinned_actor_model_state(
        self,
        local_path: str,
        *,
        cpu_offload,
        progressive_swap: bool,
        device_preload_threshold: float,
        preload_queue_depth: int,
    ) -> None:
        """Load model pages through rank-local pinned buffers without any RPC hop."""
        StateDictOptions, set_model_state_dict = _get_reshardable_model_state_dict_api()
        manifest = read_paged_state_manifest(local_path, "model_state") if self.rank == 0 else None
        manifest_list = [manifest]
        torch.distributed.broadcast_object_list(manifest_list, src=0)
        manifest = manifest_list[0] or {}

        if self.rank == 0:
            _assert_rank0_local_pin_memory_available()
            model = self.actor_module_fsdp.to(device=get_torch_device().current_device(), non_blocking=True)
        else:
            model = self.actor_module_fsdp.to_empty(device=get_torch_device().current_device())

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload is not None,
            broadcast_from_rank0=True,
            strict=False,
        )

        page_iter = (
            _build_streaming_page_iterator(
                local_path,
                "model_state",
                device_preload_threshold=device_preload_threshold,
                preload_queue_depth=preload_queue_depth,
            )
            if self.rank == 0
            else None
        )
        for page_idx in range(int(manifest.get("page_count", 0))):
            page_state = next(page_iter) if self.rank == 0 and page_iter is not None else {}

            set_model_state_dict(model, page_state, options=options)

            if self.rank == 0:
                del page_state
                gc.collect()
                if has_restore_session_manifest(local_path):
                    record_restore_progress(local_path, applied_model_pages=page_idx + 1, status="restoring_model")
            if progressive_swap:
                get_torch_device().empty_cache()

        for _, buf in model.named_buffers():
            torch.distributed.broadcast(buf, src=0)

        if cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(get_torch_device().current_device())

    def _load_paged_actor_optimizer_state(self, local_path: str, *, progressive_swap: bool) -> tuple[dict, int]:
        manifest = read_paged_state_manifest(local_path, "optim_state") if self.rank == 0 else None
        manifest_list = [manifest]
        torch.distributed.broadcast_object_list(manifest_list, src=0)
        manifest = manifest_list[0] or {}
        manifest_page_count = int(manifest.get("page_count", 0))

        if self.rank == 0:
            logger.info(
                "[one-step-off][resize][handoff] rank=%s paged optimizer restore enabled: page_count=%s",
                self.rank,
                manifest_page_count,
            )
            optim_state_dict = load_paged_optimizer_state_dict(local_path, "optim_state")
            gc.collect()
        else:
            optim_state_dict = {}

        if progressive_swap:
            get_torch_device().empty_cache()
        return optim_state_dict, manifest_page_count

    def _load_pinned_actor_optimizer_state(
        self,
        local_path: str,
        *,
        progressive_swap: bool,
        device_preload_threshold: float,
        preload_queue_depth: int,
        consume_preloaded_pages: bool = False,
    ) -> tuple[dict, int]:
        if self.rank == 0:
            _assert_rank0_local_pin_memory_available()
            manifest = read_paged_state_manifest(local_path, "optim_state")
            logger.info(
                "[one-step-off][resize][host-stage] rank=%s pinned optimizer restore enabled: page_count=%s path=%s",
                self.rank,
                manifest.get("page_count", 0),
                local_path,
            )
            merged_state: dict[str, Any] = {}
            merged_param_groups: list[dict[str, Any]] | None = None
            if consume_preloaded_pages and self._pending_optimizer_preload_session is not None:
                page_iter = self._consume_pending_optimizer_preload_pages()
            else:
                page_iter = _build_streaming_page_iterator(
                    local_path,
                    "optim_state",
                    device_preload_threshold=device_preload_threshold,
                    preload_queue_depth=preload_queue_depth,
                )
            for page in page_iter:
                merged_state.update(page.get("state", {}))
                if merged_param_groups is None:
                    merged_param_groups = list(page.get("param_groups", []))
            optim_state_dict = {"state": merged_state, "param_groups": merged_param_groups or []}
            page_count = int(manifest.get("page_count", 0))
        else:
            optim_state_dict = {}
            page_count = 0

        if progressive_swap:
            get_torch_device().empty_cache()
        return optim_state_dict, page_count

    def _clear_pending_optimizer_restore(self) -> None:
        session = self._pending_optimizer_preload_session
        if session is not None:
            thread = session.get("thread")
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self._pending_optimizer_restore_path = None
        self._pending_optimizer_restore_page_count = 0
        self._pending_optimizer_restore_progressive_swap = False
        self._pending_optimizer_restore_cleanup_after_load = False
        self._pending_optimizer_restore_policy = "immediate"
        self._pending_optimizer_restore_backend = "disk_fallback"
        self._pending_optimizer_preload_queue_depth = 1
        self._pending_optimizer_device_preload_threshold = 0.0
        self._pending_optimizer_async_preload_enabled = False
        self._pending_optimizer_preload_session = None

    def _mark_pending_optimizer_restore(
        self,
        *,
        local_path: str,
        page_count: int,
        progressive_swap: bool,
        cleanup_after_load: bool,
        optimizer_restore_policy: str,
        backend: str,
        preload_queue_depth: int,
        device_preload_threshold: float,
        async_optimizer_preload: bool,
    ) -> None:
        self._pending_optimizer_restore_path = local_path
        self._pending_optimizer_restore_page_count = max(int(page_count), 0)
        self._pending_optimizer_restore_progressive_swap = progressive_swap
        self._pending_optimizer_restore_cleanup_after_load = cleanup_after_load
        self._pending_optimizer_restore_policy = optimizer_restore_policy
        self._pending_optimizer_restore_backend = backend
        self._pending_optimizer_preload_queue_depth = max(int(preload_queue_depth), 1)
        self._pending_optimizer_device_preload_threshold = min(max(float(device_preload_threshold), 0.0), 1.0)
        self._pending_optimizer_async_preload_enabled = bool(async_optimizer_preload)
        if (
            self.rank == 0
            and backend == "pinned_cpu"
            and self._pending_optimizer_async_preload_enabled
            and self._pending_optimizer_restore_page_count > 0
        ):
            self._start_pending_optimizer_preload()

    def _has_pending_optimizer_restore(self) -> bool:
        return bool(self._pending_optimizer_restore_path)

    def _start_pending_optimizer_preload(self) -> None:
        local_path = self._pending_optimizer_restore_path
        if not local_path:
            return
        if self._pending_optimizer_preload_session is not None:
            return

        _assert_rank0_local_pin_memory_available()
        preload_queue: queue.Queue[Any] = queue.Queue(maxsize=max(self._pending_optimizer_preload_queue_depth, 1))
        session: dict[str, Any] = {
            "queue": preload_queue,
            "error": None,
            "done": threading.Event(),
            "thread": None,
        }

        def _worker() -> None:
            try:
                for page in _build_streaming_page_iterator(
                    local_path,
                    "optim_state",
                    device_preload_threshold=self._pending_optimizer_device_preload_threshold,
                    preload_queue_depth=self._pending_optimizer_preload_queue_depth,
                ):
                    preload_queue.put(page)
            except Exception as exc:  # pragma: no cover - exercised in runtime
                session["error"] = exc
            finally:
                preload_queue.put(_ASYNC_PRELOAD_SENTINEL)
                session["done"].set()

        thread = threading.Thread(
            target=_worker,
            name=f"optimizer_preload_rank{self.rank}",
            daemon=True,
        )
        session["thread"] = thread
        self._pending_optimizer_preload_session = session
        thread.start()

    def _consume_pending_optimizer_preload_pages(self):
        session = self._pending_optimizer_preload_session
        if session is None:
            return

        preload_queue: queue.Queue[Any] = session["queue"]
        while True:
            item = preload_queue.get()
            if item is _ASYNC_PRELOAD_SENTINEL:
                break
            yield item

        thread = session.get("thread")
        if thread is not None:
            thread.join(timeout=1.0)
        error = session.get("error")
        self._pending_optimizer_preload_session = None
        if error is not None:
            raise error

    def _get_optimizer_restore_page_count(self, local_path: str) -> int:
        if self.rank == 0 and has_restore_session_manifest(local_path):
            session_manifest = read_restore_session_manifest(local_path)
            if session_manifest.get("backend") == "pinned_cpu":
                page_count = int(session_manifest.get("optimizer_page_count", 0))
            else:
                page_count = None
        else:
            page_count = None
        page_count_list = [page_count]
        torch.distributed.broadcast_object_list(page_count_list, src=0)
        page_count = page_count_list[0]
        if page_count is not None:
            return int(page_count)

        paged_optim_state = has_paged_state_dict(local_path, "optim_state") if self.rank == 0 else False
        paged_optim_state_list = [paged_optim_state]
        torch.distributed.broadcast_object_list(paged_optim_state_list, src=0)
        paged_optim_state = bool(paged_optim_state_list[0])

        if paged_optim_state:
            if self.rank == 0:
                manifest = read_paged_state_manifest(local_path, "optim_state")
                page_count = int(manifest.get("page_count", 0))
            else:
                page_count = 0
            page_count_list = [page_count]
            torch.distributed.broadcast_object_list(page_count_list, src=0)
            return int(page_count_list[0])

        if self.rank == 0:
            page_count = 1 if os.path.exists(os.path.join(local_path, "optim_state.pt")) else 0
        else:
            page_count = 0
        page_count_list = [page_count]
        torch.distributed.broadcast_object_list(page_count_list, src=0)
        return int(page_count_list[0])

    def _load_optimizer_state_from_handoff_path(
        self,
        local_path: str,
        *,
        progressive_swap: bool,
        device_preload_threshold: float,
        preload_queue_depth: int,
        consume_preloaded_pages: bool = False,
    ) -> tuple[dict, int]:
        if self.rank == 0 and has_restore_session_manifest(local_path):
            session_manifest = read_restore_session_manifest(local_path)
            backend = session_manifest.get("backend", "disk_fallback")
            service_name = session_manifest.get("service_name")
        else:
            backend = None
            service_name = None
        info_list = [{"backend": backend, "service_name": service_name}]
        torch.distributed.broadcast_object_list(info_list, src=0)
        backend = info_list[0]["backend"]
        service_name = info_list[0]["service_name"]

        if backend == "pinned_cpu":
            if service_name:
                raise ValueError(
                    "legacy pinned_cpu Ray-service manifests are no longer supported; please restage the handoff state"
                )
            return self._load_pinned_actor_optimizer_state(
                local_path,
                progressive_swap=progressive_swap,
                device_preload_threshold=device_preload_threshold,
                preload_queue_depth=preload_queue_depth,
                consume_preloaded_pages=consume_preloaded_pages,
            )

        paged_optim_state = has_paged_state_dict(local_path, "optim_state") if self.rank == 0 else False
        paged_optim_state_list = [paged_optim_state]
        torch.distributed.broadcast_object_list(paged_optim_state_list, src=0)
        paged_optim_state = bool(paged_optim_state_list[0])

        if paged_optim_state:
            return self._load_paged_actor_optimizer_state(local_path, progressive_swap=progressive_swap)

        if self.rank == 0:
            optim_state_path = os.path.join(local_path, "optim_state.pt")
            if os.path.exists(optim_state_path):
                optim_state_dict = torch.load(optim_state_path, weights_only=False)
                page_count = 1 if optim_state_dict else 0
            else:
                optim_state_dict = {}
                page_count = 0
        else:
            optim_state_dict = {}
            page_count = 0

        if progressive_swap:
            get_torch_device().empty_cache()
        return optim_state_dict, page_count

    def _ensure_optimizer_materialized_before_update(self) -> dict[str, float | int]:
        if not self._has_pending_optimizer_restore():
            return {}

        local_path = self._pending_optimizer_restore_path
        assert local_path is not None
        page_count_hint = self._pending_optimizer_restore_page_count
        started_at = time.monotonic()

        try:
            with resize_trace_span(
                self._resize_trace_config,
                "optimizer_materialize",
                lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
                metadata={"role": self.role, "rank": getattr(self, "rank", None)},
            ):
                optim_state_dict, page_count = self._load_optimizer_state_from_handoff_path(
                    local_path,
                    progressive_swap=self._pending_optimizer_restore_progressive_swap,
                    device_preload_threshold=self._pending_optimizer_device_preload_threshold,
                    preload_queue_depth=self._pending_optimizer_preload_queue_depth,
                    consume_preloaded_pages=(
                        self._pending_optimizer_restore_backend == "pinned_cpu"
                        and self._pending_optimizer_preload_session is not None
                    ),
                )
                StateDictOptions, _, set_optimizer_state_dict = _get_reshardable_optimizer_state_dict_api()
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
                set_optimizer_state_dict(self.actor_module_fsdp, self.actor_optimizer, optim_state_dict, options=options)

            applied_page_count = max(int(page_count), int(page_count_hint), 0)
            if self.rank == 0 and has_restore_session_manifest(local_path):
                update_restore_session_manifest(
                    local_path,
                    status="completed",
                    applied_optimizer_pages=applied_page_count,
                    last_error="",
                )

            if self._pending_optimizer_restore_progressive_swap:
                get_torch_device().empty_cache()

            cleanup_applied = 0.0
            if self._pending_optimizer_restore_cleanup_after_load and self.rank == 0:
                try:
                    shutil.rmtree(local_path)
                    cleanup_applied = 1.0
                except FileNotFoundError:
                    pass
            torch.distributed.barrier()

            self._pending_optimizer_restore_materialize_count += 1
            duration = time.monotonic() - started_at
            self._clear_pending_optimizer_restore()
            return {
                "resize/optimizer_materialize_s": duration,
                "resize/optimizer_materialize_count": float(self._pending_optimizer_restore_materialize_count),
                "resize/optimizer_pending_pages": float(applied_page_count),
                "resize/host_stage_cleanup": cleanup_applied,
            }
        except Exception as exc:
            if self.rank == 0 and has_restore_session_manifest(local_path):
                update_restore_session_manifest(
                    local_path,
                    status="optimizer_materialize_failed",
                    last_error=repr(exc),
                )
            raise

    def _restore_actor_handoff_state(
        self,
        local_path: str,
        *,
        load_optimizer: bool,
        optimizer_restore_policy: str,
        progressive_swap: bool,
        chunk_mb: int | None,
        cleanup_after_load: bool,
        backend: str = "disk_fallback",
        service_name: str | None = None,
    ) -> None:
        # The conservative "progressive swap" implementation restores model
        # state first, then optimizer state, and explicitly drops allocator
        # caches between phases. This keeps the semantics simple while reducing
        # peak memory pressure at the switch boundary.
        assert self._is_actor

        logger.info(
            "[one-step-off][resize][handoff] rank=%s loading actor handoff state from %s backend=%s",
            self.rank,
            local_path,
            backend,
        )
        optimizer_restore_policy = str(optimizer_restore_policy or "immediate").strip().lower()
        defer_optimizer_restore = load_optimizer and optimizer_restore_policy == "deferred"
        cfg_preload_queue_depth = 1
        cfg_device_preload_threshold = 0.0
        cfg_async_optimizer_preload = False

        if self.rank == 0 and has_restore_session_manifest(local_path):
            update_restore_session_manifest(
                local_path,
                status="restoring",
                last_error="",
                optimizer_restore_policy=optimizer_restore_policy,
                applied_model_pages=0,
                applied_optimizer_pages=0,
            )
        try:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                load_fsdp_optimizer(self.actor_optimizer, device_id=get_torch_device().current_device())

            _sanitize_optimizer_container_values(self.actor_optimizer)

            if backend == "pinned_cpu":
                paged_model_state = True
                paged_optim_state = bool(load_optimizer)
                if self.rank == 0:
                    session_manifest = read_restore_session_manifest(local_path)
                    if session_manifest.get("service_name"):
                        raise ValueError(
                            "legacy pinned_cpu Ray-service manifests are no longer supported; please restage the handoff state"
                        )
                    extra_state = torch.load(os.path.join(local_path, "extra_state.pt"), weights_only=False)
                    cfg_preload_queue_depth = max(int(extra_state.get("preload_queue_depth", 2)), 1)
                    cfg_device_preload_threshold = min(max(float(extra_state.get("device_preload_threshold", 0.9)), 0.0), 1.0)
                    cfg_async_optimizer_preload = bool(extra_state.get("async_optimizer_preload", True))
                else:
                    extra_state = None
                model_state_dict = None
                optim_state_dict = {}
            else:
                paged_model_state = has_paged_state_dict(local_path, "model_state") if self.rank == 0 else False
                paged_model_state_list = [paged_model_state]
                torch.distributed.broadcast_object_list(paged_model_state_list, src=0)
                paged_model_state = bool(paged_model_state_list[0])

                paged_optim_state = has_paged_state_dict(local_path, "optim_state") if self.rank == 0 else False
                paged_optim_state_list = [paged_optim_state]
                torch.distributed.broadcast_object_list(paged_optim_state_list, src=0)
                paged_optim_state = bool(paged_optim_state_list[0])

                if self.rank == 0:
                    model_state_dict = None if paged_model_state else torch.load(
                        os.path.join(local_path, "model_state.pt"), weights_only=False
                    )
                    extra_state = torch.load(os.path.join(local_path, "extra_state.pt"), weights_only=False)
                    optim_state_path = os.path.join(local_path, "optim_state.pt")
                    if load_optimizer and (not paged_optim_state) and os.path.exists(optim_state_path):
                        optim_state_dict = torch.load(optim_state_path, weights_only=False)
                    else:
                        optim_state_dict = {}
                else:
                    model_state_dict = {}
                    optim_state_dict = {}
                    extra_state = None

            optimizer_page_count = self._get_optimizer_restore_page_count(local_path) if load_optimizer else 0

            paged_optimizer_page_count = 0
            if load_optimizer and (not defer_optimizer_restore) and paged_optim_state:
                if backend == "pinned_cpu":
                    optim_state_dict, paged_optimizer_page_count = self._load_pinned_actor_optimizer_state(
                        local_path,
                        progressive_swap=progressive_swap,
                        device_preload_threshold=cfg_device_preload_threshold,
                        preload_queue_depth=cfg_preload_queue_depth,
                    )
                else:
                    optim_state_dict, paged_optimizer_page_count = self._load_paged_actor_optimizer_state(
                        local_path,
                        progressive_swap=progressive_swap,
                    )

            if progressive_swap:
                logger.info(
                    "[one-step-off][resize][handoff] rank=%s progressive restore enabled: backend=disk_fallback chunk_mb=%s",
                    self.rank,
                    chunk_mb,
                )

            logger.info("[one-step-off][resize][handoff] rank=%s restoring full model state", self.rank)
            if fsdp_version(self.actor_module_fsdp) == 2:
                cpu_offload = True if self.config.actor.fsdp_config.get("offload_policy", False) else None
                if backend == "pinned_cpu":
                    self._restore_pinned_actor_model_state(
                        local_path,
                        cpu_offload=cpu_offload,
                        progressive_swap=progressive_swap,
                        device_preload_threshold=cfg_device_preload_threshold,
                        preload_queue_depth=cfg_preload_queue_depth,
                    )
                elif paged_model_state:
                    logger.info(
                        "[one-step-off][resize][handoff] rank=%s paged model restore enabled: page_count=%s",
                        self.rank,
                        read_paged_state_manifest(local_path, "model_state").get("page_count", 0) if self.rank == 0 else -1,
                    )
                    self._restore_paged_actor_model_state(
                        local_path,
                        cpu_offload=cpu_offload,
                        progressive_swap=progressive_swap,
                    )
                else:
                    fsdp2_load_full_state_dict(self.actor_module_fsdp, model_state_dict, self.device_mesh, cpu_offload)
                    if self.rank == 0 and has_restore_session_manifest(local_path):
                        record_restore_progress(local_path, applied_model_pages=1, status="restoring_model")
            else:
                raise NotImplementedError("actor handoff currently supports fsdp2 only")

            if self.rank == 0 and model_state_dict is not None:
                del model_state_dict
                gc.collect()

            if progressive_swap:
                get_torch_device().empty_cache()

            if load_optimizer and not defer_optimizer_restore:
                logger.info("[one-step-off][resize][handoff] rank=%s restoring optimizer state", self.rank)
                StateDictOptions, _, set_optimizer_state_dict = _get_reshardable_optimizer_state_dict_api()
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
                set_optimizer_state_dict(self.actor_module_fsdp, self.actor_optimizer, optim_state_dict, options=options)
                if self.rank == 0 and has_restore_session_manifest(local_path):
                    applied_optimizer_pages = 0
                    if paged_optim_state:
                        applied_optimizer_pages = paged_optimizer_page_count
                    elif load_optimizer and optim_state_dict:
                        applied_optimizer_pages = 1
                    if applied_optimizer_pages > 0:
                        record_restore_progress(
                            local_path,
                            applied_optimizer_pages=applied_optimizer_pages,
                            status="restoring_optimizer",
                        )
                if progressive_swap:
                    get_torch_device().empty_cache()

            extra_state_list = [extra_state]
            torch.distributed.broadcast_object_list(extra_state_list, src=0)
            extra_state = extra_state_list[0]
            if extra_state is not None and self.actor_lr_scheduler is not None and extra_state.get("lr_scheduler") is not None:
                self.actor_lr_scheduler.load_state_dict(extra_state["lr_scheduler"])
            if extra_state is not None and self.checkpoint_manager is not None and extra_state.get("rng") is not None:
                self.checkpoint_manager.load_rng_state(extra_state["rng"])

            if defer_optimizer_restore:
                self._mark_pending_optimizer_restore(
                    local_path=local_path,
                    page_count=optimizer_page_count,
                    progressive_swap=progressive_swap,
                    cleanup_after_load=cleanup_after_load,
                    optimizer_restore_policy=optimizer_restore_policy,
                    backend=backend,
                    preload_queue_depth=cfg_preload_queue_depth,
                    device_preload_threshold=cfg_device_preload_threshold,
                    async_optimizer_preload=cfg_async_optimizer_preload,
                )
                if self.rank == 0 and has_restore_session_manifest(local_path):
                    update_restore_session_manifest(
                        local_path,
                        status="optimizer_deferred",
                        applied_optimizer_pages=0,
                        last_error="",
                    )
            elif self.rank == 0 and has_restore_session_manifest(local_path):
                finalize_restore_session_manifest(local_path, status="completed")

            logger.info("[one-step-off][resize][handoff] rank=%s finished loading actor handoff state", self.rank)
        except Exception as exc:
            if self.rank == 0 and has_restore_session_manifest(local_path):
                update_restore_session_manifest(
                    local_path,
                    status="failed",
                    last_error=repr(exc),
                )
            raise
        finally:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stage_actor_handoff_state_to_host(self, local_path: str, staging_config: dict | None = None):
        # Host staging is represented as a backend + manifest pair so the
        # trainer can switch storage strategies without changing RPC semantics.
        cfg = HostStagingConfig.from_dict(staging_config)
        logger.info(
            "[one-step-off][resize][host-stage] rank=%s export actor state to host staging: requested_backend=%s effective_backend=%s path=%s",
            self.rank,
            cfg.backend,
            cfg.effective_backend(),
            local_path,
        )
        with resize_trace_span(
            self._resize_trace_config,
            "worker_resize_export",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": self.rank, "backend": cfg.effective_backend()},
        ):
            if self.rank == 0:
                write_host_staging_manifest(local_path, cfg)
            torch.distributed.barrier()
            if cfg.effective_backend() == "pinned_cpu":
                self._export_actor_handoff_state_to_pinned_cpu(local_path, cfg)
            else:
                self._export_actor_handoff_state(local_path, stage_optimizer=cfg.stage_optimizer, chunk_mb=cfg.chunk_mb)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_actor_handoff_state_from_host(self, local_path: str, staging_config: dict | None = None):
        # The manifest is broadcast from rank 0 so every worker uses the same
        # effective staging backend and restore policy.
        cfg = HostStagingConfig.from_dict(staging_config)
        manifest = None
        if self.rank == 0:
            manifest = read_host_staging_manifest(local_path)
        manifest_list = [manifest]
        torch.distributed.broadcast_object_list(manifest_list, src=0)
        manifest = manifest_list[0] or {}
        cfg = HostStagingConfig.from_dict({**manifest, **(staging_config or {})})
        with resize_trace_span(
            self._resize_trace_config,
            "worker_resize_import",
            lane=f"{self.role}_rank{getattr(self, 'rank', 'na')}",
            metadata={"role": self.role, "rank": self.rank, "backend": cfg.effective_backend()},
        ):
            self._restore_actor_handoff_state(
                local_path,
                load_optimizer=cfg.stage_optimizer,
                optimizer_restore_policy=cfg.optimizer_restore_policy,
                progressive_swap=cfg.progressive_swap,
                chunk_mb=cfg.chunk_mb,
                cleanup_after_load=cfg.cleanup_after_load,
                backend=cfg.effective_backend(),
                service_name=cfg.service_name,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_host_staging_buffer(self, local_path: str, staging_config: dict | None = None):
        cfg = HostStagingConfig.from_dict(staging_config)
        logger.info(
            "[one-step-off][resize][host-stage] rank=%s release staging metadata: backend=%s path=%s cleanup_after_load=%s",
            self.rank,
            cfg.effective_backend(),
            local_path,
            cfg.cleanup_after_load,
        )
        if cfg.effective_backend() == "pinned_cpu" and cfg.cleanup_after_load and not self._has_pending_optimizer_restore():
            torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def cleanup_actor_handoff_restore_session(self, local_path: str):
        if self.rank == 0 and has_restore_session_manifest(local_path):
            session_manifest = read_restore_session_manifest(local_path)
            update_restore_session_manifest(local_path, status="cleanup_partial_restore")
        gc.collect()
        get_torch_device().empty_cache()


class DetachAsyncRolloutWorker(DetachSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        LocalInitActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
