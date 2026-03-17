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

import torch
import torch.distributed
from omegaconf import DictConfig
from ray.util.collective import collective
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.experimental.one_step_off_policy.distributed_utils import vllm_stateless_init_process_group
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.ray_utils import get_event_loop
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
    CriticWorker,
    RewardModelWorker,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = ["DetachActorWorker", "DetachAsyncRolloutWorker", "CriticWorker", "RewardModelWorker"]


class DetachSync(AsyncActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size):
        rank = torch.distributed.get_rank() + rank_offset
        self._actor_rollout_collective_rank = rank
        self._actor_rollout_collective_world_size = world_size

        if device_name == "npu":
            self._weight_sync_group = vllm_stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                get_torch_device().current_device(),
            )
        else:
            self._maybe_init_collective_group(group_name="actor_rollout")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

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
            self._maybe_init_collective_group(group_name="actor_rollout")

        collective_rank, collective_world_size, collective_initialized = self._get_collective_rank_world_size(
            group_name="actor_rollout"
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
                if dynamic_resize_enabled:
                    torch.distributed.broadcast(tensor, src=0)
                else:
                    if not collective_initialized:
                        raise RuntimeError(
                            "Ray collective group 'actor_rollout' is not initialized. "
                            "Please ensure create_weight_sync_group is called on all actor/rollout workers."
                        )
                    collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")

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


class DetachAsyncRolloutWorker(DetachSync):
    def __init__(self, config: DictConfig, role: str):
        print(f"[DetachAsyncRolloutWorker] {DetachAsyncRolloutWorker.__mro__}")
        ActorRolloutRefWorker.__init__(self, config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
