# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        token_ids = kwargs.get("token_ids", None)
        partial_rollout = False
        prompt_token_ids = kwargs.get("prompt_token_ids", None)

        metrics = {}
        request_id = uuid4().hex
        prompt_token_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, sampling_params=sampling_params
            )
        )
        if token_ids:
            # partial rollout case:
            prompt_ids = prompt_token_ids + token_ids
            partial_rollout = True
        else:
            prompt_ids = prompt_token_ids

        with simple_timer("generate_sequences", metrics):
            # NOTE to DHL: genrate 的返回值变了，但需要确认是否还包括 interrupted
            # NOTE to DHL: interrupted 在 partial 的 commit 中不需要了，直接用 abort 接口
            # response_ids, interrupted = await self.server_manager.generate(
            token_output = await self.server_manager.generate(    
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )
        response_ids = token_output.token_ids
        if partial_rollout:
            # NOTE to DHL: 此处的 bug 已经 fixed，后续合并时注意
            original_prompt_len = len(prompt_ids) - len(token_ids)
            response_ids = prompt_ids[original_prompt_len :] + response_ids
            # prompt as original
            prompt_ids = prompt_ids[: original_prompt_len]
        response_mask = [1] * len(response_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
        )
        return output
