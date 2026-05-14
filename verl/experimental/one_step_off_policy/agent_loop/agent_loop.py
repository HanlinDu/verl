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
import asyncio
import logging
import os
import threading

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    def _run_replica_tasks_blocking(self, method_name: str) -> None:
        async def run_all():
            await asyncio.gather(*[getattr(replica, method_name)() for replica in self.rollout_replicas])

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(run_all())
            return

        errors: list[BaseException] = []

        def _runner() -> None:
            try:
                asyncio.run(run_all())
            except BaseException as exc:  # pragma: no cover - best effort propagation
                errors.append(exc)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with ray.get wrapped in asyncio.to_thread to avoid blocking
        import asyncio

        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        worker_debugs = [output.meta_info.pop("worker_debug", {}) for output in outputs]
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        timing.update(self._aggregate_worker_debug_metrics(worker_debugs, batch_size=len(prompts)))

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def wake_up(self):
        self._run_replica_tasks_blocking("wake_up")

    async def wake_up_async(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        self._run_replica_tasks_blocking("sleep")

    async def sleep_async(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    def clear_kv_cache(self):
        self._run_replica_tasks_blocking("clear_kv_cache")

    async def clear_kv_cache_async(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    async def shutdown_async(self):
        handles = []
        handles.extend(getattr(self, "agent_loop_workers", []) or [])
        for replica in getattr(self, "rollout_replicas", []) or []:
            handles.extend(getattr(replica, "servers", []) or [])

        seen_actor_ids = set()
        for handle in handles:
            actor_id = getattr(handle, "_actor_id", None)
            if actor_id in seen_actor_ids:
                continue
            seen_actor_ids.add(actor_id)
            try:
                ray.kill(handle)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("[one-step-off][resize] failed to kill async rollout actor %s: %s", handle, exc)
