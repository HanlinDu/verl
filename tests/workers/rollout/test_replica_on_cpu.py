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

import asyncio
from types import SimpleNamespace

from verl.workers.rollout.replica import RolloutReplica


class _TestReplica(RolloutReplica):
    async def launch_servers(self):
        self.launch_called = True


def test_hybrid_worker_slice_uses_group_local_replica_rank():
    replica = _TestReplica.__new__(_TestReplica)
    replica.replica_rank = 3
    replica.worker_group_rank = 0
    replica.world_size = 2
    replica.launch_called = False

    workers = [object() for _ in range(4)]
    asyncio.run(replica.init_hybrid(SimpleNamespace(workers=workers)))

    assert replica.workers == workers[:2]
    assert replica.launch_called
