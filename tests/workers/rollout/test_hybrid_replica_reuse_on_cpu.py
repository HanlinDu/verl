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

from types import SimpleNamespace

from verl.workers.rollout.llm_server import plan_hybrid_replica_reuse


class _Handle:
    def __init__(self, actor_id):
        self._actor_id = actor_id


def _physical_workers(n=8):
    return [_Handle(f"worker-{rank}") for rank in range(n)]


def _replica(replica_rank, workers, world_size, alive=True):
    return SimpleNamespace(
        replica_rank=replica_rank,
        workers=workers[replica_rank * world_size : (replica_rank + 1) * world_size],
        servers=["server"] if alive else [],
    )


def _ranks(items):
    return [item[1].replica_rank if isinstance(item, tuple) else item.replica_rank for item in items]


def test_rollout_shrink_keeps_the_replicas_that_stay_on_rollout_gpus():
    physical = _physical_workers()
    # 4 actor + 4 rollout (replicas 2 and 3) -> 6 actor + 2 rollout (replica 3 only)
    live = [_replica(2, physical, 2), _replica(3, physical, 2)]
    plan = plan_hybrid_replica_reuse(live, physical[6:8], start_rank=3, world_size=2)

    assert _ranks(plan.keep) == [3] and plan.keep[0][0] == 0
    assert _ranks(plan.retire) == [2]
    assert plan.launch == []


def test_rollout_expand_keeps_survivors_and_launches_only_new_slices():
    physical = _physical_workers()
    # 6 actor + 2 rollout (replica 3) -> 2 actor + 6 rollout (replicas 1, 2, 3)
    live = [_replica(3, physical, 2)]
    plan = plan_hybrid_replica_reuse(live, physical[2:8], start_rank=1, world_size=2)

    assert _ranks(plan.keep) == [3] and plan.keep[0][0] == 2
    assert plan.retire == []
    assert plan.launch == [(0, 1), (1, 2)]


def test_replica_with_dead_servers_or_foreign_workers_is_relaunched():
    physical = _physical_workers()
    dead = _replica(3, physical, 2, alive=False)
    plan = plan_hybrid_replica_reuse([dead], physical[6:8], start_rank=3, world_size=2)
    assert plan.keep == [] and _ranks(plan.retire) == [3] and plan.launch == [(0, 3)]

    foreign = SimpleNamespace(
        replica_rank=3, workers=[_Handle("other-6"), _Handle("other-7")], servers=["server"]
    )
    plan = plan_hybrid_replica_reuse([foreign], physical[6:8], start_rank=3, world_size=2)
    assert plan.keep == [] and _ranks(plan.retire) == [3] and plan.launch == [(0, 3)]


def test_already_retired_replicas_do_not_reappear():
    physical = _physical_workers()
    # retire_hybrid_replicas_below already dropped replica 2 from the live list
    live = [_replica(3, physical, 2)]
    plan = plan_hybrid_replica_reuse(live, physical[6:8], start_rank=3, world_size=2)
    assert _ranks(plan.keep) == [3] and plan.retire == [] and plan.launch == []
