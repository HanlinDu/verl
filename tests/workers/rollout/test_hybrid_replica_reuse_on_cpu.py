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

import asyncio
from types import SimpleNamespace

from verl.workers.rollout.llm_server import LLMServerManager, plan_hybrid_replica_reuse


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


class _FakeReplica:
    """Records the lifecycle calls LLMServerManager makes on a hybrid replica."""

    def __init__(self, replica_rank, physical, world_size):
        self.replica_rank = replica_rank
        self.workers = physical[replica_rank * world_size : (replica_rank + 1) * world_size]
        self.servers = [f"server-{replica_rank}"]
        self.server_address = self._server_address = f"10.0.0.1:{9000 + replica_rank}"
        self.server_handle = self._server_handle = f"handle-{replica_rank}"
        self.worker_group_rank = replica_rank
        self.calls = []

    async def abort_all_requests(self):
        self.calls.append("abort")

    async def sleep(self, force=False):
        self.calls.append(("sleep", force))

    async def wake_up(self):
        self.calls.append("wake_up")

    async def shutdown_runtime(self, kill_workers=False):
        self.calls.append("shutdown")
        self.servers = []

    async def init_hybrid(self, worker_group):
        self.calls.append("init_hybrid")


class _FakeLoadBalancer:
    def __init__(self):
        self.removed, self.added = [], []
        self.remove_servers = SimpleNamespace(remote=self._remove)
        self.add_servers = SimpleNamespace(remote=self._add)

    async def _remove(self, addresses):
        self.removed.append(sorted(addresses))

    async def _add(self, servers):
        self.added.append(sorted(servers))


def _manager(physical, replicas, *, name="vllm", enable_sleep_mode=True, world_size=2):
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_config = SimpleNamespace(
        name=name,
        enable_sleep_mode=enable_sleep_mode,
        tensor_model_parallel_size=world_size,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        n_gpus_per_node=len(physical),
        prometheus=SimpleNamespace(enable=False),
    )
    manager.model_config = None
    manager.worker_group = SimpleNamespace(workers=physical)
    manager.rollout_replicas = list(replicas)
    manager._dormant_replicas = {}
    manager.global_load_balancer = _FakeLoadBalancer()
    manager._refresh_server_lists()
    return manager


def _prelaunched(physical, *, dormant_below, enable_sleep_mode=True, world_size=2):
    """Run the start-up path (``_initialize_llm_servers``) over the whole physical pool."""
    manager = _manager(physical, [], enable_sleep_mode=enable_sleep_mode, world_size=world_size)
    del manager.global_load_balancer  # not created yet when the pool is launched
    manager.worker_group = SimpleNamespace(workers=physical, world_size=len(physical))
    manager.start_rank, manager.dormant_below = 0, dormant_below
    manager.rollout_replica_class = lambda replica_rank, **kwargs: _FakeReplica(replica_rank, physical, world_size)
    asyncio.run(manager._initialize_llm_servers())
    return manager


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


def test_rollout_expand_wakes_dormant_replicas_and_launches_only_first_time_slices():
    physical = _physical_workers()
    # [4,4] -> [6,2] parked replica 2; [6,2] -> [2,6] brings GPUs 2-5 back: wake 2, launch 1 (never a rollout slice)
    live = [_replica(3, physical, 2)]
    dormant = {2: _replica(2, physical, 2)}
    plan = plan_hybrid_replica_reuse(live, physical[2:8], start_rank=1, world_size=2, dormant=dormant)

    assert _ranks(plan.keep) == [3] and plan.keep[0][0] == 2
    assert _ranks(plan.wake) == [2] and plan.wake[0][0] == 1
    assert plan.launch == [(0, 1)]
    assert plan.retire == []


def test_dormant_replicas_outside_the_new_range_stay_parked():
    physical = _physical_workers()
    # [2,6] -> [6,2]: replicas 1 and 2 were parked by retire_hybrid_replicas_below, only 3 serves
    live = [_replica(3, physical, 2)]
    dormant = {1: _replica(1, physical, 2), 2: _replica(2, physical, 2)}
    plan = plan_hybrid_replica_reuse(live, physical[6:8], start_rank=3, world_size=2, dormant=dormant)
    assert _ranks(plan.keep) == [3] and plan.wake == [] and plan.launch == [] and plan.retire == []


def test_dormant_replica_with_dead_servers_or_foreign_workers_is_relaunched():
    physical = _physical_workers()
    dead = {2: _replica(2, physical, 2, alive=False)}
    plan = plan_hybrid_replica_reuse([], physical[4:8], start_rank=2, world_size=2, dormant=dead)
    assert plan.wake == [] and _ranks(plan.retire) == [2] and plan.launch == [(0, 2), (1, 3)]

    foreign = {2: SimpleNamespace(replica_rank=2, workers=[_Handle("x"), _Handle("y")], servers=["server"])}
    plan = plan_hybrid_replica_reuse([], physical[4:8], start_rank=2, world_size=2, dormant=foreign)
    assert plan.wake == [] and _ranks(plan.retire) == [2] and plan.launch == [(0, 2), (1, 3)]


def test_manager_parks_retired_replicas_and_wakes_them_when_their_gpus_return():
    physical = _physical_workers()
    r2, r3 = _FakeReplica(2, physical, 2), _FakeReplica(3, physical, 2)
    manager = _manager(physical, [r2, r3])

    # [4,4] -> [6,2]: GPUs 4,5 become training ranks; replica 2 is parked, not killed.
    retired = manager.retire_hybrid_replicas_below(6, dormant=True)
    assert retired == {"shutdown": [], "slept": [2]}
    assert r2.calls == ["abort", ("sleep", True)] and r2.servers  # server actors survive
    assert manager.rollout_replicas == [r3] and manager.server_addresses == [r3.server_address]
    assert manager._dormant_replicas == {2: r2}
    assert manager.global_load_balancer.removed == [[r2.server_address]]

    # [6,2] -> [2,6]: GPUs 2-5 return to rollout; replica 2 wakes, replica 1 is launched.
    manager.rollout_replica_class = lambda replica_rank, **kwargs: _FakeReplica(replica_rank, physical, 2)
    rebuild = manager.rebuild_hybrid_runtime(SimpleNamespace(workers=physical[2:8]), start_rank=1)
    assert rebuild == {"kept": [3], "retired": [], "woken": [2], "launched": [1]}
    assert r2.calls[-1] == "wake_up" and r2.worker_group_rank == 1
    assert manager._dormant_replicas == {}
    assert [replica.replica_rank for replica in manager.rollout_replicas] == [1, 2, 3]
    assert manager.global_load_balancer.added == [sorted([r2.server_address, "10.0.0.1:9001"])]


def test_manager_prelaunches_every_pair_and_parks_the_pairs_on_training_gpus():
    physical = _physical_workers()
    # Initial split [4,4]: replicas 0,1 sit on the actor GPUs -> launched and parked; 2,3 serve.
    manager = _prelaunched(physical, dormant_below=2)
    by_rank = {r.replica_rank: r for r in manager.rollout_replicas + list(manager._dormant_replicas.values())}
    assert sorted(by_rank) == [0, 1, 2, 3]
    assert [by_rank[rank].calls for rank in (0, 1)] == [["init_hybrid", "abort", ("sleep", True)]] * 2
    assert [by_rank[rank].calls for rank in (2, 3)] == [["init_hybrid"]] * 2
    assert _ranks(manager.rollout_replicas) == [2, 3] and sorted(manager._dormant_replicas) == [0, 1]
    assert manager.server_addresses == [by_rank[2].server_address, by_rank[3].server_address]
    assert manager.dormant_below == 0  # one-shot: a trtllm rebuild-from-scratch must not park anything

    # [4,4] -> [2,6]: GPUs 2,3 return to rollout; replica 1 wakes and nothing is launched.
    manager.global_load_balancer = _FakeLoadBalancer()
    rebuild = manager.rebuild_hybrid_runtime(SimpleNamespace(workers=physical[2:8]), start_rank=1)
    assert rebuild == {"kept": [2, 3], "retired": [], "woken": [1], "launched": []}
    assert by_rank[1].calls[-1] == "wake_up" and manager._dormant_replicas == {0: by_rank[0]}
    assert manager.global_load_balancer.added == [[by_rank[1].server_address]]


def test_manager_launches_only_the_rollout_pairs_when_the_backend_cannot_sleep():
    physical = _physical_workers()
    manager = _prelaunched(physical, dormant_below=2, enable_sleep_mode=False)
    assert _ranks(manager.rollout_replicas) == [2, 3] and manager._dormant_replicas == {}
    assert all(r.calls == ["init_hybrid"] for r in manager.rollout_replicas)
    assert [r.worker_group_rank for r in manager.rollout_replicas] == [2, 3]  # slicing stays physical


def test_manager_shuts_down_instead_of_parking_when_the_backend_cannot_sleep():
    physical = _physical_workers()
    r2, r3 = _FakeReplica(2, physical, 2), _FakeReplica(3, physical, 2)
    manager = _manager(physical, [r2, r3], enable_sleep_mode=False)

    retired = manager.retire_hybrid_replicas_below(6, dormant=True)
    assert retired == {"shutdown": [2], "slept": []}
    assert r2.calls == ["shutdown"] and manager._dormant_replicas == {}

    r3.calls.clear()
    assert manager.retire_hybrid_replicas_below(8, dormant=False) == {"shutdown": [3], "slept": []}
    assert r3.calls == ["shutdown"] and manager.rollout_replicas == []
