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

"""Exercise the real collector, replacing only the external Ray RPC boundary."""

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from verl.experimental.one_step_off_policy import ray_trainer
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.one_step_off_policy.trace_utils import ResizeTraceConfig
from verl.trainer.ppo.utils import Role
from verl.workers import engine_workers
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager


class _Ref:
    def __init__(self, value, *, ready=True, error=None):
        self.value = value
        self.ready = ready
        self.error = error


class _RayBoundary:
    def __init__(self, monkeypatch):
        self.clock = 0.0
        self.calls = []
        self.wait_timeouts = []
        self.context = None
        monkeypatch.setattr(ray_trainer.ray, "get_runtime_context", lambda: self.context)
        monkeypatch.setattr(ray_trainer.ray, "wait", self.wait)
        monkeypatch.setattr(ray_trainer.ray, "get", self.get)
        monkeypatch.setattr(
            ray_trainer,
            "time",
            SimpleNamespace(monotonic=lambda: self.clock, time_ns=lambda: 1_700_000_000_000_000_000),
        )

    def wait(self, refs, *, num_returns, timeout):
        assert timeout >= 0
        self.wait_timeouts.append(timeout)
        ready = [ref for ref in refs if ref.ready][:num_returns]
        if not ready:
            self.clock += timeout
        return ready, [ref for ref in refs if ref not in ready]

    def get(self, ref, *, timeout):
        # An unready read or an unbounded second read is a production bug.
        assert ref.ready
        assert timeout >= 0
        if ref.error is not None:
            raise ref.error
        return ref.value


class _RawWorker:
    def __init__(self, boundary, rank, node_id, gpu_ids):
        self.rank = rank
        self._actor_id = f"physical-worker-{rank}"
        self.node_id = node_id
        self.gpu_ids = gpu_ids
        self.boundary = boundary
        self.identity_ready = True
        self.__ray_call__ = SimpleNamespace(remote=self._identity_call)

    def _identity_call(self, callback):
        self.boundary.calls.append(("identity", self.rank))
        self.boundary.context = SimpleNamespace(
            get_node_id=lambda: self.node_id,
            get_accelerator_ids=lambda: {"GPU": self.gpu_ids},
        )
        return _Ref(callback(self), ready=self.identity_ready)


class _PhysicalGroup:
    def __init__(self, boundary, workers, snapshots):
        self.boundary = boundary
        self.workers = workers
        self.world_size = len(workers)
        self.snapshots = snapshots
        self.dispatch_error = None

    def get_hybrid_card_snapshot(self):
        self.boundary.calls.append(("snapshot",))
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return self.snapshots


def _snapshot(rank, node_id, gpu_id="0"):
    return {
        "physical_rank": rank,
        "node_id": node_id,
        "accelerator_id_raw": gpu_id,
        "process_scope": "hybrid_worker",
        "sampled_at_ns": 1_700_000_000_000_000_001 + rank,
        "pid": 100 + rank,
        "driver": {"free_bytes": 60, "total_bytes": 100, "used_bytes": 40},
        "process_torch": {"allocated_bytes": 7, "reserved_bytes": 11},
        "status": "ok",
        "error_type": "",
        "error_message": "",
    }


def _trainer(tmp_path, boundary, *, count=2, timeout=5.0):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer._memory_probe_timeout_s = timeout
    trainer._hybrid_memory_identity_cache = None
    trainer._resize_trace_config = ResizeTraceConfig(enabled=True, path=str(tmp_path / "memory.jsonl"))
    trainer._active_dynamic_resize_topology = {"actor_size": 1, "rollout_size": count - 1}
    trainer.global_steps = 0
    workers = [_RawWorker(boundary, rank, f"node-{rank}", ["0"]) for rank in range(count)]
    trainer._physical_actor_rollout_wg = _PhysicalGroup(
        boundary, workers, [_Ref(_snapshot(rank, f"node-{rank}")) for rank in range(count)]
    )
    return trainer


def _config_trainer(tmp_path, *, enabled=True, timeout=5.0, dynamic=True, shared=True, hybrid=True, trace=True):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer._dynamic_resize_cfg = {
        "enable": dynamic,
        "shared_pool": {"enable": shared},
        "memory_probe": {"enable": enabled, "timeout_s": timeout},
    }
    trainer._dynamic_resize_enabled = dynamic
    trainer.hybrid_engine = hybrid
    trainer._resize_trace_config = ResizeTraceConfig(
        enabled=trace, path=str(tmp_path / "memory.jsonl") if trace else ""
    )
    return trainer


def test_disabled_collector_does_not_lookup_workers_or_issue_rpc(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, timeout=None)
    del trainer._physical_actor_rollout_wg

    assert trainer._sample_hybrid_memory("startup_prelaunch") is None
    assert boundary.calls == []
    assert not (tmp_path / "memory.jsonl").exists()


@pytest.mark.parametrize("gate", ["enabled", "dynamic", "shared", "hybrid", "trace"])
def test_probe_requires_every_opt_in_gate(tmp_path, gate):
    trainer = _config_trainer(tmp_path, **{gate: False})
    assert trainer._build_memory_probe_timeout_s() is None


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, None, "bad"])
def test_enabled_probe_rejects_invalid_timeout_even_when_trace_is_off(tmp_path, timeout):
    trainer = _config_trainer(tmp_path, timeout=timeout, trace=False)
    with pytest.raises(ValueError, match="memory_probe.timeout_s"):
        trainer._build_memory_probe_timeout_s()


def test_valid_enabled_timeout_controls_sampler(tmp_path):
    trainer = _config_trainer(tmp_path, timeout=0.25)
    assert trainer._build_memory_probe_timeout_s() == 0.25


def test_complete_snapshot_uses_independent_node_identity_and_caches_only_allocations(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)

    first = trainer._sample_hybrid_memory("startup_prelaunch", actor_size=1)
    trainer._physical_actor_rollout_wg.snapshots[0].value["driver"] = {
        "free_bytes": 55,
        "total_bytes": 100,
        "used_bytes": 45,
    }
    second = trainer._sample_hybrid_memory("startup_runtime_ready", actor_size=1)

    assert first["complete"] is True
    assert first["expected_count"] == first["observed_count"] == 2
    assert [(row["node_id"], row["accelerator_id_raw"]) for row in first["expected"]] == [
        ("node-0", "0"),
        ("node-1", "0"),
    ]
    assert second["complete"] is True
    assert second["observed"][0]["driver"]["used_bytes"] == 45
    assert boundary.calls == [("identity", 0), ("identity", 1), ("snapshot",), ("snapshot",)]
    records = [json.loads(line) for line in (tmp_path / "memory.jsonl").read_text().splitlines()]
    assert [(record["event"], record["kind"], record["phase"]) for record in records] == [
        ("gpu_cards_snapshot", "snapshot", "startup_prelaunch"),
        ("gpu_cards_snapshot", "snapshot", "startup_runtime_ready"),
    ]


def test_memory_read_failure_and_timeout_preserve_ready_rows_and_known_world_size(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, count=3, timeout=0.5)
    trainer._physical_actor_rollout_wg.snapshots[1].error = RuntimeError("sample read failed")
    trainer._physical_actor_rollout_wg.snapshots[2].ready = False

    record = trainer._sample_hybrid_memory("resize_after_destroy", actor_size=1)

    assert record["complete"] is False
    assert record["expected_count"] == 3
    assert record["observed_count"] == 1
    assert record["observed"][0]["physical_rank"] == 0
    assert any(error.get("stage") == "snapshot_read" and error.get("physical_rank") == 1 for error in record["errors"])
    assert any(
        error.get("stage") == "snapshot_timeout" and error.get("physical_rank") == 2 for error in record["errors"]
    )
    assert boundary.clock == 0.5
    assert all(timeout <= 0.5 for timeout in boundary.wait_timeouts)


def test_duplicate_sample_cannot_replace_a_missing_card(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    trainer._physical_actor_rollout_wg.snapshots[1].value = _snapshot(0, "node-0")

    record = trainer._sample_hybrid_memory("startup_runtime_ready")

    assert record["complete"] is False
    assert record["expected_count"] == record["observed_count"] == 2
    assert any(error.get("kind") == "duplicate" for error in record["errors"])
    assert any(error.get("kind") == "missing" and error.get("physical_rank") == 1 for error in record["errors"])


@pytest.mark.parametrize("identity_failure", ["missing_gpu", "duplicate_card", "wrong_rank", "timeout"])
def test_invalid_identity_map_is_not_cached_or_used_for_memory(tmp_path, monkeypatch, identity_failure):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, timeout=0.5)
    worker = trainer._physical_actor_rollout_wg.workers[1]
    if identity_failure == "missing_gpu":
        worker.gpu_ids = []
    elif identity_failure == "duplicate_card":
        worker.node_id = "node-0"
    elif identity_failure == "wrong_rank":
        worker.rank = 0
    else:
        worker.identity_ready = False

    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert record["observed"] == []
    assert record["identity_observed"]
    assert record["errors"]
    assert trainer._hybrid_memory_identity_cache is None
    assert all(call[0] == "identity" for call in boundary.calls)

    worker.rank, worker.node_id, worker.gpu_ids, worker.identity_ready = 1, "node-1", ["0"], True
    retry = trainer._sample_hybrid_memory("startup_parked")
    assert retry["complete"] is True


def test_snapshot_dispatch_failure_emits_incomplete_evidence(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    trainer._physical_actor_rollout_wg.dispatch_error = RuntimeError("dispatch failed")

    record = trainer._sample_hybrid_memory("resize_before_destroy")

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert record["observed"] == []
    assert any(error.get("stage") == "snapshot_dispatch" for error in record["errors"])
    assert (tmp_path / "memory.jsonl").exists()


def test_sampler_does_not_swallow_cancellation(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    cancelled = asyncio.CancelledError()
    trainer._physical_actor_rollout_wg.snapshots[0].error = cancelled

    with pytest.raises(asyncio.CancelledError) as raised:
        trainer._sample_hybrid_memory("startup_prelaunch")
    assert raised.value is cancelled


def test_actor_and_dormant_membership_coexist_and_use_actor_ids(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, count=4)
    physical = trainer._physical_actor_rollout_wg.workers
    manager = object.__new__(LLMServerManager)
    manager.worker_group = trainer._physical_actor_rollout_wg
    # Different Python handles with the same Ray actor IDs still name the same cards.
    aliases = [SimpleNamespace(_actor_id=worker._actor_id) for worker in physical]
    manager.rollout_replicas = [SimpleNamespace(workers=aliases[2:])]
    manager._dormant_replicas = {0: SimpleNamespace(workers=aliases[:2])}
    trainer._prelaunched_llm_server_manager = manager

    old = trainer._sample_hybrid_memory("resize_after_destroy", actor_size=1)
    rebuilt = trainer._sample_hybrid_memory("resize_after_rebuild", actor_size=2)

    assert old["expected"][0]["membership"] == {
        "actor": True,
        "rollout_active": False,
        "rollout_dormant": True,
    }
    assert old["expected"][1]["membership"]["actor"] is False
    assert rebuilt["expected"][1]["membership"]["actor"] is True
    assert rebuilt["expected"][1]["membership"]["rollout_dormant"] is True
    assert rebuilt["expected"][2]["membership"] == {
        "actor": False,
        "rollout_active": True,
        "rollout_dormant": False,
    }


class _DeviceCounters:
    def is_available(self):
        return True

    def mem_get_info(self):
        return 60, 100

    def memory_allocated(self):
        return 7

    def memory_reserved(self):
        return 11


def test_worker_snapshot_uses_persistent_rank_and_raw_gpu_token(monkeypatch):
    monkeypatch.setattr(
        engine_workers.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node-real-allocation",
            get_accelerator_ids=lambda: {"GPU": ["GPU-card-uuid"]},
        ),
    )
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(engine_workers, "get_torch_device", _DeviceCounters)
    worker = object.__new__(ActorRolloutRefWorker)
    worker._rank = 7

    row = worker.get_hybrid_card_snapshot()

    assert row["physical_rank"] == 7
    assert row["node_id"] == "node-real-allocation"
    assert row["accelerator_id_raw"] == "GPU-card-uuid"
    assert row["status"] == "ok"
    assert row["driver"] == {"free_bytes": 60, "total_bytes": 100, "used_bytes": 40}
    assert row["process_torch"] == {"allocated_bytes": 7, "reserved_bytes": 11}


@pytest.mark.parametrize("gpu_ids", [[], ["0", "1"]])
def test_worker_without_exactly_one_gpu_reports_error_without_counters(monkeypatch, gpu_ids):
    monkeypatch.setattr(
        engine_workers.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node-allocation",
            get_accelerator_ids=lambda: {"GPU": gpu_ids},
        ),
    )
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(engine_workers, "get_torch_device", _DeviceCounters)
    worker = object.__new__(ActorRolloutRefWorker)
    worker._rank = 3

    row = worker.get_hybrid_card_snapshot()

    assert row["physical_rank"] == 3
    assert row["status"] == "error"
    assert row["error_message"]
    assert row["driver"] == {"free_bytes": None, "total_bytes": None, "used_bytes": None}
    assert row["process_torch"] == {"allocated_bytes": None, "reserved_bytes": None}


def test_one_deadline_covers_identity_lookup_and_memory_waits(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, timeout=0.5)
    workers = trainer._physical_actor_rollout_wg.workers
    original = workers[1].__ray_call__.remote

    def slow_identity(callback):
        boundary.clock += 0.3
        return original(callback)

    workers[1].__ray_call__.remote = slow_identity
    trainer._physical_actor_rollout_wg.snapshots[1].ready = False

    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["observed_count"] == 1
    assert boundary.clock == pytest.approx(0.5)
    assert boundary.wait_timeouts[-1] == pytest.approx(0.2)


def test_membership_failure_preserves_counters_but_never_claims_complete(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)

    def unavailable(_workers):
        raise RuntimeError("membership unavailable")

    trainer.llm_server_manager = SimpleNamespace(get_hybrid_memory_membership=unavailable)

    record = trainer._sample_hybrid_memory("resize_after_rebuild", actor_size=1)

    assert record["complete"] is False
    assert record["observed_count"] == 2
    assert any(error.get("stage") == "membership" for error in record["errors"])
    assert record["expected"][0]["membership"]["rollout_active"] is None
    assert record["expected"][0]["membership"]["rollout_dormant"] is None


def test_identity_dispatch_failure_keeps_other_ready_identity_rows(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)

    def failed(_callback):
        raise RuntimeError("identity dispatch failed")

    trainer._physical_actor_rollout_wg.workers[1].__ray_call__.remote = failed
    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert len(record["identity_observed"]) == 1
    assert record["identity_observed"][0]["physical_rank"] == 0
    assert any(error.get("stage") == "identity_dispatch" for error in record["errors"])
    assert all(call[0] == "identity" for call in boundary.calls)


def test_expired_identity_deadline_does_not_dispatch_memory(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary, timeout=0.5)
    worker = trainer._physical_actor_rollout_wg.workers[1]
    original = worker.__ray_call__.remote

    def slow_identity(callback):
        boundary.clock += 0.5
        return original(callback)

    worker.__ray_call__.remote = slow_identity
    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert record["observed"] == []
    assert any(error.get("stage") == "snapshot_timeout" for error in record["errors"])
    assert all(call[0] == "identity" for call in boundary.calls)


def test_identity_cache_does_not_survive_changed_raw_handles(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    assert trainer._sample_hybrid_memory("startup_prelaunch")["complete"] is True
    replacement = _RawWorker(boundary, 1, "node-replacement", ["GPU-new-uuid"])
    replacement._actor_id = "replacement-worker"
    trainer._physical_actor_rollout_wg.workers[1] = replacement
    trainer._physical_actor_rollout_wg.snapshots[1] = _Ref(_snapshot(1, "node-replacement", "GPU-new-uuid"))

    record = trainer._sample_hybrid_memory("startup_parked")

    assert record["complete"] is True
    assert record["expected"][1]["node_id"] == "node-replacement"
    assert record["expected"][1]["accelerator_id_raw"] == "GPU-new-uuid"
    assert boundary.calls == [
        ("identity", 0),
        ("identity", 1),
        ("snapshot",),
        ("identity", 0),
        ("identity", 1),
        ("snapshot",),
    ]


def test_known_world_size_rejects_truncated_raw_worker_handles(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    trainer._physical_actor_rollout_wg.workers.pop()

    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert record["observed"] == []
    assert any(error.get("stage") == "identity_validation" for error in record["errors"])
    assert trainer._hybrid_memory_identity_cache is None
    assert all(call[0] == "identity" for call in boundary.calls)


def test_hard_switch_samples_old_membership_until_rebuild_and_new_membership_after(tmp_path, monkeypatch):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.config = None
    trainer.global_steps = 2
    trainer.hybrid_engine = True
    trainer.timing_raw = {}
    trainer._resize_controller = None
    trainer._resize_trace_config = ResizeTraceConfig()
    trainer._memory_probe_timeout_s = 5.0
    trainer._host_staging_config = SimpleNamespace(effective_backend=lambda: "gpu_direct")
    trainer._active_dynamic_resize_topology = {"actor_size": 1, "rollout_size": 3}
    trainer._dynamic_resize_target_sizes = lambda item: (2, 2)
    trainer._gpu_direct_actor_model_resize_enabled = lambda: True
    monkeypatch.setattr(ray_trainer, "dynamic_resize_shared_pool_enabled", lambda config: True)
    monkeypatch.setattr(ray_trainer, "marked_timer", lambda *_args, **_kwargs: nullcontext())
    events = []

    def sample(phase, *, actor_size=None):
        events.append((phase, actor_size, trainer._active_dynamic_resize_topology["actor_size"]))

    async def destroy(*, actor_target):
        events.append(("destroy", actor_target, trainer._active_dynamic_resize_topology["actor_size"]))

    async def rebuild(*, actor_size, rollout_size):
        events.append(("rebuild", actor_size, trainer._active_dynamic_resize_topology["actor_size"]))
        trainer._latest_gpu_direct_resize_metrics = {}

    async def sync(step):
        events.append(("sync", step, trainer._active_dynamic_resize_topology["actor_size"]))

    trainer._sample_hybrid_memory = sample
    trainer._destroy_dynamic_resize_runtime = destroy
    trainer._rebuild_hybrid_dynamic_resize_runtime = rebuild
    trainer.checkpoint_manager = SimpleNamespace(update_weights=sync)

    metrics = asyncio.run(trainer._apply_dynamic_resize_hard_switch(item={}, required_action="expand_train"))

    assert events == [
        ("resize_before_destroy", 1, 1),
        ("destroy", 2, 1),
        ("resize_after_destroy", 1, 1),
        ("rebuild", 2, 1),
        ("resize_after_rebuild", 2, 1),
        ("sync", 2, 1),
        ("resize_after_sync", 2, 1),
    ]
    assert trainer._active_dynamic_resize_topology == {"actor_size": 2, "rollout_size": 2}
    assert metrics["resize/hard_switch_success"] == 1.0


def test_startup_samples_before_prelaunch_after_parking_and_after_runtime_ready(tmp_path, monkeypatch):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.hybrid_engine = True
    trainer.use_critic = trainer.use_reference_policy = trainer.use_rm = False
    trainer.ref_in_actor = False
    trainer.actor_role = Role.ActorRollout
    trainer.global_steps = 0
    trainer._resize_trace_config = ResizeTraceConfig()
    topology = SimpleNamespace(initial_actor_size=1, initial_rollout_size=1)
    trainer.resource_pool_manager = SimpleNamespace(dynamic_resize_topology=topology)
    trainer.config = SimpleNamespace()
    trainer._gpu_direct_actor_model_resize_enabled = lambda: False
    trainer._validate_hybrid_split = lambda **_kwargs: 1
    trainer._set_hybrid_rollout_subgroup = lambda **_kwargs: None
    monkeypatch.setattr(ray_trainer, "dynamic_resize_shared_pool_enabled", lambda config: True)
    events = []

    def sample(phase, *, actor_size=None):
        events.append((phase, actor_size))

    def prelaunch(*, dormant_below):
        events.append(("prelaunch", dormant_below))

    def init_model(*, actor_ranks):
        events.append(("init_model", actor_ranks))

    physical = SimpleNamespace(init_model=init_model, subgroup=lambda *_args, **_kwargs: object())
    trainer.all_wg = {str(Role.ActorRollout): physical}
    trainer._sample_hybrid_memory = sample
    trainer._prelaunch_hybrid_rollout_pool = prelaunch

    trainer._init_models()

    assert events == [
        ("startup_prelaunch", 1),
        ("prelaunch", 1),
        ("startup_parked", 1),
        ("init_model", [0]),
        ("startup_runtime_ready", 1),
    ]


def test_summarizer_failure_retains_samples_and_warns_incomplete(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)

    def broken_summary(*_args):
        raise RuntimeError("summarizer failed")

    monkeypatch.setattr(ray_trainer, "summarize_memory_snapshot", broken_summary)
    record = trainer._sample_hybrid_memory("startup_prelaunch")

    assert record["complete"] is False
    assert record["expected_count"] == record["observed_count"] == 2
    assert any(error.get("stage") == "summary" for error in record["errors"])
    assert (tmp_path / "memory.jsonl").exists()


def test_wait_failure_after_one_sample_keeps_that_sample(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    assert trainer._sample_hybrid_memory("startup_prelaunch")["complete"] is True
    calls = 0

    def failed_wait(refs, *, num_returns, timeout):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("readiness failed")
        return boundary.wait(refs, num_returns=num_returns, timeout=timeout)

    monkeypatch.setattr(ray_trainer.ray, "wait", failed_wait)
    record = trainer._sample_hybrid_memory("startup_parked")

    assert record["complete"] is False
    assert record["observed_count"] == 1
    assert record["observed"][0]["physical_rank"] == 0
    assert any(error.get("stage") == "snapshot_wait" for error in record["errors"])


def test_disabled_probe_ignores_unused_invalid_timeout(tmp_path):
    trainer = _config_trainer(tmp_path, enabled=False, timeout="unused")
    assert trainer._build_memory_probe_timeout_s() is None


def test_enabled_probe_requires_nonempty_trace_path(tmp_path):
    trainer = _config_trainer(tmp_path)
    trainer._resize_trace_config = ResizeTraceConfig(enabled=True, path="   ")
    assert trainer._build_memory_probe_timeout_s() is None


def test_worker_cpu_backend_never_reports_fake_gpu_success(monkeypatch):
    monkeypatch.setattr(
        engine_workers.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "cpu-node",
            get_accelerator_ids=lambda: {"GPU": ["0"]},
        ),
    )
    monkeypatch.setattr(engine_workers, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(engine_workers, "get_torch_device", _DeviceCounters)
    worker = object.__new__(ActorRolloutRefWorker)
    worker._rank = 0

    row = worker.get_hybrid_card_snapshot()

    assert row["status"] == "error"
    assert row["error_message"]
    assert row["driver"]["used_bytes"] is None


def test_startup_sync_snapshot_happens_before_validation_or_generation(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from verl.utils import tracking

    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "memory",
                "logger": [],
                "critic_warmup": 0,
                "val_before_train": True,
                "val_only": True,
            }
        }
    )
    events = []
    monkeypatch.setattr(tracking, "Tracking", lambda **_kwargs: SimpleNamespace(log=lambda **_values: None))
    trainer._load_checkpoint = lambda: events.append("load")
    trainer.timing_raw = {}
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append("sync"))
    trainer._sample_hybrid_memory = lambda phase: events.append(phase)

    def validate():
        events.append("validate")
        return {"validation/reward": 1.0}

    trainer._validate = validate
    asyncio.run(trainer.fit())

    assert events == ["load", "sync", "startup_after_sync", "validate"]


def test_missing_raw_actor_id_keeps_known_expected_ranks_without_sampling(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    del trainer._physical_actor_rollout_wg.workers[1]._actor_id

    record = trainer._sample_hybrid_memory("startup_prelaunch", actor_size=1)

    assert record["complete"] is False
    assert record["expected_count"] == 2
    assert [entry["physical_rank"] for entry in record["expected"]] == [0, 1]
    assert record["observed"] == []
    assert any(error.get("stage") == "identity_validation" for error in record["errors"])
    assert boundary.calls == []


def test_empty_hybrid_replica_workers_cannot_claim_no_rollout_membership(tmp_path, monkeypatch):
    boundary = _RayBoundary(monkeypatch)
    trainer = _trainer(tmp_path, boundary)
    manager = object.__new__(LLMServerManager)
    manager.worker_group = trainer._physical_actor_rollout_wg
    manager.rollout_replicas = [SimpleNamespace(workers=[])]
    manager._dormant_replicas = {}
    trainer.llm_server_manager = manager

    record = trainer._sample_hybrid_memory("startup_runtime_ready", actor_size=1)

    assert record["complete"] is False
    assert record["observed_count"] == 2
    assert any(error.get("stage") == "membership" for error in record["errors"])
    assert record["expected"][1]["membership"]["rollout_active"] is None
