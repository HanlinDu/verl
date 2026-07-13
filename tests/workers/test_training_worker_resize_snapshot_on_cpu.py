from types import SimpleNamespace

import torch

from verl.experimental.one_step_off_policy.staging_backend import load_paged_optimizer_state_dict
from verl.workers import engine_workers
from verl.workers.engine_workers import TrainingWorker


def test_available_disk_bytes_uses_existing_parent_for_missing_stage_path(tmp_path):
    missing_stage_path = tmp_path / "missing" / "stage"

    free_bytes = engine_workers._available_disk_bytes(str(missing_stage_path))

    assert isinstance(free_bytes, int)
    assert free_bytes > 0


def test_training_worker_resize_resource_snapshot_reports_available_resources(monkeypatch, tmp_path):
    class _FakeDevice:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 1234, 5678

    monkeypatch.setattr(
        engine_workers.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4321),
    )
    monkeypatch.setattr(
        engine_workers.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=8765),
    )
    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: _FakeDevice)

    worker = object.__new__(TrainingWorker)
    snapshot = worker.get_resize_resource_snapshot(str(tmp_path / "not-yet-created"))

    assert snapshot == {
        "host_free_bytes": 4321,
        "disk_free_bytes": 8765,
        "gpu_free_bytes": 1234,
    }


def test_training_worker_resize_resource_snapshot_allows_unknown_gpu(monkeypatch, tmp_path):
    class _FakeDevice:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(engine_workers, "get_torch_device", lambda: _FakeDevice)

    worker = object.__new__(TrainingWorker)
    snapshot = worker.get_resize_resource_snapshot(str(tmp_path))

    assert snapshot["host_free_bytes"] is None or snapshot["host_free_bytes"] > 0
    assert snapshot["disk_free_bytes"] is None or snapshot["disk_free_bytes"] > 0
    assert snapshot["gpu_free_bytes"] is None


def test_training_worker_stages_optimizer_for_dynamic_resize(tmp_path):
    class _FakeOptimizer:
        def state_dict(self):
            return {
                "state": {"p0": {"exp_avg": torch.ones(4)}},
                "param_groups": [{"params": ["p0"], "lr": 1e-4}],
            }

    worker = object.__new__(TrainingWorker)
    worker.engine = SimpleNamespace(optimizer=_FakeOptimizer())

    metrics = worker.stage_optimizer_for_dynamic_resize(
        str(tmp_path),
        {"enable": True, "stage_optimizer": True, "chunk_mb": 1},
    )
    restored = load_paged_optimizer_state_dict(str(tmp_path), "optim_state_rank_0")

    assert metrics["status"] == "staged"
    assert metrics["page_count"] == 1
    assert metrics["total_bytes"] > 0
    assert restored["param_groups"] == [{"params": ["p0"], "lr": 1e-4}]
    assert torch.equal(restored["state"]["p0"]["exp_avg"], torch.ones(4))
