from __future__ import annotations

import uuid

import ray
import torch

from verl.experimental.one_step_off_policy.pinned_cpu_staging import get_or_create_pinned_cpu_staging_service
from verl.experimental.one_step_off_policy.staging_backend import (
    create_restore_session_manifest,
    read_paged_state_manifest,
    read_restore_session_manifest,
)


def _reinit_ray(*, local_mode: bool) -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=local_mode, ignore_reinit_error=True)


def test_pinned_cpu_staging_service_roundtrip():
    _reinit_ray(local_mode=True)

    service_name = f"test-pinned-cpu-{uuid.uuid4().hex}"
    service = get_or_create_pinned_cpu_staging_service(service_name)
    session_id = f"session-{uuid.uuid4().hex}"

    ray.get(service.create_session.remote(session_id, metadata={"purpose": "unit-test"}))
    desc = ray.get(service.describe_session.remote(session_id))
    pin_tensors = bool(desc["pin_memory_available"])

    page = {"weight": torch.ones(16, dtype=torch.float32)}
    ray.get(service.put_object.remote(session_id, "model_page:0", page, pin_tensors=pin_tensors))
    ray.get(service.put_object.remote(session_id, "meta", {"page_count": 1}, pin_tensors=False))

    desc = ray.get(service.describe_session.remote(session_id))
    restored = ray.get(service.get_object.remote(session_id, "model_page:0"))

    assert desc["exists"] is True
    assert desc["metadata"] == {"purpose": "unit-test"}
    assert desc["object_keys"] == ["meta", "model_page:0"]
    assert desc["object_count"] == 2
    assert desc["pinned_tensor_count"] >= (1 if pin_tensors else 0)
    assert torch.equal(restored["weight"], page["weight"])

    popped = ray.get(service.pop_object.remote(session_id, "meta"))
    assert popped == {"page_count": 1}
    assert ray.get(service.describe_session.remote(session_id))["object_count"] == 1
    assert ray.get(service.release_session.remote(session_id)) is True
    assert ray.get(service.describe_session.remote(session_id))["exists"] is False


def test_pinned_cpu_staging_service_exports_optimizer_pages(tmp_path):
    _reinit_ray(local_mode=True)

    service_name = f"test-pinned-cpu-optim-{uuid.uuid4().hex}"
    service = get_or_create_pinned_cpu_staging_service(service_name)
    create_restore_session_manifest(
        str(tmp_path),
        backend="pinned_cpu",
        session_id=f"session-{uuid.uuid4().hex}",
        optimizer_restore_policy="deferred",
    )

    optim_state_dict = {
        "state": {
            "layer.weight": {
                "exp_avg": torch.ones(4, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(4, dtype=torch.float32),
            }
        },
        "param_groups": [{"params": ["layer.weight"], "lr": 1e-6}],
    }

    metrics = ray.get(
        service.export_optimizer_state_to_stage.remote(
            local_path=str(tmp_path),
            optim_state_dict=optim_state_dict,
            chunk_mb=1,
            host_preload_threshold=1.0,
        )
    )

    optim_manifest = read_paged_state_manifest(str(tmp_path), "optim_state")
    restore_manifest = read_restore_session_manifest(str(tmp_path))

    assert metrics["resize/host_stage_optimizer_service_page_count"] == 1.0
    assert optim_manifest["page_count"] == 1
    assert optim_manifest["pages"][0]["storage"] in {"host_file", "disk"}
    assert restore_manifest["status"] == "optimizer_staged"
    assert restore_manifest["optimizer_page_count"] == 1
    assert restore_manifest["staged_optimizer_bytes"] == optim_manifest["total_bytes"]
