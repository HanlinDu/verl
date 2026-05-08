from __future__ import annotations

import os
import uuid

import pytest
import ray
import torch

from verl.experimental.one_step_off_policy.pinned_cpu_staging import get_or_create_pinned_cpu_staging_service


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
    page = {"weight": torch.ones(16, dtype=torch.float32)}
    ray.get(service.put_object.remote(session_id, "model_page:0", page, pin_tensors=True))
    ray.get(service.put_object.remote(session_id, "meta", {"page_count": 1}, pin_tensors=False))

    desc = ray.get(service.describe_session.remote(session_id))
    restored = ray.get(service.get_object.remote(session_id, "model_page:0"))

    assert desc["exists"] is True
    assert desc["object_count"] == 2
    assert desc["pinned_tensor_count"] >= 1
    assert torch.equal(restored["weight"], page["weight"])

    assert ray.get(service.release_session.remote(session_id)) is True
    assert ray.get(service.describe_session.remote(session_id))["exists"] is False


def test_pinned_cpu_staging_service_preserves_cuda_visibility_in_real_actor(monkeypatch: pytest.MonkeyPatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate real pinned-memory actor behavior")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        visible_devices = ",".join(str(idx) for idx in range(torch.cuda.device_count()))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    _reinit_ray(local_mode=False)

    service_name = f"test-pinned-cpu-real-{uuid.uuid4().hex}"
    service = get_or_create_pinned_cpu_staging_service(service_name)
    session_id = f"session-{uuid.uuid4().hex}"

    ray.get(service.create_session.remote(session_id, metadata={"purpose": "real-actor-test"}))
    desc = ray.get(service.describe_session.remote(session_id))

    assert desc["exists"] is True
    assert desc["cuda_visible_devices"] == visible_devices
    assert isinstance(desc["pin_memory_available"], bool)
    assert isinstance(desc["pin_memory_error"], str)

    if not desc["pin_memory_available"]:
        pytest.skip(f"pin_memory is unavailable in the real Ray actor environment: {desc['pin_memory_error']}")

    assert desc["pin_memory_error"] == ""

    page = {"weight": torch.ones(8, dtype=torch.float32)}
    ray.get(service.put_object.remote(session_id, "model_page:0", page, pin_tensors=True))
    desc_after = ray.get(service.describe_session.remote(session_id))

    assert desc_after["object_count"] == 1
    assert desc_after["pinned_tensor_count"] >= 1

    assert ray.get(service.release_session.remote(session_id)) is True
