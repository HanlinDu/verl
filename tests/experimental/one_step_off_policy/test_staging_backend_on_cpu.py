from pathlib import Path

import pytest
import torch

from verl.utils.checkpoint.fsdp_checkpoint_manager import (
    dynamic_resize_optimizer_status_path,
    write_dynamic_resize_optimizer_restore_status,
)
from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    build_checkpoint_artifact_manifest,
    create_restore_session_manifest,
    finalize_restore_session_manifest,
    has_paged_state_dict,
    has_restore_session_manifest,
    iter_paged_state_dict,
    load_paged_optimizer_state_dict,
    probe_local_pin_memory_capability,
    read_host_staging_manifest,
    read_paged_state_manifest,
    read_restore_session_manifest,
    record_restore_progress,
    save_paged_optimizer_state_dict,
    save_paged_state_dict,
    update_restore_session_manifest,
    write_host_staging_manifest,
)


def test_host_staging_manifest_roundtrip(tmp_path: Path):
    cfg = HostStagingConfig.from_dict(
        {
            "enable": True,
            "backend": "disk_fallback",
            "service_name": "service-a",
            "chunk_mb": 512,
            "stage_optimizer": False,
            "optimizer_restore_policy": "immediate",
            "progressive_swap": True,
            "async_optimizer_preload": False,
            "preload_queue_depth": 3,
            "host_preload_threshold": 0.73,
            "device_preload_threshold": 0.82,
            "cleanup_after_load": False,
            "preclear_rollout_kv_cache": True,
        }
    )

    write_host_staging_manifest(str(tmp_path), cfg)
    manifest = read_host_staging_manifest(str(tmp_path))

    assert manifest["backend"] == "disk_fallback"
    assert manifest["requested_backend"] == "disk_fallback"
    assert manifest["service_name"] == "service-a"
    assert manifest["chunk_mb"] == 512
    assert manifest["stage_optimizer"] is False
    assert manifest["optimizer_restore_policy"] == "immediate"
    assert manifest["async_optimizer_preload"] is False
    assert manifest["preload_queue_depth"] == 3
    assert manifest["host_preload_threshold"] == pytest.approx(0.73)
    assert manifest["device_preload_threshold"] == pytest.approx(0.82)
    assert manifest["cleanup_after_load"] is False


def test_checkpoint_artifact_manifest_records_existing_files(tmp_path: Path):
    model_path = tmp_path / "dynamic_resize_full_model.pt"
    optim_path = tmp_path / "optim_world_size_2_rank_0.pt"
    model_path.write_bytes(b"model")
    optim_path.write_bytes(b"optimizer")

    manifest = build_checkpoint_artifact_manifest(
        str(tmp_path),
        prefix="checkpoint_model",
        file_names=[model_path.name, optim_path.name, "missing.pt"],
    )

    assert manifest["prefix"] == "checkpoint_model"
    assert manifest["page_count"] == 2
    assert manifest["total_bytes"] == model_path.stat().st_size + optim_path.stat().st_size
    assert manifest["pages"][0]["storage"] == "checkpoint_file"
    assert manifest["pages"][0]["source_path"].startswith(str(tmp_path))


def test_dynamic_resize_optimizer_restore_status_file_records_skip(tmp_path: Path):
    (tmp_path / "optim_world_size_2_rank_0.pt").write_bytes(b"optim-0")
    (tmp_path / "optim_world_size_2_rank_1.pt").write_bytes(b"optim-1")

    manifest = write_dynamic_resize_optimizer_restore_status(
        str(tmp_path),
        status="skipped",
        reason="missing_target_world_size_optimizer_shard_after_full_model_fallback",
        rank=0,
        world_size=1,
        expected_optimizer_path=str(tmp_path / "optim_world_size_1_rank_0.pt"),
        loaded_model_from_dynamic_full=True,
    )

    path = dynamic_resize_optimizer_status_path(str(tmp_path), 0)

    assert Path(path).exists()
    assert manifest["status"] == "skipped"
    assert manifest["source_optimizer_file_count"] == 2
    assert manifest["source_optimizer_files"] == ["optim_world_size_2_rank_0.pt", "optim_world_size_2_rank_1.pt"]


def test_pinned_cpu_request_falls_back_to_disk_backend():
    cfg = HostStagingConfig.from_dict({"backend": "pinned_cpu"})

    assert cfg.backend == "pinned_cpu"
    assert cfg.effective_backend() == "pinned_cpu"


def test_optimizer_restore_policy_defaults_to_deferred():
    cfg = HostStagingConfig.from_dict({"stage_optimizer": True})

    assert cfg.optimizer_restore_policy == "deferred"
    assert cfg.should_defer_optimizer_restore() is True
    assert cfg.should_restore_optimizer_on_load() is False


def test_optimizer_restore_policy_can_be_immediate():
    cfg = HostStagingConfig.from_dict({"stage_optimizer": True, "optimizer_restore_policy": "immediate"})

    assert cfg.optimizer_restore_policy == "immediate"
    assert cfg.should_restore_optimizer_on_load() is True
    assert cfg.should_defer_optimizer_restore() is False


def test_paged_state_dict_roundtrip(tmp_path: Path):
    state_dict = {
        "layer0.weight": torch.ones(256, dtype=torch.float32),
        "layer1.weight": torch.ones(256, dtype=torch.float32) * 2,
        "layer2.weight": torch.ones(256, dtype=torch.float32) * 3,
    }

    manifest = save_paged_state_dict(str(tmp_path), "model_state", state_dict, page_bytes=1024)

    assert manifest["page_count"] >= 2
    assert manifest["total_bytes"] > 0
    assert len(manifest["pages"]) == manifest["page_count"]
    assert has_paged_state_dict(str(tmp_path), "model_state") is True

    restored = {}
    for page in iter_paged_state_dict(str(tmp_path), "model_state"):
        restored.update(page)

    assert restored.keys() == state_dict.keys()
    for key, value in restored.items():
        assert torch.equal(value, state_dict[key])


def test_paged_state_dict_can_be_read_back_as_local_pinned_pages(tmp_path: Path):
    available, error = probe_local_pin_memory_capability()
    if not available:
        pytest.skip(f"pin_memory is unavailable in this test environment: {error}")

    state_dict = {
        "layer0.weight": torch.ones(256, dtype=torch.float32),
        "layer1.weight": torch.ones(256, dtype=torch.float32) * 2,
    }

    save_paged_state_dict(str(tmp_path), "model_state", state_dict, page_bytes=1024)

    restored = {}
    for page in iter_paged_state_dict(str(tmp_path), "model_state", pin_memory=True, prefetch=True):
        for value in page.values():
            assert value.is_pinned()
        restored.update(page)

    assert restored.keys() == state_dict.keys()
    for key, value in restored.items():
        assert torch.equal(value, state_dict[key])


def test_paged_optimizer_state_dict_roundtrip(tmp_path: Path):
    optim_state_dict = {
        "state": {
            "layer0.weight": {
                "exp_avg": torch.ones(256, dtype=torch.float32),
                "exp_avg_sq": torch.ones(256, dtype=torch.float32) * 2,
            },
            "layer1.weight": {
                "exp_avg": torch.ones(256, dtype=torch.float32) * 3,
                "exp_avg_sq": torch.ones(256, dtype=torch.float32) * 4,
            },
        },
        "param_groups": [{"params": ["layer0.weight", "layer1.weight"], "lr": 1e-4}],
    }

    manifest = save_paged_optimizer_state_dict(str(tmp_path), "optim_state", optim_state_dict, page_bytes=1024)

    assert manifest["page_count"] >= 2
    assert manifest["total_bytes"] > 0
    assert len(manifest["pages"]) == manifest["page_count"]
    assert has_paged_state_dict(str(tmp_path), "optim_state") is True

    restored = load_paged_optimizer_state_dict(str(tmp_path), "optim_state")

    assert restored["param_groups"] == optim_state_dict["param_groups"]
    assert restored["state"].keys() == optim_state_dict["state"].keys()
    for param_name, state in restored["state"].items():
        for key, value in state.items():
            assert torch.equal(value, optim_state_dict["state"][param_name][key])


def test_restore_session_manifest_tracks_progress(tmp_path: Path):
    model_manifest = save_paged_state_dict(
        str(tmp_path),
        "model_state",
        {"layer.weight": torch.ones(512, dtype=torch.float32)},
        page_bytes=512,
    )
    optim_manifest = save_paged_optimizer_state_dict(
        str(tmp_path),
        "optim_state",
        {
            "state": {"layer.weight": {"exp_avg": torch.ones(128, dtype=torch.float32)}},
            "param_groups": [{"params": ["layer.weight"], "lr": 1e-4}],
        },
        page_bytes=256,
    )

    created = create_restore_session_manifest(
        str(tmp_path),
        backend="disk_fallback",
        session_id="session-test",
        optimizer_restore_policy="deferred",
        model_manifest=model_manifest,
        optimizer_manifest=optim_manifest,
    )

    assert has_restore_session_manifest(str(tmp_path)) is True
    assert created["session_id"] == "session-test"
    assert created["model_page_count"] == model_manifest["page_count"]
    assert created["optimizer_page_count"] == optim_manifest["page_count"]
    assert created["optimizer_restore_policy"] == "deferred"

    record_restore_progress(str(tmp_path), applied_model_pages=1, status="restoring_model")
    update_restore_session_manifest(str(tmp_path), applied_optimizer_pages=1)
    finalize_restore_session_manifest(str(tmp_path))

    manifest = read_restore_session_manifest(str(tmp_path))
    assert manifest["status"] == "completed"
    assert manifest["applied_model_pages"] == manifest["model_page_count"]
    assert manifest["applied_optimizer_pages"] == manifest["optimizer_page_count"]


def test_read_paged_manifest_exposes_page_metadata(tmp_path: Path):
    save_paged_state_dict(
        str(tmp_path),
        "model_state",
        {
            "layer0.weight": torch.ones(256, dtype=torch.float32),
            "layer1.weight": torch.ones(256, dtype=torch.float32),
        },
        page_bytes=1024,
    )

    manifest = read_paged_state_manifest(str(tmp_path), "model_state")

    assert manifest["page_count"] == len(manifest["pages"])
    assert manifest["pages"][0]["file_name"].startswith("model_state.page_")
    assert manifest["pages"][0]["estimated_bytes"] > 0
