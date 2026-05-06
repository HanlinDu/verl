from pathlib import Path

from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    read_host_staging_manifest,
    write_host_staging_manifest,
)


def test_host_staging_manifest_roundtrip(tmp_path: Path):
    cfg = HostStagingConfig.from_dict(
        {
            "enable": True,
            "backend": "disk_fallback",
            "chunk_mb": 512,
            "stage_optimizer": False,
            "progressive_swap": True,
            "cleanup_after_load": False,
            "preclear_rollout_kv_cache": True,
        }
    )

    write_host_staging_manifest(str(tmp_path), cfg)
    manifest = read_host_staging_manifest(str(tmp_path))

    assert manifest["backend"] == "disk_fallback"
    assert manifest["requested_backend"] == "disk_fallback"
    assert manifest["chunk_mb"] == 512
    assert manifest["stage_optimizer"] is False
    assert manifest["cleanup_after_load"] is False


def test_pinned_cpu_request_falls_back_to_disk_backend():
    cfg = HostStagingConfig.from_dict({"backend": "pinned_cpu"})

    assert cfg.backend == "pinned_cpu"
    assert cfg.effective_backend() == "disk_fallback"