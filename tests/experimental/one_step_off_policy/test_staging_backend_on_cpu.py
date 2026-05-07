from pathlib import Path

import torch

from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    has_paged_state_dict,
    iter_paged_state_dict,
    load_paged_optimizer_state_dict,
    read_host_staging_manifest,
    save_paged_optimizer_state_dict,
    save_paged_state_dict,
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


def test_paged_state_dict_roundtrip(tmp_path: Path):
    state_dict = {
        "layer0.weight": torch.ones(256, dtype=torch.float32),
        "layer1.weight": torch.ones(256, dtype=torch.float32) * 2,
        "layer2.weight": torch.ones(256, dtype=torch.float32) * 3,
    }

    manifest = save_paged_state_dict(str(tmp_path), "model_state", state_dict, page_bytes=1024)

    assert manifest["page_count"] >= 2
    assert has_paged_state_dict(str(tmp_path), "model_state") is True

    restored = {}
    for page in iter_paged_state_dict(str(tmp_path), "model_state"):
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
    assert has_paged_state_dict(str(tmp_path), "optim_state") is True

    restored = load_paged_optimizer_state_dict(str(tmp_path), "optim_state")

    assert restored["param_groups"] == optim_state_dict["param_groups"]
    assert restored["state"].keys() == optim_state_dict["state"].keys()
    for param_name, state in restored["state"].items():
        for key, value in state.items():
            assert torch.equal(value, optim_state_dict["state"][param_name][key])