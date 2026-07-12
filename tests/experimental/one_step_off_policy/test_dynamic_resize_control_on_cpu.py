import asyncio
from types import SimpleNamespace

from omegaconf import OmegaConf

from verl.experimental.one_step_off_policy.resize_budget import (
    ResizeBudgetConfig,
    ResizeBudgetController,
    ResizeBudgetSnapshot,
)
from verl.experimental.one_step_off_policy.resize_controller import (
    ACTION_HOLD,
    ACTION_TO_CODE,
    ACTION_EXPAND_ROLLOUT,
    ACTION_EXPAND_TRAIN,
    ResizeController,
    ResizeControllerConfig,
)
from verl.experimental.one_step_off_policy.resize_metrics import build_resize_observation
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.one_step_off_policy.staging_backend import (
    HostStagingConfig,
    read_host_staging_manifest,
    read_restore_session_manifest,
)
from verl.experimental.one_step_off_policy.trace_utils import ResizeTraceConfig
from verl.experimental.separation.utils import build_dynamic_resize_pool_topology, create_resource_pool_manager
from verl.trainer.ppo.utils import Role


def test_resize_controller_allows_matching_stable_signal():
    controller = ResizeController(
        ResizeControllerConfig(
            enable=True,
            window_size=2,
            up_threshold=1.1,
            consecutive_signal_steps=1,
            min_observation_count=1,
            min_dwell_steps=0,
        )
    )

    controller.observe(step=1, observation={"rollout_time_s": 2.0, "train_time_s": 1.0})
    required = controller.infer_required_action(
        active_actor=4,
        active_rollout=4,
        actor_target=2,
        rollout_target=6,
    )
    allowed, snapshot = controller.gate(step=1, required_action=required)

    assert required == ACTION_EXPAND_ROLLOUT
    assert allowed
    assert snapshot["gate_pass"] == 1.0


def test_resize_budget_falls_back_and_blocks_when_disk_is_too_small():
    controller = ResizeBudgetController(ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5))
    decision = controller.evaluate_export(
        requested_backend="pinned_cpu",
        snapshot=ResizeBudgetSnapshot(host_free_bytes=100, disk_free_bytes=100, gpu_free_bytes=100),
        estimated_host_peak_bytes=40,
        estimated_stage_bytes=80,
    )

    assert not decision.allow_resize
    assert decision.effective_backend == "disk_fallback"
    assert decision.reason == "disk_budget"


def test_resize_observation_uses_rollout_and_train_timing_sections():
    observation = build_resize_observation(
        timing_raw={
            "gen": 1.5,
            "sync_rollout_weights": 0.5,
            "update_actor": 0.75,
            "update_critic": 0.25,
        }
    )

    assert observation["rollout_time_s"] == 2.0
    assert observation["train_time_s"] == 1.0


def test_one_step_off_configs_define_disabled_dynamic_resize_by_default():
    fsdp_cfg = OmegaConf.load("verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer.yaml")
    megatron_cfg = OmegaConf.load(
        "verl/experimental/one_step_off_policy/config/one_step_off_ppo_megatron_trainer.yaml"
    )

    assert fsdp_cfg.trainer.dynamic_resize.enable is False
    assert megatron_cfg.trainer.dynamic_resize.enable is False
    assert fsdp_cfg.trainer.dynamic_resize.hard_switch.enable is False
    assert megatron_cfg.trainer.dynamic_resize.hard_switch.enable is False
    assert fsdp_cfg.trainer.dynamic_resize.handoff.enable is False
    assert megatron_cfg.trainer.dynamic_resize.handoff.enable is False
    assert fsdp_cfg.trainer.dynamic_resize.handoff.backend == "disk_fallback"
    assert fsdp_cfg.trainer.dynamic_resize.handoff.stage_optimizer is False
    assert fsdp_cfg.trainer.dynamic_resize.schedule == []


def test_dynamic_resize_handoff_metrics_reflect_disabled_default_and_pinned_backend():
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer._dynamic_resize_cfg = {"handoff": {"enable": True, "backend": "pinned_cpu", "stage_optimizer": True}}
    trainer._host_staging_config = HostStagingConfig.from_dict(trainer._dynamic_resize_cfg["handoff"])
    metrics = trainer._default_resize_handoff_metrics()

    assert metrics["resize/handoff_enabled"] == 1.0
    assert metrics["resize/handoff_backend"] == "pinned_cpu"
    assert metrics["resize/handoff_stage_optimizer"] == 1.0
    assert metrics["resize/handoff_optimizer_restore_policy"] == "deferred"
    assert metrics["resize/handoff_manifest_written"] == 0.0


def test_dynamic_resize_handoff_session_manifest_is_written_and_completed(tmp_path):
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 7
    trainer._host_staging_config = HostStagingConfig.from_dict(
        {
            "enable": True,
            "backend": "pinned_cpu",
            "service_name": "resize-service",
            "stage_optimizer": False,
            "optimizer_restore_policy": "immediate",
        }
    )
    global_step_folder = str(tmp_path / "global_step_7")

    prepare_metrics = trainer._prepare_dynamic_resize_handoff_session(global_step_folder)

    stage_dir = tmp_path / "global_step_7" / "dynamic_resize_handoff"
    host_manifest = read_host_staging_manifest(str(stage_dir))
    restore_manifest = read_restore_session_manifest(str(stage_dir))

    assert prepare_metrics["resize/handoff_manifest_written"] == 1.0
    assert prepare_metrics["resize/handoff_restore_session_status"] == "staged"
    assert prepare_metrics["resize/handoff_stage_dir"] == str(stage_dir)
    assert host_manifest["backend"] == "pinned_cpu"
    assert host_manifest["service_name"] == "resize-service"
    assert restore_manifest["backend"] == "pinned_cpu"
    assert restore_manifest["service_name"] == "resize-service"
    assert restore_manifest["optimizer_restore_policy"] == "immediate"

    complete_metrics = trainer._complete_dynamic_resize_handoff_session(global_step_folder)
    completed_manifest = read_restore_session_manifest(str(stage_dir))

    assert complete_metrics["resize/handoff_manifest_written"] == 1.0
    assert complete_metrics["resize/handoff_restore_session_status"] == "completed"
    assert completed_manifest["status"] == "completed"


def test_dynamic_resize_defer_next_rollout_requires_hard_switch_schedule_match():
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 4
    trainer._dynamic_resize_enabled = True
    trainer._dynamic_resize_cfg = {
        "enable": True,
        "shared_pool": True,
        "hard_switch": {"enable": True},
        "schedule": [{"step": 4, "actor_pool": {"world_size": 2}, "rollout_pool": {"world_size": 4}}],
    }

    assert trainer._dynamic_resize_hard_switch_enabled()
    assert trainer._should_defer_next_rollout_for_resize()

    trainer._dynamic_resize_cfg["hard_switch"]["enable"] = False
    assert not trainer._dynamic_resize_hard_switch_enabled()
    assert not trainer._should_defer_next_rollout_for_resize()


def test_dynamic_resize_metrics_keep_active_topology_without_schedule_match():
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 2
    trainer._dynamic_resize_enabled = True
    trainer._dynamic_resize_mode = "schedule"
    trainer._dynamic_resize_cfg = {
        "enable": True,
        "shared_pool": True,
        "hard_switch": {"enable": True},
        "schedule": [{"step": 1, "actor_pool": {"world_size": 1}, "rollout_pool": {"world_size": 3}}],
    }
    trainer._resize_controller = None
    trainer._resize_trace_config = ResizeTraceConfig()
    trainer._active_dynamic_resize_topology = {"actor_size": 1, "rollout_size": 3}
    trainer._latest_resize_control_metrics = {
        "resize/enabled": 1.0,
        "resize/mode": "schedule",
        "resize/controller_enabled": 0.0,
        "resize/hysteresis_signal": ACTION_HOLD,
        "resize/hysteresis_signal_code": ACTION_TO_CODE[ACTION_HOLD],
        "resize/hysteresis_decision": ACTION_HOLD,
        "resize/hysteresis_decision_code": ACTION_TO_CODE[ACTION_HOLD],
        "resize/required_action": ACTION_HOLD,
        "resize/required_action_code": ACTION_TO_CODE[ACTION_HOLD],
        "resize/schedule_triggered": 0.0,
        "resize/schedule_applied": 0.0,
        "resize/pending_resource_switch": 0.0,
        "resize/hard_switch_enabled": 1.0,
        "resize/active_actor_size": 0.0,
        "resize/active_rollout_size": 0.0,
        "resize/target_actor_size": 0.0,
        "resize/target_rollout_size": 0.0,
    }

    metrics = asyncio.run(trainer._maybe_dynamic_resize())

    assert metrics["resize/schedule_triggered"] == 0.0
    assert metrics["resize/active_actor_size"] == 1.0
    assert metrics["resize/active_rollout_size"] == 3.0
    assert metrics["resize/target_actor_size"] == 1.0
    assert metrics["resize/target_rollout_size"] == 3.0


def test_dynamic_resize_schedule_uses_hard_switch_for_release_old_metadata():
    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 2
    trainer._dynamic_resize_enabled = True
    trainer._dynamic_resize_mode = "schedule"
    trainer._dynamic_resize_cfg = {
        "enable": True,
        "shared_pool": True,
        "hard_switch": {"enable": True},
        "schedule": [
            {
                "step": 2,
                "release_old": True,
                "actor_pool": {"world_size": 1},
                "rollout_pool": {"world_size": 3},
            }
        ],
    }
    trainer._resize_controller = None
    trainer._resize_trace_config = ResizeTraceConfig()
    trainer._latest_resize_control_metrics = {}

    applied = []

    def _gate(item):
        assert item["release_old"] is True
        return True, ACTION_EXPAND_TRAIN, {"resize/schedule_triggered": 1.0}

    async def _hard_switch(*, item, required_action):
        applied.append((item, required_action))
        return {
            "resize/schedule_applied": 1.0,
            "resize/pending_resource_switch": 0.0,
            "resize/hard_switch_success": 1.0,
        }

    trainer._gate_dynamic_resize_schedule_item = _gate
    trainer._apply_dynamic_resize_hard_switch = _hard_switch

    metrics = asyncio.run(trainer._maybe_dynamic_resize())

    assert len(applied) == 1
    item, required_action = applied[0]
    assert item["release_old"] is True
    assert required_action == ACTION_EXPAND_TRAIN
    assert metrics["resize/schedule_triggered"] == 1.0
    assert metrics["resize/schedule_applied"] == 1.0
    assert metrics["resize/hard_switch_success"] == 1.0


def test_dynamic_resize_budget_gate_blocks_schedule_before_hard_switch(tmp_path):
    class _FakeWorkerGroup:
        world_size = 2

        def execute_rank_zero_sync(self, method_name, staging_path):
            assert method_name == "get_resize_resource_snapshot"
            assert staging_path.endswith("global_step_3")
            return {"host_free_bytes": 100, "disk_free_bytes": 100, "gpu_free_bytes": 10_000}

    trainer = object.__new__(OneStepOffRayTrainer)
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {"model": {"path": str(tmp_path / "missing-model")}},
        }
    )
    trainer.actor_wg = _FakeWorkerGroup()
    trainer.resource_pool_manager = SimpleNamespace(dynamic_resize_topology=None)
    trainer._active_dynamic_resize_topology = {"actor_size": 2, "rollout_size": 2}
    trainer._resize_controller = None
    trainer._latest_resize_control_metrics = {}
    trainer._resize_budget_config = ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5)
    trainer._resize_budget_controller = ResizeBudgetController(trainer._resize_budget_config)
    trainer._host_staging_config = HostStagingConfig.from_dict(
        {"enable": True, "backend": "pinned_cpu", "stage_optimizer": False}
    )

    gate_pass, required_action, metrics = trainer._gate_dynamic_resize_schedule_item(
        {
            "step": 3,
            "actor_pool": {"world_size": 1},
            "rollout_pool": {"world_size": 3},
            "estimated_stage_bytes": 80,
            "estimated_host_peak_bytes": 40,
            "estimated_gpu_peak_bytes": 40,
        }
    )

    assert required_action == ACTION_EXPAND_ROLLOUT
    assert gate_pass is False
    assert metrics["resize/gate_pass"] == 0.0
    assert metrics["resize/budget_blocked"] == 1.0
    assert metrics["resize/budget_effective_backend"] == "disk_fallback"
    assert metrics["resize/budget_reason"] == "disk_budget"


def _make_shared_pool_config(schedule=None):
    return OmegaConf.create(
        {
            "trainer": {
                "nnodes": 2,
                "n_gpus_per_node": 2,
                "dynamic_resize": {
                    "enable": True,
                    "shared_pool": True,
                    "schedule": schedule or [],
                },
            },
            "rollout": {
                "nnodes": 2,
                "n_gpus_per_node": 1,
            },
            "reward": {
                "reward_model": {
                    "n_gpus_per_node": 1,
                    "nnodes": 1,
                }
            },
        }
    )


def test_dynamic_resize_shared_pool_topology_records_initial_and_scheduled_splits():
    cfg = _make_shared_pool_config(
        schedule=[
            {
                "step": 3,
                "actor_pool": {"world_size": 2},
                "rollout_pool": {"world_size": 4},
            }
        ]
    )

    topology = build_dynamic_resize_pool_topology(cfg)
    manager = create_resource_pool_manager(cfg, [Role.Actor, Role.Critic])

    assert topology.shared_pool_name == "dynamic_resize_shared_pool"
    assert topology.shared_pool_spec == [3, 3]
    assert topology.initial_actor_size == 4
    assert topology.initial_rollout_size == 2
    assert topology.schedule_splits == [{"step": 3, "actor_size": 2, "rollout_size": 4}]
    assert manager.resource_pool_spec == {"dynamic_resize_shared_pool": [3, 3]}
    assert manager.mapping[Role.Actor] == "dynamic_resize_shared_pool"
    assert manager.mapping[Role.Critic] == "dynamic_resize_shared_pool"
    assert manager.dynamic_resize_topology == topology


def test_dynamic_resize_shared_pool_rejects_schedule_that_changes_total_size():
    cfg = _make_shared_pool_config(
        schedule=[
            {
                "step": 3,
                "actor_pool": {"world_size": 2},
                "rollout_pool": {"world_size": 3},
            }
        ]
    )

    try:
        build_dynamic_resize_pool_topology(cfg)
    except ValueError as exc:
        assert "shared pool size 6" in str(exc)
    else:
        raise AssertionError("expected invalid shared_pool schedule to raise ValueError")
