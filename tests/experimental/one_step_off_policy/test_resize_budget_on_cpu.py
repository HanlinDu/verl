from verl.experimental.one_step_off_policy.resize_budget import (
    ResizeBudgetConfig,
    ResizeBudgetController,
    ResizeBudgetSnapshot,
)


def test_export_budget_blocks_when_host_budget_is_too_small():
    controller = ResizeBudgetController(ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5))
    snapshot = ResizeBudgetSnapshot(host_free_bytes=100, disk_free_bytes=10_000, gpu_free_bytes=10_000)

    decision = controller.evaluate_export(
        requested_backend="disk_fallback",
        snapshot=snapshot,
        estimated_host_peak_bytes=60,
        estimated_stage_bytes=10,
    )

    assert decision.allow_resize is False
    assert decision.blocked is True
    assert decision.reason == "host_budget"


def test_export_budget_falls_back_from_pinned_to_disk():
    controller = ResizeBudgetController(ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5))
    snapshot = ResizeBudgetSnapshot(host_free_bytes=1_000, disk_free_bytes=10_000, gpu_free_bytes=10_000)

    decision = controller.evaluate_export(
        requested_backend="pinned_cpu",
        snapshot=snapshot,
        estimated_host_peak_bytes=100,
        estimated_stage_bytes=700,
    )

    assert decision.allow_resize is True
    assert decision.effective_backend == "disk_fallback"


def test_restore_budget_blocks_when_gpu_budget_is_too_small():
    controller = ResizeBudgetController(ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5))
    snapshot = ResizeBudgetSnapshot(host_free_bytes=10_000, disk_free_bytes=10_000, gpu_free_bytes=100)

    decision = controller.evaluate_restore(snapshot=snapshot, estimated_gpu_peak_bytes=60)

    assert decision.allow_resize is False
    assert decision.blocked is True
    assert decision.reason == "gpu_budget"


def test_unknown_gpu_budget_does_not_block_resize():
    controller = ResizeBudgetController(ResizeBudgetConfig(enable=True, memory_budget_ratio=0.5))
    snapshot = ResizeBudgetSnapshot(host_free_bytes=10_000, disk_free_bytes=10_000, gpu_free_bytes=None)

    decision = controller.evaluate_restore(snapshot=snapshot, estimated_gpu_peak_bytes=1_000_000)

    assert decision.allow_resize is True
    assert decision.blocked is False
