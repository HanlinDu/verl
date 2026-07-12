import pytest

from verl.experimental.one_step_off_policy.resize_controller import (
    ACTION_EXPAND_ROLLOUT,
    ACTION_EXPAND_TRAIN,
    ACTION_HOLD,
    ResizeController,
    ResizeControllerConfig,
)


def _observe(controller: ResizeController, step: int, rollout_time: float, train_time: float):
    controller.observe(
        step=step,
        observation={
            "rollout_time_s": rollout_time,
            "train_time_s": train_time,
        },
    )


def test_resize_controller_blocks_until_signal_matches_required_action():
    controller = ResizeController(
        ResizeControllerConfig(
            enable=True,
            window_size=2,
            up_threshold=1.1,
            down_threshold=0.9,
            min_dwell_steps=0,
            cooldown_steps=0,
            consecutive_signal_steps=1,
            min_observation_count=1,
        )
    )

    _observe(controller, step=1, rollout_time=2.0, train_time=1.0)
    allow, snapshot = controller.gate(step=1, required_action=ACTION_EXPAND_ROLLOUT)

    assert allow is True
    assert snapshot["hysteresis_decision"] == ACTION_EXPAND_ROLLOUT
    assert snapshot["gate_pass"] == pytest.approx(1.0)

    allow, snapshot = controller.gate(step=1, required_action=ACTION_EXPAND_TRAIN)
    assert allow is False
    assert snapshot["gate_pass"] == pytest.approx(0.0)


def test_resize_controller_enforces_dwell_after_resize():
    controller = ResizeController(
        ResizeControllerConfig(
            enable=True,
            window_size=2,
            up_threshold=1.1,
            down_threshold=0.9,
            min_dwell_steps=2,
            cooldown_steps=0,
            consecutive_signal_steps=1,
            min_observation_count=1,
        )
    )

    _observe(controller, step=1, rollout_time=2.0, train_time=1.0)
    controller.mark_resize_applied(step=1, action=ACTION_EXPAND_ROLLOUT)

    _observe(controller, step=2, rollout_time=0.5, train_time=2.0)
    allow, snapshot = controller.gate(step=2, required_action=ACTION_EXPAND_TRAIN)

    assert allow is False
    assert snapshot["dwell_remaining"] == 1
    assert snapshot["gate_pass"] == pytest.approx(0.0)


def test_resize_controller_returns_hold_for_insufficient_consensus():
    controller = ResizeController(
        ResizeControllerConfig(
            enable=True,
            window_size=3,
            up_threshold=1.1,
            down_threshold=0.9,
            min_dwell_steps=0,
            cooldown_steps=0,
            consecutive_signal_steps=2,
            min_observation_count=2,
        )
    )

    _observe(controller, step=1, rollout_time=2.0, train_time=1.0)
    allow, snapshot = controller.gate(step=1, required_action=ACTION_EXPAND_ROLLOUT)

    assert allow is False
    assert snapshot["hysteresis_decision"] == ACTION_HOLD
