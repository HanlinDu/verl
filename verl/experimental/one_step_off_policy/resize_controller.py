"""Schedule-oriented gating controller for dynamic resize."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


ACTION_EXPAND_TRAIN = "expand_train"
ACTION_EXPAND_ROLLOUT = "expand_rollout"
ACTION_HOLD = "hold"

ACTION_TO_CODE = {
    ACTION_EXPAND_TRAIN: -1.0,
    ACTION_HOLD: 0.0,
    ACTION_EXPAND_ROLLOUT: 1.0,
}


@dataclass(slots=True)
class ResizeControllerConfig:
    enable: bool = False
    window_size: int = 2
    up_threshold: float = 1.15
    down_threshold: float = 0.9
    min_dwell_steps: int = 1
    cooldown_steps: int = 0
    consecutive_signal_steps: int = 1
    min_observation_count: int = 1

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ResizeControllerConfig":
        cfg = cfg or {}
        return cls(
            enable=bool(cfg.get("enable", False)),
            window_size=max(int(cfg.get("window_size", 2)), 1),
            up_threshold=float(cfg.get("up_threshold", 1.15)),
            down_threshold=float(cfg.get("down_threshold", 0.9)),
            min_dwell_steps=max(int(cfg.get("min_dwell_steps", 1)), 0),
            cooldown_steps=max(int(cfg.get("cooldown_steps", 0)), 0),
            consecutive_signal_steps=max(int(cfg.get("consecutive_signal_steps", 1)), 1),
            min_observation_count=max(int(cfg.get("min_observation_count", 1)), 1),
        )


class ResizeController:
    """Gate scheduled resizes with a lightweight hysteresis controller."""

    def __init__(self, config: ResizeControllerConfig):
        self.config = config
        self._observations: deque[dict[str, float]] = deque(maxlen=config.window_size)
        self._signals: deque[str] = deque(maxlen=config.consecutive_signal_steps)
        self._last_resize_step: int | None = None
        self._last_applied_action: str = ACTION_HOLD
        self._latest_snapshot: dict[str, float | str | int] = self._build_snapshot(required_action=ACTION_HOLD)

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def observe(self, *, step: int, observation: dict[str, float]) -> dict[str, float | str | int]:
        self._observations.append(observation)
        avg_rollout_time, avg_train_time = self._average_times()
        ratio = avg_rollout_time / max(avg_train_time, 1e-6)
        raw_signal = self._signal_from_ratio(ratio)
        self._signals.append(raw_signal)
        self._latest_snapshot = self._build_snapshot(current_step=step, required_action=ACTION_HOLD)
        return self._latest_snapshot

    def infer_required_action(
        self,
        *,
        active_actor: int,
        active_rollout: int,
        actor_target: int,
        rollout_target: int,
    ) -> str:
        actor_delta = actor_target - active_actor
        rollout_delta = rollout_target - active_rollout
        if rollout_delta > 0 and actor_delta < 0:
            return ACTION_EXPAND_ROLLOUT
        if actor_delta > 0 and rollout_delta < 0:
            return ACTION_EXPAND_TRAIN
        return ACTION_HOLD

    def gate(
        self,
        *,
        step: int,
        required_action: str,
    ) -> tuple[bool, dict[str, float | str | int]]:
        snapshot = self._build_snapshot(current_step=step, required_action=required_action)

        if not self.enabled:
            snapshot["gate_pass"] = 1.0
            self._latest_snapshot = snapshot
            return True, snapshot

        if required_action == ACTION_HOLD:
            snapshot["gate_pass"] = 1.0
            self._latest_snapshot = snapshot
            return True, snapshot

        window_fill = int(snapshot["window_fill"])
        if window_fill < max(self.config.min_observation_count, 1):
            snapshot["gate_pass"] = 0.0
            self._latest_snapshot = snapshot
            return False, snapshot

        if int(snapshot["cooldown_remaining"]) > 0 or int(snapshot["dwell_remaining"]) > 0:
            snapshot["gate_pass"] = 0.0
            self._latest_snapshot = snapshot
            return False, snapshot

        decision = snapshot["hysteresis_decision"]
        allow = decision == required_action
        snapshot["gate_pass"] = 1.0 if allow else 0.0
        self._latest_snapshot = snapshot
        return allow, snapshot

    def mark_resize_applied(self, *, step: int, action: str) -> dict[str, float | str | int]:
        self._last_resize_step = step
        self._last_applied_action = action
        self._latest_snapshot = self._build_snapshot(current_step=step, required_action=action)
        self._latest_snapshot["gate_pass"] = 1.0
        return self._latest_snapshot

    def latest_metrics(self) -> dict[str, float | str | int]:
        return dict(self._latest_snapshot)

    def _build_snapshot(self, *, current_step: int | None = None, required_action: str) -> dict[str, float | str | int]:
        avg_rollout_time, avg_train_time = self._average_times()
        ratio = avg_rollout_time / max(avg_train_time, 1e-6)
        raw_signal = self._signal_from_ratio(ratio)
        decision = self._stable_decision()

        cooldown_remaining = self._remaining_steps(current_step, self.config.cooldown_steps)
        dwell_remaining = self._remaining_steps(current_step, self.config.min_dwell_steps)

        return {
            "window_fill": len(self._observations),
            "rollout_train_ratio": ratio,
            "avg_rollout_time_s": avg_rollout_time,
            "avg_train_time_s": avg_train_time,
            "hysteresis_signal": raw_signal,
            "hysteresis_signal_code": ACTION_TO_CODE[raw_signal],
            "hysteresis_decision": decision,
            "hysteresis_decision_code": ACTION_TO_CODE[decision],
            "required_action": required_action,
            "required_action_code": ACTION_TO_CODE[required_action],
            "cooldown_remaining": cooldown_remaining,
            "dwell_remaining": dwell_remaining,
            "last_applied_action": self._last_applied_action,
            "gate_pass": -1.0,
        }

    def _average_times(self) -> tuple[float, float]:
        if not self._observations:
            return 0.0, 0.0

        rollout_sum = sum(float(item.get("rollout_time_s", 0.0)) for item in self._observations)
        train_sum = sum(float(item.get("train_time_s", 0.0)) for item in self._observations)
        count = len(self._observations)
        return rollout_sum / count, train_sum / count

    def _signal_from_ratio(self, ratio: float) -> str:
        if ratio > self.config.up_threshold:
            return ACTION_EXPAND_ROLLOUT
        if ratio < self.config.down_threshold:
            return ACTION_EXPAND_TRAIN
        return ACTION_HOLD

    def _stable_decision(self) -> str:
        if len(self._signals) < self.config.consecutive_signal_steps:
            return ACTION_HOLD
        first_signal = self._signals[0]
        if first_signal == ACTION_HOLD:
            return ACTION_HOLD
        if all(signal == first_signal for signal in self._signals):
            return first_signal
        return ACTION_HOLD

    def _remaining_steps(self, current_step: int | None, limit: int) -> int:
        if current_step is None or self._last_resize_step is None or limit <= 0:
            return 0
        elapsed = current_step - self._last_resize_step
        return max(limit - elapsed, 0)