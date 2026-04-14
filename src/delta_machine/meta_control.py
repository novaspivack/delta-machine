"""
Meta-control utilities for Δ-Machine runs.

Design reference:
- 1.16 New Avenues for DSAC:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.16_new_avenues_for_dsac.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from collections import deque


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(slots=True)
class MetaControlConfig:
    reaction_decay: float = 0.92
    temperature_gain: float = 0.04
    min_temperature: float = 0.35
    max_temperature: float = 2.8
    drive_gain: float = 0.35
    beta_gain: float = 4.5
    sinkhorn_extra_max: int = 8
    perturbation_threshold: float = 7.5e-4
    perturbation_cooldown: int = 180
    perturbation_strength: float = 0.02
    temperature_leak: float = 0.015
    slow_window: int = 80
    slow_drive_gain: float = 0.15
    slow_beta_gain: float = 1.8
    slow_sinkhorn_gain: float = 0.4
    slow_subtour_gain: float = 0.25
    slow_subtour_max: float = 2.5
    observables_window: int = 12


class ReactionFieldController:
    """Tracks global reaction/temperature state and returns control actions."""

    def __init__(self, config: MetaControlConfig):
        self.config = config
        self._prev_dissonance: Optional[float] = None
        self._reaction: float = 0.0
        self._temperature: float = 1.0
        self._quiet_steps: int = 0
        self._slow_buffer: deque[float] = deque(maxlen=max(4, int(config.slow_window)))

    def update(self, step: int, dissonance: float, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        derivative = 0.0 if self._prev_dissonance is None else dissonance - self._prev_dissonance
        self._prev_dissonance = float(dissonance)
        self._reaction = (
            self.config.reaction_decay * self._reaction
            + (1.0 - self.config.reaction_decay) * derivative
        )
        magnitude = abs(self._reaction)
        if magnitude < self.config.perturbation_threshold:
            self._quiet_steps += 1
        else:
            self._quiet_steps = 0

        leak = self.config.temperature_leak * (self._temperature - 1.0)
        self._temperature = _clamp(
            self._temperature
            + self.config.temperature_gain * self._reaction
            - leak,
            self.config.min_temperature,
            self.config.max_temperature,
        )

        drive_scale_fast = 1.0 + self.config.drive_gain * (self._temperature - 1.0)
        beta_boost_fast = max(0.0, self.config.beta_gain * self._reaction)
        sinkhorn_fast = min(
            self.config.sinkhorn_extra_max,
            int(round(magnitude * self.config.sinkhorn_extra_max * 2.0)),
        )

        self._slow_buffer.append(dissonance)
        slow_slope = 0.0
        slow_amplitude = 0.0
        if len(self._slow_buffer) >= 2:
            slow_slope = (self._slow_buffer[-1] - self._slow_buffer[0]) / len(self._slow_buffer)
            slow_amplitude = max(self._slow_buffer) - min(self._slow_buffer)
        slow_drive = _clamp(1.0 + (-slow_slope) * self.config.slow_drive_gain, 0.5, 1.6)
        beta_boost_slow = max(0.0, -slow_slope) * self.config.slow_beta_gain
        slow_sinkhorn = 0
        subtour_scale = 1.0
        cost_gap = None
        subtour_count = None
        if metrics:
            meta_obs = metrics.get("meta")
            if meta_obs:
                cost_gap = meta_obs.get("cost_gap")
                subtour_count = meta_obs.get("subtour_count")
                if subtour_count is not None and subtour_count > 1:
                    subtour_scale += (subtour_count - 1) * self.config.slow_subtour_gain
        if metrics and "tsp" in metrics and cost_gap is None:
            tsp_metrics = metrics["tsp"]
            cost_gap = tsp_metrics.get("tour_cost_gap")
            subtour_count = subtour_count or tsp_metrics.get("subtour_count")
        if cost_gap is not None and cost_gap > 0:
            slow_sinkhorn = int(round(min(self.config.sinkhorn_extra_max, cost_gap * self.config.slow_sinkhorn_gain * 10)))
        if slow_amplitude < self.config.perturbation_threshold and slow_sinkhorn == 0 and magnitude < self.config.perturbation_threshold:
            slow_sinkhorn = 1

        subtour_scale = _clamp(subtour_scale, 1.0, self.config.slow_subtour_max)

        drive_scale = _clamp(drive_scale_fast * slow_drive, 0.25, 2.5)
        beta_boost = beta_boost_fast + beta_boost_slow
        sinkhorn_total = min(
            self.config.sinkhorn_extra_max,
            sinkhorn_fast + slow_sinkhorn,
        )

        actions: Dict[str, Any] = {
            "temperature": self._temperature,
            "reaction": self._reaction,
            "drive_scale": drive_scale,
            "beta_boost": beta_boost,
            "sinkhorn_extra": sinkhorn_total,
            "slow_sinkhorn_extra": slow_sinkhorn,
            "subtour_scale": subtour_scale,
            "observables": {
                "slow_slope": slow_slope,
                "slow_amplitude": slow_amplitude,
                "slow_drive": slow_drive,
                "subtour_scale": subtour_scale,
            },
        }

        if self._quiet_steps >= self.config.perturbation_cooldown:
            actions["perturbation"] = {
                "strength": self.config.perturbation_strength,
                "reason": "reaction_quiet",
                "step": step,
            }
            self._quiet_steps = 0

        if metrics and "tsp" in metrics:
            tsp_metrics = metrics["tsp"]
            cost_gap_metric = float(tsp_metrics.get("tour_cost_gap", 0.0))
            if cost_gap_metric > 0.05:
                actions["beta_boost"] += cost_gap_metric * self.config.beta_gain
                actions["sinkhorn_extra"] = min(
                    self.config.sinkhorn_extra_max,
                    actions["sinkhorn_extra"] + 2,
                )

        return actions


DEFAULT_META_CONFIG = MetaControlConfig()


def load_meta_control_config(metadata: Optional[Dict[str, Any]]) -> MetaControlConfig | None:
    if not metadata:
        return None
    cfg = metadata.get("meta_control")
    if cfg is None:
        return None
    return MetaControlConfig(
        reaction_decay=float(cfg.get("reaction_decay", DEFAULT_META_CONFIG.reaction_decay)),
        temperature_gain=float(cfg.get("temperature_gain", DEFAULT_META_CONFIG.temperature_gain)),
        min_temperature=float(cfg.get("min_temperature", DEFAULT_META_CONFIG.min_temperature)),
        max_temperature=float(cfg.get("max_temperature", DEFAULT_META_CONFIG.max_temperature)),
        drive_gain=float(cfg.get("drive_gain", DEFAULT_META_CONFIG.drive_gain)),
        beta_gain=float(cfg.get("beta_gain", DEFAULT_META_CONFIG.beta_gain)),
        sinkhorn_extra_max=int(cfg.get("sinkhorn_extra_max", DEFAULT_META_CONFIG.sinkhorn_extra_max)),
        perturbation_threshold=float(cfg.get("perturbation_threshold", DEFAULT_META_CONFIG.perturbation_threshold)),
        perturbation_cooldown=int(cfg.get("perturbation_cooldown", DEFAULT_META_CONFIG.perturbation_cooldown)),
        perturbation_strength=float(cfg.get("perturbation_strength", DEFAULT_META_CONFIG.perturbation_strength)),
        temperature_leak=float(cfg.get("temperature_leak", DEFAULT_META_CONFIG.temperature_leak)),
        slow_window=int(cfg.get("slow_window", DEFAULT_META_CONFIG.slow_window)),
        slow_drive_gain=float(cfg.get("slow_drive_gain", DEFAULT_META_CONFIG.slow_drive_gain)),
        slow_beta_gain=float(cfg.get("slow_beta_gain", DEFAULT_META_CONFIG.slow_beta_gain)),
        slow_sinkhorn_gain=float(cfg.get("slow_sinkhorn_gain", DEFAULT_META_CONFIG.slow_sinkhorn_gain)),
        slow_subtour_gain=float(cfg.get("slow_subtour_gain", DEFAULT_META_CONFIG.slow_subtour_gain)),
        slow_subtour_max=float(cfg.get("slow_subtour_max", DEFAULT_META_CONFIG.slow_subtour_max)),
        observables_window=int(cfg.get("observables_window", DEFAULT_META_CONFIG.observables_window)),
    )
