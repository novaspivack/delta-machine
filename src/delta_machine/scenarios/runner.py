"""
Scenario runner with halting criteria and success metrics.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Set

import numpy as np

try:
    import numba
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..analysis import (
    compute_metric_closure,
    compute_tsp_metrics,
    extract_assignment_matrix,
    get_tsp_config,
)
from ..config import HaltingCriteria, ScenarioSpec


@dataclass
class RunResult:
    """Result of a scenario run."""

    halted: bool = False
    halt_reason: str | None = None
    success: bool = False
    success_reason: str | None = None
    final_step: int = 0
    final_dissonance: float = 0.0
    scenario_metrics: Dict[str, Any] = field(default_factory=dict)


def check_halting_criteria(
    criteria: HaltingCriteria | None,
    step: int,
    dissonance: float,
    dissonance_history: deque[float],
    residual_norms: Dict[str, float] | None = None,
    scenario_metrics: Dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """
    Check if halting criteria are met.

    Returns:
        (should_halt, reason)
    """
    if not criteria:
        return False, None

    history = list(dissonance_history)

    if criteria.max_steps and step >= criteria.max_steps:
        return True, f"max_steps ({criteria.max_steps}) reached"

    if criteria.dissonance_threshold and dissonance <= criteria.dissonance_threshold:
        return True, f"dissonance_threshold ({criteria.dissonance_threshold}) reached"

    if criteria.residual_threshold and residual_norms:
        max_residual = max(residual_norms.values()) if residual_norms else float("inf")
        if max_residual <= criteria.residual_threshold:
            return True, f"residual_threshold ({criteria.residual_threshold}) reached"

    if criteria.dissonance_plateau_steps and len(history) >= criteria.dissonance_plateau_steps:
        recent = history[-criteria.dissonance_plateau_steps :]
        if len(recent) >= 2:
            variance = max(recent) - min(recent)
            if variance <= criteria.dissonance_plateau_tolerance:
                return True, f"dissonance_plateau ({criteria.dissonance_plateau_steps} steps, tolerance {criteria.dissonance_plateau_tolerance})"

    if criteria.stagnation_window and len(history) >= criteria.stagnation_window:
        window = np.array(history[-criteria.stagnation_window :], dtype=np.float64)
        if window.size >= 2:
            delta = float(window[0] - window[-1])
            scale = float(np.max(np.abs(window)))
            scale = scale if scale > 1.0 else 1.0
            if abs(delta) <= criteria.stagnation_relative_delta * scale:
                return True, (
                    "stagnation_window "
                    f"({criteria.stagnation_window} steps, |Δ|={abs(delta):.3e}, scale={scale:.3e})"
                )

    if (
        criteria.periodic_window
        and criteria.periodic_window > 0
        and criteria.periodic_min_cycles > 1
        and len(history) >= criteria.periodic_window * criteria.periodic_min_cycles
    ):
        cycle_len = criteria.periodic_window
        total = cycle_len * criteria.periodic_min_cycles
        recent = np.array(history[-total:], dtype=np.float64)
        amplitude = float(recent.max() - recent.min())
        if amplitude >= criteria.periodic_min_amplitude:
            segments = [recent[i * cycle_len : (i + 1) * cycle_len] for i in range(criteria.periodic_min_cycles)]
            if all(len(seg) == cycle_len for seg in segments):
                ref = segments[-1]
                max_diff = max(float(np.max(np.abs(seg - ref))) for seg in segments[:-1])
                if max_diff <= criteria.periodic_tolerance:
                    return True, (
                        "periodic_cycle "
                        f"(window {cycle_len}, diff={max_diff:.3e}, tolerance={criteria.periodic_tolerance})"
                    )

    if criteria.success_condition and scenario_metrics:
        if criteria.success_condition in scenario_metrics:
            if scenario_metrics[criteria.success_condition]:
                return True, f"success_condition ({criteria.success_condition}) met"

    return False, None


# Numba-optimized kernels for discovery metrics computation
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _laplacian_numba(field: np.ndarray) -> np.ndarray:
        """JIT-compiled Laplacian stencil operator."""
        ny, nx = field.shape
        result = np.zeros_like(field)
        for i in range(ny):
            for j in range(nx):
                result[i, j] = (
                    -4.0 * field[i, j]
                    + field[(i + 1) % ny, j]
                    + field[(i - 1) % ny, j]
                    + field[i, (j + 1) % nx]
                    + field[i, (j - 1) % nx]
                )
        return result

    @njit(cache=True)
    def _smooth_field_numba(field: np.ndarray, radius: int) -> np.ndarray:
        """JIT-compiled spatial smoothing with periodic boundary conditions."""
        if radius <= 0:
            return field
        ny, nx = field.shape
        result = np.zeros_like(field)
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                for i in range(ny):
                    for j in range(nx):
                        result[i, j] += field[(i + dy) % ny, (j + dx) % nx]
                count += 1
        # Normalize by the number of offsets
        if count > 0:
            result /= count
        return result

    @njit(cache=True)
    def _compute_entropy_density_numba(real: np.ndarray, imag: np.ndarray, epsilon: float) -> np.ndarray:
        """JIT-compiled entropy density computation."""
        mag_sq = real**2 + imag**2
        return -mag_sq * np.log(mag_sq + epsilon)

    @njit(cache=True)
    def _compute_dissonance_local_numba(real: np.ndarray, imag: np.ndarray, chi_field: np.ndarray) -> np.ndarray:
        """JIT-compiled local dissonance computation."""
        return np.sqrt(real**2 + imag**2 + chi_field**2)

    @njit(cache=True)
    def _compute_metric_residual_numba(
        chi_field: np.ndarray, chi_dot_field: np.ndarray, psi_real: np.ndarray, psi_imag: np.ndarray
    ) -> np.ndarray:
        """JIT-compiled metric closure residual computation."""
        return (
            chi_field
            - 0.25 * (psi_real**2 + psi_imag**2)
            - 0.15 * chi_field**3
            + 0.1 * chi_dot_field
        )
else:
    # Fallback to numpy-based implementations if numba is not available
    def _laplacian_numba(field: np.ndarray) -> np.ndarray:
        return (
            -4.0 * field
            + np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        )

    def _smooth_field_numba(field: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return field
        acc = np.zeros_like(field)
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
                count += 1
        return acc / max(count, 1)

    def _compute_entropy_density_numba(real: np.ndarray, imag: np.ndarray, epsilon: float) -> np.ndarray:
        mag_sq = real**2 + imag**2
        return -mag_sq * np.log(mag_sq + epsilon)

    def _compute_dissonance_local_numba(real: np.ndarray, imag: np.ndarray, chi_field: np.ndarray) -> np.ndarray:
        return np.sqrt(real**2 + imag**2 + chi_field**2)

    def _compute_metric_residual_numba(
        chi_field: np.ndarray, chi_dot_field: np.ndarray, psi_real: np.ndarray, psi_imag: np.ndarray
    ) -> np.ndarray:
        return (
            chi_field
            - 0.25 * (psi_real**2 + psi_imag**2)
            - 0.15 * chi_field**3
            + 0.1 * chi_dot_field
        )


class ScenarioRunner:
    """Manages scenario-specific execution logic and metrics."""

    def __init__(self, scenario: ScenarioSpec):
        self.scenario = scenario
        self.dissonance_history: deque[float] = deque(maxlen=5000)
        self.scenario_metrics: Dict[str, Any] = {}
        self._discovery_streak: int = 0
        self._closure_streak: int = 0
        self._tsp_config = None
        self._discovery_running_fields: Dict[str, np.ndarray] = {}
        self._discovery_xtx: np.ndarray | None = None
        self._discovery_xty: np.ndarray | None = None
        self._discovery_yty: float = 0.0
        self._discovery_sample_count: int = 0
        self._discovery_last_fit_step: int = -1
        self._discovery_feature_names: list[str] | None = None
        self._discovery_last_metrics: Dict[str, Any] = {"discovery_verified": 0.0, "discovery_sampled": 0}
        self._tpuc_advantage_log: list[Dict[str, float]] = []
        self._tpuc_classical_log: list[Dict[str, float]] = []
        self._tpuc_manifest_entries: list[Dict[str, Any]] = []
        self._tpuc_manifest_path: str | None = None
        self._tpuc_logged_keys: Set[str] = set()
        self._tpuc_manifest_entry_current: Dict[str, Any] | None = None
        self._beta_measurements: list[Dict[str, float]] = []
        self._beta_manifest_entries: list[Dict[str, Any]] = []
        self._beta_manifest_path: str | None = None
        self._beta_manifest_entry_current: Dict[str, Any] | None = None

    def update_metrics(
        self,
        step: int,
        dissonance: float,
        residual_norms: Dict[str, float] | None = None,
        arrays: Dict[str, Any] | None = None,
    ):
        """Update scenario-specific metrics."""
        self.dissonance_history.append(dissonance)
        self.scenario_metrics["step"] = step
        self.scenario_metrics["dissonance"] = dissonance
        if residual_norms:
            self.scenario_metrics["residual_norms"] = residual_norms
            self.scenario_metrics["max_residual"] = max(residual_norms.values()) if residual_norms else 0.0
        if arrays:
            psi_real = arrays.get("psi_real")
            psi_imag = arrays.get("psi_imag")
            chi = arrays.get("chi")
            chi_dot = arrays.get("chi_dot")

            if psi_real is not None and psi_imag is not None:
                activity_norm = float(np.linalg.norm(psi_real) + np.linalg.norm(psi_imag))
                self.scenario_metrics["activity_norm"] = activity_norm

                gradients = np.gradient(psi_real)
                grad_magnitude = np.sqrt(sum(g ** 2 for g in gradients))
                pattern_complexity = float(np.mean(np.abs(grad_magnitude)))
                self.scenario_metrics["pattern_complexity"] = pattern_complexity

            if chi is not None:
                self.scenario_metrics["chi_energy"] = float(np.mean(chi ** 2))

            if chi_dot is not None:
                self.scenario_metrics["chi_dot_energy"] = float(np.mean(chi_dot ** 2))

            scenario_type = self.scenario.scenario_type
            if scenario_type == "sat_solver":
                self.scenario_metrics.update(self._compute_sat_metrics(arrays))
            elif scenario_type == "discovery":
                self.scenario_metrics.update(self._compute_discovery_metrics(arrays))
            elif scenario_type in {"pattern_synthesis", "metric_closure"}:
                self.scenario_metrics.update(self._compute_pattern_metrics(arrays))
            elif scenario_type == "tsp_reflexive":
                self.scenario_metrics.update(self._compute_tsp_metrics(arrays))

    def check_halting(
        self, step: int, dissonance: float, residual_norms: Dict[str, float] | None = None
    ) -> tuple[bool, str | None]:
        """Check if run should halt."""
        return check_halting_criteria(
            self.scenario.halting_criteria,
            step,
            dissonance,
            self.dissonance_history,
            residual_norms,
            self.scenario_metrics,
        )

    def evaluate_success(self) -> tuple[bool, str | None]:
        """Evaluate if run was successful based on success_metrics."""
        if not self.scenario.success_metrics:
            return False, None

        for metric_name, threshold in self.scenario.success_metrics.items():
            if metric_name not in self.scenario_metrics:
                continue
            value = self.scenario_metrics[metric_name]
            if isinstance(threshold, dict):
                if "min" in threshold and value < threshold["min"]:
                    return False, f"{metric_name} ({value}) below minimum ({threshold['min']})"
                if "max" in threshold and value > threshold["max"]:
                    return False, f"{metric_name} ({value}) above maximum ({threshold['max']})"
            else:
                # Scalar thresholds default to "max" semantics
                if value > threshold:
                    return False, f"{metric_name} ({value}) above threshold ({threshold})"

        return True, "all success_metrics satisfied"

    def get_result(self, final_step: int, final_dissonance: float) -> RunResult:
        """Generate final run result."""
        success, success_reason = self.evaluate_success()
        return RunResult(
            halted=False,
            halt_reason=None,
            success=success,
            success_reason=success_reason,
            final_step=final_step,
            final_dissonance=final_dissonance,
            scenario_metrics=self.scenario_metrics.copy(),
        )

    def _compute_discovery_metrics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        metadata = self.scenario.metadata or {}
        psi_real = arrays.get("psi_real")
        psi_imag = arrays.get("psi_imag")
        chi = arrays.get("chi")
        chi_dot = arrays.get("chi_dot")
        dissonance_field = arrays.get("dissonance")
        chi_reference = arrays.get("chi_reference")

        if psi_real is None or psi_imag is None or chi is None:
            return {"discovery_verified": 0.0}

        if dissonance_field is None:
            dissonance_field = np.zeros_like(psi_real)
        if chi_reference is None:
            chi_reference = chi

        # Use optimized Numba functions for hot loops
        def _laplacian(field: np.ndarray) -> np.ndarray:
            return _laplacian_numba(field)

        def _smooth_field(field: np.ndarray, radius: int) -> np.ndarray:
            return _smooth_field_numba(field, radius)

        def _temporal_average(name: str, field: np.ndarray) -> np.ndarray:
            window = int(metadata.get("discovery_window_samples", 0))
            if window <= 1:
                return field
            alpha = 1.0 / window
            prev = self._discovery_running_fields.get(name)
            if prev is None or prev.shape != field.shape:
                avg = field.copy()
            else:
                avg = (1.0 - alpha) * prev + alpha * field
            self._discovery_running_fields[name] = avg
            return avg

        def _compute_entropy_density(real: np.ndarray, imag: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
            epsilon = float(meta.get("epsilon", 1.0e-9))
            entropy = _compute_entropy_density_numba(real, imag, epsilon)
            radius = int(meta.get("smoothing_radius", 0))
            entropy = _smooth_field(entropy, radius)
            return entropy

        def _compute_dissonance_local(real: np.ndarray, imag: np.ndarray, chi_field: np.ndarray) -> np.ndarray:
            return _compute_dissonance_local_numba(real, imag, chi_field)

        env = {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_reference": chi_reference,
            "chi_dot": chi_dot if chi_dot is not None else np.zeros_like(psi_real),
            "dissonance": dissonance_field,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "tanh": np.tanh,
            "arctan": np.arctan,
        }

        current_step = int(self.scenario_metrics.get("step", 0))
        burn_in = int(metadata.get("discovery_burn_in_steps", 0))
        if metadata.get("log_tpuc_advantage"):
            self._log_tpuc_advantage(metadata, arrays, current_step)
        if metadata.get("log_multiscale_beta"):
            self._log_multiscale_beta(metadata, arrays, current_step)

        sample_interval = int(metadata.get("discovery_sample_interval", 1))
        if current_step < burn_in or (sample_interval > 0 and (current_step - burn_in) % sample_interval != 0):
            return {"discovery_sampled": 0, "discovery_verified": 0.0}

        curvature_meta = metadata.get("curvature_probe")
        if curvature_meta:
            mode = curvature_meta.get("mode", "laplacian")
            radius = int(curvature_meta.get("smoothing_radius", 0))
            if mode == "metric_residual":
                chi_field = chi if chi is not None else np.zeros_like(psi_real)
                chi_dot_field = chi_dot if chi_dot is not None else np.zeros_like(psi_real)
                residual = _compute_metric_residual_numba(chi_field, chi_dot_field, psi_real, psi_imag)
                curvature_field = _smooth_field(residual, radius)
            else:
                scale = float(curvature_meta.get("laplacian_scale", 1.0))
                curvature_field = _smooth_field(scale * _laplacian(psi_real), radius)
            curvature_field = _temporal_average("curvature_proxy", curvature_field)
            env["curvature_proxy"] = curvature_field
            self.scenario_metrics["curvature_proxy_rms"] = float(np.sqrt(np.mean(curvature_field**2)))
        entropy_meta = metadata.get("entropy_probe")
        if entropy_meta:
            entropy_density = _compute_entropy_density(psi_real, psi_imag, entropy_meta)
            entropy_density = _temporal_average("entropy_density", entropy_density)
            env["entropy_density"] = entropy_density
            self.scenario_metrics["entropy_density_rms"] = float(np.sqrt(np.mean(entropy_density**2)))
        if metadata.get("include_dissonance_local"):
            dissonance_local = _compute_dissonance_local(psi_real, psi_imag, chi)
            dissonance_local = _temporal_average("dissonance_local", dissonance_local)
            env["dissonance_local"] = dissonance_local
            self.scenario_metrics["dissonance_local_rms"] = float(np.sqrt(np.mean(dissonance_local**2)))

        def _eval(expr: str) -> np.ndarray:
            expr = expr.strip()
            if not expr.startswith("(") or not expr.endswith(")"):
                expr_wrapped = f"({expr})"
            else:
                expr_wrapped = expr
            try:
                value = np.asarray(eval(expr_wrapped, {"__builtins__": {}}, env))
            except Exception as exc:  # pragma: no cover - eval guard
                raise ValueError(f"Failed to evaluate discovery expression '{expr}'") from exc
            if value.shape == ():
                return np.full_like(psi_real, float(value))
            if value.shape != psi_real.shape:
                try:
                    return np.broadcast_to(value, psi_real.shape)
                except ValueError as exc:  # pragma: no cover - broadcast guard
                    raise ValueError(
                        f"Discovery expression '{expr}' produced shape {value.shape}, expected {psi_real.shape}"
                    ) from exc
            return value

        feature_specs = metadata.get("discovery_features")
        feature_arrays: list[np.ndarray] = []
        feature_names: list[str] = []

        if feature_specs:
            for spec in feature_specs:
                if isinstance(spec, str):
                    expr = spec
                    name = spec
                else:
                    expr = str(spec.get("expression", ""))
                    if not expr:
                        continue
                    name = str(spec.get("name", expr))
                feature_arrays.append(_eval(expr))
                feature_names.append(name)
        else:
            feature_arrays = [
                psi_real ** 2,
                psi_imag ** 2,
            ]
            feature_names = ["psi_real_sq", "psi_imag_sq"]

        include_bias = metadata.get("discovery_include_bias", True)
        if include_bias:
            feature_arrays.append(np.ones_like(psi_real))
            feature_names.append("bias")

        if not feature_arrays:
            return {"discovery_verified": 0.0}

        features = np.stack(feature_arrays, axis=-1)
        feature_dim = features.shape[-1]

        target_expr = metadata.get("discovery_target_expression")
        use_model_target = bool(metadata.get("discovery_use_model", False))
        if target_expr:
            target = _eval(target_expr)
        elif use_model_target and metadata.get("discovery_model"):
            model = metadata.get("discovery_model", {})
            alpha_true = float(model.get("alpha", 0.0))
            beta_true = float(model.get("beta", 0.0))
            gamma_true = float(model.get("gamma", 0.0))
            target = alpha_true * psi_real ** 2 + beta_true * psi_imag ** 2 + gamma_true
        else:
            target = chi

        mask = np.isfinite(features).all(axis=-1) & np.isfinite(target)
        if not np.any(mask):
            return {"discovery_verified": 0.0}

        X = features[mask].reshape(-1, feature_dim)
        y = target[mask].reshape(-1)
        if X.size == 0:
            return {"discovery_verified": 0.0}

        if self._discovery_feature_names is None:
            self._discovery_feature_names = feature_names.copy()
        else:
            self._discovery_feature_names = feature_names

        if self._discovery_xtx is None or self._discovery_xtx.shape != (feature_dim, feature_dim):
            self._discovery_xtx = np.zeros((feature_dim, feature_dim), dtype=np.float64)
            self._discovery_xty = np.zeros(feature_dim, dtype=np.float64)
            self._discovery_yty = 0.0
            self._discovery_sample_count = 0

        self._discovery_xtx += X.T @ X
        self._discovery_xty += X.T @ y
        self._discovery_yty += float(y @ y)
        self._discovery_sample_count += X.shape[0]

        fit_interval = int(metadata.get("discovery_fit_interval", max(sample_interval, 1)))
        min_fit_samples = int(metadata.get("discovery_fit_min_samples", feature_dim * 4))
        if self._discovery_sample_count < max(min_fit_samples, feature_dim) or (
            self._discovery_last_fit_step >= 0
            and current_step - self._discovery_last_fit_step < fit_interval
        ):
            return {
                "discovery_sampled": 0,
                "discovery_verified": float(self._discovery_last_metrics.get("discovery_verified", 0.0)),
            }

        xtx = self._discovery_xtx.copy()
        ridge = float(metadata.get("discovery_ridge_lambda", 0.0))
        if ridge > 0.0:
            xtx.flat[:: feature_dim + 1] += ridge

        try:
            coefs = np.linalg.solve(xtx, self._discovery_xty)
        except np.linalg.LinAlgError:
            coefs, *_ = np.linalg.lstsq(xtx, self._discovery_xty, rcond=None)

        predictions = X @ coefs
        residuals = y - predictions
        residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
        residual_var = float(np.var(residuals))
        denom = np.abs(y) + 1.0e-8
        abs_relative_errors = np.abs(residuals) / denom
        transport_mape = float(np.mean(abs_relative_errors))
        keep_count = max(1, int(np.floor(0.95 * abs_relative_errors.size)))
        transport_mape_trimmed = float(np.mean(np.sort(abs_relative_errors)[:keep_count]))
        predictions_rms = float(np.sqrt(np.mean(predictions ** 2)))
        transport_target_rms = float(np.sqrt(np.mean(y ** 2)))
        transport_mean_abs_target = float(np.mean(np.abs(y)))
        flux_pred = np.zeros_like(y)
        source_pred = np.zeros_like(y)
        for idx, name in enumerate(feature_names):
            contribution = coefs[idx] * feature_arrays[idx][mask]
            if name.startswith("lap_"):
                flux_pred += contribution
            else:
                source_pred += contribution
        flux_actual = y - source_pred
        flux_divergence_residual = flux_pred - flux_actual
        flux_residual_rms = float(np.sqrt(np.mean(flux_divergence_residual ** 2)))
        flux_actual_rms = float(np.sqrt(np.mean(flux_actual ** 2)))
        flux_pred_rms = float(np.sqrt(np.mean(flux_pred ** 2)))
        self.scenario_metrics["transport_mape"] = transport_mape
        self.scenario_metrics["transport_mape_trimmed"] = transport_mape_trimmed
        self.scenario_metrics["transport_predicted_rms"] = predictions_rms
        self.scenario_metrics["transport_target_rms"] = transport_target_rms
        self.scenario_metrics["transport_mean_abs_target"] = transport_mean_abs_target
        self.scenario_metrics["transport_residual_rms"] = residual_rms
        self.scenario_metrics["transport_flux_divergence_rms"] = flux_residual_rms
        self.scenario_metrics["transport_flux_divergence_actual_rms"] = flux_actual_rms
        self.scenario_metrics["transport_flux_divergence_predicted_rms"] = flux_pred_rms
        self.scenario_metrics["invariant_residual_variance"] = residual_var
        prior_cfg = metadata.get("discovery_prior") or {}
        if prior_cfg.get("enabled"):
            prior_expr = prior_cfg.get("expression")
            prior_weight = float(prior_cfg.get("weight", 0.1))
            if prior_expr:
                prior_target = _eval(prior_expr)
                prior_residual = (target - prior_target)[mask]
                prior_penalty = float(np.sqrt(np.mean(prior_residual ** 2)))
                residual_rms = float(np.sqrt(residual_rms ** 2 + (prior_weight * prior_penalty) ** 2))
                self.scenario_metrics["invariant_prior_penalty"] = prior_penalty

        truth_map = metadata.get("discovery_truth") or {}
        if not truth_map:
            model_truth = metadata.get("discovery_model") or {}
            if model_truth:
                inferred_truth: dict[str, float] = {}
                alpha = float(model_truth.get("alpha", 0.0))
                beta = float(model_truth.get("beta", 0.0))
                gamma = float(model_truth.get("gamma", 0.0))
                for name in feature_names:
                    if "psi_real" in name and "sq" in name:
                        inferred_truth[name] = alpha
                    elif "psi_imag" in name and "sq" in name:
                        inferred_truth[name] = beta
                    elif name == "bias":
                        inferred_truth[name] = gamma
                truth_map = inferred_truth
        truth_vector = np.array([float(truth_map.get(name, 0.0)) for name in feature_names], dtype=float)
        coefficient_errors = np.abs(coefs - truth_vector)
        if truth_map:
            coefficient_error = float(np.sqrt(np.mean((coefs - truth_vector) ** 2)))
            max_coeff_error = float(np.max(coefficient_errors))
        else:
            coefficient_error = float(np.linalg.norm(coefs))
            max_coeff_error = coefficient_error

        coeff_tol = float(metadata.get("coefficient_tolerance", 0.05))
        max_coeff_tol = float(metadata.get("max_coefficient_tolerance", coeff_tol))
        residual_tol = float(metadata.get("residual_tolerance", 0.05))
        success_now = residual_rms <= residual_tol and max_coeff_error <= max_coeff_tol and coefficient_error <= coeff_tol
        if metadata.get("discovery_ignore_coefficients"):
            success_now = residual_rms <= residual_tol

        if success_now:
            self._discovery_streak += 1
        else:
            self._discovery_streak = 0

        verification_window = max(1, int(metadata.get("verification_window", 10)))
        discovery_verified = self._discovery_streak >= verification_window

        truth_expr = metadata.get("discovery_truth_expression")
        if truth_expr:
            truth_field = _eval(truth_expr)
            model_alignment = chi - truth_field
        elif metadata.get("discovery_model"):
            model = metadata.get("discovery_model", {})
            truth_field = (
                float(model.get("alpha", 0.0)) * psi_real ** 2
                + float(model.get("beta", 0.0)) * psi_imag ** 2
                + float(model.get("gamma", 0.0))
            )
            model_alignment = chi - truth_field
        else:
            truth_field = target
            model_alignment = residuals.reshape(-1)

        metrics = {
            "feature_names": self._discovery_feature_names,
            "discovered_coefficients": [float(c) for c in coefs],
            "coefficient_error": coefficient_error,
            "max_coefficient_error": max_coeff_error,
            "residual_rms": residual_rms,
            "invariant_residual_variance": residual_var,
            "model_alignment_rms": float(np.sqrt(np.mean(model_alignment ** 2))),
            "transport_mape": transport_mape,
            "transport_mape_trimmed": transport_mape_trimmed,
            "transport_predicted_rms": predictions_rms,
            "transport_target_rms": transport_target_rms,
            "transport_mean_abs_target": transport_mean_abs_target,
            "transport_residual_rms": residual_rms,
            "transport_flux_divergence_rms": flux_residual_rms,
            "transport_flux_divergence_actual_rms": flux_actual_rms,
            "transport_flux_divergence_predicted_rms": flux_pred_rms,
            "discovery_streak": self._discovery_streak,
            "discovery_verified": 1.0 if discovery_verified else 0.0,
            "discovery_sampled": 1,
        }

        decay = float(metadata.get("discovery_accumulator_decay", 1.0))
        if 0.0 <= decay < 1.0:
            self._discovery_xtx *= decay
            self._discovery_xty *= decay
            self._discovery_yty *= decay
            self._discovery_sample_count = max(int(self._discovery_sample_count * decay), 0)
        self._discovery_last_fit_step = current_step
        self._discovery_last_metrics = metrics

        return metrics

    def _log_tpuc_advantage(self, metadata: Dict[str, Any], arrays: Dict[str, Any], current_step: int) -> None:
        interval = int(metadata.get("tpuc_log_interval", 50))
        if interval <= 0 or current_step <= 0 or current_step % interval != 0:
            return
        entry = self._tpuc_manifest_entry_current
        if entry is None:
            return
        chi = arrays.get("chi")
        chi_reference = arrays.get("chi_reference")
        if chi is None or chi_reference is None:
            return
        key = f"{entry.get('family','unknown')}:{entry.get('sample_index',-1)}:{current_step}"
        if key in self._tpuc_logged_keys:
            return
        psi_real = arrays.get("psi_real")
        psi_imag = arrays.get("psi_imag")
        solution_error = float(np.sqrt(np.mean((chi - chi_reference) ** 2)))
        energy = 0.0
        if psi_real is not None and psi_imag is not None:
            energy = float(np.mean(psi_real**2 + psi_imag**2))
        latency = float(current_step * self.scenario.timestep)
        self.scenario_metrics["chi_solution_error"] = solution_error
        self.scenario_metrics["tpuc_energy_joules"] = energy
        self.scenario_metrics["tpuc_latency_seconds"] = latency
        self._tpuc_advantage_log.append(
            {
                "latency_seconds": latency,
                "solution_error": solution_error,
                "tpuc_energy_joules": energy,
                "step": current_step,
            }
        )
        classical_latency = float(entry.get("classical_cost", latency))
        holographic_bias = float(entry.get("holographic_bias", 0.0))
        conditioning = float(entry.get("conditioning", 1.0))
        classical_error = float(np.abs(holographic_bias) / max(conditioning, 1.0))
        self._tpuc_classical_log.append(
            {
                "latency_seconds": classical_latency,
                "solution_error": classical_error,
            }
        )
        self._tpuc_manifest_entries.append(dict(entry))
        self._tpuc_logged_keys.add(key)

    def _log_multiscale_beta(self, metadata: Dict[str, Any], arrays: Dict[str, Any], current_step: int) -> None:
        interval = int(metadata.get("beta_log_interval", 50))
        if interval <= 0 or current_step <= 0 or current_step % interval != 0:
            return
        entry = self._beta_manifest_entry_current
        psi_real = arrays.get("psi_real")
        chi_dot = arrays.get("chi_dot")
        if entry is None or psi_real is None or chi_dot is None:
            return
        scale = float(entry.get("scale", metadata.get("scale", 1)))
        coupling = float(np.mean(psi_real))
        flow = float(np.mean(chi_dot))
        self._beta_measurements.append(
            {
                "scale": scale,
                "coupling": coupling,
                "flow": flow,
                "step": current_step,
            }
        )
        self._beta_manifest_entries.append(dict(entry))

    def export_artifacts(self, run_dir: Path | None = None) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        if self._tpuc_advantage_log and run_dir is not None:
            adv_dir = run_dir / "tpuc_advantage"
            adv_dir.mkdir(parents=True, exist_ok=True)
            dsac_latency = np.array([row["latency_seconds"] for row in self._tpuc_advantage_log], dtype=np.float64)
            dsac_error = np.array([row["solution_error"] for row in self._tpuc_advantage_log], dtype=np.float64)
            tpuc_energy = np.array([row["tpuc_energy_joules"] for row in self._tpuc_advantage_log], dtype=np.float64)
            dsac_path = adv_dir / "dsac_results.npz"
            np.savez(dsac_path, latency_seconds=dsac_latency, solution_error=dsac_error, tpuc_energy_joules=tpuc_energy)

            classical_latency = np.array([row["latency_seconds"] for row in self._tpuc_classical_log], dtype=np.float64)
            classical_error = np.array([row["solution_error"] for row in self._tpuc_classical_log], dtype=np.float64)
            classical_path = adv_dir / "classical_results.npz"
            np.savez(classical_path, latency_seconds=classical_latency, solution_error=classical_error)

            manifest_subset_path = adv_dir / "manifest_subset.json"
            with manifest_subset_path.open("w", encoding="utf-8") as handle:
                json.dump(self._tpuc_manifest_entries, handle, indent=2)

            extras["tpuc_advantage"] = {
                "sample_count": len(self._tpuc_advantage_log),
                "dsac_results": os.path.relpath(dsac_path, run_dir),
                "classical_results": os.path.relpath(classical_path, run_dir),
                "manifest_subset": os.path.relpath(manifest_subset_path, run_dir),
                "manifest_source": self._tpuc_manifest_path,
            }

        if self._beta_measurements and run_dir is not None:
            beta_dir = run_dir / "te2rg_beta"
            beta_dir.mkdir(parents=True, exist_ok=True)
            measurements = [
                {"scale": row["scale"], "coupling": row["coupling"], "flow": row["flow"]}
                for row in self._beta_measurements
            ]
            measurements_path = beta_dir / "beta_measurements.json"
            with measurements_path.open("w", encoding="utf-8") as handle:
                json.dump(measurements, handle, indent=2)
            manifest_subset_path = beta_dir / "manifest_subset.json"
            with manifest_subset_path.open("w", encoding="utf-8") as handle:
                json.dump(self._beta_manifest_entries, handle, indent=2)
            extras["te2rg_beta"] = {
                "sample_count": len(self._beta_measurements),
                "measurements": os.path.relpath(measurements_path, run_dir),
                "manifest_subset": os.path.relpath(manifest_subset_path, run_dir),
                "manifest_source": self._beta_manifest_path,
            }

        return extras

    def _compute_sat_metrics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        metadata = self.scenario.metadata or {}
        beta = float(metadata.get("beta", 3.0))
        clause_tol = float(metadata.get("clause_tolerance", 0.01))
        var_map = metadata.get(
            "variable_map",
            {"x1": "psi_real", "x2": "psi_imag", "x3": "chi"},
        )

        assignment: Dict[str, int] = {}
        field_means: Dict[str, float] = {}
        for var, field_name in var_map.items():
            field_data = arrays.get(field_name)
            if field_data is None:
                continue
            mean_val = float(np.mean(field_data))
            field_means[var] = mean_val
            assignment[var] = 1 if mean_val >= 0.0 else 0

        clauses = list(self._iter_clauses())
        clause_deficits: list[float] = []
        clause_satisfaction: list[float] = []

        total_weight = 0.0
        max_weight = sum(weight for weight, _ in clauses)

        for weight, literals in clauses:
            unsatisfied = None
            for var, positive in literals:
                field_data = arrays[var_map[var]]
                literal_sat = 0.5 * (1.0 + np.tanh(beta * field_data)) if positive else 0.5 * (1.0 - np.tanh(beta * field_data))
                literal_unsat = 1.0 - literal_sat
                unsatisfied = literal_unsat if unsatisfied is None else unsatisfied * literal_unsat
            deficit = float(np.mean(unsatisfied)) if unsatisfied is not None else 0.0
            clause_deficits.append(deficit)
            clause_satisfaction.append(float(1.0 - deficit))

            if self._clause_satisfied_binary(literals, assignment):
                total_weight += weight

        max_clause_deficit = max(clause_deficits) if clause_deficits else 0.0
        target_weight = metadata.get("target_weight")
        if target_weight is None:
            target_weight = max_weight
        target_weight = float(target_weight)
        weight_tol = float(metadata.get("weight_tolerance", 1e-6))
        require_clause_tol = bool(metadata.get("require_clause_tolerance", True))
        clause_requirement = max_clause_deficit <= clause_tol if require_clause_tol else True
        solution_verified = clause_requirement and (total_weight + weight_tol >= target_weight)

        return {
            "clause_deficits": clause_deficits,
            "clause_satisfaction": clause_satisfaction,
            "max_clause_deficit": max_clause_deficit,
            "assignment_values": assignment,
            "assignment_field_means": field_means,
            "assignment_satisfied": 1.0 if total_weight + weight_tol >= target_weight else 0.0,
            "weighted_score": total_weight,
            "solution_verified": 1.0 if solution_verified else 0.0,
        }

    def _compute_pattern_metrics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        psi_real = arrays.get("psi_real")
        psi_imag = arrays.get("psi_imag")
        chi = arrays.get("chi")

        if psi_real is None or psi_imag is None or chi is None:
            return {"closure_verified": 0.0}

        metadata = self.scenario.metadata or {}
        closure_stats = compute_metric_closure(psi_real, psi_imag, chi, metadata)

        closure_tol = float(metadata.get("closure_tolerance", 0.05))
        divergence_tol = float(metadata.get("divergence_tolerance", closure_tol))
        chi_tol = float(metadata.get("chi_alignment_tolerance", closure_tol))
        curl_tol = float(metadata.get("curl_alignment_tolerance", closure_tol))

        success_now = (
            closure_stats["closure_error"] <= closure_tol
            and closure_stats["divergence_rms"] <= divergence_tol
            and closure_stats["chi_alignment_rms"] <= chi_tol
            and closure_stats["curl_alignment_rms"] <= curl_tol
        )

        if success_now:
            self._closure_streak += 1
        else:
            self._closure_streak = 0

        verification_window = max(1, int(metadata.get("verification_window", 20)))
        closure_verified = self._closure_streak >= verification_window

        closure_stats.update(
            {
                "closure_streak": self._closure_streak,
                "closure_verified": 1.0 if closure_verified else 0.0,
            }
        )

        return closure_stats

    def _compute_tsp_metrics(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        psi_real = arrays.get("psi_real")
        if psi_real is None:
            return {"stochasticity_verified": 0.0}

        if self._tsp_config is None:
            self._tsp_config = get_tsp_config(self.scenario.metadata or {})

        assignment = extract_assignment_matrix(psi_real, self._tsp_config)
        metrics = compute_tsp_metrics(assignment, self._tsp_config)
        return metrics

    def _iter_clauses(self) -> Iterable[tuple[float, list[tuple[str, bool]]]]:
        clauses = self.scenario.metadata.get("cnf_clauses", []) if self.scenario.metadata else []
        parsed = []
        for clause in clauses:
            if isinstance(clause, dict):
                weight = float(clause.get("weight", 1.0))
                literals = clause.get("literals", [])
            else:
                weight = 1.0
                literals = clause
            parsed_literals: list[tuple[str, bool]] = []
            for literal in literals:
                if isinstance(literal, dict):
                    var = literal.get("var")
                    positive = bool(literal.get("positive", True))
                else:
                    var, positive = literal
                if isinstance(positive, str):
                    positive = positive.lower() in {"true", "1", "yes"}
                parsed_literals.append((var, bool(positive)))
            parsed.append((weight, parsed_literals))
        return parsed

    def _clause_satisfied_binary(self, literals: list[tuple[str, bool]], assignment: Dict[str, int]) -> bool:
        for var, positive in literals:
            value = assignment.get(var, 0)
            literal_truth = value == 1 if positive else value == 0
            if literal_truth:
                return True
        return False

    def reset_logs(self) -> None:
        self._tpuc_advantage_log.clear()
        self._tpuc_classical_log.clear()
        self._tpuc_manifest_entries.clear()
        self._tpuc_manifest_path = None
        self._tpuc_logged_keys.clear()
        self._tpuc_manifest_entry_current = None
        self._beta_measurements.clear()
        self._beta_manifest_entries.clear()
        self._beta_manifest_path = None
        self._beta_manifest_entry_current = None

    def ingest_initial_metadata(self, metadata: Dict[str, Any]) -> None:
        entry = metadata.get("tpuc_manifest_entry")
        if isinstance(entry, dict):
            self._tpuc_manifest_entry_current = dict(entry)
        manifest_path = metadata.get("tpuc_manifest_path")
        if isinstance(manifest_path, str):
            self._tpuc_manifest_path = manifest_path
        entry = metadata.get("multiscale_manifest_entry")
        if isinstance(entry, dict):
            self._beta_manifest_entry_current = dict(entry)
        manifest_path = metadata.get("multiscale_manifest_path")
        if isinstance(manifest_path, str):
            self._beta_manifest_path = manifest_path

