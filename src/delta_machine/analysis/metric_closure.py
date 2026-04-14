"""Metric closure analysis helpers for DSAC scenarios.

Design references:
- 1.5 DSAC Scenario Roadmap:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md
"""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Numba-optimized gradient computation (replaces np.gradient)
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _gradient_axis0(field: np.ndarray) -> np.ndarray:
        """JIT-compiled gradient along axis 0 (rows) with periodic boundaries."""
        ny, nx = field.shape
        result = np.zeros_like(field)
        for i in range(ny):
            for j in range(nx):
                result[i, j] = 0.5 * (field[(i + 1) % ny, j] - field[(i - 1) % ny, j])
        return result

    @njit(cache=True)
    def _gradient_axis1(field: np.ndarray) -> np.ndarray:
        """JIT-compiled gradient along axis 1 (columns) with periodic boundaries."""
        ny, nx = field.shape
        result = np.zeros_like(field)
        for i in range(ny):
            for j in range(nx):
                result[i, j] = 0.5 * (field[i, (j + 1) % nx] - field[i, (j - 1) % nx])
        return result

    @njit(cache=True)
    def _laplacian_numba(field: np.ndarray) -> np.ndarray:
        """JIT-compiled Laplacian using explicit stencil (replaces np.gradient-based version)."""
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
else:
    # Fallback to numpy implementations
    def _gradient_axis0(field: np.ndarray) -> np.ndarray:
        return np.gradient(field, axis=0)

    def _gradient_axis1(field: np.ndarray) -> np.ndarray:
        return np.gradient(field, axis=1)

    def _laplacian_numba(field: np.ndarray) -> np.ndarray:
        grad_y = np.gradient(field, axis=0)
        grad_x = np.gradient(field, axis=1)
        dyy = np.gradient(grad_y, axis=0)
        dxx = np.gradient(grad_x, axis=1)
        return dxx + dyy


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Compute the discrete Laplacian using optimized Numba version."""
    return _laplacian_numba(field)


def compute_metric_closure(
    psi_real: np.ndarray,
    psi_imag: np.ndarray,
    chi: np.ndarray,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, float]:
    """Compute metric closure diagnostics for pattern synthesis scenarios."""
    psi_real = np.asarray(psi_real, dtype=np.float64)
    psi_imag = np.asarray(psi_imag, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)

    psi_mag_sq = psi_real ** 2 + psi_imag ** 2

    # Use optimized Numba gradient functions
    grad_pr_y = _gradient_axis0(psi_real)
    grad_pr_x = _gradient_axis1(psi_real)
    grad_pi_y = _gradient_axis0(psi_imag)
    grad_pi_x = _gradient_axis1(psi_imag)

    divergence = grad_pr_x + grad_pi_y
    curl_z = grad_pi_x - grad_pr_y
    laplace_mag = _laplacian(psi_mag_sq)

    magnitude_scale = float(np.sqrt(np.mean(psi_mag_sq) + 1e-8))
    closure_model = metadata.get("closure_model", {}) if metadata else {}
    chi_scale = float(closure_model.get("chi_scale", 0.25))
    chi_offset = float(closure_model.get("chi_offset", 0.0))
    chi_expected = chi_scale * psi_mag_sq + chi_offset
    chi_scale_norm = float(
        max(
            np.mean(np.abs(chi)),
            np.mean(np.abs(chi_expected)),
            magnitude_scale,
            1e-3,
        )
    )

    divergence_target = float(closure_model.get("divergence_target", 0.0))
    laplacian_target = float(closure_model.get("laplacian_target", 0.0))

    expected_chi = chi_expected
    chi_alignment = chi - expected_chi

    divergence_error = (divergence - divergence_target) / magnitude_scale
    laplacian_error = (laplace_mag - laplacian_target) / max(magnitude_scale ** 2, 1e-8)

    curl_alignment_target = closure_model.get("curl_alignment", "chi")
    if isinstance(curl_alignment_target, str) and curl_alignment_target.lower() == "chi":
        curl_alignment = (curl_z - chi) / max(chi_scale_norm, 1e-8)
    else:
        try:
            numeric_target = float(curl_alignment_target)
        except (TypeError, ValueError):
            numeric_target = 0.0
        curl_alignment = (curl_z - numeric_target) / magnitude_scale

    chi_alignment = chi_alignment / max(chi_scale_norm, 1e-8)

    stats = {
        "divergence_rms": float(np.sqrt(np.mean(divergence_error ** 2))),
        "chi_alignment_rms": float(np.sqrt(np.mean(chi_alignment ** 2))),
        "curl_alignment_rms": float(np.sqrt(np.mean(curl_alignment ** 2))),
        "laplacian_rms": float(np.sqrt(np.mean(laplacian_error ** 2))),
        "divergence_mean": float(np.mean(divergence) / magnitude_scale),
        "curl_mean": float(np.mean(curl_z) / max(chi_scale_norm, magnitude_scale)),
        "chi_alignment_mean": float(np.mean(chi_alignment)),
    }

    stats["closure_error"] = float(
        np.sqrt(
            stats["divergence_rms"] ** 2
            + stats["chi_alignment_rms"] ** 2
            + stats["curl_alignment_rms"] ** 2
            + stats["laplacian_rms"] ** 2
        )
    )

    return stats
