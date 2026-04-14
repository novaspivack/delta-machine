"""Utilities for analysing PR-0 transport flux decompositions.

Implements diagnostic helpers referenced in `notes/1.37_phase2_pr0_transport_proof_package.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

_EPS = 1.0e-6


def _central_laplacian(field: np.ndarray) -> np.ndarray:
    """Compute the 2-D central-difference Laplacian with periodic wrap."""
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _gradient_norm(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


@dataclass
class FeaturePlanes:
    rho_centered: np.ndarray
    lap_rho: np.ndarray
    gamma_centered: np.ndarray
    lap_gamma: np.ndarray
    diff_grad_norm: np.ndarray
    gamma_normalized: np.ndarray
    bias: np.ndarray

    @classmethod
    def from_fields(
        cls,
        rho: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        gamma: np.ndarray,
    ) -> "FeaturePlanes":
        rho_centered = rho - np.mean(rho)
        gamma_centered = gamma - np.mean(gamma)
        lap_rho = _central_laplacian(rho)
        lap_gamma = _central_laplacian(gamma)

        grad_norm = _gradient_norm(grad_x, grad_y)
        grad_rms = np.sqrt(np.mean(grad_norm ** 2)) + _EPS
        diff_grad_norm = grad_norm / grad_rms

        gamma_rms = np.sqrt(np.mean(gamma ** 2)) + _EPS
        gamma_normalized = gamma / gamma_rms

        bias = np.ones_like(rho)

        return cls(
            rho_centered=rho_centered,
            lap_rho=lap_rho,
            gamma_centered=gamma_centered,
            lap_gamma=lap_gamma,
            diff_grad_norm=diff_grad_norm,
            gamma_normalized=gamma_normalized,
            bias=bias,
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "rho_centered": self.rho_centered,
            "lap_rho": self.lap_rho,
            "gamma_centered": self.gamma_centered,
            "lap_gamma": self.lap_gamma,
            "diff_grad_norm": self.diff_grad_norm,
            "gamma_normalized": self.gamma_normalized,
            "bias": self.bias,
        }


@dataclass
class FluxDiagnostics:
    predictions: np.ndarray
    source_terms: np.ndarray
    flux_terms: np.ndarray
    residuals: np.ndarray
    residual_rms: float
    trimmed_mape: float
    flux_rms: float
    source_rms: float


def _resolve_coefficients(
    coefficients: Mapping[str, float] | Sequence[float],
    feature_order: Iterable[str],
) -> Dict[str, float]:
    if isinstance(coefficients, Mapping):
        return {name: float(coefficients.get(name, 0.0)) for name in feature_order}
    ordered = list(feature_order)
    return {name: float(coefficients[idx]) if idx < len(coefficients) else 0.0 for idx, name in enumerate(ordered)}


def compute_flux_diagnostics(
    feature_planes: Mapping[str, np.ndarray],
    coefficients: Mapping[str, float] | Sequence[float],
    feature_order: Sequence[str],
    rho_dot: np.ndarray,
    trim_fraction: float = 0.95,
) -> FluxDiagnostics:
    """Reconstruct source/flux contributions using the reduced transport basis."""
    coeff_map = _resolve_coefficients(coefficients, feature_order)

    flux_terms = np.zeros_like(rho_dot)
    source_terms = np.zeros_like(rho_dot)

    for name in feature_order:
        plane = feature_planes[name]
        contribution = coeff_map[name] * plane
        if name.startswith("lap_"):
            flux_terms += contribution
        else:
            source_terms += contribution

    predictions = flux_terms + source_terms
    residuals = rho_dot - predictions

    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    flux_rms = float(np.sqrt(np.mean(flux_terms ** 2)))
    source_rms = float(np.sqrt(np.mean(source_terms ** 2)))

    denom = np.abs(rho_dot) + _EPS
    relative_errors = np.abs(residuals) / denom
    sorted_errors = np.sort(relative_errors.reshape(-1))
    keep = max(1, int(np.floor(trim_fraction * sorted_errors.size)))
    trimmed_mape = float(np.mean(sorted_errors[:keep]))

    return FluxDiagnostics(
        predictions=predictions,
        source_terms=source_terms,
        flux_terms=flux_terms,
        residuals=residuals,
        residual_rms=residual_rms,
        trimmed_mape=trimmed_mape,
        flux_rms=flux_rms,
        source_rms=source_rms,
    )
