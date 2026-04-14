"""Instrumentation for TE₂.H boundary↔bulk holographic experiments.

This module is part of the Band C execution log described in
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md`
and draws on the computational holography results of
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.Y_COMPUTATIONAL_HOLOGRAPHY/TE_1.Y_COMPTUTATIONAL_HOLOGRAPHY_THMS.md`.

The routines compute quantitative signatures—mutual information, PSC efficiency,
and dimensional-lift diagnostics—needed to corroborate reflexive holographic equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft2
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class BoundaryBulkMetrics:
    """Summary statistics for a single boundary↔bulk pairing."""

    boundary_energy: float
    bulk_energy: float
    compression_ratio: float
    mutual_information: float
    psc_efficiency: float
    edge_alignment: float
    dimensional_lift_signal: float
    spectral_overlap: float


def compute_boundary_bulk_metrics(
    boundary_projection: ArrayLike,
    bulk_field: ArrayLike,
    boundary_mask: ArrayLike,
    smoothing_sigma: float = 1.5,
    histogram_bins: int = 128,
) -> BoundaryBulkMetrics:
    """Compute holographic equivalence diagnostics for a boundary/bulk sample."""

    boundary_projection = np.asarray(boundary_projection, dtype=np.float64)
    bulk_field = np.asarray(bulk_field, dtype=np.float64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)

    if boundary_projection.shape != bulk_field.shape:
        raise ValueError("boundary and bulk fields must share lattice shape")
    if boundary_mask.shape != bulk_field.shape:
        raise ValueError("boundary mask must match field shape")

    smoothed_boundary = gaussian_filter(boundary_projection, sigma=smoothing_sigma)
    smoothed_bulk = gaussian_filter(bulk_field, sigma=smoothing_sigma)

    boundary_energy = float(np.sum(smoothed_boundary**2))
    bulk_energy = float(np.sum(smoothed_bulk**2))
    compression_ratio = boundary_energy / max(bulk_energy, 1e-12)

    mutual_information = _estimate_mutual_information(
        smoothed_boundary[boundary_mask], smoothed_bulk[boundary_mask], bins=histogram_bins
    )

    psc_efficiency = _psc_topological_efficiency(smoothed_boundary, smoothed_bulk, boundary_mask)
    edge_alignment = _edge_alignment_score(smoothed_boundary, smoothed_bulk)
    dimensional_lift_signal = _dimensional_lift_signature(smoothed_boundary, smoothed_bulk, boundary_mask)
    spectral_overlap = _spectral_overlap(smoothed_boundary, smoothed_bulk)

    return BoundaryBulkMetrics(
        boundary_energy=boundary_energy,
        bulk_energy=bulk_energy,
        compression_ratio=compression_ratio,
        mutual_information=mutual_information,
        psc_efficiency=psc_efficiency,
        edge_alignment=edge_alignment,
        dimensional_lift_signal=dimensional_lift_signal,
        spectral_overlap=spectral_overlap,
    )


def _estimate_mutual_information(surface: np.ndarray, bulk: np.ndarray, bins: int) -> float:
    eps = 1e-12
    hist, _, _ = np.histogram2d(surface, bulk, bins=bins, density=True)
    hist += eps
    pxy = hist / np.sum(hist)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    px_py = px @ py
    mi = np.sum(pxy * np.log(pxy / px_py))
    return float(max(mi, 0.0))


def _psc_topological_efficiency(
    boundary: np.ndarray,
    bulk: np.ndarray,
    mask: np.ndarray,
) -> float:
    # PSC (Projected Surface Capacity) efficiency approximated via flux conservation.
    grad_boundary = np.gradient(boundary)
    grad_bulk = np.gradient(bulk)
    surface_flux = np.sqrt(np.sum([(g * mask) ** 2 for g in grad_boundary], axis=0))
    bulk_flux = np.sqrt(np.sum([g**2 for g in grad_bulk], axis=0))
    numerator = float(np.sum(surface_flux))
    denominator = float(np.sum(bulk_flux))
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def _edge_alignment_score(boundary: np.ndarray, bulk: np.ndarray) -> float:
    boundary_edges = np.gradient(boundary)
    bulk_edges = np.gradient(bulk)
    dot_sum = 0.0
    norm_boundary = 0.0
    norm_bulk = 0.0
    for db, bb in zip(boundary_edges, bulk_edges):
        dot_sum += float(np.sum(db * bb))
        norm_boundary += float(np.sum(db**2))
        norm_bulk += float(np.sum(bb**2))
    if norm_boundary <= 1e-12 or norm_bulk <= 1e-12:
        return 0.0
    return dot_sum / math.sqrt(norm_boundary * norm_bulk)


def _dimensional_lift_signature(boundary: np.ndarray, bulk: np.ndarray, mask: np.ndarray) -> float:
    boundary_energy = np.sum((boundary * mask) ** 2)
    interior_energy = np.sum((bulk * (~mask)) ** 2)
    total = boundary_energy + interior_energy
    if total <= 1e-12:
        return 0.0
    return float(interior_energy / total)


def _spectral_overlap(boundary: np.ndarray, bulk: np.ndarray) -> float:
    boundary_spectrum = fft2(boundary)
    bulk_spectrum = fft2(bulk)
    numerator = np.sum(np.abs(boundary_spectrum * np.conj(bulk_spectrum)))
    denominator = math.sqrt(np.sum(np.abs(boundary_spectrum) ** 2) * np.sum(np.abs(bulk_spectrum) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.real(numerator / denominator))
