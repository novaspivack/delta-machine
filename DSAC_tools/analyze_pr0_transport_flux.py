#!/usr/bin/env python3
"""Flux diagnostics CLI for the PR-0 transport law (see `notes/1.37_phase2_pr0_transport_proof_package.md`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from delta_machine.analysis.transport_flux import FeaturePlanes, compute_flux_diagnostics
from delta_machine.initial_conditions.generators import PR0TransportDatasetGenerator

FEATURE_ORDER = (
    "rho_centered",
    "lap_rho",
    "gamma_centered",
    "lap_gamma",
    "diff_grad_norm",
    "gamma_normalized",
    "bias",
)


def load_coefficients(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    if "coefficients" in data and isinstance(data["coefficients"], dict):
        return {k: float(v) for k, v in data["coefficients"].items()}
    return {k: float(v) for k, v in data.items()}


def load_dataset_sample(dataset_dir: Path, regime: str, sample_index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    generator = PR0TransportDatasetGenerator(dataset_dir=dataset_dir)
    arrays = generator.load_sample(regime=regime, sample_index=sample_index)
    rho = arrays["rho"]
    gamma = arrays["gamma"]
    grad_x = arrays["rho_grad_x"]
    grad_y = arrays["rho_grad_y"]
    rho_dot = arrays["rho_dot"]

    feature_planes = FeaturePlanes.from_fields(
        rho=rho,
        grad_x=grad_x,
        grad_y=grad_y,
        gamma=gamma,
    ).as_dict()

    target = rho_dot - np.mean(rho_dot)
    return target, feature_planes


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse PR-0 transport flux decomposition")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing generated PR-0 transport datasets")
    parser.add_argument("--regime", type=str, required=True, help="Dataset regime to load (e.g. baseline, high_drive, asymmetric)")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the regime dataset")
    parser.add_argument("--coeff-json", type=Path, required=True, help="JSON file mapping feature names to coefficients")
    parser.add_argument("--trim", type=float, default=0.95, help="Trim percentile for MAPE calculation")
    args = parser.parse_args()

    coeffs = load_coefficients(args.coeff_json)
    rho_dot, feature_planes = load_dataset_sample(args.dataset_dir, args.regime, args.sample_index)

    diagnostics = compute_flux_diagnostics(
        feature_planes=feature_planes,
        coefficients=coeffs,
        feature_order=FEATURE_ORDER,
        rho_dot=rho_dot,
        trim_fraction=args.trim,
    )

    print("=== PR-0 Transport Flux Diagnostics ===")
    print(f"dataset_dir : {args.dataset_dir}")
    print(f"regime      : {args.regime}")
    print(f"sample_index: {args.sample_index}")
    print(f"coeff_source: {args.coeff_json}")
    print("--- metrics ---")
    print(f"residual_rms        = {diagnostics.residual_rms:.6e}")
    print(f"flux_rms            = {diagnostics.flux_rms:.6e}")
    print(f"source_rms          = {diagnostics.source_rms:.6e}")
    print(f"trimmed_mape ({args.trim:.2f}) = {diagnostics.trimmed_mape:.6e}")


if __name__ == "__main__":
    main()
