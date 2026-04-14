#!/usr/bin/env python3
"""Phase II curvature invariant simplification tool.

This utility re-fits the curvature proxy `R` as a reduced-basis function of
local dissonance and entropy fields (`D`, `E`) taken from archived DSAC runs.
It implements the analytic simplification path requested in
`notes/1.25_phase2_curvature_invariant_plan.md` and
`notes/1.26_phase2_curvature_invariant_results.md`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit curvature invariant with a reduced basis.")
    parser.add_argument(
        "--runs-csv",
        type=Path,
        help="CSV file produced by the ensemble harness; run_dir column will be consumed to locate tensors.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        default=None,
        help="Explicit run directory containing final_state tensors. Can be provided multiple times.",
    )
    parser.add_argument(
        "--smoothing-radius",
        type=int,
        default=2,
        help="Spatial smoothing radius to match the discovery configuration (default: 2).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-6,
        help="Small positive constant used to keep logarithms and ratios well-defined.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional fraction of lattice samples to retain from each run (0 < frac <= 1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on the total number of samples kept across runs (applied after concatenation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="PRNG seed for subsampling (default: 12345).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If provided, dump coefficients and fit metrics to this JSON file.",
    )
    return parser.parse_args()


def _smooth_field(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    acc = np.zeros_like(field)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
            count += 1
    return acc / max(count, 1)


def _load_field(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / "final_state" / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing tensor {path}")
    return np.load(path)


def _gather_run_dirs(args: argparse.Namespace) -> List[Path]:
    run_dirs: List[Path] = []
    if args.run_dirs:
        run_dirs.extend(Path(p) for p in args.run_dirs)
    if args.runs_csv:
        with args.runs_csv.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run_dir = row.get("run_dir")
                if run_dir:
                    run_dirs.append(Path(run_dir.strip()))
    unique_dirs: List[Path] = []
    seen = set()
    for path in run_dirs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_dirs.append(resolved)
    return unique_dirs


def _compute_fields(run_dir: Path, radius: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi_r = _load_field(run_dir, "psi_real")
    psi_i = _load_field(run_dir, "psi_imag")
    chi = _load_field(run_dir, "chi")
    chi_dot_path = run_dir / "final_state" / "chi_dot.npy"
    chi_dot = np.load(chi_dot_path) if chi_dot_path.exists() else np.zeros_like(chi)

    mag_sq = psi_r**2 + psi_i**2
    curvature = chi - 0.25 * mag_sq - 0.15 * chi**3 + 0.1 * chi_dot
    curvature = _smooth_field(curvature, radius)

    dissonance = np.sqrt(mag_sq + chi**2)
    dissonance = _smooth_field(dissonance, radius)

    entropy = -mag_sq * np.log(mag_sq + epsilon)
    entropy = _smooth_field(entropy, radius)
    return dissonance, entropy, curvature


def _subsample_mask(size: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    if frac >= 1.0:
        return np.ones(size, dtype=bool)
    keep = int(max(1, np.floor(frac * size)))
    indices = np.arange(size)
    rng.shuffle(indices)
    mask = np.zeros(size, dtype=bool)
    mask[indices[:keep]] = True
    return mask


def _prepare_dataset(
    run_dirs: Sequence[Path],
    radius: int,
    epsilon: float,
    sample_frac: float,
    max_samples: int | None,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []

    for run_dir in run_dirs:
        dissonance, entropy, curvature = _compute_fields(run_dir, radius, epsilon)
        mask = (dissonance > epsilon) & (np.abs(entropy) > epsilon)
        flat_mask = mask.reshape(-1)
        if not np.any(flat_mask):
            continue
        if sample_frac < 1.0:
            subsample = _subsample_mask(flat_mask.size, sample_frac, rng)
            flat_mask &= subsample
        if not np.any(flat_mask):
            continue
        D = dissonance.reshape(-1)[flat_mask]
        E = entropy.reshape(-1)[flat_mask]
        R = curvature.reshape(-1)[flat_mask]
        feature_list.append(np.column_stack((D, E, D**2, E**2, D * E, np.log(D), np.log(np.abs(E)), np.ones_like(D))))
        target_list.append(R)

    if not feature_list:
        raise RuntimeError("No samples collected; ensure the run directories contain valid tensors.")

    X = np.vstack(feature_list)
    y = np.concatenate(target_list)

    if max_samples is not None and X.shape[0] > max_samples:
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)
        indices = indices[:max_samples]
        X = X[indices]
        y = y[indices]
    return X, y


def _fit_least_squares(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    coeffs, residuals, rank, singular_vals = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    rms = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    target_rms = float(np.sqrt(np.mean(y**2)))
    rel = float(rms / max(target_rms, 1.0e-12))
    metrics = {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "rms": rms,
        "target_rms": target_rms,
        "relative_rms": rel,
        "rank": int(rank),
        "residual_sum_squares": float(residuals[0]) if residuals.size else 0.0,
        "singular_values": singular_vals.tolist(),
    }
    return coeffs, metrics


def _format_coefficients(coeffs: Sequence[float], names: Sequence[str]) -> str:
    return "\n".join(f"  {name:>16s}: {value:+.6f}" for name, value in zip(names, coeffs))


def main() -> None:
    args = _parse_args()
    run_dirs = _gather_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories supplied via --run-dir or --runs-csv")

    rng = np.random.default_rng(args.seed)
    X, y = _prepare_dataset(
        run_dirs=run_dirs,
        radius=args.smoothing_radius,
        epsilon=args.epsilon,
        sample_frac=max(0.0, min(1.0, args.sample_frac)),
        max_samples=args.max_samples,
        rng=rng,
    )

    coeffs, metrics = _fit_least_squares(X, y)
    feature_names = [
        "dissonance_local",
        "entropy_density",
        "dissonance_sq",
        "entropy_sq",
        "cross_term",
        "log_dissonance",
        "log_entropy",
        "bias",
    ]

    print("Refit complete using runs:")
    for path in run_dirs:
        print(f"  - {path}")
    print("\nCoefficients (R ≈ Σ c_i f_i):")
    print(_format_coefficients(coeffs, feature_names))
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if args.output_json:
        payload = {
            "feature_names": feature_names,
            "coefficients": coeffs.tolist(),
            "metrics": metrics,
            "runs": [str(p) for p in run_dirs],
            "parameters": {
                "smoothing_radius": args.smoothing_radius,
                "epsilon": args.epsilon,
                "sample_frac": args.sample_frac,
                "max_samples": args.max_samples,
                "seed": args.seed,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
