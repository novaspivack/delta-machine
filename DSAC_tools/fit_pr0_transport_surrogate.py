#!/usr/bin/env python3
"""Refit PR-0 transport surrogate coefficients from coarse-grained datasets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "pr0_transport"
DEFAULT_REGIMES = ("baseline", "high_drive", "asymmetric")
DEFAULT_BASIS = "reduced"
EPS = 1.0e-8


@dataclass
class SampleRecord:
    regime: str
    index: int
    X: np.ndarray  # (points, features)
    y: np.ndarray  # (points,)


def _load_regime(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    required = {"rho", "grad_rho_x", "grad_rho_y", "gamma", "rho_t"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Dataset {path} missing arrays: {sorted(missing)}")
    return {name: data[name] for name in required}


def _laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def _prepare_features(
    rho: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    gamma: np.ndarray,
    basis: str,
) -> tuple[np.ndarray, list[str]]:
    rho_centered = rho - rho.mean()
    grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_rms = np.sqrt(np.mean(grad_norm ** 2)) + EPS
    gamma_centered = gamma - gamma.mean()
    gamma_rms = np.sqrt(np.mean(gamma ** 2)) + EPS

    lap_rho = _laplacian(rho)
    lap_gamma = _laplacian(gamma)

    def add(name: str, field: np.ndarray, store: list[np.ndarray], labels: list[str]) -> None:
        store.append(field.reshape(-1))
        labels.append(name)

    features: list[np.ndarray] = []
    labels: list[str] = []

    add("rho_centered", rho_centered, features, labels)

    if basis in ("full", "experimental"):
        add("diff_grad_x", grad_x / grad_rms, features, labels)
        add("diff_grad_y", grad_y / grad_rms, features, labels)

    add("lap_rho", lap_rho, features, labels)
    add("gamma_centered", gamma_centered, features, labels)
    add("lap_gamma", lap_gamma, features, labels)
    add("diff_grad_norm", grad_norm / grad_rms, features, labels)
    add("gamma_normalized", gamma / gamma_rms, features, labels)

    if basis in ("full", "experimental"):
        add("rho_grad_coupling", rho_centered * grad_x, features, labels)
        add(
            "grad_gamma_coupling",
            (grad_x * gamma) / (grad_rms * gamma_rms),
            features,
            labels,
        )

    if basis == "experimental":
        add(
            "gamma_lap_rho",
            gamma_centered * lap_rho,
            features,
            labels,
        )
        add(
            "gamma_diff_grad",
            gamma_centered * (grad_norm / grad_rms),
            features,
            labels,
        )
        add(
            "lap_rho_norm",
            lap_rho / (np.sqrt(np.mean(lap_rho ** 2)) + EPS),
            features,
            labels,
        )

    add("bias", np.ones_like(rho), features, labels)
    X = np.stack(features, axis=1)
    return X, labels


def _collect_samples(
    dataset_dir: Path,
    regimes: Iterable[str],
    basis: str,
) -> tuple[list[SampleRecord], list[str]]:
    samples: list[SampleRecord] = []
    feature_names: list[str] | None = None
    for regime in regimes:
        path = dataset_dir / regime / "transport_dataset.npz"
        raw = _load_regime(path)
        count = raw["rho"].shape[0]
        for idx in range(count):
            rho = raw["rho"][idx].astype(np.float64, copy=False)
            grad_x = raw["grad_rho_x"][idx].astype(np.float64, copy=False)
            grad_y = raw["grad_rho_y"][idx].astype(np.float64, copy=False)
            gamma = raw["gamma"][idx].astype(np.float64, copy=False)
            rho_t = raw["rho_t"][idx].astype(np.float64, copy=False)

            X, labels = _prepare_features(rho, grad_x, grad_y, gamma, basis)
            if feature_names is None:
                feature_names = labels
            elif feature_names != labels:
                raise RuntimeError("Feature name mismatch across samples")

            y = (rho_t - rho_t.mean()).reshape(-1)
            samples.append(SampleRecord(regime=regime, index=idx, X=X, y=y))

    if feature_names is None:
        raise RuntimeError("No samples collected")
    return samples, feature_names


def _stack_design_matrix(samples: Sequence[SampleRecord]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([s.X for s in samples], axis=0)
    y = np.concatenate([s.y for s in samples], axis=0)
    return X, y


def fit_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    XT = X.T
    xtx = XT @ X
    if ridge > 0.0:
        xtx += ridge * np.eye(xtx.shape[0], dtype=xtx.dtype)
    xty = XT @ y
    coefs, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    predictions = X @ coefs
    residuals = y - predictions
    metrics = {
        "samples": int(X.shape[0]),
        "residual_rms": float(np.sqrt(np.mean(residuals ** 2))),
        "target_rms": float(np.sqrt(np.mean(y ** 2))),
        "mape": float(np.mean(np.abs(residuals) / (np.abs(y) + EPS))),
    }
    return coefs, metrics


def per_regime_metrics(
    samples: Sequence[SampleRecord],
    coefs: np.ndarray,
    tolerance: float = 2.0,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
    aggregates: dict[str, dict[str, float]] = {}
    scratch: dict[str, dict[str, float]] = {}
    sample_rows: list[dict[str, float]] = []

    for record in samples:
        predictions = record.X @ coefs
        residuals = record.y - predictions
        abs_residuals = np.abs(residuals)
        y_abs = np.abs(record.y)

        res_sq = float(np.sum(residuals ** 2))
        tgt_sq = float(np.sum(record.y ** 2))
        mape_sum = float(np.sum(abs_residuals / (y_abs + EPS)))
        n_points = float(residuals.shape[0])
        sample_rms = float(np.sqrt(res_sq / n_points))
        max_point_residual = float(np.max(abs_residuals))

        bucket = scratch.setdefault(
            record.regime,
            {
                "res_sq": 0.0,
                "tgt_sq": 0.0,
                "mape_sum": 0.0,
                "count": 0.0,
                "samples": 0.0,
                "max_sample_rms": 0.0,
                "max_point_residual": 0.0,
                "within_tol": 0.0,
            },
        )
        bucket["res_sq"] += res_sq
        bucket["tgt_sq"] += tgt_sq
        bucket["mape_sum"] += mape_sum
        bucket["count"] += n_points
        bucket["samples"] += 1.0
        bucket["max_sample_rms"] = max(bucket["max_sample_rms"], sample_rms)
        bucket["max_point_residual"] = max(bucket["max_point_residual"], max_point_residual)
        bucket["within_tol"] += 1.0 if sample_rms <= tolerance else 0.0

        sample_rows.append(
            {
                "regime": record.regime,
                "sample_index": record.index,
                "residual_rms": sample_rms,
                "target_rms": float(np.sqrt(tgt_sq / n_points)),
                "mape": float(mape_sum / n_points),
                "max_point_residual": max_point_residual,
                "points": n_points,
                "within_tol": 1.0 if sample_rms <= tolerance else 0.0,
            }
        )

    overall = {
        "res_sq": sum(bucket["res_sq"] for bucket in scratch.values()),
        "tgt_sq": sum(bucket["tgt_sq"] for bucket in scratch.values()),
        "mape_sum": sum(bucket["mape_sum"] for bucket in scratch.values()),
        "count": sum(bucket["count"] for bucket in scratch.values()),
        "samples": sum(bucket["samples"] for bucket in scratch.values()),
        "max_sample_rms": max(bucket["max_sample_rms"] for bucket in scratch.values()),
        "max_point_residual": max(bucket["max_point_residual"] for bucket in scratch.values()),
        "within_tol": sum(bucket["within_tol"] for bucket in scratch.values()),
    }
    scratch["ALL"] = overall

    for regime, bucket in scratch.items():
        if bucket["count"] == 0:
            continue
        aggregates[regime] = {
            "samples": bucket["samples"],
            "residual_rms": float(np.sqrt(bucket["res_sq"] / bucket["count"])),
            "target_rms": float(np.sqrt(bucket["tgt_sq"] / bucket["count"])),
            "mape": float(bucket["mape_sum"] / bucket["count"]),
            "max_sample_rms": float(bucket["max_sample_rms"]),
            "max_point_residual": float(bucket["max_point_residual"]),
            "verification_rate": float(bucket["within_tol"] / bucket["samples"]),
        }

    return aggregates, sample_rows


def write_sample_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit PR-0 transport surrogate")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--regimes", nargs="*", default=list(DEFAULT_REGIMES))
    parser.add_argument("--ridge", type=float, default=1.0e-9)
    parser.add_argument(
        "--basis",
        choices=("reduced", "full", "experimental"),
        default=DEFAULT_BASIS,
        help="Feature basis to use (matches scenario by default)",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file")
    parser.add_argument(
        "--sample-output",
        type=str,
        default=None,
        help="Optional CSV with per-sample residual metrics",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Verification tolerance for per-sample classification",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    samples, feature_names = _collect_samples(dataset_dir, args.regimes, args.basis)
    X, y = _stack_design_matrix(samples)

    coefs, metrics = fit_surrogate(X, y, ridge=args.ridge)
    regime_metrics, sample_rows = per_regime_metrics(samples, coefs, tolerance=args.tolerance)

    print("Fit summary")
    print("  samples:", metrics["samples"])
    print("  residual_rms:", metrics["residual_rms"])
    print("  target_rms:", metrics["target_rms"])
    print("  mape:", metrics["mape"])
    print()
    for label, coef in zip(feature_names, coefs):
        print(f"  {label:24s} {coef:+.6e}")

    print("\nPer-regime metrics:")
    for regime, info in regime_metrics.items():
        print(
            f"  {regime:10s} residual_rms={info['residual_rms']:.3e} "
            f"mape={info['mape']:.3e} max_sample_rms={info['max_sample_rms']:.3e} "
            f"max_point_residual={info['max_point_residual']:.3e} "
            f"verification_rate={info['verification_rate']:.2f} samples={info['samples']:.0f}"
        )

    if args.sample_output:
        write_sample_csv(sample_rows, Path(args.sample_output).resolve())

    if args.output:
        payload = {
            "metrics": metrics,
            "coefficients": {
                label: float(value) for label, value in zip(feature_names, coefs)
            },
            "regimes": list(args.regimes),
            "ridge": float(args.ridge),
            "basis": args.basis,
            "regime_metrics": regime_metrics,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
