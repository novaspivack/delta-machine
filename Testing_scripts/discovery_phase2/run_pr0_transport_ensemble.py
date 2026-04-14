#!/usr/bin/env python3
"""Run Phase II PR-0 transport discovery ensembles across regimes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "pr0_transport"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "discovery_phase2" / "pr0_transport_ensemble.csv"
DEFAULT_SCENARIO = "discovery_phase2/pr0_transport.yaml"
DEFAULT_LONGRUN_SCENARIO = "discovery_phase2/pr0_transport_longrun.yaml"
DEFAULT_BACKEND = "taichi"
DEFAULT_STEPS = 1
DEFAULT_LONGRUN_STEPS = 120
DEFAULT_SAMPLES = 24
DEFAULT_WORKERS = 4
DEFAULT_REGIMES = ("baseline", "high_drive", "asymmetric")
DEFAULT_ENS_OUTPUT_DIR = PROJECT_ROOT / "runs" / "discovery_phase2"


def discover_dataset_counts(dataset_dir: Path, regimes: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for regime in regimes:
        path = dataset_dir / regime / "transport_dataset.npz"
        if not path.exists():
            raise FileNotFoundError(f"Dataset for regime '{regime}' not found at {path}")
        data = np.load(path)
        counts[regime] = data["rho"].shape[0]
    return counts


def compress_records(records: list[Dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summary_statistics(records: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if not records:
        return stats
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["regime"], []).append(row)
    grouped["ALL"] = records
    for regime, rows in grouped.items():
        residuals = [float(r["residual_rms"]) for r in rows if r.get("residual_rms")]
        mapes = [float(r["transport_mape"]) for r in rows if r.get("transport_mape")]
        trimmed_mapes = [float(r["transport_mape_trimmed"]) for r in rows if r.get("transport_mape_trimmed")]
        flux_residuals = [float(r["transport_flux_divergence_rms"]) for r in rows if r.get("transport_flux_divergence_rms")]
        verified = [float(r["discovery_verified"]) for r in rows if r.get("discovery_verified")]
        stats[regime] = {
            "samples": float(len(rows)),
            "mean_residual": float(np.mean(residuals)) if residuals else float("nan"),
            "mean_mape": float(np.mean(mapes)) if mapes else float("nan"),
            "mean_mape_trimmed": float(np.mean(trimmed_mapes)) if trimmed_mapes else float("nan"),
            "mean_flux_divergence": float(np.mean(flux_residuals)) if flux_residuals else float("nan"),
            "verification_rate": float(np.mean(verified)) if verified else 0.0,
        }
    return stats


def coefficient_summary(records: list[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    buckets: Dict[str, Dict[str, list[float]]] = {}
    for row in records:
        coeffs_raw = row.get("coefficients")
        names_raw = row.get("feature_names")
        if not coeffs_raw or not names_raw:
            continue
        names = json.loads(names_raw)
        coeffs = json.loads(coeffs_raw)
        if not names or not coeffs:
            continue
        regime = row["regime"]
        regime_bucket = buckets.setdefault(regime, {})
        for name, value in zip(names, coeffs):
            regime_bucket.setdefault(name, []).append(float(value))
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for regime, features in buckets.items():
        feature_stats: Dict[str, Dict[str, float]] = {}
        for name, values in features.items():
            arr = np.asarray(values, dtype=float)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            feature_stats[name] = {"median": median, "mad": mad}
        summary[regime] = feature_stats
    return summary


def run_single(
    scenario: str,
    scenario_dir: Path,
    steps: int,
    backend: str,
    workers: int,
    regime: str,
    sample_index: int,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT.parent}"
    env["PR0_TRANSPORT_REGIME"] = regime
    env["PR0_TRANSPORT_SAMPLE_INDEX"] = str(sample_index)

    cmd = [
        "python3",
        "-m",
        "delta_machine.cli",
        "headless",
        "--scenario",
        scenario,
        "--scenario-dir",
        str(scenario_dir),
        "--steps",
        str(steps),
        "--backend",
        backend,
        "--workers",
        str(workers),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Run failed for regime={regime} sample={sample_index}:\n{result.stderr.strip()}"
        )

    run_dir = None
    for line in result.stdout.splitlines():
        if line.startswith("Δ-Machine run artifacts:"):
            run_dir = line.split(":", 1)[1].strip()
            break
    if run_dir is None:
        raise RuntimeError("Failed to locate run directory in CLI output")

    report = Path(run_dir) / "report.yaml"
    data = yaml.safe_load(report.read_text())
    metrics = data.get("metrics", {})
    summary = data.get("run_summary", {})
    feature_names = metrics.get("feature_names")
    coefficients = metrics.get("discovered_coefficients")

    if feature_names:
        feature_names = [name for name in list(feature_names) if name is not None]
    if feature_names and coefficients:
        paired = []
        seen = set()
        for name, coef in zip(feature_names, coefficients):
            if name is None:
                continue
            key = name
            if key == "bias" and key in seen:
                continue
            seen.add(key)
            paired.append((name, coef))
        if paired:
            feature_names = [name for name, _ in paired]
            coefficients = [coef for _, coef in paired]

    row = {
        "regime": regime,
        "sample_index": sample_index,
        "run_dir": run_dir,
        "steps": summary.get("final_step"),
        "discovery_verified": metrics.get("discovery_verified"),
        "discovery_streak": metrics.get("discovery_streak"),
        "residual_rms": metrics.get("residual_rms"),
        "transport_mape": metrics.get("transport_mape"),
        "transport_mape_trimmed": metrics.get("transport_mape_trimmed"),
        "transport_target_rms": metrics.get("transport_target_rms"),
        "transport_mean_abs_target": metrics.get("transport_mean_abs_target"),
        "transport_predicted_rms": metrics.get("transport_predicted_rms"),
        "transport_residual_rms": metrics.get("transport_residual_rms"),
        "transport_flux_divergence_rms": metrics.get("transport_flux_divergence_rms"),
        "transport_flux_divergence_actual_rms": metrics.get("transport_flux_divergence_actual_rms"),
        "transport_flux_divergence_predicted_rms": metrics.get("transport_flux_divergence_predicted_rms"),
        "coefficients": json.dumps(coefficients),
        "feature_names": json.dumps(feature_names),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase II PR-0 transport ensemble runner")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--longrun-scenario", default=DEFAULT_LONGRUN_SCENARIO)
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--longrun", action="store_true", help="Use the longrun scenario (120 steps)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=list(DEFAULT_REGIMES),
        help="Subset of regimes to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for sample ordering",
    )
    parser.add_argument(
        "--per-regime-output",
        type=str,
        default=None,
        help="Directory to write per-regime CSV summaries",
    )
    parser.add_argument(
        "--coeff-summary-output",
        type=str,
        default=None,
        help="Optional JSON file capturing coefficient medians and MADs",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_path = Path(args.output).resolve()

    per_regime_dir = (
        Path(args.per_regime_output).resolve()
        if args.per_regime_output
        else DEFAULT_ENS_OUTPUT_DIR / "pr0_transport_regime_breakdown"
    )

    if args.seed is not None:
        random.seed(args.seed)

    scenario_name = args.longrun_scenario if args.longrun else args.scenario
    steps = args.steps
    if args.longrun and (args.steps == DEFAULT_STEPS):
        steps = DEFAULT_LONGRUN_STEPS

    regimes = args.regimes
    counts = discover_dataset_counts(dataset_dir, regimes)

    print(f"[PR0-Transport] Scenario '{scenario_name}' steps={steps} backend={args.backend}")

    rows: list[Dict[str, Any]] = []
    for regime in regimes:
        regime_count = counts[regime]
        indices = list(range(regime_count))
        random.shuffle(indices)
        selected = indices[: args.samples]
        print(f"[PR0-Transport] Regime '{regime}' → sampling {len(selected)} / {regime_count}")
        for sample_idx in selected:
            row = run_single(
                scenario=scenario_name,
                scenario_dir=scenario_dir,
                steps=steps,
                backend=args.backend,
                workers=args.workers,
                regime=regime,
                sample_index=sample_idx,
            )
            rows.append(row)
            status = (
                f"  -> sample={sample_idx:03d} residual={row['residual_rms']:.4e} "
                f"mape={row['transport_mape']:.3e} verified={row['discovery_verified']}"
            )
            print(status)

    stats = summary_statistics(rows)
    coeff_stats = coefficient_summary(rows)

    compress_records(rows, output_path)
    print("Ensemble summary:")
    print(f"  rows={len(rows)} output={output_path}")
    for regime, info in stats.items():
        print(
            f"  {regime}: n={info['samples']:.0f} mean_residual={info['mean_residual']:.3e} "
            f"mean_mape={info['mean_mape']:.3e} mean_mape_trimmed={info['mean_mape_trimmed']:.3e} "
            f"mean_flux_div={info['mean_flux_divergence']:.3e} verification_rate={info['verification_rate']:.2f}"
        )

    # Write per-regime CSVs when requested
    per_regime_dir.mkdir(parents=True, exist_ok=True)
    for regime in regimes:
        subset = [row for row in rows if row['regime'] == regime]
        compress_records(subset, per_regime_dir / f"{regime}_ensemble.csv")
    compress_records(rows, per_regime_dir / "ALL_ensemble.csv")

    if args.coeff_summary_output:
        coeff_path = Path(args.coeff_summary_output).resolve()
        coeff_path.parent.mkdir(parents=True, exist_ok=True)
        coeff_path.write_text(json.dumps(coeff_stats, indent=2))
        print(f"Coefficient summary written to {coeff_path}")
    elif coeff_stats:
        print("Coefficient medians / MADs:")
        for regime, feature_stats in coeff_stats.items():
            print(f"  {regime}:")
            for name, stats in feature_stats.items():
                print(
                    f"    {name:20s} median={stats['median']:+.4e} mad={stats['mad']:.4e}"
                )


if __name__ == "__main__":
    main()
