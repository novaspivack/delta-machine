#!/usr/bin/env python3
"""Run PR-0 flux rediscovery ensemble and summarize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import yaml

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = DEFAULT_PROJECT_ROOT / "scenarios"
DEFAULT_OUTPUT = DEFAULT_PROJECT_ROOT / "runs" / "discovery_phase1" / "pr0_flux_ensemble.csv"
DEFAULT_STEPS = 6000
DEFAULT_RUNS = 10
DEFAULT_WORKERS = 9


def run_single(seed: int, scenario: str, scenario_dir: Path, steps: int, workers: int) -> Dict[str, Any]:
    env = {
        **dict(**os.environ),
        "PYTHONPATH": f"{DEFAULT_PROJECT_ROOT / 'src'}:{DEFAULT_PROJECT_ROOT / '..'}",
    }
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
        "--workers",
        str(workers),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Run failed (seed={seed}): {result.stderr}")

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
    return {
        "seed": seed,
        "run_dir": run_dir,
        "steps": data.get("run_summary", {}).get("final_step"),
        "discovery_verified": metrics.get("discovery_verified"),
        "discovery_streak": metrics.get("discovery_streak"),
        "residual_rms": metrics.get("residual_rms"),
        "coefficient_error": metrics.get("coefficient_error"),
        "max_coefficient_error": metrics.get("max_coefficient_error"),
        "coefficients": json.dumps(metrics.get("discovered_coefficients")),
        "feature_names": json.dumps(metrics.get("feature_names")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase I PR-0 flux rediscovery ensemble runner")
    parser.add_argument("--scenario", default="discovery_phase1/pr0_flux.yaml")
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(1, args.runs + 1):
        print(f"[PR0-Ensemble] Seed {seed}/{args.runs}")
        row = run_single(seed, args.scenario, scenario_dir, args.steps, args.workers)
        rows.append(row)
        print(
            f"  -> verified={row['discovery_verified']} streak={row['discovery_streak']} "
            f"residual={row['residual_rms']:.4e} max_coeff_err={row['max_coefficient_error']:.4e}"
        )

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"PR-0 ensemble summary written to {output_path}")


if __name__ == "__main__":
    import os

    main()
