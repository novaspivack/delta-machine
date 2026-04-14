#!/usr/bin/env python3
"""Run the short discovery benchmark suite across multiple scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios"
DEFAULT_JOBS = [
    {
        "name": "curvature_autofusion",
        "scenario": "discovery_phase2/curvature_invariant.yaml",
        "output": PROJECT_ROOT
        / "runs"
        / "discovery_phase2"
        / "benchmark_curvature_autofusion_5x8k_simd.csv",
        "runs": 5,
        "steps": 8000,
        "workers": 1,
        "backend": "taichi",
    },
    {
        "name": "curvature_pernode",
        "scenario": "discovery_phase2/curvature_invariant_nofusion.yaml",
        "output": PROJECT_ROOT
        / "runs"
        / "discovery_phase2"
        / "benchmark_curvature_pernode_5x8k_simd.csv",
        "runs": 5,
        "steps": 8000,
        "workers": 1,
        "backend": "taichi",
    },
    {
        "name": "constraint_discovery",
        "scenario": "constraint_discovery.yaml",
        "output": PROJECT_ROOT
        / "runs"
        / "discovery_phase1"
        / "benchmark_constraint_discovery_5x4k_simd.csv",
        "runs": 5,
        "steps": 4000,
        "workers": 1,
        "backend": "taichi",
    },
]


def run_single(
    seed: int,
    scenario: str,
    scenario_dir: Path,
    steps: int,
    workers: int,
    backend: str,
) -> Dict[str, Any]:
    paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT.parent)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    python_path = os.pathsep.join(paths)
    env = {
        **dict(os.environ),
        "PYTHONPATH": python_path,
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
        "--backend",
        backend,
    ]
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"Run failed (seed={seed}): {result.stderr}")

    run_dir = None
    resolved_backend = backend
    for line in result.stdout.splitlines():
        if line.startswith("[Δ-Machine] Using constraint backend:"):
            resolved_backend = line.split(":", 1)[1].strip()
        if line.startswith("Δ-Machine run artifacts:"):
            run_dir = line.split(":", 1)[1].strip()
    if run_dir is None:
        raise RuntimeError("Failed to locate run directory in CLI output")

    report = Path(run_dir) / "report.yaml"
    data = yaml.safe_load(report.read_text())
    metrics = data.get("metrics", {})
    return {
        "seed": seed,
        "backend": resolved_backend,
        "run_dir": run_dir,
        "steps": data.get("run_summary", {}).get("final_step"),
        "runtime_seconds": duration,
        "discovery_verified": metrics.get("discovery_verified"),
        "discovery_streak": metrics.get("discovery_streak"),
        "residual_rms": metrics.get("residual_rms"),
        "invariant_residual_variance": metrics.get("invariant_residual_variance"),
        "coefficient_error": metrics.get("coefficient_error"),
        "max_coefficient_error": metrics.get("max_coefficient_error"),
        "coefficients": json.dumps(metrics.get("discovered_coefficients")),
        "feature_names": json.dumps(metrics.get("feature_names")),
    }


def run_job(job: Dict[str, Any], scenario_dir: Path) -> Path:
    name = job["name"]
    scenario = job["scenario"]
    steps = int(job.get("steps", 8000))
    runs = int(job.get("runs", 5))
    workers = int(job.get("workers", 1))
    backend = job.get("backend", "taichi")
    output_path = Path(job["output"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for seed in range(1, runs + 1):
        print(f"[{name}] Seed {seed}/{runs}")
        row = run_single(seed, scenario, scenario_dir, steps, workers, backend)
        rows.append(row)
        residual = row.get("residual_rms")
        residual_str = f"{float(residual):.4e}" if residual not in (None, "", "null") else "n/a"
        variance = row.get("invariant_residual_variance")
        variance_str = f"{float(variance):.4e}" if variance not in (None, "", "null") else "n/a"
        print(
            f"  -> verified={row['discovery_verified']} streak={row['discovery_streak']} "
            f"residual={residual_str} var={variance_str} runtime={row['runtime_seconds']:.2f}s"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{name}] Summary written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short discovery benchmark suite")
    parser.add_argument(
        "--scenario-dir",
        default=str(DEFAULT_SCENARIO_DIR),
        help="Base directory containing scenario YAML files.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Optional subset of job names to run (default: all).",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    job_filter = set(args.include) if args.include else None

    executed_outputs: List[Path] = []
    for job in DEFAULT_JOBS:
        if job_filter and job["name"] not in job_filter:
            continue
        executed_outputs.append(run_job(job, scenario_dir))

    if executed_outputs:
        print("Completed jobs:")
        for output in executed_outputs:
            print(f"  - {output}")
    else:
        print("No jobs executed (check --include filter).")


if __name__ == "__main__":
    main()
