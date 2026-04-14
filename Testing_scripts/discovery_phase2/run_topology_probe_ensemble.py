#!/usr/bin/env python3
"""Run short curvature-invariant ensembles across lattice topologies.

Backend design and topology plan documented in:
- DSAC Optimization Strategies Lab Note:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.30_DSAC_Optmization_Strategies.md
- Topology Selection Material for MFRR:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.27_topology_selection_material_for_mfrr.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = DEFAULT_PROJECT_ROOT / "scenarios"
DEFAULT_SCENARIO = "discovery_phase2/curvature_invariant.yaml"
DEFAULT_OUTPUT = (
    DEFAULT_PROJECT_ROOT
    / "runs"
    / "discovery_phase2"
    / "curvature_invariant_topology_probe.csv"
)
DEFAULT_STEPS = 8000
DEFAULT_RUNS = 5
DEFAULT_WORKERS = 9
DEFAULT_BOUNDARIES: List[str] = ["torus", "open", "cylinder_x"]


def run_single(
    seed: int,
    scenario: str,
    scenario_dir: Path,
    steps: int,
    workers: int,
    boundary: str,
) -> Dict[str, Any]:
    from os import environ

    env = {
        **dict(**environ),
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
    # Only pass lattice boundary if explicitly set
    if boundary:
        cmd.extend(["--lattice-boundary", boundary])

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Run failed (seed={seed}, boundary={boundary}): {result.stderr}")

    run_dir: str | None = None
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
        "boundary": boundary,
        "seed": seed,
        "run_dir": run_dir,
        "steps": data.get("run_summary", {}).get("final_step"),
        "discovery_verified": metrics.get("discovery_verified"),
        "discovery_streak": metrics.get("discovery_streak"),
        "residual_rms": metrics.get("residual_rms"),
        "invariant_residual_variance": metrics.get("invariant_residual_variance"),
        "coefficient_error": metrics.get("coefficient_error"),
        "max_coefficient_error": metrics.get("max_coefficient_error"),
        "coefficients": json.dumps(metrics.get("discovered_coefficients")),
        "feature_names": json.dumps(metrics.get("feature_names")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvature-invariant topology probe ensemble runner")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--boundaries",
        nargs="*",
        default=DEFAULT_BOUNDARIES,
        help="List of lattice boundaries to probe (e.g. torus open cylinder_x)",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for boundary in args.boundaries:
        print(f"[Topology-Probe] Boundary={boundary}")
        for seed in range(1, args.runs + 1):
            print(f"  Seed {seed}/{args.runs}")
            row = run_single(seed, args.scenario, scenario_dir, args.steps, args.workers, boundary)
            rows.append(row)
            variance = row.get("invariant_residual_variance")
            variance_str = f"{float(variance):.4e}" if variance not in (None, "", "null") else "n/a"
            residual = row.get("residual_rms")
            residual_str = f"{float(residual):.4e}" if residual not in (None, "", "null") else "n/a"
            print(
                f"    -> verified={row['discovery_verified']} "
                f"streak={row['discovery_streak']} "
                f"residual={residual_str} var={variance_str}"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Topology probe ensemble summary written to {output_path}")


if __name__ == "__main__":
    main()
