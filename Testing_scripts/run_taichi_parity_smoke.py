#!/usr/bin/env python3
"""Run Taichi vs NumPy parity smoke tests for key scenarios."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = DEFAULT_PROJECT_ROOT / "scenarios"
DEFAULT_CASES = [
    ("discovery_phase2/curvature_invariant.yaml", 50, None),
    ("discovery_phase2/curvature_invariant_nofusion.yaml", 50, None),
    ("discovery_phase1/pr0_flux.yaml", 10, 1e-10),
    ("discovery_phase1/polynomial_law.yaml", 20, None),
    ("discovery_phase1/te1_jarzynski.yaml", 10, None),
    ("constraint_discovery.yaml", 20, None),
]


def _auto_tolerance(scenario_path: Path) -> float:
    import yaml
    from delta_machine.backends import resolve_backend_selection

    data = yaml.safe_load(scenario_path.read_text()) or {}
    metadata = data.get("metadata") or {}
    backend, cfg = resolve_backend_selection(metadata, "auto")
    if backend != "taichi":
        return 5e-8
    dtype = (cfg.get("dtype") or cfg.get("precision") or cfg.get("fp_precision") or "float64").lower()
    if dtype in {"float32", "f32", "32"}:
        return 1e-6
    if dtype in {"float16", "f16", "16"}:
        return 1e-5
    return 5e-8


def run_case(scenario: str, steps: int, tolerance: float | None, scenario_dir: Path) -> None:
    scenario_path = scenario_dir / scenario
    tol_value = tolerance if tolerance is not None else _auto_tolerance(scenario_path)
    cmd = [
        "python3",
        str(DEFAULT_PROJECT_ROOT / "DSAC_tools" / "compare_taichi_numpy_constraints.py"),
        "--scenario",
        scenario,
        "--scenario-dir",
        str(scenario_dir),
        "--steps",
        str(steps),
        "--tolerance",
        str(tol_value),
        "--fail-on-nan",
    ]
    print(f"[Parity] scenario={scenario} steps={steps} tol={tol_value}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Taichi backend parity smoke tests")
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    args = parser.parse_args()
    scenario_dir = Path(args.scenario_dir).resolve()

    for scenario, steps, tol in DEFAULT_CASES:
        run_case(scenario, steps, tol, scenario_dir)


if __name__ == "__main__":
    main()
