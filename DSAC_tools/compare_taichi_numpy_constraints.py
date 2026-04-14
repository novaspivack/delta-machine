#!/usr/bin/env python3
"""Compare NumPy vs Taichi constraint backends for metric-closure constraints.

This script is a debugging/parity tool for the Taichi backend design in
1.30_DSAC_Optmization_Strategies.md. It:

- Loads a discovery/metric-closure scenario (e.g. curvature_invariant.yaml)
- Initializes the fields via the normal DeltaOrchestrator path
- Copies the initial state into two buffers
- Applies a small number of constraint steps using:
  - NumpyConstraintBackend
  - TaichiConstraintBackend
- Reports max abs differences per field after the steps

Usage (from project root):

  PYTHONPATH="src:.." python3 DSAC_tools/compare_taichi_numpy_constraints.py \
      --scenario discovery_phase2/curvature_invariant.yaml \
      --scenario-dir scenarios \
      --steps 1

Design references:
- 1.30 DSAC Optimization Strategies Lab Note
- 1.26 Phase II Curvature Invariant Results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from delta_machine.backends import NumpyConstraintBackend, TaichiConstraintBackend
from delta_machine.backends import resolve_backend_selection
from delta_machine.config import ScenarioLoader
from delta_machine.functionals import FunctionalCompiler
from delta_machine.orchestrator import DeltaOrchestrator


def _copy_arrays(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep-copy lattice fields (by value) from a state buffer dict."""
    return {name: np.array(val, copy=True) for name, val in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NumPy vs Taichi constraint backends")
    parser.add_argument("--scenario", required=True, help="Scenario YAML path relative to scenario-dir")
    parser.add_argument("--scenario-dir", default="scenarios", help="Scenario directory")
    parser.add_argument("--steps", type=int, default=1, help="Number of constraint steps to compare")
    parser.add_argument("--tolerance", type=float, default=None, help="Fail if max |Δ| exceeds this value")
    parser.add_argument(
        "--fail-on-nan",
        action="store_true",
        help="Fail if any NaN is detected in NumPy or Taichi buffers",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scenario_dir = (project_root / args.scenario_dir).resolve()

    loader = ScenarioLoader(scenario_dir)
    scenario = loader.load(args.scenario)

    compiler = FunctionalCompiler()
    run_base_dir = project_root / "runs"

    # Initialize an orchestrator to reuse its constraint graph and initial fields
    orchestrator = DeltaOrchestrator(
        scenario,
        compiler,
        max_workers=1,
        run_base_dir=run_base_dir,
        backend="numpy",
    )
    orchestrator.initialize()

    if orchestrator.constraint_graph is None or orchestrator.shared_state is None:
        raise RuntimeError("Orchestrator did not initialize constraint graph or shared state")

    arrays_initial = orchestrator.shared_state.arrays()
    arrays_initial_copy = _copy_arrays(arrays_initial)
    arrays_np = _copy_arrays(arrays_initial)
    arrays_ti = _copy_arrays(arrays_initial)

    numpy_backend = NumpyConstraintBackend(orchestrator.constraint_graph)
    _, taichi_config = resolve_backend_selection(scenario.metadata, "auto")
    numpy_rate = float(taichi_config.get("relaxation_rate", 0.01))
    numpy_backend.relaxation_rate = numpy_rate
    taichi_backend = TaichiConstraintBackend(orchestrator.constraint_graph, taichi_config)

    residuals_initial = orchestrator.constraint_graph.evaluate(arrays_initial_copy)

    steps = max(1, int(args.steps))
    for _ in range(steps):
        numpy_backend.step(arrays_np)
        taichi_backend.step(arrays_ti)

    fields_to_check = sorted(set(arrays_np.keys()).intersection(arrays_ti.keys()))
    max_diff_overall = 0.0
    nan_detected = False
    print(f"Compared {steps} constraint step(s) for scenario '{args.scenario}'")
    for name in fields_to_check:
        a_np = arrays_np.get(name)
        a_ti = arrays_ti.get(name)
        if a_np is None or a_ti is None:
            print(f"  {name:8s}: MISSING in one of the buffers")
            continue
        diff_arr = np.abs(a_np - a_ti)
        max_diff = np.nanmax(diff_arr)
        idx = np.unravel_index(np.nanargmax(diff_arr), diff_arr.shape)
        max_diff_overall = max(max_diff_overall, float(max_diff))
        nan_detected = nan_detected or np.isnan(a_np).any() or np.isnan(a_ti).any()
        print(
            f"  {name:8s}: max |Δ| = {max_diff:.6e} at {idx}, "
            f"NumPy={a_np[idx]:.6e}, Taichi={a_ti[idx]:.6e}"
        )

    # Inspect the location of the largest psi_imag discrepancy
    psi_idx = np.unravel_index(
        np.nanargmax(np.abs(arrays_np["psi_imag"] - arrays_ti["psi_imag"])),
        arrays_np["psi_imag"].shape,
    )
    print("\nDetailed values at psi_imag max-delta index", psi_idx)
    for name in fields_to_check:
        print(
            f"  {name:8s}: initial={arrays_initial_copy[name][psi_idx]:.6e}, "
            f"NumPy={arrays_np[name][psi_idx]:.6e}, Taichi={arrays_ti[name][psi_idx]:.6e}"
        )
    print("Residuals at that index (initial state):")
    for node_name, residual in residuals_initial.items():
        node = orchestrator.constraint_graph.nodes[node_name]
        target = node.target_field
        print(f"  {node_name:20s} (target={target}): residual={residual[psi_idx]:.6e}")

    if args.tolerance is not None and max_diff_overall > args.tolerance:
        print(
            f"\nERROR: max |Δ| ({max_diff_overall:.6e}) exceeds tolerance {args.tolerance:.6e}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.fail_on_nan and nan_detected:
        print("\nERROR: NaN detected in NumPy or Taichi buffers", file=sys.stderr)
        sys.exit(1)

    print("\nSummary:")
    print(f"  Max |Δ| across fields: {max_diff_overall:.6e}")
    print(f"  NaN detected: {'yes' if nan_detected else 'no'}")
    if args.tolerance is not None:
        print(f"  Tolerance check: passed (<= {args.tolerance:.2e})")
    if args.fail_on_nan:
        print("  NaN check: passed")


if __name__ == "__main__":
    main()
