"""Estimate β-functions for TE₂.RG multiscale ensembles.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2rg_reflexive_rg_flow/multiscale_beta_estimator.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from delta_machine.scenarios.band_c.te2rg_reflexive_rg_flow import estimate_beta_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate β-function from DSAC multiscale outputs")
    parser.add_argument(
        "measurements",
        type=Path,
        help="JSON or NPZ file containing scale measurements with fields (scale, coupling, flow)",
    )
    parser.add_argument("--format", choices=["json", "npz"], default="json", help="Input format")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_measurements(path: Path, fmt: str) -> List[dict[str, float]]:
    if fmt == "json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    data = np.load(path)
    scales = data["scale"].astype(float)
    coupling = data["coupling"].astype(float)
    flow = data["flow"].astype(float)
    return [
        {"scale": float(s), "coupling": float(c), "flow": float(f)}
        for s, c, f in zip(scales, coupling, flow)
    ]


def main() -> None:
    args = parse_args()
    measurements = load_measurements(args.measurements, args.format)
    estimates = list(estimate_beta_function(measurements))
    summary = [
        {
            "scale": est.scale,
            "coupling": est.coupling,
            "beta": est.beta_value,
            "variance": est.variance,
            "samples": est.sample_count,
        }
        for est in estimates
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    else:
        for row in summary:
            print(
                f"scale={row['scale']:.3f} coupling={row['coupling']:.4f} "
                f"beta={row['beta']:.6f} variance={row['variance']:.6f} samples={row['samples']}"
            )


if __name__ == "__main__":
    main()
