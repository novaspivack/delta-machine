"""Aggregate TPU-C reflexive advantage metrics for TE₂.U datasets.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2u_strong_transputational_universality/tpuc_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from delta_machine.scenarios.band_c.te2u_strong_transputational_universality import (
    compute_tpuc_advantage_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise TPU-C advantage metrics from DSAC logs")
    parser.add_argument("manifest", type=Path, help="Manifest JSON path produced by the benchmark builder")
    parser.add_argument("dsac_results", type=Path, help="NumPy .npz file with DSAC telemetry arrays")
    parser.add_argument("classical_results", type=Path, help="NumPy .npz file with classical baseline arrays")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON summary")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.array(data[key]) for key in data.files}


def load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return list(data.get("samples", []))
    if isinstance(data, list):
        return list(data)
    raise TypeError(f"Unsupported manifest data type: {type(data)!r}")


def main() -> None:
    args = parse_args()
    dsac = load_npz(args.dsac_results)
    classical = load_npz(args.classical_results)
    manifest = load_manifest(args.manifest)
    metrics = compute_tpuc_advantage_metrics(dsac, classical, manifest)

    summary = {
        "reflexive_speedup": metrics.reflexive_speedup,
        "accuracy_delta": metrics.accuracy_delta,
        "information_profit": metrics.information_profit,
        "holographic_capacity": metrics.holographic_capacity,
        "sigma_overall": metrics.sigma_overall,
        "classical_cost_mean": metrics.classical_cost_mean,
        "tpuc_energy_joules": metrics.tpuc_energy_joules,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    else:
        for key, value in summary.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
