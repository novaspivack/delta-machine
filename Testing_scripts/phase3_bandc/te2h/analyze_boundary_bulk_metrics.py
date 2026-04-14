"""Analyse TE₂.H boundary↔bulk dataset metrics.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2h_reflexive_holographic_equivalence/boundary_bulk_instrumentation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from delta_machine.scenarios.band_c.te2h_reflexive_holographic_equivalence import (
    BoundaryBulkMetrics,
    compute_boundary_bulk_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TE₂.H boundary↔bulk metrics")
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory produced by prepare_boundary_bulk_dataset.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store aggregated metrics",
    )
    return parser.parse_args()


def load_samples(dataset_dir: Path):
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle).get("samples", [])
    if not manifest:
        raise ValueError(f"Manifest has no samples: {manifest_path}")

    for entry in manifest:
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute():
            sample_path = (dataset_dir / sample_path).resolve()
        data = np.load(sample_path)
        yield entry, data


def main() -> None:
    args = parse_args()
    metrics_list: list[BoundaryBulkMetrics] = []
    for entry, data in load_samples(args.dataset_dir):
        metrics = compute_boundary_bulk_metrics(
            boundary_projection=data["boundary_projection"],
            bulk_field=data["bulk_field"],
            boundary_mask=data["boundary_mask"],
        )
        metrics_list.append(metrics)

    summary = {
        "count": len(metrics_list),
        "boundary_energy_mean": float(np.mean([m.boundary_energy for m in metrics_list])),
        "bulk_energy_mean": float(np.mean([m.bulk_energy for m in metrics_list])),
        "compression_ratio_mean": float(np.mean([m.compression_ratio for m in metrics_list])),
        "mutual_information_mean": float(np.mean([m.mutual_information for m in metrics_list])),
        "psc_efficiency_mean": float(np.mean([m.psc_efficiency for m in metrics_list])),
        "edge_alignment_mean": float(np.mean([m.edge_alignment for m in metrics_list])),
        "dimensional_lift_signal_mean": float(
            np.mean([m.dimensional_lift_signal for m in metrics_list])
        ),
        "spectral_overlap_mean": float(np.mean([m.spectral_overlap for m in metrics_list])),
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
