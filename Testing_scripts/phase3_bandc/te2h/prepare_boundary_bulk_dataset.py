"""Prepare TE₂.H boundary↔bulk datasets for Band C experiments.

Usage:
    python prepare_boundary_bulk_dataset.py --output data/phase3_bandc/te2h_boundary_bulk

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2h_reflexive_holographic_equivalence/boundary_bulk_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from delta_machine.scenarios.band_c.te2h_reflexive_holographic_equivalence import (
    BoundaryBulkConfig,
    build_boundary_bulk_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TE₂.H boundary↔bulk datasets")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (e.g. data/phase3_bandc/te2h_boundary_bulk)",
    )
    parser.add_argument("--samples", type=int, default=16, help="Samples per topology (default: 16)")
    parser.add_argument("--boundary-depth", type=int, default=4, help="Boundary depth in lattice cells")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["torus", "cylinder", "open"],
        help="Topology list for dataset synthesis",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BoundaryBulkConfig(
        output_dir=args.output,
        num_samples=args.samples,
        boundary_depth=args.boundary_depth,
        topologies=tuple(args.topologies),
        random_seed=args.seed,
    )
    build_boundary_bulk_dataset(config)


if __name__ == "__main__":
    main()
