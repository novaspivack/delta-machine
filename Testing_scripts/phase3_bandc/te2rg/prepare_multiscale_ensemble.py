"""Prepare TE₂.RG multiscale PR-0 ensembles for Band C experiments.

Usage:
    python prepare_multiscale_ensemble.py --output data/phase3_bandc/te2rg_multiscale

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2rg_reflexive_rg_flow/multiscale_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from delta_machine.scenarios.band_c.te2rg_reflexive_rg_flow import (
    MultiscaleEnsembleConfig,
    generate_multiscale_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TE₂.RG multiscale ensembles")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (e.g. data/phase3_bandc/te2rg_multiscale)",
    )
    parser.add_argument("--base-size", type=int, default=128, help="Base lattice dimension")
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4, 8], help="Scales to generate")
    parser.add_argument("--samples", type=int, default=12, help="Samples per scale")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--no-flux", action="store_true", help="Disable flux divergence exports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiscaleEnsembleConfig(
        output_dir=args.output,
        base_shape=(args.base_size, args.base_size),
        scales=tuple(args.scales),
        samples_per_scale=args.samples,
        random_seed=args.seed,
        include_flux=not args.no_flux,
    )
    generate_multiscale_ensemble(config)


if __name__ == "__main__":
    main()
