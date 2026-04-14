"""Prepare TE₂.U TPU-C benchmark datasets for Band C experiments.

Usage:
    python prepare_tpuc_benchmark_suite.py --output data/phase3_bandc/te2u_tpuc_benchmark

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2u_strong_transputational_universality/tpuc_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from delta_machine.scenarios.band_c.te2u_strong_transputational_universality import (
    TPUCScenarioConfig,
    build_tpuc_benchmark_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TE₂.U TPU-C benchmark suite")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (e.g. data/phase3_bandc/te2u_tpuc_benchmark)",
    )
    parser.add_argument("--matrix-size", type=int, default=48, help="Matrix dimension for workloads")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["reflexive_linear_system", "holographic_quadratic", "reflexive_sat"],
        help="Task families to include",
    )
    parser.add_argument("--samples", type=int, default=24, help="Tasks per family")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TPUCScenarioConfig(
        output_dir=args.output,
        matrix_size=args.matrix_size,
        task_families=tuple(args.families),
        tasks_per_family=args.samples,
        random_seed=args.seed,
    )
    build_tpuc_benchmark_suite(config)


if __name__ == "__main__":
    main()
