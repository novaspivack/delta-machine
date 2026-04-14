"""
Command line utilities for the Δ-Machine runtime.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/PARTICLE Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/PARTICLE Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import run_headless


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Δ-Machine CLI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    headless = subparsers.add_parser("headless", help="Run a scenario headlessly")
    headless.add_argument("--scenario", required=True, help="Scenario file name (YAML)")
    headless.add_argument("--steps", type=int, required=True, help="Number of steps to execute")
    headless.add_argument(
        "--scenario-dir",
        type=str,
        default=str(DEFAULT_SCENARIO_DIR),
        help="Directory containing scenario definitions",
    )
    headless.add_argument(
        "--use-pr0-field-state",
        action="store_true",
        help="Enable PR-0 field state integration (requires pr0_system to be installed)",
    )
    headless.add_argument(
        "--workers",
        type=int,
        help="Number of orchestrator worker processes (overrides scenario metadata and default)",
    )
    headless.add_argument(
        "--lattice-boundary",
        type=str,
        choices=["torus", "open", "cylinder_x", "cylinder_y"],
        help="Override lattice boundary when using PR-0 integration",
    )
    headless.add_argument(
        "--lattice-type",
        type=str,
        help="Override lattice type (e.g., square, hexagonal) for PR-0 integration",
    )
    headless.add_argument(
        "--backend",
        type=str,
        choices=["auto", "numpy", "taichi"],
        default="auto",
        help=(
            "Constraint backend to use for DSAC runs: "
            "'auto' (use scenario metadata), 'numpy' (worker path), or 'taichi' (high-speed backend)"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()

    if args.command == "headless":
        run_headless(
            scenario_dir,
            args.scenario,
            args.steps,
            use_pr0_field_state=args.use_pr0_field_state,
            max_workers=args.workers,
            lattice_boundary=args.lattice_boundary,
            lattice_type=args.lattice_type,
            backend=args.backend,
        )
    else:
        parser.error(f"Unsupported command '{args.command}'")


if __name__ == "__main__":
    main()

