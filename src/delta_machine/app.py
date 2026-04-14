"""
Application entrypoint for the Δ-Machine runtime and GUI.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import numpy as np
from PySide6 import QtWidgets

from .config import ScenarioLoader
from .backends import resolve_backend_selection
from .functionals import FunctionalCompiler
from .gui import DeltaMachineWindow
from .initial_conditions import load_initial_condition
from .orchestrator import DeltaOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Δ-Machine runtime")
    parser.add_argument("--scenario", type=str, help="Scenario file name")
    parser.add_argument("--steps", type=int, default=0, help="Headless run steps (0 for GUI)")
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=str(DEFAULT_SCENARIO_DIR),
        help="Scenario directory",
    )
    parser.add_argument(
        "--use-pr0-field-state",
        action="store_true",
        help="Enable PR-0 field state integration (requires pr0_system)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of orchestrator worker processes (overrides scenario metadata and default)",
    )
    parser.add_argument(
        "--lattice-boundary",
        type=str,
        choices=["torus", "open", "cylinder_x", "cylinder_y"],
        help="Override lattice boundary when using PR-0",
    )
    parser.add_argument(
        "--lattice-type",
        type=str,
        help="Override lattice type (e.g., square, hexagonal) for PR-0 runs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "numpy", "taichi"],
        default="auto",
        help=(
            "Constraint backend to use for DSAC runs: "
            "'auto' (use scenario metadata), 'numpy' (worker path), or 'taichi' (high-speed backend)"
        ),
    )
    return parser.parse_args()


def run_gui(scenario_dir: Path):
    app = QtWidgets.QApplication([])
    window = DeltaMachineWindow(scenario_dir)
    window.resize(1200, 800)
    window.show()
    app.exec()


def run_headless(
    scenario_dir: Path,
    scenario_name: str,
    steps: int,
    *,
    use_pr0_field_state: bool = False,
    max_workers: int | None = None,
    lattice_boundary: str | None = None,
    lattice_type: str | None = None,
    backend: str = "auto",
):
    previous_flag = os.environ.get("DELTA_USE_PR0_FIELDSTATE")
    if use_pr0_field_state:
        os.environ["DELTA_USE_PR0_FIELDSTATE"] = "1"
    loader = ScenarioLoader(scenario_dir)
    scenario = loader.load(scenario_name)
    if use_pr0_field_state:
        scenario.metadata = dict(scenario.metadata)
        scenario.metadata["use_pr0_field_state"] = True
    if lattice_type or lattice_boundary:
        scenario.metadata = dict(scenario.metadata)
        lattice_cfg = dict(scenario.metadata.get("lattice_config", {}))
        if lattice_type:
            lattice_cfg["type"] = lattice_type
        if lattice_boundary:
            lattice_cfg["boundary"] = lattice_boundary
        scenario.metadata["lattice_config"] = lattice_cfg

    backend_choice, backend_config = resolve_backend_selection(scenario.metadata, backend)
    print(f"[Δ-Machine] Using constraint backend: {backend_choice}")
    compiler = FunctionalCompiler()
    run_base_dir = scenario_dir.parent / "runs"
    meta = scenario.metadata or {}
    worker_count = max_workers if max_workers is not None else int(meta.get("max_workers", 6))
    if worker_count <= 0:
        worker_count = 1

    orchestrator = DeltaOrchestrator(
        scenario,
        compiler,
        max_workers=worker_count,
        run_base_dir=run_base_dir,
        backend=backend_choice,
        backend_config=backend_config,
    )
    if scenario.initial_condition_refs:
        ic_spec = scenario.initial_condition_refs[0]
        try:
            initial_condition = load_initial_condition(run_base_dir.parent, ic_spec)
            orchestrator.initial_condition = initial_condition
            if isinstance(ic_spec, dict):
                orchestrator.initial_condition_name = ic_spec.get("name", ic_spec.get("type", "generated"))
                seed = ic_spec.get("seed") if isinstance(ic_spec.get("seed"), int) else None
                orchestrator.initial_condition_seed = seed
            elif isinstance(ic_spec, str):
                orchestrator.initial_condition_name = ic_spec
            if isinstance(ic_spec, dict) and ic_spec.get("seed") == "random":
                orchestrator.initial_condition_seed = None
            orchestrator.initialize()
        except Exception as exc:
            raise RuntimeError(f"Failed to load initial condition for scenario '{scenario_name}': {exc}") from exc
    else:
        orchestrator.initialize()
    if orchestrator.backend != "taichi":
        orchestrator.start_workers()
    asyncio.run(orchestrator.run_async(steps))
    orchestrator.shutdown()
    if use_pr0_field_state:
        if previous_flag is None:
            os.environ.pop("DELTA_USE_PR0_FIELDSTATE", None)
        else:
            os.environ["DELTA_USE_PR0_FIELDSTATE"] = previous_flag


def main():
    args = parse_args()
    scenario_dir = Path(args.scenario_dir).resolve()
    if args.steps > 0:
        if not args.scenario:
            raise ValueError("Headless mode requires --scenario")
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
        run_gui(scenario_dir)


if __name__ == "__main__":
    main()


