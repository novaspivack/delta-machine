"""
Run reporting and summary generation for Δ-Machine scenarios.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from .config import ScenarioSpec
from .scenarios import RunResult


def generate_report(
    run_dir: Path,
    scenario: ScenarioSpec,
    result: RunResult,
    initial_condition_name: str | None = None,
    extras: Dict[str, Any] | None = None,
) -> Path:
    """
    Generate final run report in YAML format.

    Args:
        run_dir: Directory for run artifacts
        scenario: Scenario specification
        result: Run result
        initial_condition_name: Name of initial condition used

    Returns:
        Path to generated report file
    """
    report = {
        "scenario": {
            "name": scenario.name,
            "type": scenario.scenario_type,
            "lattice_shape": list(scenario.lattice_shape),
            "timestep": scenario.timestep,
            "max_steps": scenario.max_steps,
        },
        "initial_condition": initial_condition_name or "default",
        "run_summary": {
            "halted": result.halted,
            "halt_reason": result.halt_reason,
            "success": result.success,
            "success_reason": result.success_reason,
            "final_step": result.final_step,
            "final_dissonance": float(result.final_dissonance),
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": result.scenario_metrics.copy(),
        "halting_criteria": _serialize_halting_criteria(scenario.halting_criteria),
        "success_metrics": scenario.success_metrics.copy(),
    }

    if extras:
        report["extras"] = extras

    report_path = run_dir / "report.yaml"
    with open(report_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return report_path


def _serialize_halting_criteria(criteria: Any) -> Dict[str, Any] | None:
    """Serialize halting criteria for report."""
    if not criteria:
        return None
    return {
        "residual_threshold": criteria.residual_threshold,
        "dissonance_threshold": criteria.dissonance_threshold,
        "dissonance_plateau_steps": criteria.dissonance_plateau_steps,
        "dissonance_plateau_tolerance": criteria.dissonance_plateau_tolerance,
        "max_steps": criteria.max_steps,
        "success_condition": criteria.success_condition,
        "stagnation_window": criteria.stagnation_window,
        "stagnation_relative_delta": criteria.stagnation_relative_delta,
        "periodic_window": criteria.periodic_window,
        "periodic_tolerance": criteria.periodic_tolerance,
        "periodic_min_cycles": criteria.periodic_min_cycles,
        "periodic_min_amplitude": criteria.periodic_min_amplitude,
    }


def save_final_state(run_dir: Path, arrays: Dict[str, np.ndarray]):
    """Save final lattice state as numpy arrays."""
    state_dir = run_dir / "final_state"
    state_dir.mkdir(exist_ok=True)

    for name, arr in arrays.items():
        np.save(state_dir / f"{name}.npy", arr)

