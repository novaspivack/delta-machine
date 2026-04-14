"""
Scenario configuration models for the Δ-Machine runtime.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import yaml


@dataclass(slots=True)
class ConstraintSpec:
    """DSL description of a reflexive constraint functional."""

    name: str
    expression: str
    variables: List[str]
    target: str
    dependencies: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass(slots=True)
class HaltingCriteria:
    """Criteria for automatic run termination."""

    residual_threshold: float | None = None
    dissonance_threshold: float | None = None
    dissonance_plateau_steps: int | None = None
    dissonance_plateau_tolerance: float = 0.001
    max_steps: int | None = None
    success_condition: str | None = None
    stagnation_window: int | None = None
    stagnation_relative_delta: float = 1.0e-4
    periodic_window: int | None = None
    periodic_tolerance: float = 1.0e-4
    periodic_min_cycles: int = 3
    periodic_min_amplitude: float = 1.0e-4


@dataclass(slots=True)
class ScenarioSpec:
    """Complete scenario definition for a Δ-Machine run."""

    name: str
    lattice_shape: tuple[int, int]
    timestep: float
    max_steps: int
    constraints: List[ConstraintSpec]
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_type: str = "generic"
    halting_criteria: HaltingCriteria | None = None
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    initial_condition_refs: List[str | Dict[str, Any]] = field(default_factory=list)

    @property
    def lattice_x(self) -> int:
        return int(self.lattice_shape[1])

    @property
    def lattice_y(self) -> int:
        return int(self.lattice_shape[0])


class ScenarioLoader:
    """
    Load scenario specifications from YAML or JSON documents.

    The loader validates required fields and supports relative references
    so that scenario files can include shared fragments.
    """

    def __init__(self, root: Path):
        self._root = Path(root)

    def load(self, name: str) -> ScenarioSpec:
        path = self._root / name
        if not path.exists():
            raise FileNotFoundError(f"Scenario '{name}' not found at {path}")

        data = self._read_file(path)
        return self._parse_scenario(data, path)

    def _read_file(self, path: Path) -> Dict[str, Any]:
        if path.suffix.lower() in {".yaml", ".yml"}:
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        raise ValueError(f"Unsupported scenario format: {path.suffix}")

    def _parse_scenario(self, data: Dict[str, Any], path: Path) -> ScenarioSpec:
        try:
            name = data["name"]
            lattice = tuple(data["lattice"])
            timestep = float(data["timestep"])
            max_steps = int(data["max_steps"])
            constraint_defs = data["constraints"]
        except KeyError as exc:
            raise ValueError(f"Missing required field in {path}: {exc}") from exc

        if len(lattice) != 2:
            raise ValueError(f"Lattice must be [rows, cols]; got {lattice}")

        constraints = [
            ConstraintSpec(
                name=c["name"],
                expression=c["expression"],
                variables=list(c.get("variables", [])),
                target=c.get("target", c["name"]),
                dependencies=list(c.get("dependencies", [])),
                weight=float(c.get("weight", 1.0)),
            )
            for c in constraint_defs
        ]

        halting_data = data.get("halting_criteria", {})
        halting_criteria = None
        if halting_data:
            halting_criteria = HaltingCriteria(
                residual_threshold=halting_data.get("residual_threshold"),
                dissonance_threshold=halting_data.get("dissonance_threshold"),
                dissonance_plateau_steps=halting_data.get("dissonance_plateau_steps"),
                dissonance_plateau_tolerance=float(
                    halting_data.get("dissonance_plateau_tolerance", 0.001)
                ),
                max_steps=halting_data.get("max_steps"),
                success_condition=halting_data.get("success_condition"),
                stagnation_window=halting_data.get("stagnation_window"),
                stagnation_relative_delta=float(
                    halting_data.get("stagnation_relative_delta", 1.0e-4)
                    if halting_data.get("stagnation_relative_delta") is not None
                    else 1.0e-4
                ),
                periodic_window=halting_data.get("periodic_window"),
                periodic_tolerance=float(
                    halting_data.get("periodic_tolerance", 1.0e-4)
                    if halting_data.get("periodic_tolerance") is not None
                    else 1.0e-4
                ),
                periodic_min_cycles=int(
                    halting_data.get("periodic_min_cycles", 3)
                ),
                periodic_min_amplitude=float(
                    halting_data.get("periodic_min_amplitude", 1.0e-4)
                    if halting_data.get("periodic_min_amplitude") is not None
                    else 1.0e-4
                ),
            )

        return ScenarioSpec(
            name=name,
            lattice_shape=(int(lattice[0]), int(lattice[1])),
            timestep=timestep,
            max_steps=max_steps,
            constraints=constraints,
            initial_conditions=data.get("initial_conditions", {}),
            metadata=data.get("metadata", {}),
            scenario_type=data.get("scenario_type", "generic"),
            halting_criteria=halting_criteria,
            success_metrics=data.get("success_metrics", {}),
            initial_condition_refs=data.get("initial_condition_refs", []),
        )


