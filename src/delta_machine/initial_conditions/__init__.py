"""
Initial condition generator library for Δ-Machine scenarios.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

from .generators import (
    InitialConditionGenerator,
    RandomGenerator,
    GaussianGenerator,
    VortexGenerator,
    SolitonGenerator,
    PatternGenerator,
    PolynomialDatasetGenerator,
    TE1JarzynskiDatasetGenerator,
    PR0DiffusionDatasetGenerator,
    load_generator,
    list_generators,
)
from .registry import InitialConditionRegistry, load_initial_condition

__all__ = [
    "InitialConditionGenerator",
    "RandomGenerator",
    "GaussianGenerator",
    "VortexGenerator",
    "SolitonGenerator",
    "PatternGenerator",
    "PolynomialDatasetGenerator",
    "TE1JarzynskiDatasetGenerator",
    "PR0DiffusionDatasetGenerator",
    "load_generator",
    "list_generators",
    "InitialConditionRegistry",
    "load_initial_condition",
]

