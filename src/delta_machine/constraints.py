"""
Constraint graph representation and evaluation routines for the Δ-Machine.

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
from typing import Dict, Iterable, List

import numpy as np

try:
    from DSAC_tools.timing_instrumentation import profile_section
except ImportError:  # pragma: no cover - optional profiling dependency
    class profile_section:  # type: ignore[too-many-ancestors]
        """Fallback no-op context manager when timing instrumentation is unavailable."""

        def __init__(self, *_: object, **__: object) -> None:  # noqa: D401
            """Initialize no-op context manager."""

        def __enter__(self) -> "profile_section":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

from .functionals import FunctionalKernel


@dataclass(slots=True)
class ConstraintNode:
    """Runtime node representing a reflexive constraint."""

    name: str
    kernel: FunctionalKernel
    target_field: str
    residual: np.ndarray | None = None
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)

    def compute_residual(self, state_buffers: Dict[str, np.ndarray]) -> np.ndarray:
        expected = state_buffers[self.target_field]
        actual = self.kernel.evaluate(state_buffers)
        residual = expected - actual
        self.residual = residual
        return residual


class ConstraintGraph:
    """
    Directed graph of reflexive constraints.

    Tracks dependency ordering and provides helpers for computing total
    ontological dissonance across the constraint ensemble.
    """

    def __init__(self, nodes: Iterable[ConstraintNode]):
        self._nodes = {node.name: node for node in nodes}
        self._order = self._topological_sort()

    @property
    def nodes(self) -> Dict[str, ConstraintNode]:
        return self._nodes

    @property
    def order(self) -> List[str]:
        return list(self._order)

    def _topological_sort(self) -> List[str]:
        visited = set()
        order: List[str] = []

        def visit(node_name: str, stack: set[str]):
            if node_name in stack:
                raise ValueError(f"Cyclic dependency detected in constraint graph at '{node_name}'")
            if node_name in visited:
                return
            stack.add(node_name)
            node = self._nodes[node_name]
            for dep in node.dependencies:
                if dep in self._nodes:
                    visit(dep, stack)
            stack.remove(node_name)
            visited.add(node_name)
            order.append(node_name)

        for name in self._nodes:
            visit(name, set())

        return order

    def evaluate(self, state_buffers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Evaluate all constraints and return residuals per node.

        When timing instrumentation is available, this method records aggregate
        timings for the full evaluation and for each constraint node; otherwise
        it behaves as a plain evaluation loop.
        """
        residual_map: Dict[str, np.ndarray] = {}
        with profile_section("constraint_graph_evaluate"):
            for name in self._order:
                node = self._nodes[name]
                with profile_section(f"constraint_node_{name}"):
                    residual_map[name] = node.compute_residual(state_buffers)
        return residual_map

    def total_dissonance(self) -> float:
        total = 0.0
        for node in self._nodes.values():
            if node.residual is None:
                continue
            total += node.weight * float(np.mean(np.abs(node.residual)))
        return total


