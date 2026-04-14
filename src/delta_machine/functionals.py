"""
Functional compilation layer bridging MFRR symbolic forms with executable kernels.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping

import sympy as sp
import numpy as np


STATE_SYMBOLS = {
    "psi_real": sp.symbols("psi_real"),
    "psi_imag": sp.symbols("psi_imag"),
    "chi": sp.symbols("chi"),
    "chi_dot": sp.symbols("chi_dot"),
    "dissonance": sp.symbols("dissonance"),
    "chi_reference": sp.symbols("chi_reference"),
}


@dataclass
class FunctionalKernel:
    """Executable representation of a compiled functional expression."""

    name: str
    expression: str
    sym_expr: sp.Expr
    variables: tuple[str, ...]
    dependencies: tuple[str, ...]
    weight: float
    _callable: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = None

    def __post_init__(self):
        if self._callable is None:
            self._callable = FunctionalCompiler().build_callable(self.sym_expr, self.variables)

    def evaluate(self, state_buffers: Mapping[str, np.ndarray]) -> np.ndarray:
        if self._callable is None:
            self._callable = FunctionalCompiler().build_callable(self.sym_expr, self.variables)
        return self._callable(state_buffers)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_callable"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._callable is None:
            self._callable = FunctionalCompiler().build_callable(self.sym_expr, self.variables)


class FunctionalCompiler:
    """
    Compile symbolic expressions (e.g., from MFRR) into fast NumPy callables.

    Expressions are defined over lattice-wise state variables such as psi_real
    or chi. Each compiled kernel expects the variables to be present in the
    state mapping supplied at evaluation time.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def compile(
        self,
        name: str,
        expression: str,
        variables: Iterable[str],
        dependencies: Iterable[str],
        weight: float = 1.0,
    ) -> FunctionalKernel:
        sym_expr = self._parse_expression(expression, variables)
        variable_tuple = tuple(variables)
        evaluator = self.build_callable(sym_expr, variable_tuple)
        return FunctionalKernel(
            name=name,
            expression=expression,
            sym_expr=sym_expr,
            variables=variable_tuple,
            dependencies=tuple(dependencies),
            weight=weight,
            _callable=evaluator,
        )

    def _parse_expression(self, expression: str, variables: Iterable[str]) -> sp.Expr:
        local_env = {STATE_SYMBOLS[var].name: STATE_SYMBOLS[var] for var in variables}
        try:
            sym_expr = sp.sympify(expression, locals=local_env)
        except sp.SympifyError as exc:
            raise ValueError(f"Failed to parse functional expression '{expression}'") from exc
        return sym_expr

    @lru_cache(maxsize=64)
    def build_callable(
        self,
        sym_expr: sp.Expr,
        variables: Iterable[str],
    ) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
        variable_tuple = tuple(variables)
        sym_vars = [STATE_SYMBOLS[var] for var in variable_tuple]
        func = sp.lambdify(sym_vars, sym_expr, modules="numpy")

        def evaluator(state_buffers: Mapping[str, np.ndarray]) -> np.ndarray:
            args = [state_buffers[var] for var in variable_tuple]
            result = func(*args)
            if not isinstance(result, np.ndarray):
                result = np.asarray(result)
            return result

        return evaluator


