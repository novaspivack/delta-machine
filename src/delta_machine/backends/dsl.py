"""Tools for translating SymPy expressions into Taichi-friendly code snippets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import sympy as sp

SUPPORTED_FUNCTIONS = {
    sp.Abs: "ti.abs",
    sp.sqrt: "ti.sqrt",
    sp.log: "ti.log",
    sp.exp: "ti.exp",
    sp.sin: "ti.sin",
    sp.cos: "ti.cos",
    sp.tan: "ti.tan",
    sp.sinh: "ti.sinh",
    sp.cosh: "ti.cosh",
    sp.tanh: "ti.tanh",
}

KNOWN_SYMBOLS = {
    "psi_real_sq": "psi_real_sq",
    "psi_imag_sq": "psi_imag_sq",
    "chi_sq": "chi_sq",
    "chi_dot_sq": "chi_dot_sq",
    "psi_norm_sq": "psi_norm_sq",
    "psi_real_imag_sq": "psi_real_imag_sq",
    "psi_imag_real_sq": "psi_imag_real_sq",
    "psi_real_sq_imag": "psi_real_sq_imag",
    "psi_imag_sq_real": "psi_imag_sq_real",
    "chi_sq_chi": "chi_sq_chi",
    "chi_dot_sq_chi_dot": "chi_dot_sq_chi_dot",
    "chi_dot_norm_sq": "chi_dot_norm_sq",
    "psi_real_pow4": "psi_real_pow4",
    "psi_imag_pow4": "psi_imag_pow4",
    "psi_norm_sq_sq": "psi_norm_sq_sq",
    "psi_sq_cross": "psi_sq_cross",
    "chi_sq_sq": "chi_sq_sq",
    "chi_dot_sq_sq": "chi_dot_sq_sq",
    "chi_sq_norm_sq": "chi_sq_norm_sq",
    "chi_dot_sq_norm_sq": "chi_dot_sq_norm_sq",
    "psi_real_psi_imag": "psi_real_psi_imag",
    "psi_norm_sq_eps": "psi_norm_sq_eps",
    "log_psi_norm_sq_eps": "log_psi_norm_sq_eps",
    "chi_psi_real": "chi_psi_real",
    "chi_psi_imag": "chi_psi_imag",
    "chi_dot_psi_real": "chi_dot_psi_real",
    "chi_dot_psi_imag": "chi_dot_psi_imag",
    "psi_r_pi": "psi_real_psi_imag",
    "psi_r_pii": "psi_real_imag_sq",
    "psi_real_cu": "psi_real_cu",
    "psi_imag_cu": "psi_imag_cu",
    "log_radius": "log_psi_norm_sq_eps",
}


@dataclass
class TaichiExpressionTranslator:
    """Translate a SymPy expression into a Taichi expression string."""

    variable_map: Dict[str, str]

    def translate(self, expression: sp.Expr) -> str:
        expr = sp.sympify(expression)
        return self._translate(expr)

    def _translate(self, expr: sp.Expr) -> str:
        if expr.is_Number:
            return repr(float(expr))
        if expr.is_Symbol:
            name = expr.name
            if name in self.variable_map:
                return self.variable_map[name]
            if name in KNOWN_SYMBOLS:
                return KNOWN_SYMBOLS[name]
            raise ValueError(f"Unsupported symbol '{name}' in Taichi DSL")
        if expr.is_Add:
            parts = [self._translate(arg) for arg in expr.args]
            return "(" + " + ".join(parts) + ")"
        if expr.is_Mul:
            parts = [self._translate(arg) for arg in expr.args]
            return "(" + " * ".join(parts) + ")"
        if expr.is_Pow:
            base = self._translate(expr.base)
            exp = expr.exp
            if exp.is_Integer:
                return f"{base}**{int(exp)}"
            return f"ti.pow({base}, {self._translate(exp)})"
        func = expr.func
        if func in SUPPORTED_FUNCTIONS:
            args = ", ".join(self._translate(a) for a in expr.args)
            return f"{SUPPORTED_FUNCTIONS[func]}({args})"
        raise ValueError(f"Unsupported SymPy expression '{sp.srepr(expr)}' for Taichi backend")
