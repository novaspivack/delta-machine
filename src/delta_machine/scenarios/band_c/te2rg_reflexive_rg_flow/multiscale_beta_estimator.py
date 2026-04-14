"""β-function estimation for TE₂.RG DSAC multi-scale ensembles.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/Mathematical_Foundations_of_Reflexive_Reality.tex

The estimator translates scale-labelled DSAC outputs into β-function samples via discrete
RG flows, providing empirical support for the SRRG program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class BetaEstimate:
    """Container for β-function estimates at a given coarse-graining scale."""

    scale: float
    coupling: float
    beta_value: float
    variance: float
    sample_count: int


def estimate_beta_function(
    scale_measurements: Sequence[Mapping[str, float]],
    min_samples: int = 4,
) -> Iterable[BetaEstimate]:
    """Estimate β-function using logarithmic scale regression."""

    if len(scale_measurements) < min_samples:
        raise ValueError("insufficient samples for β-function estimation")

    grouped: dict[float, list[tuple[float, float]]] = {}
    for entry in scale_measurements:
        scale = float(entry["scale"])
        coupling = float(entry["coupling"])
        flow = float(entry["flow"])
        grouped.setdefault(scale, []).append((coupling, flow))

    estimates: list[BetaEstimate] = []
    for scale, pairs in sorted(grouped.items()):
        couplings = np.array([p[0] for p in pairs], dtype=np.float64)
        flows = np.array([p[1] for p in pairs], dtype=np.float64)
        if couplings.size < min_samples:
            continue
        slope, intercept = _linear_regression(couplings, flows)
        beta_value = float(slope)
        variance = float(np.var(flows - (slope * couplings + intercept)))
        estimates.append(
            BetaEstimate(
                scale=scale,
                coupling=float(couplings.mean()),
                beta_value=beta_value,
                variance=variance,
                sample_count=int(couplings.size),
            )
        )
    return estimates


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    solution, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(solution[0]), float(solution[1])
