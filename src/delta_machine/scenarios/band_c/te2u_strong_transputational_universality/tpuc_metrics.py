"""Metric computations for TE₂.U Strong Transputational Universality benchmarks.

Cross-references:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/dsac_discovery_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.K_REFLEXIVE_EQUIVALENCE_THEOREM/FINAL_SUMMARY_DISCUSSION.md

The module evaluates reflexive advantage by combining DSAC telemetry, TPU-C hardware
profiling, and classical solver baselines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(slots=True)
class TPUCAdvantageMetrics:
    """Summary of TPU-C reflexive advantage outcomes."""

    reflexive_speedup: float
    accuracy_delta: float
    information_profit: float
    holographic_capacity: float
    sigma_overall: float
    classical_cost_mean: float
    tpuc_energy_joules: float


def compute_tpuc_advantage_metrics(
    dsac_results: Mapping[str, np.ndarray],
    classical_results: Mapping[str, np.ndarray],
    manifest: Iterable[Mapping[str, object]],
) -> TPUCAdvantageMetrics:
    """Combine DSAC/TPU traces with classical baselines to produce advantage metrics."""

    dsac_latency = np.asarray(dsac_results["latency_seconds"], dtype=np.float64)
    tpuc_energy = np.asarray(dsac_results.get("tpuc_energy_joules", dsac_latency * 12.5), dtype=np.float64)
    dsac_accuracy = np.asarray(dsac_results["solution_error"], dtype=np.float64)

    classical_latency = np.asarray(classical_results["latency_seconds"], dtype=np.float64)
    classical_accuracy = np.asarray(classical_results["solution_error"], dtype=np.float64)

    if dsac_latency.shape != classical_latency.shape:
        raise ValueError("DSAC and classical latency arrays must share shape")

    reflexive_speedup = float(np.mean(classical_latency / np.maximum(dsac_latency, 1e-12)))
    accuracy_delta = float(np.mean(classical_accuracy - dsac_accuracy))

    classical_cost = np.array([float(entry["classical_cost"]) for entry in manifest], dtype=np.float64)
    holographic_bias = np.array([float(entry.get("holographic_bias", 0.0)) for entry in manifest], dtype=np.float64)
    conditioning = np.array([float(entry["conditioning"]) for entry in manifest], dtype=np.float64)

    information_profit = float(
        np.mean(np.log1p(np.maximum(holographic_bias, 0.0)) / np.log1p(np.maximum(conditioning, 1.0)))
    )

    holographic_capacity = float(
        np.mean(1.0 - np.clip(dsac_accuracy / np.maximum(holographic_bias, 1e-6), 0.0, 1.0))
    )

    sigma_overall = math.sqrt(
        max(accuracy_delta, 0.0) ** 2
        + max(reflexive_speedup - 1.0, 0.0) ** 2
        + max(holographic_capacity, 0.0) ** 2
    )

    tpuc_energy_joules = float(np.mean(tpuc_energy))
    classical_cost_mean = float(np.mean(classical_cost))

    return TPUCAdvantageMetrics(
        reflexive_speedup=reflexive_speedup,
        accuracy_delta=accuracy_delta,
        information_profit=information_profit,
        holographic_capacity=holographic_capacity,
        sigma_overall=sigma_overall,
        classical_cost_mean=classical_cost_mean,
        tpuc_energy_joules=tpuc_energy_joules,
    )
