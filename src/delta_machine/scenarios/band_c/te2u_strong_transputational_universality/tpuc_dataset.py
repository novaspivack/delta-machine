"""Dataset synthesis utilities for TE₂.U Strong Transputational Universality.

The implementation follows the plan logged in
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md`
and references the ΛΩ/Z₂ synthesis described in
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.K_REFLEXIVE_EQUIVALENCE_THEOREM/FINAL_SUMMARY_DISCUSSION.md`.

The generated benchmark suite enumerates TPU-C friendly workloads (saddle-point systems,
constraint-factored Hamiltonians, reflexive SAT reductions) together with classical
solver baselines so DSAC can quantify reflexive advantage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(slots=True)
class TPUCScenarioConfig:
    """Configuration for assembling TPU-C reflexive advantage benchmarks."""

    output_dir: Path
    matrix_size: int = 48
    task_families: Tuple[str, ...] = (
        "reflexive_linear_system",
        "holographic_quadratic",
        "reflexive_sat",
    )
    tasks_per_family: int = 24
    noise_level: float = 0.02
    random_seed: int | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.matrix_size < 8:
            raise ValueError("matrix_size must be >= 8 for TPU contraction blocks")
        if self.tasks_per_family <= 0:
            raise ValueError("tasks_per_family must be positive")
        if not self.task_families:
            raise ValueError("at least one task family is required")


def build_tpuc_benchmark_suite(config: TPUCScenarioConfig) -> List[Path]:
    """Generate TPU-C benchmark workloads with classical baselines."""

    config.validate()
    rng = np.random.default_rng(config.random_seed)
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    sample_paths: List[Path] = []

    for family in config.task_families:
        fam_dir = out_dir / family
        fam_dir.mkdir(exist_ok=True)
        for task_idx in range(config.tasks_per_family):
            seed = int(rng.integers(0, 2**32 - 1))
            task = _synthesise_task(
                family=family,
                matrix_size=config.matrix_size,
                noise_level=config.noise_level,
                seed=seed,
            )
            sample_path = fam_dir / f"task_{task_idx:03d}.npz"
            np.savez_compressed(sample_path, **task)
            manifest.append(
                {
                    "family": family,
                    "task_index": task_idx,
                    "path": sample_path.as_posix(),
                    "seed": seed,
                    "conditioning": float(task["conditioning"]),
                    "classical_cost": float(task["classical_cost"]),
                    "holographic_bias": float(task.get("holographic_bias", 0.0)),
                }
            )
            sample_paths.append(sample_path)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "te2u_tpuc_benchmark_v1",
                "config": {
                    "matrix_size": config.matrix_size,
                    "task_families": list(config.task_families),
                    "tasks_per_family": config.tasks_per_family,
                    "noise_level": config.noise_level,
                    **config.metadata,
                },
                "samples": manifest,
            },
            handle,
            indent=2,
        )

    return sample_paths


def _synthesise_task(
    family: str,
    matrix_size: int,
    noise_level: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_matrix = rng.normal(0.0, 1.0, size=(matrix_size, matrix_size))
    symmetric_component = (base_matrix + base_matrix.T) / 2.0
    skew_component = (base_matrix - base_matrix.T) / 2.0
    conditioning = np.linalg.cond(symmetric_component + np.eye(matrix_size) * 0.1)

    if family == "reflexive_linear_system":
        rhs = rng.normal(0.0, 1.0, size=matrix_size)
        solution = np.linalg.solve(symmetric_component + np.eye(matrix_size) * 0.25, rhs)
        classical_cost = matrix_size**3
        holographic_bias = float(np.var(solution))
        packed = {
            "family": np.array(family),
            "matrix": symmetric_component.astype(np.float32),
            "rhs": rhs.astype(np.float32),
            "solution": solution.astype(np.float32),
        }
    elif family == "holographic_quadratic":
        weights = rng.normal(0.0, 1.0, size=(matrix_size, matrix_size))
        holographic_bias = float(np.mean(np.abs(weights)))
        kernel = symmetric_component + noise_level * weights
        classical_cost = matrix_size**2 * math.log2(matrix_size)
        solution = _solve_quadratic(kernel, rng)
        packed = {
            "family": np.array(family),
            "kernel": kernel.astype(np.float32),
            "solution": solution.astype(np.float32),
        }
    elif family == "reflexive_sat":
        clause_tensor = _build_clause_tensor(matrix_size, rng)
        assignment = _solve_sat_tensor(clause_tensor, rng)
        holographic_bias = float(np.mean(assignment))
        classical_cost = matrix_size**2 * matrix_size
        packed = {
            "family": np.array(family),
            "clause_tensor": clause_tensor.astype(np.int8),
            "assignment": assignment.astype(np.int8),
        }
    else:
        raise ValueError(f"Unknown TPU-C task family: {family}")

    packed.update(
        {
            "conditioning": np.array(conditioning, dtype=np.float32),
            "classical_cost": np.array(classical_cost, dtype=np.float32),
            "holographic_bias": np.array(holographic_bias, dtype=np.float32),
            "skew_component": skew_component.astype(np.float32),
        }
    )
    return packed


def _solve_quadratic(kernel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(kernel + np.eye(kernel.shape[0]) * 0.05)
    coeffs = rng.normal(0.0, 0.5, size=eigvals.shape)
    state = eigvecs @ (coeffs / (np.abs(eigvals) + 0.1))
    return state


def _build_clause_tensor(dim: int, rng: np.random.Generator) -> np.ndarray:
    tensor = rng.integers(0, 2, size=(dim, dim, 3), endpoint=False)
    return tensor


def _solve_sat_tensor(clause_tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = clause_tensor.shape[0]
    assignment = np.zeros(dim, dtype=int)
    for i in range(dim):
        clause = clause_tensor[i]
        truth_bias = np.sum(clause, axis=1)
        assignment[i] = 1 if truth_bias.mean() + rng.normal(0, 0.1) > 1.0 else 0
    return assignment
