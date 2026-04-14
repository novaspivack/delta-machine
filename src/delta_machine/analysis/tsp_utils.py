"""
TSP analysis helpers for reflexive DSAC scenarios.

Design references:
- 1.5 DSAC Scenario Roadmap (Section 5):
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(slots=True)
class TSPConfig:
    num_cities: int
    row_slice: slice
    col_slice: slice
    beta: float
    epsilon: float
    sinkhorn_iterations: int
    stochasticity_tolerance: float
    assignment_threshold: float
    cost_matrix: np.ndarray
    optimal_cost: float | None = None
    optimal_tour: Tuple[int, ...] | None = None
    cost_gap_target: float | None = None
    recommended_min_side: int = 0


def _ensure_square(matrix: np.ndarray, n: int) -> np.ndarray:
    if matrix.shape != (n, n):
        raise ValueError(f"Cost matrix must be {n}x{n}; got {matrix.shape}")
    return matrix.astype(float)


def _build_cost_matrix(meta: Dict[str, Any], base_dir: Path | None = None) -> np.ndarray:
    if "cost_matrix" in meta and meta["cost_matrix"] is not None:
        matrix = np.asarray(meta["cost_matrix"], dtype=float)
        np.fill_diagonal(matrix, np.max(matrix) + 1.0)
        return matrix
    if "city_positions" in meta and meta["city_positions"] is not None:
        positions = np.asarray(meta["city_positions"], dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("city_positions must be a list of [x, y] pairs")
        diff = positions[:, None, :] - positions[None, :, :]
        matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(matrix, np.max(matrix) + 1.0)
        return matrix
    if "cost_matrix_path" in meta and meta["cost_matrix_path"]:
        path = Path(meta["cost_matrix_path"])
        if not path.is_absolute() and base_dir:
            path = base_dir / path
        if path.suffix.lower() == ".npy":
            matrix = np.load(path)
        else:
            matrix = np.loadtxt(path, delimiter=meta.get("cost_matrix_delimiter", ","))
        matrix = np.asarray(matrix, dtype=float)
        np.fill_diagonal(matrix, np.max(matrix) + 1.0)
        return matrix
    raise ValueError("TSP metadata must define cost_matrix, city_positions, or cost_matrix_path")


def get_tsp_config(metadata: Dict[str, Any], base_dir: Path | None = None) -> TSPConfig:
    tsp_meta = dict(metadata.get("tsp", {}))
    if not tsp_meta:
        raise ValueError("Scenario metadata missing 'tsp' section for tsp_reflexive scenario")

    num_cities = int(tsp_meta.get("num_cities", 0))
    if num_cities <= 1:
        raise ValueError("tsp.num_cities must be greater than 1")
    row_start = int(tsp_meta.get("row_start", 0))
    col_start = int(tsp_meta.get("col_start", 0))
    row_slice = slice(row_start, row_start + num_cities)
    col_slice = slice(col_start, col_start + num_cities)

    beta = float(tsp_meta.get("beta", 8.0))
    epsilon = float(tsp_meta.get("epsilon", 1e-6))
    sinkhorn_iterations = int(tsp_meta.get("sinkhorn_iterations", 3))
    stochasticity_tolerance = float(tsp_meta.get("stochasticity_tolerance", 1e-3))
    assignment_threshold = float(tsp_meta.get("assignment_threshold", 0.55))
    cost_gap_target = tsp_meta.get("cost_gap_target")
    if cost_gap_target is not None:
        cost_gap_target = float(cost_gap_target)

    cost_matrix = _ensure_square(_build_cost_matrix(tsp_meta, base_dir), num_cities)

    optimal_cost = tsp_meta.get("optimal_cost")
    if optimal_cost is not None:
        optimal_cost = float(optimal_cost)
    optimal_tour = tsp_meta.get("optimal_tour")
    if optimal_tour is not None:
        optimal_tour = tuple(int(x) for x in optimal_tour)
        if optimal_cost is None:
            optimal_cost = float(
                sum(cost_matrix[i, optimal_tour[i]] for i in range(num_cities))
            )

    margin = int(tsp_meta.get("lattice_margin", 6))
    recommended_min_side = max(row_start + num_cities + margin, col_start + num_cities + margin)

    return TSPConfig(
        num_cities=num_cities,
        row_slice=row_slice,
        col_slice=col_slice,
        beta=beta,
        epsilon=epsilon,
        sinkhorn_iterations=max(1, sinkhorn_iterations),
        stochasticity_tolerance=stochasticity_tolerance,
        assignment_threshold=assignment_threshold,
        cost_matrix=cost_matrix,
        optimal_cost=optimal_cost,
        optimal_tour=optimal_tour,
        cost_gap_target=cost_gap_target,
        recommended_min_side=recommended_min_side,
    )


def extract_assignment_matrix(field: np.ndarray, config: TSPConfig) -> np.ndarray:
    return np.asarray(field[config.row_slice, config.col_slice], dtype=float)


def apply_sinkhorn_step(matrix: np.ndarray, config: TSPConfig, beta_scale: float = 1.0) -> np.ndarray:
    biased = np.clip(matrix, config.epsilon, None)
    beta = max(config.beta * beta_scale, config.epsilon)
    cost_bias = np.exp(-beta * (config.cost_matrix - np.min(config.cost_matrix)))
    biased *= cost_bias
    updated = biased
    for _ in range(config.sinkhorn_iterations):
        row_sum = np.sum(updated, axis=1, keepdims=True) + config.epsilon
        updated = updated / row_sum
        col_sum = np.sum(updated, axis=0, keepdims=True) + config.epsilon
        updated = updated / col_sum
    return updated


def embed_assignment_matrix(array: np.ndarray, matrix: np.ndarray, config: TSPConfig) -> None:
    array[config.row_slice, config.col_slice] = matrix


def _permutation_from_assignment(assignment: np.ndarray, threshold: float) -> Tuple[np.ndarray, bool]:
    choice = np.argmax(assignment, axis=1)
    unique = len(set(int(x) for x in choice)) == assignment.shape[0]
    if not unique:
        return choice, False
    max_vals = assignment[np.arange(assignment.shape[0]), choice]
    if np.any(max_vals < threshold):
        return choice, False
    return choice, True


def _enumerate_subtours(permutation: Sequence[int]) -> List[List[int]]:
    n = len(permutation)
    visited = set()
    cycles: List[List[int]] = []
    for start in range(n):
        if start in visited:
            continue
        current = start
        cycle: List[int] = []
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation[current]
            if len(cycle) > n:
                break
        cycles.append(cycle)
    return cycles


def compute_tsp_metrics(assignment: np.ndarray, config: TSPConfig) -> Dict[str, Any]:
    row_sums = np.sum(assignment, axis=1)
    col_sums = np.sum(assignment, axis=0)
    row_error = np.abs(row_sums - 1.0)
    col_error = np.abs(col_sums - 1.0)
    doubly_stochastic_error = float(np.mean(row_error) + np.mean(col_error))

    permutation, strong_permutation = _permutation_from_assignment(assignment, config.assignment_threshold)
    cycles = _enumerate_subtours(permutation)
    subtour_count = len(cycles)
    has_single_cycle = subtour_count == 1 and len(cycles[0]) == config.num_cities if cycles else False

    optimal_cost = config.optimal_cost
    if optimal_cost is None and config.optimal_tour is not None:
        optimal_cost = float(
            sum(config.cost_matrix[i, int(config.optimal_tour[i])] for i in range(config.num_cities))
        )

    cost = None
    if config.cost_matrix is not None and strong_permutation:
        indices = tuple(int(i) for i in permutation)
        cost = float(sum(config.cost_matrix[i, indices[i]] for i in range(config.num_cities)))

    cost_gap = None
    if cost is not None and optimal_cost:
        cost_gap = max(0.0, (cost - optimal_cost) / optimal_cost)

    entropy = float(-np.sum(assignment * np.log(np.clip(assignment, config.epsilon, None))) / config.num_cities)

    stochasticity_ok = doubly_stochastic_error <= config.stochasticity_tolerance
    cost_ok = True
    if config.cost_gap_target is not None and cost_gap is not None:
        cost_ok = cost_gap <= config.cost_gap_target
    tour_verified = 1.0 if strong_permutation and has_single_cycle and stochasticity_ok and cost_ok else 0.0

    metrics: Dict[str, Any] = {
        "row_sum_error": row_error.tolist(),
        "col_sum_error": col_error.tolist(),
        "doubly_stochastic_error": doubly_stochastic_error,
        "stochasticity_verified": 1.0 if stochasticity_ok else 0.0,
        "assignment_entropy": entropy,
        "subtour_count": subtour_count,
        "permutation": [int(x) for x in permutation],
        "tour_verified": tour_verified,
    }

    if cycles:
        metrics["cycles"] = [[int(v) for v in cycle] for cycle in cycles]
    if cost is not None:
        metrics["tour_cost"] = cost
    if cost_gap is not None:
        metrics["tour_cost_gap"] = cost_gap
    if optimal_cost is not None:
        metrics["optimal_cost"] = optimal_cost
    if config.optimal_tour is not None:
        metrics["optimal_tour"] = list(config.optimal_tour)
    if config.cost_gap_target is not None:
        metrics["cost_gap_target"] = config.cost_gap_target

    return metrics
