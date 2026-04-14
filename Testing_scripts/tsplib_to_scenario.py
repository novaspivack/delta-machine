#!/usr/bin/env python3
"""Convert a TSPLIB TSP instance into a Δ-Machine scenario YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import yaml
import numpy as np
import tsplib95
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def normalize_coords(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    arr = np.asarray(coords, dtype=float)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    span = np.where((max_vals - min_vals) < 1e-9, 1.0, max_vals - min_vals)
    norm = (arr - min_vals) / span
    return [(float(x), float(y)) for x, y in norm]


def solve_with_ortools(coords: List[Tuple[float, float]]):
    arr = np.asarray(coords, dtype=float)
    diff = arr[:, None, :] - arr[None, :, :]
    cost = np.linalg.norm(diff, axis=-1)
    n = len(coords)
    scale = 1_000_000
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_index: int, to_index: int) -> int:
        return int(cost[manager.IndexToNode(from_index), manager.IndexToNode(to_index)] * scale)

    transit_cb = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(120)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools failed to find a tour for the TSPLIB instance")

    index = routing.Start(0)
    tour: List[int] = []
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    total_cost = 0.0
    for i in range(len(tour)):
        total_cost += float(cost[tour[i], tour[(i + 1) % len(tour)]])
    return tour, total_cost


def build_scenario_dict(
    name: str,
    description: str,
    coords: List[Tuple[float, float]],
    tour: List[int],
    optimal_cost: float,
    max_steps: int,
    notes_ref: str,
    max_workers: int,
) -> dict:
    num_cities = len(coords)
    sinkhorn_iterations = max(16, num_cities // 2)
    scenario = {
        "name": name,
        "description": description,
        "scenario_type": "tsp_reflexive",
        "lattice": [64, 64],
        "timestep": 0.02,
        "max_steps": max_steps,
        "initial_condition_refs": [
            {
                "name": "tsp_uniform_warm_start",
                "type": "tsp_warm_start",
                "parameters": {
                    "num_cities": num_cities,
                    "embedding_row_start": 1,
                    "embedding_col_start": 1,
                    "noise_level": 0.1,
                },
            }
        ],
        "constraints": [
            {
                "name": "psi_imag_sync",
                "expression": "chi - psi_imag",
                "variables": ["psi_imag"],
                "target": "psi_imag",
                "dependencies": ["chi"],
                "weight": 0.05,
            },
            {
                "name": "chi_decay",
                "expression": "chi * 0.02",
                "variables": ["chi"],
                "target": "chi",
                "dependencies": [],
                "weight": 0.08,
            },
            {
                "name": "chi_dot_decay",
                "expression": "0.0",
                "variables": ["chi_dot"],
                "target": "chi_dot",
                "dependencies": [],
                "weight": 0.08,
            },
        ],
        "halting_criteria": {
            "success_condition": "tour_verified",
            "max_steps": max_steps,
            "dissonance_plateau_steps": max(320, num_cities * 10),
            "dissonance_plateau_tolerance": 0.0015,
        },
        "success_metrics": {
            "tour_verified": {"min": 1.0},
            "doubly_stochastic_error": {"max": 0.02},
            "tour_cost_gap": {"max": 0.05},
        },
        "metadata": {
            "notes_ref": notes_ref,
            "description": description,
            "max_workers": max_workers,
            "tsp": {
                "num_cities": num_cities,
                "row_start": 1,
                "col_start": 1,
                "beta": 8.0,
                "epsilon": 1.0e-6,
                "sinkhorn_iterations": sinkhorn_iterations,
                "stochasticity_tolerance": 0.02,
                "assignment_threshold": 0.42,
                "cost_gap_target": 0.05,
                "city_positions": [[float(x), float(y)] for x, y in coords],
                "optimal_tour": tour,
                "optimal_cost": float(optimal_cost),
            },
            "expected_behavior": "DSAC matches the TSPLIB optimal tour while providing reflexive diagnostics.",
        },
    }
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TSPLIB instance to DSAC scenario")
    parser.add_argument("--tsp-file", required=True, type=Path, help="Path to TSPLIB .tsp file")
    parser.add_argument("--output", required=True, type=Path, help="Destination YAML path")
    parser.add_argument("--name", default="Reflexive TSP (TSPLIB)")
    parser.add_argument("--description", default="TSPLIB-derived TSP scenario for DSAC benchmarking")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument(
        "--notes-ref",
        default="/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md",
    )
    args = parser.parse_args()

    problem = tsplib95.load_problem(str(args.tsp_file))
    if not problem.node_coords:
        raise ValueError("TSPLIB problem does not contain node coordinates (EDGE_WEIGHT_TYPE must be EUC_2D)")

    node_ids = sorted(problem.get_nodes())
    coords = [problem.node_coords[i] for i in node_ids]
    normalized = normalize_coords(coords)

    optimal_tour, optimal_cost = solve_with_ortools(normalized)

    scenario = build_scenario_dict(
        name=args.name,
        description=args.description,
        coords=normalized,
        tour=[int(t) for t in optimal_tour],
        optimal_cost=optimal_cost,
        max_steps=args.max_steps,
        notes_ref=args.notes_ref,
        max_workers=args.max_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(scenario, fh, sort_keys=False)
    print(f"Wrote scenario to {args.output}")


if __name__ == "__main__":
    main()
