"""TSP baseline harness for reflexive DSAC experiments.

Design references:
- 1.5 DSAC Scenario Roadmap (Reflexive TSP milestones)
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import time

from delta_machine.analysis import compute_tsp_metrics, extract_assignment_matrix, get_tsp_config
from delta_machine.config import ScenarioLoader
from delta_machine.functionals import FunctionalCompiler
from delta_machine.orchestrator import DeltaOrchestrator

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ORTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ORTOOLS_AVAILABLE = False


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = DEFAULT_PROJECT_ROOT / "scenarios"
MAX_BRUTE_FORCE_CITIES = 12


def solve_tsp_with_ortools(cost_matrix: np.ndarray, time_limit: int = 30) -> Tuple[Tuple[int, ...], float]:
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools is not installed; cannot run OR-Tools baseline")

    n = cost_matrix.shape[0]
    scale = 1_000_000

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(cost_matrix[from_node, to_node] * scale)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("OR-Tools failed to find a solution within the time limit")

    index = routing.Start(0)
    tour: list[int] = []
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    tour_tuple = tuple(int(node) for node in tour)

    cost = 0.0
    for i in range(len(tour_tuple)):
        src = tour_tuple[i]
        dst = tour_tuple[(i + 1) % len(tour_tuple)]
        cost += float(cost_matrix[src, dst])

    return tour_tuple, cost


def brute_force_tour(cost_matrix: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    """Compute the optimal Hamiltonian cycle (fixed start city) via brute force."""

    n = cost_matrix.shape[0]
    best_cost = float("inf")
    best_tour: Tuple[int, ...] | None = None

    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        cost = 0.0
        for idx in range(n):
            src = tour[idx]
            dst = tour[(idx + 1) % n]
            cost += float(cost_matrix[src, dst])
        if cost < best_cost:
            best_cost = cost
            best_tour = tuple(int(x) for x in tour)

    if best_tour is None:
        raise RuntimeError("Failed to locate any Hamiltonian cycle during brute force search")
    return best_tour, best_cost


def run_dsac_tsp(
    scenario_dir: Path,
    scenario_name: str,
    steps: int,
    use_pr0: bool = False,
    workers: int = 9,
) -> Dict[str, float]:
    loader = ScenarioLoader(scenario_dir)
    scenario = loader.load(scenario_name)
    compiler = FunctionalCompiler()
    orchestrator = DeltaOrchestrator(
        scenario,
        compiler,
        max_workers=max(1, workers),
        run_base_dir=scenario_dir.parent / "runs",
    )

    if use_pr0:
        orchestrator.scenario.metadata = dict(orchestrator.scenario.metadata)
        orchestrator.scenario.metadata["use_pr0_field_state"] = True

    orchestrator.initialize()
    orchestrator.start_workers()
    try:
        for _ in range(steps):
            orchestrator.step()
            if orchestrator.halted:
                break
        arrays = orchestrator.shared_state.arrays()
        config = get_tsp_config(orchestrator.scenario.metadata or {})
        assignment = extract_assignment_matrix(arrays["psi_real"], config)
        metrics = compute_tsp_metrics(assignment, config)
        return metrics
    finally:
        orchestrator.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DSAC Reflexive TSP against brute-force optimal tour")
    parser.add_argument("--scenario", default="tsp_reflexive.yaml", help="Scenario file name")
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--steps", type=int, default=8000, help="Maximum DSAC steps to execute")
    parser.add_argument("--use-pr0", action="store_true", help="Enable PR-0 field state integration during the run")
    parser.add_argument("--workers", type=int, default=9, help="Number of DSAC worker processes (default: 9)")
    parser.add_argument("--ortools", action="store_true", help="Run OR-Tools baseline (if installed) before DSAC")
    parser.add_argument("--ortools-time-limit", type=int, default=30, help="Time limit in seconds for the OR-Tools solver (default: 30s)")
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    loader = ScenarioLoader(scenario_dir)
    scenario = loader.load(args.scenario)
    config = get_tsp_config(scenario.metadata or {})

    print("=== Reflexive DSAC TSP Baseline ===")
    print(f"Cities: {config.num_cities}")
    optimal_cycle = config.optimal_tour
    optimal_cost = config.optimal_cost
    brute_force_elapsed = None

    if config.num_cities <= MAX_BRUTE_FORCE_CITIES:
        print("Computing brute-force Hamiltonian optimum...")
        brute_force_start = time.perf_counter()
        optimal_cycle, optimal_cost = brute_force_tour(config.cost_matrix)
        brute_force_elapsed = time.perf_counter() - brute_force_start
        print(f"Optimal tour cost: {optimal_cost:.6f}")
        print(f"Optimal tour cycle: {optimal_cycle} (0-indexed, returning to start)")
        print(f"Brute-force runtime: {brute_force_elapsed:.4f}s")
    elif optimal_cost is not None:
        if optimal_cycle is None:
            optimal_cycle = tuple(range(config.num_cities))
        print(
            "Skipping brute-force enumeration (city count exceeds limit); "
            "using scenario metadata for reference tour/cost."
        )
        print(f"Reference tour cost: {optimal_cost:.6f}")
        print(f"Reference tour cycle: {optimal_cycle}")
    else:
        print(
            "No brute-force baseline available (city count exceeds limit and no metadata cost provided)."
        )

    ortools_report: Dict[str, Any] | None = None
    if args.ortools:
        if not ORTOOLS_AVAILABLE:
            print("OR-Tools baseline requested but the package is not installed. Skipping.")
        else:
            print("Running OR-Tools baseline...")
            ort_start = time.perf_counter()
            tour, ort_cost = solve_tsp_with_ortools(config.cost_matrix, time_limit=args.ortools_time_limit)
            ort_elapsed = time.perf_counter() - ort_start
            ortools_report = {
                "tour": tour,
                "cost": ort_cost,
                "runtime": ort_elapsed,
            }
            print(f"OR-Tools tour cost: {ort_cost:.6f}")
            print(f"OR-Tools tour cycle: {tour}")
            print(f"OR-Tools runtime: {ort_elapsed:.4f}s")

    print("Running DSAC scenario...")
    dsac_start = time.perf_counter()
    metrics = run_dsac_tsp(
        scenario_dir,
        args.scenario,
        steps=args.steps,
        use_pr0=args.use_pr0,
        workers=args.workers,
    )
    dsac_elapsed = time.perf_counter() - dsac_start
    dsac_cost = metrics.get("tour_cost")
    cost_gap = metrics.get("tour_cost_gap")
    subtours = metrics.get("subtour_count")
    dsac_perm = metrics.get("permutation")

    print("--- DSAC Results ---")
    print(f"Doubly stochastic error: {metrics.get('doubly_stochastic_error'):.4e}")
    print(f"Subtour count: {subtours}")
    print(f"Assignment permutation: {dsac_perm}")
    print(f"Tour verified: {bool(metrics.get('tour_verified'))}")
    if dsac_perm is not None:
        cycle = [0]
        visited = {0}
        while len(cycle) < config.num_cities:
            nxt = dsac_perm[cycle[-1]]
            if nxt in visited:
                break
            visited.add(nxt)
            cycle.append(nxt)
        print(f"DSAC cycle (0-indexed): {tuple(cycle)}")
    if dsac_cost is not None:
        print(f"DSAC tour cost: {dsac_cost:.6f}")
    if cost_gap is not None:
        print(f"Cost gap: {cost_gap:.4%}")
    print(f"DSAC runtime: {dsac_elapsed:.4f}s")

    print("Comparison summary:")
    if dsac_cost is not None and optimal_cost is not None:
        print(f"  Absolute delta: {dsac_cost - optimal_cost:.6f}")
    if cost_gap is not None:
        print(f"  Relative gap: {cost_gap:.4%}")
    if brute_force_elapsed is not None and brute_force_elapsed > 0:
        print(f"  Runtime ratio (DSAC / brute-force): {dsac_elapsed / brute_force_elapsed:.2f}x")
    elif optimal_cost is not None:
        print("  Runtime ratio: skipped (metadata reference used instead of brute force)")
    else:
        print("  Runtime ratio: unavailable (no baseline)")
    if ortools_report:
        if dsac_cost is not None:
            print(f"  OR-Tools delta: {dsac_cost - ortools_report['cost']:.6f}")
        print(f"  OR-Tools runtime: {ortools_report['runtime']:.4f}s")


if __name__ == "__main__":
    main()
