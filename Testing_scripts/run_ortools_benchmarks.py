#!/usr/bin/env python3
"""Run OR-Tools benchmarks for the reflexive TSP scenarios.

Design references:
- 1.5 DSAC Scenario Roadmap: /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md
- 1.18 Interim Findings Report: /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.18_interim_findings_report.md
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from delta_machine.config import ScenarioLoader
from delta_machine.analysis import get_tsp_config
from Testing_scripts.tsp_baseline import solve_tsp_with_ortools, ORTOOLS_AVAILABLE

DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "scenarios"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "ortools_benchmarks"
DEFAULT_SCENARIOS = [
    "tsp_reflexive.yaml",
    "tsp_reflexive_10.yaml",
    "tsp_reflexive_12.yaml",
    "tsp_reflexive_16.yaml",
    "tsp_reflexive_20.yaml",
    "tsp_reflexive_perturbed_20.yaml",
    "tsp_reflexive_perturbed_24.yaml",
    "tsp_reflexive_perturbed_32.yaml",
    "tsp_tsplib_eil51.yaml",
]


def load_cost_matrix(scenario_dir: Path, scenario_name: str) -> Dict[str, Any]:
    loader = ScenarioLoader(scenario_dir)
    spec = loader.load(scenario_name)
    config = get_tsp_config(spec.metadata or {})
    return {
        "scenario": scenario_name,
        "num_cities": config.num_cities,
        "cost_matrix": config.cost_matrix,
        "optimal_cost": config.optimal_cost,
        "optimal_tour": config.optimal_tour,
    }


def run_benchmark(entry: Dict[str, Any], time_limit: int) -> Dict[str, Any]:
    scenario_name = entry["scenario"]
    cost_matrix: np.ndarray = entry["cost_matrix"]
    num_cities: int = entry["num_cities"]
    optimal_cost = entry["optimal_cost"]

    start = time.perf_counter()
    tour, cost = solve_tsp_with_ortools(cost_matrix, time_limit=time_limit)
    runtime = time.perf_counter() - start

    gap_abs = float("nan")
    gap_rel = float("nan")
    if optimal_cost is not None:
        gap_abs = cost - float(optimal_cost)
        gap_rel = gap_abs / float(optimal_cost) if optimal_cost else float("nan")

    return {
        "scenario": scenario_name,
        "cities": num_cities,
        "time_limit": time_limit,
        "runtime": runtime,
        "cost": cost,
        "tour": tour,
        "optimal_cost": optimal_cost,
        "cost_gap": gap_abs,
        "cost_gap_pct": gap_rel * 100.0 if np.isfinite(gap_rel) else float("nan"),
    }


def write_results(results: Iterable[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "cities",
        "time_limit",
        "runtime",
        "cost",
        "optimal_cost",
        "cost_gap",
        "cost_gap_pct",
        "tour",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            serializable = dict(row)
            serializable["tour"] = " ".join(str(x) for x in row["tour"])
            writer.writerow(serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OR-Tools benchmarks for DSAC TSP scenarios")
    parser.add_argument("--scenario", action="append", help="Scenario file name (can be repeated). Defaults to the standard set.")
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR), help="Directory containing scenario YAML files")
    parser.add_argument("--time-limit", type=int, default=60, help="OR-Tools time limit per scenario (seconds)")
    parser.add_argument("--output", default=str(DEFAULT_RUN_DIR / "ortools_summary.csv"), help="Destination CSV path for benchmark results")
    args = parser.parse_args()

    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools is not installed; cannot run benchmarks")

    scenario_dir = Path(args.scenario_dir).resolve()
    selections: List[str]
    if args.scenario:
        selections = args.scenario
    else:
        selections = DEFAULT_SCENARIOS

    print("=== OR-Tools Benchmark Sweep ===")
    results: List[Dict[str, Any]] = []
    for name in selections:
        print(f"[Benchmark] Scenario: {name}")
        entry = load_cost_matrix(scenario_dir, name)
        try:
            row = run_benchmark(entry, time_limit=args.time_limit)
        except RuntimeError as exc:  # OR-Tools failure
            print(f"  OR-Tools failed: {exc}")
            row = {
                "scenario": name,
                "cities": entry["num_cities"],
                "time_limit": args.time_limit,
                "runtime": float("nan"),
                "cost": float("nan"),
                "optimal_cost": entry["optimal_cost"],
                "cost_gap": float("nan"),
                "cost_gap_pct": float("nan"),
                "tour": (),
            }
        else:
            print(f"  runtime = {row['runtime']:.3f}s, cost = {row['cost']:.6f}, gap = {row['cost_gap']:.6f}")
        results.append(row)

    output = Path(args.output).resolve()
    write_results(results, output)
    print(f"Benchmark summary written to {output}")


if __name__ == "__main__":
    main()
