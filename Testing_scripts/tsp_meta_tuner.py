"""Short-loop TSP meta-control tuner.

Design reference:
- 1.16 New Avenues for DSAC:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.16_new_avenues_for_dsac.md
"""

from __future__ import annotations

import argparse
import itertools
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from delta_machine.config import ScenarioLoader
from delta_machine.functionals import FunctionalCompiler
from delta_machine.orchestrator import DeltaOrchestrator


def run_candidate(
    loader: ScenarioLoader,
    scenario_name: str,
    meta_config: Dict[str, float],
    steps: int,
    workers: int,
    artifact_dir: Path | None = None,
) -> Dict[str, float | int | bool | str]:
    spec = loader.load(scenario_name)
    metadata = dict(spec.metadata or {})
    metadata["meta_control"] = meta_config
    spec.metadata = metadata

    halting = getattr(spec, "halting_criteria", None)
    if halting is not None:
        halting.success_condition = None
        halting.max_steps = steps
        plateau = getattr(halting, "dissonance_plateau_steps", None)
        if plateau is not None and plateau < steps + 5:
            halting.dissonance_plateau_steps = steps + 5

    run_base_dir = None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_base_dir = artifact_dir

    orchestrator = DeltaOrchestrator(
        spec,
        FunctionalCompiler(),
        max_workers=workers,
        run_base_dir=run_base_dir,
    )
    orchestrator.start_workers(worker_count=workers)

    first_dissonance = None
    try:
        while orchestrator.step_counter < steps and not orchestrator.halted:
            orchestrator.step()
            if first_dissonance is None and orchestrator.scenario_runner:
                hist = orchestrator.scenario_runner.dissonance_history
                if hist:
                    first_dissonance = hist[0]

        metrics = orchestrator.scenario_runner.scenario_metrics if orchestrator.scenario_runner else {}
        final_dissonance = metrics.get("dissonance", orchestrator.last_dissonance)
        if first_dissonance is None:
            first_dissonance = final_dissonance
        dissonance_drop = first_dissonance - final_dissonance
        tour_cost = metrics.get("tour_cost")
        tour_gap = metrics.get("tour_cost_gap")
        subtours = metrics.get("subtour_count", 0)
        halted = orchestrator.halted
        halt_reason = orchestrator.halt_reason or ""
        steps_taken = orchestrator.step_counter
        dissonance_history = []
        if orchestrator.scenario_runner:
            dissonance_history = [float(x) for x in orchestrator.scenario_runner.dissonance_history]

        score = (
            final_dissonance
            - max(0.0, dissonance_drop)
            + (tour_gap if tour_gap is not None else 0.25)
            + 0.02 * max(0, subtours - 1)
            + 0.02 * max(0, steps_taken - 1)
            + (0.05 if halted else 0.0)
        )

        payload = {
            "meta": meta_config,
            "steps": steps_taken,
            "halted": halted,
            "halt_reason": halt_reason,
            "final_dissonance": final_dissonance,
            "dissonance_drop": dissonance_drop,
            "tour_cost": tour_cost,
            "tour_cost_gap": tour_gap,
            "subtour_count": subtours,
            "score": score,
            "meta_drive_scale": float(orchestrator.meta_drive_scale),
            "meta_beta_boost": float(orchestrator.meta_beta_boost),
            "meta_actions": orchestrator.meta_last_actions or {},
            "meta_events": list(orchestrator.meta_events),
            "meta_drive_history": list(orchestrator.meta_drive_history),
            "dissonance_history": dissonance_history,
            "plateau_events": list(orchestrator.plateau_events),
            "lattice_usage": orchestrator.lattice_usage_snapshot or {},
        }
    finally:
        orchestrator.shutdown()

    return payload


def _write_candidate_log(log_dir: Path, idx: int, scenario: str, cfg: Dict[str, float], result: Dict[str, float | int | bool | str]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "candidate_index": idx,
        "scenario": scenario,
        "meta_config": cfg,
        "result": result,
    }
    log_path = log_dir / f"{timestamp}_cand{idx:02d}.yaml"
    with log_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def _write_summary(summary_path: Path, results: List[Dict[str, float | int | bool | str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "score",
        "steps",
        "halted",
        "final_dissonance",
        "tour_cost_gap",
        "subtour_count",
        "meta_drive_scale",
        "meta_beta_boost",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "candidate_index": idx,
                    "score": result.get("score"),
                    "steps": result.get("steps"),
                    "halted": result.get("halted"),
                    "final_dissonance": result.get("final_dissonance"),
                    "tour_cost_gap": result.get("tour_cost_gap"),
                    "subtour_count": result.get("subtour_count"),
                    "meta_drive_scale": result.get("meta_drive_scale"),
                    "meta_beta_boost": result.get("meta_beta_boost"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short TSP meta-control tuning bursts")
    parser.add_argument("--scenario", default="tsp_reflexive_12.yaml", help="Scenario file name")
    parser.add_argument("--scenario-dir", default="scenarios", help="Scenario directory")
    parser.add_argument("--steps", type=int, default=12, help="Max steps per burst")
    parser.add_argument("--workers", type=int, default=9, help="Worker count")
    parser.add_argument("--log-dir", default="runs/meta_tuning_logs", help="Directory for candidate logs")
    parser.add_argument("--summary-csv", help="Optional summary CSV path (defaults inside log dir)")
    args = parser.parse_args()

    scenario_path = Path(args.scenario_dir)
    loader = ScenarioLoader(scenario_path)
    log_dir = Path(args.log_dir)
    summary_path = Path(args.summary_csv) if args.summary_csv else log_dir / "summary.csv"

    beta_gains = [5.5, 6.5]
    sinkhorn_extra = [8, 10]
    perturb_strength = [0.02, 0.03]
    perturb_threshold = [6e-4, 8e-4]

    candidates: List[Dict[str, float]] = []
    for bg, extra, strength, threshold in itertools.product(
        beta_gains, sinkhorn_extra, perturb_strength, perturb_threshold
    ):
        candidates.append(
            {
                "reaction_decay": 0.93,
                "temperature_gain": 0.05,
                "drive_gain": 0.35,
                "beta_gain": bg,
                "sinkhorn_extra_max": extra,
                "perturbation_threshold": threshold,
                "perturbation_cooldown": 140,
                "perturbation_strength": strength,
                "temperature_leak": 0.02,
            }
        )

    results: List[Dict[str, float | int | bool | str]] = []
    for idx, cfg in enumerate(candidates, start=1):
        print(f"[Tuner] Running candidate {idx}/{len(candidates)}: {cfg}")
        candidate_artifacts = log_dir / "artifacts" / f"{Path(args.scenario).stem}_cand{idx:02d}"
        result = run_candidate(loader, args.scenario, cfg, args.steps, args.workers, candidate_artifacts)
        results.append(result)
        _write_candidate_log(log_dir, idx, args.scenario, cfg, result)
        print(
            f"[Tuner]   -> halted={result['halted']} steps={result['steps']} dissonance={result['final_dissonance']:.6f} score={result['score']:.6f}"
        )

    best = min(results, key=lambda item: item["score"]) if results else None
    _write_summary(summary_path, results)
    if best:
        print("\n=== Best configuration ===")
        print(best["meta"])
        print(
            f"Steps={best['steps']} halted={best['halted']} final_dissonance={best['final_dissonance']:.6f} score={best['score']:.6f}"
        )
    else:
        print("\nNo successful candidates were completed before interruption.")
    print(f"Candidate logs written to {log_dir}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
