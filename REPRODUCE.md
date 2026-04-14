# Reproducing the Paper 77 Validation Results

This document gives exact commands to reproduce the five scenario families
documented in **DSAC as a Realization of Transputation** (Paper 77).

DOI: [10.5281/zenodo.19429884](https://doi.org/10.5281/zenodo.19429884)

---

## Setup

```bash
git clone https://github.com/novaspivack/delta-machine.git
cd delta-machine
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Python 3.10+ recommended. OR-Tools is required for TSP baseline comparisons
(`pip install ortools`). Taichi backend is optional (`pip install taichi`).

---

## 1. Reflexive SAT

**Scenario:** `scenarios/reflexive_sat.yaml`  
**Expected:** `solution_verified = 1`, convergence in ≤300 steps, `max_clause_deficit < 0.01`

```bash
python -m delta_machine.cli headless \
  --scenario scenarios/reflexive_sat.yaml \
  --seed 42
```

Representative original result: 59 steps, `max_clause_deficit ≈ 9.54×10⁻³`, assignment verified by classical checker. Results are stochastic across random seeds; success condition (`satisfied_clauses >= target`) should trigger reliably within 300 steps.

---

## 2. Weighted Max-SAT

**Scenario:** `scenarios/weighted_max_sat.yaml`  
**Expected:** `weighted_score ≈ 14.5/15.5`, convergence in ~36 steps

```bash
python -m delta_machine.cli headless \
  --scenario scenarios/weighted_max_sat.yaml \
  --seed 42
```

---

## 3. Constraint Discovery

**Scenario:** `scenarios/constraint_discovery.yaml`  
**Expected:** latent relation `χ ≈ 0.2ψ_real² + 0.2ψ_imag²` recovered;
`discovered_alpha ≈ 0.2`, `discovered_beta ≈ 0.2`, residual RMS ≈ 1.1×10⁻¹⁶ at 50 steps

```bash
python -m delta_machine.cli headless \
  --scenario scenarios/constraint_discovery.yaml \
  --seed 42
```

Extended discovery runs (Phases 1–2) that recover physical law forms are in
`Testing_scripts/discovery_phase1/` and `Testing_scripts/discovery_phase2/`:

```bash
# Polynomial law rediscovery
python Testing_scripts/discovery_phase1/run_polynomial_ensemble.py

# Jarzynski-type thermodynamic relation rediscovery
python Testing_scripts/discovery_phase1/run_te1_jarzynski_ensemble.py

# PR-0 transport flux rediscovery
python Testing_scripts/discovery_phase1/run_pr0_flux_ensemble.py

# Full discovery benchmark suite
python Testing_scripts/run_discovery_benchmark_suite.py
```

---

## 4. Metric Closure

**Scenario:** `scenarios/metric_closure.yaml`  
**Expected:** closure streak ≥ 80 steps, `pattern_complexity ≈ 0.27`, curvature invariants within tolerance

```bash
python -m delta_machine.cli headless \
  --scenario scenarios/metric_closure.yaml \
  --seed 42
```

---

## 5. Reflexive TSP

**Scenarios:** `scenarios/tsp_reflexive*.yaml`, `scenarios/tsp_tsplib_eil51.yaml`  
**Expected:** `tour_verified = 1`, `tour_cost_gap = 0.0` (exact optimal), convergence in ~6 steps

```bash
# 8-city (exact vs brute-force)
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_reflexive.yaml \
  --seed 42

# 10-city
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_reflexive_10.yaml \
  --seed 42

# 12-city
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_reflexive_12.yaml \
  --seed 42

# 16-city
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_reflexive_16.yaml \
  --seed 42

# 32-city perturbed
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_reflexive_perturbed_32.yaml \
  --seed 42

# TSPLIB eil51
python -m delta_machine.cli headless \
  --scenario scenarios/tsp_tsplib_eil51.yaml \
  --seed 42
```

**Baseline comparison** (DSAC vs brute-force vs OR-Tools):

```bash
python Testing_scripts/tsp_baseline.py --scenario scenarios/tsp_reflexive.yaml
python Testing_scripts/tsp_baseline.py --scenario scenarios/tsp_reflexive_12.yaml --ortools
```

### Representative TSP Results (from original runs)

| Scenario | Steps | `tour_verified` | `tour_cost_gap` | `doubly_stochastic_error` |
|----------|-------|-----------------|-----------------|--------------------------|
| 8-city   | 18    | 1               | 0.000           | 1.7×10⁻²                |
| 10-city  | 6     | 1               | 0.000           | 1.9×10⁻⁴                |
| 12-city  | 6     | 1               | 0.000           | 4.8×10⁻⁴                |
| 16-city  | 6     | 1               | 0.000           | 4.4×10⁻⁴                |
| 32-city perturbed | 13 | 1          | —               | —                        |

Original run dossiers are stored under `runs/` with timestamped directories
containing `report.yaml`, field snapshots, and logs.

---

## Notes on Reproducibility

- **Stochasticity:** DSAC uses random initial conditions by default. The `--seed`
  flag seeds NumPy RNG for reproducibility. Exact step counts may vary slightly
  across seeds; the success condition (e.g., `tour_verified = 1`) should be
  robust.
- **Workers:** TSP scenarios use multi-process workers (`--workers N`). Default
  is system-dependent; for exact replication of original results, set
  `--workers 9` or use the `workers` field in scenario metadata.
- **PR-0 backend (not publicly available):** Several original validation runs
  used the PR-0 field-state backend (`--use-pr0-field-state`), which depends
  on the `pr0_system` package. That package has not been published and is not
  available in this repository. **Do not use `--use-pr0-field-state`** when
  reproducing; the default NumPy backend is sufficient to reproduce all core
  Paper 77 validation results. The PR-0 integration was an optional enhancement
  to the original runs; the SAT, constraint discovery, metric closure, and TSP
  results reported in the paper do not require it.
- **Hardware:** Original runs were on a macOS workstation (Apple Silicon).
  Results should be qualitatively reproducible on any platform with
  sufficient RAM (≥8 GB for 32-city+).

---

## Related Documentation

- `docs/dsac_users_guide.md` — full CLI reference
- `docs/dsac_discovery_recipe.md` — discovery workflow
