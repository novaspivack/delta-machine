# Δ-Machine: Differential Self-Adjudicative Computation (DSAC)

**Repository**: https://github.com/novaspivack/delta-machine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19429884.svg)](https://doi.org/10.5281/zenodo.19429884)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

DSAC is a continuous-field, reflexive computing architecture for law discovery, constraint satisfaction, and reflexive substrates. It implements the Δ-Computing paradigm and serves as the candidate computational realization of **transputation** as defined in the NEMS formal research program.

## Companion Paper

**DSAC as a Realization of Transputation** (Paper 77 of the Reflexive Reality suite)  
Nova Spivack · Zenodo: [10.5281/zenodo.19429884](https://doi.org/10.5281/zenodo.19429884)

This paper formally maps the Δ-machine architecture to the six transputation realization criteria of Paper 76, explains the relaxation-to-coherence mechanism (finding a self-consistent fixed point of a reflexive constraint system), and documents computational validation across five scenario families.

**Formal theory of transputation**: Paper 76 · Zenodo: [10.5281/zenodo.19429882](https://doi.org/10.5281/zenodo.19429882)  
**Lean formalization**: [transputation-lean](https://github.com/novaspivack/transputation-lean)  
**Full NEMS program**: [novaspivack.com/research](https://www.novaspivack.com/research)

## Installation

```bash
git clone https://github.com/novaspivack/delta-machine.git
cd delta-machine
pip install -r requirements.txt

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Quick Start

```bash
# Headless run with a discovery scenario
python -m delta_machine.cli headless --scenario scenarios/discovery_phase1/polynomial_law.yaml --steps 4000

# GUI (requires PySide6)
python -m delta_machine.app
```

## Optional: PR-0 Integration

> ⚠️ **The `pr0_system` package is not publicly available.** The PR-0 field-state backend is an optional enhancement; all Paper 77 validation results can be reproduced without it using the default NumPy backend.

For PR-0 field-state backends and transport discovery (if you have access to `pr0_system`):

1. Clone `pr0_system` alongside or set `PR0_SYSTEM_ROOT` / `PYTHONPATH` to include it.
2. Enable via `DELTA_USE_PR0_FIELDSTATE=1` or set `use_pr0_field_state: true` in scenario metadata.
3. Run scenarios with `--use-pr0-field-state` flag.

## Backends

- **NumPy** (default): Reference implementation.
- **Taichi**: High-performance backend for discovery workloads. Use `--backend taichi` or set `constraint_backend.type: taichi` in scenario metadata.

## Documentation

### Validation & Benchmarks

> **→ See [REPRODUCE.md](REPRODUCE.md) for step-by-step commands to replicate the Paper 77 validation results.**

The `Testing_scripts/` directory contains reproducible validation harnesses:

| Script | Purpose |
|--------|---------|
| `Testing_scripts/tsp_baseline.py` | TSP benchmarks vs brute-force and OR-Tools (8–32 cities) |
| `Testing_scripts/run_discovery_benchmark_suite.py` | Constraint discovery benchmark suite |
| `Testing_scripts/run_ortools_benchmarks.py` | OR-Tools comparison baselines |

Validated scenario families (run with `python -m delta_machine.cli headless --scenario <file>`):

| Scenario | Result |
|----------|--------|
| `scenarios/reflexive_sat.yaml` | Boolean SAT without branching, ≤300 steps, assignment verified |
| `scenarios/weighted_max_sat.yaml` | Max-SAT 14.5/15.5, convergence 36 steps |
| `scenarios/constraint_discovery.yaml` | Latent relation recovery, residual RMS ~10⁻¹⁶; extended runs recover physical law forms |
| `scenarios/metric_closure.yaml` | Curvature invariants maintained ≥80 steps |
| `scenarios/tsp_reflexive_*.yaml` | 8–32 cities, verified zero-gap optimal tours, flat ~6-step convergence |
| `scenarios/tsp_tsplib_eil51.yaml` | TSPLIB benchmark |

Run dossiers with full logs, field snapshots, and metrics are stored under `runs/`.

### Practical Guides

- [docs/dsac_users_guide.md](docs/dsac_users_guide.md) — Installation, CLI, authoring scenarios, API, best practices
- [docs/dsac_discovery_recipe.md](docs/dsac_discovery_recipe.md) — Discovery-to-proof workflow
- [docs/1.32_dsac_backend_operations_manual_copy.md](docs/1.32_dsac_backend_operations_manual_copy.md) — Backend configuration (Taichi, NumPy)

## Citation

```bibtex
@misc{Spivack2026-77,
  author    = {Spivack, Nova},
  title     = {{DSAC} as a Realization of Transputation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19429884},
  url       = {https://doi.org/10.5281/zenodo.19429884}
}
```
