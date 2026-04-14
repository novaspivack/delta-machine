# DSAC User's Guide for Programmers

**See also**: [dsac_discovery_recipe.md](./dsac_discovery_recipe.md) · [1.32_dsac_backend_operations_manual_copy.md](./1.32_dsac_backend_operations_manual_copy.md)

This guide provides a practical overview for developers replicating experiments or authoring new DSAC scenarios. It covers workflows, APIs, CLI commands, and best practices.

## 1. Installation & Environment
1. **Python environment**: Python ≥3.10 recommended (matching `conda` base on Nova’s machine).
2. **Taichi backend**:
   ```bash
   pip install taichi==1.7.1
   ```
   (Ensure CPU float64 mode is available; GPU path remains experimental.)
3. **Project layout**: clone or sync the `Delta_machine` workspace and add `src/` to `PYTHONPATH` when running scripts:
   ```bash
   export PYTHONPATH="$(pwd)/src:.."
   ```
4. **Optional dependencies**: `numpy`, `scipy`, `pandas`, `matplotlib` for diagnostics; install via `pip install -r requirements.txt` (under construction).

## 2. Key Directories & Modules
- `scenarios/`: YAML files describing DSAC experiments.
- `src/delta_machine/`: core runtime (orchestrator, backends, meta-control, initial conditions).
- `Testing_scripts/`: ensemble runners, dataset generators, parity tests.
- `DSAC_tools/`: analysis utilities (surrogate fitting, flux diagnostics, benchmarks).
- `docs/`: guides (this file, discovery recipe, parity recipes, backend manual).

## 3. Running DSAC Scenarios
### 3.1 CLI Entry Points
Use `python3 -m delta_machine.cli headless --scenario <path>` with optional overrides:
```bash
python3 -m delta_machine.cli headless \
  --scenario scenarios/discovery_phase2/pr0_transport_longrun.yaml \
  --backend taichi --workers 6
```
Flags:
- `--backend {numpy,taichi,auto}`
- `--workers N`
- `--seed SEED`
- `--headless-log-dir runs/...`

### 3.2 Ensemble Scripts
Batch runners automate sampling and summary statistics:
```bash
PYTHONPATH=src python3 Testing_scripts/discovery_phase2/run_pr0_transport_ensemble.py \
  --samples 15 --steps 8000 --output runs/discovery_phase2/pr0_transport_ensemble_longrun_steps8000.csv
```
Common options:
- `--longrun` or `--longrun-scenario`
- `--per-regime-output DIR`
- `--coeff-summary-output FILE`
- `--seed` for reproducibility

## 4. Authoring New Scenarios
1. Start from an existing YAML (e.g. `discovery_phase2/pr0_transport.yaml`).
2. Set `scenario_type`, discovery features, halting criteria, and `constraint_backend` metadata.
3. Update the discovery recipe (`docs/dsac_discovery_recipe.md`) with any special steps.
4. Ensure initial conditions exist (see `src/delta_machine/initial_conditions/generators.py`).

## 5. API Highlights
- **Orchestrator** (`src/delta_machine/orchestrator.py`): `DeltaOrchestrator` manages workers and steps; scenarios are loaded via `ScenarioLoader`.
- **Constraint Backends** (`src/delta_machine/backends/__init__.py`): `NumpyConstraintBackend`, `TaichiConstraintBackend` with configuration knobs (`inner_iterations`, `tile_shape`, `use_vectorized`).
- **Scenario Runner** (`src/delta_machine/scenarios/runner.py`): `_compute_discovery_metrics` handles regression offloading, metrics, flux diagnostics.
- **Initial Conditions** (`src/delta_machine/initial_conditions/generators.py`): `PR0TransportDatasetGenerator.load_sample` for direct dataset access.
- **Analysis tools**: `delta_machine.analysis.transport_flux.compute_flux_diagnostics` for reconstructing source/flux terms.

## 6. Best Practices & Protocols
1. **Parity first**: run `Testing_scripts/run_taichi_parity_smoke.py` before large Taichi ensembles.
2. **Document runs**: store CSVs/JSONs in `runs/` with descriptive names and timestamp directories for reproducibility.
3. **Use discovery recipe**: follow `docs/dsac_discovery_recipe.md` to move systematically from hypothesis to proof.
4. **Taichi tuning**: adjust `inner_iterations`, `tile_shape`, `block_dim`, and `relaxation_rate` carefully; see `docs/1.34_taichi_nan_parity_recipe.md` for the stabilization playbook.

## 7. Architecture Summary
- DSAC wraps a lattice relaxation engine with a two-timescale meta-controller that modulates drive scale, Sinkhorn iterations, and perturbations based on dissonance trends.
- The Taichi backend stages derived quantities (vectorised tiles, shared caches) for performance; fallback NumPy backend remains available for debugging.
- PR-0 interoperability uses shared field bundles (`pr0_system`) to exchange states between DSAC and Rule-110 simulations.

## 8. Creating New Experiments
1. Design objectives and success metrics.
2. Build or adapt scenario YAML/initial conditions.
3. Run smoke, parity, and production suites.
4. Quantify residuals, verification streaks, trimmed errors.
5. Draft proof package and LaTeX inserts.

## 9. Further Reading
- `docs/1.32_dsac_backend_operations_manual_copy.md` — low-level backend configuration.
- `docs/1.34_taichi_nan_parity_recipe.md` — Taichi long-run stability playbook.
- `docs/dsac_discovery_recipe.md` — end-to-end discovery/proof workflow.
