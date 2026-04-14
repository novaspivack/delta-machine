# DSAC Discovery-to-Proof Recipe

**See also**: [dsac_users_guide.md](./dsac_users_guide.md) · [1.32_dsac_backend_operations_manual_copy.md](./1.32_dsac_backend_operations_manual_copy.md)

This recipe captures the workflow for taking a DSAC-driven experiment from initial hypothesis through empirical validation, statistical quantification, analytic proof, and manuscript integration. It has been applied across Phase I rediscoveries, Phase II curvature invariant, and Phase II PR-0 transport workstreams.

## 1. Frame the Experiment
1. Document the scientific question, expected observables, and success metrics.
2. Ensure data pipelines are prepared (e.g. `Testing_scripts/discovery_phase2/generate_pr0_transport_dataset.py` for PR-0 transport datasets).

## 2. Scenario Authoring & Configuration
1. Create or clone a DSAC scenario YAML under `scenarios/` with:
   - Discovery features / target expressions.
   - Constraint backend metadata (`taichi` knobs, precision, fusion).
   - Halting criteria / verification streaks.
2. Commit accompanying dataset generator or loader entries (see `src/delta_machine/initial_conditions/generators.py`).

## 3. Calibration & Smoke Testing
1. Run short (`≤4000` step) smoke ensembles via `Testing_scripts/.../run_*_ensemble.py` with `--samples` limited to a handful of seeds.
2. Use `Testing_scripts/run_taichi_parity_smoke.py` to verify NumPy vs Taichi parity.
3. Log preliminary metrics; note any outstanding issues for follow-up.

## 4. Production Ensembles & Statistical Quantification
1. Execute long-run ensembles (e.g. 15×8000 steps) capturing CSVs in `runs/discovery_phase*/`.
2. Record residual RMS, verification streaks, trimmed MAPE (or analogous robustness metrics).
3. Generate coefficient summaries or diagnostic CSVs (e.g. `pr0_transport_longrun_steps8000_coeff_summary.json`, `pr0_transport_flux_diagnostics_samples.csv`).

## 5. Offloaded Regression & Diagnostics
1. Enable DSAC’s offloaded regression path (`discovery_fit_interval`, `discovery_ridge_lambda`) for stability.
2. For transport-like laws, run flux decomposition tools (e.g. `DSAC_tools/analyze_pr0_transport_flux.py`).
3. Capture supporting tables/plots for manuscripts or appendices.

## 6. Analytic Proof Packaging
1. Draft a proof package containing:
   - Microscopic derivation (PR-0 update law).
   - Continuum or analytic interpretation (TE₁ analogies).
   - Theorem statement with explicit bounds.
2. Track integration tasks referencing the proof and any LaTeX fragments for manuscript insertion.

## 7. Manuscript Integration
1. Compose LaTeX fragments for the relevant manuscript chapter.
2. Ensure the integration checklist has aligned TODOs for manuscript insertion.

## 8. Documentation & Operational Updates
1. Refresh `docs/1.32_dsac_backend_operations_manual_copy.md` with any new knobs, scripts, or diagnostics.
2. Cross-link this recipe from the user guide as needed.

## 9. Closure & Next Actions
1. Archive results and scripts with hashes in the computational artifact manifest.
2. Decide the next experiment and repeat the recipe.
