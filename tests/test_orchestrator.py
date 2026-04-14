import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delta_machine.analysis import get_tsp_config
from delta_machine.config import ScenarioLoader
from delta_machine.functionals import FunctionalCompiler
from delta_machine.orchestrator import DeltaOrchestrator
from delta_machine.initial_conditions import load_initial_condition
from delta_machine.scenarios import ScenarioRunner


class OrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.scenario_dir = Path(__file__).resolve().parents[1] / "scenarios"
        loader = ScenarioLoader(self.scenario_dir)
        self.scenario = loader.load("basic_dsac.yaml")

    def test_start_step_shutdown(self):
        orchestrator = DeltaOrchestrator(self.scenario, FunctionalCompiler(), max_workers=2)
        orchestrator.start_workers(worker_count=2)
        orchestrator.step()
        telemetry = orchestrator.telemetry()
        self.assertGreaterEqual(telemetry.total_dissonance, 0.0)
        orchestrator.shutdown()

    def test_sat_metrics_solution_verified(self):
        loader = ScenarioLoader(self.scenario_dir)
        sat_spec = loader.load("reflexive_sat.yaml")

        runner = ScenarioRunner(sat_spec)
        shape = sat_spec.lattice_shape
        arrays = {
            "psi_real": np.full(shape, -2.0, dtype=np.float64),
            "psi_imag": np.full(shape, -2.0, dtype=np.float64),
            "chi": np.full(shape, 2.0, dtype=np.float64),
            "chi_dot": np.zeros(shape, dtype=np.float64),
        }

        runner.update_metrics(step=0, dissonance=0.0, residual_norms=None, arrays=arrays)
        metrics = runner.scenario_metrics

        self.assertLess(metrics["max_clause_deficit"], sat_spec.metadata.get("clause_tolerance", 0.01))
        self.assertEqual(metrics["assignment_satisfied"], 1.0)
        self.assertEqual(metrics["solution_verified"], 1.0)

    def test_weighted_max_sat_convergence(self):
        loader = ScenarioLoader(self.scenario_dir)
        weighted_spec = loader.load("weighted_max_sat.yaml")
        orchestrator = DeltaOrchestrator(weighted_spec, FunctionalCompiler(), max_workers=2)
        ic_spec = weighted_spec.initial_condition_refs[0]
        initial_condition = load_initial_condition(Path(self.scenario_dir).parent, ic_spec)
        orchestrator.initial_condition = initial_condition
        if isinstance(ic_spec, dict):
            orchestrator.initial_condition_name = ic_spec.get("name", ic_spec.get("type", "generated"))
        orchestrator.initial_condition_seed = 1076251544
        orchestrator.start_workers(worker_count=2)
        for _ in range(weighted_spec.halting_criteria.max_steps):
            orchestrator.step()
            if orchestrator.halted:
                break
        self.assertTrue(orchestrator.halted)
        metrics = orchestrator.scenario_runner.scenario_metrics if orchestrator.scenario_runner else {}
        target_weight = weighted_spec.metadata.get("target_weight", 0.0)
        self.assertGreaterEqual(metrics.get("weighted_score", 0.0), target_weight - 0.05)
        orchestrator.shutdown()

    def test_constraint_discovery_metrics(self):
        loader = ScenarioLoader(self.scenario_dir)
        discovery_spec = loader.load("constraint_discovery.yaml")
        runner = ScenarioRunner(discovery_spec)
        shape = discovery_spec.lattice_shape
        rng = np.random.default_rng(123)
        psi_real = rng.normal(0.0, 0.5, size=shape)
        psi_imag = rng.normal(0.0, 0.4, size=shape)
        true_alpha = discovery_spec.metadata.get("discovery_model", {}).get("alpha", 0.2)
        true_beta = discovery_spec.metadata.get("discovery_model", {}).get("beta", 0.2)
        true_gamma = discovery_spec.metadata.get("discovery_model", {}).get("gamma", 0.0)
        chi = true_alpha * psi_real ** 2 + true_beta * psi_imag ** 2 + true_gamma
        arrays = {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": np.zeros(shape, dtype=np.float64),
        }

        for step in range(60):
            runner.update_metrics(step, dissonance=0.01, arrays=arrays)

        metrics = runner.scenario_metrics
        self.assertLess(metrics["coefficient_error"], discovery_spec.metadata.get("coefficient_tolerance", 0.05))
        self.assertLess(metrics["residual_rms"], discovery_spec.metadata.get("residual_tolerance", 0.05))
        self.assertEqual(metrics["discovery_verified"], 1.0)

    def test_metric_closure_metrics(self):
        loader = ScenarioLoader(self.scenario_dir)
        closure_spec = loader.load("metric_closure.yaml")
        runner = ScenarioRunner(closure_spec)
        shape = closure_spec.lattice_shape

        closure_spec.metadata.setdefault("closure_model", {})
        closure_spec.metadata["closure_model"].update({
            "chi_scale": 0.0,
            "chi_offset": 0.0,
            "curl_alignment": 0.0,
            "divergence_target": 0.0,
            "laplacian_target": 0.0,
        })

        y = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
        x = np.linspace(0, 2 * np.pi, shape[1], endpoint=False)
        X, Y = np.meshgrid(x, y)
        stream = np.sin(X) * np.sin(Y)
        psi_real = np.gradient(stream, axis=0)  # ∂ψ/∂y
        psi_imag = -np.gradient(stream, axis=1)  # -∂ψ/∂x
        chi = np.gradient(psi_imag, axis=1) - np.gradient(psi_real, axis=0)

        arrays = {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": np.zeros(shape, dtype=np.float64),
        }

        for step in range(100):
            runner.update_metrics(step, dissonance=0.001, arrays=arrays)

        metrics = runner.scenario_metrics
        self.assertTrue(np.isfinite(metrics["closure_error"]))
        self.assertTrue(np.isfinite(metrics["divergence_rms"]))
        self.assertTrue(np.isfinite(metrics["chi_alignment_rms"]))
        self.assertIn("closure_verified", metrics)

    def test_tsp_metrics_stochasticity(self):
        loader = ScenarioLoader(self.scenario_dir)
        tsp_spec = loader.load("tsp_reflexive.yaml")
        runner = ScenarioRunner(tsp_spec)

        config = get_tsp_config(tsp_spec.metadata or {})
        shape = tsp_spec.lattice_shape
        arrays = {
            "psi_real": np.zeros(shape, dtype=np.float64),
            "psi_imag": np.zeros(shape, dtype=np.float64),
            "chi": np.zeros(shape, dtype=np.float64),
            "chi_dot": np.zeros(shape, dtype=np.float64),
        }

        uniform = np.full((config.num_cities, config.num_cities), 1.0 / config.num_cities)
        arrays["psi_real"][config.row_slice, config.col_slice] = uniform
        arrays["psi_imag"][config.row_slice, config.col_slice] = uniform

        runner.update_metrics(step=10, dissonance=0.1, arrays=arrays)
        metrics = runner.scenario_metrics

        self.assertLess(metrics["doubly_stochastic_error"], 1e-9)
        self.assertEqual(metrics["stochasticity_verified"], 1.0)


if __name__ == "__main__":
    unittest.main()

