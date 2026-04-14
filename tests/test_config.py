import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delta_machine.config import ScenarioLoader


class ScenarioLoaderTest(unittest.TestCase):
    def setUp(self):
        self.scenario_dir = Path(__file__).resolve().parents[1] / "scenarios"

    def test_load_basic_scenario(self):
        loader = ScenarioLoader(self.scenario_dir)
        scenario = loader.load("basic_dsac.yaml")
        self.assertEqual(scenario.name, "Basic DSAC Equilibration")
        self.assertEqual(scenario.lattice_shape, (64, 64))
        self.assertEqual(len(scenario.constraints), 4)


if __name__ == "__main__":
    unittest.main()

