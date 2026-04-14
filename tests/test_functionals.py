import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delta_machine.functionals import FunctionalCompiler


class FunctionalCompilerTest(unittest.TestCase):
    def setUp(self):
        self.compiler = FunctionalCompiler()

    def test_compile_simple_expression(self):
        kernel = self.compiler.compile(
            name="psi_real",
            expression="psi_real - 0.5 * chi",
            variables=["psi_real", "chi"],
            dependencies=[],
            weight=1.0,
        )
        psi_real = np.ones((4, 4))
        chi = np.full((4, 4), 2.0)
        result = kernel.evaluate({"psi_real": psi_real, "chi": chi})
        np.testing.assert_allclose(result, psi_real - 0.5 * chi)


if __name__ == "__main__":
    unittest.main()

