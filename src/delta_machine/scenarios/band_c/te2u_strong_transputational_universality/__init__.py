"""TE₂.U Strong Transputational Universality instrumentation & datasets.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/dsac_discovery_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.32_dsac_backend_operations_manual_copy.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.34_taichi_nan_parity_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.K_REFLEXIVE_EQUIVALENCE_THEOREM/FINAL_SUMMARY_DISCUSSION.md
"""

from .tpuc_dataset import TPUCScenarioConfig, build_tpuc_benchmark_suite
from .tpuc_metrics import (
    TPUCAdvantageMetrics,
    compute_tpuc_advantage_metrics,
)

__all__ = [
    "TPUCScenarioConfig",
    "TPUCAdvantageMetrics",
    "build_tpuc_benchmark_suite",
    "compute_tpuc_advantage_metrics",
]
