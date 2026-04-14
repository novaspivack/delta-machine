"""TE₂.H instrumentation and dataset tooling.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/dsac_discovery_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.32_dsac_backend_operations_manual_copy.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.34_taichi_nan_parity_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.Y_COMPUTATIONAL_HOLOGRAPHY/TE_1.Y_COMPTUTATIONAL_HOLOGRAPHY_THMS.md
"""

from .boundary_bulk_dataset import build_boundary_bulk_dataset, BoundaryBulkConfig
from .boundary_bulk_instrumentation import (
    BoundaryBulkMetrics,
    compute_boundary_bulk_metrics,
)

__all__ = [
    "BoundaryBulkConfig",
    "BoundaryBulkMetrics",
    "build_boundary_bulk_dataset",
    "compute_boundary_bulk_metrics",
]
