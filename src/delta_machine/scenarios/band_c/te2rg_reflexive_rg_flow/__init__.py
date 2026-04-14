"""TE₂.RG Reflexive RG flow ensemble helpers.

References:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/dsac_discovery_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.32_dsac_backend_operations_manual_copy.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/docs/1.34_taichi_nan_parity_recipe.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/Mathematical_Foundations_of_Reflexive_Reality.tex
"""

from .multiscale_dataset import MultiscaleEnsembleConfig, generate_multiscale_ensemble
from .multiscale_beta_estimator import BetaEstimate, estimate_beta_function

__all__ = [
    "MultiscaleEnsembleConfig",
    "BetaEstimate",
    "generate_multiscale_ensemble",
    "estimate_beta_function",
]
