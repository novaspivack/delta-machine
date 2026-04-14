"""Phase III Band C scenario tooling package.

References:
- notes/1.38_phase3_bandc_experiments_plan.md
- docs/dsac_discovery_recipe.md
- docs/1.32_dsac_backend_operations_manual_copy.md
- docs/1.34_taichi_nan_parity_recipe.md
"""

from importlib import import_module
from types import ModuleType

__all__ = ["load_submodule"]


def load_submodule(name: str) -> ModuleType:
    """Import a band_c submodule by name.

    Parameters
    ----------
    name:
        Relative submodule path (e.g. "te2h_reflexive_holographic_equivalence.boundary_bulk_dataset").

    Returns
    -------
    ModuleType
        Imported module instance.
    """

    full_name = f"delta_machine.scenarios.band_c.{name}"
    module = import_module(full_name)
    return module
