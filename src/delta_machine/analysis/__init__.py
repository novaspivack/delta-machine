"""Analysis utilities for Δ-Machine experiments.

Design references:
- 1.5 DSAC Scenario Roadmap:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.5_dsac_scenario_roadmap.md
"""

from .metric_closure import compute_metric_closure
from .tsp_utils import (
    TSPConfig,
    apply_sinkhorn_step,
    compute_tsp_metrics,
    embed_assignment_matrix,
    extract_assignment_matrix,
    get_tsp_config,
)

__all__ = [
    "compute_metric_closure",
    "TSPConfig",
    "apply_sinkhorn_step",
    "compute_tsp_metrics",
    "embed_assignment_matrix",
    "extract_assignment_matrix",
    "get_tsp_config",
]
