"""
Initial condition registry and loader for saved conditions.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .generators import InitialConditionGenerator, load_generator


class InitialConditionRegistry:
    """Registry for initial condition generators and saved conditions."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.conditions_dir = self.base_dir / "initial_conditions"
        self.conditions_dir.mkdir(parents=True, exist_ok=True)

    def list_conditions(self) -> List[str]:
        """List available initial condition names."""
        if not self.conditions_dir.exists():
            return []
        return [f.stem for f in self.conditions_dir.glob("*.yaml")]

    def save_condition(
        self, name: str, generator: InitialConditionGenerator, metadata: Dict | None = None
    ):
        """Save a generator configuration to YAML."""
        spec = {
            "name": name,
            "generator_type": generator.__class__.__name__.replace("Generator", "").lower(),
            "parameters": self._extract_parameters(generator),
        }
        if metadata:
            spec["metadata"] = metadata

        path = self.conditions_dir / f"{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

    def _extract_parameters(self, generator: InitialConditionGenerator) -> Dict:
        """Extract generator parameters for serialization."""
        params = {}
        if hasattr(generator, "psi_amplitude"):
            params["psi_amplitude"] = generator.psi_amplitude
        if hasattr(generator, "chi_amplitude"):
            params["chi_amplitude"] = generator.chi_amplitude
        if hasattr(generator, "chi_dot_amplitude"):
            params["chi_dot_amplitude"] = generator.chi_dot_amplitude
        if hasattr(generator, "amplitude"):
            params["amplitude"] = generator.amplitude
        if hasattr(generator, "sigma"):
            params["sigma"] = generator.sigma
        if hasattr(generator, "center"):
            params["center"] = generator.center
        if hasattr(generator, "vorticity"):
            params["vorticity"] = generator.vorticity
        if hasattr(generator, "width"):
            params["width"] = generator.width
        if hasattr(generator, "position"):
            params["position"] = generator.position
        if hasattr(generator, "charge"):
            params["charge"] = generator.charge
        if hasattr(generator, "vortex_strength"):
            params["vortex_strength"] = generator.vortex_strength
        if hasattr(generator, "noise_level"):
            params["noise_level"] = generator.noise_level
        return params

    def load_condition(self, name: str) -> InitialConditionGenerator:
        """Load a saved initial condition generator."""
        path = self.conditions_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Initial condition '{name}' not found at {path}")

        with open(path) as f:
            spec = yaml.safe_load(f)

        generator_type = spec["generator_type"]
        parameters = spec.get("parameters", {})
        return load_generator(generator_type, **parameters)


def load_initial_condition(
    base_dir: Path, name_or_generator: str | Dict
) -> InitialConditionGenerator:
    """
    Load an initial condition from registry or generator spec.

    Args:
        base_dir: Base directory for initial conditions
        name_or_generator: Either a condition name (str) or generator spec (dict)

    Returns:
        InitialConditionGenerator instance
    """
    if isinstance(name_or_generator, str):
        registry = InitialConditionRegistry(base_dir)
        return registry.load_condition(name_or_generator)
    elif isinstance(name_or_generator, dict):
        generator_type = name_or_generator.get("type", "random")
        parameters = name_or_generator.get("parameters", {})
        return load_generator(generator_type, **parameters)
    else:
        raise ValueError(f"Invalid initial condition spec: {name_or_generator}")

