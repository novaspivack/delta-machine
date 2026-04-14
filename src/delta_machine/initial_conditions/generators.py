"""
Initial condition generators for field initialization.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.4 Implementation & Validation Update:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.4_delta_machine_implementation_update.md
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple
import textwrap

import numpy as np
from scipy.ndimage import gaussian_filter


class InitialConditionGenerator(ABC):
    """Base class for initial condition generators."""

    @abstractmethod
    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate initial conditions for all fields.

        Args:
            lattice_shape: (L_y, L_x) tuple
            seed: Optional random seed for reproducibility

        Returns:
            Dictionary mapping field names to initial arrays
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return generator name."""
        pass


class RandomGenerator(InitialConditionGenerator):
    """Random noise initial conditions with configurable density."""

    def __init__(
        self,
        psi_amplitude: float = 0.1,
        chi_amplitude: float = 0.05,
        chi_dot_amplitude: float = 0.01,
    ):
        self.psi_amplitude = psi_amplitude
        self.chi_amplitude = chi_amplitude
        self.chi_dot_amplitude = chi_dot_amplitude

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        L_y, L_x = lattice_shape
        return {
            "psi_real": rng.normal(0.0, self.psi_amplitude, size=(L_y, L_x)),
            "psi_imag": rng.normal(0.0, self.psi_amplitude, size=(L_y, L_x)),
            "chi": rng.normal(0.0, self.chi_amplitude, size=(L_y, L_x)),
            "chi_dot": rng.normal(0.0, self.chi_dot_amplitude, size=(L_y, L_x)),
        }

    def get_name(self) -> str:
        return f"random_psi{self.psi_amplitude:.2f}_chi{self.chi_amplitude:.2f}"


class GaussianGenerator(InitialConditionGenerator):
    """Gaussian bump centered on lattice."""

    def __init__(
        self,
        amplitude: float = 1.0,
        sigma: float | None = None,
        center: Tuple[float, float] | None = None,
    ):
        self.amplitude = amplitude
        self.sigma = sigma
        self.center = center

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        L_y, L_x = lattice_shape
        y = np.linspace(0, L_y - 1, L_y)
        x = np.linspace(0, L_x - 1, L_x)
        X, Y = np.meshgrid(x, y)

        center_x, center_y = self.center or (L_x / 2.0, L_y / 2.0)
        sigma = self.sigma or (min(L_x, L_y) / 6.0)

        gaussian = self.amplitude * np.exp(
            -(((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2))
        )

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.03 * self.amplitude, size=gaussian.shape)

        return {
            "psi_real": gaussian + noise,
            "psi_imag": 0.6 * gaussian + noise,
            "chi": 0.3 * gaussian + rng.normal(0.0, 0.015 * self.amplitude, size=gaussian.shape),
            "chi_dot": rng.normal(0.0, 0.01 * self.amplitude, size=gaussian.shape),
        }

    def get_name(self) -> str:
        return f"gaussian_amp{self.amplitude:.2f}"


class VortexGenerator(InitialConditionGenerator):
    """Vortex pattern with Gaussian envelope."""

    def __init__(
        self,
        amplitude: float = 1.0,
        vorticity: int = 1,
        sigma: float | None = None,
    ):
        self.amplitude = amplitude
        self.vorticity = vorticity
        self.sigma = sigma

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        L_y, L_x = lattice_shape
        y = np.linspace(0, L_y - 1, L_y)
        x = np.linspace(0, L_x - 1, L_x)
        X, Y = np.meshgrid(x, y)

        center_x, center_y = L_x / 2.0, L_y / 2.0
        sigma = self.sigma or (min(L_x, L_y) / 6.0)

        R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        theta = np.arctan2(Y - center_y, X - center_x)

        envelope = np.exp(-(R ** 2) / (2 * sigma ** 2))
        phase = self.vorticity * theta

        psi_real = self.amplitude * envelope * np.cos(phase)
        psi_imag = self.amplitude * envelope * np.sin(phase)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.03 * self.amplitude, size=psi_real.shape)

        return {
            "psi_real": psi_real + noise,
            "psi_imag": psi_imag + noise,
            "chi": 0.3 * envelope + rng.normal(0.0, 0.015 * self.amplitude, size=psi_real.shape),
            "chi_dot": rng.normal(0.0, 0.01 * self.amplitude, size=psi_real.shape),
        }

    def get_name(self) -> str:
        return f"vortex_v{self.vorticity}_amp{self.amplitude:.2f}"


class SolitonGenerator(InitialConditionGenerator):
    """Soliton-like localized excitation."""

    def __init__(
        self,
        amplitude: float = 2.0,
        width: float = 3.0,
        position: Tuple[float, float] | None = None,
        charge: int = 1,
    ):
        self.amplitude = amplitude
        self.width = width
        self.position = position
        self.charge = charge

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        L_y, L_x = lattice_shape
        y = np.linspace(0, L_y - 1, L_y)
        x = np.linspace(0, L_x - 1, L_x)
        X, Y = np.meshgrid(x, y)

        pos_x, pos_y = self.position or (L_x / 2.0, L_y / 2.0)
        R = np.sqrt((X - pos_x) ** 2 + (Y - pos_y) ** 2)

        profile = self.amplitude / np.cosh(R / self.width)
        phase = self.charge * np.arctan2(Y - pos_y, X - pos_x)

        psi_real = profile * np.cos(phase)
        psi_imag = profile * np.sin(phase)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.02 * self.amplitude, size=psi_real.shape)

        return {
            "psi_real": psi_real + noise,
            "psi_imag": psi_imag + noise,
            "chi": 0.2 * profile + rng.normal(0.0, 0.01 * self.amplitude, size=psi_real.shape),
            "chi_dot": rng.normal(0.0, 0.01 * self.amplitude, size=psi_real.shape),
        }

    def get_name(self) -> str:
        return f"soliton_q{self.charge}_amp{self.amplitude:.2f}"


class PatternGenerator(InitialConditionGenerator):
    """Structured pattern (Gaussian + vortex + noise)."""

    def __init__(
        self,
        amplitude: float = 1.0,
        vortex_strength: float = 0.7,
        noise_level: float = 0.03,
    ):
        self.amplitude = amplitude
        self.vortex_strength = vortex_strength
        self.noise_level = noise_level

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        L_y, L_x = lattice_shape
        y = np.linspace(0, L_y - 1, L_y)
        x = np.linspace(0, L_x - 1, L_x)
        X, Y = np.meshgrid(x, y)

        center_x, center_y = L_x / 2.0, L_y / 2.0
        sigma = min(L_x, L_y) / 6.0

        gaussian = np.exp(-(((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2)))
        vortex = np.sin(2 * np.pi * X / L_x + np.pi / 4) * np.cos(2 * np.pi * Y / L_y)
        combined = self.amplitude * gaussian * (1 + self.vortex_strength * vortex)

        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.noise_level * self.amplitude, size=combined.shape)

        return {
            "psi_real": combined + noise,
            "psi_imag": 0.6 * combined + noise,
            "chi": 0.3 * combined + rng.normal(0.0, 0.015 * self.amplitude, size=combined.shape),
            "chi_dot": rng.normal(0.0, 0.01 * self.amplitude, size=combined.shape),
        }

    def get_name(self) -> str:
        return f"pattern_amp{self.amplitude:.2f}_v{self.vortex_strength:.2f}"


class PolynomialDatasetGenerator(InitialConditionGenerator):
    """Synthetic dataset generator for polynomial/log discovery scenarios.

    Design reference:
    - 1.20 Scientific Discovery Validation Plan:
      /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.20_scientific_discovery_validation_plan.md
    """

    _SAFE_FUNCS = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "tanh": np.tanh,
        "arctan": np.arctan,
    }

    def __init__(
        self,
        psi_amplitude: float = 1.0,
        spectral_components: int = 5,
        noise_level: float = 0.05,
        chi_noise: float = 0.005,
    ):
        self.psi_amplitude = psi_amplitude
        self.spectral_components = max(1, int(spectral_components))
        self.noise_level = noise_level
        self.chi_noise = chi_noise

    def _compose_field(
        self,
        lattice_shape: Tuple[int, int],
        rng: np.random.Generator,
        amplitude: float,
        components: int,
    ) -> np.ndarray:
        L_y, L_x = lattice_shape
        y = np.linspace(0.0, 1.0, L_y)
        x = np.linspace(0.0, 1.0, L_x)
        X, Y = np.meshgrid(x, y)
        field = np.zeros_like(X)
        for _ in range(max(1, components)):
            freq_x = rng.integers(1, 6)
            freq_y = rng.integers(1, 6)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            weight = rng.uniform(0.4, 1.2)
            mode = np.sin(2.0 * np.pi * freq_x * X + phase) * np.cos(2.0 * np.pi * freq_y * Y - phase)
            field += weight * mode
        return (amplitude / max(1, components)) * field

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        cfg = (metadata or {}).get("polynomial_dataset", {})
        amp = float(cfg.get("psi_amplitude", self.psi_amplitude))
        components = int(cfg.get("spectral_components", self.spectral_components))
        noise = float(cfg.get("noise_level", self.noise_level))
        chi_noise = float(cfg.get("chi_noise", self.chi_noise))

        psi_real = self._compose_field(lattice_shape, rng, amp, components)
        psi_imag = self._compose_field(lattice_shape, rng, amp * 0.9, components + 1)
        psi_real += rng.normal(0.0, noise * amp, size=lattice_shape)
        psi_imag += rng.normal(0.0, noise * amp, size=lattice_shape)

        truth_expr = (metadata or {}).get("discovery_truth_expression")
        if not truth_expr:
            raise ValueError("polynomial_dataset generator requires discovery_truth_expression in metadata")
        expr = textwrap.dedent(str(truth_expr)).strip()
        if expr and (not expr.startswith("(") or not expr.endswith(")")):
            expr = f"({expr})"
        env = {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
        }
        env.update(self._SAFE_FUNCS)
        try:
            chi_truth = np.asarray(eval(expr, {"__builtins__": {}}, env))
        except Exception as exc:  # pragma: no cover - eval guard
            raise ValueError("Failed to evaluate discovery_truth_expression") from exc
        if chi_truth.shape == ():
            chi_truth = np.full_like(psi_real, float(chi_truth))
        elif chi_truth.shape != psi_real.shape:
            chi_truth = np.broadcast_to(chi_truth, psi_real.shape)

        chi = chi_truth + rng.normal(0.0, chi_noise * amp, size=lattice_shape)
        chi_dot = np.zeros_like(chi)

        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
        }

    def get_name(self) -> str:
        return "polynomial_dataset"


class TE1JarzynskiDatasetGenerator(InitialConditionGenerator):
    """Load Jarzynski ensemble statistics from TE₁.B results into the lattice.

    Design reference:
    - TE₁.B Reflexive Statistical Mechanics:
      /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.B_RSM
    """

    def __init__(self, data_path: str, lattice_size: int | None = None):
        self.data_path = Path(data_path)
        self.lattice_size = lattice_size

    def _load_entries(self) -> list[dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"TE1 Jarzynski dataset not found: {self.data_path}")
        with self.data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(f"TE1 Jarzynski dataset must be a non-empty list: {self.data_path}")
        return data

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        entries = self._load_entries()
        L_y, L_x = lattice_shape
        capacity = L_y * L_x
        if len(entries) > capacity:
            raise ValueError(
                f"Lattice capacity {capacity} insufficient for {len(entries)} TE1 entries; increase lattice size"
            )

        psi_real = np.zeros((L_y, L_x), dtype=float)
        psi_imag = np.zeros_like(psi_real)
        chi = np.zeros_like(psi_real)
        chi_dot = np.zeros_like(psi_real)

        for idx, entry in enumerate(entries):
            row = idx // L_x
            col = idx % L_x
            temperature = float(entry.get("temperature", 0.0))
            sigma = float(entry.get("sigma", 0.0))
            mu = float(entry.get("mu", 0.0))
            je_mean = float(entry.get("je_mean", 1.0))
            je_mean = max(je_mean, 1e-15)
            psi_real[row, col] = temperature
            psi_imag[row, col] = sigma
            chi_dot[row, col] = mu
            chi[row, col] = math.log(je_mean)

        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
        }

    def get_name(self) -> str:
        return "te1_jarzynski_dataset"


class PR0DiffusionDatasetGenerator(InitialConditionGenerator):
    """Load PR-0 diffusion metrics and encode a linear flux law target.

    Design reference:
    - TE₁.U / PR-0 Rule-110 simulations:
      /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.U_TRANSPUTATIONAL_UNIVERSALITY
    """

    def __init__(self, csv_path: str, diffusivity: float = 50.0):
        self.csv_path = Path(csv_path)
        self.diffusivity = float(diffusivity)

    def _load_rows(self) -> list[dict]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"PR-0 CSV dataset not found: {self.csv_path}")
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]
        if not rows:
            raise ValueError(f"PR-0 CSV dataset is empty: {self.csv_path}")
        return rows

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rows = self._load_rows()
        L_y, L_x = lattice_shape
        capacity = L_y * L_x
        if len(rows) > capacity:
            raise ValueError(
                f"Lattice capacity {capacity} insufficient for {len(rows)} PR-0 rows; increase lattice size"
            )

        psi_real = np.zeros((L_y, L_x), dtype=float)
        psi_imag = np.zeros_like(psi_real)
        chi = np.zeros_like(psi_real)
        chi_dot = np.zeros_like(psi_real)

        for idx, row in enumerate(rows):
            r = idx // L_x
            c = idx % L_x
            density_delta = float(row.get("density_delta", 0.0))
            damping_flux = float(row.get("damping_flux", 0.0))
            gamma_mean = float(row.get("gamma_mean", 0.0))
            psi_real[r, c] = density_delta
            psi_imag[r, c] = damping_flux
            chi_dot[r, c] = gamma_mean
            chi[r, c] = damping_flux + self.diffusivity * density_delta

        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
        }

    def get_name(self) -> str:
        return "pr0_diffusion_dataset"


class PR0TransportDatasetGenerator(InitialConditionGenerator):
    """Load coarse-grained PR-0 transport datasets for reflexive transport discovery."""

    def __init__(
        self,
        dataset_dir: str,
        regime: str | None = None,
        allow_metadata_override: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.regime = regime
        self.allow_metadata_override = allow_metadata_override
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._metadata_cache: dict[str, dict[str, object]] = {}

    def _load_regime(self, regime: str) -> dict[str, np.ndarray]:
        if regime in self._cache:
            return self._cache[regime]
        path = self.dataset_dir / regime / "transport_dataset.npz"
        if not path.exists():
            raise FileNotFoundError(f"PR-0 transport dataset not found: {path}")
        data = np.load(path, allow_pickle=True)
        required = {"rho", "grad_rho_x", "grad_rho_y", "gamma", "rho_t"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset {path} missing arrays: {sorted(missing)}")
        arrays = {name: data[name] for name in required}
        arrays["dissonance_series"] = data.get("dissonance")
        meta_raw = data.get("metadata")
        meta: dict[str, object] = {}
        if meta_raw is not None:
            try:
                meta = json.loads(str(meta_raw))
            except Exception:  # pragma: no cover - defensive
                meta = {}
        self._metadata_cache[regime] = meta
        self._cache[regime] = arrays
        return arrays

    def _resolve_regime(self, metadata: Dict[str, object] | None) -> str:
        env_regime = os.environ.get("PR0_TRANSPORT_REGIME")
        if env_regime:
            candidate = env_regime.strip()
            if candidate:
                return candidate
        meta = metadata or {}
        regime = meta.get("transport_regime") if self.allow_metadata_override else None
        if regime is None:
            regime = self.regime
        if regime is None:
            candidates = [p.name for p in self.dataset_dir.iterdir() if (p / "transport_dataset.npz").exists()]
            if not candidates:
                raise ValueError(f"No transport datasets found in {self.dataset_dir}")
            regime = random.choice(candidates)
        return str(regime)

    def _select_index(self, arrays: dict[str, np.ndarray], metadata: Dict[str, object] | None) -> int:
        env_index = os.environ.get("PR0_TRANSPORT_SAMPLE_INDEX")
        if env_index is not None:
            try:
                index = int(env_index)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError("PR0_TRANSPORT_SAMPLE_INDEX must be an integer") from exc
        elif metadata and "sample_index" in metadata and self.allow_metadata_override:
            index = int(metadata["sample_index"])
        else:
            index = random.randrange(arrays["rho"].shape[0])
        if index < 0 or index >= arrays["rho"].shape[0]:
            raise IndexError(
                f"sample_index {index} out of range for dataset with {arrays['rho'].shape[0]} samples"
            )
        return index

    def load_sample(self, regime: str, sample_index: int) -> dict[str, np.ndarray]:
        """Return raw arrays for a specific dataset sample (used by analysis tooling)."""
        arrays = self._load_regime(regime)
        if sample_index < 0 or sample_index >= arrays["rho"].shape[0]:
            raise IndexError(
                f"sample_index {sample_index} out of range for dataset with {arrays['rho'].shape[0]} samples"
            )
        return {
            "rho": arrays["rho"][sample_index].astype(np.float64, copy=False),
            "rho_grad_x": arrays["grad_rho_x"][sample_index].astype(np.float64, copy=False),
            "rho_grad_y": arrays["grad_rho_y"][sample_index].astype(np.float64, copy=False),
            "gamma": arrays["gamma"][sample_index].astype(np.float64, copy=False),
            "rho_dot": arrays["rho_t"][sample_index].astype(np.float64, copy=False),
        }

    def get_dataset_metadata(self, regime: str) -> dict[str, object] | None:
        """Return cached metadata for the most recently used regime (helper for tests)."""
        return self._metadata_cache.get(regime)

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            random.seed(seed)
        regime = self._resolve_regime(metadata)
        arrays = self._load_regime(regime)
        index = self._select_index(arrays, metadata)

        rho = arrays["rho"][index]
        grad_x = arrays["grad_rho_x"][index]
        grad_y = arrays["grad_rho_y"][index]
        gamma = arrays["gamma"][index]
        rho_t = arrays["rho_t"][index]

        L_y, L_x = lattice_shape
        if rho.shape != (L_y, L_x):
            raise ValueError(
                f"Dataset sample shape {rho.shape} does not match lattice shape {(L_y, L_x)}"
            )

        return {
            "psi_real": rho.astype(np.float64, copy=False),
            "psi_imag": grad_x.astype(np.float64, copy=False),
            "chi": grad_y.astype(np.float64, copy=False),
            "chi_dot": gamma.astype(np.float64, copy=False),
            "dissonance": rho_t.astype(np.float64, copy=False),
        }

    def get_name(self) -> str:
        return "pr0_transport_dataset"


class TSPWarmStartGenerator(InitialConditionGenerator):
    """Warm start for TSP assignment matrices with near-uniform entries."""

    def __init__(
        self,
        num_cities: int | None = None,
        embedding_row_start: int = 0,
        embedding_col_start: int = 0,
        noise_level: float = 0.02,
    ):
        self.num_cities = num_cities
        self.embedding_row_start = embedding_row_start
        self.embedding_col_start = embedding_col_start
        self.noise_level = noise_level

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        L_y, L_x = lattice_shape
        psi_real = np.zeros((L_y, L_x), dtype=float)
        psi_imag = np.zeros_like(psi_real)
        chi = np.zeros_like(psi_real)
        chi_dot = np.zeros_like(psi_real)

        tsp_meta = metadata.get("tsp", {}) if metadata else {}
        num_cities = int(tsp_meta.get("num_cities", self.num_cities or 0))
        if num_cities <= 1:
            raise ValueError("TSPWarmStartGenerator requires tsp.num_cities > 1")
        row_start = int(tsp_meta.get("row_start", self.embedding_row_start))
        col_start = int(tsp_meta.get("col_start", self.embedding_col_start))
        row_slice = slice(row_start, row_start + num_cities)
        col_slice = slice(col_start, col_start + num_cities)

        uniform = np.full((num_cities, num_cities), 1.0 / num_cities, dtype=float)
        noise = rng.normal(0.0, self.noise_level / max(1, num_cities), size=uniform.shape)
        warm_start = np.clip(uniform + noise, 1e-6, None)
        warm_start /= np.sum(warm_start, axis=1, keepdims=True)
        warm_start /= np.sum(warm_start, axis=0, keepdims=True)

        psi_real[row_slice, col_slice] = warm_start
        psi_imag[row_slice, col_slice] = warm_start

        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
        }

    def get_name(self) -> str:
        return "tsp_warm_start"


class BoundaryBulkDatasetGenerator(InitialConditionGenerator):
    """Load TE₂.H boundary↔bulk dataset samples.

    Dataset specification:
    - /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2h_reflexive_holographic_equivalence/boundary_bulk_dataset.py
    """

    def __init__(self, dataset_dir: str, sample_index: int | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.sample_index = sample_index
        self._manifest: list[dict[str, object]] | None = None

    def _load_manifest(self) -> list[dict[str, object]]:
        if self._manifest is None:
            manifest_path = self.dataset_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Boundary↔bulk manifest not found: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._manifest = list(data.get("samples", []))
            if not self._manifest:
                raise ValueError(f"Boundary↔bulk manifest is empty: {manifest_path}")
        return self._manifest

    def _select_sample(self, rng: np.random.Generator, metadata: Dict[str, object] | None) -> Path:
        manifest = self._load_manifest()
        if self.sample_index is not None:
            index = self.sample_index
        elif metadata and "sample_index" in metadata:
            index = int(metadata["sample_index"])
        else:
            index = int(rng.integers(0, len(manifest)))
        if index < 0 or index >= len(manifest):
            raise IndexError(f"Boundary↔bulk sample_index {index} out of range")
        sample_path = Path(manifest[index]["path"])
        if not sample_path.is_absolute():
            sample_path = (self.dataset_dir / sample_path).resolve()
        if not sample_path.exists():
            raise FileNotFoundError(f"Boundary↔bulk sample missing: {sample_path}")
        return sample_path

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        sample_path = self._select_sample(rng, metadata)
        data = np.load(sample_path)
        bulk = np.array(data["bulk_field"], dtype=np.float64)
        boundary = np.array(data["boundary_projection"], dtype=np.float64)
        if bulk.shape != lattice_shape:
            raise ValueError(
                f"Boundary↔bulk sample shape {bulk.shape} does not match lattice {lattice_shape}"
            )
        mask = np.array(data["boundary_mask"], dtype=bool)
        # Normalize bulk/boundary amplitudes following TE₁.Y holography baseline.
        bulk -= bulk.mean()
        boundary -= boundary.mean()
        bulk_scale = max(np.std(bulk), 1e-6)
        boundary_scale = max(np.std(boundary[mask]) if mask.any() else np.std(boundary), 1e-6)
        bulk = np.clip(bulk / bulk_scale, -5.0, 5.0)
        boundary = np.clip(boundary / boundary_scale, -5.0, 5.0)
        chi = boundary * mask
        chi_dot = bulk - gaussian_filter(bulk, sigma=1.0)
        psi_real = bulk
        psi_imag = gaussian_filter(boundary, sigma=0.6)
        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
        }

    def get_name(self) -> str:
        return "boundary_bulk_dataset"


class TPUCBenchmarkDatasetGenerator(InitialConditionGenerator):
    """Load TE₂.U TPU-C benchmark tasks.

    Dataset specification:
    - /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2u_strong_transputational_universality/tpuc_dataset.py
    """

    def __init__(self, dataset_dir: str, family: str | None = None, sample_index: int | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.family = family
        self.sample_index = sample_index
        self._manifest: list[dict[str, object]] | None = None

    def _load_manifest(self) -> list[dict[str, object]]:
        if self._manifest is None:
            manifest_path = self.dataset_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"TPU-C manifest not found: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._manifest = list(data.get("samples", []))
            if not self._manifest:
                raise ValueError(f"TPU-C manifest is empty: {manifest_path}")
        return self._manifest

    def _select_entry(self, rng: np.random.Generator, metadata: Dict[str, object] | None) -> dict[str, object]:
        manifest = self._load_manifest()
        family = metadata.get("family") if metadata else None
        if family is None:
            family = self.family
        if family is not None:
            candidates = [entry for entry in manifest if entry["family"] == family]
            if not candidates:
                raise ValueError(f"No TPU-C tasks for family '{family}' in manifest")
        else:
            candidates = manifest
        if self.sample_index is not None:
            index = self.sample_index % len(candidates)
        elif metadata and "sample_index" in metadata:
            index = int(metadata["sample_index"]) % len(candidates)
        else:
            index = int(rng.integers(0, len(candidates)))
        return candidates[index]

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        entry = self._select_entry(rng, metadata)
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute():
            sample_path = (self.dataset_dir / sample_path).resolve()
        if not sample_path.exists():
            raise FileNotFoundError(f"TPU-C sample missing: {sample_path}")
        data = np.load(sample_path)
        matrix = np.array(data.get("matrix", data.get("kernel", data.get("clause_tensor"))), dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(int(np.sqrt(matrix.size)), -1)
        L_y, L_x = lattice_shape
        if matrix.shape != (L_y, L_x):
            matrix = np.resize(matrix, (L_y, L_x))
        solution = np.array(data.get("solution", data.get("assignment")), dtype=np.float64)
        solution = np.resize(solution, (L_y, L_x))
        chi_reference = solution.copy()
        skew = np.array(data.get("skew_component", np.zeros_like(matrix)), dtype=np.float64)
        if skew.shape != (L_y, L_x):
            skew = np.resize(skew, (L_y, L_x))
        holographic_bias = float(np.array(data.get("holographic_bias", 0.0)))
        conditioning = float(np.array(data.get("conditioning", 1.0)))
        psi_real = matrix
        psi_imag = skew
        chi = solution.copy()
        chi_dot = np.full((L_y, L_x), holographic_bias / max(conditioning, 1e-6), dtype=np.float64)
        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
            "chi_reference": chi_reference,
            "tpuc_manifest_entry": {
                "family": entry.get("family"),
                "sample_index": entry.get("sample_index"),
                "path": entry.get("path"),
                "classical_cost": entry.get("classical_cost"),
                "conditioning": entry.get("conditioning"),
                "holographic_bias": entry.get("holographic_bias"),
            },
            "tpuc_manifest_path": str(self.dataset_dir / "manifest.json"),
        }

    def get_name(self) -> str:
        return "tpuc_benchmark_dataset"


class MultiscaleDatasetGenerator(InitialConditionGenerator):
    """Load TE₂.RG multi-scale PR-0 ensembles.

    Dataset specification:
    - /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/src/delta_machine/scenarios/band_c/te2rg_reflexive_rg_flow/multiscale_dataset.py
    """

    def __init__(self, dataset_dir: str, scale: int | None = None, sample_index: int | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.scale = scale
        self.sample_index = sample_index
        self._manifest: list[dict[str, object]] | None = None

    def _load_manifest(self) -> list[dict[str, object]]:
        if self._manifest is None:
            manifest_path = self.dataset_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Multiscale manifest not found: {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._manifest = list(data.get("samples", []))
            if not self._manifest:
                raise ValueError(f"Multiscale manifest is empty: {manifest_path}")
        return self._manifest

    def _select_entry(self, rng: np.random.Generator, metadata: Dict[str, object] | None) -> dict[str, object]:
        manifest = self._load_manifest()
        scale = metadata.get("scale") if metadata else None
        if scale is None:
            scale = self.scale
        if scale is not None:
            scale = int(scale)
            candidates = [entry for entry in manifest if int(entry["scale"]) == scale]
            if not candidates:
                raise ValueError(f"No multiscale samples for scale={scale}")
        else:
            candidates = manifest
        if self.sample_index is not None:
            index = self.sample_index % len(candidates)
        elif metadata and "sample_index" in metadata:
            index = int(metadata["sample_index"]) % len(candidates)
        else:
            index = int(rng.integers(0, len(candidates)))
        return candidates[index]

    def generate(
        self,
        lattice_shape: Tuple[int, int],
        seed: int | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        entry = self._select_entry(rng, metadata)
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute():
            sample_path = (self.dataset_dir / sample_path).resolve()
        if not sample_path.exists():
            raise FileNotFoundError(f"Multiscale sample missing: {sample_path}")
        data = np.load(sample_path)
        field = np.array(data["field"], dtype=np.float64)
        if field.shape != lattice_shape:
            field = np.resize(field, lattice_shape)
        grad_x = np.array(data.get("gradient_x"), dtype=np.float64)
        grad_y = np.array(data.get("gradient_y"), dtype=np.float64)
        divergence = np.array(data.get("flux_divergence"), dtype=np.float64)
        psi_real = field
        psi_imag = grad_x if grad_x.size else np.zeros_like(field)
        chi = grad_y if grad_y.size else np.zeros_like(field)
        chi_dot = divergence if divergence.size else np.zeros_like(field)
        return {
            "psi_real": psi_real,
            "psi_imag": psi_imag,
            "chi": chi,
            "chi_dot": chi_dot,
            "multiscale_manifest_entry": {
                "scale": entry.get("scale"),
                "sample_index": entry.get("sample_index"),
                "path": entry.get("path"),
            },
            "multiscale_manifest_path": str(self.dataset_dir / "manifest.json"),
        }

    def get_name(self) -> str:
        return "multiscale_pr0_dataset"


_GENERATOR_REGISTRY: Dict[str, type] = {
    "random": RandomGenerator,
    "gaussian": GaussianGenerator,
    "vortex": VortexGenerator,
    "soliton": SolitonGenerator,
    "pattern": PatternGenerator,
    "polynomial_dataset": PolynomialDatasetGenerator,
    "te1_jarzynski_dataset": TE1JarzynskiDatasetGenerator,
    "pr0_diffusion_dataset": PR0DiffusionDatasetGenerator,
    "pr0_transport_dataset": PR0TransportDatasetGenerator,
    "tsp_warm_start": TSPWarmStartGenerator,
    "boundary_bulk_dataset": BoundaryBulkDatasetGenerator,
    "tpuc_benchmark_dataset": TPUCBenchmarkDatasetGenerator,
    "multiscale_pr0_dataset": MultiscaleDatasetGenerator,
}


def load_generator(generator_type: str, **kwargs) -> InitialConditionGenerator:
    """Load a generator by type name."""
    if generator_type not in _GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator type: {generator_type}")
    return _GENERATOR_REGISTRY[generator_type](**kwargs)


def list_generators() -> list[str]:
    """List available generator types."""
    return list(_GENERATOR_REGISTRY.keys())

