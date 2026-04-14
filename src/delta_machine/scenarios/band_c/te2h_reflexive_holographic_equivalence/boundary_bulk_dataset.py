"""Boundary↔bulk dataset synthesis for TE₂.H reflexive holographic experiments.

This module belongs to the Band C execution plan recorded in
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md`.
Cross-references: computational holography theory in
`/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/TE_1_VALIDATION_PROGRAM/TE_1.Y_COMPUTATIONAL_HOLOGRAPHY/TE_1.Y_COMPTUTATIONAL_HOLOGRAPHY_THMS.md`.

The builder creates paired bulk states and boundary projections that DSAC can use to
validate reflexive holographic equivalence within PR-0/DSAC joint scenarios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class BoundaryBulkConfig:
    """Configuration for synthesising boundary↔bulk datasets.

    Parameters
    ----------
    output_dir:
        Destination directory. Subdirectories per topology are created automatically.
    lattice_shape:
        Spatial resolution ``(rows, cols)`` shared by bulk and boundary fields.
    boundary_depth:
        Number of lattice cells included in the boundary projection.
    topologies:
        Iterable of topology identifiers (e.g. ``("torus", "open", "sphere")``) used
        to parameterise synthetic bulk construction.
    num_samples:
        Number of realisations per topology.
    random_seed:
        Optional seed for reproducibility.
    include_noise:
        Whether to inject calibrated bulk-surface noise to emulate PR-0 instrumentation.
    metadata:
        Additional metadata persisted alongside the dataset manifest.
    """

    output_dir: Path
    lattice_shape: Tuple[int, int] = (128, 128)
    boundary_depth: int = 4
    topologies: Tuple[str, ...] = ("torus", "cylinder", "open")
    num_samples: int = 16
    random_seed: int | None = None
    include_noise: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        rows, cols = self.lattice_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("lattice_shape must contain positive dimensions")
        if self.boundary_depth < 1:
            raise ValueError("boundary_depth must be ≥ 1")
        if self.boundary_depth * 2 >= min(rows, cols):
            raise ValueError("boundary_depth too large for lattice dimensions")
        if not self.topologies:
            raise ValueError("at least one topology is required")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")


def build_boundary_bulk_dataset(config: BoundaryBulkConfig) -> List[Path]:
    """Generate paired bulk and boundary fields for TE₂.H experiments.

    Returns
    -------
    list[Path]
        Absolute paths to the generated ``.npz`` samples.
    """

    config.validate()
    rng = np.random.default_rng(config.random_seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    sample_paths: List[Path] = []

    for topology in config.topologies:
        topo_dir = output_dir / topology
        topo_dir.mkdir(exist_ok=True)
        for sample_idx in range(config.num_samples):
            sample_seed = int(rng.integers(0, 2**32 - 1))
            bulk_field = _synthesise_bulk_field(
                config.lattice_shape,
                topology=topology,
                seed=sample_seed,
            )
            boundary_mask = _build_boundary_mask(config.lattice_shape, config.boundary_depth, topology)
            boundary_projection = np.where(boundary_mask, bulk_field, 0.0)
            if config.include_noise:
                boundary_projection = _inject_boundary_noise(
                    rng,
                    boundary_projection,
                    mask=boundary_mask,
                    topology=topology,
                )

            relative_entropy = _surface_bulk_relative_entropy(bulk_field, boundary_projection, boundary_mask)
            compression_ratio = _holographic_compression_ratio(bulk_field, boundary_projection)

            sample_path = topo_dir / f"sample_{sample_idx:03d}.npz"
            np.savez_compressed(
                sample_path,
                bulk_field=bulk_field.astype(np.float64),
                boundary_projection=boundary_projection.astype(np.float64),
                boundary_mask=boundary_mask.astype(np.uint8),
                topology=np.array(topology),
                seed=np.array(sample_seed, dtype=np.uint64),
                compression_ratio=np.array(compression_ratio, dtype=np.float64),
                relative_entropy=np.array(relative_entropy, dtype=np.float64),
            )

            manifest.append(
                {
                    "topology": topology,
                    "sample_index": sample_idx,
                    "path": sample_path.as_posix(),
                    "seed": sample_seed,
                    "compression_ratio": float(compression_ratio),
                    "relative_entropy": float(relative_entropy),
                }
            )
            sample_paths.append(sample_path)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "te2h_boundary_bulk_v1",
                "config": {
                    "lattice_shape": list(config.lattice_shape),
                    "boundary_depth": config.boundary_depth,
                    "topologies": list(config.topologies),
                    "num_samples": config.num_samples,
                    "include_noise": config.include_noise,
                    **config.metadata,
                },
                "samples": manifest,
            },
            handle,
            indent=2,
        )

    return sample_paths


def _synthesise_bulk_field(
    lattice_shape: Tuple[int, int],
    topology: str,
    seed: int,
) -> np.ndarray:
    """Create a synthetic bulk field consistent with the requested topology."""

    rng = np.random.default_rng(seed)
    rows, cols = lattice_shape
    y = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False)
    x = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    X, Y = np.meshgrid(x, y)

    base_modes = []
    for _ in range(6):
        freq_x = rng.integers(1, 6)
        freq_y = rng.integers(1, 6)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.4, 1.2)
        base_modes.append(amplitude * np.sin(freq_x * X + phase) * np.cos(freq_y * Y - phase))
    bulk = np.sum(base_modes, axis=0)

    if topology == "torus":
        bulk = gaussian_filter(bulk, sigma=1.2, mode="wrap")
    elif topology == "cylinder":
        bulk = gaussian_filter(bulk, sigma=(1.2, 0.6), mode=["wrap", "reflect"])
    elif topology == "sphere":
        radial = _spherical_envelope(lattice_shape)
        bulk = gaussian_filter(bulk * radial, sigma=1.0, mode="nearest")
    else:  # open / default
        bulk = gaussian_filter(bulk, sigma=1.0, mode="nearest")

    bulk -= bulk.mean()
    bulk /= max(np.linalg.norm(bulk), 1e-9)
    return bulk


def _spherical_envelope(lattice_shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = lattice_shape
    y = np.linspace(-1.0, 1.0, rows)
    x = np.linspace(-1.0, 1.0, cols)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    mask = (r <= 1.0).astype(float)
    envelope = np.cos(np.pi * r / 2.0) ** 2
    return envelope * mask


def _build_boundary_mask(
    lattice_shape: Tuple[int, int],
    depth: int,
    topology: str,
) -> np.ndarray:
    rows, cols = lattice_shape
    mask = np.zeros((rows, cols), dtype=bool)
    if topology == "torus":
        mask[:depth, :] = True
        mask[-depth:, :] = True
        mask[:, :depth] = True
        mask[:, -depth:] = True
    elif topology == "cylinder":
        mask[:depth, :] = True
        mask[-depth:, :] = True
        mask[:, :depth] = True
    elif topology == "sphere":
        r = _spherical_envelope(lattice_shape)
        mask = r < 0.2
    else:  # open and default
        mask[:depth, :] = True
        mask[-depth:, :] = True
        mask[:, :depth] = True
        mask[:, -depth:] = True
    return mask


def _inject_boundary_noise(
    rng: np.random.Generator,
    boundary_projection: np.ndarray,
    mask: np.ndarray,
    topology: str,
) -> np.ndarray:
    sigma = 0.015 if topology == "torus" else 0.025
    noise = rng.normal(0.0, sigma, size=boundary_projection.shape)
    smoothed = gaussian_filter(noise, sigma=1.0)
    return boundary_projection + smoothed * mask


def _surface_bulk_relative_entropy(
    bulk_field: np.ndarray,
    boundary_projection: np.ndarray,
    mask: np.ndarray,
) -> float:
    eps = 1e-9
    surface = np.where(mask, np.abs(boundary_projection), eps)
    bulk = np.abs(bulk_field) + eps
    surface /= surface.sum()
    bulk /= bulk.sum()
    return float(np.sum(surface * np.log(surface / bulk)))


def _holographic_compression_ratio(bulk_field: np.ndarray, boundary_projection: np.ndarray) -> float:
    bulk_energy = float(np.sum(bulk_field**2))
    boundary_energy = float(np.sum(boundary_projection**2))
    return boundary_energy / max(bulk_energy, 1e-12)
