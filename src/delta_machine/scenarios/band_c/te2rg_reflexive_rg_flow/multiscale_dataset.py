"""PR-0 multiscale ensemble generation for TE₂.RG reflexive RG flow studies.

Cross-links:
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.38_phase3_bandc_experiments_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.35_phase2_pr0_transport_plan.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.36_phase2_pr0_transport_results.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.37_phase2_pr0_transport_proof_package.md
- /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/si_optimizer_data/MATHEMATICAL_FOUNDATIONS_REFLEXIVE_REALITY/Mathematical_Foundations_of_Reflexive_Reality.tex

The routines synthesise multi-scale PR-0 field ensembles to supply DSAC with empirical
β-function targets complementing the analytic SRRG program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(slots=True)
class MultiscaleEnsembleConfig:
    """Configuration for TE₂.RG multi-scale ensembles."""

    output_dir: Path
    base_shape: Tuple[int, int] = (128, 128)
    scales: Sequence[int] = (1, 2, 4, 8)
    samples_per_scale: int = 12
    random_seed: int | None = None
    include_flux: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if min(self.base_shape) < 32:
            raise ValueError("base_shape must be >= 32×32 for multiscale analysis")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be positive integers")
        if self.samples_per_scale <= 0:
            raise ValueError("samples_per_scale must be positive")


def generate_multiscale_ensemble(config: MultiscaleEnsembleConfig) -> List[Path]:
    """Create multi-scale PR-0 style ensembles for DSAC β-function fitting."""

    config.validate()
    rng = np.random.default_rng(config.random_seed)
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    sample_paths: List[Path] = []

    base_field = _synthesise_base_field(config.base_shape, rng)

    for scale in config.scales:
        scaled_dir = out_dir / f"scale_{scale:02d}"
        scaled_dir.mkdir(exist_ok=True)
        for sample_idx in range(config.samples_per_scale):
            seed = int(rng.integers(0, 2**32 - 1))
            sample = _prepare_scale_sample(base_field, scale, seed, config.include_flux)
            sample_path = scaled_dir / f"sample_{sample_idx:03d}.npz"
            np.savez_compressed(sample_path, **sample)
            manifest.append(
                {
                    "scale": scale,
                    "sample_index": sample_idx,
                    "path": sample_path.as_posix(),
                    "seed": seed,
                    "norm": float(sample["field"].std()),
                }
            )
            sample_paths.append(sample_path)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "te2rg_multiscale_v1",
                "config": {
                    "base_shape": list(config.base_shape),
                    "scales": list(config.scales),
                    "samples_per_scale": config.samples_per_scale,
                    "include_flux": config.include_flux,
                    **config.metadata,
                },
                "samples": manifest,
            },
            handle,
            indent=2,
        )

    return sample_paths


def _synthesise_base_field(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    rows, cols = shape
    spectrum = rng.normal(0.0, 1.0, size=(rows, cols)) + 1j * rng.normal(0.0, 1.0, size=(rows, cols))
    kx = np.fft.fftfreq(cols)[:, None]
    ky = np.fft.fftfreq(rows)[None, :]
    laplacian = (kx**2 + ky**2)
    laplacian[laplacian == 0] = 1.0
    spectral_filter = 1.0 / laplacian
    field = np.fft.ifft2(np.fft.fft2(spectrum) * spectral_filter).real
    field -= field.mean()
    field /= max(np.std(field), 1e-9)
    return field


def _prepare_scale_sample(
    base_field: np.ndarray,
    scale: int,
    seed: int,
    include_flux: bool,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sigma = scale * 0.8
    smoothed = gaussian_filter(base_field, sigma=sigma, mode="wrap")
    resampled = smoothed
    if scale > 1:
        resampled = smoothed[::scale, ::scale]
        resampled = np.kron(resampled, np.ones((scale, scale)))
        resampled = resampled[: base_field.shape[0], : base_field.shape[1]]

    perturbation = rng.normal(0.0, 0.1 / max(scale, 1), size=base_field.shape)
    field = (resampled + perturbation).astype(np.float64)

    sample: Dict[str, np.ndarray] = {
        "field": field,
        "scale": np.array(scale, dtype=np.int32),
    }

    if include_flux:
        gradients = np.gradient(field)
        flux_divergence = np.gradient(gradients[0])[0] + np.gradient(gradients[1])[1]
        sample["flux_divergence"] = flux_divergence.astype(np.float64)
        sample["gradient_x"] = gradients[1].astype(np.float64)
        sample["gradient_y"] = gradients[0].astype(np.float64)

    return sample
