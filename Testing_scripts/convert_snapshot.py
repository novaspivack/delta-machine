#!/usr/bin/env python3
"""Convert Δ-Machine .npy snapshots into PNG images for quick inspection.

Design references:
- 1.0 Δ-Computing Paradigm Definition
- 1.4 Implementation & Validation Update
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_array(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim == 3 and data.shape[0] in {3, 4}:
        # assume channel-first, convert to HxWxC for convenience
        data = np.moveaxis(data, 0, -1)
    return data.astype(float)


def normalize(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=float)
    lo = float(finite.min())
    hi = float(finite.max())
    if np.isclose(hi, lo):
        return np.zeros_like(data, dtype=float)
    return (data - lo) / (hi - lo)


def save_image(data: np.ndarray, output_path: Path, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.imshow(data, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Δ-Machine numpy snapshot to PNG")
    parser.add_argument("npy_path", type=Path, help="Path to the .npy snapshot (e.g., runs/.../psi_real.npy)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path (.png). Defaults to <file>.png")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap to use (default: magma)")
    args = parser.parse_args()

    npy_path = args.npy_path
    if not npy_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {npy_path}")

    data = load_array(npy_path)
    normed = normalize(data)
    output_path = args.output or npy_path.with_suffix(".png")
    save_image(normed, output_path, cmap=args.cmap)
    print(f"Saved snapshot visualization to {output_path}")


if __name__ == "__main__":
    main()
