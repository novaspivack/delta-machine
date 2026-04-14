#!/usr/bin/env python3
"""Generate PR-0 coarse-grained transport datasets for Phase II."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT / "src", PARENT_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
# Resolve pr0_system: env var, sibling of Delta_machine, or sibling of cwd
PR0_ROOT = Path(os.environ.get("PR0_SYSTEM_ROOT", "") or "")
if not PR0_ROOT or not PR0_ROOT.exists():
    PR0_ROOT = PARENT_ROOT / "pr0_system"
if not PR0_ROOT.exists():
    PR0_ROOT = Path.cwd().parent / "pr0_system"
if PR0_ROOT.exists() and str(PR0_ROOT) not in sys.path:
    sys.path.insert(0, str(PR0_ROOT))

from pr0_system.evolution.ablowitz_ladik import PR0_Final  # type: ignore
from pr0_system.bootstrap.dissonance import compute_ontological_dissonance  # type: ignore


@dataclass
class Regime:
    name: str
    gamma_base: float
    gamma_scale: float
    soliton_offset: float
    velocity: float
    amplitude: float = 3.0
    width: float = 3.5


REGIMES: tuple[Regime, ...] = (
    Regime(name="baseline", gamma_base=0.015, gamma_scale=0.50, soliton_offset=8.0, velocity=0.06),
    Regime(name="high_drive", gamma_base=0.030, gamma_scale=0.80, soliton_offset=10.0, velocity=0.10),
    Regime(name="asymmetric", gamma_base=0.012, gamma_scale=0.45, soliton_offset=6.0, velocity=0.04),
)

LATTICE_SHAPE = (64, 64)
DT = 0.01
TOTAL_STEPS = 2400
DEFAULT_SAMPLE_EVERY = 10
DEFAULT_SAMPLE_WINDOW = 1
HISTORY_LIMIT = 20
EPS = 1.0e-8


def central_gradient(field: np.ndarray, axis: int) -> np.ndarray:
    return 0.5 * (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis))


def run_regime(regime: Regime, output_dir: Path, sample_every: int, sample_window: int) -> None:
    integrator = PR0_Final(
        L_x=LATTICE_SHAPE[1],
        L_y=LATTICE_SHAPE[0],
        gamma_base=regime.gamma_base,
        gamma_scale=regime.gamma_scale,
    )

    mid_y = LATTICE_SHAPE[0] // 2
    left_x = LATTICE_SHAPE[1] // 2 - int(regime.soliton_offset)
    right_x = LATTICE_SHAPE[1] // 2 + int(regime.soliton_offset)

    integrator.set_soliton(
        x0=left_x,
        y0=mid_y,
        amplitude=regime.amplitude,
        width=regime.width,
        velocity_x=regime.velocity,
        sign=+1,
    )
    integrator.set_soliton(
        x0=right_x,
        y0=mid_y,
        amplitude=regime.amplitude,
        width=regime.width,
        velocity_x=-regime.velocity,
        sign=-1,
    )

    snapshots_rho: list[np.ndarray] = []
    snapshots_grad_x: list[np.ndarray] = []
    snapshots_grad_y: list[np.ndarray] = []
    snapshots_gamma: list[np.ndarray] = []
    snapshots_rho_dt: list[np.ndarray] = []
    dissonance_series: list[float] = []

    density_history: deque[np.ndarray] = deque(maxlen=sample_window)
    gamma_history: deque[np.ndarray] = deque(maxlen=sample_window)
    dissonance_history: deque[float] = deque(maxlen=sample_window)

    previous_avg: np.ndarray | None = None
    psi_history: list[np.ndarray] = []

    for step in range(TOTAL_STEPS):
        integrator.step(dt=DT)
        psi_history.append(integrator.psi.copy())
        if len(psi_history) > HISTORY_LIMIT:
            psi_history.pop(0)

        if (step + 1) % sample_every != 0:
            continue

        density = np.abs(integrator.psi) ** 2
        gamma = integrator._compute_damping()  # type: ignore[attr-defined]
        dissonance_value = float(
            compute_ontological_dissonance(integrator.psi, integrator.chi, psi_history)
        )

        density_history.append(density)
        gamma_history.append(gamma)
        dissonance_history.append(dissonance_value)

        if len(density_history) < sample_window:
            continue

        density_avg = np.mean(np.stack(density_history, axis=0), axis=0)
        gamma_avg = np.mean(np.stack(gamma_history, axis=0), axis=0)
        dissonance_avg = float(np.mean(dissonance_history))

        grad_x = central_gradient(density_avg, axis=1)
        grad_y = central_gradient(density_avg, axis=0)

        if previous_avg is not None:
            rho_dt = (density_avg - previous_avg) / (sample_every * DT)
            snapshots_rho_dt.append(rho_dt.astype(np.float32))
            snapshots_rho.append(density_avg.astype(np.float32))
            snapshots_grad_x.append(grad_x.astype(np.float32))
            snapshots_grad_y.append(grad_y.astype(np.float32))
            snapshots_gamma.append(gamma_avg.astype(np.float32))
            dissonance_series.append(dissonance_avg)

        previous_avg = density_avg

    if not snapshots_rho:
        raise RuntimeError(f"No snapshots recorded for regime {regime.name}")

    rho_stack = np.stack(snapshots_rho)
    rho_dt_stack = np.stack(snapshots_rho_dt)
    gamma_stack = np.stack(snapshots_gamma)
    grad_norm_stack = np.sqrt(np.stack(snapshots_grad_x) ** 2 + np.stack(snapshots_grad_y) ** 2)

    regime_dir = output_dir / regime.name
    regime_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        regime_dir / "transport_dataset.npz",
        rho=rho_stack,
        grad_rho_x=np.stack(snapshots_grad_x),
        grad_rho_y=np.stack(snapshots_grad_y),
        gamma=gamma_stack,
        rho_t=rho_dt_stack,
        dissonance=np.asarray(dissonance_series, dtype=np.float32),
        metadata=json.dumps(
            {
                "regime": regime.__dict__,
                "dt": DT,
                "sample_every": sample_every,
                "sample_window": sample_window,
                "total_steps": TOTAL_STEPS,
                "lattice_shape": LATTICE_SHAPE,
                "rho_mean": float(rho_stack.mean()),
                "rho_std": float(rho_stack.std() + EPS),
                "rho_t_std": float(rho_dt_stack.std() + EPS),
                "gamma_mean": float(gamma_stack.mean()),
                "gamma_std": float(gamma_stack.std() + EPS),
                "grad_rho_rms": float(np.sqrt(np.mean(grad_norm_stack ** 2)) + EPS),
                "dissonance_mean": float(np.mean(dissonance_series)) if dissonance_series else 0.0,
                "dissonance_std": float(np.std(dissonance_series) + EPS)
                if dissonance_series
                else 0.0,
            }
        ),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate PR-0 coarse-grained transport datasets")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "pr0_transport"),
        help="Directory to store generated datasets",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help="Simulation steps between samples",
    )
    parser.add_argument(
        "--sample-window",
        type=int,
        default=DEFAULT_SAMPLE_WINDOW,
        help="Temporal window (number of samples) for averaging observables",
    )
    args = parser.parse_args(argv)

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")
    if args.sample_window <= 0:
        raise ValueError("--sample-window must be positive")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime in REGIMES:
        print(f"[PR-0] Generating regime '{regime.name}'...")
        run_regime(regime, output_dir, sample_every=args.sample_every, sample_window=args.sample_window)
        print(f"[PR-0]   -> stored in {output_dir / regime.name / 'transport_dataset.npz'}")


if __name__ == "__main__":
    main()
