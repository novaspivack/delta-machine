"""
Diagnostics, reporting, and logging utilities for the Δ-Machine runtime.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MPL = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import OrchestratorTelemetry


@dataclass(slots=True)
class RunRecord:
    timestamp: float
    total_dissonance: float
    cpu_percent: float
    memory_percent: float
    worker_steps: Dict[int, int]


class RunLogger:
    """Structured logging for Δ-Machine runs."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "telemetry.csv"
        self.struct_log = self.run_dir / "telemetry.jsonl"
        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["timestamp", "dissonance", "cpu_percent", "memory_percent", "worker_steps"])

    def append(self, telemetry: "OrchestratorTelemetry"):
        record = RunRecord(
            timestamp=telemetry.timestamp,
            total_dissonance=telemetry.total_dissonance,
            cpu_percent=telemetry.cpu_percent,
            memory_percent=telemetry.memory_percent,
            worker_steps={wid: status.processed_steps for wid, status in telemetry.worker_load.items()},
        )
        with self.log_file.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                record.timestamp,
                record.total_dissonance,
                record.cpu_percent,
                record.memory_percent,
                json.dumps(record.worker_steps),
            ])
        with self.struct_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)))
            fh.write("\n")


def save_field_snapshot(run_dir: Path, arrays: Dict[str, np.ndarray], step: int):
    snapshot_dir = Path(run_dir) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        path = snapshot_dir / f"{name}_step_{step:06d}.npy"
        np.save(path, array)
        if name in {"psi_real", "psi_imag"} and HAS_MPL:
            preview = snapshot_dir / f"{name}_step_{step:06d}.png"
            _save_image(array, preview)


def create_run_directory(base_dir: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_final_preview(run_dir: Path, arrays: Dict[str, np.ndarray]):
    if not HAS_MPL:
        return
    state_dir = Path(run_dir) / "final_state"
    state_dir.mkdir(exist_ok=True)
    psi_real = arrays.get("psi_real")
    psi_imag = arrays.get("psi_imag")
    if psi_real is None or psi_imag is None:
        return
    magnitude = np.sqrt(np.square(psi_real) + np.square(psi_imag))
    path = state_dir / "psi_magnitude.png"
    _save_image(magnitude, path)


def _save_image(data: np.ndarray, output_path: Path, cmap: str = "magma") -> None:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        norm = np.zeros_like(data, dtype=float)
    else:
        lo = float(finite.min())
        hi = float(finite.max())
        if np.isclose(hi, lo):
            norm = np.zeros_like(data, dtype=float)
        else:
            norm = (data - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.imshow(norm, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


