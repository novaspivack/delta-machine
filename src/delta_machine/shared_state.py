"""
Shared-memory management for Δ-Machine multiprocessing.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict

import numpy as np


BUFFER_NAMES = ("psi_real", "psi_imag", "chi", "chi_dot", "dissonance", "chi_reference")


def _create_shared_array(name: str, shape: tuple[int, int], dtype: np.dtype) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    shm = shared_memory.SharedMemory(create=True, size=size * dtype.itemsize, name=None)
    buffer = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    buffer.fill(0.0)
    return shm, buffer


@dataclass(slots=True)
class SharedState:
    """Collection of shared arrays representing the evolving DSAC state."""

    shape: tuple[int, int]
    dtype: np.dtype = np.float64
    _segments: Dict[str, shared_memory.SharedMemory] = None  # type: ignore
    _arrays: Dict[str, np.ndarray] = None  # type: ignore

    def __post_init__(self):
        self.dtype = np.dtype(self.dtype)
        self._segments = {}
        self._arrays = {}
        for name in BUFFER_NAMES:
            shm, arr = _create_shared_array(name, self.shape, self.dtype)
            self._segments[name] = shm
            self._arrays[name] = arr

    def arrays(self) -> Dict[str, np.ndarray]:
        return self._arrays

    def close(self):
        for shm in self._segments.values():
            shm.close()

    def unlink(self):
        for shm in self._segments.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def descriptor(self) -> Dict[str, tuple[str, tuple[int, int]]]:
        return {name: (shm.name, self.shape) for name, shm in self._segments.items()}


def attach_shared_state(descriptor: Dict[str, tuple[str, tuple[int, int]]], dtype: np.dtype = np.float64) -> tuple[Dict[str, np.ndarray], Dict[str, shared_memory.SharedMemory]]:
    attached_arrays: Dict[str, np.ndarray] = {}
    handles: Dict[str, shared_memory.SharedMemory] = {}
    for name, (shm_name, shape) in descriptor.items():
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        attached_arrays[name] = arr
        handles[name] = shm
    return attached_arrays, handles


