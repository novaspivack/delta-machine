"""Helpers for integrating Δ-Machine orchestrator with PR-0 field state.

This module is intentionally lightweight so that PR-0 remains an optional
dependency. Import failures are handled gracefully and callers can detect
whether PR-0 integration is available via :func:`pr0_available` and the
feature flag helpers below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pr0_system.core.fields import FieldState as FieldStateProto
    from pr0_system.core.lattice import Lattice as LatticeProto
else:  # Runtime fallback for type checking
    FieldStateProto = Any  # type: ignore[type-arg]
    LatticeProto = Any  # type: ignore[type-arg]

try:  # Optional runtime dependency
    from pr0_system.core.fields import FieldState as FieldStateRuntime
    from pr0_system.core.lattice import Lattice as LatticeRuntime
except Exception:  # pragma: no cover - handled by runtime checks
    FieldStateRuntime = None  # type: ignore[assignment]
    LatticeRuntime = None  # type: ignore[assignment]


PR0_FEATURE_FLAG = os.getenv("DELTA_USE_PR0_FIELDSTATE", "0").lower()
PR0_ENABLED = PR0_FEATURE_FLAG in {"1", "true", "yes", "on"}


def pr0_available() -> bool:
    """Return True if PR-0 modules were imported successfully."""

    return FieldStateRuntime is not None and LatticeRuntime is not None


@dataclass(slots=True)
class PR0FieldBundle:
    """Container wiring PR-0 field state to DSAC shared arrays."""

    lattice: Optional[LatticeProto]
    field_state: Optional[FieldStateProto]
    dissonance_history: List[Tuple[int, float]] = field(default_factory=list)

    def sync_from_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """Copy DSAC shared arrays into the PR-0 field state."""

        if self.field_state is None:
            return
        psi = self.field_state.psi
        chi = self.field_state.chi
        chi_dot = self.field_state.chi_dot
        psi.real[:, :] = arrays["psi_real"]
        psi.imag[:, :] = arrays["psi_imag"]
        chi[:, :] = arrays["chi"]
        chi_dot[:, :] = arrays["chi_dot"]

    def sync_to_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """Copy PR-0 field state into DSAC shared arrays."""

        if self.field_state is None:
            return
        psi = self.field_state.psi
        arrays["psi_real"][:, :] = psi.real
        arrays["psi_imag"][:, :] = psi.imag
        arrays["chi"][:, :] = self.field_state.chi
        arrays["chi_dot"][:, :] = self.field_state.chi_dot

    def record_dissonance(self, step: int, value: float) -> None:
        """Store latest dissonance value for downstream PR-0 analysis."""

        if self.field_state is None:
            return
        self.dissonance_history.append((step, value))
        try:
            setattr(self.field_state, "last_dissonance", float(value))
        except Exception:  # pragma: no cover - attribute may be read-only
            pass


def should_use_pr0(metadata: Dict[str, object] | None = None) -> bool:
    """Determine whether PR-0 integration is requested."""

    if not pr0_available():
        return False
    if PR0_ENABLED:
        return True
    if metadata and metadata.get("use_pr0_field_state") is True:
        return True
    return False


def build_lattice(
    shape: Tuple[int, int],
    metadata: Dict[str, object] | None = None,
) -> Optional[LatticeProto]:
    """Create a PR-0 lattice according to scenario metadata."""

    if LatticeRuntime is None:
        return None

    rows, cols = shape
    lattice_cfg = (metadata or {}).get("lattice_config", {}) if metadata else {}
    lattice_type = str(lattice_cfg.get("type", "square"))
    boundary = lattice_cfg.get("boundary")

    if boundary is None:
        periodic = bool(lattice_cfg.get("periodic", True))
        return LatticeRuntime(cols, rows, lattice_type=lattice_type, periodic=periodic)

    # Newer PR-0 builds accept boundary keyword; fall back to periodic flag
    try:
        return LatticeRuntime(cols, rows, lattice_type=lattice_type, boundary=str(boundary))  # type: ignore[arg-type]
    except TypeError:
        periodic = str(boundary).lower() == "torus"
        return LatticeRuntime(cols, rows, lattice_type=lattice_type, periodic=periodic)


def create_field_bundle(
    shape: Tuple[int, int],
    metadata: Dict[str, object] | None = None,
    history_length: int = 20,
) -> PR0FieldBundle:
    """Instantiate PR-0 lattice and field state if available."""

    lattice = build_lattice(shape, metadata)
    if lattice is None or FieldStateRuntime is None:
        return PR0FieldBundle(None, None)

    try:
        field_state = FieldStateRuntime(lattice, history_length=history_length)
    except Exception:
        field_state = None
    return PR0FieldBundle(lattice, field_state)
