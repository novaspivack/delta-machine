# pyright: reportInvalidTypeForm=false
"""Constraint backends for DSAC.

This module defines the abstraction layer for constraint evaluation backends
(e.g., NumPy, Taichi) so that the orchestrator can swap implementations
without changing high-level logic.

CPU/float64 (NumPy or Taichi) is the production path. GPU/float32 Taichi
support exists but is still experimental: expect slower perf and tighter
stability tolerances until fused kernels land (see 1.30 optimization note).

Design references:
- 1.30 DSAC Optimization Strategies Lab Note:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.30_DSAC_Optmization_Strategies.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import os
import linecache
import math

import numpy as np
import sympy as sp

from ..constraints import ConstraintGraph
from .dsl import TaichiExpressionTranslator

try:
    import taichi as ti
except ImportError:  # pragma: no cover - Taichi is optional
    ti = None

taichi_init_done = False
_taichi_init_config: Dict[str, Any] | None = None
_taichi_init_signature: Optional[Tuple[Tuple[str, Any], ...]] = None


def _flatten_backend_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in cfg.items():
        if key in {"type", "backend", "name"}:
            continue
        if key == "options" and isinstance(value, dict):
            result.update(value)
        else:
            result[key] = value
    return result


def extract_backend_metadata(metadata: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not metadata:
        return None, {}
    meta = metadata.get("constraint_backend")
    if isinstance(meta, str):
        return meta.lower(), {}
    if isinstance(meta, dict):
        backend_type = meta.get("type") or meta.get("backend") or meta.get("name")
        backend_lower = backend_type.lower() if isinstance(backend_type, str) else None
        return backend_lower, _flatten_backend_config(meta)
    return None, {}


def resolve_backend_selection(
    metadata: Optional[Dict[str, Any]],
    override: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    meta_backend, meta_cfg = extract_backend_metadata(metadata)
    override_norm = (override or "auto").lower()
    if override_norm not in {"auto", "numpy", "taichi"}:
        override_norm = "auto"

    if override_norm == "auto":
        backend = meta_backend or "numpy"
        config = dict(meta_cfg) if meta_backend == "taichi" else {}
    else:
        backend = override_norm
        config = dict(meta_cfg) if meta_backend == backend == "taichi" else {}

    if backend not in {"numpy", "taichi"}:
        backend = "numpy"
        config = {}

    return backend, config


def _fingerprint_value(value: Any) -> Any:
    if hasattr(value, "__name__"):
        return value.__name__
    if isinstance(value, (int, float, str, bool)):
        return value
    return repr(value)


def _signature_for_kwargs(init_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, _fingerprint_value(val)) for key, val in init_kwargs.items()))


# Global handles for Taichi fields and relaxation; field objects are created after ti.init().
if ti is not None:
    psi_r_field = None  # type: ignore[assignment]
    psi_i_field = None  # type: ignore[assignment]
    chi_field = None  # type: ignore[assignment]
    chi_dot_field = None  # type: ignore[assignment]
    dissonance_field = None  # type: ignore[assignment]
    psi_r_snapshot_field = None  # type: ignore[assignment]
    psi_i_snapshot_field = None  # type: ignore[assignment]
    chi_snapshot_field = None  # type: ignore[assignment]
    chi_dot_snapshot_field = None  # type: ignore[assignment]
    dissonance_snapshot_field = None  # type: ignore[assignment]
    relaxation_rate_scalar = 0.01


class ConstraintBackend(Protocol):
    """Interface for constraint evaluation backends.

    Concrete implementations are responsible for applying one relaxation/update
    pass to the lattice fields given a `ConstraintGraph` and the current
    `state_buffers` mapping field names to NumPy arrays.
    """

    def step(self, state_buffers: Dict[str, np.ndarray]) -> None:  # pragma: no cover - protocol
        """Apply one constraint relaxation step in-place on `state_buffers`."""
        raise NotImplementedError


@dataclass
class NumpyConstraintBackend:
    """Backend that uses the existing NumPy-based `ConstraintGraph` evaluation.

    This is the default, reference backend. It preserves the current DSAC
    behaviour and serves as a correctness baseline for alternate backends
    (e.g., Taichi).
    """

    graph: ConstraintGraph
    relaxation_rate: float = 0.01

    def step(self, state_buffers: Dict[str, np.ndarray]) -> None:
        """Evaluate all constraints and relax fields in-place.

        This mirrors the behaviour currently implemented in `worker_entry`:
        it computes residuals for each constraint and subtracts a fraction of
        the residual from the corresponding fields.
        """
        residuals = self.graph.evaluate(state_buffers)
        node_map = self.graph.nodes
        for node_name, residual in residuals.items():
            target = node_map[node_name].target_field
            if target in state_buffers:
                state_buffers[target] -= self.relaxation_rate * residual


FIELD_VAR_MAP = {
    "psi_real": "psi_r_field",
    "psi_imag": "psi_i_field",
    "chi": "chi_field",
    "chi_dot": "chi_dot_field",
    "dissonance": "dissonance_field",
}

FIELD_SNAPSHOT_MAP = {
    "psi_real": "psi_r_snapshot_field",
    "psi_imag": "psi_i_snapshot_field",
    "chi": "chi_snapshot_field",
    "chi_dot": "chi_dot_snapshot_field",
    "dissonance": "dissonance_snapshot_field",
}


@dataclass
class TaichiConstraintBackend:
    """Taichi-based constraint backend using the restricted constraint DSL."""

    graph: ConstraintGraph
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = {}
        self._numpy_dtype = np.float64
        self._taichi_kernels: List[Callable[[], None]] = []
        self._snapshot_field_objects: Dict[str, Any] = {}
        self._loop_parallelize: Optional[int] = None
        self._loop_block_dim: Optional[int] = None
        self._tile_shape: Optional[Tuple[int, int]] = None
        self._grid_shape: Optional[Tuple[int, int]] = None
        self._use_vectorized: bool = False
        self._vector_width: int = 1

    def step(self, state_buffers: Dict[str, np.ndarray]) -> None:  # pragma: no cover - not yet wired
        if ti is None:
            raise RuntimeError(
                "TaichiConstraintBackend requires the 'taichi' package. "
                "Install it (e.g. 'pip install taichi') to use the Taichi backend."
            )

        if not hasattr(self, "_initialized"):
            self._initialize_kernels(state_buffers)
            self._initialized = True

        self._copy_numpy_to_taichi(state_buffers)
        for kernel in self._taichi_kernels:
            kernel()
        self._copy_taichi_to_numpy(state_buffers)

    def _initialize_kernels(self, state_buffers: Dict[str, np.ndarray]) -> None:
        if ti is None:  # pragma: no cover - defensive
            raise RuntimeError("Taichi is not available")

        psi_real = state_buffers.get("psi_real")
        if psi_real is None:
            raise RuntimeError("psi_real field is required for Taichi backend initialization")
        ny, nx = psi_real.shape
        self._grid_shape = (ny, nx)

        # Initialize Taichi runtime once.
        config = dict(self.config or {})
        init_kwargs: Dict[str, Any] = dict(config.pop("init_kwargs", {}))

        dtype_config = config.get("dtype") or config.get("precision") or config.get("fp_precision")
        dtype_name = str(dtype_config).lower() if dtype_config is not None else "float64"
        if dtype_name in {"float32", "f32", "32"}:
            ti_dtype = ti.f32
            np_dtype = np.float32
        elif dtype_name in {"float16", "f16", "16"}:
            ti_dtype = ti.f16
            np_dtype = np.float16
        else:
            ti_dtype = ti.f64
            np_dtype = np.float64
        self._numpy_dtype = np_dtype

        arch_value = config.get("arch")
        if isinstance(arch_value, str):
            arch_key = arch_value.lower()
            if ti is None:
                raise RuntimeError("Taichi is not available")
            if arch_key == "gpu":
                ti.warn(
                    "Taichi GPU backend is experimental: float32 recommended, performance still slower than CPU path."
                )
                init_kwargs.setdefault("arch", ti.gpu)
            elif hasattr(ti, arch_key):
                init_kwargs.setdefault("arch", getattr(ti, arch_key))
            else:
                raise ValueError(f"Unsupported Taichi arch '{arch_value}'")
        elif arch_value is not None:
            init_kwargs.setdefault("arch", arch_value)
        else:
            init_kwargs.setdefault("arch", ti.cpu)

        threads = config.get("cpu_max_num_threads")
        if isinstance(threads, str) and threads.lower() == "auto":
            threads = os.cpu_count() or 1
        if threads is not None:
            init_kwargs.setdefault("cpu_max_num_threads", int(threads))

        if "device_memory_fraction" in config:
            init_kwargs.setdefault(
                "device_memory_fraction", float(config["device_memory_fraction"])
            )

        # Optional micro-iteration loop to amortize Python<->Taichi launches.
        self._inner_iterations = int(config.get("inner_iterations", 1))
        if self._inner_iterations < 1:
            raise ValueError("inner_iterations must be >= 1")

        self._loop_parallelize = None
        parallelize_cfg = config.get("parallelize")
        if parallelize_cfg is not None:
            if isinstance(parallelize_cfg, str):
                key = parallelize_cfg.lower()
                if key == "auto":
                    auto_threads = config.get("cpu_max_num_threads")
                    if isinstance(auto_threads, str) and auto_threads.lower() == "auto":
                        auto_threads = os.cpu_count() or 1
                    elif auto_threads is None:
                        auto_threads = init_kwargs.get("cpu_max_num_threads") or os.cpu_count() or 1
                    self._loop_parallelize = int(auto_threads)
                else:
                    if not key.isdigit():
                        raise ValueError("parallelize must be an integer or 'auto'")
                    self._loop_parallelize = int(key)
            else:
                self._loop_parallelize = int(parallelize_cfg)
            if self._loop_parallelize is not None and self._loop_parallelize < 1:
                self._loop_parallelize = None

        self._loop_block_dim = None
        block_dim_cfg = config.get("block_dim")
        if block_dim_cfg is not None:
            if isinstance(block_dim_cfg, str):
                key = block_dim_cfg.lower()
                if key == "auto":
                    self._loop_block_dim = 64
                else:
                    if not key.isdigit():
                        raise ValueError("block_dim must be an integer or 'auto'")
                    self._loop_block_dim = int(key)
            else:
                self._loop_block_dim = int(block_dim_cfg)
            if self._loop_block_dim is not None and self._loop_block_dim < 1:
                self._loop_block_dim = None

        self._use_vectorized = bool(config.get("use_vectorized", False))
        vector_width_cfg = int(config.get("vector_width", 4))
        if vector_width_cfg < 1:
            vector_width_cfg = 1
        self._vector_width = vector_width_cfg

        self._tile_shape = None
        tile_cfg = config.get("tile_shape")
        if tile_cfg is not None:
            tile_y: Optional[int] = None
            tile_x: Optional[int] = None
            if isinstance(tile_cfg, str):
                key = tile_cfg.lower()
                if key == "auto":
                    threads_hint = self._loop_parallelize or init_kwargs.get("cpu_max_num_threads")
                    if isinstance(threads_hint, str) and threads_hint.lower() == "auto":
                        threads_hint = os.cpu_count() or 1
                    elif threads_hint is None:
                        threads_hint = os.cpu_count() or 1
                    threads_hint = int(threads_hint)
                    base = max(1, int(math.sqrt(max(1, threads_hint))))
                    tile_y = max(1, min(ny, base * 2))
                    tile_x = max(1, min(nx, base * 2))
                else:
                    parts = key.replace("x", " ").replace(",", " ").split()
                    if len(parts) == 1:
                        tile_y = tile_x = int(parts[0])
                    elif len(parts) >= 2:
                        tile_y = int(parts[0])
                        tile_x = int(parts[1])
            elif isinstance(tile_cfg, (list, tuple)):
                if len(tile_cfg) == 1:
                    tile_y = tile_x = int(tile_cfg[0])
                elif len(tile_cfg) >= 2:
                    tile_y = int(tile_cfg[0])
                    tile_x = int(tile_cfg[1])
            elif isinstance(tile_cfg, (int, float)):
                tile_y = tile_x = int(tile_cfg)

            if tile_y is not None and tile_x is not None:
                tile_y = max(1, min(ny, tile_y))
                tile_x = max(1, min(nx, tile_x))
                if tile_y > 1 or tile_x > 1:
                    self._tile_shape = (tile_y, tile_x)
                    if self._loop_block_dim is None:
                        block_dim = tile_y * tile_x
                        if block_dim > 0:
                            block_dim = 1 << int(math.floor(math.log2(block_dim)))
                            block_dim = max(1, block_dim)
                        self._loop_block_dim = block_dim

        global taichi_init_done, _taichi_init_config, _taichi_init_signature
        signature = _signature_for_kwargs(init_kwargs)
        if not taichi_init_done:
            ti.init(**init_kwargs)
            taichi_init_done = True
            _taichi_init_config = dict(init_kwargs)
            _taichi_init_signature = signature
        else:
            if _taichi_init_signature and _taichi_init_signature != signature:
                raise RuntimeError(
                    "Taichi runtime already initialised with a different configuration."
                )

        # Create global fields if this is the first initialization.
        global psi_r_field, psi_i_field, chi_field, chi_dot_field, dissonance_field
        global psi_r_snapshot_field, psi_i_snapshot_field, chi_snapshot_field, chi_dot_snapshot_field, dissonance_snapshot_field
        if not hasattr(self, "_fields_initialized"):
            psi_r_field = ti.field(dtype=ti_dtype)
            psi_i_field = ti.field(dtype=ti_dtype)
            chi_field = ti.field(dtype=ti_dtype)
            chi_dot_field = ti.field(dtype=ti_dtype)
            dissonance_field = ti.field(dtype=ti_dtype)
            psi_r_snapshot_field = ti.field(dtype=ti_dtype)
            psi_i_snapshot_field = ti.field(dtype=ti_dtype)
            chi_snapshot_field = ti.field(dtype=ti_dtype)
            chi_dot_snapshot_field = ti.field(dtype=ti_dtype)
            dissonance_snapshot_field = ti.field(dtype=ti_dtype)
            ti.root.dense(ti.ij, (ny, nx)).place(
                psi_r_field,
                psi_i_field,
                chi_field,
                chi_dot_field,
                dissonance_field,
                psi_r_snapshot_field,
                psi_i_snapshot_field,
                chi_snapshot_field,
                chi_dot_snapshot_field,
                dissonance_snapshot_field,
            )
            self._fields_initialized = True

        global relaxation_rate_scalar
        relaxation_rate_scalar = np_dtype(float(config.get("relaxation_rate", 0.01)))

        ordered_names = list(self.graph.order)
        ordered_nodes = [self.graph.nodes[name] for name in ordered_names]
        variable_names = {var for node in ordered_nodes for var in node.kernel.variables}
        target_names = {node.target_field for node in ordered_nodes}
        required_fields_all = variable_names.union(target_names)
        unsupported = required_fields_all.difference(FIELD_VAR_MAP.keys())
        if unsupported:
            raise ValueError(f"Unsupported fields for Taichi backend: {sorted(unsupported)}")

        self._active_fields = sorted(required_fields_all)
        self._field_objects = {name: globals()[FIELD_VAR_MAP[name]] for name in self._active_fields}
        self._snapshot_field_objects = {
            name: globals()[FIELD_SNAPSHOT_MAP[name]] for name in self._active_fields
        }

        fusion_cfg_raw = config.get("fusion")
        fusion_policy = str(config.get("fusion_policy", "")).lower()
        if isinstance(fusion_cfg_raw, str) and fusion_cfg_raw.lower() == "auto":
            fusion_cfg: List[Dict[str, Any]] = [
                {"name": "all_constraints", "nodes": ordered_names}
            ]
        elif isinstance(fusion_cfg_raw, list) and fusion_cfg_raw:
            fusion_cfg = fusion_cfg_raw  # type: ignore[assignment]
        elif fusion_policy == "auto":
            fusion_cfg = [{"name": "all_constraints", "nodes": ordered_names}]
        elif fusion_policy == "per-node":
            fusion_cfg = [{"name": name, "nodes": [name]} for name in ordered_names]
        else:
            fusion_cfg = []

        node_lookup = {node.name: node for node in ordered_nodes}
        assigned: set[str] = set()
        kernels: List[Callable[[], None]] = []
        kernel_sources: List[str] = []

        def build_kernel(node_names: List[str], inner: int) -> Callable[[], None]:
            if not node_names:
                raise ValueError("Taichi fusion group produced an empty node list")

            required_fields_group: set[str] = set()
            for name in node_names:
                node = node_lookup[name]
                required_fields_group.update(node.kernel.variables)
                required_fields_group.add(node.target_field)
            required_fields_sorted = sorted(required_fields_group)

            psi_real_sym = sp.Symbol("psi_real")
            psi_imag_sym = sp.Symbol("psi_imag")
            chi_sym = sp.Symbol("chi")
            chi_dot_sym = sp.Symbol("chi_dot")
            psi_norm_expr = psi_real_sym**2 + psi_imag_sym**2
            psi_norm_eps_expr = psi_norm_expr + sp.Float("1.0e-3")
            derived_symbol_map = {
                psi_real_sym**2: sp.Symbol("psi_real_sq"),
                psi_imag_sym**2: sp.Symbol("psi_imag_sq"),
                chi_sym**2: sp.Symbol("chi_sq"),
                chi_dot_sym**2: sp.Symbol("chi_dot_sq"),
                psi_real_sym**3: sp.Symbol("psi_real_cu"),
                psi_imag_sym**3: sp.Symbol("psi_imag_cu"),
                chi_sym**3: sp.Symbol("chi_sq_chi"),
                chi_dot_sym**3: sp.Symbol("chi_dot_sq_chi_dot"),
                psi_real_sym * psi_imag_sym**2: sp.Symbol("psi_real_imag_sq"),
                psi_imag_sym * psi_real_sym**2: sp.Symbol("psi_imag_real_sq"),
                psi_real_sym**2 * psi_imag_sym: sp.Symbol("psi_real_sq_imag"),
                psi_imag_sym**2 * psi_real_sym: sp.Symbol("psi_imag_sq_real"),
                chi_dot_sym * psi_norm_expr: sp.Symbol("chi_dot_norm_sq"),
                psi_real_sym * psi_imag_sym: sp.Symbol("psi_real_psi_imag"),
                psi_norm_eps_expr: sp.Symbol("psi_norm_sq_eps"),
                sp.log(psi_norm_eps_expr): sp.Symbol("log_psi_norm_sq_eps"),
                psi_real_sym**4: sp.Symbol("psi_real_pow4"),
                psi_imag_sym**4: sp.Symbol("psi_imag_pow4"),
                psi_norm_expr**2: sp.Symbol("psi_norm_sq_sq"),
                psi_real_sym**2 * psi_imag_sym**2: sp.Symbol("psi_sq_cross"),
                chi_sym**4: sp.Symbol("chi_sq_sq"),
                chi_dot_sym**4: sp.Symbol("chi_dot_sq_sq"),
                chi_sym**2 * psi_norm_expr: sp.Symbol("chi_sq_norm_sq"),
                chi_dot_sym**2 * psi_norm_expr: sp.Symbol("chi_dot_sq_norm_sq"),
                chi_sym * psi_real_sym: sp.Symbol("chi_psi_real"),
                chi_sym * psi_imag_sym: sp.Symbol("chi_psi_imag"),
                chi_dot_sym * psi_real_sym: sp.Symbol("chi_dot_psi_real"),
                chi_dot_sym * psi_imag_sym: sp.Symbol("chi_dot_psi_imag"),
            }
            norm_sym = sp.Symbol("psi_norm_sq")

            if self._grid_shape is None:
                raise RuntimeError("grid shape not initialised for Taichi backend")
            grid_y, grid_x = self._grid_shape

            use_tiling = self._tile_shape is not None
            tile_y = 1
            tile_x = 1
            block_local_lines: List[str] = []
            if use_tiling:
                tile_y, tile_x = self._tile_shape if self._tile_shape is not None else (1, 1)
                block_local_fields = sorted({FIELD_VAR_MAP[name] for name in required_fields_sorted})
                if block_local_fields:
                    block_local_lines.append(
                        "        ti.block_local(" + ", ".join(block_local_fields) + ")"
                    )

            vectorized = (
                use_tiling
                and self._use_vectorized
                and self._vector_width > 1
                and tile_x >= self._vector_width
            )
            vector_width = self._vector_width if vectorized else 1

            if vectorized:
                local_map = {name: f"{name}_vec" for name in required_fields_sorted}
                chunk_count = max(1, (tile_x + vector_width - 1) // vector_width)
                outer_y = (grid_y + tile_y - 1) // tile_y
                outer_x = (grid_x + tile_x - 1) // tile_x
                loop_lines = [
                    f"        for block_y, block_x in ti.ndrange({outer_y}, {outer_x}):",
                    f"            base_y = block_y * {tile_y}",
                    f"            base_x = block_x * {tile_x}",
                    f"            for oy in range({tile_y}):",
                    f"                iy = base_y + oy",
                    f"                if iy >= {grid_y}:",
                    "                    continue",
                    f"                for chunk in range({chunk_count}):",
                    f"                    ox = chunk * {vector_width}",
                    f"                    ix = base_x + ox",
                    f"                    if ix >= {grid_x}:",
                    "                        continue",
                    "                    I = ti.Vector([iy, ix])",
                ]
                load_indent = "                    "
            elif use_tiling:
                local_map = {name: f"{name}_val" for name in required_fields_sorted}
                outer_y = (grid_y + tile_y - 1) // tile_y
                outer_x = (grid_x + tile_x - 1) // tile_x
                loop_lines = [
                    f"        for block_y, block_x in ti.ndrange({outer_y}, {outer_x}):",
                    f"            base_y = block_y * {tile_y}",
                    f"            base_x = block_x * {tile_x}",
                    f"            for oy in range({tile_y}):",
                    f"                iy = base_y + oy",
                    f"                if iy >= {grid_y}:",
                    "                    continue",
                    f"                for ox in range({tile_x}):",
                    f"                    ix = base_x + ox",
                    f"                    if ix >= {grid_x}:",
                    "                        continue",
                    "                    I = ti.Vector([iy, ix])",
                ]
                load_indent = "                    "
            else:
                local_map = {name: f"{name}_val" for name in required_fields_sorted}
                loop_lines = ["        for I in ti.grouped(psi_r_field):"]
                load_indent = "            "

            translator = TaichiExpressionTranslator(local_map)

            load_lines: List[str] = []
            if vectorized:
                for name in required_fields_sorted:
                    vec_var = local_map[name]
                    source_field = FIELD_SNAPSHOT_MAP[name]
                    field_obj = FIELD_VAR_MAP[name]
                    load_lines.extend(
                        [
                            f"{load_indent}{vec_var} = ti.Vector.zero({field_obj}.dtype, {vector_width})",
                            f"{load_indent}for lane in ti.static(range({vector_width})):",
                            f"{load_indent}    lane_x = ix + lane",
                            f"{load_indent}    lane_I = ti.Vector([iy, lane_x])",
                            f"{load_indent}    if lane_x < {grid_x}:",
                            f"{load_indent}        {vec_var}[lane] = {source_field}[lane_I]",
                            f"{load_indent}    else:",
                            f"{load_indent}        {vec_var}[lane] = 0",
                        ]
                    )
            else:
                load_lines = [
                    f"{load_indent}{local_map[name]} = {FIELD_SNAPSHOT_MAP[name]}[I]"
                    for name in required_fields_sorted
                ]
            derived_lines: List[str] = []
            def _append_square(symbol: str, base: str) -> None:
                if symbol not in local_map and base in local_map:
                    local_map[symbol] = symbol
                    derived_lines.append(
                        f"{load_indent}{symbol} = {local_map[base]} * {local_map[base]}"
                    )

            def _append_product(symbol: str, a: str, b: str) -> None:
                if symbol not in local_map and a in local_map and b in local_map:
                    local_map[symbol] = symbol
                    derived_lines.append(
                        f"{load_indent}{symbol} = {local_map[a]} * {local_map[b]}"
                    )

            _append_square("psi_real_sq", "psi_real")
            _append_square("psi_imag_sq", "psi_imag")
            _append_square("chi_sq", "chi")
            _append_square("chi_dot_sq", "chi_dot")

            if "psi_norm_sq" not in local_map and "psi_real" in local_map and "psi_imag" in local_map:
                local_map["psi_norm_sq"] = "psi_norm_sq"
                derived_lines.append(
                    f"{load_indent}psi_norm_sq = {local_map['psi_real']} * {local_map['psi_real']} + {local_map['psi_imag']} * {local_map['psi_imag']}"
                )

            if "psi_norm_sq" in local_map and "psi_norm_sq_eps" not in local_map:
                local_map["psi_norm_sq_eps"] = "psi_norm_sq_eps"
                derived_lines.append(
                    f"{load_indent}psi_norm_sq_eps = {local_map['psi_norm_sq']} + 1.0e-3"
                )
            if "psi_norm_sq_eps" in local_map and "log_psi_norm_sq_eps" not in local_map:
                local_map["log_psi_norm_sq_eps"] = "log_psi_norm_sq_eps"
                derived_lines.append(
                    f"{load_indent}log_psi_norm_sq_eps = ti.log({local_map['psi_norm_sq_eps']})"
                )

            _append_square("psi_real_pow4", "psi_real_sq")
            _append_square("psi_imag_pow4", "psi_imag_sq")
            if "psi_norm_sq" in local_map:
                _append_square("psi_norm_sq_sq", "psi_norm_sq")

            _append_product("psi_real_cu", "psi_real_sq", "psi_real")
            _append_product("psi_imag_cu", "psi_imag_sq", "psi_imag")
            _append_product("psi_real_psi_imag", "psi_real", "psi_imag")
            _append_product("psi_real_imag_sq", "psi_real", "psi_imag_sq")
            _append_product("psi_imag_real_sq", "psi_imag", "psi_real_sq")
            _append_product("psi_real_sq_imag", "psi_real_sq", "psi_imag")
            _append_product("psi_imag_sq_real", "psi_imag_sq", "psi_real")
            _append_product("chi_sq_chi", "chi_sq", "chi")
            _append_product("chi_dot_sq_chi_dot", "chi_dot_sq", "chi_dot")
            _append_product("chi_dot_norm_sq", "chi_dot", "psi_norm_sq")
            _append_product("psi_sq_cross", "psi_real_sq", "psi_imag_sq")
            _append_product("chi_sq_sq", "chi_sq", "chi_sq")
            _append_product("chi_dot_sq_sq", "chi_dot_sq", "chi_dot_sq")
            _append_product("chi_sq_norm_sq", "chi_sq", "psi_norm_sq")
            _append_product("chi_dot_sq_norm_sq", "chi_dot_sq", "psi_norm_sq")
            _append_product("chi_psi_real", "chi", "psi_real")
            _append_product("chi_psi_imag", "chi", "psi_imag")
            _append_product("chi_dot_psi_real", "chi_dot", "psi_real")
            _append_product("chi_dot_psi_imag", "chi_dot", "psi_imag")

            load_lines.extend(derived_lines)

            constraint_lines: List[str] = []
            for idx, node_name in enumerate(node_names):
                node = node_lookup[node_name]
                expr = node.kernel.sym_expr
                expr = expr.xreplace(derived_symbol_map)
                expr = expr.xreplace({
                    sp.Symbol("psi_real_sq") + sp.Symbol("psi_imag_sq"): norm_sym,
                    sp.Symbol("psi_imag_sq") + sp.Symbol("psi_real_sq"): norm_sym,
                    psi_real_sym**2 + psi_imag_sym**2: norm_sym,
                    psi_imag_sym**2 + psi_real_sym**2: norm_sym,
                })
                expr_code = translator.translate(expr)
                target = node.target_field
                base = local_map[target]
                target_field = FIELD_VAR_MAP[target]
                update_expr = f"{base} - relaxation_rate_scalar * residual_{idx}"
                if vectorized:
                    update_var = f"{target}_update_vec_{idx}"
                    constraint_lines.extend(
                        [
                            f"{load_indent}actual_{idx} = {expr_code}",
                            f"{load_indent}residual_{idx} = {base} - actual_{idx}",
                            f"{load_indent}{update_var} = {update_expr}",
                            f"{load_indent}for lane in ti.static(range({vector_width})):",
                            f"{load_indent}    lane_x = ix + lane",
                            f"{load_indent}    if lane_x < {grid_x}:",
                            f"{load_indent}        lane_I = ti.Vector([iy, lane_x])",
                            f"{load_indent}        {target_field}[lane_I] = {update_var}[lane]",
                        ]
                    )
                else:
                    update_var = f"{target}_update_val_{idx}"
                    constraint_lines.extend(
                        [
                            f"{load_indent}actual_{idx} = {expr_code}",
                            f"{load_indent}residual_{idx} = {base} - actual_{idx}",
                            f"{load_indent}{update_var} = {update_expr}",
                            f"{load_indent}{target_field}[I] = {update_var}",
                        ]
                    )

            inner_value = max(int(inner), 1)
            kernel_header = [
                "@ti.kernel",
                "def taichi_constraint_step():",
                f"    for _inner in ti.static(range({inner_value})):",
            ]
            if self._loop_parallelize or self._loop_block_dim:
                if self._loop_parallelize:
                    kernel_header.append(
                        f"        ti.loop_config(parallelize={self._loop_parallelize})"
                    )
                if self._loop_block_dim:
                    kernel_header.append(
                        f"        ti.loop_config(block_dim={self._loop_block_dim})"
                    )

            kernel_body = "\n".join(
                kernel_header + block_local_lines + loop_lines + load_lines + constraint_lines
            )

            namespace = {
                "ti": ti,
                "relaxation_rate_scalar": relaxation_rate_scalar,
                "psi_r_field": psi_r_field,
                "psi_i_field": psi_i_field,
                "chi_field": chi_field,
                "chi_dot_field": chi_dot_field,
                "dissonance_field": dissonance_field,
                "psi_r_snapshot_field": psi_r_snapshot_field,
                "psi_i_snapshot_field": psi_i_snapshot_field,
                "chi_snapshot_field": chi_snapshot_field,
                "chi_dot_snapshot_field": chi_dot_snapshot_field,
                "dissonance_snapshot_field": dissonance_snapshot_field,
            }
            cache_key = f"<taichi_constraint_step_{'_'.join(node_names)}>"
            linecache.cache[cache_key] = (
                len(kernel_body),
                None,
                [line + "\n" for line in kernel_body.splitlines()],
                cache_key,
            )
            code = compile(kernel_body, filename=cache_key, mode="exec")
            exec(code, namespace)
            kernel_sources.append(kernel_body)
            return namespace["taichi_constraint_step"]

        if isinstance(fusion_cfg, list):
            for group in fusion_cfg:
                nodes_cfg = group.get("nodes")
                if not nodes_cfg:
                    continue
                node_names = [str(name) for name in nodes_cfg]
                for name in node_names:
                    if name not in node_lookup:
                        raise ValueError(f"Fusion group references unknown constraint '{name}'")
                inner = group.get("inner_iterations", self._inner_iterations)
                kernels.append(build_kernel(node_names, inner))
                assigned.update(node_names)

        remaining = [name for name in ordered_names if name not in assigned]
        if remaining:
            kernels.append(build_kernel(remaining, self._inner_iterations))

        if not kernels:
            raise RuntimeError("Failed to build any Taichi constraint kernels")

        self._taichi_kernels = kernels
        self._kernel_source = "\n\n".join(kernel_sources)

    def _copy_numpy_to_taichi(self, state_buffers: Dict[str, np.ndarray]) -> None:
        for field_name, ti_field in self._field_objects.items():
            array = state_buffers.get(field_name)
            if array is None:
                raise RuntimeError(f"state buffer missing required field '{field_name}'")
            cast_array = array.astype(self._numpy_dtype, copy=False)
            ti_field.from_numpy(cast_array)
            snapshot_field = self._snapshot_field_objects[field_name]
            snapshot_field.from_numpy(cast_array)

    def _copy_taichi_to_numpy(self, state_buffers: Dict[str, np.ndarray]) -> None:
        for field_name, ti_field in self._field_objects.items():
            array = state_buffers.get(field_name)
            if array is None:
                raise RuntimeError(f"state buffer missing required field '{field_name}'")
            array[:, :] = ti_field.to_numpy()
