"""
Runtime orchestration for Δ-Machine multiprocessing control.

Design references:
- 1.0 Δ-Computing Paradigm Definition:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.0_Delta_machine_notes.md
- 1.2 Functional Design Concept:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.2_possible_design_concept.md
- 1.3 Design Evaluation & Recommendations:
  /Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/Particle Derivations/Optimizer new tests/Delta_machine/notes/1.3_design_evaluation.md
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from time import time_ns
from multiprocessing import Process, get_context
from multiprocessing.synchronize import Event as SyncEvent
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional, List

import numpy as np
import psutil
from collections import deque

try:
    from pr0_system.bootstrap.dissonance import compute_ontological_dissonance
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def compute_ontological_dissonance(
        psi: np.ndarray,
        chi: np.ndarray,
        history: list[np.ndarray],
    ) -> float:
        psi_norm = float(np.linalg.norm(psi))
        chi_norm = float(np.linalg.norm(chi))
        history_norm = sum(float(np.linalg.norm(h)) for h in history)
        return psi_norm + chi_norm + history_norm

from .analysis import (
    TSPConfig,
    apply_sinkhorn_step,
    compute_tsp_metrics,
    embed_assignment_matrix,
    extract_assignment_matrix,
    get_tsp_config,
)
from .config import ScenarioSpec
from .constraints import ConstraintGraph, ConstraintNode
from .functionals import FunctionalCompiler
from .shared_state import SharedState, attach_shared_state
from .diagnostics import RunLogger, create_run_directory, save_field_snapshot, save_final_preview
from .initial_conditions import InitialConditionGenerator
from .scenarios import ScenarioRunner
from .reporting import generate_report, save_final_state
from .logging_utils import configure_run_logger
from .pr0_adapter import PR0FieldBundle, create_field_bundle, should_use_pr0
from .meta_control import MetaControlConfig, ReactionFieldController, load_meta_control_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERT_SCRIPT = PROJECT_ROOT / "Testing_scripts" / "convert_snapshot.py"


@dataclass(slots=True)
class WorkerStatus:
    worker_id: int
    last_heartbeat: float
    processed_steps: int = 0


@dataclass(slots=True)
class OrchestratorTelemetry:
    timestamp: float
    total_dissonance: float
    worker_load: Dict[int, WorkerStatus]
    cpu_percent: float
    memory_percent: float


def worker_entry(
    worker_id: int,
    descriptor: Dict[str, tuple[str, tuple[int, int]]],
    row_slice: slice,
    command_queue: Any,
    status_queue: Any,
    shutdown_event: SyncEvent,
    constraint_graph: ConstraintGraph,
    dtype: np.dtype = np.float64,
):
    # Optional timing instrumentation
    import os
    import sys
    try:
        # Add project root to path for timing instrumentation
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from DSAC_tools.timing_instrumentation import profile_section
    except ImportError:
        # No-op context manager if profiling not available
        class profile_section:
            def __init__(self, *args):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
    
    arrays, handles = attach_shared_state(descriptor, dtype=dtype)
    status_queue.put((worker_id, "started", monotonic()))

    relaxation_rate = 0.01

    while not shutdown_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
        except Exception:
            continue

        if command == "step":
            with profile_section(f"worker_{worker_id}_step"):
                with profile_section(f"worker_{worker_id}_constraint_eval"):
                    partial_state = {name: arr[row_slice] for name, arr in arrays.items()}
                    residuals = constraint_graph.evaluate(partial_state)
                with profile_section(f"worker_{worker_id}_field_update"):
                    node_map = constraint_graph.nodes
                    for node_name, residual in residuals.items():
                        node = node_map[node_name]
                        target = node.target_field
                        if target in arrays:
                            arrays[target][row_slice] -= relaxation_rate * residual
            status_queue.put((worker_id, "step", monotonic()))
        elif command == "shutdown":
            break

    status_queue.put((worker_id, "stopped", monotonic()))
    for shm in handles.values():
        shm.close()


@dataclass
class DeltaOrchestrator:
    scenario: ScenarioSpec
    compiler: FunctionalCompiler
    max_workers: int = 6
    run_base_dir: Path | None = None
    iterations_per_step: int = 1
    initial_condition: InitialConditionGenerator | None = None
    initial_condition_name: str | None = None
    initial_condition_seed: int | None = None
    freeze_fields: bool = False
    shared_state: SharedState | None = None
    use_pr0: bool = False
    pr0_bundle: PR0FieldBundle | None = None
    workers: Dict[int, Process] = field(default_factory=dict)
    statuses: Dict[int, WorkerStatus] = field(default_factory=dict)
    command_queue: Any = None
    status_queue: Any = None
    shutdown_event: SyncEvent | None = None
    constraint_graph: ConstraintGraph | None = None
    scenario_runner: ScenarioRunner | None = None
    last_dissonance: float = 0.0
    step_counter: int = 0
    run_dir: Path | None = None
    logger: RunLogger | None = None
    runtime_logger: Any | None = None
    psi_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=20))
    halted: bool = False
    halt_reason: str | None = None
    tsp_config: TSPConfig | None = None
    tsp_cycle_weight: float = 0.05
    last_tsp_metrics: Dict[str, Any] | None = None
    tsp_rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    tsp_cost_gain: float = 0.0
    plateau_events: List[Dict[str, Any]] = field(default_factory=list)
    oscillation_events: List[Dict[str, Any]] = field(default_factory=list)
    oscillation_guard: Dict[str, Any] | None = None
    oscillation_history: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _last_initial_seed: int | None = field(default=None, init=False)
    oscillation_rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    meta_control_config: MetaControlConfig | None = None
    meta_controller: ReactionFieldController | None = None
    meta_last_actions: Dict[str, Any] | None = None
    meta_drive_scale: float = 1.0
    meta_beta_boost: float = 0.0
    meta_sinkhorn_extra: int = 0
    meta_events: List[Dict[str, Any]] = field(default_factory=list)
    meta_sinkhorn_extra_slow: int = 0
    meta_subtour_scale: float = 1.0
    meta_observable_history: deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1000))
    meta_drive_history: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    meta_plateau_streak: int = 0
    _prev_plateau_dissonance: float | None = field(default=None, init=False, repr=False)
    lattice_usage_interval: int = 25
    lattice_usage_snapshot: Dict[str, Any] = field(default_factory=dict)
    backend: str = "numpy"  # 'numpy' (default) or 'taichi' (experimental)
    backend_config: Dict[str, Any] | None = None

    def initialize(self):
        if self.shared_state is not None:
            return

        self.scenario_runner = ScenarioRunner(self.scenario)
        self.scenario_runner.reset_logs()
        shared_dtype = np.float64
        if self.backend == "taichi" and self.backend_config:
            dtype_config = (
                self.backend_config.get("dtype")
                or self.backend_config.get("precision")
                or self.backend_config.get("fp_precision")
            )
            if dtype_config is not None:
                dtype_name = str(dtype_config).lower()
                if dtype_name in {"float32", "f32", "32"}:
                    shared_dtype = np.float32
                elif dtype_name in {"float16", "f16", "16"}:
                    shared_dtype = np.float16
        self.shared_state = SharedState(self.scenario.lattice_shape, dtype=shared_dtype)
        self.use_pr0 = should_use_pr0(self.scenario.metadata)
        self.pr0_bundle = None
        if self.use_pr0:
            self.pr0_bundle = create_field_bundle(self.scenario.lattice_shape, self.scenario.metadata)
        if self.scenario.scenario_type == "tsp_reflexive":
            self.tsp_config = get_tsp_config(self.scenario.metadata or {})
            self.last_tsp_metrics = None
            self.tsp_cycle_weight = 0.05
            self.tsp_cost_gain = 0.0

        self.meta_control_config = load_meta_control_config(self.scenario.metadata)
        self.meta_controller = (
            ReactionFieldController(self.meta_control_config) if self.meta_control_config else None
        )
        self.meta_last_actions = None
        self.meta_drive_scale = 1.0
        self.meta_beta_boost = 0.0
        self.meta_sinkhorn_extra = 0
        self.meta_events.clear()
        self.meta_sinkhorn_extra_slow = 0
        self.meta_subtour_scale = 1.0
        self.meta_observable_history.clear()
        self.meta_drive_history.clear()
        self.meta_plateau_streak = 0
        self._prev_plateau_dissonance = None

        scenario_meta = self.scenario.metadata or {}
        self.freeze_fields = bool(scenario_meta.get("freeze_fields", False))
        interval_override = scenario_meta.get("lattice_usage_interval")
        if interval_override is not None:
            try:
                self.lattice_usage_interval = max(0, int(interval_override))
            except (TypeError, ValueError):
                self.lattice_usage_interval = 25
        self.lattice_usage_snapshot.clear()

        if self.run_base_dir:
            self.run_dir = create_run_directory(self.run_base_dir)
            self.logger = RunLogger(self.run_dir)
            self.runtime_logger = configure_run_logger(self.run_dir)

        arrays = self.shared_state.arrays()
        self._dissonance_locked = False
        self._initialize_fields(arrays)
        if self.pr0_bundle and self.pr0_bundle.field_state:
            self.pr0_bundle.sync_from_arrays(arrays)
        if self.scenario.scenario_type == "tsp_reflexive":
            seed = self._last_initial_seed or (time_ns() & 0xFFFFFFFF)
            self.tsp_rng = np.random.default_rng(seed)
            self.oscillation_rng = np.random.default_rng(seed ^ 0xABCDEF)
            guard_cfg = None
            if self.scenario.metadata:
                guard_cfg = self.scenario.metadata.get("oscillation_guard")
            self.oscillation_guard = guard_cfg

        self.constraint_graph = self._build_constraint_graph(arrays)
        ctx = get_context("spawn")
        self.command_queue = ctx.Queue()
        self.status_queue = ctx.Queue()
        self.shutdown_event = ctx.Event()
        if self.runtime_logger:
            self.runtime_logger.info("Orchestrator initialized for scenario '%s'", self.scenario.name)
            if self.initial_condition_name:
                self.runtime_logger.info("Using initial condition: %s", self.initial_condition_name)
        self.psi_history.clear()
        self.halted = False
        self.halt_reason = None
        self.plateau_events.clear()
        self.oscillation_events.clear()
        self.oscillation_history.clear()

    def _build_constraint_graph(self, arrays: Dict[str, np.ndarray]) -> ConstraintGraph:
        nodes = []
        for spec in self.scenario.constraints:
            kernel = self.compiler.compile(
                spec.name, spec.expression, spec.variables, spec.dependencies, spec.weight
            )
            node = ConstraintNode(
                name=spec.name,
                kernel=kernel,
                target_field=spec.target,
                weight=spec.weight,
                dependencies=list(spec.dependencies),
            )
            nodes.append(node)
        return ConstraintGraph(nodes)

    def _initialize_fields(self, arrays: Dict[str, np.ndarray]) -> None:
        if self.initial_condition:
            seed = self.initial_condition_seed
            if seed is None:
                meta_seed = self.scenario.metadata.get("initial_condition_seed") if self.scenario.metadata else None
                if isinstance(meta_seed, int):
                    seed = meta_seed
                elif meta_seed == "random":
                    seed = None
                if seed is None:
                    seed = time_ns() & 0xFFFFFFFF
            self._last_initial_seed = int(seed)
            ic_fields = self.initial_condition.generate(
                self.scenario.lattice_shape,
                seed=seed,
                metadata=self.scenario.metadata,
            )
            dissonance_assigned = "dissonance" in ic_fields
            for name, value in ic_fields.items():
                if name in arrays:
                    arrays[name][:] = value
            extras = {k: v for k, v in ic_fields.items() if k not in arrays}
            if extras and self.scenario_runner:
                self.scenario_runner.ingest_initial_metadata(extras)
            if "dissonance" in arrays and not dissonance_assigned:
                arrays["dissonance"][:] = 0.0
            self._dissonance_locked = dissonance_assigned
            if self.runtime_logger:
                self.runtime_logger.info(
                    "Initialized fields using generator: %s (seed=%s)",
                    self.initial_condition.get_name(),
                    self._last_initial_seed,
                )
            return

        if self.scenario.initial_conditions:
            for name, value in self.scenario.initial_conditions.items():
                if name in arrays:
                    arrays[name][:] = value
            if self.runtime_logger:
                self.runtime_logger.info("Initialized fields from scenario initial_conditions")
            return

        from .initial_conditions.generators import PatternGenerator

        default_gen = PatternGenerator(amplitude=1.0, vortex_strength=0.7, noise_level=0.03)
        ic_fields = default_gen.generate(
            self.scenario.lattice_shape,
            seed=42,
            metadata=self.scenario.metadata,
        )
        for name, value in ic_fields.items():
            if name in arrays:
                arrays[name][:] = value
        if "dissonance" in arrays:
            arrays["dissonance"][:] = 0.0
        if self.runtime_logger:
            self.runtime_logger.info("Initialized fields with default pattern generator")

    def start_workers(self, worker_count: Optional[int] = None):
        self.initialize()
        if self.backend == "taichi":
            return
        count = min(worker_count or psutil.cpu_count(logical=False) or 2, self.max_workers)
        ctx = get_context("spawn")
        descriptor = self.shared_state.descriptor()
        rows_per_worker = max(1, self.scenario.lattice_shape[0] // count)

        for worker_id in range(count):
            start_row = worker_id * rows_per_worker
            end_row = self.scenario.lattice_shape[0] if worker_id == count - 1 else start_row + rows_per_worker
            row_slice = slice(start_row, end_row)
            process = ctx.Process(
                target=worker_entry,
                args=(
                    worker_id,
                    descriptor,
                    row_slice,
                    self.command_queue,
                    self.status_queue,
                    self.shutdown_event,
                    self.constraint_graph,
                    self.shared_state.dtype,
                ),
                daemon=True,
            )
            process.start()
            self.workers[worker_id] = process
            self.statuses[worker_id] = WorkerStatus(worker_id, monotonic())

    def stop_workers(self):
        if self.workers:
            if self.shutdown_event:
                self.shutdown_event.set()
            for _ in self.workers:
                if self.command_queue:
                    self.command_queue.put("shutdown")
            for process in self.workers.values():
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
            self.workers.clear()
            self.statuses.clear()

        if self.shutdown_event:
            try:
                self.shutdown_event.clear()
            except Exception:
                pass
            self.shutdown_event = None

        if self.command_queue:
            try:
                self.command_queue.close()
                self.command_queue.join_thread()
            except Exception:
                pass
            self.command_queue = None

        if self.status_queue:
            try:
                self.status_queue.close()
                self.status_queue.join_thread()
            except Exception:
                pass
            self.status_queue = None

        if self.shared_state:
            try:
                self.shared_state.close()
            except Exception:
                pass
            try:
                self.shared_state.unlink()
            except Exception:
                pass
            self.shared_state = None

        self.pr0_bundle = None
        self.use_pr0 = False
        self.constraint_graph = None
        self.step_counter = 0
        self.psi_history.clear()
        self.halted = False
        self.halt_reason = None
        self.logger = None
        self.runtime_logger = None
        self.run_dir = None
        self.initial_condition = None
        self.initial_condition_name = None
        self.initial_condition_seed = None
        self._last_initial_seed = None
        self.scenario_runner = None
        self.meta_control_config = None
        self.meta_controller = None
        self.meta_last_actions = None
        self.meta_drive_scale = 1.0
        self.meta_beta_boost = 0.0
        self.meta_sinkhorn_extra = 0
        self.meta_events.clear()
        self.meta_sinkhorn_extra_slow = 0
        self.meta_subtour_scale = 1.0
        self.meta_observable_history.clear()
        self.meta_drive_history.clear()
        self.meta_plateau_streak = 0
        self._prev_plateau_dissonance = None
        self.lattice_usage_interval = 25
        self.lattice_usage_snapshot.clear()

    def step(self):
        if self.halted:
            return

        iterations = max(1, self.iterations_per_step)
        if self.backend == "taichi" and self.constraint_graph is not None:
            # Experimental Taichi backend path (single-process constraint updates)
            self._step_taichi(iterations)
        else:
            # Default behaviour: use worker processes and NumPy-based constraints
            self._step_with_workers(iterations)

        arrays = self.shared_state.arrays()
        self.oscillation_history.append(self.last_dissonance)
        self._maybe_apply_oscillation_guard(arrays)
        self._update_meta_controller(arrays)

        if self.scenario_runner:
            residual_norms = self._compute_residual_norms()
            self.scenario_runner.update_metrics(
                self.step_counter, self.last_dissonance, residual_norms, arrays
            )
            should_halt, halt_reason = self.scenario_runner.check_halting(
                self.step_counter, self.last_dissonance, residual_norms
            )
            if should_halt:
                self.halted = True
                self.halt_reason = halt_reason
                if halt_reason and "dissonance_plateau" in halt_reason:
                    self._record_plateau_event(arrays)
                if self.runtime_logger:
                    self.runtime_logger.info("Run halted: %s", halt_reason)

        # Lattice instrumentation executes at its own cadence
        if self.lattice_usage_interval:
            self._record_lattice_usage(arrays)

        if self.logger:
            telemetry = self.telemetry()
            self.logger.append(telemetry)
            if self.run_dir and self.step_counter % 50 == 0:
                save_field_snapshot(self.run_dir, arrays, self.step_counter)
        if self.runtime_logger:
            activity_norm = float(np.linalg.norm(arrays["psi_real"]) + np.linalg.norm(arrays["psi_imag"]))
            self.runtime_logger.info(
                "Step %d | Dissonance %.6f | Activity Norm %.6f | Workers %d",
                self.step_counter,
                self.last_dissonance,
                activity_norm,
                len(self.workers),
            )

    def _step_with_workers(self, iterations: int) -> None:
        """Advance the system using the existing worker-based constraint updates."""
        if not self.shared_state:
            return
        if not self.freeze_fields:
            scenario_meta = self.scenario.metadata or {}
            drive_enabled = not bool(scenario_meta.get("disable_drive", False))
            relax_enabled = not bool(scenario_meta.get("disable_relaxation", False))
            for _ in range(iterations):
                for _ in self.workers:
                    self.command_queue.put("step")
                self._drain_status()
                arrays = self.shared_state.arrays()
                if self.pr0_bundle and self.pr0_bundle.field_state:
                    self.pr0_bundle.sync_from_arrays(arrays)
                if self.scenario.scenario_type == "tsp_reflexive":
                    self._apply_tsp_sinkhorn(arrays)
                else:
                    if drive_enabled:
                        t = self.step_counter * 0.02
                        L_y, L_x = arrays["psi_real"].shape
                        drive = 0.01 * self.meta_drive_scale * np.sin(np.linspace(0, 2 * np.pi, L_x) + t)
                        arrays["psi_real"] += drive
                        arrays["psi_imag"] -= drive
                    if relax_enabled:
                        arrays["chi"] *= 0.999
                        arrays["chi_dot"] *= 0.998
        else:
            if self.pr0_bundle and self.pr0_bundle.field_state and self.shared_state:
                arrays = self.shared_state.arrays()
                self.pr0_bundle.sync_from_arrays(arrays)
        self._update_dissonance()
        self.step_counter += iterations

    def _step_taichi(self, iterations: int) -> None:
        """Advance the system using the Taichi constraint backend.

        This path skips worker processes and instead applies constraint updates
        directly in-process via the Taichi backend. Meta-control and driving
        logic mirror the worker-based path.
        """
        from .backends import TaichiConstraintBackend

        if not self.shared_state or not self.constraint_graph:
            return

        try:
            from DSAC_tools.timing_instrumentation import profile_section  # type: ignore
        except Exception:  # pragma: no cover - profiling optional
            class profile_section:  # type: ignore
                def __init__(self, *_: object, **__: object) -> None:
                    pass

                def __enter__(self) -> "profile_section":
                    return self

                def __exit__(self, *exc_info: object) -> None:
                    return None

        # Lazily construct the Taichi backend and cache it on first use
        if not hasattr(self, "_taichi_backend"):
            self._taichi_backend = TaichiConstraintBackend(
                self.constraint_graph,
                self.backend_config,
            )

        arrays = self.shared_state.arrays()
        scenario_meta = self.scenario.metadata or {}
        drive_enabled = not bool(scenario_meta.get("disable_drive", False))
        relax_enabled = not bool(scenario_meta.get("disable_relaxation", False))
        if not self.freeze_fields:
            for _ in range(iterations):
                # Constraint relaxation via Taichi
                with profile_section("taichi_backend_step"):
                    self._taichi_backend.step(arrays)
                # PR-0 sync and driving logic match the worker-based path
                if self.pr0_bundle and self.pr0_bundle.field_state:
                    self.pr0_bundle.sync_from_arrays(arrays)
                if self.scenario.scenario_type == "tsp_reflexive":
                    self._apply_tsp_sinkhorn(arrays)
                else:
                    if drive_enabled:
                        t = self.step_counter * 0.02
                        L_y, L_x = arrays["psi_real"].shape
                        drive = 0.01 * self.meta_drive_scale * np.sin(np.linspace(0, 2 * np.pi, L_x) + t)
                        arrays["psi_real"] += drive
                        arrays["psi_imag"] -= drive
                    if relax_enabled:
                        arrays["chi"] *= 0.999
                        arrays["chi_dot"] *= 0.998
        else:
            if self.pr0_bundle and self.pr0_bundle.field_state:
                self.pr0_bundle.sync_from_arrays(arrays)

        self._update_dissonance()
        self.step_counter += iterations

    def _drain_status(self):
        if not self.status_queue:
            return
        messages = 0
        expected = max(1, len(self.workers))
        while messages < expected:
            try:
                worker_id, status, timestamp = self.status_queue.get(timeout=0.1)
                messages += 1
            except Exception:
                break
            if status == "step":
                stat = self.statuses.get(worker_id)
                if stat:
                    stat.processed_steps += 1
                    stat.last_heartbeat = timestamp
            elif status == "started":
                self.statuses[worker_id] = WorkerStatus(worker_id, timestamp)
            elif status == "stopped":
                self.statuses.pop(worker_id, None)

    def _update_dissonance(self):
        arrays = self.shared_state.arrays()
        if getattr(self, "_dissonance_locked", False):
            self.last_dissonance = float(np.mean(arrays["dissonance"]))
            return
        if self.pr0_bundle and self.pr0_bundle.field_state:
            self.pr0_bundle.sync_from_arrays(arrays)
            try:
                self.pr0_bundle.field_state.save_history()
            except Exception:
                pass
        psi = arrays["psi_real"] + 1j * arrays["psi_imag"]
        chi = arrays["chi"]
        self.psi_history.append(psi.copy())
        history = list(self.psi_history)
        dissonance = compute_ontological_dissonance(psi, chi, history)
        if self.pr0_bundle and self.pr0_bundle.field_state:
            self.pr0_bundle.record_dissonance(self.step_counter, float(dissonance))
        arrays["dissonance"][:] = dissonance
        self.last_dissonance = float(dissonance)

    def _compute_residual_norms(self) -> Dict[str, float] | None:
        """Compute residual norms for all constraints."""
        if not self.constraint_graph:
            return None
        arrays = self.shared_state.arrays()
        residuals = {}
        for node in self.constraint_graph.nodes.values():
            try:
                expected = arrays[node.target_field]
                actual = node.kernel.evaluate(arrays)
                residual = expected - actual
                residuals[node.name] = float(np.linalg.norm(residual))
            except Exception:
                residuals[node.name] = float("inf")
        return residuals

    def telemetry(self) -> OrchestratorTelemetry:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return OrchestratorTelemetry(
            timestamp=monotonic(),
            total_dissonance=self.last_dissonance,
            worker_load=self.statuses.copy(),
            cpu_percent=cpu,
            memory_percent=mem,
        )

    async def run_async(self, steps: int):
        for _ in range(steps):
            self.step()
            if self.halted:
                break
            await asyncio.sleep(0)

    def shutdown(self):
        if self.run_dir and self.scenario_runner and self.shared_state:
            arrays = self.shared_state.arrays()
            result = self.scenario_runner.get_result(self.step_counter, self.last_dissonance)
            if self.halted:
                result.halted = True
                result.halt_reason = self.halt_reason
            self._record_lattice_usage(arrays, force=True)
            extras = {}
            if self.use_pr0 and self.pr0_bundle:
                pr0_meta = {
                    "dissonance_history": [
                        {"step": int(step), "value": float(value)}
                        for step, value in self.pr0_bundle.dissonance_history
                    ],
                }
                if self.pr0_bundle.lattice is not None:
                    pr0_meta["lattice_type"] = self.pr0_bundle.lattice.lattice_type
                    pr0_meta["boundary"] = getattr(self.pr0_bundle.lattice, "boundary", None)
                extras["pr0"] = pr0_meta
            if self.plateau_events:
                extras["plateau_events"] = self.plateau_events
            if self.oscillation_events:
                extras["oscillation_events"] = self.oscillation_events
            if self.meta_events or self.meta_observable_history or self.meta_drive_history:
                meta_extra: Dict[str, Any] = {}
                if self.meta_events:
                    meta_extra["events"] = self.meta_events
                if self.meta_observable_history:
                    meta_extra["observables"] = list(self.meta_observable_history)
                if self.meta_drive_history:
                    meta_extra["drive_scale"] = list(self.meta_drive_history)
                extras["meta_control"] = meta_extra
            if self.lattice_usage_snapshot:
                extras["lattice_usage"] = self.lattice_usage_snapshot
            artifact_extras = self.scenario_runner.export_artifacts(self.run_dir)
            if artifact_extras:
                extras.update(artifact_extras)
            self.scenario_runner.reset_logs()
            if not extras:
                extras = None

            generate_report(
                self.run_dir,
                self.scenario,
                result,
                self.initial_condition_name,
                extras=extras,
            )
            save_final_state(self.run_dir, arrays)
            save_final_preview(self.run_dir, arrays)
            if self.use_pr0 and self.pr0_bundle:
                self._save_pr0_artifacts(self.run_dir)
            if self.runtime_logger:
                self.runtime_logger.info(
                    "Generated final report and saved final state (seed=%s)", self._last_initial_seed
                )
            print(f"Δ-Machine run artifacts: {self.run_dir}")
            snapshot_dir = self.run_dir / "snapshots"
            if snapshot_dir.exists() and CONVERT_SCRIPT.exists():
                script_arg = shlex.quote(str(CONVERT_SCRIPT))
                dir_var = shlex.quote(str(snapshot_dir))
                print("Snapshot visualization (bash/zsh):")
                print(
                    f"  snapshot_dir={dir_var}; for f in \"$snapshot_dir\"/*.npy; do python {script_arg} \"$f\"; done"
                )
            elif not CONVERT_SCRIPT.exists():
                print("Snapshot converter script not found; ensure Testing_scripts/convert_snapshot.py exists.")

        self.stop_workers()
        self.logger = None
        self.run_dir = None
        self.shutdown_event = None
        self.scenario_runner = None
        self.initial_condition = None
        self.initial_condition_name = None
        self.initial_condition_seed = None
        self._last_initial_seed = None

    def _save_pr0_artifacts(self, run_dir: Path) -> None:
        if not self.pr0_bundle:
            return
        pr0_dir = run_dir / "pr0"
        pr0_dir.mkdir(exist_ok=True)
        if self.pr0_bundle.dissonance_history:
            history_path = pr0_dir / "dissonance_history.csv"
            data = np.array(self.pr0_bundle.dissonance_history, dtype=float)
            header = "step,dissonance"
            np.savetxt(history_path, data, delimiter=",", header=header, comments="")
        if self.pr0_bundle.field_state:
            np.save(pr0_dir / "psi.npy", self.pr0_bundle.field_state.psi)
            np.save(pr0_dir / "chi.npy", self.pr0_bundle.field_state.chi)
            np.save(pr0_dir / "chi_dot.npy", self.pr0_bundle.field_state.chi_dot)

    def _record_plateau_event(self, arrays: Dict[str, np.ndarray]) -> None:
        event: Dict[str, Any] = {
            "step": self.step_counter,
        }
        tolerance = None
        if self.scenario.halting_criteria:
            tolerance = self.scenario.halting_criteria.dissonance_plateau_tolerance
        if tolerance is not None:
            event["tolerance"] = tolerance
        if self.run_dir:
            plateau_dir = Path(self.run_dir) / "plateau_snapshots"
            plateau_dir.mkdir(exist_ok=True)
            for name, array in arrays.items():
                np.save(plateau_dir / f"{name}_plateau_{self.step_counter:06d}.npy", array)
            event["snapshot_dir"] = str(plateau_dir)
        self.plateau_events.append(event)
        if self.scenario_runner:
            self.scenario_runner.scenario_metrics["plateau_detected"] = 1.0

    def _maybe_apply_oscillation_guard(self, arrays: Dict[str, np.ndarray]) -> None:
        if self.freeze_fields:
            return
        if not self.oscillation_guard:
            return
        window = int(self.oscillation_guard.get("window", 25))
        tolerance = float(self.oscillation_guard.get("tolerance", 5e-4))
        strength = float(self.oscillation_guard.get("perturbation_strength", 0.02))
        if len(self.oscillation_history) < window:
            return
        recent = list(self.oscillation_history)[-window:]
        variance = max(recent) - min(recent)
        if variance > tolerance:
            return

        self._apply_oscillation_perturbation(arrays, strength)
        event = {
            "step": self.step_counter,
            "variance": variance,
            "window": window,
            "tolerance": tolerance,
            "strength": strength,
        }
        self.oscillation_events.append(event)
        if self.runtime_logger:
            self.runtime_logger.info(
                "Oscillation guard triggered at step %d (variance %.6f <= %.6f), applied perturbation",
                self.step_counter,
                variance,
                tolerance,
            )
        if self.scenario_runner:
            self.scenario_runner.scenario_metrics["oscillation_guard_triggered"] = 1.0

    def _apply_oscillation_perturbation(self, arrays: Dict[str, np.ndarray], strength: float) -> None:
        rng = self.oscillation_rng or np.random.default_rng()
        if self.tsp_config is None:
            self.tsp_config = get_tsp_config(self.scenario.metadata or {})
        psi_real = arrays.get("psi_real")
        if psi_real is None or self.tsp_config is None:
            return
        sub = psi_real[
            self.tsp_config.row_slice,
            self.tsp_config.col_slice,
        ]
        noise = rng.normal(0.0, strength, size=sub.shape)
        np.add(sub, noise, out=sub)
        psi_real[self.tsp_config.row_slice, self.tsp_config.col_slice] = sub

    def _update_meta_controller(self, arrays: Dict[str, np.ndarray]) -> None:
        if not self.meta_controller:
            return
        metrics: Dict[str, Any] = {}
        if self.last_tsp_metrics:
            metrics["tsp"] = self.last_tsp_metrics
        meta_inputs = self._compose_meta_metrics()
        if meta_inputs:
            metrics["meta"] = meta_inputs
        actions = self.meta_controller.update(self.step_counter, self.last_dissonance, metrics)
        self.meta_last_actions = actions
        self.meta_drive_scale = float(actions.get("drive_scale", 1.0))
        self.meta_beta_boost = float(actions.get("beta_boost", 0.0))
        self.meta_sinkhorn_extra = max(0, int(actions.get("sinkhorn_extra", 0)))
        self.meta_sinkhorn_extra_slow = max(0, int(actions.get("slow_sinkhorn_extra", 0)))
        self.meta_subtour_scale = float(actions.get("subtour_scale", 1.0))
        if self.scenario_runner:
            self.scenario_runner.scenario_metrics["meta_temperature"] = actions.get("temperature")
            self.scenario_runner.scenario_metrics["meta_reaction"] = actions.get("reaction")
            self.scenario_runner.scenario_metrics["meta_drive_scale"] = self.meta_drive_scale
            self.scenario_runner.scenario_metrics["meta_beta_boost"] = self.meta_beta_boost
            self.scenario_runner.scenario_metrics["meta_sinkhorn_extra_fast"] = self.meta_sinkhorn_extra
            self.scenario_runner.scenario_metrics["meta_sinkhorn_extra_slow"] = self.meta_sinkhorn_extra_slow
            self.scenario_runner.scenario_metrics["meta_subtour_scale"] = self.meta_subtour_scale
        perturb = actions.get("perturbation")
        if perturb:
            strength = float(perturb.get("strength", 0.01))
            reason = perturb.get("reason", "meta_control")
            self._apply_meta_perturbation(arrays, strength, reason)
        slow_obs = actions.get("observables")
        if slow_obs and self.scenario_runner:
            for key, value in slow_obs.items():
                self.scenario_runner.scenario_metrics[f"meta_obs_{key}"] = value

    def _apply_meta_perturbation(self, arrays: Dict[str, np.ndarray], strength: float, reason: str) -> None:
        if self.freeze_fields:
            return
        if strength <= 0:
            return
        rng = self.oscillation_rng
        if rng is None:
            rng = np.random.default_rng()
            self.oscillation_rng = rng
        psi_real = arrays.get("psi_real")
        if psi_real is None:
            return
        if self.scenario.scenario_type == "tsp_reflexive" and self.tsp_config is not None:
            block = psi_real[self.tsp_config.row_slice, self.tsp_config.col_slice]
            block += rng.normal(0.0, strength, size=block.shape)
            psi_real[self.tsp_config.row_slice, self.tsp_config.col_slice] = block
        else:
            psi_real += rng.normal(0.0, strength, size=psi_real.shape)
        event = {
            "step": self.step_counter,
            "strength": strength,
            "reason": reason,
        }
        self.meta_events.append(event)
        if self.runtime_logger:
            self.runtime_logger.info(
                "Meta-control perturbation applied at step %d (reason=%s, strength=%.4f)",
                self.step_counter,
                reason,
                strength,
            )

    def _compose_meta_metrics(self) -> Dict[str, Any]:
        tolerance = 1e-4
        if self.scenario.halting_criteria and self.scenario.halting_criteria.dissonance_plateau_tolerance:
            tolerance = self.scenario.halting_criteria.dissonance_plateau_tolerance
        previous = self._prev_plateau_dissonance
        delta = 0.0 if previous is None else self.last_dissonance - previous
        if abs(delta) <= tolerance:
            self.meta_plateau_streak += 1
        else:
            self.meta_plateau_streak = 0
        self._prev_plateau_dissonance = self.last_dissonance
        window = self.meta_control_config.observables_window if self.meta_control_config else 12
        recent = list(self.oscillation_history)[-max(2, window) :]
        oscillation_amplitude = (max(recent) - min(recent)) if len(recent) >= 2 else 0.0
        cost_gap = None
        subtours = None
        if self.last_tsp_metrics:
            cost_gap = self.last_tsp_metrics.get("tour_cost_gap")
            subtours = self.last_tsp_metrics.get("subtour_count")
        meta_metrics = {
            "plateau_streak": self.meta_plateau_streak,
            "dissonance_delta": delta,
            "oscillation_amplitude": oscillation_amplitude,
            "cost_gap": cost_gap,
            "subtour_count": subtours,
            "drive_scale_prev": self.meta_drive_scale,
        }
        self.meta_drive_history.append(self.meta_drive_scale)
        self.meta_observable_history.append(
            {
                "step": self.step_counter,
                **{k: v for k, v in meta_metrics.items() if v is not None},
            }
        )
        return meta_metrics

    def _record_lattice_usage(self, arrays: Dict[str, np.ndarray], force: bool = False) -> None:
        if not force:
            if self.lattice_usage_interval <= 0:
                return
            if self.step_counter % self.lattice_usage_interval != 0:
                return
        psi_real = arrays.get("psi_real")
        if psi_real is None:
            return
        magnitude = np.abs(psi_real)
        total_shape = magnitude.shape

        row_start = 0
        row_end = total_shape[0]
        col_start = 0
        col_end = total_shape[1]
        margin = int((self.scenario.metadata or {}).get("lattice_usage_margin", 4)) if self.scenario.metadata else 4
        if self.scenario.scenario_type == "tsp_reflexive" and self.tsp_config is not None:
            row_start = max(0, self.tsp_config.row_slice.start - margin)
            row_end = min(total_shape[0], self.tsp_config.row_slice.stop + margin)
            col_start = max(0, self.tsp_config.col_slice.start - margin)
            col_end = min(total_shape[1], self.tsp_config.col_slice.stop + margin)

        roi = magnitude[row_start:row_end, col_start:col_end]
        if roi.size == 0:
            return
        peak = float(np.max(roi))
        if not np.isfinite(peak) or peak <= 0.0:
            return
        threshold = 0.05 * peak
        roi_active = roi >= threshold
        if not np.any(roi_active):
            return
        row_indices = np.where(np.any(roi_active, axis=1))[0]
        col_indices = np.where(np.any(roi_active, axis=0))[0]
        row_min = row_start + int(row_indices[0])
        row_max = row_start + int(row_indices[-1])
        col_min = col_start + int(col_indices[0])
        col_max = col_start + int(col_indices[-1])
        active_fraction_roi = float(roi_active.mean())
        active_fraction_total = float(np.count_nonzero(roi_active) / magnitude.size)

        snapshot: Dict[str, Any] = {
            "step": self.step_counter,
            "peak": peak,
            "threshold": threshold,
            "active_fraction": active_fraction_roi,
            "active_fraction_total": active_fraction_total,
            "bbox": {
                "row_min": row_min,
                "row_max": row_max,
                "col_min": col_min,
                "col_max": col_max,
            },
            "roi_bounds": {
                "row_start": row_start,
                "row_end": row_end - 1,
                "col_start": col_start,
                "col_end": col_end - 1,
                "margin": margin,
            },
            "shape": list(total_shape),
        }
        if self.tsp_config:
            snapshot["tsp_recommended_side"] = int(self.tsp_config.recommended_min_side)
            snapshot["lattice_side"] = int(self.scenario.lattice_shape[0])
        self.lattice_usage_snapshot = snapshot
        if self.scenario_runner:
            self.scenario_runner.scenario_metrics["lattice_active_fraction"] = active_fraction_roi
            self.scenario_runner.scenario_metrics["lattice_active_fraction_total"] = active_fraction_total
            self.scenario_runner.scenario_metrics["lattice_active_bbox"] = snapshot["bbox"]

    def _apply_tsp_sinkhorn(self, arrays: Dict[str, np.ndarray]) -> None:
        if self.tsp_config is None:
            self.tsp_config = get_tsp_config(self.scenario.metadata or {})
        psi_real = arrays.get("psi_real")
        if psi_real is None:
            return
        assignment = extract_assignment_matrix(psi_real, self.tsp_config)
        beta_scale = min(1.0 + max(self.tsp_cost_gain + self.meta_beta_boost, 0.0), 80.0)
        updated = apply_sinkhorn_step(assignment, self.tsp_config, beta_scale=beta_scale)
        metrics = compute_tsp_metrics(updated, self.tsp_config)
        cost_gap = metrics.get("tour_cost_gap")
        target_gap = self.tsp_config.cost_gap_target

        if cost_gap is not None and target_gap is not None:
            if cost_gap > target_gap:
                excess = cost_gap - target_gap
                self.tsp_cost_gain = min(self.tsp_cost_gain * 0.9 + 2.0 + 12.0 * excess, 59.0)
            else:
                self.tsp_cost_gain = max(self.tsp_cost_gain * 0.4, 0.0)
        elif cost_gap is not None:
            self.tsp_cost_gain = min(self.tsp_cost_gain * 0.95 + 1.5 * cost_gap, 50.0)
        else:
            self.tsp_cost_gain = max(self.tsp_cost_gain * 0.8, 0.0)

        new_beta_scale = min(1.0 + max(self.tsp_cost_gain + self.meta_beta_boost, 0.0), 80.0)
        if abs(new_beta_scale - beta_scale) > 1e-6:
            beta_scale = new_beta_scale
            updated = apply_sinkhorn_step(updated, self.tsp_config, beta_scale=beta_scale)
            metrics = compute_tsp_metrics(updated, self.tsp_config)
            cost_gap = metrics.get("tour_cost_gap")

        cycles = metrics.get("cycles", [])
        permutation = metrics.get("permutation", [])
        strong_cycle = bool(permutation) and len(set(permutation)) == self.tsp_config.num_cities and len(cycles) == 1

        rng = self.tsp_rng or np.random.default_rng()
        subtour_scale = max(0.5, self.meta_subtour_scale)

        if cycles and len(cycles) > 1:
            self.tsp_cycle_weight = min((self.tsp_cycle_weight * 1.12 + 0.02) * subtour_scale, 0.8)
            cycle_mask = np.zeros_like(updated)
            for cycle in cycles:
                if len(cycle) < self.tsp_config.num_cities:
                    for a in cycle:
                        for b in cycle:
                            if a != b:
                                cycle_mask[a, b] += 1.0
            updated /= (1.0 + self.tsp_cycle_weight * cycle_mask)

            for i in range(len(cycles)):
                for j in range(i + 1, len(cycles)):
                    best_edge = None
                    best_cost = float("inf")
                    for a in cycles[i]:
                        for b in cycles[j]:
                            cost = self.tsp_config.cost_matrix[a, b]
                            if cost < best_cost:
                                best_cost = cost
                                best_edge = (a, b)
                    if best_edge:
                        bonus = 0.5 * self.tsp_cycle_weight
                        updated[best_edge[0], best_edge[1]] += bonus
                        updated[best_edge[1], best_edge[0]] += 0.25 * bonus

            cost_bias = np.exp(
                -self.tsp_cycle_weight * beta_scale * (self.tsp_config.cost_matrix - np.min(self.tsp_config.cost_matrix))
            )
            updated *= cost_bias

            jitter = rng.normal(0.0, 1e-4, size=updated.shape)
            updated += self.tsp_cycle_weight * jitter
        else:
            decay = 0.65 / subtour_scale
            self.tsp_cycle_weight = max(self.tsp_cycle_weight * decay, 0.02)

        if strong_cycle and target_gap is not None and cost_gap is not None and cost_gap > target_gap:
            cost_matrix = self.tsp_config.cost_matrix
            max_push = min(self.tsp_cost_gain, 12.0)
            for row, current_col in enumerate(permutation):
                current_cost = cost_matrix[row, current_col]
                candidate_order = np.argsort(cost_matrix[row])
                for candidate in candidate_order:
                    if candidate == current_col:
                        continue
                    improvement = current_cost - cost_matrix[row, candidate]
                    if improvement <= 0:
                        continue
                    weight = 0.1 * max_push * improvement
                    updated[row, candidate] += weight
                    updated[row, current_col] = max(updated[row, current_col] - weight, self.tsp_config.epsilon)
                    break

        row_res = np.sum(updated, axis=1, keepdims=True) - 1.0
        col_res = np.sum(updated, axis=0, keepdims=True) - 1.0
        base_imbalance = 0.4 if cycles and len(cycles) > 1 else 0.1
        imbalance_gain = base_imbalance * subtour_scale
        updated /= (1.0 + imbalance_gain * np.abs(row_res))
        updated /= (1.0 + imbalance_gain * np.abs(col_res))

        updated = np.clip(updated, self.tsp_config.epsilon, None)
        iterations = max(3, self.tsp_config.sinkhorn_iterations + self.meta_sinkhorn_extra + self.meta_sinkhorn_extra_slow)
        for _ in range(iterations):
            row_sum = np.sum(updated, axis=1, keepdims=True) + self.tsp_config.epsilon
            updated /= row_sum
            col_sum = np.sum(updated, axis=0, keepdims=True) + self.tsp_config.epsilon
            updated /= col_sum

        updated = apply_sinkhorn_step(updated, self.tsp_config, beta_scale=beta_scale)

        metrics = compute_tsp_metrics(updated, self.tsp_config)
        cost_gap = metrics.get("tour_cost_gap")
        metrics["cost_gain"] = self.tsp_cost_gain
        self.last_tsp_metrics = metrics

        if self.runtime_logger and (self.step_counter % 100 == 0):
            key_metrics = {}
            for key in ("tour_cost_gap", "tour_cost", "stochasticity_verified", "subtour_count", "permutation", "cost_gain"):
                if key in metrics:
                    key_metrics[key] = metrics[key]
            gap_value = cost_gap if cost_gap is not None else float("nan")
            if not np.isnan(gap_value) and (gap_value > (target_gap or 0.0) or self.tsp_cost_gain > 0.0):
                self.runtime_logger.info(
                    "TSP reflexive status | step=%d | cost_gap=%.6f | cost_gain=%.3f",
                    self.step_counter,
                    gap_value,
                    self.tsp_cost_gain,
                )
                self.runtime_logger.info("TSP metrics snapshot: %s", key_metrics)

        embed_assignment_matrix(psi_real, updated, self.tsp_config)
        psi_imag = arrays.get("psi_imag")
        if psi_imag is not None:
            embed_assignment_matrix(psi_imag, updated, self.tsp_config)
        chi = arrays.get("chi")
        if chi is not None:
            chi *= 0.9
            block = chi[self.tsp_config.row_slice, self.tsp_config.col_slice]
            block += row_res + col_res
            cycle_list = metrics.get("cycles", [])
            cycle_count = max(1, len(cycle_list))
            for idx, cycle in enumerate(cycle_list):
                for city in cycle:
                    block[city, :] += idx / cycle_count
                    block[:, city] += idx / cycle_count
            if target_gap is not None and cost_gap is not None and cost_gap > target_gap:
                cost_error = cost_gap - target_gap
                cost_drive = self.tsp_config.cost_matrix - np.min(self.tsp_config.cost_matrix)
                block -= 0.05 * beta_scale * cost_error * cost_drive
        chi_dot = arrays.get("chi_dot")
        if chi_dot is not None:
            chi_dot *= 0.9


