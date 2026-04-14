"""
Main PySide6 window for the Δ-Machine control GUI.

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
from pathlib import Path
from typing import Optional

import math

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

import numpy as np

from ..config import ScenarioLoader, ScenarioSpec
from ..orchestrator import DeltaOrchestrator, OrchestratorTelemetry
from ..functionals import FunctionalCompiler
from ..initial_conditions import load_initial_condition, InitialConditionRegistry


HEARTBEAT_INTERVAL = 1000  # milliseconds
METRIC_HISTORY = 512


@dataclass
class MetricSeries:
    series: QLineSeries
    values: list[float]


class DeltaMachineWindow(QtWidgets.QMainWindow):
    """High-performance live control GUI for the Δ-Machine runtime."""

    def __init__(self, scenario_dir: Path, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Δ-Machine Control")
        self._scenario_loader = ScenarioLoader(scenario_dir)
        self._current_scenario: Optional[ScenarioSpec] = None
        self._compiler = FunctionalCompiler()
        self._run_base_dir = scenario_dir.parent / "runs"
        self._ic_registry = InitialConditionRegistry(scenario_dir.parent)
        self._orchestrator: Optional[DeltaOrchestrator] = None
        self._last_logged_step: int = -1

        self._metric_series: dict[str, MetricSeries] = {}
        self._surface_image = QtGui.QImage()
        self._surface_buffer: Optional[np.ndarray] = None
        self._tsp_positions: list[tuple[float, float]] | None = None

        self._build_ui()
        self._heartbeat = QtCore.QTimer(self)
        self._heartbeat.timeout.connect(self._on_heartbeat)
        self._heartbeat.start(HEARTBEAT_INTERVAL)
        self._axis_x: Optional[QValueAxis] = None
        self._axis_y: Optional[QValueAxis] = None

    def _build_ui(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)

        control_bar = self._build_control_bar()
        layout.addWidget(control_bar)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_activity_tab(), "Activity")
        tabs.addTab(self._build_metrics_tab(), "Metrics")
        layout.addWidget(tabs)

        self.setCentralWidget(root)

    def _build_control_bar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        self.scenario_combo = QtWidgets.QComboBox()
        self._populate_scenarios()
        layout.addWidget(self.scenario_combo, stretch=2)

        load_btn = QtWidgets.QPushButton("Load Scenario")
        load_btn.clicked.connect(self._load_selected_scenario)
        layout.addWidget(load_btn)

        layout.addWidget(QtWidgets.QLabel("Initial Condition:"))
        self.ic_combo = QtWidgets.QComboBox()
        self.ic_combo.setEnabled(False)
        layout.addWidget(self.ic_combo, stretch=1)

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self._start_runtime)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_runtime)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_runtime)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)

        self.status_label = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status_label, stretch=1)

        return widget

    def _build_activity_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.activity_label = QtWidgets.QLabel()
        self.activity_label.setMinimumSize(512, 512)
        self.activity_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.activity_label)

        return widget

    def _build_metrics_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        chart_view = QChartView()
        chart_view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.chart = QChart()
        chart_view.setChart(self.chart)

        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("Steps")
        self._axis_x.setTickCount(6)
        self.chart.addAxis(self._axis_x, QtCore.Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("Value")
        self._axis_y.setTickCount(6)
        self.chart.addAxis(self._axis_y, QtCore.Qt.AlignmentFlag.AlignLeft)

        self._axis_y.setRange(0, 1)

        layout.addWidget(chart_view)

        self._tsp_group = QtWidgets.QGroupBox("TSP Metrics")
        tsp_layout = QtWidgets.QVBoxLayout(self._tsp_group)
        self._tsp_text = QtWidgets.QPlainTextEdit()
        self._tsp_text.setReadOnly(True)
        self._tsp_text.setMinimumHeight(150)
        tsp_layout.addWidget(self._tsp_text)
        self._tsp_group.setVisible(False)
        layout.addWidget(self._tsp_group)
        return widget

    def _populate_scenarios(self):
        self.scenario_combo.clear()
        for path in sorted(self._scenario_loader._root.glob("*.yaml")):
            self.scenario_combo.addItem(path.name)

    def _load_selected_scenario(self):
        name = self.scenario_combo.currentText()
        if not name:
            return
        scenario = self._scenario_loader.load(name)
        self._current_scenario = scenario

        self.ic_combo.clear()
        if scenario.initial_condition_refs:
            for ic_ref in scenario.initial_condition_refs:
                if isinstance(ic_ref, str):
                    self.ic_combo.addItem(ic_ref, ic_ref)
                elif isinstance(ic_ref, dict):
                    ic_name = ic_ref.get("name", ic_ref.get("type", "unnamed"))
                    self.ic_combo.addItem(ic_name, ic_ref)
            self.ic_combo.setEnabled(True)
            self.ic_combo.setCurrentIndex(0)
        else:
            self.ic_combo.addItem("Default", None)
            self.ic_combo.setEnabled(False)

        self._orchestrator = DeltaOrchestrator(scenario, self._compiler, run_base_dir=self._run_base_dir)
        self._tsp_positions = None
        if scenario.metadata and scenario.metadata.get("tsp"):
            cities = scenario.metadata["tsp"].get("city_positions")
            if cities:
                self._tsp_positions = [(float(x), float(y)) for x, y in cities]
        self.run_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.status_label.setText(f"Loaded {scenario.name}")

    def _start_runtime(self):
        if not self._orchestrator or not self._current_scenario:
            return

        if self._orchestrator.workers:
            self._orchestrator.stop_workers()

        if self._orchestrator.shared_state is not None:
            # Existing state from a previous run; rebuild orchestrator to guarantee a clean start.
            self._orchestrator.shutdown()
            self._orchestrator = DeltaOrchestrator(
                self._current_scenario, self._compiler, run_base_dir=self._run_base_dir
            )

        ic_data = self.ic_combo.currentData()
        ic_name = None
        self._orchestrator.initial_condition_seed = None
        if ic_data:
            try:
                if isinstance(ic_data, str):
                    ic = load_initial_condition(self._run_base_dir.parent, ic_data)
                    ic_name = ic_data
                elif isinstance(ic_data, dict):
                    ic = load_initial_condition(self._run_base_dir.parent, ic_data)
                    ic_name = ic_data.get("name", ic_data.get("type", "generated"))
                    seed_value = ic_data.get("seed")
                    if isinstance(seed_value, int):
                        self._orchestrator.initial_condition_seed = seed_value
                    elif seed_value == "random":
                        self._orchestrator.initial_condition_seed = None
                else:
                    ic = None
                if ic:
                    self._orchestrator.initial_condition = ic
                    self._orchestrator.initial_condition_name = ic_name
                    self._orchestrator.initialize()
            except Exception as e:
                self.status_label.setText(f"IC Error: {e}")
                return

        if ic_data is None:
            self._orchestrator.initialize()

        self._orchestrator.start_workers()
        self.status_label.setText("Running")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)

    def _stop_runtime(self):
        if not self._orchestrator:
            return
        self._orchestrator.stop_workers()
        self.status_label.setText("Stopped")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)

    def _reset_runtime(self):
        if not self._current_scenario:
            return
        if self._orchestrator:
            self._orchestrator.shutdown()
        self._orchestrator = DeltaOrchestrator(self._current_scenario, self._compiler, run_base_dir=self._run_base_dir)
        self._reset_activity_view()
        self._reset_metrics()
        self.status_label.setText("Reset")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _reset_activity_view(self):
        self.activity_label.clear()
        self.activity_label.setText("Ready")
        self._surface_buffer = None

    def _reset_metrics(self):
        for metric in self._metric_series.values():
            self.chart.removeSeries(metric.series)
        self._metric_series.clear()
        if self._axis_y:
            self._axis_y.setRange(0, 1)
        if hasattr(self, "_tsp_group"):
            self._tsp_group.setVisible(False)
            self._tsp_text.clear()

    def _on_heartbeat(self):
        if not self._orchestrator or not self._orchestrator.workers:
            return
        self._orchestrator.step()
        if self._orchestrator.halted:
            self.status_label.setText(f"Halted: {self._orchestrator.halt_reason}")
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.reset_btn.setEnabled(True)
        telemetry = self._orchestrator.telemetry()
        self._update_activity()
        self._update_metrics(telemetry)

    def _update_activity(self):
        arrays = self._orchestrator.shared_state.arrays()
        complex_field = arrays["psi_real"] + 1j * arrays["psi_imag"]
        heatmap = np.abs(complex_field)
        min_val = float(np.min(heatmap))
        max_val = float(np.max(heatmap))
        range_val = max(max_val - min_val, 1e-9)
        normalized = (heatmap - min_val) / range_val
        color_map = np.zeros((*normalized.shape, 3), dtype=np.uint8)
        color_map[..., 0] = (normalized * 255).astype(np.uint8)
        color_map[..., 1] = (np.sqrt(normalized) * 255).astype(np.uint8)
        color_map[..., 2] = ((1.0 - normalized) * 255).astype(np.uint8)
        self._surface_buffer = np.require(color_map, np.uint8, "C")
        h, w, _ = self._surface_buffer.shape
        qimage = QtGui.QImage(
            self._surface_buffer.data,
            w,
            h,
            3 * w,
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            self.activity_label.width(),
            self.activity_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        if (
            self._current_scenario
            and self._current_scenario.scenario_type == "tsp_reflexive"
            and self._tsp_positions
            and self._orchestrator
            and self._orchestrator.scenario_runner
        ):
            metrics = self._orchestrator.scenario_runner.scenario_metrics
            self._overlay_tour(pixmap, metrics)
        self.activity_label.setPixmap(pixmap)
        if (
            self._orchestrator.runtime_logger
            and self._orchestrator.step_counter > self._last_logged_step
        ):
            stats = {
                "min": min_val,
                "max": max_val,
                "std": float(np.std(heatmap)),
                "range": range_val,
            }
            self._orchestrator.runtime_logger.info(
                "GUI activity stats at step %d: %s",
                self._orchestrator.step_counter,
                stats,
            )
            self._last_logged_step = self._orchestrator.step_counter

    def _overlay_tour(self, pixmap: QtGui.QPixmap, metrics: dict[str, float]) -> None:
        cycles = metrics.get("cycles")
        if cycles:
            order = cycles[0]
        else:
            permutation = metrics.get("permutation")
            order = None
            if permutation:
                visited = set()
                cycle = []
                current = 0
                for _ in range(len(permutation)):
                    if current in visited:
                        break
                    visited.add(current)
                    cycle.append(current)
                    current = permutation[current]
                if len(cycle) >= 2:
                    order = cycle
        if not order or not self._tsp_positions or len(order) > len(self._tsp_positions):
            return
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        width = pixmap.width()
        height = pixmap.height()
        points = []
        for idx in order:
            x, y = self._tsp_positions[idx]
            px = x * width
            py = (1.0 - y) * height
            points.append(QtCore.QPointF(px, py))
        if points:
            poly = QtGui.QPolygonF(points + [points[0]])
            pen = QtGui.QPen(QtGui.QColor(50, 220, 255), 2.0)
            painter.setPen(pen)
            painter.drawPolyline(poly)
            node_pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
            painter.setPen(node_pen)
            painter.setBrush(QtGui.QColor(255, 120, 0))
            radius = max(3.0, max(width, height) * 0.01)
            for pt in points:
                painter.drawEllipse(pt, radius, radius)
        painter.end()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            if self._orchestrator:
                self._orchestrator.shutdown()
        finally:
            super().closeEvent(event)

    def _update_metrics(self, telemetry: OrchestratorTelemetry):
        self._append_metric("Ontological Dissonance", telemetry.total_dissonance)
        self._append_metric("CPU %", telemetry.cpu_percent)

        if self._orchestrator and self._orchestrator.scenario_runner:
            metrics = self._orchestrator.scenario_runner.scenario_metrics
            if "activity_norm" in metrics:
                self._append_metric("Activity Norm", float(metrics["activity_norm"]))
            if "max_residual" in metrics:
                self._append_metric("Max Residual", float(metrics["max_residual"]))
            if "max_clause_deficit" in metrics:
                self._append_metric("Max Clause Deficit", float(metrics["max_clause_deficit"]))
            scenario_type = self._orchestrator.scenario.scenario_type
            if scenario_type == "tsp_reflexive":
                if "doubly_stochastic_error" in metrics:
                    self._append_metric("Doubly Stochastic Error", float(metrics["doubly_stochastic_error"]))
                if "assignment_entropy" in metrics:
                    self._append_metric("Assignment Entropy", float(metrics["assignment_entropy"]))
                if "subtour_count" in metrics:
                    self._append_metric("Subtour Count", float(metrics["subtour_count"]))
                if "tour_cost" in metrics and metrics["tour_cost"] is not None:
                    self._append_metric("Tour Cost", float(metrics["tour_cost"]))
                if "tour_cost_gap" in metrics and metrics["tour_cost_gap"] is not None:
                    self._append_metric("Tour Cost Gap", float(metrics["tour_cost_gap"]))
                self._update_tsp_metrics_panel(metrics)
            else:
                self._tsp_group.setVisible(False)

        if self._axis_y and self._metric_series:
            all_values = [
                value
                for series in self._metric_series.values()
                for value in series.values
                if value is not None and math.isfinite(value)
            ]
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                if math.isclose(max_val, min_val, rel_tol=1e-6, abs_tol=1e-6):
                    max_val = min_val + 1e-3
                pad = 0.1 * (max_val - min_val)
                self._axis_y.setRange(min_val - pad, max_val + pad)

    def _append_metric(self, name: str, value: float):
        if name not in self._metric_series:
            series = QLineSeries()
            series.setName(name)
            self.chart.addSeries(series)
            if self._axis_x:
                series.attachAxis(self._axis_x)
            if self._axis_y:
                series.attachAxis(self._axis_y)
            self._metric_series[name] = MetricSeries(series, [])

        metric = self._metric_series[name]
        metric.values.append(value)
        if len(metric.values) > METRIC_HISTORY:
            metric.values.pop(0)

        series_points = [QtCore.QPointF(i, v) for i, v in enumerate(metric.values)]
        metric.series.replace(series_points)

    def _update_tsp_metrics_panel(self, metrics: dict[str, float]) -> None:
        lines = []
        if "doubly_stochastic_error" in metrics:
            lines.append(f"Row/Col Error: {metrics['doubly_stochastic_error']:.4e}")
        if "subtour_count" in metrics:
            lines.append(f"Subtours: {int(metrics['subtour_count'])}")
        if "permutation" in metrics:
            lines.append(f"Permutation: {metrics['permutation']}")
        if "tour_cost" in metrics and metrics["tour_cost"] is not None:
            lines.append(f"Tour Cost: {metrics['tour_cost']:.6f}")
        if "optimal_cost" in metrics and metrics["optimal_cost"] is not None:
            lines.append(f"Optimal Cost: {metrics['optimal_cost']:.6f}")
        if "tour_cost_gap" in metrics and metrics["tour_cost_gap"] is not None:
            lines.append(f"Cost Gap: {metrics['tour_cost_gap']:.4%}")
        if "tour_verified" in metrics:
            lines.append(f"Tour Verified: {bool(metrics['tour_verified'])}")
        if "cycles" in metrics and metrics["cycles"]:
            lines.append(f"Cycles: {metrics['cycles']}")
        self._tsp_text.setPlainText("\n".join(lines))
        self._tsp_group.setVisible(True)


