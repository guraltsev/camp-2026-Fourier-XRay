"""Define plot-node errors and lightweight snapshot containers."""

from __future__ import annotations

import dataclasses
from itertools import count
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import sympy

from ._reactive import Computed, Effect, Signal
from .specs import (
    CartesianView2D,
    CurveView,
    ParametricView,
    TRACE_ROLE_DOMAIN_BOUNDARY,
    TRACE_ROLE_DOMAIN_FILL,
    TRACE_ROLE_MAIN,
)

if TYPE_CHECKING:
    from .figure import FigureHandle

__all__ = [
    "PlottingError",
    "PlotCompilationError",
    "PlotShapeError",
    "SampleSignature",
    "TraceDataSnapshot",
    "TraceStyleSnapshot",
    "ControlLayoutItem",
    "SliderValueItem",
    "Buffer",
    "ArrayBuffer",
    "PlotNode",
]


class PlottingError(Exception):
    """Base class for plotting-layer errors."""


class PlotCompilationError(PlottingError):
    """Report expressions that cannot cross into numeric plotting."""


class PlotShapeError(PlottingError):
    """Report numeric outputs that diverge from expected grid shapes."""


@dataclasses.dataclass(frozen=True)
class SampleSignature:
    """Describe the state that determines one sampled plot."""

    expression: Any
    view: CurveView | CartesianView2D | ParametricView
    parameter_symbols: tuple[sympy.Symbol, ...]
    parameter_values: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class TraceDataSnapshot:
    """Describe sampled trace data for one Plotly trace."""

    node_id: int
    trace_role: str
    trace_type: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    contour_level: float | None = None


@dataclasses.dataclass(frozen=True)
class TraceStyleSnapshot:
    """Describe style-only trace state for one Plotly trace."""

    node_id: int
    trace_role: str
    trace_type: str
    label: str
    style: tuple[tuple[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class ControlLayoutItem:
    """Describe one slider's identity and metadata without its live value."""

    node_id: int
    symbol: sympy.Symbol
    label_markdown: str
    minimum: float
    maximum: float
    step: float


@dataclasses.dataclass(frozen=True)
class SliderValueItem:
    """Describe one slider mirror value for model-to-widget synchronization."""

    node_id: int
    symbol: sympy.Symbol
    value: float


class Buffer:
    """Store one reusable one-dimensional NumPy sample buffer."""

    def __init__(self, *, dtype: Any = float) -> None:
        self.dtype = np.dtype(dtype)
        self.array: np.ndarray | None = None
        self.capacity = 0
        self.active_length = 0
        self.generation = 0
        self.allocation_id = 0

    def set(self, values: Any) -> None:
        """Copy values into the active buffer slice, growing geometrically."""

        incoming = np.asarray(values, dtype=self.dtype)
        if incoming.ndim != 1:
            incoming = incoming.reshape(-1)
        needed = int(incoming.shape[0])

        if self.array is None or needed > self.capacity:
            new_capacity = max(needed, 1)
            if self.capacity:
                new_capacity = max(new_capacity, self.capacity * 2)
            self.array = np.empty(new_capacity, dtype=self.dtype)
            self.capacity = new_capacity
            self.allocation_id += 1

        self.array[:needed] = incoming
        self.active_length = needed
        self.generation += 1

    @property
    def active_view(self) -> np.ndarray:
        """Return the active slice of the underlying buffer."""

        if self.array is None:
            return np.empty(0, dtype=self.dtype)
        return self.array[: self.active_length]


class ArrayBuffer:
    """Store one reusable shaped NumPy sample buffer."""

    def __init__(self, *, dtype: Any = float) -> None:
        self.dtype = np.dtype(dtype)
        self.array: np.ndarray | None = None
        self.capacity = 0
        self.shape: tuple[int, ...] = (0,)
        self.generation = 0
        self.allocation_id = 0

    def set(self, values: Any) -> None:
        """Copy shaped values into the active buffer slice."""

        incoming = np.asarray(values, dtype=self.dtype)
        flat = incoming.reshape(-1)
        needed = int(flat.shape[0])

        if self.array is None or needed > self.capacity:
            new_capacity = max(needed, 1)
            if self.capacity:
                new_capacity = max(new_capacity, self.capacity * 2)
            self.array = np.empty(new_capacity, dtype=self.dtype)
            self.capacity = new_capacity
            self.allocation_id += 1

        self.array[:needed] = flat
        self.shape = incoming.shape
        self.generation += 1

    @property
    def active_view(self) -> np.ndarray:
        """Return the active shaped view of the underlying buffer."""

        if self.array is None:
            return np.empty(self.shape, dtype=self.dtype)
        size = int(np.prod(self.shape, dtype=int)) if self.shape else 1
        return self.array[:size].reshape(self.shape)


class PlotNode:
    """Represent one signal-backed sampled plot inside a figure."""

    _ids = count(1)

    def __init__(
        self,
        figure: FigureHandle,
        *,
        kind: str,
        name: str | None,
        expression: Any,
        view: CurveView | CartesianView2D | ParametricView,
        label: str,
        parameters: Mapping[sympy.Symbol, Any],
        style: dict[str, Any],
    ) -> None:
        self.id = next(self._ids)
        self.figure = figure
        self.kind = kind
        self.name = name
        self.expression_signal = Signal(expression)
        self.view_signal = Signal(view)
        self.label_signal = Signal(label)
        self.style_signal = Signal(dict(style))
        self.parameters = parameters
        self.parameter_symbols_signal = Signal(tuple(parameters.keys()))

        self.x_buffer = Buffer()
        self.y_buffer = Buffer()
        self.z_buffer = ArrayBuffer()
        self.domain_fill_buffer = ArrayBuffer()
        self.domain_boundary_buffer = ArrayBuffer()

        self.trace_data_signal = Signal(tuple())
        self.sample_signature = Computed(self._sample_signature)
        self.trace_style_snapshot = Computed(self._trace_style_snapshot)

        self._effects = [Effect(self._sample_into_buffers)]

    def _sample_signature(self) -> SampleSignature:
        symbols = self.parameter_symbols_signal()
        values = tuple(self.parameters[s].value for s in symbols)
        return SampleSignature(
            expression=self.expression_signal(),
            view=self.view_signal(),
            parameter_symbols=symbols,
            parameter_values=values,
        )

    def _trace_style_snapshot(self) -> tuple[TraceStyleSnapshot, ...]:
        style = self.style_signal()
        if self.kind == "domain":
            return (
                TraceStyleSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_FILL,
                    trace_type="domain-fill",
                    label=self.label_signal(),
                    style=tuple(sorted(style.get("domain", {}).items())),
                ),
                TraceStyleSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_BOUNDARY,
                    trace_type="domain-boundary",
                    label=f"{self.label_signal()} boundary",
                    style=tuple(sorted(style.get("boundary", {}).items())),
                ),
            )
        trace_type = "scatter"
        if self.kind == "temperature":
            trace_type = "heatmap"
        elif self.kind == "contour":
            trace_type = "contour"
        return (
            TraceStyleSnapshot(
                node_id=self.id,
                trace_role=TRACE_ROLE_MAIN,
                trace_type=trace_type,
                label=self.label_signal(),
                style=tuple(sorted(style.items())),
            ),
        )

    def _sample_into_buffers(self) -> None:
        """Sample the current signature into the reusable buffers."""

        self.trace_data_signal.set(())

