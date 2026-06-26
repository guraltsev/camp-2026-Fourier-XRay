"""Represent notebook figures, plot handles, plot nodes, and sample buffers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import inspect
from itertools import count
from pathlib import PurePosixPath
import math
import sys
import time
from types import MappingProxyType
from typing import Callable, Iterator

import numpy as np
import sympy

from math_toolkit.num import indexed_runtime_parameter_info

from ._reactive import Computed, Effect, Signal, batch, untracked
from .display import (
    FigureDisplayGeneration,
    current_execution_key,
    normalize_display_backend,
    normalize_display_policy,
)
from .errors import InfoNotFoundError, PlotNotFoundError, PlotSpecError, ViewNotFoundError
from .layout import FigureLayout, ResponsiveSidebarLayout
from .sampling import (
    SampleSignature,
    compile_numeric_curve,
    compile_numeric_domain,
    compile_numeric_field,
    compile_numeric_info,
    compile_numeric_list,
    compile_numeric_parametric,
    expression_parameter_symbols,
    sample_curve,
    sample_domain,
    sample_list_plot,
    sample_parametric,
    sample_scalar_field,
    uncovered_expression_symbols,
)
from .specs import (
    OMITTED,
    AxisView,
    CartesianView2D,
    CurveView,
    DEFAULT_ANIMATION_SPEED,
    DomainConditionSpec,
    ListView,
    ParameterAnimationMode,
    ParameterAnimationSpeedDefault,
    ParameterMetadata,
    ParameterSpec,
    ParametricView,
    PLOT_KIND_CONTOUR,
    PLOT_KIND_CURVE,
    PLOT_KIND_DOMAIN,
    PLOT_KIND_LIST,
    PLOT_KIND_PARAMETRIC,
    PLOT_KIND_TEMPERATURE,
    TRACE_ROLE_DOMAIN_BOUNDARY,
    TRACE_ROLE_DOMAIN_FILL,
    TRACE_ROLE_MAIN,
    default_initial_2d_view,
    normalize_cartesian_view,
    normalize_domain,
    normalize_domain_conditions,
    normalize_domain_style,
    normalize_expression,
    normalize_field_style,
    normalize_list_plot_spec,
    normalize_line_style,
    normalize_parameter_specs,
    normalize_parametric_expressions,
    normalize_parametric_view,
    normalize_grid_sample_count,
    normalize_sample_count,
    normalize_style,
    sort_symbols,
    _parameter_animation_mode,
    _parameter_animation_rate,
    _parameter_animation_speed,
)

InteractionCallback = Callable[[], None]

# Default plot colors are model state so toolkit-owned legends and Plotly
# traces agree without relying on Plotly's implicit colorway.
_PLOT_COLOR_CYCLE = (
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "magenta",
    "gray",
    "olive",
    "cyan",
)

# Named colors are figure-level plotting vocabulary. UI controls, legends, and
# future style helpers can share this mapping when they need concrete swatches.
PLOT_NAMED_COLOR_HEX = MappingProxyType(
    {
        "blue": "#0000ff",
        "red": "#ff0000",
        "green": "#008000",
        "orange": "#ffa500",
        "purple": "#800080",
        "brown": "#a52a2a",
        "crimson": "#dc143c",
        "pink": "#ffc0cb",
        "gray": "#808080",
        "black": "#000000",
        "royalblue": "#4169e1",
    }
)
PLOT_NAMED_COLORS = tuple(PLOT_NAMED_COLOR_HEX)

# Animation recomputation must give the host a bounded service window to
# deliver browser-originated comm callbacks into the figure interaction queue.
# The queue is the priority boundary; this delay only gives external signals a
# chance to enter that queue after a synchronous sample has released Python.
_FRONTEND_INTERACTION_SERVICE_SECONDS = 0.05


@dataclass(frozen=True)
class TraceDataSnapshot:
    """Describe the sampled data that should be rendered for one trace."""

    node_id: int
    trace_role: str
    trace_type: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    contour_level: float | None = None


@dataclass(frozen=True)
class TraceStyleSnapshot:
    """Describe the display-only properties that should be rendered for one trace."""

    node_id: int
    trace_role: str
    trace_type: str
    label: str
    style: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class LegendMarker:
    """Describe the toolkit-owned style marker for one user plot."""

    fill_color: str | None
    border_color: str | None
    border_width: float
    border_dash: str
    opacity: float


@dataclass(frozen=True)
class LegendItem:
    """Describe one toolkit-owned legend row."""

    node_id: int
    label_markdown: str
    visible: bool
    marker: LegendMarker
    sound_playable: bool = False
    sound_enabled: bool = False
    sound_playing: bool = False
    sound_status: str = "stopped"


@dataclass(frozen=True)
class ControlLayoutItem:
    """Describe one slider's identity and metadata without its live value."""

    node_id: int
    symbol: sympy.Symbol
    label_markdown: str
    minimum: float
    maximum: float
    step: float
    animated: bool = True
    animation_mode: str = "bounce"
    animation_rate_hz: float = 20.0
    animation_speed: str | float = "default"
    animation_speed_effective: float = 0.02


@dataclass(frozen=True)
class SliderValueItem:
    """Describe one slider mirror value for model-to-widget synchronization."""

    node_id: int
    symbol: sympy.Symbol
    value: float


@dataclass(frozen=True)
class ParameterAnimationStateItem:
    """Describe Python-owned play state for one parameter animation."""

    symbol: sympy.Basic
    running: bool
    direction: int
    accumulated_value_delta: float


@dataclass(frozen=True)
class InfoSegmentSnapshot:
    """Describe one rendered fragment inside an authored info card."""

    index: int
    kind: str
    text: str


@dataclass(frozen=True)
class InfoCardSnapshot:
    """Describe one rendered authored info card."""

    card_id: int
    name: str | None
    title_markdown: str | None
    markdown: str
    segments: tuple[InfoSegmentSnapshot, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class FigureViewState:
    """Describe the Python-owned visible and home Cartesian ranges.

    ``x_view`` and ``y_view`` are the ranges currently sampled and rendered.
    ``home_x_view`` and ``home_y_view`` are the reset target used by
    ``fig.view.reset()`` and mirrored into Plotly's modebar reset state.
    """

    view_id: int
    name: str | None
    x_view: AxisView
    y_view: AxisView
    home_x_view: AxisView
    home_y_view: AxisView


class Buffer:
    """Store one reusable one-dimensional NumPy sample buffer."""

    def __init__(self, *, dtype: object = float) -> None:
        """Create an empty reusable buffer."""

        self.dtype = np.dtype(dtype)
        self.array: np.ndarray | None = None
        self.capacity = 0
        self.active_length = 0
        self.generation = 0
        self.allocation_id = 0

    def set(self, values: object) -> None:
        """Copy values into the active buffer slice, growing geometrically."""

        incoming = np.asarray(values, dtype=self.dtype)
        if incoming.ndim != 1:
            incoming = incoming.reshape(-1)
        needed = int(incoming.shape[0])

        # Grow geometrically so pan, zoom, and sample-count edits do not replace
        # the underlying array on every update.
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

    def __init__(self, *, dtype: object = float) -> None:
        """Create an empty reusable shaped buffer."""

        self.dtype = np.dtype(dtype)
        self.array: np.ndarray | None = None
        self.capacity = 0
        self.shape: tuple[int, ...] = (0,)
        self.generation = 0
        self.allocation_id = 0

    def set(self, values: object) -> None:
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


class ViewHandle:
    """Represent a reusable Cartesian view owned by one figure.

    A view stores two rectangles: ``range`` is the visible rectangle used for
    sampling, and ``home_range`` is the reset rectangle. Public mutation is
    property-based through dictionaries shaped as
    ``{"x": (xmin, xmax), "y": (ymin, ymax)}`` or through per-axis properties
    such as ``home_x_range``.

    Methods
    -------
    reset
        Make this view current and return its visible axes to its home ranges.

    Attributes
    ----------
    figure : FigureHandle
        Figure that owns this view.
    id : int
        Stable view identity.
    name : str or None
        Optional view identity within the figure.
    range : dict
        Current visible ranges as ``{"x": (xmin, xmax), "y": (ymin, ymax)}``.
    home_range : dict
        Home/reset ranges as ``{"x": (xmin, xmax), "y": (ymin, ymax)}``.
    x_range, y_range : tuple
        Visible range for one axis.
    home_x_range, home_y_range : tuple
        Home/reset range for one axis.
    """

    _ids = count(1)

    def __init__(
        self,
        figure: FigureHandle,
        *,
        name: str | None,
        x_view: AxisView,
        y_view: AxisView,
        home_x_view: AxisView | None = None,
        home_y_view: AxisView | None = None,
    ) -> None:
        """Create a view handle with visible ranges and home ranges."""

        self.figure = figure
        self.id = next(self._ids)
        self.name = name
        self._state_signal = Signal(
            FigureViewState(
                view_id=self.id,
                name=name,
                x_view=x_view,
                y_view=y_view,
                home_x_view=x_view if home_x_view is None else home_x_view,
                home_y_view=y_view if home_y_view is None else home_y_view,
            ),
            equal=_semantic_equal,
        )

    @property
    def x_range(self) -> tuple[float, float]:
        """Return the current visible x-axis range."""

        return _axis_range(self._state_signal().x_view)

    @x_range.setter
    def x_range(self, value: object) -> None:
        """Set the current visible x-axis range."""

        self._set_range(x_range=value)

    @property
    def y_range(self) -> tuple[float, float]:
        """Return the current visible y-axis range."""

        return _axis_range(self._state_signal().y_view)

    @y_range.setter
    def y_range(self, value: object) -> None:
        """Set the current visible y-axis range."""

        self._set_range(y_range=value)

    @property
    def range(self) -> dict[str, tuple[float, float]]:
        """Return the current visible x and y ranges."""

        return _view_range_snapshot(self.x_range, self.y_range)

    @range.setter
    def range(self, value: object) -> None:
        """Set visible axis ranges from a range dictionary."""

        updates = _view_range_assignment(value)
        self._set_range(**updates)

    @property
    def home_x_range(self) -> tuple[float, float]:
        """Return the home x-axis range used by view reset actions."""

        return _axis_range(self._state_signal().home_x_view)

    @home_x_range.setter
    def home_x_range(self, value: object) -> None:
        """Set the home x-axis range used by view reset actions."""

        self._set_home(x_range=value)

    @property
    def home_y_range(self) -> tuple[float, float]:
        """Return the home y-axis range used by view reset actions."""

        return _axis_range(self._state_signal().home_y_view)

    @home_y_range.setter
    def home_y_range(self, value: object) -> None:
        """Set the home y-axis range used by view reset actions."""

        self._set_home(y_range=value)

    @property
    def home_range(self) -> dict[str, tuple[float, float]]:
        """Return the home x and y ranges used by reset actions."""

        return _view_range_snapshot(self.home_x_range, self.home_y_range)

    @home_range.setter
    def home_range(self, value: object) -> None:
        """Set home axis ranges from a range dictionary."""

        updates = _view_range_assignment(value)
        self._set_home(**updates)

    def _set_range(
        self,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Change this view's visible ranges while preserving its home ranges."""

        return self.figure._set_view_range(
            self,
            x_range=x_range,
            y_range=y_range,
        )

    def _set_home(
        self,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Change this view's home ranges used by reset actions."""

        return self.figure._set_view_home_range(
            self,
            x_range=x_range,
            y_range=y_range,
        )

    def reset(self) -> ViewHandle:
        """Reset this view's visible ranges to its home ranges."""

        return self.figure._reset_view(self)

    def _state(self) -> FigureViewState:
        """Return the current signal-backed state for renderer snapshots."""

        return self._state_signal()

    def _set_state(self, state: FigureViewState) -> None:
        """Replace this view's state after a model-side view operation."""

        self._state_signal.set(state)


class FigureView:
    """Expose figure-owned Cartesian view commands as a public namespace.

    The namespace is callable so ``fig.view("name")`` continues to create or
    retrieve a view handle. Visible and home range changes are public
    properties on ``fig.view`` and ``ViewHandle``; the underscored methods are
    implementation hooks used by those property setters.

    Methods
    -------
    __call__
        Return or create a view handle without making it current.
    reset
        Reset a view to its home ranges.

    Attributes
    ----------
    current : ViewHandle
        View targeted by unnamed range and reset commands.
    range : dict
        Current view's visible ranges as ``{"x": (xmin, xmax), "y": (ymin, ymax)}``.
    x_range, y_range : tuple
        Current view's visible range for one axis.
    home_range : dict
        Current view's home/reset ranges as ``{"x": (xmin, xmax), "y": (ymin, ymax)}``.
    home_x_range, home_y_range : tuple
        Current view's home/reset range for one axis.
    """

    def __init__(self, figure: FigureHandle) -> None:
        """Create a view command namespace for one figure."""

        self._figure = figure

    def __call__(self, target: str | ViewHandle | None = None) -> ViewHandle:
        """Return or create a view handle without making it current."""

        return self._figure._view_handle(target)

    @property
    def current(self) -> ViewHandle:
        """Return the figure's current view, creating it lazily."""

        return self._figure._current_view()

    @current.setter
    def current(self, target: str | ViewHandle | None) -> None:
        """Set the current view using the regular view selector rules."""

        self._set_current(target)

    @property
    def range(self) -> dict[str, tuple[float, float]]:
        """Return the current view's visible x and y ranges."""

        return self.current.range

    @range.setter
    def range(self, value: object) -> None:
        """Set the current view's visible ranges from a range dictionary."""

        self.current.range = value

    @property
    def x_range(self) -> tuple[float, float]:
        """Return the current view's visible x-axis range."""

        return self.current.x_range

    @x_range.setter
    def x_range(self, value: object) -> None:
        """Set the current view's visible x-axis range."""

        self.current.x_range = value

    @property
    def y_range(self) -> tuple[float, float]:
        """Return the current view's visible y-axis range."""

        return self.current.y_range

    @y_range.setter
    def y_range(self, value: object) -> None:
        """Set the current view's visible y-axis range."""

        self.current.y_range = value

    @property
    def home_range(self) -> dict[str, tuple[float, float]]:
        """Return the current view's home x and y ranges."""

        return self.current.home_range

    @home_range.setter
    def home_range(self, value: object) -> None:
        """Set the current view's home ranges from a range dictionary."""

        self.current.home_range = value

    @property
    def home_x_range(self) -> tuple[float, float]:
        """Return the current view's home x-axis range."""

        return self.current.home_x_range

    @home_x_range.setter
    def home_x_range(self, value: object) -> None:
        """Set the current view's home x-axis range."""

        self.current.home_x_range = value

    @property
    def home_y_range(self) -> tuple[float, float]:
        """Return the current view's home y-axis range."""

        return self.current.home_y_range

    @home_y_range.setter
    def home_y_range(self, value: object) -> None:
        """Set the current view's home y-axis range."""

        self.current.home_y_range = value

    def _set_current(
        self,
        target: str | ViewHandle | None = None,
        *,
        only_existing: bool = False,
    ) -> ViewHandle:
        """Set and return the view targeted by unnamed view commands."""

        return self._figure._set_current_view(target, only_existing=only_existing)

    def _set_range(
        self,
        target: str | ViewHandle | None = None,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Set a view's visible ranges and make it current."""

        return self._figure._set_view_range(
            target,
            x_range=x_range,
            y_range=y_range,
        )

    def _set_home_range(
        self,
        target: str | ViewHandle | None = None,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Set a view's home ranges used by reset actions."""

        return self._figure._set_view_home_range(
            target,
            x_range=x_range,
            y_range=y_range,
        )

    def _set_home(
        self,
        target: str | ViewHandle | None = None,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Set a view's home ranges using the shorter method name."""

        return self._set_home_range(
            target,
            x_range=x_range,
            y_range=y_range,
        )

    def reset(self, target: str | ViewHandle | None = None) -> ViewHandle:
        """Reset a view to its home ranges and make it current."""

        return self._figure._reset_view(target)


@dataclass
class RunningParameterAnimation:
    """Track transient play state for one running parameter."""

    direction: int
    previous_time: float
    next_due_time: float
    accumulated_value_delta: float = 0.0


class AnimationScheduler:
    """Run a cancellable notebook-friendly animation heartbeat."""

    def __init__(self) -> None:
        """Create an idle scheduler."""

        self._task: object | None = None

    @property
    def running(self) -> bool:
        """Return whether a scheduler task is active."""

        task = self._task
        return task is not None and not task.done()

    def start(self, tick: Callable[[], object]) -> bool:
        """Start a 60Hz task and return whether it could be scheduled."""

        if self.running:
            return True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        async def _run() -> None:
            while True:
                await asyncio.sleep(1.0 / 60.0)
                await asyncio.sleep(0)
                result = tick()
                if inspect.isawaitable(result):
                    await result

        self._task = loop.create_task(_run())
        return True

    def stop(self) -> None:
        """Cancel the active task if one exists."""

        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()


class FigureInteractionQueue:
    """Store figure-owned UI interactions ahead of animation recomputation."""

    def __init__(self) -> None:
        """Create an empty interaction queue."""

        self._pending: list[InteractionCallback] = []
        self._draining = False

    def enqueue(
        self,
        callback: InteractionCallback,
        *,
        clear_pending: bool = False,
    ) -> None:
        """Add one interaction callback, optionally dropping stale queued work."""

        if clear_pending:
            self._pending.clear()
        self._pending.append(callback)

    def clear(self) -> None:
        """Discard queued interaction callbacks."""

        self._pending.clear()

    def drain(self) -> None:
        """Run queued interactions in FIFO order."""

        if self._draining:
            return
        self._draining = True
        try:
            while self._pending:
                callback = self._pending.pop(0)
                callback()
        finally:
            self._draining = False


class FigureAnimationCoordinator:
    """Own Python-side parameter animation state for one figure."""

    def __init__(
        self,
        figure: FigureHandle,
        *,
        scheduler: AnimationScheduler | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        """Create an idle coordinator for figure-owned parameters."""

        self._figure = figure
        self._scheduler = AnimationScheduler() if scheduler is None else scheduler
        self._monotonic = time.monotonic if monotonic is None else monotonic
        self._running: dict[sympy.Basic, RunningParameterAnimation] = {}
        self._animation_write_depth = 0
        self._animation_write_scopes: list[tuple[sympy.Basic, int, bool]] = []
        self._stop_revision = 0

    @contextmanager
    def animation_value_write(
        self,
        symbol: sympy.Basic | None = None,
        *,
        stop_revision: int | None = None,
        allow_stopped: bool = False,
    ) -> Iterator[None]:
        """Mark parameter value writes that originate from animation ticks."""

        self._animation_write_depth += 1
        if symbol is not None:
            revision = self._stop_revision if stop_revision is None else stop_revision
            self._animation_write_scopes.append((symbol, revision, bool(allow_stopped)))
        try:
            yield
        finally:
            if symbol is not None:
                self._animation_write_scopes.pop()
            self._animation_write_depth -= 1

    @property
    def writing_animation_value(self) -> bool:
        """Return whether the coordinator is applying its own value write."""

        return self._animation_write_depth > 0

    def animation_write_should_continue(
        self,
        parameter_symbols: tuple[sympy.Basic, ...],
    ) -> bool:
        """Return whether an animation-originated resample is still current."""

        if not self._animation_write_scopes:
            return True
        active_symbol, stop_revision, allow_stopped = self._animation_write_scopes[-1]
        if active_symbol not in parameter_symbols:
            return True
        if allow_stopped:
            return True
        return active_symbol in self._running and stop_revision == self._stop_revision

    @property
    def running_symbols(self) -> frozenset[sympy.Basic]:
        """Return symbols that currently have running animations."""

        return frozenset(self._running)

    def play(self, symbol: sympy.Basic) -> None:
        """Start animation for one parameter if its metadata allows it."""

        state = self._figure.parameters.get(symbol)
        if state is None or not state.animated:
            self.stop(symbol)
            return
        metadata = state.metadata
        if metadata.minimum == metadata.maximum:
            return

        if metadata.animation_mode == ParameterAnimationMode.FORWARD:
            threshold = metadata.maximum - 2.0 * metadata.step
            if state.value >= threshold:
                default_value = self._figure._default_parameter_value(symbol)
                if default_value is not None:
                    with self.animation_value_write():
                        state.set_value(default_value)

        now = self._monotonic()
        self._running[symbol] = RunningParameterAnimation(
            direction=1,
            previous_time=now,
            next_due_time=now,
        )
        if not self._scheduler.running:
            started = self._scheduler.start(self.tick_async)
            if not started:
                self._running.pop(symbol, None)
                self._figure._queue_output_notice(
                    "Parameter animation is unavailable because no running Python event loop was found."
                )
        self._publish_state()

    def pause(self, symbol: sympy.Basic) -> None:
        """Pause one parameter animation."""

        if symbol in self._running:
            del self._running[symbol]
            self._stop_revision += 1
            self._stop_scheduler_if_idle()
            self._publish_state()

    def stop(self, symbol: sympy.Basic) -> None:
        """Stop one parameter animation and discard its accumulator."""

        self.pause(symbol)

    def stop_all(self) -> None:
        """Stop every running animation."""

        if not self._running:
            return
        self._running.clear()
        self._stop_revision += 1
        self._scheduler.stop()
        self._publish_state()

    def toggle(self, symbol: sympy.Basic) -> None:
        """Toggle one parameter between running and paused."""

        if symbol in self._running:
            self.pause(symbol)
        else:
            self.play(symbol)

    def notify_external_value_write(self, symbol: sympy.Basic) -> None:
        """Stop a running animation after a non-animation value write."""

        if not self.writing_animation_value:
            self.stop(symbol)

    def notify_metadata_write(self, symbol: sympy.Basic, *, clamped_value: bool = True) -> None:
        """Stop a running animation after metadata changes that affect playback."""

        if clamped_value or symbol in self._running:
            self.stop(symbol)

    def snapshot(self) -> tuple[ParameterAnimationStateItem, ...]:
        """Return current play state for active figure parameters."""

        items: list[ParameterAnimationStateItem] = []
        for symbol in self._figure._ordered_active_parameter_symbols():
            running = self._running.get(symbol)
            items.append(
                ParameterAnimationStateItem(
                    symbol=symbol,
                    running=running is not None,
                    direction=1 if running is None else running.direction,
                    accumulated_value_delta=(
                        0.0 if running is None else running.accumulated_value_delta
                    ),
                )
            )
        return tuple(items)

    def tick(self) -> None:
        """Advance due animations using the coordinator's monotonic clock."""

        self.tick_at(self._monotonic())

    async def tick_async(self) -> None:
        """Advance due animations after pending frontend messages run."""

        await self._drain_frontend_events()
        self._figure.drain_interactions()
        if not self._running:
            return
        await self.tick_at_async(self._monotonic())

    def tick_at(self, now: float) -> None:
        """Advance due animations for deterministic tests and scheduler wakes."""

        for symbol in tuple(self._running):
            running = self._running.get(symbol)
            state = self._figure.parameters.get(symbol)
            if running is None or state is None or not state.animated:
                self.stop(symbol)
                continue
            if now + 1e-12 < running.next_due_time:
                continue
            self._tick_symbol(symbol, state, running, float(now))
        self._stop_scheduler_if_idle()
        self._publish_state()

    async def tick_at_async(self, now: float) -> None:
        """Advance due animations while yielding before expensive work."""

        for symbol in tuple(self._running):
            await self._drain_frontend_events()
            self._figure.drain_interactions()
            running = self._running.get(symbol)
            state = self._figure.parameters.get(symbol)
            if running is None or state is None or not state.animated:
                self.stop(symbol)
                continue
            if now + 1e-12 < running.next_due_time:
                continue
            self._tick_symbol(symbol, state, running, float(now))
        self._stop_scheduler_if_idle()
        self._publish_state()

    async def _drain_frontend_events(self) -> None:
        """Yield so browser-originated callbacks can enqueue interactions."""

        self._figure.drain_interactions()
        await asyncio.sleep(_FRONTEND_INTERACTION_SERVICE_SECONDS)
        self._figure.drain_interactions()
        await asyncio.sleep(0)
        self._figure.drain_interactions()

    def _tick_symbol(
        self,
        symbol: sympy.Basic,
        state: ParameterState,
        running: RunningParameterAnimation,
        now: float,
    ) -> None:
        """Advance one parameter by elapsed wall-clock motion."""

        metadata = state.metadata
        elapsed = max(0.0, now - running.previous_time)
        running.previous_time = now
        running.next_due_time = now + 1.0 / metadata.animation_rate_hz

        running.accumulated_value_delta += (
            running.direction * metadata.animation_speed_effective * elapsed
        )
        step = metadata.step
        if abs(running.accumulated_value_delta) < step:
            return

        whole_steps = math.trunc(abs(running.accumulated_value_delta) / step)
        applied_delta = math.copysign(whole_steps * step, running.accumulated_value_delta)
        running.accumulated_value_delta -= applied_delta
        candidate = state.value + applied_delta
        value, direction, keep_running = self._apply_mode(candidate, metadata, running.direction)
        running.direction = direction
        if not keep_running:
            self._running.pop(symbol, None)
        with self.animation_value_write(
            symbol,
            stop_revision=self._stop_revision,
            allow_stopped=not keep_running,
        ):
            state.set_value(value)

    def _apply_mode(
        self,
        candidate: float,
        metadata: ParameterMetadata,
        direction: int,
    ) -> tuple[float, int, bool]:
        """Return a range-constrained value and next direction."""

        minimum = metadata.minimum
        maximum = metadata.maximum
        width = maximum - minimum
        if width <= 0:
            return minimum, direction, False
        if metadata.animation_mode == ParameterAnimationMode.FORWARD:
            if candidate >= maximum:
                return maximum, direction, False
            return max(minimum, candidate), direction, True
        if metadata.animation_mode == ParameterAnimationMode.WRAP:
            while candidate > maximum:
                candidate = minimum + (candidate - maximum)
            while candidate < minimum:
                candidate = maximum - (minimum - candidate)
            return candidate, direction, True

        doubled = 2.0 * width
        position = (candidate - minimum) % doubled
        if position <= width:
            return minimum + position, 1, True
        return maximum - (position - width), -1, True

    def _stop_scheduler_if_idle(self) -> None:
        """Stop the heartbeat when no parameter is running."""

        if not self._running:
            self._scheduler.stop()

    def _publish_state(self) -> None:
        """Publish play-state changes to the active frontend."""

        self._figure.sync_animation_state(self.snapshot())


class ParameterAnimationControl:
    """Expose animation commands for one figure parameter."""

    def __init__(self, state: ParameterState) -> None:
        """Create a parameter-scoped animation command object."""

        self._state = state

    @property
    def enabled(self) -> bool:
        """Return whether animation controls are enabled for this parameter."""

        return self._state.animated

    @enabled.setter
    def enabled(self, value: object) -> None:
        """Enable or disable animation controls for this parameter."""

        self._state.animated = value

    @property
    def running(self) -> bool:
        """Return whether this parameter is currently animating."""

        return self._state.symbol in self._state._figure._animation_coordinator.running_symbols

    def start(self) -> None:
        """Start this parameter's animation."""

        self._state._figure._animation_coordinator.play(self._state.symbol)

    def stop(self) -> None:
        """Stop this parameter's animation."""

        self._state._figure._animation_coordinator.stop(self._state.symbol)

    def toggle(self) -> None:
        """Toggle this parameter's animation."""

        self._state._figure._animation_coordinator.toggle(self._state.symbol)


class ParameterState:
    """Own signal-backed value and metadata for one plot parameter."""

    __slots__ = ("_figure", "animate", "metadata_signal", "symbol", "value_signal")

    def __init__(self, figure: FigureHandle, spec: ParameterSpec) -> None:
        """Create parameter state from a normalized public spec."""

        self._figure = figure
        self.symbol = spec.symbol
        self.animate = ParameterAnimationControl(self)
        self.value_signal = Signal(float(spec.value), equal=_semantic_equal)
        self.metadata_signal = Signal(spec.metadata, equal=_semantic_equal)

    @property
    def value(self) -> float:
        """Return the current slider value."""

        return self.value_signal()

    @value.setter
    def value(self, value: object) -> None:
        """Set the current slider value."""

        self._figure._set_parameter_field(self.symbol, "value", value)

    @property
    def metadata(self) -> ParameterMetadata:
        """Return the current slider metadata."""

        return self.metadata_signal()

    @property
    def min(self) -> float:
        """Return the current slider minimum."""

        return self.metadata.minimum

    @min.setter
    def min(self, value: object) -> None:
        """Set the current slider minimum."""

        self._figure._set_parameter_field(self.symbol, "min", value)

    @property
    def max(self) -> float:
        """Return the current slider maximum."""

        return self.metadata.maximum

    @max.setter
    def max(self, value: object) -> None:
        """Set the current slider maximum."""

        self._figure._set_parameter_field(self.symbol, "max", value)

    @property
    def range(self) -> tuple[float, float, float]:
        """Return the current slider minimum, maximum, and step."""

        metadata = self.metadata
        return (metadata.minimum, metadata.maximum, metadata.step)

    @range.setter
    def range(self, value: object) -> None:
        """Set the slider range from ``(min, max)`` or ``(min, max, step)``."""

        self._figure._set_parameter_range(self.symbol, value)

    @property
    def step(self) -> float:
        """Return the current slider step."""

        return self.metadata.step

    @step.setter
    def step(self, value: object) -> None:
        """Set the current slider step."""

        self._figure._set_parameter_field(self.symbol, "step", value)

    @property
    def label(self) -> str | None:
        """Return the current slider label."""

        return self.metadata.label

    @label.setter
    def label(self, value: object) -> None:
        """Set the current slider label."""

        self._figure._set_parameter_field(self.symbol, "label", value)

    @property
    def animated(self) -> bool:
        """Return whether this parameter exposes animation controls."""

        return self.metadata.animated

    @animated.setter
    def animated(self, value: object) -> None:
        """Set whether this parameter exposes animation controls."""

        self._figure._set_parameter_field(self.symbol, "animated", value)

    @property
    def animation_mode(self) -> str:
        """Return the stored animation boundary mode."""

        return self.metadata.animation_mode.value

    @animation_mode.setter
    def animation_mode(self, value: object) -> None:
        """Set the stored animation boundary mode."""

        self._figure._set_parameter_field(self.symbol, "animation_mode", value)

    @property
    def animation_rate_hz(self) -> float:
        """Return the animation recomputation cadence."""

        return self.metadata.animation_rate_hz

    @animation_rate_hz.setter
    def animation_rate_hz(self, value: object) -> None:
        """Set the animation recomputation cadence."""

        self._figure._set_parameter_field(self.symbol, "animation_rate_hz", value)

    @property
    def animation_speed(self) -> float | ParameterAnimationSpeedDefault:
        """Return the stored animation speed setting."""

        return self.metadata.animation_speed

    @animation_speed.setter
    def animation_speed(self, value: object) -> None:
        """Set the stored animation speed setting."""

        self._figure._set_parameter_field(self.symbol, "animation_speed", value)

    @property
    def animation_speed_effective(self) -> float:
        """Return the concrete animation speed after applying defaults."""

        return self.metadata.animation_speed_effective

    def set_value(self, value: object) -> None:
        """Set the numeric parameter value from a widget or handle call."""

        self._figure._animation_coordinator.notify_external_value_write(self.symbol)
        with self._figure._coalesced_trace_data_updates(), batch():
            self.value_signal.set(float(value))

    def set_spec(self, spec: ParameterSpec) -> None:
        """Update value and metadata from a normalized spec."""

        self.value_signal.set(float(spec.value))
        self.metadata_signal.set(spec.metadata)

    def to_spec(self) -> ParameterSpec:
        """Return a normalized spec snapshot for this parameter."""

        return ParameterSpec(
            symbol=self.symbol,
            value=self.value,
            metadata=self.metadata,
        )


class PendingParameterState:
    """Create a new figure parameter from the first direct assignment."""

    __slots__ = ("_figure", "symbol")

    def __init__(self, figure: FigureHandle, symbol: sympy.Basic) -> None:
        """Create a pending parameter entry for one supported symbol."""

        self._figure = figure
        self.symbol = symbol

    @property
    def value(self) -> float:
        """Raise because a pending parameter has no value yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @value.setter
    def value(self, value: object) -> None:
        """Create the parameter with a value-centered default range."""

        number = self._figure._finite_parameter_float(value, "value")
        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=number,
                metadata=ParameterMetadata(
                    minimum=number - 1.0,
                    maximum=number + 1.0,
                    step=0.01,
                    label=None,
                ),
            )
        )

    @property
    def min(self) -> float:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @min.setter
    def min(self, value: object) -> None:
        """Create the parameter from its first slider minimum."""

        minimum = self._figure._finite_parameter_float(value, "minimum")
        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=minimum + 1.0,
                metadata=ParameterMetadata(
                    minimum=minimum,
                    maximum=minimum + 2.0,
                    step=0.01,
                    label=None,
                ),
            )
        )

    @property
    def max(self) -> float:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @max.setter
    def max(self, value: object) -> None:
        """Create the parameter from its first slider maximum."""

        maximum = self._figure._finite_parameter_float(value, "maximum")
        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=maximum - 1.0,
                metadata=ParameterMetadata(
                    minimum=maximum - 2.0,
                    maximum=maximum,
                    step=0.01,
                    label=None,
                ),
            )
        )

    @property
    def range(self) -> tuple[float, float, float]:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @range.setter
    def range(self, value: object) -> None:
        """Create the parameter from its first slider range."""

        minimum, maximum, step = self._figure._parameter_range_parts(
            value,
            default_step=0.01,
        )
        if minimum > maximum:
            raise PlotSpecError("Parameter slider minimum must not exceed maximum.")
        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=(minimum + maximum) / 2.0,
                metadata=ParameterMetadata(
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    label=None,
                ),
            )
        )

    @property
    def step(self) -> float:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @step.setter
    def step(self, value: object) -> None:
        """Create a default parameter with a custom slider step."""

        step = self._figure._positive_parameter_float(value, "step")
        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=0.0,
                metadata=ParameterMetadata(
                    minimum=-1.0,
                    maximum=1.0,
                    step=step,
                    label=None,
                ),
            )
        )

    @property
    def label(self) -> str | None:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @label.setter
    def label(self, value: object) -> None:
        """Create a default parameter with a custom slider label."""

        self._figure._explicit_parameter_definitions.add(self.symbol)
        self._figure._parameter_state_for_spec(
            ParameterSpec(
                symbol=self.symbol,
                value=0.0,
                metadata=ParameterMetadata(
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.01,
                    label=None if value is None else str(value),
                ),
            )
        )

    @property
    def animated(self) -> bool:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @animated.setter
    def animated(self, value: object) -> None:
        """Create a default parameter with a custom animation flag."""

        self.value = 0.0
        self._figure._set_parameter_field(self.symbol, "animated", value)

    @property
    def animation_mode(self) -> str:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @animation_mode.setter
    def animation_mode(self, value: object) -> None:
        """Create a default parameter with a custom animation mode."""

        self.value = 0.0
        self._figure._set_parameter_field(self.symbol, "animation_mode", value)

    @property
    def animation_rate_hz(self) -> float:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @animation_rate_hz.setter
    def animation_rate_hz(self, value: object) -> None:
        """Create a default parameter with a custom animation rate."""

        self.value = 0.0
        self._figure._set_parameter_field(self.symbol, "animation_rate_hz", value)

    @property
    def animation_speed(self) -> float | ParameterAnimationSpeedDefault:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")

    @animation_speed.setter
    def animation_speed(self, value: object) -> None:
        """Create a default parameter with a custom animation speed."""

        self.value = 0.0
        self._figure._set_parameter_field(self.symbol, "animation_speed", value)

    @property
    def animation_speed_effective(self) -> float:
        """Raise because a pending parameter has no metadata yet."""

        raise PlotSpecError("Parameter has not been assigned yet.")


class FigureParameters:
    """Expose figure-owned parameter controls as a callable mapping.

    Calling ``fig.parameters({...})`` defines or updates slider-backed values
    shared by plots and info cards in the figure. The object also behaves like
    a read-only mapping from SymPy parameter symbols to live ``ParameterState``
    objects, so callers can inspect current values and metadata.

    Parameters
    ----------
    figure : FigureHandle
        Figure whose parameter state is exposed.

    Methods
    -------
    __call__
        Define or update parameter values and slider metadata.
    get
        Return one parameter state or a default value.
    items
        Return parameter symbol and state pairs.
    keys
        Return figure-owned parameter symbols.
    values
        Return figure-owned parameter states.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import figure
    >>> a = sympy.Symbol("a")
    >>> fig = figure()
    >>> _ = fig.parameters({a: {"value": 1.0, "min": 0.0, "max": 2.0}})
    >>> round(fig.parameters[a].value, 1)
    1.0
    """

    def __init__(self, figure: FigureHandle) -> None:
        """Create a parameter command namespace for one figure."""

        self._figure = figure

    def __call__(self, params: object) -> FigureHandle:
        """Define or update figure-level parameter specs and return the figure.

        Parameters
        ----------
        params : object
            Mapping from ``sympy.Symbol`` or concrete ``sympy.Indexed``
            parameters to numeric values or dictionaries with ``value``,
            ``min``, ``max``, ``step``, and ``label`` entries.

        Returns
        -------
        FigureHandle
            Figure that owns the updated parameter state.

        Raises
        ------
        PlotSpecError
            Raised when the supplied object is not a parameter mapping or when
            any key is not a supported SymPy parameter symbol.
        """

        return self._figure._define_parameters(params)

    def __getitem__(self, symbol: sympy.Basic) -> ParameterState | PendingParameterState:
        """Return live state for one active or predefined parameter."""

        return self._figure._parameter_entry(symbol)

    def __contains__(self, symbol: object) -> bool:
        """Return whether a parameter symbol exists on the figure."""

        return (
            symbol in self._figure._parameters
            or symbol in self._figure._parameter_definitions
        )

    def __iter__(self) -> object:
        """Iterate over figure-owned parameter symbols."""

        return iter(self._figure._parameters)

    def __len__(self) -> int:
        """Return the number of figure-owned parameter states."""

        return len(self._figure._parameters)

    def get(self, symbol: sympy.Basic, default: object = None) -> object:
        """Return live state for one parameter or a default value."""

        if symbol in self:
            return self._figure._ensure_parameter_state(symbol)
        return default

    def items(self) -> object:
        """Return parameter symbol and state pairs."""

        return self._figure._parameters.items()

    def keys(self) -> object:
        """Return figure-owned parameter symbols."""

        return self._figure._parameters.keys()

    def values(self) -> object:
        """Return figure-owned parameter states."""

        return self._figure._parameters.values()


class InfoCommand:
    """Expose figure-owned Markdown info commands as a callable namespace.

    Parameters
    ----------
    figure : FigureHandle
        Figure whose ordered info cards should be created, updated, or cleared.

    Methods
    -------
    __call__
        Create or replace one info card from Markdown, symbolic, or callable
        fragments.
    clear
        Remove one named info card or all info cards.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import figure
    >>> a = sympy.Symbol("a")
    >>> fig = figure()
    >>> fig.info("a = ", a).name is None
    True
    """

    def __init__(self, figure: FigureHandle) -> None:
        """Create an info command namespace for one figure."""

        self._figure = figure

    def __call__(
        self,
        *fragments: object,
        name: object = None,
        title: object = OMITTED,
        params: object = OMITTED,
    ) -> InfoHandle:
        """Create or replace one authored info card.

        Parameters
        ----------
        *fragments : object
            Markdown strings, SymPy expressions, or callables accepting the
            owning figure.
        name : object, optional
            Optional card identity. Named calls replace the existing card with
            that name while preserving its order.
        title : object, optional
            Optional Markdown title. Omit it on named updates to preserve the
            previous title, or pass ``None`` to clear it.
        params : object, optional
            Parameter slider specs for symbolic fragments.

        Returns
        -------
        InfoHandle
            Public handle for the created or updated card.

        Raises
        ------
        PlotSpecError
            Raised when no fragments are supplied or a fragment is unsupported.
        """

        return self._figure._add_or_update_info(
            fragments,
            name=name,
            title=title,
            params=params,
        )

    def clear(self, name: object = None) -> None:
        """Remove authored info cards from the figure.

        Parameters
        ----------
        name : object, optional
            Optional card identity. Omit it or pass ``None`` to clear all cards.

        Returns
        -------
        None
            The figure is mutated in place.

        Raises
        ------
        InfoNotFoundError
            Raised when a named card does not exist.
        """

        self._figure._clear_info(name)
        return None


class InfoCard:
    """Represent one signal-backed authored Markdown info card."""

    _ids = count(1)

    def __init__(
        self,
        figure: FigureHandle,
        *,
        name: str | None,
        title: str | None,
        fragments: tuple[object, ...],
        parameters: Mapping[sympy.Basic, ParameterSpec],
    ) -> None:
        """Create an info card with fragment and parameter state."""

        self.id = next(self._ids)
        self.figure = figure
        self.name = name
        self.title_signal = Signal(title, equal=_semantic_equal)
        self.fragments_signal = Signal(fragments, equal=_semantic_equal)
        self.parameters: dict[sympy.Basic, ParameterState] = {
            symbol: self.figure._parameter_state_for_spec(spec)
            for symbol, spec in parameters.items()
        }
        self.parameter_symbols_signal = Signal(
            tuple(self.parameters),
            equal=_semantic_equal,
        )
        self.has_callable_signal = Signal(
            any(callable(fragment) for fragment in fragments),
            equal=_semantic_equal,
        )
        self.controls_signature = Computed(
            self._controls_signature,
            equal=_semantic_equal,
        )
        self.snapshot = Computed(self._snapshot, equal=_semantic_equal)
        self._compiled_info: dict[tuple[object, tuple[sympy.Symbol, ...]], object] = {}

    @property
    def fragments(self) -> tuple[object, ...]:
        """Return the current ordered fragments."""

        return self.fragments_signal()

    @property
    def parameter_symbols(self) -> tuple[sympy.Symbol, ...]:
        """Return active symbolic parameter symbols."""

        return self.parameter_symbols_signal()

    def update(
        self,
        *,
        title: str | None,
        fragments: tuple[object, ...],
        parameters: Mapping[sympy.Basic, ParameterSpec],
    ) -> None:
        """Replace the card content while preserving card identity."""

        with batch():
            self.title_signal.set(title)
            self.fragments_signal.set(fragments)
            self.has_callable_signal.set(
                any(callable(fragment) for fragment in fragments)
            )
            self._set_parameter_specs(parameters)

    def parameter_specs(self) -> dict[sympy.Basic, ParameterSpec]:
        """Return normalized parameter spec snapshots for active parameters."""

        return {
            symbol: self.figure._defaulted_parameter_spec(symbol, state)
            for symbol, state in self.parameters.items()
        }

    def set_params(self, params: object) -> None:
        """Apply public parameter updates for this card's symbolic fragments."""

        specs = normalize_parameter_specs(
            _symbolic_fragments_expression(self.fragments_signal()),
            (),
            params,
            existing=self.parameter_specs(),
        )
        self._set_parameter_specs(specs)

    def _set_parameter_specs(
        self,
        specs: Mapping[sympy.Basic, ParameterSpec],
    ) -> None:
        """Replace active parameters while preserving figure-level state."""

        for symbol in tuple(self.parameters):
            if symbol not in specs:
                del self.parameters[symbol]

        self.parameters = {
            symbol: self.figure._parameter_state_for_spec(spec)
            for symbol, spec in specs.items()
        }

        self.parameter_symbols_signal.set(tuple(self.parameters))
        self.figure._prune_unused_parameters()

    def _controls_signature(self) -> tuple[tuple[object, ...], ...]:
        """Return metadata that determines visible parameter controls."""

        rows = []
        for symbol in self.parameter_symbols_signal():
            metadata = self.parameters[symbol].metadata_signal()
            rows.append(
                (
                    symbol,
                    metadata.minimum,
                    metadata.maximum,
                    metadata.step,
                    metadata.label,
                    metadata.animated,
                    metadata.animation_mode.value,
                    metadata.animation_rate_hz,
                    _animation_speed_snapshot(metadata),
                    metadata.animation_speed_effective,
                )
            )
        return tuple(rows)

    def _snapshot(self) -> InfoCardSnapshot:
        """Return rendered Markdown for the current fragments."""

        pieces: list[str] = []
        segments: list[InfoSegmentSnapshot] = []
        errors: list[str] = []
        for index, fragment in enumerate(self.fragments_signal()):
            try:
                value = fragment(self.figure) if callable(fragment) else fragment
                kind, text = self._render_fragment(value)
                pieces.append(text)
                segments.append(InfoSegmentSnapshot(index=index, kind=kind, text=text))
            except Exception as exc:
                message = f"Info error: {exc}"
                text = f"\n\n**{message}**"
                pieces.append(text)
                segments.append(
                    InfoSegmentSnapshot(index=index, kind="markdown", text=text)
                )
                errors.append(message)
        return InfoCardSnapshot(
            card_id=self.id,
            name=self.name,
            title_markdown=self.title_signal(),
            markdown="".join(pieces),
            segments=tuple(segments),
            error="\n".join(errors) if errors else None,
        )

    def _render_fragment(self, value: object) -> tuple[str, str]:
        """Render one evaluated fragment and classify its frontend cost."""

        if isinstance(value, str):
            return "markdown", value
        if isinstance(value, sympy.MatrixBase | sympy.Basic):
            return self._render_symbolic_value(value)
        if _is_numeric_scalar(value):
            return "markdown", _format_info_number(float(value))
        return "markdown", str(value)

    def _render_symbolic_value(self, expression: object) -> tuple[str, str]:
        """Evaluate a symbolic fragment numerically when all symbols are parameters."""

        symbols = expression_parameter_symbols(
            expression,
            sort_symbols(self.parameters),
        )
        if uncovered_expression_symbols(expression, symbols):
            return "markdown", sympy.latex(expression)

        if isinstance(expression, sympy.MatrixBase):
            substitutions = {
                symbol: self.parameters[symbol].value_signal() for symbol in symbols
            }
            return "markdown", sympy.latex(expression.subs(substitutions))

        try:
            numeric = self._compiled_for_expression(expression, symbols)
            if symbols:
                value = numeric(
                    *(self.parameters[symbol].value_signal() for symbol in symbols)
                )
            else:
                value = numeric()
        except Exception:
            return "markdown", sympy.latex(expression)

        if _is_numeric_scalar(value):
            return "markdown", _format_info_number(float(value))

        try:
            substitutions = {
                symbol: self.parameters[symbol].value_signal() for symbol in symbols
            }
            return "markdown", sympy.latex(expression.subs(substitutions))
        except Exception:
            return "markdown", sympy.latex(value)

    def _compiled_for_expression(
        self,
        expression: object,
        symbols: tuple[sympy.Basic, ...],
    ) -> object:
        """Return a cached symbolic info evaluator."""

        key = (expression, symbols)
        compiled = self._compiled_info.get(key)
        if compiled is not None:
            return compiled

        numeric = compile_numeric_info(expression, symbols)
        self._compiled_info[key] = numeric
        return numeric


class InfoHandle:
    """Represent a public handle to one authored info card.

    Parameters
    ----------
    figure : FigureHandle
        Figure that owns the card.
    card : InfoCard
        Card model represented by the handle.

    Attributes
    ----------
    figure : FigureHandle
        Owning figure.
    name : str or None
        Optional card identity.
    title : str or None
        Current Markdown title.
    markdown : str
        Current rendered Markdown body.
    """

    def __init__(self, figure: FigureHandle, card: InfoCard) -> None:
        """Create a handle for one info card."""

        self.figure = figure
        self._card = card

    @property
    def name(self) -> str | None:
        """Return this card's optional identity name."""

        return self._card.name

    @property
    def title(self) -> str | None:
        """Return this card's Markdown title."""

        return self._card.title_signal()

    @property
    def markdown(self) -> str:
        """Return this card's current rendered Markdown body."""

        return self._card.snapshot().markdown

    @property
    def snapshot(self) -> InfoCardSnapshot:
        """Return the current immutable rendered snapshot."""

        return self._card.snapshot()

    def remove(self) -> None:
        """Remove this card from its parent figure."""

        self.figure._remove_info_card(self._card)
        return None


class PlotNode:
    """Represent one signal-backed sampled plot inside a figure."""

    _ids = count(1)

    def __init__(
        self,
        figure: FigureHandle,
        *,
        kind: str,
        name: str | None,
        expression: object,
        view: CurveView | CartesianView2D | ParametricView | ListView,
        label: str,
        parameters: Mapping[sympy.Basic, ParameterSpec],
        style: Mapping[str, object],
    ) -> None:
        """Create a plot node with source state and sampling effects."""

        self.id = next(self._ids)
        self.figure = figure
        self.kind = kind
        self.name = name
        self.expression_signal = _quiet_signal(expression, equal=_semantic_equal)
        self._expression_revision_signal = Signal(0)
        self.view_signal = Signal(view, equal=_semantic_equal)
        self.label_signal = Signal(label, equal=_semantic_equal)
        self.style_signal = Signal(dict(style), equal=_semantic_equal)
        self._domain_boundary_visible_intent = _domain_boundary_visible(style)
        self.parameters: dict[sympy.Basic, ParameterState] = {
            symbol: self.figure._parameter_state_for_spec(spec)
            for symbol, spec in parameters.items()
        }
        self.parameter_symbols_signal = Signal(
            tuple(self.parameters),
            equal=_semantic_equal,
        )

        self.x_buffer = Buffer(dtype=float)
        self.y_buffer = Buffer(dtype=float)
        self.z_buffer = ArrayBuffer(dtype=float)
        self.domain_fill_buffer = ArrayBuffer(dtype=float)
        self.domain_boundary_buffer = ArrayBuffer(dtype=float)
        self.trace_data_signal = Signal((), equal=lambda _old, _new: False)
        self.sample_signature = Computed(self._sample_signature)
        self.controls_signature = Computed(
            self._controls_signature,
            equal=_semantic_equal,
        )
        self.trace_style_snapshot = Computed(
            self._trace_style_snapshot,
            equal=_semantic_equal,
        )
        self.legend_item_snapshot = Computed(
            self._legend_item_snapshot,
            equal=_semantic_equal,
        )
        self.slider_value_snapshot = Computed(
            self._slider_value_snapshot,
            equal=_semantic_equal,
        )

        self._compiled_key: tuple[object, ...] | None = None
        self._compiled_numeric: object | None = None
        self._effects: list[object] = [Effect(self._sample_into_buffers)]

    @property
    def expression(self) -> object:
        """Return the current symbolic expression."""

        return self.expression_signal()

    @property
    def view(self) -> CurveView | CartesianView2D | ParametricView | ListView:
        """Return the current sampled plot view."""

        return self.view_signal()

    @property
    def label(self) -> str:
        """Return the current display label."""

        return self.label_signal()

    @property
    def default_label(self) -> str:
        """Return the expression-derived display label for this plot."""

        return _default_plot_label(self.expression_signal())

    @property
    def style(self) -> dict[str, object]:
        """Return the current trace style dictionary."""

        return dict(self.style_signal())

    @property
    def parameter_symbols(self) -> tuple[sympy.Symbol, ...]:
        """Return active parameter symbols in numeric call order."""

        return self.parameter_symbols_signal()

    @property
    def independent_symbols(self) -> tuple[sympy.Symbol, ...]:
        """Return independent variables in numeric call order."""

        return _independent_symbols_for_view(self.view)

    def update(
        self,
        *,
        expression: object,
        view: CurveView | CartesianView2D | ParametricView | ListView,
        label: str,
        parameters: Mapping[sympy.Basic, ParameterSpec],
        style: Mapping[str, object],
    ) -> None:
        """Update source state while preserving node identity and buffers."""

        with batch():
            _quiet_signal_set(self.expression_signal, expression)
            self._expression_revision_signal.update(lambda revision: revision + 1)
            self.view_signal.set(view)
            self.label_signal.set(label)
            self.style_signal.set(dict(style))
            self._set_parameter_specs(parameters)

    def set_style(self, style: object) -> None:
        """Merge supported style updates into this node."""

        if self.kind == PLOT_KIND_CURVE:
            merged = normalize_style(style, existing=self.style_signal())
        elif self.kind in {PLOT_KIND_PARAMETRIC, PLOT_KIND_LIST}:
            merged = normalize_line_style(
                style,
                existing=self.style_signal(),
                plotter="list_plot" if self.kind == PLOT_KIND_LIST else "parametric_plot",
            )
        elif self.kind == PLOT_KIND_TEMPERATURE:
            merged = normalize_field_style(
                style,
                existing=self.style_signal(),
                plotter="temperature_plot",
            )
        elif self.kind == PLOT_KIND_CONTOUR:
            merged = normalize_field_style(
                style,
                existing=self.style_signal(),
                plotter="contour_plot",
            )
        else:
            merged = normalize_domain_style(style, existing=self.style_signal())
        if merged is not None:
            if self.kind == PLOT_KIND_DOMAIN and isinstance(style, Mapping):
                boundary_update = style.get("boundary")
                if (
                    isinstance(boundary_update, Mapping)
                    and "visible" in boundary_update
                    and bool(boundary_update["visible"])
                ):
                    self._domain_boundary_visible_intent = True
            self.style_signal.set(merged)

    def set_visible(self, visible: bool) -> None:
        """Set plot visibility through the normalized style state."""

        if not isinstance(visible, bool):
            raise PlotSpecError("Plot visibility must be True or False.")
        if self.kind == PLOT_KIND_DOMAIN:
            style = self.style_signal()
            if not visible:
                self._domain_boundary_visible_intent = _domain_boundary_visible(style)
            boundary_enabled = self._domain_boundary_visible_intent
            self.set_style(
                {
                    "domain": {"visible": visible},
                    "boundary": {"visible": visible and boundary_enabled},
                }
            )
            return
        self.set_style({"visible": visible})

    def toggle_visible(self) -> None:
        """Toggle plot visibility through the normalized style state."""

        self.set_visible(not self.legend_item_snapshot().visible)

    def set_label(self, label: object) -> None:
        """Set the display label used by the trace legend."""

        if label is None or str(label) == "":
            self.label_signal.set(self.default_label)
            return
        self.label_signal.set(str(label))

    def set_params(self, params: object) -> None:
        """Apply public parameter updates for this node."""

        specs = normalize_parameter_specs(
            self.expression,
            self.independent_symbols,
            params,
            existing=self.parameter_specs(),
        )
        _validate_plot_candidate(self.kind, self.expression, self.view, specs)
        with batch():
            self._set_parameter_specs(specs)

    def set_samples(self, samples: object) -> None:
        """Set the sample count for continuous plot kinds."""

        current = self.view_signal()
        if isinstance(current, ListView):
            raise PlotSpecError("List plots do not expose editable sample counts.")

        # Normalize against the same public grammar used by plotting calls, then
        # validate one sample pass before publishing reactive source state.
        if isinstance(current, CurveView | ParametricView):
            next_view = replace(current, samples=normalize_sample_count(samples))
        else:
            x_samples, y_samples = normalize_grid_sample_count(samples)
            next_view = replace(current, x_samples=x_samples, y_samples=y_samples)

        _validate_plot_candidate(
            self.kind,
            self.expression,
            next_view,
            self.parameter_specs(),
        )
        self.view_signal.set(next_view)

    def patch_view_range(
        self,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """Patch visible Cartesian ranges while preserving declared domains."""

        current = self.view_signal()
        if isinstance(current, ParametricView):
            return
        if isinstance(current, ListView):
            if not current.inferred or x_range is None:
                return
            self.view_signal.set(
                replace(
                    current,
                    x_view=AxisView(minimum=x_range[0], maximum=x_range[1]),
                )
            )
            return
        if isinstance(current, CurveView):
            if x_range is None:
                return
            self.view_signal.set(
                replace(
                    current,
                    x_view=replace(
                        current.x_view,
                        minimum=x_range[0],
                        maximum=x_range[1],
                    ),
                )
            )
            return

        next_view = current
        if x_range is not None:
            next_view = replace(
                next_view,
                x_view=replace(
                    next_view.x_view,
                    minimum=x_range[0],
                    maximum=x_range[1],
                ),
            )
        if y_range is not None:
            next_view = replace(
                next_view,
                y_view=replace(
                    next_view.y_view,
                    minimum=y_range[0],
                    maximum=y_range[1],
                ),
            )
        self.view_signal.set(next_view)

    def parameter_specs(self) -> dict[sympy.Basic, ParameterSpec]:
        """Return normalized parameter spec snapshots for active parameters."""

        return {
            symbol: self.figure._defaulted_parameter_spec(symbol, state)
            for symbol, state in self.parameters.items()
        }

    def register_effect(self, effect: object) -> None:
        """Keep an externally owned reactive effect alive with the node."""

        self._effects.append(effect)

    def dispose(self) -> None:
        """Dispose reactive effects owned by this node."""

        for effect in self._effects:
            dispose = getattr(effect, "dispose", None)
            if dispose is not None:
                dispose()
        self._effects.clear()

    def _set_parameter_specs(
        self,
        specs: Mapping[sympy.Basic, ParameterSpec],
    ) -> None:
        """Replace the active parameter set while preserving matching states."""

        # Remove parameters that are no longer active before publishing the new
        # ordered symbol tuple used by computed sampling signatures.
        for symbol in tuple(self.parameters):
            if symbol not in specs:
                del self.parameters[symbol]

        self.parameters = {
            symbol: self.figure._parameter_state_for_spec(spec)
            for symbol, spec in specs.items()
        }

        self.parameter_symbols_signal.set(tuple(self.parameters))
        self.figure._prune_unused_parameters()

    def _sample_signature(self) -> SampleSignature:
        """Return the immutable source snapshot that determines sampling."""

        symbols = self.parameter_symbols_signal()
        values = tuple(self.parameters[symbol].value_signal() for symbol in symbols)

        # Track expression edits through a small revision signal. Reading the
        # expression signal directly on every parameter tick forces reaktiv to
        # format large SymPy expressions for debug logging even when logging is
        # disabled, which can dominate interactive updates.
        self._expression_revision_signal()
        return SampleSignature(
            expression=untracked(self.expression_signal),
            view=self.view_signal(),
            parameter_symbols=symbols,
            parameter_values=values,
        )

    def _controls_signature(self) -> tuple[tuple[object, ...], ...]:
        """Return metadata that determines the visible controls layout."""

        rows = []
        for symbol in self.parameter_symbols_signal():
            metadata = self.parameters[symbol].metadata_signal()
            rows.append(
                (
                    symbol,
                    metadata.minimum,
                    metadata.maximum,
                    metadata.step,
                    metadata.label,
                    metadata.animated,
                    metadata.animation_mode.value,
                    metadata.animation_rate_hz,
                    _animation_speed_snapshot(metadata),
                    metadata.animation_speed_effective,
                )
            )
        return tuple(rows)

    def _trace_style_snapshot(self) -> tuple[TraceStyleSnapshot, ...]:
        """Return immutable display snapshots for trace styling."""

        style = self.style_signal()
        if self.kind == PLOT_KIND_DOMAIN:
            domain_style = style.get("domain", {})
            boundary_style = style.get("boundary", {})
            return (
                TraceStyleSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_FILL,
                    trace_type="domain-fill",
                    label=self.label_signal(),
                    style=tuple(sorted(dict(domain_style).items())),
                ),
                TraceStyleSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_BOUNDARY,
                    trace_type="domain-boundary",
                    label=f"{self.label_signal()} boundary",
                    style=tuple(sorted(dict(boundary_style).items())),
                ),
            )

        trace_type = _trace_type_for_kind(self.kind)
        return (
            TraceStyleSnapshot(
                node_id=self.id,
                trace_role=TRACE_ROLE_MAIN,
                trace_type=trace_type,
                label=self.label_signal(),
                style=tuple(sorted(style.items())),
            ),
        )

    def _legend_item_snapshot(self) -> LegendItem:
        """Return the toolkit-owned legend row for this plot node."""

        style = self.style_signal()
        self.figure._sound_control_signal()
        if self.kind == PLOT_KIND_DOMAIN:
            marker, visible = _domain_legend_marker(style)
        elif self.kind == PLOT_KIND_TEMPERATURE:
            marker, visible = _heatmap_legend_marker(style)
        elif self.kind == PLOT_KIND_CONTOUR:
            marker, visible = _contour_legend_marker(style)
        else:
            marker, visible = _line_legend_marker(style)
        sound_playable = self.kind == PLOT_KIND_CURVE
        sound_enabled = False
        sound_playing = False
        sound_status = "stopped"
        if sound_playable:
            audio_node = self.figure._audio_nodes.get(self.id)
            if audio_node is not None:
                sound_enabled = bool(audio_node.sound_enabled_signal())
                audio_state = self.figure._audio_controller.playback_state()
                if sound_enabled and audio_state.node_id == audio_node.id:
                    sound_status = audio_state.status
                sound_playing = sound_status == "playing"
        return LegendItem(
            node_id=self.id,
            label_markdown=self.label_signal(),
            visible=visible,
            marker=marker,
            sound_playable=sound_playable,
            sound_enabled=sound_enabled,
            sound_playing=sound_playing,
            sound_status=sound_status,
        )

    def _slider_value_snapshot(self) -> tuple[SliderValueItem, ...]:
        """Return current parameter values without any slider metadata."""

        return tuple(
            SliderValueItem(
                node_id=self.id,
                symbol=symbol,
                value=self.parameters[symbol].value_signal(),
            )
            for symbol in self.parameter_symbols_signal()
        )

    def _sample_into_buffers(self) -> None:
        """Sample the current signature into reusable buffers."""

        self.figure.drain_interactions()
        signature = self.sample_signature()
        if not self.figure._animation_coordinator.animation_write_should_continue(
            signature.parameter_symbols
        ):
            return
        compiled = self._compiled_for_signature(signature)

        if self.kind == PLOT_KIND_CURVE:
            x_values, y_values = sample_curve(compiled, signature)
            self.x_buffer.set(x_values)
            self.y_buffer.set(y_values)
            snapshots = (
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_MAIN,
                    trace_type="scatter",
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                ),
            )
        elif self.kind in {PLOT_KIND_TEMPERATURE, PLOT_KIND_CONTOUR}:
            sample = sample_scalar_field(compiled, signature)
            self.x_buffer.set(sample.x)
            self.y_buffer.set(sample.y)
            self.z_buffer.set(sample.z)
            snapshots = (
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_MAIN,
                    trace_type=_trace_type_for_kind(self.kind),
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                    z=self.z_buffer.active_view,
                ),
            )
        elif self.kind == PLOT_KIND_DOMAIN:
            sample = sample_domain(compiled, signature)
            self.x_buffer.set(sample.x)
            self.y_buffer.set(sample.y)
            self.domain_fill_buffer.set(sample.fill_z)
            self.domain_boundary_buffer.set(sample.boundary_z)
            snapshots = (
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_FILL,
                    trace_type="domain-fill",
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                    z=self.domain_fill_buffer.active_view,
                ),
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_DOMAIN_BOUNDARY,
                    trace_type="domain-boundary",
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                    z=self.domain_boundary_buffer.active_view,
                    contour_level=sample.boundary_level,
                ),
            )
        elif self.kind == PLOT_KIND_LIST:
            x_values, y_values = sample_list_plot(compiled, signature)
            self.x_buffer.set(x_values)
            self.y_buffer.set(y_values)
            snapshots = (
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_MAIN,
                    trace_type="list-scatter",
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                ),
            )
        else:
            x_values, y_values = sample_parametric(compiled, signature)
            self.x_buffer.set(x_values)
            self.y_buffer.set(y_values)
            snapshots = (
                TraceDataSnapshot(
                    node_id=self.id,
                    trace_role=TRACE_ROLE_MAIN,
                    trace_type="scatter",
                    x=self.x_buffer.active_view,
                    y=self.y_buffer.active_view,
                ),
            )

        self.trace_data_signal.set(snapshots)

    def _compiled_for_signature(self, signature: SampleSignature) -> object:
        """Return a cached numeric callable or compile a new one."""

        independent = _independent_symbols_for_view(signature.view)
        key = (
            self.kind,
            signature.expression,
            independent,
            signature.parameter_symbols,
        )
        if key == self._compiled_key:
            return self._compiled_numeric

        if self.kind == PLOT_KIND_CURVE:
            compiled = compile_numeric_curve(
                signature.expression,
                independent[0],
                signature.parameter_symbols,
            )
        elif self.kind == PLOT_KIND_TEMPERATURE:
            compiled = compile_numeric_field(
                signature.expression,
                independent[0],
                independent[1],
                signature.parameter_symbols,
                plotter="temperature_plot",
            )
        elif self.kind == PLOT_KIND_CONTOUR:
            compiled = compile_numeric_field(
                signature.expression,
                independent[0],
                independent[1],
                signature.parameter_symbols,
                plotter="contour_plot",
            )
        elif self.kind == PLOT_KIND_DOMAIN:
            compiled = compile_numeric_domain(
                signature.expression,
                independent[0],
                independent[1],
                signature.parameter_symbols,
            )
        elif self.kind == PLOT_KIND_LIST:
            compiled = compile_numeric_list(
                signature.expression,
                independent[0] if independent else None,
                signature.parameter_symbols,
            )
        else:
            compiled = compile_numeric_parametric(
                signature.expression,
                independent[0],
                signature.parameter_symbols,
            )

        self._compiled_numeric = compiled
        self._compiled_key = key
        return compiled


class FigureHandle:
    """Represent a durable notebook plotting figure.

    Methods
    -------
    show
        Display a fresh live widget generation for this figure.
    """

    _ids = count(1)

    def __init__(
        self,
        *,
        name: str | None = None,
        layout_class: type[FigureLayout] = ResponsiveSidebarLayout,
        layout_options: dict[str, object] | None = None,
        backend: str | None = None,
    ) -> None:
        """Create an empty figure handle without constructing widgets."""

        self.id = next(self._ids)
        self._name = name
        self._layout_class = self._normalize_layout_class(layout_class)
        self._layout_options = self._normalize_layout_options(layout_options)
        self._default_backend = normalize_display_backend(backend)
        self.plots: list[PlotNode] = []
        self.plots_by_name: dict[str, PlotNode] = {}
        self.info_cards: list[InfoCard] = []
        self.info_cards_by_name: dict[str, InfoCard] = {}
        self._parameter_definitions: dict[sympy.Basic, ParameterSpec] = {}
        self._explicit_parameter_definitions: set[sympy.Basic] = set()
        self._parameters: dict[sympy.Basic, ParameterState] = {}
        self.parameters = FigureParameters(self)
        self.parameter = self.parameters
        self._interaction_queue = FigureInteractionQueue()
        self._animation_coordinator = FigureAnimationCoordinator(self)
        self.views: list[ViewHandle] = []
        self.views_by_name: dict[str, ViewHandle] = {}
        self.view = FigureView(self)
        self.info = InfoCommand(self)
        self._current_view_signal = Signal(None, equal=_semantic_equal)
        self._plot_topology_signal = Signal((), equal=_semantic_equal)
        self._info_topology_signal = Signal((), equal=_semantic_equal)
        self.info_snapshot = Computed(self._info_snapshot, equal=_semantic_equal)
        self._sound_control_signal = Signal(0)
        self._generations: list[FigureDisplayGeneration] = []
        self._active_generation: FigureDisplayGeneration | None = None
        self._next_generation_id = 1
        self._renderer: object | None = None
        self._output_notices: list[str] = []
        self._emitted_output_notices: set[str] = set()
        self._output_area_context_stack: list[object | None] = []
        self._context_batch_stack: list[object] = []
        self._display_update_hold_depth = 0
        self._pending_display_update = False
        from .audio import FigureAudioController, FigureSound

        self._audio_nodes: dict[int, object] = {}
        self._audio_controller = FigureAudioController(self)
        self.sound = FigureSound(self, self._audio_controller)

    def queue_interaction(
        self,
        callback: InteractionCallback,
        *,
        clear_pending: bool = False,
    ) -> None:
        """Queue a UI interaction to run before animation recomputation."""

        self._interaction_queue.enqueue(callback, clear_pending=clear_pending)

    def drain_interactions(self) -> None:
        """Run queued UI interactions before starting heavier figure work."""

        self._interaction_queue.drain()

    def clear_interactions(self) -> None:
        """Discard queued UI interactions."""

        self._interaction_queue.clear()

    @property
    def name(self) -> str | None:
        """Return the manager-recoverable figure name, if this figure has one."""

        if self._name is None:
            return None
        from .session import get_session

        if get_session().named_figures.get(self._name) is self:
            return self._name
        return None

    def _set_manager_name(self, name: str | None) -> None:
        """Store the figure-manager name candidate assigned by the session."""

        self._name = name

    def __enter__(self) -> FigureHandle:
        """Route plots and notebook output in the context body to this figure."""

        from .session import get_session

        get_session().push_figure(self)
        batch_context = batch()
        batch_context.__enter__()
        self._context_batch_stack.append(batch_context)
        self._display_update_hold_depth += 1
        output_context = self._output_area_context()
        if output_context is None:
            self._output_area_context_stack.append(None)
            return self
        try:
            output_context.__enter__()
        except Exception:
            self._display_update_hold_depth -= 1
            self._context_batch_stack.pop().__exit__(*sys.exc_info())
            get_session().pop_figure(self)
            raise
        self._output_area_context_stack.append(output_context)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool | None:
        """Restore the previous plot and notebook output routing targets."""

        from .session import get_session

        output_context = self._output_area_context_stack.pop()
        try:
            if output_context is not None:
                output_context.__exit__(None, None, None)
        finally:
            try:
                get_session().pop_figure(self)
            finally:
                batch_context = self._context_batch_stack.pop()
                self._display_update_hold_depth -= 1
                batch_context.__exit__(exc_type, exc, traceback)
                if self._display_update_hold_depth == 0 and self._pending_display_update:
                    self._pending_display_update = False
                    if exc_type is None:
                        self._refresh_display_after_plot_call()
        return None

    @property
    def widget(self) -> object:
        """Return the active display generation root widget."""

        return self._ensure_display_generation().root

    @property
    def figure_widget(self) -> object:
        """Return the active generation's underlying Plotly FigureWidget."""

        return self._ensure_display_generation().renderer.figure_widget

    @property
    def layout(self) -> object:
        """Return the active generation's layout shell."""

        return self._ensure_display_generation().layout

    @property
    def layout_instance(self) -> object:
        """Return the active generation's authored layout object."""

        return self.layout.layout_instance

    @property
    def layout_style(self) -> type[FigureLayout]:
        """Return the layout class used for future display generations."""

        return self._layout_class

    @property
    def layout_options(self) -> dict[str, object]:
        """Return layout constructor options for future display generations."""

        return dict(self._layout_options)

    @property
    def default_backend(self) -> str:
        """Return the backend used by implicit and default figure display."""

        return self._default_backend

    def set_default_backend(self, backend: str | None) -> FigureHandle:
        """Store the backend used by future default display generations."""

        self._default_backend = normalize_display_backend(backend)
        return self

    def enqueue_interaction(
        self,
        callback: InteractionCallback,
        *,
        clear_pending: bool = False,
    ) -> None:
        """Alias for queueing an external UI interaction."""

        if not callable(callback):
            raise TypeError("FigureHandle.enqueue_interaction(...) requires a callable.")
        self.queue_interaction(callback, clear_pending=clear_pending)

    @property
    def params(self) -> dict[sympy.Symbol, dict[str, object]]:
        """Return figure-owned parameter values and slider metadata."""

        # Return detached dictionaries so local edits stay draft state until the
        # caller assigns them back through this property or calls ``set_params``.
        params: dict[sympy.Symbol, dict[str, object]] = {}
        for symbol, spec in self._parameter_specs().items():
            state = self._parameters.get(symbol)
            params[symbol] = {
                "value": spec.value,
                "default_value": spec.value,
                "current_value": spec.value if state is None else state.value,
                "min": spec.metadata.minimum,
                "max": spec.metadata.maximum,
                "step": spec.metadata.step,
                "label": spec.metadata.label,
                "animated": spec.metadata.animated,
                "animation_mode": spec.metadata.animation_mode.value,
                "animation_rate_hz": spec.metadata.animation_rate_hz,
                "animation_speed": _animation_speed_snapshot(spec.metadata),
                "animation_speed_effective": spec.metadata.animation_speed_effective,
            }
        return params

    @params.setter
    def params(self, params: object) -> None:
        """Apply figure-owned parameter values or metadata."""

        self.set_params(params)

    def _define_parameters(self, params: object) -> FigureHandle:
        """Define figure-owned parameter values or metadata and return self."""

        if not isinstance(params, Mapping):
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )
        for symbol in params:
            if not isinstance(symbol, (sympy.Symbol, sympy.Indexed)):
                raise PlotSpecError(
                    "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                    'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
                )

        specs = normalize_parameter_specs(
            tuple(params),
            (),
            params,
            existing=self._parameter_specs(),
        )
        self._parameter_definitions.update(specs)
        self._explicit_parameter_definitions.update(specs)
        pending_plots: list[tuple[PlotNode, dict[sympy.Basic, ParameterSpec]]] = []
        existing_specs = dict(self._parameter_definitions)
        existing_specs.update(self._parameter_specs())
        for node in self.plots:
            updated_specs = normalize_parameter_specs(
                node.expression,
                node.independent_symbols,
                params,
                existing={**existing_specs, **node.parameter_specs()},
            )
            if updated_specs.keys() == node.parameter_specs().keys():
                continue
            _validate_plot_candidate(node.kind, node.expression, node.view, updated_specs)
            pending_plots.append((node, updated_specs))
        for symbol, spec in specs.items():
            if symbol in self._parameters:
                self._parameter_state_for_spec(spec)
        with batch():
            for node, updated_specs in pending_plots:
                node._set_parameter_specs(updated_specs)
        self.rebuild_controls()
        self._auto_show_for_model_change()
        return self

    def set_params(self, params: object) -> FigureHandle:
        """Update figure-owned parameter values or metadata and return self."""

        if not isinstance(params, Mapping):
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )

        updates: dict[sympy.Basic, object] = {}
        current_value_updates: dict[sympy.Basic, object] = {}
        preserve_current_symbols: set[sympy.Basic] = set()
        for symbol, raw_spec in params.items():
            if not isinstance(symbol, (sympy.Symbol, sympy.Indexed)):
                raise PlotSpecError(
                    "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                    'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
                )
            if symbol not in self._parameters:
                raise PlotSpecError(
                    f"Figure has no active parameter named {symbol!s}."
                )
            if isinstance(raw_spec, Mapping):
                unknown = set(raw_spec) - {
                    "value",
                    "default_value",
                    "current_value",
                    "min",
                    "max",
                    "step",
                    "label",
                    "animated",
                    "animation_mode",
                    "animation_rate_hz",
                    "animation_speed",
                }
                if "animation_speed_effective" in raw_spec:
                    raise PlotSpecError(
                        "animation_speed_effective is read-only; assign animation_speed instead."
                    )
                if unknown:
                    raise PlotSpecError(
                        "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                        'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
                    )
                if "value" in raw_spec and (
                    "default_value" in raw_spec or "current_value" in raw_spec
                ):
                    raise PlotSpecError(
                        "Parameter specs must use 'value' or split value fields, not both."
                    )
                if "current_value" in raw_spec:
                    current_value_updates[symbol] = raw_spec["current_value"]
                if "default_value" in raw_spec and "value" not in raw_spec:
                    preserve_current_symbols.add(symbol)
                updates[symbol] = {
                    key: value
                    for key, value in raw_spec.items()
                    if key != "current_value"
                }
            else:
                updates[symbol] = raw_spec

        pending_plots: list[tuple[PlotNode, dict[sympy.Basic, ParameterSpec]]] = []
        for node in self.plots:
            node_updates = {
                symbol: raw_spec
                for symbol, raw_spec in updates.items()
                if symbol in node.parameter_symbols
            }
            if not node_updates:
                continue
            specs = normalize_parameter_specs(
                node.expression,
                node.independent_symbols,
                node_updates,
                existing=node.parameter_specs(),
            )
            _validate_plot_candidate(node.kind, node.expression, node.view, specs)
            pending_plots.append((node, specs))

        pending_info: list[tuple[InfoCard, dict[sympy.Basic, ParameterSpec]]] = []
        for card in self.info_cards:
            card_updates = {
                symbol: raw_spec
                for symbol, raw_spec in updates.items()
                if symbol in card.parameter_symbols
            }
            if not card_updates:
                continue
            specs = normalize_parameter_specs(
                _symbolic_fragments_expression(card.fragments),
                (),
                card_updates,
                existing=card.parameter_specs(),
            )
            pending_info.append((card, specs))

        preserved_current_values = {
            symbol: self._parameters[symbol].value
            for symbol in preserve_current_symbols
            if symbol in self._parameters
        }
        with batch():
            for node, specs in pending_plots:
                node._set_parameter_specs(specs)
            for card, specs in pending_info:
                card._set_parameter_specs(specs)
            for symbol, value in preserved_current_values.items():
                if symbol in self._parameters:
                    self._parameters[symbol].set_value(value)
            for symbol, value in current_value_updates.items():
                if symbol in self._parameters:
                    self._parameters[symbol].set_value(
                        self._finite_parameter_float(value, "current value")
                    )
        return self

    @property
    def active_generation(self) -> FigureDisplayGeneration | None:
        """Return the currently live display generation, if one exists."""

        return self._active_generation

    @property
    def layout_class(self) -> type[FigureLayout]:
        """Return the layout class used for future display generations."""

        return self.layout_style

    def _view_handle(self, target: str | ViewHandle | None = None) -> ViewHandle:
        """Return or create a view handle without making it current."""

        if isinstance(target, ViewHandle):
            if target.figure is not self:
                raise PlotSpecError("fig.view(...) received a view from another figure.")
            return target
        if target is None:
            return self._create_view(name=None)
        if isinstance(target, str):
            view = self.views_by_name.get(target)
            if view is None:
                view = self._create_view(name=target)
            return view
        raise PlotSpecError("fig.view(...) expects no argument, a name, or a ViewHandle.")

    def _current_view(self) -> ViewHandle:
        """Return this figure's current view, creating one lazily."""

        current = self._current_view_signal()
        if current is None:
            current = self._create_view(name=None)
            self._current_view_signal.set(current)
        return current

    def _set_current_view(
        self,
        target: str | ViewHandle | None = None,
        *,
        only_existing: bool = False,
    ) -> ViewHandle:
        """Set and return the view targeted by unnamed view commands."""

        view = self._resolve_view_target(target, only_existing=only_existing)
        self._publish_current_view(view)
        return view

    def _set_view_range(
        self,
        target: str | ViewHandle | None = None,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Change a view's visible ranges and make it current."""

        # Visible range edits target an existing view. Omitting the selector
        # means "the current view", matching reset-style commands.
        view = self._resolve_existing_view(target, operation="fig.view.range")
        state = view._state()

        # Patch only the requested axes and preserve unspecified visible
        # ranges, so per-axis properties can move one axis deliberately.
        next_x_view, next_y_view = _resolve_axis_view_update(
            state.x_view,
            state.y_view,
            x_range=x_range,
            y_range=y_range,
        )
        next_state = replace(state, x_view=next_x_view, y_view=next_y_view)

        # Make the moved view current and resample view-aware plots from the
        # same visible ranges rendered into the frontend layout.
        with batch():
            view._set_state(next_state)
            self._current_view_signal.set(view)
            self._apply_view_state_to_plots(next_state)
        self._auto_show_for_model_change()
        return view

    def _set_view_home_range(
        self,
        target: str | ViewHandle | None = None,
        *,
        x_range: object = OMITTED,
        y_range: object = OMITTED,
    ) -> ViewHandle:
        """Change a view's reset/home ranges without moving its visible ranges."""

        # Home edits do not move the view. With no ranges supplied, the current
        # visible rectangle becomes the new reset target.
        view = self._resolve_existing_view(
            target,
            operation="fig.view.home_range",
        )
        state = view._state()
        if x_range is OMITTED and y_range is OMITTED:
            next_state = replace(
                state,
                home_x_view=state.x_view,
                home_y_view=state.y_view,
            )
        else:
            # Partial updates preserve unspecified home axes, which lets
            # callers reset only the x or y home range deliberately.
            next_home_x, next_home_y = _resolve_axis_view_update(
                state.home_x_view,
                state.home_y_view,
                x_range=x_range,
                y_range=y_range,
            )
            next_state = replace(
                state,
                home_x_view=next_home_x,
                home_y_view=next_home_y,
            )
        # Before the first display exists, the home rectangle is also the
        # initial visible rectangle. After display, a home edit updates only
        # the reset target so an already panned plot does not jump.
        if self.active_generation is None:
            next_state = replace(
                next_state,
                x_view=next_state.home_x_view,
                y_view=next_state.home_y_view,
            )
            with batch():
                view._set_state(next_state)
                self._apply_view_state_to_plots(next_state)
            return view

        view._set_state(next_state)
        self._auto_show_for_model_change()
        return view

    def _reset_view(self, target: str | ViewHandle | None = None) -> ViewHandle:
        """Reset a view to its home ranges and make it current."""

        view = self._resolve_existing_view(target, operation="fig.view.reset")
        state = view._state()
        next_state = replace(
            state,
            x_view=state.home_x_view,
            y_view=state.home_y_view,
        )
        with batch():
            view._set_state(next_state)
            self._current_view_signal.set(view)
            self._apply_view_state_to_plots(next_state)
        self._auto_show_for_model_change()
        return view

    def view_snapshot(self) -> FigureViewState:
        """Return the current view state consumed by display generations."""

        self._current_view_signal()
        return self._current_view()._state()

    def plot(
        self,
        expr: object,
        domain: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
        samples: object = OMITTED,
    ) -> PlotHandle:
        """Plot or update one sampled two-dimensional function curve.

        Parameters
        ----------
        expr : object
            SymPy-compatible scalar expression describing ``y = f(x)``.
            Symbols not used as the independent variable become figure-owned
            parameters.
        domain : object, optional
            Independent variable symbol such as ``x`` for view-aware sampling,
            or a finite interval tuple such as ``(x, -10, 10)``. Named updates
            may omit it to preserve the existing domain.
        name : str, optional
            Plot identity within this figure. Reusing a name updates the
            existing curve in place when it is already a curve plot.
        label : object, optional
            Legend display label. It does not define plot identity.
        style : object, optional
            Line style dictionary supporting ``color``, ``width``, ``opacity``,
            ``visible``, and ``dash``.
        samples : object, optional
            Number of sample points to use for the curve.

        Returns
        -------
        PlotHandle
            Handle for updating style, label, parameters, audio, and plot
            removal.

        Raises
        ------
        PlotSpecError
            Raised when the expression, domain, style, samples, or update
            target is invalid for a curve plot.

        Examples
        --------
        Basic usage:

        >>> import sympy
        >>> from math_toolkit import figure
        >>> x = sympy.Symbol("x")
        >>> fig = figure()
        >>> handle = fig.plot(sympy.sin(x), x, name="sine")
        >>> handle.name
        'sine'

        Named update:

        >>> updated = fig.plot(sympy.cos(x), name="sine", label="cosine")
        >>> updated.name
        'sine'
        """

        expression = normalize_expression(expr)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = existing if existing is not None and existing.kind == PLOT_KIND_CURVE else None
        view = normalize_domain(
            domain,
            samples=samples,
            existing=same_kind.view if same_kind is not None else None,
        )
        return self._add_or_update_plot(
            kind=PLOT_KIND_CURVE,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
        )

    def temperature_plot(
        self,
        expr: object,
        x_domain: object = OMITTED,
        y_domain: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
        samples: object = OMITTED,
    ) -> PlotHandle:
        """Add or update one heatmap scalar field in this figure."""

        expression = normalize_expression(expr)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = (
            existing if existing is not None and existing.kind == PLOT_KIND_TEMPERATURE else None
        )
        view = normalize_cartesian_view(
            x_domain,
            y_domain,
            samples=samples,
            existing=same_kind.view if same_kind is not None else None,
            plotter="temperature_plot",
        )
        return self._add_or_update_plot(
            kind=PLOT_KIND_TEMPERATURE,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
        )

    def contour_plot(
        self,
        expr: object,
        x_domain: object = OMITTED,
        y_domain: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
        samples: object = OMITTED,
    ) -> PlotHandle:
        """Add or update one contour scalar field in this figure."""

        expression = normalize_expression(expr)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = existing if existing is not None and existing.kind == PLOT_KIND_CONTOUR else None
        view = normalize_cartesian_view(
            x_domain,
            y_domain,
            samples=samples,
            existing=same_kind.view if same_kind is not None else None,
            plotter="contour_plot",
        )
        return self._add_or_update_plot(
            kind=PLOT_KIND_CONTOUR,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
        )

    def domain_plot(
        self,
        condition: object,
        x_domain: object = OMITTED,
        y_domain: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
        samples: object = OMITTED,
        boundary: bool = True,
    ) -> PlotHandle:
        """Add or update one filled Boolean or signed domain in this figure."""

        if not isinstance(boundary, bool):
            raise PlotSpecError("domain_plot(...) boundary must be True or False.")
        expression = normalize_domain_conditions(condition)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = existing if existing is not None and existing.kind == PLOT_KIND_DOMAIN else None
        view = normalize_cartesian_view(
            x_domain,
            y_domain,
            samples=samples,
            existing=same_kind.view if same_kind is not None else None,
            plotter="domain_plot",
        )
        return self._add_or_update_plot(
            kind=PLOT_KIND_DOMAIN,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
            boundary=boundary,
        )

    def parametric_plot(
        self,
        exprs: object,
        parameter_domain: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
        samples: object = OMITTED,
    ) -> PlotHandle:
        """Add or update one two-dimensional parametric curve in this figure."""

        expression = normalize_parametric_expressions(exprs)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = (
            existing if existing is not None and existing.kind == PLOT_KIND_PARAMETRIC else None
        )
        view = normalize_parametric_view(
            parameter_domain,
            samples=samples,
            existing=same_kind.view if same_kind is not None else None,
        )
        return self._add_or_update_plot(
            kind=PLOT_KIND_PARAMETRIC,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
        )

    def list_plot(
        self,
        source: object,
        index: object = OMITTED,
        *,
        name: str | None = None,
        label: object = OMITTED,
        style: object = OMITTED,
    ) -> PlotHandle:
        """Add or update one discrete list plot in this figure."""

        expression, view = normalize_list_plot_spec(source, index)
        plot_name = _normalize_plot_name(name)
        existing = self.plots_by_name.get(plot_name) if plot_name is not None else None
        same_kind = existing if existing is not None and existing.kind == PLOT_KIND_LIST else None
        if same_kind is not None and index is OMITTED:
            view = same_kind.view
        return self._add_or_update_plot(
            kind=PLOT_KIND_LIST,
            expression=expression,
            view=view,
            plot_name=plot_name,
            label=label,
            style=style,
            existing=existing,
        )

    def get_plot(self, name: object = None) -> PlotHandle:
        """Return the latest plot or a named plot in this figure.

        Parameters
        ----------
        name : object, optional
            Plot identity to retrieve. When omitted or ``None``, the latest
            plot in this figure is returned.

        Returns
        -------
        PlotHandle
            Lightweight handle for the selected plot.

        Raises
        ------
        PlotNotFoundError
            Raised when the figure has no plots or the named plot is missing.
        """

        if name is None:
            if not self.plots:
                raise PlotNotFoundError("No plots exist in this figure.")
            return self._handle_for_node(self.plots[-1])

        plot_name = _normalize_plot_name(name)
        node = self.plots_by_name.get(plot_name)
        if node is None:
            raise PlotNotFoundError(
                f"No plot named {plot_name!r} exists in this figure."
            )
        return self._handle_for_node(node)

    def get_info(self, name: object = None) -> InfoHandle:
        """Return the latest info card or a named info card.

        Parameters
        ----------
        name : object, optional
            Info card identity to retrieve. When omitted or ``None``, the
            latest card in this figure is returned.

        Returns
        -------
        InfoHandle
            Lightweight handle for the selected info card.

        Raises
        ------
        InfoNotFoundError
            Raised when the figure has no info cards or the named card is
            missing.
        """

        if name is None:
            if not self.info_cards:
                raise InfoNotFoundError("No info cards exist in this figure.")
            return InfoHandle(self, self.info_cards[-1])

        info_name = _normalize_plot_name(name)
        card = self.info_cards_by_name.get(info_name)
        if card is None:
            raise InfoNotFoundError(
                f"No info card named {info_name!r} exists in this figure."
            )
        return InfoHandle(self, card)

    def show(
        self,
        *,
        backend: str | None = None,
        policy: str | None = None,
        new: bool = True,
    ) -> FigureDisplayGeneration:
        """Display this figure through the toolkit display manager."""

        display_policy = normalize_display_policy(policy)
        display_backend = (
            self._default_backend
            if backend is None
            else normalize_display_backend(backend)
        )
        active_backend = (
            None
            if self._active_generation is None
            else getattr(self._active_generation, "backend_name", "ipywidgets")
        )
        if new or self._active_generation is None or active_backend != display_backend:
            generation = self._create_generation(
                execution_key=current_execution_key(),
                policy=display_policy,
                backend=display_backend,
            )
        else:
            generation = self._active_generation
        generation.display()
        return generation

    def set_layout(
        self,
        layout_class: type[FigureLayout],
        *,
        layout_options: dict[str, object] | None = None,
    ) -> FigureHandle:
        """Store the layout class used by future display generations."""

        self._layout_class = self._normalize_layout_class(layout_class)
        self._layout_options = self._normalize_layout_options(layout_options)
        return self

    def set_layout_options(
        self,
        layout_options: dict[str, object] | None,
    ) -> FigureHandle:
        """Store layout constructor options used by future display generations."""

        self._layout_options = self._normalize_layout_options(layout_options)
        return self

    def _ipython_display_(self) -> None:
        """Display an undisplayed active figure for IPython's display hook."""

        self._display_once_implicitly()
        return None

    def _repr_mimebundle_(
        self,
        include: object = None,
        exclude: object = None,
    ) -> object:
        """Return the active figure widget MIME bundle for notebook display."""

        generation = self._ensure_display_generation()
        generation.displayed = True
        repr_mimebundle = getattr(generation.root, "_repr_mimebundle_", None)
        if callable(repr_mimebundle):
            return repr_mimebundle(include=include, exclude=exclude)

        mime = getattr(generation.root, "_mime_", None)
        if callable(mime):
            mimetype, data = mime()
            return {mimetype: data}, {}
        return None

    def rebuild_controls(self) -> None:
        """Reconcile parameter slider controls for all current plots."""

        self.reconcile_controls()

    def reconcile_controls(self) -> None:
        """Reconcile slider controls from the current layout snapshot."""

        if self._active_generation is not None:
            self._active_generation.reconcile_controls(self.control_layout_snapshot())

    def sync_controls(self, values: tuple[SliderValueItem, ...]) -> None:
        """Synchronize existing slider values without rebuilding controls."""

        if self._active_generation is not None:
            self._active_generation.sync_controls(values)

    def sync_animation_state(
        self,
        values: tuple[ParameterAnimationStateItem, ...],
    ) -> None:
        """Synchronize existing animation buttons without rebuilding controls."""

        if self._active_generation is not None:
            self._active_generation.sync_animation_state(values)

    def animation_state_snapshot(self) -> tuple[ParameterAnimationStateItem, ...]:
        """Return the current figure-level animation play-state snapshot."""

        return self._animation_coordinator.snapshot()

    def reconcile_legend(self) -> None:
        """Reconcile legend rows from the current plot snapshots."""

        if self._active_generation is not None:
            self._active_generation.reconcile_legend(self.legend_snapshot())

    def control_layout_snapshot(self) -> tuple[ControlLayoutItem, ...]:
        """Return the figure-level slider layout snapshot without values."""

        items: list[ControlLayoutItem] = []
        seen: set[sympy.Symbol] = set()
        for node in self.plots:
            for (
                symbol,
                minimum,
                maximum,
                step,
                label,
                animated,
                animation_mode,
                animation_rate_hz,
                animation_speed,
                animation_speed_effective,
            ) in node.controls_signature():
                if symbol in seen:
                    continue
                seen.add(symbol)
                items.append(
                    ControlLayoutItem(
                        node_id=0,
                        symbol=symbol,
                        label_markdown=_parameter_label_markdown(symbol, label),
                        minimum=minimum,
                        maximum=maximum,
                        step=step,
                        animated=animated,
                        animation_mode=animation_mode,
                        animation_rate_hz=animation_rate_hz,
                        animation_speed=animation_speed,
                        animation_speed_effective=animation_speed_effective,
                    )
                )
        for card in self.info_cards:
            for (
                symbol,
                minimum,
                maximum,
                step,
                label,
                animated,
                animation_mode,
                animation_rate_hz,
                animation_speed,
                animation_speed_effective,
            ) in card.controls_signature():
                if symbol in seen:
                    continue
                seen.add(symbol)
                items.append(
                    ControlLayoutItem(
                        node_id=0,
                        symbol=symbol,
                        label_markdown=_parameter_label_markdown(symbol, label),
                        minimum=minimum,
                        maximum=maximum,
                        step=step,
                        animated=animated,
                        animation_mode=animation_mode,
                        animation_rate_hz=animation_rate_hz,
                        animation_speed=animation_speed,
                        animation_speed_effective=animation_speed_effective,
                    )
                )
        return tuple(items)

    def slider_value_snapshot(self) -> tuple[SliderValueItem, ...]:
        """Return the figure-level slider value snapshot."""

        items: list[SliderValueItem] = []
        for symbol in self._ordered_active_parameter_symbols():
            items.append(
                SliderValueItem(
                    node_id=0,
                    symbol=symbol,
                    value=self._parameters[symbol].value_signal(),
                )
            )
        return tuple(items)

    def legend_snapshot(self) -> tuple[LegendItem, ...]:
        """Return the figure-level legend row snapshot in plot order."""

        return tuple(node.legend_item_snapshot() for node in self.plots)

    def _ordered_active_parameter_symbols(self) -> tuple[sympy.Basic, ...]:
        """Return visible figure parameters in control layout order."""

        ordered: list[sympy.Basic] = []
        for node in self.plots:
            for symbol in node.parameter_symbols_signal():
                if symbol not in ordered:
                    ordered.append(symbol)
        for card in self.info_cards:
            for symbol in card.parameter_symbols_signal():
                if symbol not in ordered:
                    ordered.append(symbol)
        return tuple(ordered)

    def remove_plot(self, node: PlotNode) -> None:
        """Remove a plot node from this figure and dispose its resources."""

        if node not in self.plots:
            return
        self._dispose_audio_for_plot(node)
        self.plots.remove(node)
        if node.name is not None and self.plots_by_name.get(node.name) is node:
            del self.plots_by_name[node.name]
        if self._active_generation is not None:
            self._active_generation.detach_node(node)
        node.dispose()
        self._prune_unused_parameters()
        self._publish_plot_topology()
        self.rebuild_controls()

    def close(self) -> None:
        """Dispose effects and widget observers owned by this figure."""

        self._animation_coordinator.stop_all()
        self._audio_controller.stop()
        for audio_node in tuple(self._audio_nodes.values()):
            dispose = getattr(audio_node, "dispose", None)
            if dispose is not None:
                dispose()
        self._audio_nodes.clear()
        for generation in tuple(self._generations):
            generation.close()
        self._generations.clear()
        self._active_generation = None
        self._renderer = None
        for node in tuple(self.plots):
            node.dispose()
        self.plots.clear()
        self.plots_by_name.clear()
        self.info_cards.clear()
        self.info_cards_by_name.clear()
        self._parameter_definitions.clear()
        self._explicit_parameter_definitions.clear()
        self._parameters.clear()
        self.views.clear()
        self.views_by_name.clear()
        self._current_view_signal.set(None)
        self._publish_plot_topology()
        self._publish_info_topology()

    def _add_or_update_plot(
        self,
        *,
        kind: str,
        expression: object,
        view: CurveView | CartesianView2D | ParametricView | ListView,
        plot_name: str | None,
        label: object,
        style: object,
        existing: PlotNode | None,
        boundary: bool = True,
    ) -> PlotHandle:
        """Add or update a plot node after kind-specific normalization."""

        view = self._bind_plot_view_to_current_view(view)
        same_kind = existing if existing is not None and existing.kind == kind else None
        existing_parameter_specs = dict(self._parameter_definitions)
        existing_parameter_specs.update(self._parameter_specs())
        if same_kind is not None:
            existing_parameter_specs.update(same_kind.parameter_specs())
        parameter_specs = normalize_parameter_specs(
            expression,
            _independent_symbols_for_view(view),
            OMITTED,
            existing=existing_parameter_specs,
        )
        self._notice_default_indexed_parameters(expression, parameter_specs)
        node_label = _next_label(
            expression,
            plot_name=plot_name,
            label=label,
            existing=same_kind,
        )
        node_style = _next_style(
            kind,
            style,
            existing=same_kind,
            boundary=boundary,
            default_color=self._default_color_for_plot(existing),
        )
        _validate_plot_candidate(kind, expression, view, parameter_specs)

        # A named kind change deliberately creates a fresh node after the new
        # plot has validated, so invalid updates leave the previous plot intact.
        if existing is not None and existing.kind != kind:
            self.remove_plot(existing)
            existing = None
        if existing is not None:
            existing.update(
                expression=expression,
                view=view,
                label=node_label,
                parameters=parameter_specs,
                style=node_style,
            )
            self._auto_show_for_plot_call()
            return self._handle_for_node(existing)

        node = PlotNode(
            self,
            kind=kind,
            name=plot_name,
            expression=expression,
            view=view,
            label=node_label,
            parameters=parameter_specs,
            style=node_style,
        )
        self.plots.append(node)
        if plot_name is not None:
            self.plots_by_name[plot_name] = node
        if node.kind == PLOT_KIND_CURVE:
            self._audio_node_for_plot(node)
        self._publish_plot_topology()
        self._attach_node(node)
        self._auto_show_for_plot_call()
        return self._handle_for_node(node)

    def _add_or_update_info(
        self,
        fragments: tuple[object, ...],
        *,
        name: object,
        title: object,
        params: object,
    ) -> InfoHandle:
        """Add or replace one authored info card."""

        normalized_fragments = _normalize_info_fragments(fragments)
        info_name = _normalize_plot_name(name)
        existing = self.info_cards_by_name.get(info_name) if info_name is not None else None
        existing_parameter_specs = dict(self._parameter_definitions)
        existing_parameter_specs.update(self._parameter_specs())
        if existing is not None:
            existing_parameter_specs.update(existing.parameter_specs())
        parameter_specs = normalize_parameter_specs(
            _symbolic_fragments_expression(normalized_fragments),
            (),
            params,
            existing=existing_parameter_specs,
        )
        card_title = _next_info_title(title, existing=existing)

        if existing is not None:
            existing.update(
                title=card_title,
                fragments=normalized_fragments,
                parameters=parameter_specs,
            )
            self._publish_info_topology()
            self.rebuild_controls()
            self._auto_show_for_plot_call()
            return InfoHandle(self, existing)

        card = InfoCard(
            self,
            name=info_name,
            title=card_title,
            fragments=normalized_fragments,
            parameters=parameter_specs,
        )
        self.info_cards.append(card)
        if info_name is not None:
            self.info_cards_by_name[info_name] = card
        self._publish_info_topology()
        self.rebuild_controls()
        self._auto_show_for_plot_call()
        return InfoHandle(self, card)

    def _clear_info(self, name: object = None) -> None:
        """Remove one named info card or all info cards."""

        if name is None:
            for card in tuple(self.info_cards):
                self._remove_info_card(card, publish=False)
            self._prune_unused_parameters()
            self._publish_info_topology()
            self.rebuild_controls()
            return None

        info_name = _normalize_plot_name(name)
        card = self.info_cards_by_name.get(info_name)
        if card is None:
            raise InfoNotFoundError(
                f"No info card named {info_name!r} exists in this figure."
            )
        self._remove_info_card(card)
        return None

    def _remove_info_card(self, card: InfoCard, *, publish: bool = True) -> None:
        """Remove one info card from this figure."""

        if card not in self.info_cards:
            return
        self.info_cards.remove(card)
        if card.name is not None and self.info_cards_by_name.get(card.name) is card:
            del self.info_cards_by_name[card.name]
        self._prune_unused_parameters()
        if publish:
            self._publish_info_topology()
            self.rebuild_controls()

    def _info_snapshot(self) -> tuple[InfoCardSnapshot, ...]:
        """Return rendered info cards in figure order."""

        self._info_topology_signal()
        if any(card.has_callable_signal() for card in self.info_cards):
            self._read_broad_info_dependencies()
        return tuple(card.snapshot() for card in self.info_cards)

    def _read_info_render_dependencies(self) -> None:
        """Read info dependencies without rendering Markdown fragments."""

        self._info_topology_signal()
        if any(card.has_callable_signal() for card in self.info_cards):
            self._read_broad_info_dependencies()
        for card in self.info_cards:
            card.title_signal()
            card.fragments_signal()
            symbols = card.parameter_symbols_signal()
            for symbol in symbols:
                card.parameters[symbol].value_signal()

    def _read_broad_info_dependencies(self) -> None:
        """Read broad model signals for callable info fragments."""

        self._plot_topology_signal()
        current_view = self._current_view_signal()
        if current_view is not None:
            current_view._state_signal()
        for node in self.plots:
            node.expression_signal()
            node.view_signal()
            node.label_signal()
            node.style_signal()
            node.trace_data_signal()
        for symbol in sort_symbols(self._parameters):
            self._parameters[symbol].value_signal()
        self._info_topology_signal()

    def _handle_for_node(self, node: PlotNode) -> PlotHandle:
        """Return the public handle type appropriate for one plot node."""

        if node.kind == PLOT_KIND_CURVE:
            return CurvePlotHandle(self, node)
        return PlotHandle(self, node)

    def _audio_node_for_plot(self, node: PlotNode) -> object:
        """Return the figure-owned audio node for one curve plot."""

        if node not in self.plots:
            raise PlotSpecError("Cannot play a plot that is no longer in its figure.")
        if node.kind != PLOT_KIND_CURVE:
            from .audio import AudioPlaybackError

            raise AudioPlaybackError("Only ordinary plot(...) curves expose sound.")
        audio_node = self._audio_nodes.get(node.id)
        if audio_node is None:
            from .audio import AudioNode

            audio_node = AudioNode(self, node)
            self._audio_nodes[node.id] = audio_node
        return audio_node

    def _dispose_audio_for_plot(self, node: PlotNode) -> None:
        """Stop and dispose the audio node attached to a removed plot."""

        audio_node = self._audio_nodes.pop(node.id, None)
        if audio_node is None:
            return
        if self._audio_controller.active_node_signal() is audio_node:
            self._audio_controller.stop()
            self._audio_controller.active_node_signal.set(None)
        dispose = getattr(audio_node, "dispose", None)
        if dispose is not None:
            dispose()

    def _send_audio_output_command(self, content: dict[str, object]) -> None:
        """Send an audio command to the active display generation, if present."""

        generation = self._active_generation
        if generation is None:
            return
        audio_output = getattr(generation, "audio_output", None)
        if audio_output is None:
            return
        audio_output.send_command(content)

    def _send_audio_output_chunk(self, chunk: object) -> None:
        """Send a sampled PCM chunk to the active display generation."""

        generation = self._active_generation
        if generation is None:
            return
        audio_output = getattr(generation, "audio_output", None)
        if audio_output is None:
            return
        audio_output.send_chunk(chunk)

    def _send_audio_output_batch(self, batch: object) -> None:
        """Send a sampled PCM batch to the active display generation."""

        generation = self._active_generation
        if generation is None:
            return
        audio_output = getattr(generation, "audio_output", None)
        if audio_output is None:
            return
        audio_output.send_batch(batch)

    def _ensure_renderer(self) -> object:
        """Return the active generation renderer, constructing it lazily."""

        return self._ensure_display_generation().renderer

    def _ensure_display_generation(self) -> FigureDisplayGeneration:
        """Return the active display generation, constructing it lazily."""

        if self._active_generation is None:
            self._create_generation(
                execution_key=current_execution_key(),
                policy="disconnect",
                backend=self._default_backend,
            )
        assert self._active_generation is not None
        return self._active_generation

    def _output_area_context(self) -> object | None:
        """Return the active layout's output context manager, if one exists."""

        generation = self._active_generation
        if generation is not None and getattr(generation, "frontend", None) is not None:
            return generation.frontend.output
        if generation is None:
            return None
        output_context = getattr(self.layout.layout_instance, "output_area", None)
        if output_context is None:
            return None
        output_context.__enter__
        output_context.__exit__
        return output_context

    def _attach_node(self, node: PlotNode) -> None:
        """Attach a new plot node to the active display generation, if any."""

        if self._active_generation is not None:
            if self._display_update_hold_depth > 0:
                self._pending_display_update = True
                return
            self._active_generation.attach_node(node)

    @contextmanager
    def _coalesced_trace_data_updates(self) -> Iterator[None]:
        """Coalesce trace data rendering for one model-side mutation."""

        generation = self._active_generation
        if generation is None:
            yield
            return
        with generation.defer_trace_data_updates():
            yield

    def _default_color_for_plot(self, existing: PlotNode | None) -> str:
        """Return the palette color for a plot's durable figure slot."""

        if existing is not None and existing in self.plots:
            index = self.plots.index(existing)
            return _PLOT_COLOR_CYCLE[index % len(_PLOT_COLOR_CYCLE)]

        # New traces should avoid colors already visible in the figure. This
        # keeps automatic colors rotating even when earlier traces declared
        # explicit styles that overlap the default palette.
        used_colors = {
            color
            for node in self.plots
            for color in _plot_style_colors(node.kind, node.style).values()
        }
        for color in _PLOT_COLOR_CYCLE:
            if color not in used_colors:
                return color
        index = len(self.plots)
        return _PLOT_COLOR_CYCLE[index % len(_PLOT_COLOR_CYCLE)]

    def _auto_show_for_plot_call(self) -> None:
        """Keep existing displayed figures live after a plot command."""

        self._auto_show_for_model_change()

    def _auto_show_for_model_change(self) -> None:
        """Keep existing displayed figures live after model-side mutations."""

        if self._active_generation is None:
            return
        if not self._active_generation.displayed:
            return
        if self._display_update_hold_depth > 0:
            self._pending_display_update = True
            return
        self._refresh_display_after_model_change()

    def _refresh_display_after_plot_call(self) -> None:
        """Push the current figure model into an already displayed generation."""

        self._refresh_display_after_model_change()

    def _refresh_display_after_model_change(self) -> None:
        """Push the current figure model into an already displayed generation."""

        if self._active_generation is None:
            return
        if not self._active_generation.displayed:
            return
        self._active_generation.refresh_from_model()
        self._active_generation.display()

    def _display_once_implicitly(self) -> None:
        """Display the active generation unless this figure is already visible."""

        if self._active_generation is not None and self._active_generation.displayed:
            return
        if self._active_generation is None:
            self._create_generation(
                execution_key=current_execution_key(),
                policy="disconnect",
                backend=self._default_backend,
            )
        assert self._active_generation is not None
        self._active_generation.display()

    def _create_generation(
        self,
        *,
        execution_key: object,
        policy: str,
        backend: str = "ipywidgets",
    ) -> FigureDisplayGeneration:
        """Create and activate a fresh display generation."""

        self._current_view()
        if self._active_generation is not None:
            self._active_generation.retire()

        generation = FigureDisplayGeneration(
            self,
            generation_id=self._next_generation_id,
            execution_key=execution_key,
            policy=policy,
            backend=backend,
        )
        self._next_generation_id += 1
        self._active_generation = generation
        self._renderer = generation.renderer
        self._generations.append(generation)
        generation.hydrate()
        self._flush_output_notices()
        return generation

    def _notice_default_indexed_parameters(
        self,
        expression: object,
        parameter_specs: Mapping[sympy.Basic, ParameterSpec],
    ) -> None:
        """Queue guidance when indexed array slider length is only a default."""

        if not isinstance(expression, sympy.Basic):
            return
        active_symbols = set(parameter_specs)
        for info in indexed_runtime_parameter_info(expression):
            if info.complete:
                continue
            entries = tuple(entry for entry in info.entries if entry in active_symbols)
            if not entries:
                continue
            last_entry = entries[-1]
            next_index = _next_indexed_entry(last_entry)
            entry_text = ", ".join(str(entry) for entry in entries)
            message = (
                f"Created default sliders {entry_text} for indexed parameter "
                f"{info.base}. The number of entries was not known at plot time. "
                f"Add more with `fig.parameters({{{next_index}: 0.0}})` before "
                "or after plotting."
            )
            self._queue_output_notice(message)

    def _queue_output_notice(self, message: str) -> None:
        """Append one figure output notice, deferring until display if needed."""

        if message in self._emitted_output_notices:
            return
        self._emitted_output_notices.add(message)
        self._output_notices.append(message)
        self._flush_output_notices()

    def _flush_output_notices(self) -> None:
        """Write queued notices to the active generation's output area."""

        generation = self._active_generation
        if generation is None or not self._output_notices:
            return
        notices = tuple(self._output_notices)
        self._output_notices.clear()
        for message in notices:
            if generation.frontend is not None:
                generation.frontend.output.append_markdown(message)
                continue
            _append_markdown_to_layout_output(generation.layout.output_area, message)

    def _normalize_layout_class(self, layout_class: type[FigureLayout]) -> type[FigureLayout]:
        """Validate and return one plotting layout class."""

        if not isinstance(layout_class, type):
            raise PlotSpecError(
                "Figure layouts must be provided as a class, such as "
                "fig.set_layout(MyLayoutClass)."
            )
        build = getattr(layout_class, "build", None)
        if not callable(build):
            raise PlotSpecError(
                "Figure layouts must define a build() method on the layout class."
            )
        return layout_class

    def _normalize_layout_options(
        self,
        layout_options: dict[str, object] | None,
    ) -> dict[str, object]:
        """Validate and copy layout constructor options."""

        if layout_options is None:
            return {}
        if not isinstance(layout_options, dict):
            raise PlotSpecError("layout_options must be a dictionary.")
        return dict(layout_options)

    def _publish_plot_topology(self) -> None:
        """Publish the ordered plot identity tuple for generation effects."""

        self._plot_topology_signal.set(tuple(node.id for node in self.plots))

    def _publish_info_topology(self) -> None:
        """Publish the ordered info card identity tuple for generation effects."""

        self._info_topology_signal.set(tuple(card.id for card in self.info_cards))

    def _parameter_state_for_spec(self, spec: ParameterSpec) -> ParameterState:
        """Return the figure-level state for a parameter, applying new metadata."""

        self._parameter_definitions[spec.symbol] = spec
        state = self._parameters.get(spec.symbol)
        if state is None:
            state = ParameterState(self, spec)
            self._parameters[spec.symbol] = state
        else:
            previous_value = state.value
            previous_metadata = state.metadata
            state.set_spec(spec)
            if spec.value != previous_value:
                self._animation_coordinator.notify_external_value_write(spec.symbol)
            if _parameter_metadata_change_stops_animation(
                previous_metadata,
                spec.metadata,
            ):
                self._animation_coordinator.notify_metadata_write(spec.symbol)
        return state

    def _set_parameter_default_spec(
        self,
        symbol: sympy.Basic,
        default_value: object,
        metadata: ParameterMetadata,
    ) -> None:
        """Update a parameter's reset value and metadata without moving its current value."""

        state = self._ensure_parameter_state(symbol)
        value = self._finite_parameter_float(default_value, "default value")
        spec = ParameterSpec(symbol=symbol, value=value, metadata=metadata)
        self._explicit_parameter_definitions.add(symbol)
        self._parameter_definitions[symbol] = spec
        previous_metadata = state.metadata
        state.metadata_signal.set(metadata)
        if _parameter_metadata_change_stops_animation(previous_metadata, metadata):
            self._animation_coordinator.notify_metadata_write(symbol)

    def _defaulted_parameter_spec(
        self,
        symbol: sympy.Basic,
        state: ParameterState,
    ) -> ParameterSpec:
        """Return a spec using the declared reset value and current metadata."""

        definition = self._parameter_definitions.get(symbol)
        value = state.value if definition is None else definition.value
        return ParameterSpec(symbol=symbol, value=value, metadata=state.metadata)

    def _parameter_entry(
        self,
        symbol: sympy.Basic,
    ) -> ParameterState | PendingParameterState:
        """Return an existing parameter state or a pending direct-write entry."""

        if not isinstance(symbol, (sympy.Symbol, sympy.Indexed)):
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )
        if symbol in self._parameters or symbol in self._parameter_definitions:
            return self._ensure_parameter_state(symbol)
        return PendingParameterState(self, symbol)

    def _ensure_parameter_state(self, symbol: sympy.Basic) -> ParameterState:
        """Return existing parameter state or instantiate a stored declaration."""

        if not isinstance(symbol, (sympy.Symbol, sympy.Indexed)):
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )
        state = self._parameters.get(symbol)
        if state is not None:
            return state
        spec = self._parameter_definitions.get(symbol)
        if spec is None:
            raise PlotSpecError("Parameter has not been assigned yet.")
        return self._parameter_state_for_spec(spec)

    def _set_parameter_field(
        self,
        symbol: sympy.Basic,
        field: str,
        raw_value: object,
    ) -> None:
        """Apply one direct parameter attribute update."""

        state = self._ensure_parameter_state(symbol)
        self._explicit_parameter_definitions.add(symbol)
        spec = state.to_spec()
        value = spec.value
        minimum = spec.metadata.minimum
        maximum = spec.metadata.maximum
        step = spec.metadata.step
        label = spec.metadata.label
        animated = spec.metadata.animated
        animation_mode = spec.metadata.animation_mode
        animation_rate_hz = spec.metadata.animation_rate_hz
        animation_speed = spec.metadata.animation_speed

        if field == "value":
            value = self._finite_parameter_float(raw_value, "value")
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        elif field == "min":
            minimum = self._finite_parameter_float(raw_value, "minimum")
            if minimum > maximum:
                raise PlotSpecError("Parameter slider minimum must not exceed maximum.")
            if value < minimum:
                value = minimum
        elif field == "max":
            maximum = self._finite_parameter_float(raw_value, "maximum")
            if maximum < minimum:
                raise PlotSpecError("Parameter slider maximum must not be below minimum.")
            if value > maximum:
                value = maximum
        elif field == "step":
            step = self._positive_parameter_float(raw_value, "step")
        elif field == "label":
            label = None if raw_value is None else str(raw_value)
        elif field == "animated":
            if not isinstance(raw_value, bool):
                raise PlotSpecError("Parameter animated must be True or False.")
            animated = raw_value
        elif field == "animation_mode":
            animation_mode = _parameter_animation_mode(raw_value)
        elif field == "animation_rate_hz":
            animation_rate_hz = _parameter_animation_rate(raw_value)
        elif field == "animation_speed":
            animation_speed = _parameter_animation_speed(raw_value)
        else:
            raise AttributeError(field)

        if minimum > maximum:
            raise PlotSpecError("Parameter slider minimum must not exceed maximum.")
        if value < minimum or value > maximum:
            raise PlotSpecError("Parameter value must lie within its slider range.")

        self._parameter_state_for_spec(
            ParameterSpec(
                symbol=symbol,
                value=value,
                metadata=ParameterMetadata(
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    label=label,
                    animated=animated,
                    animation_mode=animation_mode,
                    animation_rate_hz=animation_rate_hz,
                    animation_speed=animation_speed,
                ),
            )
        )

    def _set_parameter_range(self, symbol: sympy.Basic, raw_range: object) -> None:
        """Apply a simultaneous direct slider range update."""

        state = self._ensure_parameter_state(symbol)
        self._explicit_parameter_definitions.add(symbol)
        minimum, maximum, step = self._parameter_range_parts(
            raw_range,
            default_step=state.metadata.step,
        )
        if minimum > maximum:
            raise PlotSpecError("Parameter slider minimum must not exceed maximum.")

        value = state.value
        if value < minimum:
            value = minimum
        if value > maximum:
            value = maximum

        self._parameter_state_for_spec(
            ParameterSpec(
                symbol=symbol,
                value=value,
                metadata=ParameterMetadata(
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    label=state.metadata.label,
                    animated=state.metadata.animated,
                    animation_mode=state.metadata.animation_mode,
                    animation_rate_hz=state.metadata.animation_rate_hz,
                    animation_speed=state.metadata.animation_speed,
                ),
            )
        )

    @classmethod
    def _parameter_range_parts(
        cls,
        raw_range: object,
        *,
        default_step: float,
    ) -> tuple[float, float, float]:
        """Return validated range parts from a public range assignment."""

        if (
            isinstance(raw_range, str | bytes | sympy.Basic)
            or not isinstance(raw_range, Sequence)
        ):
            raise PlotSpecError("Parameter range must be a tuple (min, max) or (min, max, step).")
        if len(raw_range) not in {2, 3}:
            raise PlotSpecError("Parameter range must be a tuple (min, max) or (min, max, step).")

        minimum = cls._finite_parameter_float(raw_range[0], "minimum")
        maximum = cls._finite_parameter_float(raw_range[1], "maximum")
        step = (
            cls._positive_parameter_float(raw_range[2], "step")
            if len(raw_range) == 3
            else default_step
        )
        return minimum, maximum, step

    @staticmethod
    def _finite_parameter_float(value: object, label: str) -> float:
        """Return a finite float for direct parameter attribute updates."""

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PlotSpecError(f"Parameter {label} must be a finite number.") from exc
        if not math.isfinite(number):
            raise PlotSpecError(f"Parameter {label} must be a finite number.")
        return number

    @staticmethod
    def _positive_parameter_float(value: object, label: str) -> float:
        """Return a positive float for direct parameter attribute updates."""

        number = FigureHandle._finite_parameter_float(value, label)
        if number <= 0:
            raise PlotSpecError(f"Parameter {label} must be positive.")
        return number

    def _parameter_specs(self) -> dict[sympy.Basic, ParameterSpec]:
        """Return figure-level parameter spec snapshots."""

        return {
            symbol: self._defaulted_parameter_spec(symbol, state)
            for symbol, state in self._parameters.items()
        }

    def _default_parameter_value(self, symbol: sympy.Basic) -> float | None:
        """Return the declared reset value for an active parameter."""

        spec = self._parameter_definitions.get(symbol)
        if spec is None:
            return None
        return float(spec.value)

    def _prune_unused_parameters(self) -> None:
        """Drop figure-level states that no active consumer still references."""

        active_symbols = {
            symbol
            for node in self.plots
            for symbol in node.parameter_symbols_signal()
        }
        active_symbols.update(
            symbol
            for card in self.info_cards
            for symbol in card.parameter_symbols_signal()
        )
        for symbol in tuple(self._parameters):
            if symbol not in active_symbols:
                del self._parameters[symbol]
        for symbol in tuple(self._parameter_definitions):
            if symbol not in active_symbols and symbol not in self._explicit_parameter_definitions:
                del self._parameter_definitions[symbol]

    def _create_view(
        self,
        *,
        name: str | None,
        x_view: AxisView | None = None,
        y_view: AxisView | None = None,
        home_x_view: AxisView | None = None,
        home_y_view: AxisView | None = None,
    ) -> ViewHandle:
        """Create a view from explicit axes or the current/default axes."""

        if x_view is None or y_view is None:
            current = self._current_view_signal()
            if current is None:
                x_view, y_view = default_initial_2d_view()
            else:
                state = current._state()
                x_view, y_view = state.x_view, state.y_view
                home_x_view = state.home_x_view if home_x_view is None else home_x_view
                home_y_view = state.home_y_view if home_y_view is None else home_y_view
        assert x_view is not None
        assert y_view is not None

        view = ViewHandle(
            self,
            name=name,
            x_view=x_view,
            y_view=y_view,
            home_x_view=home_x_view,
            home_y_view=home_y_view,
        )
        self.views.append(view)
        if name is not None:
            self.views_by_name[name] = view
        return view

    def _resolve_view_target(
        self,
        target: object,
        *,
        only_existing: bool,
    ) -> ViewHandle:
        """Return a view for a current-view request."""

        if isinstance(target, ViewHandle):
            if target.figure is not self:
                raise PlotSpecError(
                    "fig.view.current received a view from another figure."
                )
            return target
        if target is OMITTED or target is None:
            if only_existing:
                return self._current_view()
            return self._create_view(name=None)
        if isinstance(target, str):
            view = self.views_by_name.get(target)
            if view is None:
                if only_existing:
                    raise ViewNotFoundError(
                        f"No view named {target!r} exists in this figure."
                    )
                view = self._create_view(name=target)
            return view
        raise PlotSpecError(
            "fig.view.current expects no argument, a name, or a ViewHandle."
        )

    def _resolve_existing_view(self, target: object, *, operation: str) -> ViewHandle:
        """Return an existing view for reset-style operations."""

        if isinstance(target, ViewHandle):
            if target.figure is not self:
                raise PlotSpecError(
                    f"{operation} received a view from another figure."
                )
            return target
        if target is OMITTED or target is None:
            return self._current_view()
        if isinstance(target, str):
            view = self.views_by_name.get(target)
            if view is None:
                raise ViewNotFoundError(
                    f"No view named {target!r} exists in this figure."
                )
            return view
        raise PlotSpecError(
            f"{operation} expects no argument, a name, or a ViewHandle."
        )

    def _publish_current_view(self, view: ViewHandle) -> None:
        """Publish a current view and apply its visible ranges to plot nodes."""

        state = view._state()
        with batch():
            self._current_view_signal.set(view)
            self._apply_view_state_to_plots(state)

    def _bind_plot_view_to_current_view(
        self,
        view: CurveView | CartesianView2D | ParametricView | ListView,
    ) -> CurveView | CartesianView2D | ParametricView | ListView:
        """Return a plot view aligned with the figure's current axes."""

        current = self._current_view_signal()
        if current is None:
            x_view, y_view = _figure_axis_views_for_plot_view(view)
            current = self._create_view(name=None, x_view=x_view, y_view=y_view)
            self._current_view_signal.set(current)
            return view

        return _plot_view_with_figure_axes(view, current._state())

    def _apply_view_state_to_plots(self, state: FigureViewState) -> None:
        """Patch every view-aware plot from one figure-level view state."""

        x_range = _axis_range(state.x_view)
        y_range = _axis_range(state.y_view)
        for node in tuple(self.plots):
            node.patch_view_range(x_range=x_range, y_range=y_range)

    def patch_current_view_range(
        self,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """Patch the current figure view from a frontend relayout event."""

        view = self._current_view()
        state = view._state()
        next_state = state
        if x_range is not None:
            next_state = replace(next_state, x_view=_axis_view_from_range(x_range))
        if y_range is not None:
            next_state = replace(next_state, y_view=_axis_view_from_range(y_range))
        with batch():
            view._set_state(next_state)
            self._apply_view_state_to_plots(next_state)


class PlotStyle:
    """Expose plot style keys as a live public command surface.

    Attribute or item assignment writes through the owning ``PlotHandle`` and
    therefore uses the same normalization and reactive updates as
    ``PlotHandle.set_style``. Snapshot methods such as ``to_dict`` return
    detached dictionaries for draft editing.

    Parameters
    ----------
    handle : PlotHandle
        Plot handle whose style should be read or updated.

    Methods
    -------
    to_dict
        Return a detached copy of the current normalized style dictionary.
    get
        Return one style value with an optional default.
    items
        Return style key-value pairs from the current snapshot.
    """

    def __init__(self, handle: PlotHandle) -> None:
        """Create a style facade for one plot handle."""

        object.__setattr__(self, "_handle", handle)

    def __repr__(self) -> str:
        """Return a compact representation of the current style snapshot."""

        return f"PlotStyle({self.to_dict()!r})"

    def __iter__(self):
        """Iterate over current style keys."""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of current style keys."""

        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        """Return whether the current style snapshot contains ``key``."""

        return key in self.to_dict()

    def __getitem__(self, key: str) -> object:
        """Return a style value from the current snapshot."""

        value = self.to_dict()[key]
        if isinstance(value, Mapping):
            return _NestedPlotStyle(self, key)
        return value

    def __setitem__(self, key: str, value: object) -> None:
        """Update one style key through the owning plot handle."""

        self._set_key(key, value)

    def __getattr__(self, key: str) -> object:
        """Return a style value as an attribute from the current snapshot."""

        style = self.to_dict()
        if key not in style:
            raise AttributeError(f"Plot style has no key {key!r}.")
        value = style[key]
        if isinstance(value, Mapping):
            return _NestedPlotStyle(self, key)
        return value

    def __setattr__(self, key: str, value: object) -> None:
        """Update one style key through the owning plot handle."""

        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self._set_key(key, value)

    def to_dict(self) -> dict[str, object]:
        """Return a detached copy of the current normalized style."""

        return self._handle._node.style

    def get(self, key: str, default: object = None) -> object:
        """Return one style value from the current snapshot."""

        return self.to_dict().get(key, default)

    def keys(self):
        """Return current style keys."""

        return self.to_dict().keys()

    def values(self):
        """Return current style values."""

        return self.to_dict().values()

    def items(self):
        """Return current style items."""

        return self.to_dict().items()

    def update(self, style: Mapping[str, object] | None = None, **kwargs: object) -> None:
        """Merge supported style keys through the owning plot handle."""

        update: dict[str, object] = {}
        if style is not None:
            if not isinstance(style, Mapping):
                raise PlotSpecError("Plot style must be a dictionary of supported keys.")
            update.update(style)
        update.update(kwargs)
        self._handle._node.set_style(update)

    def _set_key(self, key: str, value: object) -> None:
        """Apply one style-key update while preserving visibility semantics."""

        if key == "visible":
            self._handle.visible = value  # type: ignore[assignment]
            return
        self._handle._node.set_style({key: value})


class _NestedPlotStyle:
    """Expose one nested style dictionary as a live command surface."""

    def __init__(self, parent: PlotStyle, key: str) -> None:
        """Create a nested facade for one top-level style key."""

        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_key", key)

    def __repr__(self) -> str:
        """Return a compact representation of the nested style snapshot."""

        return f"NestedPlotStyle({self.to_dict()!r})"

    def __getitem__(self, key: str) -> object:
        """Return a nested style value from the current snapshot."""

        return self.to_dict()[key]

    def __setitem__(self, key: str, value: object) -> None:
        """Update one nested style key through the parent style facade."""

        self._set_key(key, value)

    def __getattr__(self, key: str) -> object:
        """Return a nested style value as an attribute."""

        style = self.to_dict()
        if key not in style:
            raise AttributeError(f"Nested plot style has no key {key!r}.")
        return style[key]

    def __setattr__(self, key: str, value: object) -> None:
        """Update one nested style key through the parent style facade."""

        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self._set_key(key, value)

    def to_dict(self) -> dict[str, object]:
        """Return a detached copy of the current nested style dictionary."""

        nested = self._parent.to_dict().get(self._key, {})
        return dict(nested) if isinstance(nested, Mapping) else {}

    def get(self, key: str, default: object = None) -> object:
        """Return one nested style value from the current snapshot."""

        return self.to_dict().get(key, default)

    def _set_key(self, key: str, value: object) -> None:
        """Apply one nested style-key update through the parent plot handle."""

        self._parent.update({self._key: {key: value}})


class PlotHandle:
    """Represent the lightweight public handle returned by plot commands.

    Attributes
    ----------
    figure : FigureHandle
        Figure that owns this plot.
    name : str or None
        Optional identity for in-place updates within the figure.
    label : str
        Display label used by legends and rendered traces.
    style : PlotStyle
        Live style command surface. Assign style attributes such as
        ``handle.style.opacity = 0.8`` to update supported style keys through
        the regular model command path.
    visible : bool
        Whether the plot is currently visible.
    params : dict
        Copy of active parameter values and slider metadata. Assigning a
        dictionary applies the same validation as ``set_params``.

    Methods
    -------
    show
        Display the parent figure and return the display generation.
    set_label
        Update the trace legend label without changing plot identity.
    set_style
        Merge supported style keys into the plotted trace.
    set_samples
        Update the sample count for continuous plot kinds.
    set_params
        Update parameter values or slider metadata for this plot.
    remove
        Remove this plot from its figure.
    """

    def __init__(self, figure: FigureHandle, node: PlotNode) -> None:
        """Create a handle for an existing plot node."""

        self.figure = figure
        self._node = node
        self._style = PlotStyle(self)

    @property
    def name(self) -> str | None:
        """Return this plot's optional identity name."""

        return self._node.name

    @property
    def label(self) -> str:
        """Return this plot's current display label."""

        return self._node.label

    @label.setter
    def label(self, label: object) -> None:
        """Update the display label through the signal-backed model node."""

        self._node.set_label(label)

    @property
    def style(self) -> PlotStyle:
        """Return this plot's live style command surface."""

        return self._style

    @style.setter
    def style(self, style: object) -> None:
        """Merge supported style keys through the regular model command path."""

        if not isinstance(style, Mapping):
            raise PlotSpecError("Plot style must be a dictionary of supported keys.")
        self._style.update(style)

    @property
    def visible(self) -> bool:
        """Return whether this plot is currently visible."""

        return self._node.legend_item_snapshot().visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        """Set this plot's visibility through normalized style state."""

        self._node.set_visible(visible)

    @property
    def kind(self) -> str:
        """Return this plot's internal kind name."""

        return self._node.kind

    @property
    def playable(self) -> bool:
        """Return whether this plot can be played through the sound controls."""

        return self._node.kind == PLOT_KIND_CURVE

    def set_label(self, label: object) -> PlotHandle:
        """Update the trace legend label without changing plot identity."""

        self.label = label
        return self

    def set_style(self, style: object | None = None, **kwargs: object) -> PlotHandle:
        """Merge supported style keys into the plotted trace."""

        update: dict[str, object] = {}
        if style is not None:
            if not isinstance(style, Mapping):
                raise PlotSpecError("Plot style must be a dictionary of supported keys.")
            update.update(style)
        update.update(kwargs)
        self.style = update
        return self

    def set_visible(self, visible: bool) -> PlotHandle:
        """Set this plot's visibility and return the handle."""

        self.visible = visible
        return self

    def set_samples(self, samples: object) -> PlotHandle:
        """Set this plot's sample count and return the handle."""

        self._node.set_samples(samples)
        return self

    def toggle_visible(self) -> PlotHandle:
        """Toggle this plot's visibility and return the handle."""

        self._node.toggle_visible()
        return self

    def show(self, **kwargs: object) -> FigureDisplayGeneration:
        """Display this plot's parent figure through ``FigureHandle.show``."""

        return self.figure.show(**kwargs)

    def remove(self) -> None:
        """Remove this plot from its parent figure."""

        self.figure.remove_plot(self._node)
        return None


class CurvePlotHandle(PlotHandle):
    """Represent a playable curve plot handle returned by ``plot(...)``.

    Attributes
    ----------
    sound : CurveSound
        Public playback namespace for this ordinary scalar curve.
    """

    def __init__(self, figure: FigureHandle, node: PlotNode) -> None:
        """Create a curve handle with its public sound facade."""

        super().__init__(figure, node)
        from .audio import CurveSound

        self.sound = CurveSound(self)


FigureHandle._mt_help = {
    "path": PurePosixPath("library/figure"),
    "anchor": None,
    "label": "figure",
}
PlotHandle._mt_help = {
    "path": PurePosixPath("library/plot"),
    "anchor": None,
    "label": "plot",
}


def _normalize_info_fragments(fragments: tuple[object, ...]) -> tuple[object, ...]:
    """Return validated authored info fragments."""

    if not fragments:
        raise PlotSpecError("fig.info(...) needs at least one fragment.")

    normalized: list[object] = []
    for fragment in fragments:
        if isinstance(fragment, str) or callable(fragment):
            normalized.append(fragment)
            continue
        expression = normalize_expression(fragment)
        if isinstance(expression, sympy.MatrixBase | sympy.Basic):
            normalized.append(expression)
            continue
        raise PlotSpecError(
            "fig.info(...) fragments must be Markdown strings, SymPy "
            "expressions, or callables."
        )
    return tuple(normalized)


def _symbolic_fragments_expression(fragments: tuple[object, ...]) -> tuple[object, ...]:
    """Return the symbolic fragments used for parameter discovery."""

    expressions: list[object] = []
    for fragment in fragments:
        if isinstance(fragment, sympy.MatrixBase):
            expressions.extend(tuple(fragment))
        elif isinstance(fragment, sympy.Basic):
            expressions.append(fragment)
    return tuple(expressions)


def _next_info_title(title: object, *, existing: InfoCard | None) -> str | None:
    """Return the next info title from update-style public options."""

    if title is OMITTED:
        return existing.title_signal() if existing is not None else None
    if title is None:
        return None
    return str(title)


def _is_numeric_scalar(value: object) -> bool:
    """Return whether a value can be displayed as a compact number."""

    try:
        array = np.asarray(value)
    except Exception:
        return False
    if array.shape != ():
        return False
    if np.iscomplexobj(array):
        return False
    try:
        number = float(array)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _format_info_number(value: float) -> str:
    """Return a compact stable number for info Markdown."""

    return f"{value:.12g}"


def _axis_range(view: AxisView) -> tuple[float, float]:
    """Return an axis view as the public two-float range tuple."""

    return view.minimum, view.maximum


def _animation_speed_snapshot(metadata: ParameterMetadata) -> str | float:
    """Return the public stored animation speed spelling."""

    if metadata.animation_speed is DEFAULT_ANIMATION_SPEED:
        return DEFAULT_ANIMATION_SPEED.value
    return float(metadata.animation_speed)


def _parameter_metadata_change_stops_animation(
    previous: ParameterMetadata,
    current: ParameterMetadata,
) -> bool:
    """Return whether a metadata change invalidates running animation state."""

    return (
        previous.step != current.step
        or previous.label != current.label
        or previous.animated != current.animated
        or previous.animation_mode != current.animation_mode
        or previous.animation_rate_hz != current.animation_rate_hz
        or previous.animation_speed != current.animation_speed
    )


def _view_range_snapshot(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Return visible or home ranges in public dictionary form."""

    return {"x": x_range, "y": y_range}


def _view_range_assignment(value: object) -> dict[str, object]:
    """Return private range-update keywords from public assignment syntax."""

    if not isinstance(value, Mapping) or not value:
        raise PlotSpecError(
            'View range assignments must be a non-empty dictionary like '
            '{"x": (xmin, xmax), "y": (ymin, ymax)}.'
        )

    updates: dict[str, object] = {}
    for axis, axis_range in value.items():
        axis_name = _view_range_axis_name(axis)
        key = f"{axis_name}_range"
        if key in updates:
            raise PlotSpecError(f"The {axis_name}-axis view range was provided twice.")
        updates[key] = axis_range
    return updates


def _view_range_axis_name(axis: object) -> str:
    """Return the public view axis name for a range assignment axis."""

    if axis in {"x", "y"}:
        return str(axis)
    raise PlotSpecError('View range assignments must use "x" or "y" keys.')


def _resolve_axis_view_update(
    base_x_view: AxisView,
    base_y_view: AxisView,
    *,
    x_range: object = OMITTED,
    y_range: object = OMITTED,
) -> tuple[AxisView, AxisView]:
    """Return next x and y views from explicit range updates."""

    next_x_view = (
        base_x_view
        if x_range is OMITTED
        else _axis_view_from_range(x_range, axis_name="x")
    )
    next_y_view = (
        base_y_view
        if y_range is OMITTED
        else _axis_view_from_range(y_range, axis_name="y")
    )
    return next_x_view, next_y_view


def _axis_view_from_range(value: object, *, axis_name: str = "axis") -> AxisView:
    """Return an ``AxisView`` from a public or frontend range."""

    # Accept any length-two sequence-like object so public commands and Plotly
    # callback ranges share the same finite range validation.
    try:
        minimum_value, maximum_value = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise PlotSpecError(
            f"The {axis_name}-axis view range must contain exactly two values."
        ) from exc

    try:
        minimum = float(minimum_value)
        maximum = float(maximum_value)
    except (TypeError, ValueError) as exc:
        raise PlotSpecError(
            f"The {axis_name}-axis view range must contain finite real values."
        ) from exc
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise PlotSpecError(
            f"The {axis_name}-axis view range must contain finite real values."
        )
    if minimum == maximum:
        raise PlotSpecError(
            f"The {axis_name}-axis view range endpoints must be distinct."
        )

    return AxisView(minimum=minimum, maximum=maximum)


def _figure_axis_views_for_plot_view(
    view: CurveView | CartesianView2D | ParametricView | ListView,
) -> tuple[AxisView, AxisView]:
    """Return figure-level axes implied by one plot view."""

    if isinstance(view, CurveView):
        _, default_y_view = default_initial_2d_view()
        return view.x_view, default_y_view
    if isinstance(view, CartesianView2D):
        return view.x_view, view.y_view
    if isinstance(view, ListView) and view.x_view is not None:
        _, default_y_view = default_initial_2d_view()
        return view.x_view, default_y_view
    if isinstance(view, ListView) and view.minimum is not None and view.maximum is not None:
        _, default_y_view = default_initial_2d_view()
        return AxisView(
            minimum=float(min(view.minimum, view.maximum)),
            maximum=float(max(view.minimum, view.maximum)),
        ), default_y_view
    return default_initial_2d_view()


def _plot_view_with_figure_axes(
    view: CurveView | CartesianView2D | ParametricView | ListView,
    state: FigureViewState,
) -> CurveView | CartesianView2D | ParametricView | ListView:
    """Return a plot view that samples from the current figure axes."""

    if isinstance(view, CurveView):
        return replace(view, x_view=state.x_view)
    if isinstance(view, CartesianView2D):
        return replace(view, x_view=state.x_view, y_view=state.y_view)
    if isinstance(view, ListView) and view.inferred:
        return replace(view, x_view=state.x_view)
    return view


def _quiet_signal(value: object, *, equal: Callable[[object, object], bool]) -> Signal:
    """Return a signal initialized without formatting a potentially huge value."""

    signal = Signal(None, equal=equal)
    _quiet_signal_set(signal, value)
    return signal


def _quiet_signal_set(signal: Signal, value: object) -> None:
    """Set a signal value without triggering eager debug string formatting."""

    signal._set_internal(value)


def _semantic_equal(old: object, new: object) -> bool:
    """Return whether two source values are semantically unchanged."""

    try:
        return bool(old == new)
    except Exception:
        return old is new


def _normalize_plot_name(name: object) -> str | None:
    """Return a validated optional plot identity name."""

    if name is None:
        return None
    if not isinstance(name, str) or name == "":
        raise PlotSpecError("Plot names must be non-empty strings.")
    return name


def _independent_symbols_for_view(
    view: CurveView | CartesianView2D | ParametricView | ListView,
) -> tuple[sympy.Symbol, ...]:
    """Return independent variables in numeric call order for a view."""

    if isinstance(view, CurveView):
        return (view.x_domain.symbol,)
    if isinstance(view, CartesianView2D):
        return (view.x_domain.symbol, view.y_domain.symbol)
    if isinstance(view, ListView):
        return () if view.index_symbol is None else (view.index_symbol,)
    return (view.parameter_symbol,)


def _trace_type_for_kind(kind: str) -> str:
    """Return the renderer trace type for a plot kind."""

    if kind == PLOT_KIND_TEMPERATURE:
        return "heatmap"
    if kind == PLOT_KIND_CONTOUR:
        return "contour"
    if kind == PLOT_KIND_LIST:
        return "list-scatter"
    return "scatter"


def _line_legend_marker(style: Mapping[str, object]) -> tuple[LegendMarker, bool]:
    """Return legend marker state for line-like plots."""

    visible = bool(style.get("visible", True))
    color = _style_text(style.get("color"))
    marker = LegendMarker(
        fill_color=color,
        border_color=color,
        border_width=_legend_border_width(style.get("width")),
        border_dash=_style_text(style.get("dash")) or "solid",
        opacity=1.0 if visible else 0.38,
    )
    return marker, visible


def _heatmap_legend_marker(style: Mapping[str, object]) -> tuple[LegendMarker, bool]:
    """Return legend marker state for heatmap plots."""

    visible = bool(style.get("visible", True))
    color = _color_from_colorscale(style.get("colorscale")) or "#8aa0b5"
    marker = LegendMarker(
        fill_color=color,
        border_color="#536878",
        border_width=1.0,
        border_dash="solid",
        opacity=1.0 if visible else 0.38,
    )
    return marker, visible


def _contour_legend_marker(style: Mapping[str, object]) -> tuple[LegendMarker, bool]:
    """Return legend marker state for contour plots."""

    visible = bool(style.get("visible", True))
    color = _style_text(style.get("contour_color")) or "#536878"
    marker = LegendMarker(
        fill_color=None,
        border_color=color,
        border_width=_legend_border_width(style.get("contour_width")),
        border_dash="solid",
        opacity=1.0 if visible else 0.38,
    )
    return marker, visible


def _domain_legend_marker(style: Mapping[str, object]) -> tuple[LegendMarker, bool]:
    """Return legend marker state for filled-domain plots."""

    domain_style = style.get("domain", {})
    boundary_style = style.get("boundary", {})
    domain = dict(domain_style) if isinstance(domain_style, Mapping) else {}
    boundary = dict(boundary_style) if isinstance(boundary_style, Mapping) else {}
    domain_visible = bool(domain.get("visible", True))
    boundary_visible = bool(boundary.get("visible", True))
    visible = domain_visible or boundary_visible
    fill_color = _style_text(domain.get("color")) if domain_visible else None
    border_color = _style_text(boundary.get("color")) if boundary_visible else None
    marker = LegendMarker(
        fill_color=fill_color,
        border_color=border_color,
        border_width=_legend_border_width(boundary.get("width")),
        border_dash=_style_text(boundary.get("dash")) or "solid",
        opacity=1.0 if visible else 0.38,
    )
    return marker, visible


def _domain_boundary_visible(style: Mapping[str, object]) -> bool:
    """Return whether a domain plot currently intends to show its boundary."""

    boundary_style = style.get("boundary", {})
    return isinstance(boundary_style, Mapping) and bool(
        boundary_style.get("visible", True)
    )


def _legend_border_width(value: object) -> float:
    """Return a compact marker border width from a plot stroke width."""

    try:
        width = float(value)
    except (TypeError, ValueError):
        return 1.5
    if not math.isfinite(width):
        return 1.5
    return min(4.0, max(1.0, width))


def _style_text(value: object) -> str | None:
    """Return a nonempty string style value or ``None``."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _next_indexed_entry(symbol: sympy.Indexed) -> sympy.Indexed:
    """Return the next one-step indexed entry for user guidance."""

    indices = list(symbol.indices)
    if not indices:
        return symbol
    try:
        first = int(indices[0])
    except TypeError:
        first = 0
    indices[0] = sympy.Integer(first + 1)
    return symbol.base[tuple(indices)]


def _append_markdown_to_layout_output(output_area: object, markdown: str) -> None:
    """Append Markdown to an ipywidgets-style figure output area."""

    append_display_data = getattr(output_area, "append_display_data", None)
    if callable(append_display_data):
        try:
            from IPython.display import Markdown

            append_display_data(Markdown(markdown))
            return
        except ImportError:
            pass
    append_stdout = getattr(output_area, "append_stdout", None)
    if callable(append_stdout):
        append_stdout(f"{markdown}\n")


def _color_from_colorscale(value: object) -> str | None:
    """Return a representative color from a simple Plotly colorscale value."""

    if isinstance(value, str):
        return None
    if isinstance(value, tuple | list) and value:
        middle = value[len(value) // 2]
        if isinstance(middle, tuple | list) and len(middle) >= 2:
            return _style_text(middle[1])
    return None


def _plot_style_colors(kind: str, style: Mapping[str, object]) -> dict[str, str]:
    """Return palette-relevant colors from one normalized plot style."""

    if kind in {PLOT_KIND_CURVE, PLOT_KIND_PARAMETRIC, PLOT_KIND_LIST}:
        color = _style_text(style.get("color"))
        return {"color": color} if color is not None else {}
    if kind == PLOT_KIND_TEMPERATURE:
        color = _color_from_colorscale(style.get("colorscale"))
        return {"colorscale": color} if color is not None else {}
    if kind == PLOT_KIND_CONTOUR:
        color = _style_text(style.get("contour_color"))
        return {"contour_color": color} if color is not None else {}
    if kind == PLOT_KIND_DOMAIN:
        colors: dict[str, str] = {}
        domain_style = style.get("domain", {})
        boundary_style = style.get("boundary", {})
        if isinstance(domain_style, Mapping):
            domain_color = _style_text(domain_style.get("color"))
            if domain_color is not None:
                colors["domain"] = domain_color
        if isinstance(boundary_style, Mapping):
            boundary_color = _style_text(boundary_style.get("color"))
            if boundary_color is not None:
                colors["boundary"] = boundary_color
        return colors
    return {}


def _next_label(
    expression: object,
    *,
    plot_name: str | None,
    label: object,
    existing: PlotNode | None,
) -> str:
    """Return the next user-visible trace label."""

    if label is not OMITTED:
        text = "" if label is None else str(label)
        if text:
            return text
        return _default_plot_label(expression)
    if existing is not None:
        return existing.label
    return _default_plot_label(expression)


def _default_plot_label(expression: object) -> str:
    """Return the Markdown label derived from a plot expression."""

    if isinstance(expression, tuple) and expression and isinstance(
        expression[0],
        DomainConditionSpec,
    ):
        if len(expression) == 1:
            return f"${sympy.latex(expression[0].expression)}$"
        pieces = ", ".join(sympy.latex(condition.expression) for condition in expression)
        return f"${pieces}$"
    try:
        return f"${sympy.latex(expression)}$"
    except Exception:
        return str(expression)


def _parameter_label_markdown(symbol: sympy.Symbol, label: object) -> str:
    """Return the markdown label for a parameter control."""

    if label is not None:
        return str(label)
    return f"${sympy.latex(symbol)}$"


def _next_style(
    kind: str,
    style: object,
    *,
    existing: PlotNode | None,
    boundary: bool,
    default_color: str,
) -> dict[str, object]:
    """Return normalized style for a new or updated plot."""

    existing_style = existing.style if existing is not None else None
    if kind == PLOT_KIND_CURVE:
        normalized = normalize_style(style, existing=existing_style)
    elif kind in {PLOT_KIND_PARAMETRIC, PLOT_KIND_LIST}:
        normalized = normalize_line_style(
            style,
            existing=existing_style,
            plotter="list_plot" if kind == PLOT_KIND_LIST else "parametric_plot",
        )
    elif kind == PLOT_KIND_TEMPERATURE:
        normalized = normalize_field_style(
            style,
            existing=existing_style,
            plotter="temperature_plot",
        )
    elif kind == PLOT_KIND_CONTOUR:
        normalized = normalize_field_style(
            style,
            existing=existing_style,
            plotter="contour_plot",
        )
    else:
        normalized = normalize_domain_style(
            style,
            existing=existing_style,
            boundary=boundary,
        )

    if normalized is not None:
        return _style_with_default_color(
            kind,
            normalized,
            style,
            existing=existing,
            default_color=default_color,
        )
    if existing is not None:
        return existing.style
    return _style_with_default_color(
        kind,
        {},
        style,
        existing=existing,
        default_color=default_color,
    )


def _style_with_default_color(
    kind: str,
    normalized: Mapping[str, object],
    raw_style: object,
    *,
    existing: PlotNode | None,
    default_color: str,
) -> dict[str, object]:
    """Return style with figure-owned default colors for new plot slots."""

    style = dict(normalized)
    if existing is not None:
        return style

    # Apply defaults only where the caller did not declare a color-bearing key.
    # The normalizers validate user input first; this helper fills model-owned
    # defaults after that validation so Plotly never supplies implicit colors.
    if kind in {PLOT_KIND_CURVE, PLOT_KIND_PARAMETRIC, PLOT_KIND_LIST}:
        if not _style_declares_key(raw_style, "color"):
            style["color"] = default_color
    elif kind == PLOT_KIND_TEMPERATURE:
        if not _style_declares_key(raw_style, "colorscale"):
            style["colorscale"] = _single_color_scale(default_color)
    elif kind == PLOT_KIND_CONTOUR:
        if not _style_declares_key(raw_style, "contour_color"):
            style["contour_color"] = default_color
    elif kind == PLOT_KIND_DOMAIN:
        domain_style = style.get("domain", {})
        boundary_style = style.get("boundary", {})
        domain = dict(domain_style) if isinstance(domain_style, Mapping) else {}
        boundary = dict(boundary_style) if isinstance(boundary_style, Mapping) else {}
        if not _domain_style_declares_color(raw_style, "domain"):
            domain["color"] = default_color
        if not _domain_style_declares_color(raw_style, "boundary"):
            boundary["color"] = default_color
        style["domain"] = domain
        style["boundary"] = boundary
    return style


def _style_declares_key(raw_style: object, key: str) -> bool:
    """Return whether a public style update explicitly declares a key."""

    return isinstance(raw_style, Mapping) and key in raw_style


def _domain_style_declares_color(raw_style: object, part: str) -> bool:
    """Return whether a domain style explicitly declares a color for a part."""

    if not isinstance(raw_style, Mapping):
        return False
    if "color" in raw_style:
        return True
    nested = raw_style.get(part)
    return isinstance(nested, Mapping) and "color" in nested


def _single_color_scale(color: str) -> list[list[object]]:
    """Return a simple source-owned heatmap colorscale for one plot color."""

    return [[0.0, "#ffffff"], [1.0, color]]


def _validate_plot_candidate(
    kind: str,
    expression: object,
    view: CurveView | CartesianView2D | ParametricView | ListView,
    parameter_specs: Mapping[sympy.Basic, ParameterSpec],
) -> None:
    """Run one synchronous sample so public plotting errors propagate."""

    symbols = sort_symbols(parameter_specs)
    signature = SampleSignature(
        expression=expression,
        view=view,
        parameter_symbols=symbols,
        parameter_values=tuple(parameter_specs[symbol].value for symbol in symbols),
    )
    independent = _independent_symbols_for_view(view)

    if kind == PLOT_KIND_CURVE:
        numeric = compile_numeric_curve(expression, independent[0], symbols)
        sample_curve(numeric, signature)
    elif kind == PLOT_KIND_TEMPERATURE:
        numeric = compile_numeric_field(
            expression,
            independent[0],
            independent[1],
            symbols,
            plotter="temperature_plot",
        )
        sample_scalar_field(numeric, signature)
    elif kind == PLOT_KIND_CONTOUR:
        numeric = compile_numeric_field(
            expression,
            independent[0],
            independent[1],
            symbols,
            plotter="contour_plot",
        )
        sample_scalar_field(numeric, signature)
    elif kind == PLOT_KIND_DOMAIN:
        numeric = compile_numeric_domain(expression, independent[0], independent[1], symbols)
        sample_domain(numeric, signature)
    elif kind == PLOT_KIND_LIST:
        numeric = compile_numeric_list(
            expression,
            independent[0] if independent else None,
            symbols,
        )
        sample_list_plot(numeric, signature)
    else:
        numeric = compile_numeric_parametric(expression, independent[0], symbols)
        sample_parametric(numeric, signature)
