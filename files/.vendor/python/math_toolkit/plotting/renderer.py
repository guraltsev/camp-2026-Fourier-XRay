"""Render sampled plot snapshots into a generation-owned Plotly widget."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
import math

import numpy as np

from .errors import PlotSpecError
from .specs import TRACE_ROLE_DOMAIN_BOUNDARY, TRACE_ROLE_DOMAIN_FILL

_COORDINATE_ONLY_HOVER_TEMPLATE = "x=%{x}<br>y=%{y}<extra></extra>"

if TYPE_CHECKING:
    from .model import (
        FigureHandle,
        FigureViewState,
        PlotNode,
        TraceDataSnapshot,
        TraceStyleSnapshot,
    )


class PlotlyRenderer:
    """Manage the Plotly objects for one display generation."""

    def __init__(
        self,
        figure: FigureHandle,
        *,
        generation_id: int,
        accepts_frontend_events: Callable[[], bool],
    ) -> None:
        """Create an empty FigureWidget for one figure display generation."""

        self.figure = figure
        self.generation_id = generation_id
        self._accepts_frontend_events = accepts_frontend_events
        self.trace_indices: dict[tuple[int, str], int] = {}
        self._layout_callback_guard = False
        self.home_sync_widget = _create_plotly_home_sync_widget()
        self.resize_sync_widget = _create_plotly_resize_sync_widget()

        # Keep Plotly and widget imports local so importing ``math_toolkit`` does
        # not touch notebook display state or construct frontend models.
        import plotly.graph_objects as go

        self.figure_widget = go.FigureWidget()
        self.figure_widget._config = {
            **self.figure_widget._config,
            "responsive": True,
            "displayModeBar": "hover",
        }
        self.figure_widget.layout.autosize = True
        self.figure_widget.layout.height = None
        self.figure_widget.layout.width = None
        self.figure_widget.layout.margin = {"l": 0, "r": 0, "t": 0, "b": 0}
        self.figure_widget.layout.showlegend = False
        self.figure_widget.layout.uirevision = (
            f"math-toolkit-figure-{figure.id}-generation-{generation_id}"
        )

        self.figure_widget.layout.xaxis.on_change(
            self._on_xaxis_range_change,
            "range",
        )
        self.figure_widget.layout.yaxis.on_change(
            self._on_yaxis_range_change,
            "range",
        )

    def render_trace_data(self, snapshot: TraceDataSnapshot) -> None:
        """Create or update one Plotly trace's sampled data."""

        with self.figure_widget.batch_update():
            created = self._render_trace_data_in_batch(snapshot)
        if created:
            self._sync_full_widget_data()

    def render_trace_data_batch(self, snapshots: tuple[TraceDataSnapshot, ...]) -> None:
        """Create or update several Plotly traces in one widget batch."""

        if not snapshots:
            return
        created_any = False
        with self.figure_widget.batch_update():
            for snapshot in snapshots:
                created_any = self._render_trace_data_in_batch(snapshot) or created_any
        if created_any:
            self._sync_full_widget_data()

    def _render_trace_data_in_batch(self, snapshot: TraceDataSnapshot) -> bool:
        """Apply one trace data snapshot inside an existing Plotly batch."""

        trace, created = self._trace_for_snapshot(snapshot)

        # Plotly copies the active views into the widget model. The Python-side
        # buffers remain owned by the node and can be reused on the next sample.
        trace.x = _json_safe_plotly_data(snapshot.x)
        trace.y = _json_safe_plotly_data(snapshot.y)
        if snapshot.z is not None:
            trace.z = _json_safe_plotly_data(snapshot.z)
        if snapshot.contour_level is not None:
            trace.contours.start = snapshot.contour_level
            trace.contours.end = snapshot.contour_level
            trace.contours.size = 1
        return created

    def render_trace_style(self, snapshot: TraceStyleSnapshot) -> None:
        """Update one Plotly trace's display style without touching sampled data."""

        trace, created = self._trace_for_style(snapshot)
        with self.figure_widget.batch_update():
            trace.name = snapshot.label
            self._apply_style(trace, snapshot.trace_type, dict(snapshot.style))
        if created:
            self._sync_full_widget_data()

    def render_view(self, snapshot: FigureViewState) -> None:
        """Apply the Python-owned visible ranges to the Plotly layout."""

        relayout = {
            "xaxis.autorange": False,
            "yaxis.autorange": False,
            "xaxis.range[0]": snapshot.x_view.minimum,
            "xaxis.range[1]": snapshot.x_view.maximum,
            "yaxis.range[0]": snapshot.y_view.minimum,
            "yaxis.range[1]": snapshot.y_view.maximum,
        }
        self._layout_callback_guard = True
        try:
            perform_relayout = getattr(
                self.figure_widget,
                "_perform_plotly_relayout",
                None,
            )
            send_relayout = getattr(self.figure_widget, "_send_relayout_msg", None)
            if callable(perform_relayout) and callable(send_relayout):
                perform_relayout(relayout)
                send_relayout(relayout)
            else:
                self.figure_widget.plotly_relayout(relayout)
            if self.home_sync_widget is not None:
                self.home_sync_widget.sync_home_ranges(snapshot)
        finally:
            self._layout_callback_guard = False

    def remove_node(self, node: PlotNode) -> None:
        """Remove all traces owned by a node and repair remaining indices."""

        removed_keys = {
            key for key in self.trace_indices if key[0] == node.id
        }
        if not removed_keys:
            return

        removed_indices = {
            self.trace_indices[key]
            for key in removed_keys
        }
        remaining_traces = tuple(
            trace
            for trace_index, trace in enumerate(self.figure_widget.data)
            if trace_index not in removed_indices
        )
        self.figure_widget.data = remaining_traces

        old_items = [
            (key, index)
            for key, index in self.trace_indices.items()
            if key not in removed_keys
        ]
        self.trace_indices = {
            key: index - sum(removed < index for removed in removed_indices)
            for key, index in old_items
        }

    def dispose_callbacks(self) -> None:
        """Release renderer-owned callbacks where Plotly exposes them."""

        for axis, callback in (
            (self.figure_widget.layout.xaxis, self._on_xaxis_range_change),
            (self.figure_widget.layout.yaxis, self._on_yaxis_range_change),
        ):
            callbacks = getattr(axis, "_change_callbacks", None)
            if isinstance(callbacks, dict):
                for registered_callbacks in callbacks.values():
                    while callback in registered_callbacks:
                        registered_callbacks.remove(callback)

    def disable_interactivity(self) -> None:
        """Freeze Plotly-side interactions for a retired display generation."""

        with self.figure_widget.batch_update():
            self.figure_widget.layout.dragmode = False
            self.figure_widget.layout.hovermode = False
            self.figure_widget.layout.showlegend = False
            self.figure_widget.layout.xaxis.fixedrange = True
            self.figure_widget.layout.yaxis.fixedrange = True

    def disable_resize_sync(self) -> None:
        """Stop frontend resize bookkeeping for a retired display generation."""

        if self.resize_sync_widget is not None:
            self.resize_sync_widget.close()
            self.resize_sync_widget = None

    def dispose(self) -> None:
        """Release renderer-owned callbacks and trace state."""

        self.dispose_callbacks()
        self.trace_indices.clear()
        self.figure_widget.data = ()
        if self.home_sync_widget is not None:
            self.home_sync_widget.close()
        if self.resize_sync_widget is not None:
            self.resize_sync_widget.close()

    def _trace_for_snapshot(self, snapshot: TraceDataSnapshot) -> tuple[object, bool]:
        """Return the trace for a data snapshot, creating it if needed."""

        return self._trace_for_key(
            (snapshot.node_id, snapshot.trace_role),
            snapshot.trace_type,
        )

    def _trace_for_style(self, snapshot: TraceStyleSnapshot) -> tuple[object, bool]:
        """Return the trace for a style snapshot, creating it if needed."""

        return self._trace_for_key(
            (snapshot.node_id, snapshot.trace_role),
            snapshot.trace_type,
        )

    def _trace_for_key(self, key: tuple[int, str], trace_type: str) -> tuple[object, bool]:
        """Return the existing trace for ``key`` or create a compatible one."""

        if key in self.trace_indices:
            return self.figure_widget.data[self.trace_indices[key]], False

        import plotly.graph_objects as go

        if trace_type == "heatmap" or trace_type == "domain-fill":
            trace = go.Heatmap()
            trace.zsmooth = "best"
            if trace_type == "domain-fill":
                trace.showscale = False
        elif trace_type == "contour" or trace_type == "domain-boundary":
            trace = go.Contour()
            trace.line.smoothing = 1.0
            if trace_type == "contour":
                trace.contours.coloring = "lines"
                trace.showscale = False
            elif trace_type == "domain-boundary":
                trace.contours.coloring = "none"
                trace.showscale = False
        elif trace_type == "list-scatter":
            trace = go.Scatter(mode="markers")
        else:
            trace = go.Scatter(mode="lines")

        trace.showlegend = False
        trace.hovertemplate = _hover_template_for_trace_type(trace_type)
        self.figure_widget.add_trace(trace)
        index = len(self.figure_widget.data) - 1
        self.trace_indices[key] = index
        return self.figure_widget.data[index], True

    def _sync_full_widget_data(self) -> None:
        """Publish full trace state for hosts that miss transient add-trace events."""

        widget_data = getattr(self.figure_widget, "_widget_data", None)
        data = getattr(self.figure_widget, "_data", None)
        if widget_data is None or data is None:
            return
        self.figure_widget._widget_data = list(data)

    def _apply_style(
        self,
        trace: object,
        trace_type: str,
        style: dict[str, object],
    ) -> None:
        """Apply one normalized public style dictionary to a Plotly trace."""

        if trace_type == "scatter":
            _apply_line_style(trace, style)
        elif trace_type == "list-scatter":
            trace.mode = "markers"
            _apply_line_style(trace, style)
        elif trace_type == "heatmap":
            _apply_field_style(trace, style)
        elif trace_type == "contour":
            _apply_field_style(trace, style)
            if "contour_color" in style:
                trace.line.color = style["contour_color"]
            if "contour_width" in style:
                trace.line.width = style["contour_width"]
            if "line_smoothing" in style:
                trace.line.smoothing = style["line_smoothing"]
        elif trace_type == "domain-fill":
            _apply_domain_fill_style(trace, style)
        elif trace_type == "domain-boundary":
            _apply_domain_boundary_style(trace, style)

    def _on_xaxis_range_change(self, layout: object, value: object) -> None:
        """Patch every non-parametric plot when the x-axis view changes."""

        x_range = _coerce_axis_range(value, axis_name="x")
        if x_range is None:
            return
        self._patch_model_view(
            x_range=x_range,
            y_range=_coerce_axis_range(
                self.figure_widget.layout.yaxis.range,
                axis_name="y",
            ),
        )

    def _on_yaxis_range_change(self, layout: object, value: object) -> None:
        """Patch every two-dimensional plot when the y-axis view changes."""

        y_range = _coerce_axis_range(value, axis_name="y")
        if y_range is None:
            return
        self._patch_model_view(
            x_range=_coerce_axis_range(
                self.figure_widget.layout.xaxis.range,
                axis_name="x",
            ),
            y_range=y_range,
        )

    def _patch_model_view(
        self,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ) -> None:
        """Patch live plot model views from a frontend relayout callback."""

        if self._layout_callback_guard or not self._accepts_frontend_events():
            return

        self.figure.patch_current_view_range(x_range=x_range, y_range=y_range)


def _apply_line_style(trace: object, style: dict[str, object]) -> None:
    """Apply curve-like line styles to a Scatter trace."""

    if "color" in style:
        trace.line.color = style["color"]
    if "width" in style:
        trace.line.width = style["width"]
    if "dash" in style:
        trace.line.dash = style["dash"]
    if "opacity" in style:
        trace.opacity = style["opacity"]
    if "visible" in style:
        trace.visible = style["visible"]


def _apply_field_style(trace: object, style: dict[str, object]) -> None:
    """Apply scalar-field styles to Heatmap or Contour traces."""

    for key in (
        "colorscale",
        "opacity",
        "visible",
        "showscale",
        "zmin",
        "zmax",
        "zsmooth",
    ):
        if key in style:
            setattr(trace, key, style[key])


def _apply_domain_fill_style(trace: object, style: dict[str, object]) -> None:
    """Apply filled-domain styles to a Heatmap trace."""

    color = style.get("color", "royalblue")
    trace.colorscale = [[0, color], [1, color]]
    trace.showscale = False
    if "zsmooth" in style:
        trace.zsmooth = style["zsmooth"]
    if "opacity" in style:
        trace.opacity = style["opacity"]
    if "visible" in style:
        trace.visible = style["visible"]


def _apply_domain_boundary_style(trace: object, style: dict[str, object]) -> None:
    """Apply boundary styles to a Contour trace."""

    trace.contours.coloring = "none"
    trace.showscale = False
    if "color" in style:
        trace.line.color = style["color"]
    if "width" in style:
        trace.line.width = style["width"]
    if "dash" in style and hasattr(trace.line, "dash"):
        trace.line.dash = style["dash"]
    if "smoothing" in style:
        trace.line.smoothing = style["smoothing"]
    if "visible" in style:
        trace.visible = style["visible"]


def _hover_template_for_trace_type(trace_type: str) -> str:
    """Return the standard hover template for one supported Plotly trace type."""

    del trace_type
    return _COORDINATE_ONLY_HOVER_TEMPLATE


def _json_safe_plotly_data(values: object) -> object:
    """Return trace data with non-finite numeric values replaced by ``None``."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        return values

    finite = np.isfinite(array)
    if bool(finite.all()):
        return values

    safe = array.astype(object)
    safe[~finite] = None
    if safe.ndim == 0:
        return safe.item()
    return safe.tolist()


def _create_plotly_home_sync_widget() -> object | None:
    """Return a hidden anywidget that mirrors Python home ranges into Plotly."""

    try:
        import anywidget
    except ImportError:
        return None

    # Plotly exposes visible axis ranges through FigureWidget, but the modebar
    # reset target is cached in frontend-only full-layout fields. The hidden
    # widget keeps that cache aligned with the Python-owned home range; it does
    # not own or infer plotting state itself.
    class PlotlyHomeSyncWidget(anywidget.AnyWidget):
        """Mirror Python-owned home ranges into the frontend Plotly modebar."""

        _esm = r"""
function render({ model, el }) {
  el.style.display = "none";

  let pending = null;
  let timer = null;
  let clickHandler = null;
  let relayoutHandler = null;

  function graphDiv() {
    const root = el.closest(".mt-plot") || el.closest(".mt-plot-shell") || document;
    return root.querySelector(".js-plotly-plot");
  }

  function setInitialRange(axis, range) {
    if (!axis || !Array.isArray(range) || range.length !== 2) {
      return;
    }
    // Plotly's reset-axes modebar button reads these cached initial values.
    // The visible layout range can stay panned while Python changes home.
    axis._rangeInitial0 = range[0];
    axis._rangeInitial1 = range[1];
    axis._autorangeInitial = false;
  }

  function homeRelayout() {
    if (!pending) {
      return null;
    }
    return {
      "xaxis.autorange": false,
      "yaxis.autorange": false,
      "xaxis.range": pending.home_x_range,
      "yaxis.range": pending.home_y_range,
    };
  }

  function applyHomeRange() {
    const gd = graphDiv();
    const update = homeRelayout();
    if (!gd || !update || !window.Plotly || typeof window.Plotly.relayout !== "function") {
      return;
    }
    window.Plotly.relayout(gd, update);
  }

  function isResetAxesButton(target) {
    const button = target && target.closest ? target.closest("[data-title], [aria-label], a, button") : null;
    if (!button) {
      return false;
    }
    const label = [
      button.getAttribute("data-title"),
      button.getAttribute("aria-label"),
      button.getAttribute("title"),
      button.textContent,
    ].filter(Boolean).join(" ").toLowerCase();
    return label.includes("reset axes");
  }

  function installResetHook(gd) {
    if (!gd || clickHandler !== null) {
      return;
    }

    clickHandler = (event) => {
      if (!isResetAxesButton(event.target)) {
        return;
      }
      window.setTimeout(applyHomeRange, 0);
      window.setTimeout(applyHomeRange, 50);
    };
    gd.addEventListener("click", clickHandler, true);

    if (typeof gd.on === "function") {
      relayoutHandler = (eventData) => {
        if (
          eventData
          && (eventData["xaxis.autorange"] || eventData["yaxis.autorange"])
        ) {
          window.setTimeout(applyHomeRange, 0);
        }
      };
      gd.on("plotly_relayout", relayoutHandler);
    }
  }

  function apply(attempt = 0) {
    if (!pending) {
      return;
    }

    const gd = graphDiv();
    if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.yaxis) {
      if (attempt < 40) {
        timer = window.setTimeout(() => apply(attempt + 1), 50);
      }
      return;
    }

    setInitialRange(gd._fullLayout.xaxis, pending.home_x_range);
    setInitialRange(gd._fullLayout.yaxis, pending.home_y_range);
    installResetHook(gd);
  }

  function schedule(message) {
    pending = message;
    if (timer !== null) {
      window.clearTimeout(timer);
      timer = null;
    }
    window.requestAnimationFrame(() => apply());
  }

  model.on("msg:custom", (message) => {
    if (message && message.type === "sync_home_ranges") {
      schedule(message);
    }
  });

  return () => {
    if (timer !== null) {
      window.clearTimeout(timer);
      timer = null;
    }
    const gd = graphDiv();
    if (gd && clickHandler !== null) {
      gd.removeEventListener("click", clickHandler, true);
    }
    if (gd && relayoutHandler !== null && typeof gd.removeListener === "function") {
      gd.removeListener("plotly_relayout", relayoutHandler);
    }
    clickHandler = null;
    relayoutHandler = null;
  };
}

export default { render };
"""

        def __init__(self) -> None:
            """Create a hidden frontend bridge for one Plotly figure."""

            super().__init__()
            self.last_command: dict[str, object] | None = None

        def sync_home_ranges(self, snapshot: FigureViewState) -> None:
            """Send current Python home ranges to the frontend reset cache."""

            command = {
                "type": "sync_home_ranges",
                "home_x_range": [
                    snapshot.home_x_view.minimum,
                    snapshot.home_x_view.maximum,
                ],
                "home_y_range": [
                    snapshot.home_y_view.minimum,
                    snapshot.home_y_view.maximum,
                ],
            }
            self.last_command = command
            self.send(command)

    return PlotlyHomeSyncWidget()


def _create_plotly_resize_sync_widget() -> object | None:
    """Return a hidden anywidget that resizes Plotly after container changes."""

    try:
        import anywidget
    except ImportError:
        return None

    class PlotlyResizeSyncWidget(anywidget.AnyWidget):
        """Ask Plotly to remeasure its flex container after notebook display."""

        _esm = r"""
function render({ el }) {
  el.style.display = "none";

  let observer = null;
  let timers = [];
  let animationFrame = null;
  let disposed = false;

  function root() {
    return el.closest(".mt-plot") || el.closest(".mt-plot-shell");
  }

  function graphDiv() {
    const plotRoot = root();
    return plotRoot ? plotRoot.querySelector(".js-plotly-plot") : null;
  }

  function plotViewport() {
    const plotRoot = root();
    return plotRoot
      ? plotRoot.querySelector(".mt-plot__plot, .mt-plot-shell__plot")
      : null;
  }

  function syncResponsiveLayout() {
    const plotRoot = root();
    if (!plotRoot) {
      return;
    }

    const width = Math.floor(plotRoot.getBoundingClientRect().width);
    if (width <= 0) {
      return;
    }
    plotRoot.classList.toggle("mt-plot--compact", width < 760);
  }

  function viewportSize() {
    const viewport = plotViewport();
    if (!viewport) {
      return null;
    }

    const rect = viewport.getBoundingClientRect();
    const width = Math.floor(rect.width);
    if (width <= 0) {
      return null;
    }

    // The viewport's measured height may include a previous Plotly relayout
    // height after a notebook reload or kernel restart. Bound the next height
    // from the current width so stale widget DOM cannot feed an ever-taller
    // size back into Plotly.
    const preferredHeight = Math.round(Math.min(Math.max(width * 0.62, 384), 720));
    const measuredHeight = Math.floor(rect.height);
    const height = measuredHeight > 0
      ? Math.min(measuredHeight, preferredHeight)
      : preferredHeight;
    return { viewport, width, height };
  }

  function reveal(viewport) {
    if (viewport) {
      viewport.style.visibility = "visible";
    }
  }

  function resize() {
    if (disposed) {
      return false;
    }

    syncResponsiveLayout();
    const gd = graphDiv();
    const size = viewportSize();
    if (!gd || !gd._fullLayout || !size || !window.Plotly) {
      return false;
    }

    const update = { width: size.width, height: size.height };
    const resized = window.Plotly.relayout
      ? window.Plotly.relayout(gd, update)
      : Promise.resolve();
    resized.then(
      () => {
        if (window.Plotly.Plots) {
          window.Plotly.Plots.resize(gd);
        }
        reveal(size.viewport);
      },
      () => reveal(size.viewport),
    );
    return true;
  }

  function schedule() {
    if (animationFrame !== null) {
      return;
    }
    animationFrame = window.requestAnimationFrame(() => {
      animationFrame = null;
      resize();
    });
  }

  function retry(attempt = 0) {
    if (disposed) {
      return;
    }

    const gd = graphDiv();
    const size = viewportSize();
    if (!gd || !gd._fullLayout || !size) {
      if (attempt < 40) {
        timers.push(window.setTimeout(() => retry(attempt + 1), 50));
      } else {
        reveal(plotViewport());
      }
      return;
    }

    const targets = [
      size.viewport,
      gd,
      gd.parentElement,
      gd.closest(".mt-plot__plot-column, .mt-plot-shell__main"),
      gd.closest(".mt-plot__overlay, .mt-plot-shell"),
      gd.closest(".mt-plot, .mt-plot-shell"),
    ].filter(Boolean);

    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(schedule);
      for (const target of new Set(targets)) {
        observer.observe(target);
      }
    }

    // Plotly can finish its first draw before the notebook output and flex
    // columns have settled. Several spaced calls cover that mount sequence
    // without depending on one frontend's timing.
    schedule();
    for (const delay of [0, 50, 150, 350, 750]) {
      timers.push(window.setTimeout(schedule, delay));
    }
    timers.push(window.setTimeout(() => reveal(size.viewport), 1500));
  }

  retry();

  return () => {
    disposed = true;
    if (observer !== null) {
      observer.disconnect();
    }
    if (animationFrame !== null) {
      window.cancelAnimationFrame(animationFrame);
    }
    for (const timer of timers) {
      window.clearTimeout(timer);
    }
    timers = [];
  };
}

export default { render };
"""

    return PlotlyResizeSyncWidget()


def _coerce_axis_range(value: object, *, axis_name: str) -> tuple[float, float] | None:
    """Return a finite axis range from Plotly relayout data."""

    if value is None or len(value) != 2:  # type: ignore[arg-type]
        return None

    try:
        minimum = float(value[0])  # type: ignore[index]
        maximum = float(value[1])  # type: ignore[index]
    except (TypeError, ValueError) as exc:
        raise PlotSpecError(
            f"Plotly {axis_name}-axis ranges must be finite real values."
        ) from exc
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise PlotSpecError(
            f"Plotly {axis_name}-axis ranges must be finite real values."
        )
    return minimum, maximum


def _display_ipython(*items: object) -> bool:
    """Display ``items`` only when an active IPython shell is available."""

    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return False

    if get_ipython() is None:
        return False
    for item in items:
        display(item)
    return True
