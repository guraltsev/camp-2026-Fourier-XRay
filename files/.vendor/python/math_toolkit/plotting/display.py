"""Own per-output widget generations for notebook plotting figures."""

from __future__ import annotations

import time
from collections.abc import Hashable
from typing import TYPE_CHECKING

from ._reactive import Effect
from .errors import PlotSpecError
from .layout import DefaultFigureLayout, FigureLayoutParts, LayoutValidationError
from . import renderer as renderer_module
from .renderer import PlotlyRenderer

if TYPE_CHECKING:
    from .model import (
        FigureHandle,
        InfoCardSnapshot,
        LegendItem,
        ParameterAnimationStateItem,
        PlotNode,
        SliderValueItem,
        TraceDataSnapshot,
    )


class WidgetLayoutShell:
    """Compose Plotly, control, and status widgets for one visible output."""

    def __init__(
        self,
        figure: FigureHandle,
        generation: FigureDisplayGeneration,
        plot_widget: object,
        *,
        generation_id: int,
    ) -> None:
        """Create a concrete shell around one Plotly widget generation."""

        import ipywidgets as ipywidgets

        self.generation_id = generation_id
        self.figure = figure
        self.generation = generation
        self.plot_widget = plot_widget
        self.runtime_area = ipywidgets.Box(
            layout=ipywidgets.Layout(
                display="none",
                height="0",
                min_height="0",
                width="0",
            ),
        )
        self.plot_area = ipywidgets.Box(
            [plot_widget, self.runtime_area],
            layout=ipywidgets.Layout(
                min_height="24rem",
                width="100%",
            ),
        )
        self.legend_area = ipywidgets.VBox()
        self.controls_area = ipywidgets.VBox()
        self.info_area = ipywidgets.VBox()
        self.modal_area = ipywidgets.VBox()
        self.output_area = ipywidgets.Output()
        self.status_area = ipywidgets.HTML(value="")
        self._info_widgets: dict[int, object] = {}
        self._disconnected_style_written = False
        self.parts = FigureLayoutParts(
            figure=figure,
            generation=generation,
            shell=self,
            plot=self.plot_area,
            legend=self.legend_area,
            controls=self.controls_area,
            info=self.info_area,
            modal=self.modal_area,
            output=self.output_area,
            status=self.status_area,
        )
        self.layout_instance = self._create_layout_instance(
            figure.layout_class,
            figure.layout_options,
        )
        self.root = self._build_root_widget()

        for widget, class_name in (
            (self.root, "mt-plot"),
            (self.plot_area, "mt-plot__plot"),
            (self.plot_widget, "mt-plot__plotly-widget"),
            (self.runtime_area, "mt-plot__runtime"),
            (self.legend_area, "mt-plot__legend"),
            (self.controls_area, "mt-plot__controls"),
            (self.info_area, "mt-plot__info"),
            (self.modal_area, "mt-plot__modal"),
            (self.output_area, "mt-plot__output"),
            (self.status_area, "mt-plot__status"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)
        add_root_class = getattr(self.root, "add_class", None)
        if add_root_class is not None:
            add_root_class(f"mt-plot--figure-{figure.id}-generation-{generation_id}")
            add_root_class(f"mt-plot--generation-{generation_id}")

    def _create_layout_instance(
        self,
        layout_class: type[object],
        layout_options: dict[str, object],
    ) -> object:
        """Instantiate the configured layout class for this generation."""

        try:
            return layout_class(self.parts, **layout_options)
        except Exception as exc:
            raise LayoutValidationError(
                "Failed to construct plotting layout "
                f"{layout_class.__name__} for figure {self._figure_name()}."
            ) from exc

    def _build_root_widget(self) -> object:
        """Build and validate the displayed root widget for this generation."""

        import ipywidgets as ipywidgets

        build = getattr(self.layout_instance, "build", None)
        if not callable(build):
            raise LayoutValidationError(
                "Plotting layout "
                f"{type(self.layout_instance).__name__} must define build()."
            )

        try:
            root = build()
        except Exception as exc:
            raise LayoutValidationError(
                "Failed to build plotting layout "
                f"{type(self.layout_instance).__name__} for figure "
                f"{self._figure_name()}."
            ) from exc

        if root is None:
            raise LayoutValidationError(
                "Plotting layout "
                f"{type(self.layout_instance).__name__} returned None from build()."
            )
        if not isinstance(root, ipywidgets.Widget):
            raise LayoutValidationError(
                "Plotting layout "
                f"{type(self.layout_instance).__name__} must return an "
                "ipywidgets.Widget from build()."
            )
        if not self.parts.was_used("plot"):
            raise LayoutValidationError(
                "Plotting layout "
                f"{type(self.layout_instance).__name__} must access parts.plot "
                "during build()."
            )
        return root

    def _figure_name(self) -> str:
        """Return a readable figure label for layout diagnostics."""

        if self.figure.name is not None:
            return repr(self.figure.name)
        return f"<unnamed figure {self.figure.id}>"

    def set_controls(self, controls: tuple[object, ...]) -> None:
        """Replace the controls slot without touching plot or status widgets."""

        self.controls_area.children = controls

    def set_legend(self, rows: tuple[object, ...]) -> None:
        """Replace the legend slot without touching plot or controls widgets."""

        self.legend_area.children = rows

    def set_status(self, message: str | None, *, kind: str = "info") -> None:
        """Show or clear a small status message in the information slot."""

        if not message:
            self.status_area.value = ""
            return
        self.status_area.value = _escape_html(message)

    def set_info(self, snapshots: tuple[InfoCardSnapshot, ...]) -> None:
        """Render authored info cards through Markdown-capable outputs."""

        wanted_ids = {snapshot.card_id for snapshot in snapshots}
        for card_id in tuple(self._info_widgets):
            if card_id not in wanted_ids:
                widget = self._info_widgets.pop(card_id)
                close = getattr(widget, "close", None)
                if callable(close):
                    close()

        children = []
        for snapshot in snapshots:
            widget = self._info_widgets.get(snapshot.card_id)
            if widget is None:
                widget = self._create_info_card_widget()
                self._info_widgets[snapshot.card_id] = widget
            self._sync_info_card_widget(widget, snapshot)
            children.append(widget)
        next_children = tuple(children)
        if self.info_area.children != next_children:
            self.info_area.children = next_children

    def set_disconnected(self, message: str) -> None:
        """Mark this visible shell as disconnected from live model callbacks."""

        self.set_status(None)
        self.mark_figure_disconnected()
        self.output_area.append_stdout(f"{message}\n")
        self.disable_controls()
        detach = getattr(self.layout_instance, "detach", None)
        if callable(detach):
            detach()

    def dispose(self) -> None:
        """Release child widget references held by the shell."""

        try:
            close = getattr(self.layout_instance, "close", None)
            if callable(close):
                close()
        finally:
            self.legend_area.children = ()
            self.controls_area.children = ()
            self.info_area.children = ()
            self._info_widgets.clear()
            self.modal_area.children = ()
            self.set_status(None)

    def disable_controls(self) -> None:
        """Disable every currently visible control widget."""

        self._disable_controls(self.controls_area)
        self._disable_controls(self.legend_area)

    def mark_figure_disconnected(self) -> None:
        """Add a visible disabled marker to the retired figure frame."""

        add_class = getattr(self.root, "add_class", None)
        if add_class is not None:
            add_class("mt-plot--disconnected")
        self.plot_area.layout.visibility = "visible"
        if not self._disconnected_style_written:
            from IPython.display import HTML

            self.output_area.append_display_data(
                HTML(
                    _disconnected_figure_script(
                        figure_id=self.figure.id,
                        generation_id=self.generation_id,
                    )
                )
            )
            self._disconnected_style_written = True

    def _disable_controls(self, widget: object) -> None:
        """Disable a widget subtree when the frontend object is retired."""

        if hasattr(widget, "disabled"):
            widget.disabled = True
        for child in getattr(widget, "children", ()):
            self._disable_controls(child)

    def _create_info_card_widget(self) -> object:
        """Create one Markdown info card widget."""

        import ipywidgets as ipywidgets

        title = ipywidgets.Output()
        body = ipywidgets.Output()
        card = ipywidgets.VBox(
            [title, body],
            layout=ipywidgets.Layout(grid_gap="0.15rem", width="100%"),
        )
        card._mt_title = title
        card._mt_body = body
        card._mt_title_markdown = None
        card._mt_body_markdown = None
        for widget, class_name in (
            (card, "mt-plot__info-card"),
            (title, "mt-plot__info-title"),
            (body, "mt-plot__info-body"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)
        return card

    def _sync_info_card_widget(
        self,
        widget: object,
        snapshot: InfoCardSnapshot,
    ) -> None:
        """Synchronize one info card widget from a snapshot."""

        title = getattr(widget, "_mt_title")
        body = getattr(widget, "_mt_body")
        title_markdown = snapshot.title_markdown
        body_markdown = snapshot.markdown
        if getattr(widget, "_mt_title_markdown", None) != title_markdown:
            title.outputs = _markdown_outputs(
                f"**{title_markdown}**" if title_markdown else ""
            )
            title.layout.display = "block" if title_markdown else "none"
            widget._mt_title_markdown = title_markdown
        if getattr(widget, "_mt_body_markdown", None) != body_markdown:
            body.outputs = _markdown_outputs(body_markdown)
            widget._mt_body_markdown = body_markdown


class FigureDisplayGeneration:
    """Own the frontend widgets and reactive effects for one displayed output."""

    def __init__(
        self,
        figure: FigureHandle,
        *,
        generation_id: int,
        execution_key: Hashable,
        policy: str,
        backend: str = "ipywidgets",
    ) -> None:
        """Create a hydrated display generation for a durable figure."""

        self.figure = figure
        self.generation_id = generation_id
        self.execution_key = execution_key
        self.policy = policy
        self.backend_name = normalize_display_backend(backend)
        self.state = "active"
        self.displayed = False
        self.renderer = PlotlyRenderer(
            figure,
            generation_id=generation_id,
            accepts_frontend_events=self.accepts_frontend_events,
        )
        self.frontend = None
        if self.backend_name == "anywidget":
            from .frontend.backend import create_frontend_backend

            self.frontend = create_frontend_backend(self, self.renderer.figure_widget)
            self.layout = self.frontend.layout
            self.modal = self.frontend.modal
            self.root = self.frontend.root
        else:
            self.layout = WidgetLayoutShell(
                figure,
                self,
                self.renderer.figure_widget,
                generation_id=generation_id,
            )
            from .modal import ModalController

            self.modal = ModalController(self, self.layout.modal_area)
            self.root = self.layout.root
        self.resize_sync_widget = self.renderer.resize_sync_widget
        if self.resize_sync_widget is not None:
            _hide_widget_layout(self.resize_sync_widget)
        if self.resize_sync_widget is not None and self.backend_name == "ipywidgets":
            self.layout.runtime_area.children = (
                *self.layout.runtime_area.children,
                self.resize_sync_widget,
            )
            self.layout.plot_area.layout.visibility = "hidden"
        self._control_observers: dict[tuple[int, object], tuple[object, object]]
        self._control_observers = {}
        self._control_edit_observers: dict[tuple[int, object], tuple[object, object]]
        self._control_edit_observers = {}
        self._control_reset_observers: dict[tuple[int, object], tuple[object, object]]
        self._control_reset_observers = {}
        self._parameter_widgets: dict[tuple[int, object], object] = {}
        self._legend_observers: dict[int, tuple[object, object]] = {}
        self._legend_sound_observers: dict[int, tuple[object, object]] = {}
        self._legend_edit_observers: dict[int, tuple[object, object]] = {}
        self._legend_widgets: dict[int, object] = {}
        self._syncing_widget_values = False
        self._defer_info_updates_depth = 0
        self._pending_info_render = False
        self._last_info_render_time = 0.0
        self._info_preview_interval = 0.15
        self._node_effects: dict[int, list[object]] = {}
        self._figure_effects: list[object] = []
        self._trace_data_defer_depth = 0
        self._pending_trace_data: dict[tuple[int, str], TraceDataSnapshot] = {}
        from .audio import AudioOutputWidget

        self.home_sync_widget = self.renderer.home_sync_widget
        if self.home_sync_widget is not None:
            _hide_widget_layout(self.home_sync_widget)
        if self.home_sync_widget is not None and self.backend_name == "ipywidgets":
            self.layout.runtime_area.children = (
                *self.layout.runtime_area.children,
                self.home_sync_widget,
            )
        self.audio_output = AudioOutputWidget(self.figure)
        _hide_widget_layout(self.audio_output)
        if self.backend_name == "ipywidgets":
            self.layout.runtime_area.children = (
                *self.layout.runtime_area.children,
                self.audio_output,
            )
        elif self.frontend is not None:
            self.frontend.set_runtime_widgets(
                tuple(
                    widget
                    for widget in (
                        self.resize_sync_widget,
                        self.home_sync_widget,
                        self.audio_output,
                    )
                    if widget is not None
                )
            )

    def hydrate(self) -> None:
        """Attach reactive effects and render the current model snapshots."""

        # Figure-level effects reconcile widgets from figure snapshots. This
        # keeps control layout and slider value mirroring separate while still
        # responding when plots are added or removed.
        self._figure_effects.append(Effect(self._render_view_from_model))
        self._figure_effects.append(Effect(self._reconcile_legend_from_model))
        self._figure_effects.append(Effect(self._reconcile_controls_from_model))
        self._figure_effects.append(Effect(self._sync_slider_values_from_model))
        self._figure_effects.append(Effect(self._render_info_from_model))
        for node in tuple(self.figure.plots):
            self.attach_node(node)

    def display(self) -> bool:
        """Display this generation's root widget at most once."""

        if self.displayed:
            return True
        if self.backend_name == "anywidget":
            displayed = renderer_module._display_ipython(self.root)
            self.displayed = displayed
            return displayed
        try:
            from IPython.display import HTML
        except ImportError:
            return False

        displayed = renderer_module._display_ipython(HTML(_PLOT_WIDGET_STYLE), self.root)
        self.displayed = displayed
        return displayed

    def accepts_frontend_events(self) -> bool:
        """Return whether callbacks from this generation may mutate the model."""

        return self.state == "active" and self.figure.active_generation is self

    def attach_node(self, node: PlotNode) -> None:
        """Attach renderer effects for a plot node if not already attached."""

        if node.id in self._node_effects:
            return

        def _render_trace_data() -> None:
            snapshots = node.trace_data_signal()
            if snapshots is not None and self.state == "active":
                for snapshot in snapshots:
                    self._render_or_queue_trace_data(snapshot)

        def _render_trace_style() -> None:
            if self.state == "active":
                for snapshot in node.trace_style_snapshot():
                    self.renderer.render_trace_style(snapshot)

        self._node_effects[node.id] = [
            Effect(_render_trace_data),
            Effect(_render_trace_style),
        ]
        self.refresh_node(node)

    def refresh_node(self, node: PlotNode) -> None:
        """Render one plot node's current snapshots into this generation."""

        if self.state != "active":
            return
        snapshots = node.trace_data_signal()
        if snapshots is not None:
            for snapshot in snapshots:
                self._render_or_queue_trace_data(snapshot)
        for snapshot in node.trace_style_snapshot():
            self.renderer.render_trace_style(snapshot)

    def defer_trace_data_updates(self) -> object:
        """Return a context manager that coalesces trace data rendering."""

        generation = self

        class _TraceDataUpdateDeferral:
            """Queue trace data snapshots until a model mutation finishes."""

            def __enter__(self) -> None:
                generation._trace_data_defer_depth += 1
                return None

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                generation._trace_data_defer_depth = max(
                    0,
                    generation._trace_data_defer_depth - 1,
                )
                if generation._trace_data_defer_depth == 0:
                    generation.flush_deferred_trace_data()
                return None

        return _TraceDataUpdateDeferral()

    def flush_deferred_trace_data(self) -> None:
        """Render queued trace data snapshots in one Plotly widget batch."""

        if self.state != "active" or self._trace_data_defer_depth:
            return
        if not self._pending_trace_data:
            return
        snapshots = tuple(self._pending_trace_data.values())
        self._pending_trace_data.clear()
        self.renderer.render_trace_data_batch(snapshots)

    def _render_or_queue_trace_data(self, snapshot: TraceDataSnapshot) -> None:
        """Render one trace snapshot now or queue it for a coalesced flush."""

        if self._trace_data_defer_depth:
            self._pending_trace_data[(snapshot.node_id, snapshot.trace_role)] = snapshot
            return
        self.renderer.render_trace_data(snapshot)

    def refresh_from_model(self) -> None:
        """Render the figure's current model state into this generation."""

        if self.state != "active":
            return
        for node in tuple(self.figure.plots):
            if node.id not in self._node_effects:
                self.attach_node(node)
            else:
                self.refresh_node(node)
        self.figure.reconcile_legend()
        self.figure.reconcile_controls()
        self.figure.sync_controls(self.figure.slider_value_snapshot())
        self.figure.sync_animation_state(self.figure.animation_state_snapshot())
        if self.frontend is not None:
            self.frontend.set_info(self.figure.info_snapshot())
        else:
            self.layout.set_info(self.figure.info_snapshot())
        self.renderer.render_view(self.figure.view_snapshot())

    def detach_node(self, node: PlotNode) -> None:
        """Detach one plot node from this generation's live frontend."""

        for effect in self._node_effects.pop(node.id, ()):
            dispose = getattr(effect, "dispose", None)
            if dispose is not None:
                dispose()
        self.renderer.remove_node(node)

        # Remove controls immediately for active generations. Retired
        # generations are snapshots and should not continue following model
        # edits after they have been disconnected.
        if self.state == "active":
            if self.frontend is not None:
                self.figure.reconcile_controls()
                self.figure.reconcile_legend()
                return
            for key in tuple(self._parameter_widgets):
                if key[0] == node.id:
                    from .widgets import _dispose_control

                    _dispose_control(self, key)
            self.layout.set_controls(tuple(self._parameter_widgets.values()))
            from .legend import _dispose_legend_row

            _dispose_legend_row(self, node.id)
            self.layout.set_legend(tuple(self._legend_widgets.values()))

    def reconcile_controls(
        self,
        layout: tuple[object, ...],
    ) -> None:
        """Reconcile control widgets in this generation's controls slot."""

        if self.state != "active":
            return
        from .widgets import reconcile_parameter_controls

        if self.frontend is not None:
            self.frontend.reconcile_controls(layout)
        else:
            reconcile_parameter_controls(self, layout)

    def sync_controls(self, values: tuple[SliderValueItem, ...]) -> None:
        """Mirror model values into this generation's existing sliders."""

        if self.state != "active":
            return
        from .widgets import sync_parameter_controls

        if self.frontend is not None:
            self.frontend.sync_control_values(values)
        else:
            sync_parameter_controls(self, values)

    def sync_animation_state(
        self,
        values: tuple[ParameterAnimationStateItem, ...],
    ) -> None:
        """Mirror animation play state into this generation's controls."""

        if self.state != "active":
            return
        if self.frontend is not None:
            self.frontend.sync_animation_state(values)

    def defer_info_updates(self) -> object:
        """Return a context manager that postpones expensive info rendering."""

        generation = self

        class _InfoUpdateDeferral:
            """Postpone info panel rendering until the caller leaves preview mode."""

            def __enter__(self) -> None:
                generation._defer_info_updates_depth += 1
                return None

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                generation._defer_info_updates_depth = max(
                    0,
                    generation._defer_info_updates_depth - 1,
                )
                return None

        return _InfoUpdateDeferral()

    def flush_deferred_info(self) -> None:
        """Render the latest info snapshot after preview updates have settled."""

        if self.state != "active" or self._defer_info_updates_depth:
            return
        if not self._pending_info_render:
            return
        self._pending_info_render = False
        self._render_info_snapshot()

    def reconcile_legend(self, items: tuple[LegendItem, ...]) -> None:
        """Reconcile legend row widgets in this generation's legend slot."""

        if self.state != "active":
            return
        from .legend import reconcile_legend

        if self.frontend is not None:
            self.frontend.reconcile_legend(items)
        else:
            reconcile_legend(self, items)

    def retire(self) -> None:
        """Disconnect this visible output from future model callbacks."""

        if self.state != "active":
            return
        self.figure._animation_coordinator.stop_all()
        self.state = "disconnected"
        self._dispose_effects()
        self._dispose_control_observers()
        self._dispose_legend_observers()
        self.modal.dispose()
        self.renderer.dispose_callbacks()
        self.renderer.disable_interactivity()
        self.renderer.disable_resize_sync()
        self.resize_sync_widget = self.renderer.resize_sync_widget
        message = (
            "This plot output is disconnected because a newer live display "
            "generation is active."
        )
        if self.frontend is not None:
            self.frontend.retire(message)
        else:
            self.layout.set_disconnected(message)

    def close(self) -> None:
        """Dispose callbacks, effects, and widget references for this generation."""

        if self.state == "closed":
            return
        if self.state == "active":
            self.retire()
        self.state = "closed"
        self._dispose_effects()
        self._dispose_control_observers()
        self._dispose_legend_observers()
        self.modal.dispose()
        self.renderer.dispose()
        self.audio_output.close()
        if self.frontend is not None:
            self.frontend.dispose()
        else:
            self.layout.dispose()

    def _reconcile_controls_from_model(self) -> None:
        """Reconcile visible controls from the figure-level layout snapshot."""

        self.figure._plot_topology_signal()
        self.figure._info_topology_signal()
        if self.state == "active":
            self.figure.reconcile_controls()

    def _reconcile_legend_from_model(self) -> None:
        """Reconcile visible legend rows from figure-level snapshots."""

        self.figure._plot_topology_signal()
        if self.state == "active":
            self.figure.reconcile_legend()

    def _render_view_from_model(self) -> None:
        """Render the Python-owned current view into Plotly layout ranges."""

        if self.state == "active":
            self.renderer.render_view(self.figure.view_snapshot())

    def _sync_slider_values_from_model(self) -> None:
        """Mirror figure-level slider values without rebuilding controls."""

        self.figure._plot_topology_signal()
        self.figure._info_topology_signal()
        if self.state == "active":
            self.figure.sync_controls(self.figure.slider_value_snapshot())

    def _render_info_from_model(self) -> None:
        """Render authored info snapshots into the layout shell."""

        if self.state != "active":
            return
        if self._defer_info_updates_depth:
            self.figure._read_info_render_dependencies()
            now = time.monotonic()
            if now - self._last_info_render_time >= self._info_preview_interval:
                self._pending_info_render = False
                self._render_info_snapshot()
            else:
                self._pending_info_render = True
            return
        self._render_info_snapshot()

    def _render_info_snapshot(self) -> None:
        """Publish the current authored info snapshot to the active frontend."""

        if self.frontend is not None:
            self.frontend.set_info(self.figure.info_snapshot())
        else:
            self.layout.set_info(self.figure.info_snapshot())
        self._last_info_render_time = time.monotonic()

    def _dispose_effects(self) -> None:
        """Dispose generation-owned reactive effects exactly once."""

        for effects in list(self._node_effects.values()):
            for effect in effects:
                dispose = getattr(effect, "dispose", None)
                if dispose is not None:
                    dispose()
        self._node_effects.clear()

        for effect in self._figure_effects:
            dispose = getattr(effect, "dispose", None)
            if dispose is not None:
                dispose()
        self._figure_effects.clear()

    def _dispose_control_observers(self) -> None:
        """Detach widget observers while preserving visible retired controls."""

        for key in tuple(self._parameter_widgets):
            observer = self._control_observers.pop(key, None)
            if observer is not None:
                for widget, callback in observer:
                    widget.unobserve(callback, names="value")
            reset_observer = self._control_reset_observers.pop(key, None)
            if reset_observer is not None:
                button, callback = reset_observer
                button.on_click(callback, remove=True)
            edit_observer = self._control_edit_observers.pop(key, None)
            if edit_observer is not None:
                button, callback = edit_observer
                button.on_click(callback, remove=True)
        self._parameter_widgets.clear()

    def _dispose_legend_observers(self) -> None:
        """Detach legend click callbacks while preserving visible rows."""

        for node_id in tuple(self._legend_widgets):
            from .legend import _dispose_legend_row

            _dispose_legend_row(self, node_id)


def current_execution_key() -> Hashable:
    """Return a best-effort key for the active notebook execution."""

    try:
        from IPython import get_ipython
    except ImportError:
        return ("python", 0)

    shell = get_ipython()
    if shell is None:
        return ("python", 0)
    execution_count = getattr(shell, "execution_count", None)
    if execution_count is None:
        return ("ipython", id(shell), 0)
    return ("ipython", id(shell), execution_count)


def normalize_display_backend(backend: str | None) -> str:
    """Validate the currently supported display backend selector."""

    if backend is None or backend == "anywidget":
        return "anywidget"
    if backend in ("jupyter", "ipywidgets", "widget"):
        return "ipywidgets"
    raise PlotSpecError(
        "FigureHandle.show(...) backend must be one of None, 'jupyter', "
        "'ipywidgets', 'widget', or 'anywidget'."
    )


def normalize_display_policy(policy: str | None) -> str:
    """Validate the currently supported display retirement policy."""

    if policy in (None, "disconnect"):
        return "disconnect"
    raise PlotSpecError(
        "FigureHandle.show(...) currently supports only policy='disconnect'."
    )


def _escape_html(value: str) -> str:
    """Return a minimal escaped HTML text fragment."""

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _markdown_outputs(markdown: str) -> tuple[dict[str, object], ...]:
    """Return an ipywidgets Output payload for Markdown content."""

    if not markdown:
        return ()
    return (
        {
            "output_type": "display_data",
            "data": {"text/markdown": markdown},
            "metadata": {},
        },
    )


def _hide_widget_layout(widget: object) -> None:
    """Make helper widgets layout-neutral before their frontend view renders."""

    layout = getattr(widget, "layout", None)
    if layout is not None and hasattr(layout, "display"):
        layout.display = "none"
    if layout is not None and hasattr(layout, "height"):
        layout.height = "0"
    if layout is not None and hasattr(layout, "width"):
        layout.width = "0"


_PLOT_WIDGET_STYLE = """
<style>
.mt-plot .modebar-container {
  margin: -0.75rem;
  padding: 0.75rem;
}
.mt-plot .modebar {
  opacity: 0;
  transition: opacity 140ms ease;
}
.mt-plot .modebar-container:hover .modebar,
.mt-plot .modebar:hover,
.mt-plot .modebar:focus-within {
  opacity: 1;
}
.mt-plot__plot,
.mt-plot__plot > .widget-box,
.mt-plot__plot > .jupyter-widgets,
.mt-plot__plot .js-plotly-plot,
.mt-plot__plot .plot-container,
.mt-plot__plot .svg-container {
  max-width: 100% !important;
  width: 100% !important;
}
.mt-plot__runtime,
.mt-plot__runtime *,
.mt-plot .jupyter-widgets-disconnected::before {
  display: none !important;
}
.fa-chain-broken,
.fa-unlink,
[class*="fa-chain-broken"],
[class*="fa-unlink"],
[data-icon*="broken" i],
[data-icon*="unlink" i] {
  display: none !important;
}
.mt-plot.mt-plot--disconnected,
.mt-plot.jupyter-widgets-disconnected {
  position: relative;
  border-radius: 0.35rem;
  padding: 0.25rem;
  box-shadow: inset 0 0 0 0.18rem rgba(100, 116, 139, 0.6);
  background: repeating-linear-gradient(
    135deg,
    rgba(148, 163, 184, 0.12) 0,
    rgba(148, 163, 184, 0.12) 0.5rem,
    rgba(255, 255, 255, 0) 0.5rem,
    rgba(255, 255, 255, 0) 1rem
  );
}
.mt-plot.mt-plot--disconnected > *,
.mt-plot.jupyter-widgets-disconnected > * {
  filter: grayscale(0.06) saturate(0.98);
  opacity: 1;
  position: relative;
  z-index: 1;
}
.mt-plot.mt-plot--disconnected .mt-plot__plot,
.mt-plot.jupyter-widgets-disconnected .mt-plot__plot {
  visibility: visible !important;
}
.mt-plot.mt-plot--disconnected .modebar,
.mt-plot.jupyter-widgets-disconnected .modebar {
  opacity: 1;
}
.mt-plot.mt-plot--disconnected *,
.mt-plot.jupyter-widgets-disconnected * {
  cursor: default !important;
}
.mt-plot.mt-plot--disconnected .mt-plot__controls button,
.mt-plot.mt-plot--disconnected .mt-plot__controls input,
.mt-plot.mt-plot--disconnected .mt-plot__controls .noUi-target,
.mt-plot.mt-plot--disconnected .mt-plot__legend button,
.mt-plot.mt-plot--disconnected .mt-plot__plot .draglayer,
.mt-plot.mt-plot--disconnected .mt-plot__plot .hoverlayer,
.mt-plot.mt-plot--disconnected .mt-plot__plot .drag,
.mt-plot.mt-plot--disconnected .mt-plot__plot .cursor-pointer,
.mt-plot.jupyter-widgets-disconnected .mt-plot__controls button,
.mt-plot.jupyter-widgets-disconnected .mt-plot__controls input,
.mt-plot.jupyter-widgets-disconnected .mt-plot__controls .noUi-target,
.mt-plot.jupyter-widgets-disconnected .mt-plot__legend button,
.mt-plot.jupyter-widgets-disconnected .mt-plot__plot .draglayer,
.mt-plot.jupyter-widgets-disconnected .mt-plot__plot .hoverlayer,
.mt-plot.jupyter-widgets-disconnected .mt-plot__plot .drag,
.mt-plot.jupyter-widgets-disconnected .mt-plot__plot .cursor-pointer {
  pointer-events: none !important;
}
.mt-plot.mt-plot--disconnected::before,
.mt-plot.jupyter-widgets-disconnected::before {
  content: "";
  background: rgba(255, 255, 255, 0.92);
  border: 0.18rem solid #dc2626;
  border-radius: 999px;
  box-shadow: 0 0.125rem 0.45rem rgba(0, 0, 0, 0.28);
  height: 2.15rem;
  width: 2.15rem;
  position: absolute;
  top: 0.45rem;
  left: 0.45rem;
  z-index: 3;
  pointer-events: none;
}
.mt-plot.mt-plot--disconnected::after,
.mt-plot.jupyter-widgets-disconnected::after {
  content: "";
  background: #dc2626;
  border-radius: 999px;
  height: 0.22rem;
  width: 1.72rem;
  position: absolute;
  top: 1.42rem;
  left: 0.83rem;
  transform: rotate(-45deg);
  transform-origin: center;
  z-index: 4;
  pointer-events: none;
}
.mt-plot__legend p {
  margin: 0 !important;
}
.mt-plot__legend,
.mt-plot__controls {
  border: 0 !important;
  box-shadow: none !important;
}
.mt-plot:not(.mt-plot--compact) .mt-plot__side-column {
  min-height: 24rem !important;
}
.mt-plot:not(.mt-plot--compact) .mt-plot__legend,
.mt-plot:not(.mt-plot--compact) .mt-plot__parameters-panel,
.mt-plot:not(.mt-plot--compact) .mt-plot__info {
  flex: 1 1 0 !important;
  min-height: 0 !important;
}
.mt-plot.mt-plot--compact .mt-plot__plot-column,
.mt-plot.mt-plot--compact .mt-plot__side-column,
.mt-plot.mt-plot--compact .mt-plot__parameters-panel,
.mt-plot.mt-plot--compact .mt-plot__legend,
.mt-plot.mt-plot--compact .mt-plot__controls,
.mt-plot.mt-plot--compact .mt-plot__info,
.mt-plot.mt-plot--compact .mt-plot__status {
  min-height: 0 !important;
  min-width: 0 !important;
  width: 100% !important;
}
.mt-plot.mt-plot--compact .mt-plot__legend,
.mt-plot.mt-plot--compact .mt-plot__parameters-panel,
.mt-plot.mt-plot--compact .mt-plot__info {
  flex: 0 1 auto !important;
  max-height: 12rem !important;
}
.mt-plot.mt-plot--compact .mt-plot__parameter-control {
  flex-flow: row wrap !important;
  gap: 0.18rem 0.25rem !important;
  overflow: visible !important;
}
.mt-plot.mt-plot--compact .mt-plot__parameter-spacer {
  display: none !important;
}
.mt-plot.mt-plot--compact .mt-plot__parameter-control > * {
  min-width: 0 !important;
}
.mt-plot.mt-plot--compact .mt-plot__parameter-control > *:first-child {
  flex: 0 1 auto !important;
}
.mt-plot.mt-plot--compact .mt-plot__parameter-control > *:nth-child(3) {
  flex: 1 1 calc(10ch + 4rem + 0.36rem) !important;
}
.mt-plot__legend-edit-button {
  align-items: center !important;
  display: flex !important;
  justify-content: center !important;
  line-height: 1 !important;
}
.mt-plot__parameter-reset-button,
.mt-plot__parameter-edit-button {
  align-items: center !important;
  display: flex !important;
  justify-content: center !important;
  line-height: 1 !important;
}
.mt-plot__legend-edit-button i,
.mt-plot__legend-edit-button .fa,
.mt-plot__parameter-reset-button i,
.mt-plot__parameter-reset-button .fa,
.mt-plot__parameter-edit-button i,
.mt-plot__parameter-edit-button .fa {
  line-height: 1 !important;
  margin: 0 !important;
}
.mt-plot__parameter-control,
.mt-plot__parameter-control * {
  background: transparent !important;
  background-color: transparent !important;
  box-sizing: border-box !important;
  margin: 0 !important;
  padding: 0 !important;
}
.mt-plot__parameter-control p {
  margin: 0 !important;
}
.mt-plot__parameter-spacer {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}
.mt-plot__parameter-slider {
  height: 0.9rem !important;
  margin: 0 !important;
  max-width: 100% !important;
  min-height: 0 !important;
  min-width: 0 !important;
  padding: 0 !important;
  width: 100% !important;
}
.mt-plot__parameter-slider input[type="range"] {
  margin: 0 !important;
  width: 100% !important;
}
.mt-plot__parameter-slider input[type="range"]::-webkit-slider-runnable-track {
  height: 0.08rem !important;
}
.mt-plot__parameter-slider input[type="range"]::-moz-range-track {
  height: 0.08rem !important;
}
.mt-plot__parameter-slider input[type="range"]::-webkit-slider-thumb {
  border: 0.08rem solid #ffffff !important;
  box-shadow: 0 0 0 0.05rem rgba(33, 113, 181, 0.45) !important;
}
.mt-plot__parameter-slider input[type="range"]::-moz-range-thumb {
  border: 0.08rem solid #ffffff !important;
  box-shadow: 0 0 0 0.05rem rgba(33, 113, 181, 0.45) !important;
}
.mt-plot__parameter-slider .noUi-target {
  background: #d7d9dc !important;
  background-color: #d7d9dc !important;
  border: 0 !important;
  box-shadow: none !important;
  margin: 0.37rem 0 0 0 !important;
  height: 0.08rem !important;
}
.mt-plot__parameter-slider .noUi-connect {
  background: #2680d9 !important;
  background-color: #2680d9 !important;
}
.mt-plot__parameter-slider .noUi-connects,
.mt-plot__parameter-slider .noUi-base {
  background: #d7d9dc !important;
  background-color: #d7d9dc !important;
  height: 0.08rem !important;
}
.mt-plot__parameter-slider .noUi-horizontal .noUi-handle {
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 0.08rem solid #ffffff !important;
  border-radius: 999px !important;
  box-shadow: 0 0 0 0.05rem rgba(33, 113, 181, 0.45) !important;
  height: 0.8rem !important;
  right: -0.4rem !important;
  top: -0.36rem !important;
  width: 0.8rem !important;
}
.mt-plot__parameter-slider .noUi-handle::before,
.mt-plot__parameter-slider .noUi-handle::after {
  display: none !important;
}
.mt-plot__parameter-value,
.mt-plot__parameter-limit {
  background: transparent !important;
  background-color: transparent !important;
}
.mt-plot__parameter-value .widget-text,
.mt-plot__parameter-value .widget-input,
.mt-plot__parameter-limit .widget-text,
.mt-plot__parameter-limit .widget-input {
  background: transparent !important;
  background-color: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
.mt-plot__parameter-value input,
.mt-plot__parameter-value input[type="text"],
.mt-plot__parameter-limit input,
.mt-plot__parameter-limit input[type="text"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  background: #f8fafc !important;
  background-color: #f8fafc !important;
  border: 1px solid #d8dee6 !important;
  border-radius: 0.2rem !important;
  box-shadow: none !important;
  color: #2f3437 !important;
  font-family: system-ui, sans-serif !important;
  letter-spacing: 0 !important;
  margin: 0 !important;
  outline: 1px solid transparent !important;
  outline-offset: 0 !important;
}
.mt-plot__parameter-value input:focus,
.mt-plot__parameter-value input[type="text"]:focus,
.mt-plot__parameter-limit input:focus,
.mt-plot__parameter-limit input[type="text"]:focus {
  outline: 1px solid rgba(71, 85, 105, 0.55) !important;
}
.mt-plot__parameter-value input {
  font-size: 0.72rem !important;
  height: 1.25rem !important;
  min-height: 1.25rem !important;
  padding: 0 0.22rem !important;
  text-align: right !important;
}
.mt-plot__parameter-limit input {
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  font-size: 0.68rem !important;
  height: 0.95rem !important;
  -webkit-mask-image: linear-gradient(to right, transparent 0, #000 0.55rem) !important;
  mask-image: linear-gradient(to right, transparent 0, #000 0.55rem) !important;
  min-height: 0.95rem !important;
  padding: 0 !important;
}
.mt-plot__parameter-limit input:focus,
.mt-plot__parameter-limit input[type="text"]:focus {
  outline-color: rgba(71, 85, 105, 0.45) !important;
}
.mt-plot__parameter-minimum input {
  padding-left: 0 !important;
  text-align: right !important;
}
.mt-plot__parameter-maximum input {
  padding-right: 0 !important;
  text-align: left !important;
}
.mt-plot__overlay {
  position: relative;
  width: 100%;
}
.mt-plot__modal {
  bottom: 0;
  left: 0;
  pointer-events: none;
  position: absolute;
  right: 0;
  top: 0;
  width: 100%;
  z-index: 20;
}
.mt-modal {
  align-items: center;
  background: rgba(15, 23, 42, 0.18);
  box-sizing: border-box;
  height: 100%;
  margin: 0;
  min-height: 100%;
  padding: 1rem;
  pointer-events: auto;
  position: absolute;
  width: 100%;
}
.mt-modal,
.mt-modal * {
  box-sizing: border-box;
}
.mt-no-scroll {
  overflow: visible !important;
  scrollbar-width: none !important;
}
.mt-no-scroll::-webkit-scrollbar {
  display: none !important;
}
.mt-no-scroll > .widget-box,
.mt-no-scroll > .widget-hbox,
.mt-no-scroll > .widget-vbox,
.mt-no-scroll > .jupyter-widgets {
  overflow: visible !important;
  scrollbar-width: none !important;
}
.mt-no-scroll > .widget-box::-webkit-scrollbar,
.mt-no-scroll > .widget-hbox::-webkit-scrollbar,
.mt-no-scroll > .widget-vbox::-webkit-scrollbar,
.mt-no-scroll > .jupyter-widgets::-webkit-scrollbar {
  display: none !important;
}
.mt-modal__dialog {
  background: #ffffff;
  border-radius: 0.35rem;
  box-shadow: 0 0.35rem 1.25rem rgba(15, 23, 42, 0.16);
  display: flex;
  flex-direction: column;
  max-height: calc(100% - 2rem);
  min-height: 0;
  overflow: hidden;
}
.mt-modal__header {
  border-bottom: 1px solid #e2e8f0;
  flex: 0 0 auto;
  overflow: hidden;
  padding-bottom: 0.5rem;
}
.mt-modal__title {
  color: #0f172a;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  font: 600 0.95rem/1.3 system-ui, sans-serif;
}
.mt-modal__body {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  padding: 0.75rem 0;
}
.mt-modal__message-error {
  background: #fff1f2;
  border: 1px solid #fecdd3;
  color: #9f1239;
  font: 0.85rem/1.35 system-ui, sans-serif;
  margin: 0 0 0.75rem 0;
  padding: 0.5rem 0.65rem;
}
.mt-modal__message-error ul {
  margin: 0;
  padding-left: 1.1rem;
}
.mt-config-field {
  align-items: center;
  box-sizing: border-box !important;
  min-height: 2rem;
}
.mt-config-field__label {
  color: #334155;
  font: 0.85rem/1.2 system-ui, sans-serif;
}
.mt-config-field__control {
  min-width: 0;
}
.mt-opacity-field,
.mt-color-field,
.mt-line-width-field {
  min-width: 0;
}
.mt-color-field__named {
  flex: 0 0 11ch !important;
  max-width: 11ch !important;
  min-width: 0 !important;
  position: relative !important;
  width: 11ch !important;
}
.mt-color-field__picker {
  flex: 0 0 2.35rem !important;
  max-width: 2.35rem !important;
  min-width: 0 !important;
  width: 2.35rem !important;
}
.mt-color-field__named select {
  max-width: 11ch !important;
  overflow: hidden !important;
  text-overflow: clip !important;
  width: 11ch !important;
}
.mt-color-field__named::after {
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0),
    #ffffff 78%
  );
  bottom: 1px;
  content: "";
  pointer-events: none;
  position: absolute;
  right: 1.45rem;
  top: 1px;
  width: 1.15rem;
  z-index: 1;
}
.mt-opacity-field__slider {
  flex: 0 0 5.5rem !important;
  min-width: 0 !important;
  width: 5.5rem !important;
}
.mt-opacity-field__entry {
  flex: 0 0 2.8rem !important;
  max-width: 2.8rem !important;
  min-width: 0 !important;
  width: 2.8rem !important;
}
.mt-opacity-field__entry input,
.mt-line-width-field__entry input {
  text-align: right !important;
}
.mt-line-width-field__sample {
  align-items: center;
  color: #475569;
  display: flex;
  font: 0.78rem/1 system-ui, sans-serif;
  height: 1.25rem;
  justify-content: center;
  max-width: 2em;
  width: 2em;
}
.mt-line-width-field__sample svg {
  height: 1.25rem;
  max-width: 2em;
  width: 2em;
}
.mt-line-width-field__suffix {
  color: #475569;
  font: 0.78rem/1 system-ui, sans-serif;
}
.mt-modal__footer {
  flex: 0 0 auto;
  overflow: hidden;
  padding-top: 0.35rem;
}
</style>
<script>
(() => {
  const STOPPED_CLASS = "mt-plot--kernel-disconnected";

  window.__mathToolkitRetirePlot = (root, options = {}) => {
    if (!root) {
      return;
    }
    root.classList.add("mt-plot--disconnected");
    if (options.kernelDisconnected) {
      root.classList.add(STOPPED_CLASS);
    }
  };
})();
</script>
"""


def _disconnected_figure_script(*, figure_id: int, generation_id: int) -> str:
    """Return a frontend retire call for one displayed plot generation."""

    return f"""
<script>
(() => {{
  const root = document.querySelector(
    ".mt-plot--figure-{figure_id}-generation-{generation_id}"
  );
  if (window.__mathToolkitRetirePlot) {{
    window.__mathToolkitRetirePlot(root);
  }} else if (root) {{
    root.classList.add("mt-plot--disconnected");
  }}
}})();
</script>
"""
