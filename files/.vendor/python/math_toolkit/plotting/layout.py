"""Define generation-owned plotting layout classes and layout parts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .display import WidgetLayoutShell
    from .display import FigureDisplayGeneration
    from .model import FigureHandle


class LayoutValidationError(ValueError):
    """Raised when a figure layout class fails the plotting layout contract."""


class FigureLayoutParts:
    """Expose generation-owned widgets and narrow layout utilities."""

    def __init__(
        self,
        *,
        figure: FigureHandle,
        generation: FigureDisplayGeneration,
        shell: WidgetLayoutShell,
        plot: object,
        legend: object,
        controls: object,
        info: object,
        modal: object,
        output: object,
        status: object,
    ) -> None:
        """Create layout parts for one display generation."""

        self.figure = figure
        self.generation = generation
        self._shell = shell
        self._plot = plot
        self._legend = legend
        self._controls = controls
        self._info = info
        self._modal = modal
        self._output = output
        self._status = status
        self._used_parts: set[str] = set()

    @property
    def plot(self) -> object:
        """Return the generation's Plotly widget or plot container."""

        self._used_parts.add("plot")
        return self._plot

    @property
    def controls(self) -> object:
        """Return the generation's controls container."""

        self._used_parts.add("controls")
        return self._controls

    @property
    def modal(self) -> object:
        """Return the generation's modal overlay container."""

        self._used_parts.add("modal")
        return self._modal

    @property
    def legend(self) -> object:
        """Return the generation's legend container."""

        self._used_parts.add("legend")
        return self._legend

    @property
    def info(self) -> object:
        """Return the generation's authored Markdown info container."""

        self._used_parts.add("info")
        return self._info

    @property
    def output(self) -> object:
        """Return the generation's notebook output widget."""

        self._used_parts.add("output")
        return self._output

    @property
    def status(self) -> object:
        """Return the generation's status widget."""

        self._used_parts.add("status")
        return self._status

    def set_status(self, message: str | None, *, kind: str = "info") -> None:
        """Delegate status updates to the generation-owned layout shell."""

        self._shell.set_status(message, kind=kind)

    def disable_controls(self) -> None:
        """Disable every currently visible control widget."""

        self._shell.disable_controls()

    def was_used(self, name: str) -> bool:
        """Return whether the named public part was accessed."""

        return name in self._used_parts


class FigureLayout:
    """Provide the base protocol for generation-owned plotting layouts."""

    def __init__(self, parts: FigureLayoutParts) -> None:
        """Store layout parts for one display generation."""

        self.parts = parts
        self.root: object | None = None

    def build(self) -> object:
        """Return the root widget that should be displayed."""

        raise NotImplementedError("FigureLayout subclasses must implement build().")

    def detach(self) -> None:
        """Respond to generation retirement while keeping the visual visible."""

        return None

    def close(self) -> None:
        """Release layout-owned widget resources on a best-effort basis."""

        close = getattr(self.root, "close", None)
        if callable(close):
            close()


class ResponsiveSidebarLayout(FigureLayout):
    """Arrange figures as a responsive plot with a wrapping sidebar."""

    def __init__(self, parts: FigureLayoutParts, *, debug_boxes: bool = False) -> None:
        """Create a responsive layout with optional visual box debugging."""

        super().__init__(parts)
        self.debug_boxes = bool(debug_boxes)

    def build(self) -> object:
        """Return the responsive sidebar layout root."""

        import ipywidgets as ipywidgets

        self.controls_area = self.parts.controls
        self.legend_area = self.parts.legend
        self.output_area = self.parts.output
        self.info_area = self.parts.info
        self.status_area = self.parts.status
        self.plot_area = self.parts.plot
        self.modal_area = self.parts.modal
        self._configure_plot_fill()
        self.legend_area.layout = ipywidgets.Layout(
            flex="0 0 auto",
            grid_gap="0.1rem",
            min_width="18rem",
            overflow="auto",
            width="100%",
        )
        self.controls_area.layout.overflow = "auto"
        self.legend_area.observe(self._sync_legend_visibility, names="children")
        self._sync_legend_visibility()
        self.parameters_panel = ipywidgets.VBox(
            [self.controls_area],
            layout=ipywidgets.Layout(
                flex="0 0 auto",
                grid_gap="0.08rem",
                min_width="18rem",
            ),
        )
        self.controls_area.observe(self._sync_parameter_visibility, names="children")
        self._sync_parameter_visibility()

        self.output_section = ipywidgets.VBox(
            [self.output_area],
            layout=ipywidgets.Layout(width="100%"),
        )
        self.plot_column = ipywidgets.VBox(
            [self.plot_area],
            layout=ipywidgets.Layout(
                flex="999 1 30rem",
                grid_gap="0.75rem",
                height="100%",
                min_height="24rem",
                min_width="24rem",
                width="100%",
            ),
        )
        add_plot_column_class = getattr(self.plot_column, "add_class", None)
        if add_plot_column_class is not None:
            add_plot_column_class("mt-plot__plot-column")
        self.side_column = ipywidgets.VBox(
            [
                self.legend_area,
                self.parameters_panel,
                ipywidgets.HTML("<h4 style='margin:0.75rem 0 0;'>Info</h4>"),
                self.info_area,
                self.status_area,
            ],
            layout=ipywidgets.Layout(
                flex="1 1 18rem",
                grid_gap="0.1rem",
                min_width="18rem",
            ),
        )
        self.info_heading = self.side_column.children[2]
        add_side_column_class = getattr(self.side_column, "add_class", None)
        if add_side_column_class is not None:
            add_side_column_class("mt-plot__side-column")
        add_parameters_panel_class = getattr(self.parameters_panel, "add_class", None)
        if add_parameters_panel_class is not None:
            add_parameters_panel_class("mt-plot__parameters-panel")
        self.info_area.observe(self._sync_info_visibility, names="children")
        self._sync_info_visibility()
        self.content = ipywidgets.Box(
            [self.plot_column, self.side_column],
            layout=ipywidgets.Layout(
                align_items="stretch",
                display="flex",
                flex_flow="row wrap",
                grid_gap="1rem",
                width="100%",
            ),
        )
        self.modal_area.layout = ipywidgets.Layout(width="100%")
        self.overlay_frame = ipywidgets.Box(
            [self.content, self.modal_area],
            layout=ipywidgets.Layout(width="100%"),
        )
        add_overlay_class = getattr(self.overlay_frame, "add_class", None)
        if add_overlay_class is not None:
            add_overlay_class("mt-plot__overlay")
        self.output_area.layout.border = ""
        self.output_area.layout.height = "auto"
        self.output_area.layout.max_height = "12rem"
        self.output_area.layout.min_height = "0"
        self.output_area.layout.overflow = "auto"
        self.output_area.layout.padding = "0"
        self.info_area.layout.border = ""
        self.info_area.layout.padding = "0"
        self.status_area.layout.border = ""
        self.status_area.layout.padding = "0"
        self.root = ipywidgets.VBox(
            [self.overlay_frame, self.output_section],
            layout=ipywidgets.Layout(grid_gap="0.75rem", width="100%"),
        )
        if self.debug_boxes:
            self._apply_debug_boxes()
        return self.root

    def close(self) -> None:
        """Release layout-owned observers and widget resources."""

        controls = getattr(self, "controls_area", None)
        if controls is not None:
            controls.unobserve(self._sync_parameter_visibility, names="children")
        legend = getattr(self, "legend_area", None)
        if legend is not None:
            legend.unobserve(self._sync_legend_visibility, names="children")
        info_area = getattr(self, "info_area", None)
        if info_area is not None:
            info_area.unobserve(self._sync_info_visibility, names="children")
        super().close()

    def _sync_parameter_visibility(self, change: object | None = None) -> None:
        """Show the parameter panel only while parameter controls exist."""

        if self.controls_area.children:
            self.parameters_panel.layout.display = "flex"
        else:
            self.parameters_panel.layout.display = "none"

    def _sync_legend_visibility(self, change: object | None = None) -> None:
        """Show the legend area only while legend rows exist."""

        if self.legend_area.children:
            self.legend_area.layout.display = "flex"
        else:
            self.legend_area.layout.display = "none"

    def _sync_info_visibility(self, change: object | None = None) -> None:
        """Show the info section only while it has visible content."""

        display = "block" if self.info_area.children else "none"
        self.info_heading.layout.display = display
        self.info_area.layout.display = display

    def _apply_debug_boxes(self) -> None:
        """Draw thin red borders around the main layout boxes."""

        for widget in (
            self.root,
            self.overlay_frame,
            self.content,
            self.plot_column,
            self.side_column,
            self.legend_area,
            self.parameters_panel,
            self.controls_area,
            self.info_area,
            self.status_area,
            self.output_section,
            self.output_area,
            self.modal_area,
        ):
            widget.layout.border = "1px solid red"

    def _configure_plot_fill(self) -> None:
        """Configure the Plotly widget to use the available layout box."""

        self.plot_area.layout.width = "100%"
        self.plot_area.layout.min_height = "24rem"
        plot_widget = self.plot_area.children[0]
        plot_widget.layout.autosize = True
        plot_widget.layout.height = None
        plot_widget.layout.width = None
        plot_widget.layout.margin = {
            "l": 0,
            "r": 0,
            "t": 0,
            "b": 0,
        }


DefaultFigureLayout = ResponsiveSidebarLayout


class CustomLayout:
    """Reject the removed HTML-template layout prototype with guidance."""

    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "CustomLayout was a failed HTML-template prototype and is no longer "
            "supported. Define a layout class with build()/detach()/close(), "
            "then register it with FigureHandle.set_layout(...)."
        )


class EventBridgeWidget:
    """Keep the lightweight event bridge used by older experiments."""

    def __init__(self) -> None:
        self._listener = None

    def on_custom_interaction(self, callback_func: object) -> None:
        """Register one Python callback for later simulated events."""

        self._listener = callback_func

    def simulate_frontend_event(self, payload: dict) -> None:
        """Invoke the registered callback with one payload."""

        if self._listener:
            self._listener(payload)
