"""Provide ipywidgets configuration panels for plotted figures."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .errors import PlotSpecError
from .modal import (
    BoolField,
    ColorField,
    DashStyleField,
    FloatField,
    LineWidthField,
    OpacityField,
    PositiveFloatField,
    ReadOnlyTextField,
    SampleCountField,
    TextField,
    no_scroll,
)
from .specs import (
    CartesianView2D,
    CurveView,
    PLOT_KIND_CURVE,
    PLOT_KIND_PARAMETRIC,
    ParametricView,
)

if TYPE_CHECKING:
    import sympy

    from .model import FigureHandle, PlotNode

__all__ = [
    "ParameterConfigPanel",
    "PlotStylePanel",
]
__notebook__ = False


class PlotStylePanel:
    """Edit common plot style metadata through a draft ipywidgets form."""

    title = "Plot Settings"

    def __init__(self, figure: FigureHandle, node: PlotNode) -> None:
        """Create a plot style panel from the current authoritative plot state."""

        import ipywidgets as ipywidgets

        self.figure = figure
        self.node_id = node.id
        self.kind = node.kind
        style = node.style
        visible = node.legend_item_snapshot().visible
        self.name = ReadOnlyTextField("Name", "" if node.name is None else node.name)
        self.label = TextField("Label", node.label)
        self.visible = BoolField("Visible", visible)
        self.line_fields_enabled = node.kind in {PLOT_KIND_CURVE, PLOT_KIND_PARAMETRIC}
        self.sound_fields_enabled = node.kind == PLOT_KIND_CURVE
        self.sample_fields = _sample_fields_for_view(node.view)
        self.color = ColorField("Color", style.get("color", "#1f77b4"))
        self.width = LineWidthField("Width", style.get("width", 2.0))
        self.opacity = OpacityField("Opacity", style.get("opacity", 1.0))
        self.dash = DashStyleField("Dash", style.get("dash", "solid"))
        self.normalization = BoolField(
            "Normalization",
            figure._handle_for_node(node).sound.normalization
            if self.sound_fields_enabled
            else False,
        )

        children = [self.name.widget, self.label.widget, self.visible.widget]
        if self.sample_fields:
            self.sampling_row = ipywidgets.HBox(
                [field.widget for field in self.sample_fields],
                layout=ipywidgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    grid_gap="0.35rem 1.35rem",
                    overflow="hidden",
                    width="100%",
                ),
            )
            for field in self.sample_fields:
                field.widget.layout.flex = "0 1 auto"
                field.widget.layout.width = "auto"
            add_class = getattr(self.sampling_row, "add_class", None)
            if add_class is not None:
                add_class("mt-plot-sampling-row")
            no_scroll(self.sampling_row)
            children.append(self.sampling_row)
        if self.line_fields_enabled:
            self.style_row = ipywidgets.HBox(
                [self.color.widget, self.width.widget],
                layout=ipywidgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    grid_gap="0.35rem 1.35rem",
                    overflow="hidden",
                    width="100%",
                ),
            )
            for widget in (self.color.widget, self.width.widget):
                widget.layout.flex = "0 1 auto"
                widget.layout.width = "auto"
            add_class = getattr(self.style_row, "add_class", None)
            if add_class is not None:
                add_class("mt-plot-style-row")
            no_scroll(self.style_row)
            self.line_style_row = ipywidgets.HBox(
                [self.opacity.widget, self.dash.widget],
                layout=ipywidgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    grid_gap="0.35rem 1.35rem",
                    overflow="hidden",
                    width="100%",
                ),
            )
            for widget in (self.opacity.widget, self.dash.widget):
                widget.layout.flex = "0 1 auto"
                widget.layout.width = "auto"
            add_class = getattr(self.line_style_row, "add_class", None)
            if add_class is not None:
                add_class("mt-plot-line-style-row")
            no_scroll(self.line_style_row)
            children.extend(
                [
                    self.style_row,
                    self.line_style_row,
                ]
            )
            if self.sound_fields_enabled:
                children.append(self.normalization.widget)
        else:
            children.append(
                ipywidgets.HTML(
                    value=(
                        "<p>Style editing for this plot type is not implemented yet.</p>"
                    )
                )
            )
        self.widget = ipywidgets.VBox(
            children,
            layout=ipywidgets.Layout(
                grid_gap="0.5rem",
                overflow="hidden",
                width="100%",
            ),
        )
        no_scroll(self.widget)

    def _ipython_display_(self) -> None:
        """Display the underlying ipywidgets form."""

        from IPython.display import display

        display(self.widget)

    @property
    def children(self) -> tuple[object, ...]:
        """Expose children so the panel can be embedded like an ipywidgets box."""

        return self.widget.children

    def validate(self) -> tuple[str, ...]:
        """Return plot style validation errors without mutating model state."""

        errors: list[str] = []
        for field in self.sample_fields:
            errors.extend(field.errors())
        if self.line_fields_enabled:
            for field in (self.color, self.width, self.opacity):
                errors.extend(field.errors())
        return tuple(errors)

    def apply(self) -> None:
        """Commit draft plot metadata through the public plot handle methods."""

        node = _plot_node_for_id(self.figure, self.node_id)
        if node is None:
            raise PlotSpecError("This plot is no longer present in the active figure.")
        handle = self.figure._handle_for_node(node)
        handle.label = self.label.value
        handle.visible = self.visible.value
        samples = _sample_update_from_fields(self.sample_fields)
        if samples is not None:
            handle.set_samples(samples)
        if self.line_fields_enabled:
            handle.style.update(
                color=self.color.value,
                width=self.width.value,
                opacity=self.opacity.value,
                dash=self.dash.value,
            )
        if self.sound_fields_enabled:
            handle.sound.normalization = self.normalization.value

    def cancel(self) -> None:
        """Close the panel without writing draft state."""

        return None


class ParameterConfigPanel:
    """Edit one parameter's slider metadata through a draft ipywidgets form."""

    title = "Parameter Settings"

    def __init__(
        self,
        figure: FigureHandle,
        *,
        node_id: int,
        symbol: sympy.Symbol,
    ) -> None:
        """Create a parameter panel from the current authoritative state."""

        import ipywidgets as ipywidgets

        state = figure.parameters[symbol]
        metadata = state.metadata
        self.figure = figure
        self.node_id = node_id
        self.symbol = symbol
        self._original_label = metadata.label
        self.label = TextField("Label", "" if metadata.label is None else metadata.label)
        self.value = FloatField("Value", state.value)
        self.minimum = FloatField("Minimum", metadata.minimum)
        self.maximum = FloatField("Maximum", metadata.maximum)
        self.step = PositiveFloatField("Step", metadata.step)
        self.widget = ipywidgets.VBox(
            [
                self.label.widget,
                self.value.widget,
                self.minimum.widget,
                self.maximum.widget,
                self.step.widget,
            ],
            layout=ipywidgets.Layout(
                grid_gap="0.5rem",
                overflow="hidden",
                width="100%",
            ),
        )
        no_scroll(self.widget)

    def _ipython_display_(self) -> None:
        """Display the underlying ipywidgets form."""

        from IPython.display import display

        display(self.widget)

    @property
    def children(self) -> tuple[object, ...]:
        """Expose children so the panel can be embedded like an ipywidgets box."""

        return self.widget.children

    def validate(self) -> tuple[str, ...]:
        """Return parameter validation errors without mutating model state."""

        errors: list[str] = []
        for field in (self.value, self.minimum, self.maximum, self.step):
            errors.extend(field.errors())
        if errors:
            return tuple(errors)

        minimum = self.minimum.value
        maximum = self.maximum.value
        value = self.value.value
        if minimum >= maximum:
            errors.append("Parameter minimum must be less than maximum.")
        if value < minimum or value > maximum:
            errors.append("Parameter value must lie within the slider range.")
        if not math.isfinite(self.step.value) or self.step.value <= 0:
            errors.append("Parameter step must be a finite positive value.")
        return tuple(errors)

    def apply(self) -> None:
        """Commit draft parameter metadata through ``FigureHandle.params``."""

        if self.symbol not in self.figure.parameters:
            raise PlotSpecError("This parameter is no longer present in the active figure.")
        node = _plot_node_for_parameter(self.figure, self.node_id, self.symbol)
        if node is None:
            raise PlotSpecError("This parameter is no longer present in the active figure.")

        label = self.label.value
        if label == "" and self._original_label is None:
            label = None
        self.figure.params = {
            self.symbol: {
                "label": label,
                "value": self.value.value,
                "min": self.minimum.value,
                "max": self.maximum.value,
                "step": self.step.value,
            }
        }

    def cancel(self) -> None:
        """Close the panel without writing draft state."""

        return None


def _plot_node_for_id(figure: FigureHandle, node_id: int) -> PlotNode | None:
    """Return the live plot node with ``node_id`` if it still exists."""

    for node in figure.plots:
        if node.id == node_id:
            return node
    return None


def _plot_node_for_parameter(
    figure: FigureHandle,
    node_id: int,
    symbol: sympy.Symbol,
) -> PlotNode | None:
    """Return a live plot node that can apply a parameter update."""

    for node in figure.plots:
        if node_id not in {0, node.id}:
            continue
        if symbol in node.parameter_symbols:
            return node
    return None


def _sample_fields_for_view(view: object) -> tuple[SampleCountField, ...]:
    """Return draft sample-count fields for one editable plot view."""

    if isinstance(view, CurveView | ParametricView):
        return (SampleCountField("Samples", view.samples),)
    if isinstance(view, CartesianView2D):
        return (
            SampleCountField("X samples", view.x_samples),
            SampleCountField("Y samples", view.y_samples),
        )
    return ()


def _sample_update_from_fields(fields: tuple[SampleCountField, ...]) -> object | None:
    """Return a public samples update from modal fields."""

    if not fields:
        return None
    if len(fields) == 1:
        return fields[0].value
    return fields[0].value, fields[1].value
