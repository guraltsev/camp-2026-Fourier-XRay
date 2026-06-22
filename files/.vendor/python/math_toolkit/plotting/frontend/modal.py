"""Provide modal field specs and apply semantics for the anywidget backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING

from ..errors import PlotSpecError
from ..model import PLOT_NAMED_COLOR_HEX, PLOT_NAMED_COLORS
from ..specs import (
    CartesianView2D,
    CurveView,
    PLOT_KIND_CURVE,
    PLOT_KIND_PARAMETRIC,
    ParametricView,
)

if TYPE_CHECKING:
    import sympy

    from ..display import FigureDisplayGeneration
    from ..model import PlotNode

__all__ = ["AnywidgetModalController"]


@dataclass
class ModalField:
    """Describe one editable modal field."""

    id: str
    label: str
    kind: str
    value: object
    options: tuple[dict[str, str], ...] = ()
    group: str = "General"
    meta: dict[str, object] | None = None

    def payload(self) -> dict[str, object]:
        """Return a JSON-compatible field specification."""

        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "value": self.value,
            "options": list(self.options),
            "group": self.group,
            "meta": {} if self.meta is None else dict(self.meta),
        }


class AnywidgetModalController:
    """Own modal payloads, draft state, validation, and apply actions."""

    def __init__(self, generation: FigureDisplayGeneration, shell: object) -> None:
        """Create a modal controller for one generation."""

        self.generation = generation
        self.shell = shell
        self.state = "closed"
        self.kind: str | None = None
        self.target: dict[str, object] = {}
        self.fields: dict[str, ModalField] = {}
        self.errors: tuple[str, ...] = ()
        self.current_panel: object | None = None
        self._disposed = False
        self.close()

    def open_parameter(self, *, node_id: object, symbol_text: object) -> None:
        """Open a parameter settings modal for a live parameter."""

        if self._disposed or not self.generation.accepts_frontend_events():
            return
        symbol = self._symbol_for_text(str(symbol_text))
        if symbol is None or symbol not in self.generation.figure.parameters:
            return
        state = self.generation.figure.parameters[symbol]
        metadata = state.metadata
        self.kind = "parameter"
        self.target = {"node_id": int(node_id), "symbol": str(symbol)}
        self.fields = {
            "label": ModalField(
                "label",
                "Label",
                "text",
                metadata.label or "",
                group="Identity",
            ),
            "value": ModalField("value", "Value", "float", state.value, group="Range"),
            "minimum": ModalField(
                "minimum",
                "Minimum",
                "float",
                metadata.minimum,
                group="Range",
            ),
            "maximum": ModalField(
                "maximum",
                "Maximum",
                "float",
                metadata.maximum,
                group="Range",
            ),
            "step": ModalField(
                "step",
                "Step",
                "positive_float",
                metadata.step,
                group="Range",
            ),
        }
        self._publish("Parameter Settings")

    def open_plot(self, *, node_id: object) -> None:
        """Open a plot settings modal for a live plot."""

        if self._disposed or not self.generation.accepts_frontend_events():
            return
        node = self._plot_node_for_id(node_id)
        if node is None:
            return
        style = node.style
        self.kind = "plot"
        self.target = {"node_id": node.id}
        fields: dict[str, ModalField] = {
            "name": ModalField(
                "name",
                "Name",
                "readonly",
                "" if node.name is None else node.name,
                group="Identity",
            ),
            "label": ModalField("label", "Label", "text", node.label, group="Identity"),
            "visible": ModalField(
                "visible",
                "Visible",
                "checkbox",
                node.legend_item_snapshot().visible,
                group="Identity",
            ),
        }
        sample_fields = _sample_fields_for_view(node.view)
        fields.update(sample_fields)
        if node.kind in {PLOT_KIND_CURVE, PLOT_KIND_PARAMETRIC}:
            color = style.get("color", "#1f77b4")
            fields.update(
                {
                    "color": ModalField(
                        "color",
                        "Color",
                        "color",
                        color,
                        _color_option_payload(color),
                        group="Line",
                        meta={"picker": _picker_hex_color(color) or "#1f77b4"},
                    ),
                    "width": ModalField(
                        "width",
                        "Width",
                        "line_width",
                        style.get("width", 2.0),
                        group="Line",
                    ),
                    "opacity": ModalField(
                        "opacity",
                        "Opacity",
                        "opacity",
                        style.get("opacity", 1.0),
                        group="Line",
                    ),
                    "dash": ModalField(
                        "dash",
                        "Dash",
                        "dash",
                        style.get("dash", "solid"),
                        _dash_option_payload(),
                        group="Line",
                    ),
                }
            )
        if node.kind == PLOT_KIND_CURVE:
            fields["normalization"] = ModalField(
                "normalization",
                "Normalization",
                "checkbox",
                self.generation.figure._handle_for_node(node).sound.normalization,
                group="Sound",
            )
        self.fields = fields
        self._publish("Plot Settings")

    def update_field(self, field_id: object, value: object) -> None:
        """Update a modal draft field from a frontend event."""

        if self._disposed or self.state != "open":
            return
        field = self.fields.get(str(field_id))
        if field is None:
            return
        if field.kind == "readonly":
            return
        field.value = bool(value) if field.kind == "checkbox" else value
        if field.kind == "color":
            field.options = _color_option_payload(field.value)
            field.meta = {"picker": _picker_hex_color(field.value) or "#1f77b4"}
        self._publish(self._title())

    def apply(self) -> None:
        """Validate and apply the current modal draft."""

        if self._disposed or self.state != "open" or not self.generation.accepts_frontend_events():
            return
        errors = self._validate()
        if errors:
            self.errors = errors
            self._publish(self._title())
            return
        try:
            if self.kind == "parameter":
                self._apply_parameter()
            elif self.kind == "plot":
                self._apply_plot()
        except Exception as exc:  # noqa: BLE001 - user-facing modal diagnostics
            self.errors = (str(exc),)
            self._publish(self._title())
            return
        self.close()

    def close(self) -> None:
        """Close the modal without applying draft state."""

        self.state = "closed"
        self.kind = None
        self.target = {}
        self.fields = {}
        self.errors = ()
        self.current_panel = None
        self.shell.set_modal({"state": "closed"})

    def dispose(self) -> None:
        """Release modal state and prevent future frontend events."""

        self.close()
        self._disposed = True
        self.state = "disposed"

    def _publish(self, title: str) -> None:
        """Publish the current modal payload."""

        self.state = "open"
        self.shell.set_modal(
            {
                "state": "open",
                "title": title,
                "fields": [field.payload() for field in self.fields.values()],
                "errors": list(self.errors),
            }
        )

    def _title(self) -> str:
        """Return the active modal title."""

        return "Parameter Settings" if self.kind == "parameter" else "Plot Settings"

    def _validate(self) -> tuple[str, ...]:
        """Return validation errors for the current modal draft."""

        errors: list[str] = []
        for field in self.fields.values():
            if field.kind in {"float", "positive_float", "opacity", "line_width"}:
                value = _finite_float(field.value)
                if value is None:
                    errors.append(f"{field.label} must be a finite number.")
                elif field.kind in {"positive_float", "line_width"} and value <= 0:
                    errors.append(f"{field.label} must be a finite positive value.")
                elif field.kind == "opacity" and (value < 0 or value > 1):
                    errors.append(f"{field.label} must be between 0 and 1.")
            if field.kind == "sample_count":
                value = _integer(field.value)
                if value is None:
                    errors.append(f"{field.label} must be an integer.")
                elif value <= 1:
                    errors.append(f"{field.label} must be greater than 1.")
            if field.kind == "color" and str(field.value).strip() == "":
                errors.append("Color must not be empty.")
        if self.kind == "parameter" and not errors:
            minimum = float(self.fields["minimum"].value)
            maximum = float(self.fields["maximum"].value)
            value = float(self.fields["value"].value)
            if minimum >= maximum:
                errors.append("Parameter minimum must be less than maximum.")
            if value < minimum or value > maximum:
                errors.append("Parameter value must lie within the slider range.")
        return tuple(errors)

    def _apply_parameter(self) -> None:
        """Commit a validated parameter draft to model state."""

        symbol = self._symbol_for_text(str(self.target.get("symbol", "")))
        if symbol is None or symbol not in self.generation.figure.parameters:
            raise PlotSpecError("This parameter is no longer present in the active figure.")
        if self._plot_node_for_parameter(int(self.target.get("node_id", 0)), symbol) is None:
            raise PlotSpecError("This parameter is no longer present in the active figure.")
        label = str(self.fields["label"].value)
        self.generation.figure.params = {
            symbol: {
                "label": label if label != "" else None,
                "value": float(self.fields["value"].value),
                "min": float(self.fields["minimum"].value),
                "max": float(self.fields["maximum"].value),
                "step": float(self.fields["step"].value),
            }
        }

    def _apply_plot(self) -> None:
        """Commit a validated plot draft to model state."""

        node = self._plot_node_for_id(self.target.get("node_id"))
        if node is None:
            raise PlotSpecError("This plot is no longer present in the active figure.")
        handle = self.generation.figure._handle_for_node(node)
        handle.label = str(self.fields["label"].value)
        handle.visible = bool(self.fields["visible"].value)
        samples = self._sample_update()
        if samples is not None:
            handle.set_samples(samples)
        if "color" in self.fields:
            handle.style.update(
                color=str(self.fields["color"].value),
                width=float(self.fields["width"].value),
                opacity=float(self.fields["opacity"].value),
                dash=str(self.fields["dash"].value),
            )
        if "normalization" in self.fields:
            handle.sound.normalization = bool(self.fields["normalization"].value)

    def _sample_update(self) -> object | None:
        """Return the plot sample update represented by the modal fields."""

        if "samples" in self.fields:
            return int(self.fields["samples"].value)
        if "x_samples" in self.fields and "y_samples" in self.fields:
            return int(self.fields["x_samples"].value), int(self.fields["y_samples"].value)
        return None

    def _plot_node_for_id(self, node_id: object) -> PlotNode | None:
        """Return a live plot node by id."""

        try:
            wanted = int(node_id)
        except (TypeError, ValueError):
            return None
        for node in self.generation.figure.plots:
            if node.id == wanted:
                return node
        return None

    def _plot_node_for_parameter(
        self,
        node_id: int,
        symbol: sympy.Symbol,
    ) -> object | None:
        """Return a live plot node that owns a parameter control."""

        for node in self.generation.figure.plots:
            if node_id not in {0, node.id}:
                continue
            if symbol in node.parameter_symbols:
                return node
        return None

    def _symbol_for_text(self, symbol_text: str) -> sympy.Symbol | None:
        """Return a live parameter symbol by frontend string."""

        for symbol in self.generation.figure.parameters:
            if str(symbol) == symbol_text:
                return symbol
        return None


def _sample_fields_for_view(view: object) -> dict[str, ModalField]:
    """Return sample-count modal fields for a plot view."""

    if isinstance(view, CurveView | ParametricView):
        return {
            "samples": ModalField(
                "samples",
                "Samples",
                "sample_count",
                view.samples,
                group="Sampling",
            )
        }
    if isinstance(view, CartesianView2D):
        return {
            "x_samples": ModalField(
                "x_samples",
                "X samples",
                "sample_count",
                view.x_samples,
                group="Sampling",
            ),
            "y_samples": ModalField(
                "y_samples",
                "Y samples",
                "sample_count",
                view.y_samples,
                group="Sampling",
            ),
        }
    return {}


def _dash_option_payload() -> tuple[dict[str, str], ...]:
    """Return line-dash options with SVG preview metadata."""

    return (
        {"label": "solid", "value": "solid", "dasharray": ""},
        {"label": "dot", "value": "dot", "dasharray": "1 4"},
        {"label": "dash", "value": "dash", "dasharray": "6 4"},
        {"label": "longdash", "value": "longdash", "dasharray": "11 4"},
        {"label": "dashdot", "value": "dashdot", "dasharray": "6 4 1 4"},
        {"label": "longdashdot", "value": "longdashdot", "dasharray": "11 4 1 4"},
    )


def _color_option_payload(selected: object) -> tuple[dict[str, str], ...]:
    """Return named color options with a custom HEX option when needed."""

    text = str(selected).strip()
    options = tuple(
        {"label": color, "value": color, "hex": PLOT_NAMED_COLOR_HEX[color]}
        for color in PLOT_NAMED_COLORS
    )
    if text in PLOT_NAMED_COLORS:
        return options
    return (
        {
            "label": text,
            "value": text,
            "hex": _picker_hex_color(text) or "#1f77b4",
        },
        *options,
    )


def _picker_hex_color(value: object) -> str | None:
    """Return a HEX color that can seed the frontend color picker."""

    text = str(value).strip()
    if text in PLOT_NAMED_COLOR_HEX:
        return PLOT_NAMED_COLOR_HEX[text]
    return _normalize_hex(text)


def _normalize_hex(value: object) -> str | None:
    """Return a normalized HEX color or ``None`` for non-HEX values."""

    text = str(value).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        return "#" + "".join(character * 2 for character in text[1:].lower())
    return None


def _finite_float(value: object) -> float | None:
    """Return a finite float or ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _integer(value: object) -> int | None:
    """Return an integer or ``None``."""

    if isinstance(value, bool):
        return None
    try:
        text = str(value)
        number = int(text)
    except (TypeError, ValueError):
        return None
    if str(number) != text.strip():
        return None
    return number
