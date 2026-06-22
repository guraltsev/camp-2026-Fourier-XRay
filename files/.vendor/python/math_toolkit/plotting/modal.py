"""Provide ipywidgets modal shells and typed configuration fields."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .model import PLOT_NAMED_COLOR_HEX, PLOT_NAMED_COLORS

if TYPE_CHECKING:
    from .display import FigureDisplayGeneration

__all__ = [
    "BoolField",
    "ChoiceField",
    "ColorField",
    "ConfigPanel",
    "DashStyleField",
    "FloatField",
    "IntegerField",
    "LineWidthField",
    "ModalController",
    "OpacityField",
    "PositiveFloatField",
    "ReadOnlyTextField",
    "SampleCountField",
    "TextField",
    "no_scroll",
]
__notebook__ = False


class ConfigPanel:
    """Define the minimal interface implemented by modal configuration panels."""

    title = "Configuration"

    def validate(self) -> tuple[str, ...]:
        """Return validation errors without mutating model state."""

        return ()

    def apply(self) -> None:
        """Write validated draft state to authoritative model state."""

        return None

    def cancel(self) -> None:
        """Release temporary panel state without writing model state."""

        return None


def no_scroll(widget: object) -> object:
    """Mark an ipywidgets container as scrollbar-free without clipping children."""

    layout = getattr(widget, "layout", None)
    if layout is not None:
        layout.overflow = "visible"
    add_class = getattr(widget, "add_class", None)
    if add_class is not None:
        add_class("mt-no-scroll")
    return widget


class ModalController:
    """Own one generation's modal shell and guarded Apply/Cancel callbacks."""

    def __init__(
        self,
        generation: FigureDisplayGeneration,
        container: object,
    ) -> None:
        """Create a modal controller attached to a generation-owned container."""

        import ipywidgets as ipywidgets

        self.generation = generation
        self.container = container
        self.state = "closed"
        self.current_panel: object | None = None
        self._disposed = False

        self.title_widget = ipywidgets.HTML(value="")
        self.message_widget = ipywidgets.HTML(value="")
        self.body = ipywidgets.VBox()
        self.apply_button = ipywidgets.Button(
            description="Apply",
            button_style="primary",
            tooltip="Apply changes",
            layout=ipywidgets.Layout(width="auto"),
        )
        self.cancel_button = ipywidgets.Button(
            description="Cancel",
            tooltip="Close without applying changes",
            layout=ipywidgets.Layout(width="auto"),
        )
        self.close_button = ipywidgets.Button(
            description="Close",
            tooltip="Close without applying changes",
            layout=ipywidgets.Layout(width="auto"),
        )
        self.header = ipywidgets.HBox(
            [self.title_widget, self.close_button],
            layout=ipywidgets.Layout(
                align_items="center",
                justify_content="space-between",
                overflow="hidden",
                width="100%",
            ),
        )
        self.footer = ipywidgets.HBox(
            [self.cancel_button, self.apply_button],
            layout=ipywidgets.Layout(
                grid_gap="0.5rem",
                justify_content="flex-end",
                overflow="hidden",
                width="100%",
            ),
        )
        self.dialog = ipywidgets.VBox(
            [self.header, self.body, self.message_widget, self.footer],
            layout=ipywidgets.Layout(
                border="1px solid #cbd5e1",
                overflow="hidden",
                padding="0.75rem",
                width="min(34rem, 100%)",
            ),
        )
        self.root = ipywidgets.Box(
            [self.dialog],
            layout=ipywidgets.Layout(
                display="none",
                justify_content="center",
                overflow="hidden",
                width="100%",
            ),
        )

        for widget, class_name in (
            (self.root, "mt-modal"),
            (self.dialog, "mt-modal__dialog"),
            (self.header, "mt-modal__header"),
            (self.title_widget, "mt-modal__title"),
            (self.close_button, "mt-modal__close"),
            (self.body, "mt-modal__body"),
            (self.message_widget, "mt-modal__message"),
            (self.footer, "mt-modal__footer"),
            (self.apply_button, "mt-modal__apply"),
            (self.cancel_button, "mt-modal__cancel"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)
        self.apply_button.on_click(self._apply_clicked)
        self.cancel_button.on_click(self._cancel_clicked)
        self.close_button.on_click(self._cancel_clicked)
        self.container.children = (self.root,)

    def open(self, panel: ConfigPanel) -> None:
        """Show a fresh panel in this generation's modal shell."""

        if self._disposed or not self.generation.accepts_frontend_events():
            return
        self.close()
        self.current_panel = panel
        self.title_widget.value = _escape_html(getattr(panel, "title", "Configuration"))
        panel_widget = getattr(panel, "widget", panel)
        self.body.children = (panel_widget,)
        self.message_widget.value = ""
        self.root.layout.display = "flex"
        self.state = "open"

    def close(self) -> None:
        """Close the modal shell without applying draft state."""

        if self._disposed or self.state == "closed":
            return
        panel = self.current_panel
        if panel is not None:
            cancel = getattr(panel, "cancel", None)
            if callable(cancel):
                cancel()
        self.current_panel = None
        self.body.children = ()
        self.message_widget.value = ""
        self.root.layout.display = "none"
        self.state = "closed"

    def dispose(self) -> None:
        """Detach modal callbacks and release shell widget references."""

        if self._disposed:
            return
        self.close()
        self._disposed = True
        self.state = "disposed"
        self.apply_button.on_click(self._apply_clicked, remove=True)
        self.cancel_button.on_click(self._cancel_clicked, remove=True)
        self.close_button.on_click(self._cancel_clicked, remove=True)
        self.current_panel = None
        self.body.children = ()
        self.message_widget.value = ""
        self.container.children = ()
        self.root.children = ()

    def _apply_clicked(self, _button: object) -> None:
        """Validate and apply the active panel under generation liveness guards."""

        if (
            self._disposed
            or self.state != "open"
            or self.current_panel is None
            or not self.generation.accepts_frontend_events()
        ):
            return

        panel = self.current_panel
        errors = tuple(panel.validate())
        if errors:
            self.state = "error"
            self._show_errors(errors)
            self.state = "open"
            return

        self.state = "applying"
        try:
            panel.apply()
        except Exception as exc:  # noqa: BLE001 - user-facing modal diagnostics
            self.state = "error"
            self._show_errors((str(exc),))
            self.state = "open"
            return
        self.state = "open"
        self.close()

    def _cancel_clicked(self, _button: object) -> None:
        """Close the modal if this generation is still allowed to respond."""

        if self._disposed:
            return
        if not self.generation.accepts_frontend_events():
            return
        self.close()

    def _show_errors(self, errors: tuple[str, ...]) -> None:
        """Render validation errors inside the modal shell."""

        items = "".join(f"<li>{_escape_html(error)}</li>" for error in errors)
        self.message_widget.value = (
            '<div class="mt-modal__message-error"><ul>' f"{items}</ul></div>"
        )


class TextField:
    """Wrap an ipywidgets text control with a visible label."""

    def __init__(self, label: str, value: object = "") -> None:
        """Create a text field with a normalized string value."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.Text(
            value=str(value),
            description="",
            continuous_update=False,
            layout=ipywidgets.Layout(width="100%"),
            style={"description_width": "0"},
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> str:
        """Return the current text value."""

        return str(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return text-field validation errors."""

        return ()


class ReadOnlyTextField:
    """Wrap a disabled text control with a visible label."""

    def __init__(self, label: str, value: object = "") -> None:
        """Create a non-editable text field with a normalized string value."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.Text(
            value=str(value),
            description="",
            disabled=True,
            layout=ipywidgets.Layout(width="100%"),
            style={"description_width": "0"},
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> str:
        """Return the displayed text value."""

        return str(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return read-only field validation errors."""

        return ()


class FloatField:
    """Wrap a finite floating-point editor with validation helpers."""

    def __init__(self, label: str, value: object) -> None:
        """Create a float field with a draft numeric value."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.FloatText(
            value=float(value),
            description="",
            layout=ipywidgets.Layout(width="100%"),
            style={"description_width": "0"},
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> float:
        """Return the current float value."""

        return float(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return finite-number validation errors."""

        if not math.isfinite(self.value):
            return (f"{self.label} must be a finite number.",)
        return ()


class IntegerField:
    """Wrap an integer editor with validation helpers."""

    def __init__(self, label: str, value: object) -> None:
        """Create an integer field with a draft value."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.IntText(
            value=int(value),
            description="",
            layout=ipywidgets.Layout(width="100%"),
            style={"description_width": "0"},
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> int:
        """Return the current integer value."""

        return int(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return integer validation errors."""

        if isinstance(self.control.value, bool):
            return (f"{self.label} must be an integer.",)
        return ()


class SampleCountField(IntegerField):
    """Wrap a plot sample-count editor constrained to counts above one."""

    def errors(self) -> tuple[str, ...]:
        """Return sample-count validation errors."""

        errors = list(super().errors())
        if not errors and self.value <= 1:
            errors.append(f"{self.label} must be greater than 1.")
        return tuple(errors)


class PositiveFloatField(FloatField):
    """Wrap a finite positive floating-point editor."""

    def errors(self) -> tuple[str, ...]:
        """Return finite-positive validation errors."""

        errors = list(super().errors())
        if not errors and self.value <= 0:
            errors.append(f"{self.label} must be a finite positive value.")
        return tuple(errors)


class LineWidthField(PositiveFloatField):
    """Wrap a positive line-width editor with a visual stroke preview."""

    def __init__(self, label: str, value: object) -> None:
        """Create a line-width field that previews the current stroke size."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.BoundedFloatText(
            value=float(value),
            min=0.0,
            continuous_update=True,
            description="",
            layout=ipywidgets.Layout(width="3.5rem"),
            style={"description_width": "0"},
        )
        self.preview = ipywidgets.HTML(
            value=self._preview_html(),
            layout=ipywidgets.Layout(overflow="hidden", width="2em"),
        )
        self.suffix = ipywidgets.HTML(
            value='<span class="mt-line-width-field__suffix">px</span>',
            layout=ipywidgets.Layout(overflow="hidden", width="auto"),
        )
        body = ipywidgets.HBox(
            [self.preview, self.control, self.suffix],
            layout=ipywidgets.Layout(
                align_items="center",
                grid_gap="0.25rem",
                overflow="hidden",
                width="auto",
            ),
        )
        self.widget = _field_widget(label, body)
        no_scroll(body)
        for widget, class_name in (
            (body, "mt-line-width-field"),
            (self.preview, "mt-line-width-field__preview"),
            (self.control, "mt-line-width-field__entry"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)
        self.control.observe(self._control_changed, names="value")

    def _control_changed(self, _change: dict[str, object]) -> None:
        """Refresh the visual preview after draft width edits."""

        self.preview.value = self._preview_html()

    def _preview_html(self) -> str:
        """Return an SVG preview scaled to the edited line width."""

        try:
            value = self.value
        except (TypeError, ValueError):
            value = 0.0
        preview_width = min(max(value, 0.5), 12.0)
        return (
            '<div class="mt-line-width-field__sample" aria-label="Line width preview">'
            '<svg viewBox="0 0 32 20" role="img" focusable="false">'
            '<line x1="2" y1="10" x2="30" y2="10" '
            f'stroke="#2563eb" stroke-width="{preview_width:g}" '
            'stroke-linecap="round"/>'
            "</svg>"
            "</div>"
        )


class OpacityField:
    """Wrap an opacity slider and numeric editor constrained to ``[0, 1]``."""

    def __init__(self, label: str, value: object) -> None:
        """Create a synchronized opacity slider and bounded numeric entry."""

        import ipywidgets as ipywidgets

        self.label = label
        initial = min(max(float(value), 0.0), 1.0)
        self._syncing_opacity_controls = False
        self.slider = ipywidgets.FloatSlider(
            value=initial,
            min=0.0,
            max=1.0,
            step=0.1,
            readout=False,
            continuous_update=True,
            description="",
            layout=ipywidgets.Layout(flex="0 0 5.5rem", min_width="0", width="5.5rem"),
            style={"description_width": "0"},
        )
        self.control = ipywidgets.Text(
            value=_format_unit_float(initial),
            description="",
            continuous_update=True,
            layout=ipywidgets.Layout(width="2.8rem"),
            style={"description_width": "0"},
        )
        body = ipywidgets.HBox(
            [self.slider, self.control],
            layout=ipywidgets.Layout(
                align_items="center",
                grid_gap="0.35rem",
                overflow="hidden",
                width="auto",
            ),
        )
        self.widget = _field_widget(label, body)
        no_scroll(body)
        for widget, class_name in (
            (body, "mt-opacity-field"),
            (self.slider, "mt-opacity-field__slider"),
            (self.control, "mt-opacity-field__entry"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)
        self.slider.observe(self._slider_changed, names="value")
        self.control.observe(self._control_changed, names="value")

    @property
    def value(self) -> float:
        """Return the current opacity value."""

        return float(str(self.control.value).strip())

    def _slider_changed(self, change: dict[str, object]) -> None:
        """Mirror slider movement into the numeric draft entry."""

        if self._syncing_opacity_controls:
            return
        self._syncing_opacity_controls = True
        try:
            self.control.value = _format_unit_float(change["new"])
        finally:
            self._syncing_opacity_controls = False

    def _control_changed(self, change: dict[str, object]) -> None:
        """Mirror numeric entry edits into the visual slider."""

        if self._syncing_opacity_controls:
            return
        self._syncing_opacity_controls = True
        try:
            try:
                value = float(str(change["new"]).strip())
            except ValueError:
                return
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                self.slider.value = value
                formatted = _format_unit_float(value)
                if self.control.value != formatted:
                    self.control.value = formatted
        finally:
            self._syncing_opacity_controls = False

    def errors(self) -> tuple[str, ...]:
        """Return opacity validation errors."""

        try:
            value = self.value
        except ValueError:
            return (f"{self.label} must be a finite number.",)
        if not math.isfinite(value):
            return (f"{self.label} must be a finite number.",)
        if value < 0 or value > 1:
            return (f"{self.label} must be between 0 and 1.",)
        return ()


class BoolField:
    """Wrap a checkbox with a visible label."""

    def __init__(self, label: str, value: object) -> None:
        """Create a boolean field."""

        import ipywidgets as ipywidgets

        self.label = label
        self.control = ipywidgets.Checkbox(
            value=bool(value),
            description="",
            indent=False,
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> bool:
        """Return the current boolean value."""

        return bool(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return boolean-field validation errors."""

        return ()


class ColorField:
    """Wrap direct custom picker and named color controls with one draft value."""

    def __init__(self, label: str, value: object = "") -> None:
        """Create a color field with a picker square and named-color dropdown."""

        import ipywidgets as ipywidgets

        self.label = label
        initial = str(value).strip()
        picker_value = _initial_hex_color(initial)
        selected = initial if initial in PLOT_NAMED_COLORS else picker_value
        self._syncing_color_controls = False
        self.picker = ipywidgets.ColorPicker(
            concise=True,
            value=picker_value,
            description="",
            layout=ipywidgets.Layout(flex="0 0 2.35rem", width="2.35rem"),
            style={"description_width": "0"},
        )
        self.named = ipywidgets.Dropdown(
            options=_color_options(selected),
            value=selected,
            description="",
            layout=ipywidgets.Layout(flex="0 0 11ch", min_width="0", width="11ch"),
            style={"description_width": "0"},
        )
        body = ipywidgets.HBox(
            [self.picker, self.named],
            layout=ipywidgets.Layout(
                align_items="center",
                grid_gap="0.25rem",
                overflow="hidden",
                width="auto",
            ),
        )
        self.widget = _field_widget(label, body)
        no_scroll(body)
        for widget, class_name in (
            (body, "mt-color-field"),
            (self.picker, "mt-color-field__picker"),
            (self.named, "mt-color-field__named"),
        ):
            add_class = getattr(widget, "add_class", None)
            if add_class is not None:
                add_class(class_name)

        self.picker.observe(self._picker_changed, names="value")
        self.named.observe(self._named_changed, names="value")

    @property
    def value(self) -> str:
        """Return the selected color value."""

        return str(self.named.value).strip()

    def errors(self) -> tuple[str, ...]:
        """Return color-field validation errors."""

        if self.value == "":
            return ("Color must not be empty.",)
        return ()

    def _picker_changed(self, change: dict[str, object]) -> None:
        """Use picker changes as custom HEX colors."""

        value = str(change["new"]).strip()
        if not value:
            return
        if self._syncing_color_controls:
            return
        self._syncing_color_controls = True
        try:
            self.named.options = _color_options(value)
            self.named.value = value
        finally:
            self._syncing_color_controls = False

    def _named_changed(self, change: dict[str, object]) -> None:
        """Mirror dropdown choices back into the picker square."""

        value = str(change["new"]).strip()
        normalized = _picker_hex_color(value)
        if normalized is None or self.picker.value == normalized:
            return
        self._syncing_color_controls = True
        try:
            self.picker.value = normalized
        finally:
            self._syncing_color_controls = False


class ChoiceField:
    """Wrap a dropdown for a finite option set."""

    def __init__(
        self,
        label: str,
        options: tuple[str, ...] | tuple[tuple[str, str], ...],
        value: object,
    ) -> None:
        """Create a choice field with a fallback to the first option."""

        import ipywidgets as ipywidgets

        normalized_options = _choice_options(options)
        self.label = label
        selected = str(value)
        values = tuple(option_value for _option_label, option_value in normalized_options)
        if selected not in values:
            selected = values[0]
        self.control = ipywidgets.Dropdown(
            options=normalized_options,
            value=selected,
            description="",
            layout=ipywidgets.Layout(width="100%"),
            style={"description_width": "0"},
        )
        self.widget = _field_widget(label, self.control)

    @property
    def value(self) -> str:
        """Return the selected choice value."""

        return str(self.control.value)

    def errors(self) -> tuple[str, ...]:
        """Return choice-field validation errors."""

        return ()


_DASH_STYLE_OPTIONS = (
    ("solid", "solid", ""),
    ("dot", "dot", "1 4"),
    ("dash", "dash", "6 4"),
    ("longdash", "longdash", "11 4"),
    ("dashdot", "dashdot", "6 4 1 4"),
    ("longdashdot", "longdashdot", "11 4 1 4"),
)


def _dash_select_base_class() -> type[object]:
    """Return the installed anywidget base class for the dash selector."""

    try:
        import anywidget
    except ImportError as exc:
        raise RuntimeError(
            "Dash style editing requires anywidget so the modal can render "
            "SVG options and sync the selected value."
        ) from exc
    return anywidget.AnyWidget


class DashStyleSelectWidget(_dash_select_base_class()):
    """Render a compact SVG listbox for Plotly dash styles."""

    import traitlets

    value = traitlets.Unicode("solid").tag(sync=True)

    _css = r"""
.mt-dash-select {
  color: #0f172a;
  font: 0.85rem/1.2 system-ui, sans-serif;
  position: relative;
  width: 100%;
}
.mt-dash-select,
.mt-dash-select * {
  box-sizing: border-box;
}
.mt-dash-select__trigger {
  align-items: center;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 0.2rem;
  color: inherit;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  min-height: 1.8rem;
  padding: 0.18rem 0.4rem;
  width: 100%;
}
.mt-dash-select__trigger:focus-visible {
  outline: 2px solid #3b82f6;
  outline-offset: 1px;
}
.mt-dash-select__value,
.mt-dash-select__option {
  align-items: center;
  display: flex;
  gap: 0.45rem;
  min-width: 0;
}
.mt-dash-select__preview {
  flex: 0 0 2.75em;
  height: 0.8em;
  max-width: 3em;
  width: 2.75em;
}
.mt-dash-select__text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mt-dash-select__chevron {
  border-left: 0.3rem solid transparent;
  border-right: 0.3rem solid transparent;
  border-top: 0.4rem solid currentColor;
  flex: 0 0 auto;
  margin-left: 0.5rem;
}
.mt-dash-select__menu {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 0.2rem;
  box-shadow: 0 0.35rem 0.8rem rgba(15, 23, 42, 0.12);
  list-style: none;
  margin: 0;
  overflow: visible;
  padding: 0.2rem;
  position: fixed;
  z-index: 2147483647;
}
.mt-dash-select__menu[hidden] {
  display: none;
}
.mt-dash-select__option {
  border-radius: 0.2rem;
  cursor: pointer;
  padding: 0.35rem 0.45rem;
}
.mt-dash-select__option[aria-selected="true"] {
  background: #dbeafe;
  color: #1d4ed8;
  font-weight: 600;
}
.mt-dash-select__option:hover,
.mt-dash-select__option:focus {
  background: #eff6ff;
  outline: none;
}
"""

    _esm = r"""
const OPTIONS = [
  { label: "solid", value: "solid", dasharray: "" },
  { label: "dot", value: "dot", dasharray: "1 4" },
  { label: "dash", value: "dash", dasharray: "6 4" },
  { label: "longdash", value: "longdash", dasharray: "11 4" },
  { label: "dashdot", value: "dashdot", dasharray: "6 4 1 4" },
  { label: "longdashdot", value: "longdashdot", dasharray: "11 4 1 4" },
];

function optionFor(value) {
  return OPTIONS.find((option) => option.value === value) || OPTIONS[0];
}

function dashSvg(option) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "mt-dash-select__preview");
  svg.setAttribute("viewBox", "0 0 48 16");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", "3");
  line.setAttribute("y1", "8");
  line.setAttribute("x2", "45");
  line.setAttribute("y2", "8");
  line.setAttribute("stroke", "currentColor");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("stroke-linecap", "round");
  if (option.dasharray) {
    line.setAttribute("stroke-dasharray", option.dasharray);
  }
  svg.appendChild(line);
  return svg;
}

function valueContent(option) {
  const value = document.createElement("span");
  value.className = "mt-dash-select__value";
  value.appendChild(dashSvg(option));

  const text = document.createElement("span");
  text.className = "mt-dash-select__text";
  text.textContent = option.label;
  value.appendChild(text);
  return value;
}

function render({ model, el }) {
  el.classList.add("mt-dash-select");
  el.dataset.select = "";
  el.innerHTML = "";

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "mt-dash-select__trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");

  const selectedSlot = document.createElement("span");
  selectedSlot.className = "mt-dash-select__selected";
  const chevron = document.createElement("span");
  chevron.className = "mt-dash-select__chevron";
  chevron.setAttribute("aria-hidden", "true");
  trigger.append(selectedSlot, chevron);

  const menu = document.createElement("ul");
  menu.className = "mt-dash-select__menu";
  menu.setAttribute("role", "listbox");
  menu.tabIndex = -1;
  menu.hidden = true;

  const hidden = document.createElement("input");
  hidden.type = "hidden";
  hidden.name = "dash";

  function close() {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
  }

  function positionMenu() {
    const rect = trigger.getBoundingClientRect();
    menu.style.left = `${rect.left}px`;
    menu.style.top = `${rect.bottom + 3}px`;
    menu.style.minWidth = `${rect.width}px`;
  }

  function open() {
    positionMenu();
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
    const selected = menu.querySelector('[aria-selected="true"]');
    if (selected) {
      selected.focus();
    }
  }

  function setValue(value) {
    const option = optionFor(value);
    model.set("value", option.value);
    model.save_changes();
  }

  function sync() {
    const selected = optionFor(model.get("value"));
    selectedSlot.replaceChildren(valueContent(selected));
    hidden.value = selected.value;
    for (const item of menu.querySelectorAll("[role='option']")) {
      item.setAttribute(
        "aria-selected",
        item.dataset.value === selected.value ? "true" : "false"
      );
    }
  }

  for (const option of OPTIONS) {
    const item = document.createElement("li");
    item.className = "mt-dash-select__option";
    item.dataset.value = option.value;
    item.setAttribute("role", "option");
    item.tabIndex = -1;
    item.appendChild(dashSvg(option));

    const text = document.createElement("span");
    text.className = "mt-dash-select__text";
    text.textContent = option.label;
    item.appendChild(text);

    item.addEventListener("click", () => {
      setValue(option.value);
      close();
      trigger.focus();
    });
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setValue(option.value);
        close();
        trigger.focus();
      }
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        trigger.focus();
      }
    });
    menu.appendChild(item);
  }

  trigger.addEventListener("click", () => {
    if (menu.hidden) {
      open();
      return;
    }
    close();
  });
  trigger.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
    if (event.key === "Escape") {
      close();
    }
  });

  const outsideClick = (event) => {
    if (!el.contains(event.target) && !menu.contains(event.target)) {
      close();
    }
  };
  const reposition = () => {
    if (!menu.hidden) {
      positionMenu();
    }
  };
  document.addEventListener("click", outsideClick);
  window.addEventListener("resize", reposition);
  window.addEventListener("scroll", reposition, true);
  model.on("change:value", sync);

  el.append(trigger, hidden);
  document.body.appendChild(menu);
  sync();

  return () => {
    document.removeEventListener("click", outsideClick);
    window.removeEventListener("resize", reposition);
    window.removeEventListener("scroll", reposition, true);
    model.off("change:value", sync);
    menu.remove();
  };
}

export default { render };
"""


class DashStyleField:
    """Wrap Plotly line-dash choices with a custom SVG select widget."""

    def __init__(self, label: str, value: object) -> None:
        """Create a dash-style field with a synced custom widget."""

        import ipywidgets as ipywidgets

        self.label = label
        self._values = tuple(
            option_value
            for _label, option_value, _dasharray in _DASH_STYLE_OPTIONS
        )
        selected = str(value)
        if selected not in self._values:
            selected = self._values[0]

        self.control = DashStyleSelectWidget(value=selected)
        self.control.layout = ipywidgets.Layout(
            flex="0 1 9.5rem",
            min_width="0",
            width="9.5rem",
        )
        self.widget = _field_widget(label, self.control)
        self.widget.layout.width = "auto"
        self.widget.layout.flex = "0 1 auto"
        no_scroll(self.widget)

    @property
    def value(self) -> str:
        """Return the selected dash style."""

        return str(self.control.value)

    @value.setter
    def value(self, value: object) -> None:
        """Select a dash style in the synced widget."""

        selected = str(value)
        if selected not in self._values:
            selected = self._values[0]
        self.control.value = selected

    def errors(self) -> tuple[str, ...]:
        """Return dash-style validation errors."""

        return ()


def _color_options(selected: object) -> tuple[tuple[str, str], ...]:
    """Return dropdown options with custom HEX first when selected."""

    text = str(selected).strip()
    options = tuple((color, color) for color in PLOT_NAMED_COLORS)
    if text in PLOT_NAMED_COLORS:
        return options
    return ((text, text),) + options


def _choice_options(
    options: tuple[str, ...] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Return ipywidgets dropdown options as ``(label, value)`` pairs."""

    normalized: list[tuple[str, str]] = []
    for option in options:
        if isinstance(option, tuple):
            normalized.append((str(option[0]), str(option[1])))
        else:
            normalized.append((str(option), str(option)))
    return tuple(normalized)


def _format_unit_float(value: object) -> str:
    """Return a one-decimal value for compact unit-interval editors."""

    return f"{float(value):.1f}"


def _initial_hex_color(value: object) -> str:
    """Return an existing HEX color or a stable default custom color."""

    normalized = _picker_hex_color(str(value).strip())
    if normalized is not None:
        return normalized
    return "#1f77b4"


def _picker_hex_color(value: object) -> str | None:
    """Return a HEX color that can be displayed in the picker square."""

    text = str(value).strip()
    if text in PLOT_NAMED_COLOR_HEX:
        return PLOT_NAMED_COLOR_HEX[text]
    return _normalize_hex(text)


def _normalize_hex(value: object) -> str | None:
    """Return a normalized HEX color or ``None`` for invalid text."""

    text = str(value).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.lower()
    if re.fullmatch(r"#[0-9a-fA-F]{3}", text):
        return "#" + "".join(character * 2 for character in text[1:].lower())
    return None


def _field_widget(label: str, control: object) -> object:
    """Return a compact labeled ipywidgets field row."""

    import ipywidgets as ipywidgets

    label_widget = ipywidgets.HTML(
        value=f"<span>{_escape_html(label)}</span>",
        layout=ipywidgets.Layout(flex="0 0 auto", overflow="hidden", width="auto"),
    )
    row = ipywidgets.HBox(
        [label_widget, control],
        layout=ipywidgets.Layout(
            align_items="center",
            grid_gap="0.25rem",
            overflow="hidden",
            width="100%",
        ),
    )
    for widget, class_name in (
        (row, "mt-config-field"),
        (label_widget, "mt-config-field__label"),
        (control, "mt-config-field__control"),
    ):
        add_class = getattr(widget, "add_class", None)
        if add_class is not None:
            add_class(class_name)
    no_scroll(row)
    return row


def _escape_html(value: object) -> str:
    """Return a minimal escaped HTML text fragment."""

    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
