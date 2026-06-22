"""Reconcile ipywidgets controls for plotted parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math
from typing import TYPE_CHECKING

from ._reactive import untracked

_VALUE_TEXT_WIDTH_CHARS = 7

if TYPE_CHECKING:
    import sympy

    from .display import FigureDisplayGeneration
    from .model import ControlLayoutItem, SliderValueItem


def reconcile_parameter_controls(
    generation: FigureDisplayGeneration,
    layout: tuple[ControlLayoutItem, ...],
) -> None:
    """Reconcile slider widgets from layout metadata without value tracking."""

    wanted_keys = {(item.node_id, item.symbol) for item in layout}

    # Remove controls whose plot node or parameter symbol disappeared before
    # composing the next children tuple. Surviving sliders remain the same
    # Python widget objects, preserving browser-side pointer capture.
    for key in tuple(generation._parameter_widgets):
        if key not in wanted_keys:
            _dispose_control(generation, key)

    controls = []
    for item in layout:
        key = (item.node_id, item.symbol)
        value = _untracked_parameter_value(generation, key, fallback=item.minimum)
        control = generation._parameter_widgets.get(key)
        if control is None:
            control = _create_control(generation, item, value)
        else:
            _update_control_layout(generation, control, item, value)
        controls.append(control)

    generation.layout.set_controls(tuple(controls))


def sync_parameter_controls(
    generation: FigureDisplayGeneration,
    values: tuple[SliderValueItem, ...],
) -> None:
    """Update existing slider values from model snapshots without rebuilding."""

    for item in values:
        control = generation._parameter_widgets.get((item.node_id, item.symbol))
        if control is not None:
            _set_slider_value_from_model(generation, _control_slider(control), item.value)
            _run_widget_sync(generation, lambda: _sync_control_value(control))


def _dispose_control_observers(generation: FigureDisplayGeneration) -> None:
    """Remove every live slider callback registered by the generation."""

    for key in tuple(generation._parameter_widgets):
        _dispose_control(generation, key)


def _create_control(
    generation: FigureDisplayGeneration,
    item: ControlLayoutItem,
    value: float,
) -> object:
    """Create one slider and attach its guarded user-input observer."""

    import ipywidgets as ipywidgets

    key = (item.node_id, item.symbol)
    label = ipywidgets.Output(
        layout=ipywidgets.Layout(
            margin="0",
            min_width="0",
        )
    )
    label_section = ipywidgets.HBox(
        [label],
        layout=ipywidgets.Layout(
            align_items="center",
            flex="0 0 auto",
            justify_content="flex-start",
            min_width="0",
            overflow="hidden",
        ),
    )
    value_input = ipywidgets.Text(
        value=_format_number(value),
        description="",
        continuous_update=False,
        layout=ipywidgets.Layout(
            flex=f"0 0 {_VALUE_TEXT_WIDTH_CHARS}ch",
            max_width=f"{_VALUE_TEXT_WIDTH_CHARS}ch",
            min_width="0",
            width=f"{_VALUE_TEXT_WIDTH_CHARS}ch",
        ),
        style={"description_width": "0"},
    )
    reset_button = ipywidgets.Button(
        description="",
        icon="refresh",
        tooltip="Reset parameter to default",
        layout=ipywidgets.Layout(
            border="0",
            flex="0 0 1.05rem",
            height="1.05rem",
            margin="0",
            padding="0",
            width="1.05rem",
        ),
        style={"button_color": "transparent"},
    )
    label_value_section = ipywidgets.HBox(
        [label_section, value_input, reset_button],
        layout=ipywidgets.Layout(
            align_items="center",
            flex="0 0 auto",
            grid_gap="0.5rem",
            justify_content="flex-start",
            max_width="12rem",
            min_width="0",
            overflow="hidden",
        ),
    )
    spacer = ipywidgets.Box(
        layout=ipywidgets.Layout(
            flex="1 1 auto",
            min_width="0",
        ),
    )
    slider = ipywidgets.FloatSlider(
        value=value,
        min=min(item.minimum, value),
        max=max(item.maximum, value),
        step=item.step,
        description="",
        continuous_update=True,
        readout=False,
        layout=ipywidgets.Layout(
            flex="1 1 auto",
            height="0.9rem",
            margin="0",
            max_width="12rem",
            min_height="0",
            min_width="0",
            width="100%",
        ),
        style={"description_width": "0"},
    )
    minimum = ipywidgets.Text(
        value=_format_limit_number(item.minimum),
        description="",
        continuous_update=False,
        layout=ipywidgets.Layout(
            flex="0 0 5ch",
            max_width="5ch",
            min_width="0",
            width="5ch",
        ),
        style={"description_width": "0"},
    )
    maximum = ipywidgets.Text(
        value=_format_limit_number(item.maximum),
        description="",
        continuous_update=False,
        layout=ipywidgets.Layout(
            flex="0 0 5ch",
            max_width="5ch",
            min_width="0",
            width="5ch",
        ),
        style={"description_width": "0"},
    )
    slider_stack = ipywidgets.HBox(
        [minimum, slider, maximum],
        layout=ipywidgets.Layout(
            align_items="center",
            flex="0 1 auto",
            grid_gap="0.18rem",
            height="1rem",
            justify_content="flex-start",
            min_height="0",
            min_width="calc(10ch + 4rem + 0.36rem)",
            overflow="hidden",
            width="auto",
        ),
    )
    edit_button = ipywidgets.Button(
        description="",
        icon="gear",
        tooltip="Edit parameter settings",
        layout=ipywidgets.Layout(
            border="0",
            flex="0 0 1.15rem",
            height="1.15rem",
            margin="0",
            padding="0",
            width="1.15rem",
        ),
        style={"button_color": "transparent"},
    )
    control = ipywidgets.HBox(
        [label_value_section, spacer, slider_stack, edit_button],
        layout=ipywidgets.Layout(
            align_items="stretch",
            flex_flow="row nowrap",
            grid_gap="0.18rem",
            overflow="hidden",
            width="100%",
        ),
    )
    for widget, class_name in (
        (control, "mt-plot__parameter-control"),
        (value_input, "mt-plot__parameter-value"),
        (reset_button, "mt-plot__parameter-reset-button"),
        (slider, "mt-plot__parameter-slider"),
        (minimum, "mt-plot__parameter-limit"),
        (minimum, "mt-plot__parameter-minimum"),
        (maximum, "mt-plot__parameter-limit"),
        (maximum, "mt-plot__parameter-maximum"),
        (spacer, "mt-plot__parameter-spacer"),
        (edit_button, "mt-plot__parameter-edit-button"),
    ):
        add_class = getattr(widget, "add_class", None)
        if add_class is not None:
            add_class(class_name)
    control._mt_slider = slider
    control._mt_label = label
    control._mt_label_section = label_section
    control._mt_label_value_section = label_value_section
    control._mt_spacer = spacer
    control._mt_slider_stack = slider_stack
    control._mt_edit_button = edit_button
    control._mt_value_section = label_value_section
    control._mt_value_input = value_input
    control._mt_value_label = value_input
    control._mt_reset_button = reset_button
    control._mt_minimum = minimum
    control._mt_maximum = maximum
    control._mt_label_markdown = None
    _update_control_layout(generation, control, item, value)

    def _set_value(change: dict[str, object], *, control_key=key) -> None:
        _run_widget_sync(generation, lambda: _sync_control_value(control))
        if generation._syncing_widget_values:
            return
        if not generation.accepts_frontend_events():
            return
        _node_id, symbol = control_key
        if symbol in generation.figure.parameters:
            generation.figure.set_params({symbol: change["new"]})

    def _set_value_text(change: dict[str, object], *, control_key=key) -> None:
        if generation._syncing_widget_values:
            return
        _apply_value_text(generation, control, control_key, change["new"])

    def _set_minimum(change: dict[str, object], *, control_key=key) -> None:
        if generation._syncing_widget_values:
            return
        _apply_limit_text(generation, control, control_key, "minimum", change["new"])

    def _set_maximum(change: dict[str, object], *, control_key=key) -> None:
        if generation._syncing_widget_values:
            return
        _apply_limit_text(generation, control, control_key, "maximum", change["new"])

    def _open_parameter_settings(_button: object, *, control_key=key) -> None:
        if not generation.accepts_frontend_events():
            return
        node_id, symbol = control_key
        if symbol not in generation.figure.parameters:
            return
        from .editors import ParameterConfigPanel

        generation.modal.open(
            ParameterConfigPanel(
                generation.figure,
                node_id=node_id,
                symbol=symbol,
            )
        )

    def _reset_parameter_value(_button: object, *, control_key=key) -> None:
        if not generation.accepts_frontend_events():
            return
        _node_id, symbol = control_key
        default_value = generation.figure._default_parameter_value(symbol)
        if default_value is None:
            return
        state = generation.figure.parameters.get(symbol)
        if state is not None:
            state.set_value(default_value)

    slider.observe(_set_value, names="value")
    value_input.observe(_set_value_text, names="value")
    minimum.observe(_set_minimum, names="value")
    maximum.observe(_set_maximum, names="value")
    reset_button.on_click(_reset_parameter_value)
    edit_button.on_click(_open_parameter_settings)
    generation._parameter_widgets[key] = control
    generation._control_observers[key] = (
        (slider, _set_value),
        (value_input, _set_value_text),
        (minimum, _set_minimum),
        (maximum, _set_maximum),
    )
    generation._control_reset_observers[key] = (reset_button, _reset_parameter_value)
    generation._control_edit_observers[key] = (edit_button, _open_parameter_settings)
    return control


def _update_control_layout(
    generation: FigureDisplayGeneration,
    control: object,
    item: ControlLayoutItem,
    value: float,
) -> None:
    """Update slider metadata in place while keeping its value valid."""

    def _apply() -> None:
        slider = _control_slider(control)
        current_value = float(slider.value)
        temporary_minimum = min(item.minimum, value, current_value)
        temporary_maximum = max(item.maximum, value, current_value)

        # Widen first, then move the value, then shrink to the final metadata.
        # This avoids transient trait validation failures when a parameter value
        # and its slider range are changed in the same public call.
        if float(slider.max) < temporary_maximum:
            slider.max = temporary_maximum
        if float(slider.min) > temporary_minimum:
            slider.min = temporary_minimum
        if slider.description != "":
            slider.description = ""
        if float(slider.step) != item.step:
            slider.step = item.step
        if float(slider.value) != value:
            slider.value = value
        if float(slider.min) != item.minimum:
            slider.min = item.minimum
        if float(slider.max) != item.maximum:
            slider.max = item.maximum

    _run_widget_sync(generation, _apply)
    _sync_control_label(control, item.label_markdown)
    _run_widget_sync(
        generation,
        lambda: (
            _sync_control_value(control),
            _sync_limit_fields(control),
        ),
    )


def _set_slider_value_from_model(
    generation: FigureDisplayGeneration,
    slider: object,
    value: float,
) -> None:
    """Mirror a model value into a slider without reporting user input."""

    def _apply() -> None:
        if float(slider.max) < value:
            slider.max = value
        if float(slider.min) > value:
            slider.min = value
        if float(slider.value) != value:
            slider.value = value

    _run_widget_sync(generation, _apply)


def _run_widget_sync(
    generation: FigureDisplayGeneration,
    action: Callable[[], None],
) -> None:
    """Run a model-originated widget mutation under the observer guard."""

    previous = generation._syncing_widget_values
    generation._syncing_widget_values = True
    try:
        action()
    finally:
        generation._syncing_widget_values = previous


def _control_slider(control: object) -> object:
    """Return the live slider owned by a parameter control row."""

    return getattr(control, "_mt_slider", control)


def _sync_control_label(control: object, markdown: str) -> None:
    """Render one parameter label as markdown when it changes."""

    if getattr(control, "_mt_label_markdown", None) == markdown:
        return

    label = getattr(control, "_mt_label", None)
    if label is None:
        return

    label.outputs = (
        {
            "output_type": "display_data",
            "data": {"text/markdown": markdown},
            "metadata": {},
        },
    )
    control._mt_label_markdown = markdown


def _sync_control_value(control: object) -> None:
    """Show the current slider value immediately after the rendered label."""

    value_input = getattr(control, "_mt_value_input", None)
    if value_input is None:
        return

    value = _format_number(float(_control_slider(control).value))
    if value_input.value != value:
        value_input.value = value


def _sync_limit_fields(control: object) -> None:
    """Mirror slider bounds into the editable limit fields."""

    slider = _control_slider(control)
    minimum = getattr(control, "_mt_minimum", None)
    maximum = getattr(control, "_mt_maximum", None)
    minimum_value = _format_limit_number(float(slider.min))
    maximum_value = _format_limit_number(float(slider.max))
    if minimum is not None and minimum.value != minimum_value:
        minimum.value = minimum_value
    if maximum is not None and maximum.value != maximum_value:
        maximum.value = maximum_value


def _apply_limit_text(
    generation: FigureDisplayGeneration,
    control: object,
    key: tuple[int, sympy.Symbol],
    side: str,
    raw_value: object,
) -> None:
    """Apply one user-edited slider limit or restore the previous bound."""

    slider = _control_slider(control)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        _run_widget_sync(generation, lambda: _sync_limit_fields(control))
        return
    if not math.isfinite(value):
        _run_widget_sync(generation, lambda: _sync_limit_fields(control))
        return

    if side == "minimum":
        if value >= float(slider.max):
            _run_widget_sync(generation, lambda: _sync_limit_fields(control))
            return

        def _apply_minimum() -> None:
            if float(slider.value) < value:
                slider.value = value
            slider.min = value

        _apply_limit_change(generation, control, key, _apply_minimum)
        return

    if value <= float(slider.min):
        _run_widget_sync(generation, lambda: _sync_limit_fields(control))
        return

    def _apply_maximum() -> None:
        if float(slider.value) > value:
            slider.value = value
        slider.max = value

    _apply_limit_change(generation, control, key, _apply_maximum)


def _apply_value_text(
    generation: FigureDisplayGeneration,
    control: object,
    key: tuple[int, sympy.Symbol],
    raw_value: object,
) -> None:
    """Apply one user-edited slider value or restore the current value."""

    slider = _control_slider(control)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        _run_widget_sync(generation, lambda: _sync_control_value(control))
        return
    if not math.isfinite(value):
        _run_widget_sync(generation, lambda: _sync_control_value(control))
        return
    if value < float(slider.min) or value > float(slider.max):
        _run_widget_sync(generation, lambda: _sync_control_value(control))
        return
    if not generation.accepts_frontend_events():
        _run_widget_sync(generation, lambda: _sync_control_value(control))
        return

    _run_widget_sync(generation, lambda: setattr(slider, "value", value))
    state = _parameter_state_for_key(generation, key)
    if state is not None:
        state.set_value(value)
    _run_widget_sync(generation, lambda: _sync_control_value(control))


def _apply_limit_change(
    generation: FigureDisplayGeneration,
    control: object,
    key: tuple[int, sympy.Symbol],
    action: Callable[[], None],
) -> None:
    """Apply a valid limit edit and publish it to the parameter metadata."""

    action()
    _run_widget_sync(
        generation,
        lambda: (
            _sync_control_value(control),
            _sync_limit_fields(control),
        ),
    )
    state = _parameter_state_for_key(generation, key)
    if state is not None:
        metadata = state.metadata_signal()
        state.metadata_signal.set(
            replace(
                metadata,
                minimum=float(_control_slider(control).min),
                maximum=float(_control_slider(control).max),
            )
        )


def _format_number(value: float) -> str:
    """Return a compact display string for a slider value."""

    return f"{value:g}"


def _format_limit_number(value: float) -> str:
    """Return a compact four-significant-digit string for a slider bound."""

    return f"{value:.4g}"


def _dispose_control(
    generation: FigureDisplayGeneration,
    key: tuple[int, sympy.Symbol],
) -> None:
    """Dispose one slider observer and remove registry entries exactly once."""

    control = generation._parameter_widgets.pop(key, None)
    observer = generation._control_observers.pop(key, None)
    edit_observer = generation._control_edit_observers.pop(key, None)
    reset_observer = generation._control_reset_observers.pop(key, None)
    if control is not None and observer is not None:
        for widget, callback in observer:
            widget.unobserve(callback, names="value")
    if reset_observer is not None:
        button, callback = reset_observer
        button.on_click(callback, remove=True)
    if edit_observer is not None:
        button, callback = edit_observer
        button.on_click(callback, remove=True)


def _untracked_parameter_value(
    generation: FigureDisplayGeneration,
    key: tuple[int, sympy.Symbol],
    *,
    fallback: float,
) -> float:
    """Return a parameter value for widget creation without subscribing effects."""

    def _read() -> float:
        state = _parameter_state_for_key(generation, key)
        if state is None:
            return fallback
        return state.value

    return untracked(_read)


def _parameter_state_for_key(
    generation: FigureDisplayGeneration,
    key: tuple[int, sympy.Symbol],
) -> object | None:
    """Return the current parameter state for a control key."""

    _node_id, symbol = key
    return generation.figure.parameters.get(symbol)
