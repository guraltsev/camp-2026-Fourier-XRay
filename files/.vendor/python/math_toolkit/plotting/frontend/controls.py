"""Synchronize composed anywidget parameter controls for plotting."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING, Callable

from .markdown import render_markdown_payload

from .widgets import (
    ButtonWidget,
    HBoxWidget,
    MarkdownWidget,
    SliderWidget,
    TextEntryWidget,
)

if TYPE_CHECKING:
    import sympy

    from ..display import FigureDisplayGeneration
    from ..model import ControlLayoutItem, SliderValueItem

__all__ = ["AnywidgetParameterControls", "ParameterControlWidget"]


class ParameterControlWidget(HBoxWidget):
    """Compose primitive anywidgets into one parameter control row."""

    def __init__(
        self,
        *,
        label_markdown: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        markdown_payload: Callable[[str], dict[str, str]] | None = None,
        native_markdown_label: bool = False,
    ) -> None:
        """Create a parameter control from small traitlet-backed widgets."""

        self.native_markdown_label = bool(native_markdown_label)
        if self.native_markdown_label:
            import ipywidgets as ipywidgets

            self.label = ipywidgets.Output()
            add_class = getattr(self.label, "add_class", None)
            if callable(add_class):
                add_class("mt-param-row__label")
            self._set_native_label_markdown(label_markdown)
        else:
            self.label = MarkdownWidget(
                label_markdown,
                class_name="mt-param-row__label",
                markdown_payload=markdown_payload,
            )
        self.value_entry = TextEntryWidget(
            _format_number(value),
            class_name="mt-text-entry--value",
        )
        self.reset_button = ButtonWidget(
            "refresh",
            title="Reset parameter to default",
            class_name="mt-param-row__reset",
        )
        self.label_value = HBoxWidget(
            (self.label, self.value_entry, self.reset_button),
            gap="0.5rem",
            align_items="center",
            class_name="mt-param-row__label-value",
        )
        self.slider = SliderWidget(
            value=value,
            minimum=minimum,
            maximum=maximum,
            step=step,
            class_name="mt-param-row__slider",
        )
        self.minimum_entry = TextEntryWidget(
            _format_limit_number(minimum),
            class_name="mt-text-entry--limit mt-text-entry--minimum",
            style={
                "background": "transparent",
                "border": "0",
                "boxShadow": "none",
                "fontSize": "0.68rem",
                "paddingLeft": "0",
                "textAlign": "right",
            },
        )
        self.maximum_entry = TextEntryWidget(
            _format_limit_number(maximum),
            class_name="mt-text-entry--limit mt-text-entry--maximum",
            style={
                "background": "transparent",
                "border": "0",
                "boxShadow": "none",
                "fontSize": "0.68rem",
                "paddingRight": "0",
                "textAlign": "left",
            },
        )
        self.slider_stack = HBoxWidget(
            (self.minimum_entry, self.slider, self.maximum_entry),
            gap="0.18rem",
            align_items="center",
            class_name="mt-param-row__slider-stack",
        )
        self.spacer = HBoxWidget(
            (),
            gap="0",
            align_items="center",
            class_name="mt-param-row__spacer",
        )
        self.edit_button = ButtonWidget(
            "gear",
            title="Edit parameter settings",
            class_name="mt-param-row__edit",
        )
        super().__init__(
            (self.label_value, self.spacer, self.slider_stack, self.edit_button),
            gap="0.25rem",
            align_items="center",
            class_name="mt-param-row",
        )

    @property
    def label_markdown(self) -> str:
        """Return the label Markdown rendered by the label child widget."""

        if self.native_markdown_label:
            return str(getattr(self.label, "_mt_label_markdown", ""))
        return self.label.value

    @label_markdown.setter
    def label_markdown(self, value: str) -> None:
        if self.native_markdown_label:
            self._set_native_label_markdown(value)
        else:
            self.label.set_markdown(value)

    @property
    def value(self) -> float:
        """Return the current slider value."""

        return float(self.slider.browser_value)

    @value.setter
    def value(self, value: float) -> None:
        number = float(value)
        self.slider.value = number
        self.slider.browser_value = number
        self.value_entry.value = _format_number(number)

    @property
    def minimum(self) -> float:
        """Return the current slider minimum."""

        return float(self.slider.minimum)

    @minimum.setter
    def minimum(self, value: float) -> None:
        self.slider.minimum = float(value)
        self.minimum_entry.value = _format_limit_number(float(value))

    @property
    def maximum(self) -> float:
        """Return the current slider maximum."""

        return float(self.slider.maximum)

    @maximum.setter
    def maximum(self, value: float) -> None:
        self.slider.maximum = float(value)
        self.maximum_entry.value = _format_limit_number(float(value))

    @property
    def step(self) -> float:
        """Return the current slider step."""

        return float(self.slider.step)

    @step.setter
    def step(self, value: float) -> None:
        self.slider.step = float(value)

    @property
    def disabled(self) -> bool:
        """Return whether the interactive child widgets are disabled."""

        return bool(self.slider.disabled)

    @disabled.setter
    def disabled(self, value: bool) -> None:
        disabled = bool(value)
        self.slider.disabled = disabled
        self.value_entry.disabled = disabled
        self.reset_button.disabled = disabled
        self.minimum_entry.disabled = disabled
        self.maximum_entry.disabled = disabled
        self.edit_button.disabled = disabled

    @property
    def release_count(self) -> int:
        """Return the slider release counter."""

        return int(self.slider.release_count)

    @release_count.setter
    def release_count(self, value: int) -> None:
        self.slider.release_count = int(value)

    def set_layout_traits(
        self,
        *,
        label_markdown: str,
        minimum: float,
        maximum: float,
        step: float,
    ) -> None:
        """Synchronize layout metadata without changing the slider value."""

        self.label_markdown = label_markdown
        self.minimum = minimum
        self.maximum = maximum
        self.step = step

    def _set_native_label_markdown(self, value: str) -> None:
        """Update a native Jupyter Markdown output label."""

        text = str(value)
        self.label._mt_label_markdown = text
        self.label.outputs = _markdown_outputs(text)


class AnywidgetParameterControls:
    """Own composed parameter child widgets and their model bindings."""

    _control_widget_class = ParameterControlWidget

    def __init__(
        self,
        generation: FigureDisplayGeneration,
        shell: object,
        *,
        markdown_payload: Callable[[str], dict[str, str]] | None = None,
        native_markdown_labels: bool = False,
    ) -> None:
        """Create parameter control widgets for one generation."""

        self.generation = generation
        self.shell = shell
        self.markdown_payload = markdown_payload or render_markdown_payload
        self.native_markdown_labels = bool(native_markdown_labels)
        self.layout_items: dict[tuple[int, str], ControlLayoutItem] = {}
        self.value_items: dict[tuple[int, str], float] = {}
        self.widgets: dict[tuple[int, str], ParameterControlWidget] = {}
        self._observers: dict[tuple[int, str], tuple[tuple[object, str, object], ...]]
        self._observers = {}
        self._syncing_widget_values = False
        self._browser_value_event_depth = 0

    def reconcile(self, layout: tuple[ControlLayoutItem, ...]) -> None:
        """Reconcile child parameter widgets from ordered layout metadata."""

        wanted_keys = {(item.node_id, str(item.symbol)) for item in layout}
        for key in tuple(self.widgets):
            if key not in wanted_keys:
                self._dispose_widget(key)

        ordered_widgets = []
        self.layout_items = {(item.node_id, str(item.symbol)): item for item in layout}
        for item in layout:
            key = (item.node_id, str(item.symbol))
            value = self.value_items.get(key, self._parameter_value(item.symbol, item.minimum))
            widget = self.widgets.get(key)
            if widget is None:
                widget = self._create_widget(key, item, value)
            else:
                self._sync_widget_layout(widget, item)
            ordered_widgets.append(widget)
        self.shell.set_controls(tuple(ordered_widgets))

    def sync_values(self, values: tuple[SliderValueItem, ...]) -> None:
        """Mirror model-originated values into existing child widgets."""

        for item in values:
            key = (item.node_id, str(item.symbol))
            self.value_items[key] = float(item.value)
            if self._browser_value_event_depth:
                continue
            widget = self.widgets.get(key)
            if widget is not None:
                self._sync_widget_value(widget, float(item.value))

    def disable(self) -> None:
        """Disable all child widgets without removing their visible views."""

        for widget in self.widgets.values():
            widget.disabled = True

    def dispose(self) -> None:
        """Release child widget observers and cached payload state."""

        for key in tuple(self.widgets):
            self._dispose_widget(key)
        self.shell.set_controls(())
        self.layout_items.clear()
        self.value_items.clear()
        self._syncing_widget_values = False
        self._browser_value_event_depth = 0

    def _create_widget(
        self,
        key: tuple[int, str],
        item: ControlLayoutItem,
        value: float,
    ) -> ParameterControlWidget:
        """Create one composed child widget and bind primitive trait events."""

        widget = self._control_widget_class(
            label_markdown=item.label_markdown,
            value=float(value),
            minimum=float(item.minimum),
            maximum=float(item.maximum),
            step=float(item.step),
            markdown_payload=self.markdown_payload,
            native_markdown_label=self.native_markdown_labels,
        )

        def _on_value(_change: dict[str, object], *, control_key=key) -> None:
            if self._syncing_widget_values:
                return
            state = self._state_for_key(control_key)
            number = self._finite_float(widget.slider.browser_value)
            if number is not None:
                self._sync_value_entry(widget, number)
            if state is not None and number is not None:
                self._apply_browser_value(state, number)

        def _on_release(_change: dict[str, object], *, control_key=key) -> None:
            if self._syncing_widget_values:
                return
            self._settle_widget_value(control_key)

        def _on_value_commit(_change: dict[str, object], *, control_key=key) -> None:
            if self._syncing_widget_values:
                return
            self._commit_value_text(control_key, widget.value_entry.value)

        def _on_minimum_commit(_change: dict[str, object], *, control_key=key) -> None:
            if self._syncing_widget_values:
                return
            self._commit_limit_text(control_key, "minimum", widget.minimum_entry.value)

        def _on_maximum_commit(_change: dict[str, object], *, control_key=key) -> None:
            if self._syncing_widget_values:
                return
            self._commit_limit_text(control_key, "maximum", widget.maximum_entry.value)

        def _on_edit(_change: dict[str, object], *, control_key=key) -> None:
            if not self.generation.accepts_frontend_events():
                return
            item = self.layout_items.get(control_key)
            if item is not None:
                self.generation.frontend.modal.open_parameter(
                    node_id=item.node_id,
                    symbol_text=str(item.symbol),
                )

        def _on_reset(_change: dict[str, object], *, control_key=key) -> None:
            if not self.generation.accepts_frontend_events():
                return
            symbol = self._symbol_for_text(control_key[1])
            if symbol is None:
                return
            default_value = self.generation.figure._default_parameter_value(symbol)
            if default_value is None:
                return
            state = self.generation.figure.parameters.get(symbol)
            if state is not None:
                state.set_value(default_value)

        observers = (
            (widget.slider, "browser_value", _on_value),
            (widget.slider, "release_count", _on_release),
            (widget.value_entry, "commit_count", _on_value_commit),
            (widget.minimum_entry, "commit_count", _on_minimum_commit),
            (widget.maximum_entry, "commit_count", _on_maximum_commit),
            (widget.reset_button, "click_count", _on_reset),
            (widget.edit_button, "click_count", _on_edit),
        )
        for observed_widget, name, callback in observers:
            observed_widget.observe(callback, names=name)
        self.widgets[key] = widget
        self._observers[key] = observers
        return widget

    def _dispose_widget(self, key: tuple[int, str]) -> None:
        """Remove observers from one composed child widget and close it."""

        widget = self.widgets.pop(key, None)
        observers = self._observers.pop(key, ())
        for observed_widget, name, callback in observers:
            observed_widget.unobserve(callback, names=name)
        if widget is not None:
            close = getattr(widget, "close", None)
            if callable(close):
                close()

    def _sync_widget_layout(
        self,
        widget: ParameterControlWidget,
        item: ControlLayoutItem,
    ) -> None:
        """Update one child widget's metadata under the observer guard."""

        self._run_widget_sync(
            lambda: widget.set_layout_traits(
                label_markdown=item.label_markdown,
                minimum=float(item.minimum),
                maximum=float(item.maximum),
                step=float(item.step),
            )
        )

    def _sync_widget_value(self, widget: ParameterControlWidget, value: float) -> None:
        """Mirror one model value into a child widget without user echo."""

        self._run_widget_sync(lambda: setattr(widget, "value", float(value)))

    def _sync_value_entry(self, widget: ParameterControlWidget, value: float) -> None:
        """Mirror a preview value into the text entry without slider trait echo."""

        self._run_widget_sync(
            lambda: setattr(widget.value_entry, "value", _format_number(float(value)))
        )

    def _run_widget_sync(self, action: object) -> None:
        """Run a model-originated widget mutation under the observer guard."""

        previous = self._syncing_widget_values
        self._syncing_widget_values = True
        try:
            action()
        finally:
            self._syncing_widget_values = previous

    def _apply_browser_value(self, state: object, number: float) -> None:
        """Apply one browser-originated value while suppressing mirror echoes."""

        self._browser_value_event_depth += 1
        try:
            # One parameter change can invalidate several traces. Keep the
            # Plotly FigureWidget transaction open while reactive effects run.
            # Info cards are intentionally deferred during preview ticks because
            # Markdown and math rendering can cost more than the graph update.
            with (
                self.generation.defer_info_updates(),
                self.generation.renderer.figure_widget.batch_update(),
            ):
                state.set_value(number)
        finally:
            self._browser_value_event_depth -= 1

    def _settle_widget_value(self, key: tuple[int, str]) -> None:
        """Publish the authoritative model value after a browser release."""

        state = self._state_for_key(key)
        widget = self.widgets.get(key)
        if state is None or widget is None:
            return
        number = self._finite_float(widget.value)
        if number is not None and number != state.value:
            state.set_value(number)
        self.sync_values(self.generation.figure.slider_value_snapshot())
        self.generation.flush_deferred_info()

    def _commit_value_text(self, key: tuple[int, str], value: object) -> None:
        """Apply a committed value text edit or restore the current value."""

        state = self._state_for_key(key)
        widget = self.widgets.get(key)
        item = self.layout_items.get(key)
        if state is None or widget is None or item is None:
            return
        number = self._finite_float(value)
        if number is None or number < item.minimum or number > item.maximum:
            self._sync_widget_value(widget, float(state.value))
            return
        state.set_value(number)

    def _commit_limit_text(
        self,
        key: tuple[int, str],
        side: str,
        value: object,
    ) -> None:
        """Apply a committed slider limit edit or restore the current layout."""

        state = self._state_for_key(key)
        widget = self.widgets.get(key)
        item = self.layout_items.get(key)
        number = self._finite_float(value)
        if state is None or widget is None or item is None or number is None:
            self.generation.figure.reconcile_controls()
            return

        metadata = state.metadata_signal()
        minimum = number if side == "minimum" else metadata.minimum
        maximum = number if side == "maximum" else metadata.maximum
        if minimum >= maximum:
            self.generation.figure.reconcile_controls()
            return

        next_value = min(max(state.value, minimum), maximum)
        state.metadata_signal.set(
            replace(metadata, minimum=float(minimum), maximum=float(maximum))
        )
        if next_value != state.value:
            state.set_value(next_value)

    def _state_for_key(self, key: tuple[int, str]) -> object | None:
        """Return a live parameter state for one child widget key."""

        if not self.generation.accepts_frontend_events():
            return None
        symbol = self._symbol_for_text(key[1])
        if symbol is None:
            return None
        return self.generation.figure.parameters.get(symbol)

    def _symbol_for_text(self, symbol_text: str) -> sympy.Symbol | None:
        """Return the live parameter symbol matching a frontend symbol string."""

        for symbol in self.generation.figure.parameters:
            if str(symbol) == symbol_text:
                return symbol
        return None

    def _parameter_value(self, symbol: sympy.Symbol, fallback: float) -> float:
        """Return the current parameter value or ``fallback``."""

        state = self.generation.figure.parameters.get(symbol)
        if state is None:
            return float(fallback)
        return float(state.value)

    @staticmethod
    def _finite_float(value: object) -> float | None:
        """Return a finite float or ``None`` for invalid frontend input."""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number


def _format_number(value: float) -> str:
    """Return a compact display string for a slider value."""

    return f"{value:g}"


def _format_limit_number(value: float) -> str:
    """Return a compact four-significant-digit string for a slider bound."""

    return f"{value:.4g}"


def _markdown_outputs(markdown: str) -> tuple[dict[str, object], ...]:
    """Return an ipywidgets output payload for Markdown content."""

    if not markdown:
        return ()
    return (
        {
            "output_type": "display_data",
            "data": {"text/markdown": markdown},
            "metadata": {},
        },
    )
