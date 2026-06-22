"""Select and implement plotting frontend backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import PlotSpecError
from . import messages
from .controls import AnywidgetParameterControls
from .legend import AnywidgetLegend
from .modal import AnywidgetModalController
from .output import FigureMessageOutput
from .shell import FigureShellWidget
from .host import AnywidgetHostAdapter, create_host_adapter

if TYPE_CHECKING:
    from ..model import ControlLayoutItem, InfoCardSnapshot, LegendItem, SliderValueItem
    from ..display import FigureDisplayGeneration

__all__ = [
    "AnywidgetFrontendBackend",
    "IpywidgetsFrontendBackend",
    "MarimoFrontendBackend",
    "create_frontend_backend",
    "normalize_display_backend",
]


def normalize_display_backend(backend: str | None) -> str:
    """Return the normalized plotting frontend backend name."""

    if backend in (None, "jupyter", "ipywidgets", "widget"):
        return "ipywidgets"
    if backend == "anywidget":
        return "anywidget"
    raise PlotSpecError(
        "FigureHandle.show(...) backend must be one of None, 'jupyter', "
        "'ipywidgets', 'widget', or 'anywidget'."
    )


class IpywidgetsFrontendBackend:
    """Adapter name for the existing ipywidgets plotting frontend."""

    name = "ipywidgets"


def create_frontend_backend(
    generation: FigureDisplayGeneration,
    plot_widget: object,
) -> object:
    """Create the host-appropriate frontend for an anywidget display."""

    return AnywidgetFrontendBackend(
        generation,
        plot_widget,
        host_adapter=create_host_adapter(),
    )


class AnywidgetFrontendBackend:
    """Own the plain-anywidget frontend components for one display generation."""

    def __init__(
        self,
        generation: FigureDisplayGeneration,
        plot_widget: object,
        *,
        host_adapter: AnywidgetHostAdapter | None = None,
    ) -> None:
        """Create all anywidget frontend components for a generation."""

        self.generation = generation
        self.plot_widget = plot_widget
        self.host_adapter = host_adapter or create_host_adapter()
        self.name = self.host_adapter.host_name
        self.shell = FigureShellWidget(
            generation_id=generation.generation_id,
            plot_widget=plot_widget,
            host_name=self.host_adapter.host_name,
            root_class=self.host_adapter.root_class,
            child_mount_policy=self.host_adapter.child_mount_policy,
            markdown_policy=self.host_adapter.markdown_policy,
        )
        self.controls = AnywidgetParameterControls(
            generation,
            self.shell,
            markdown_payload=self.host_adapter.markdown_payload,
            native_markdown_labels=self.host_adapter.native_markdown_widgets,
        )
        self.legend = AnywidgetLegend(
            generation,
            self.shell,
            markdown_payload=self.host_adapter.markdown_payload,
            native_markdown_labels=self.host_adapter.native_markdown_widgets,
        )
        self.modal = AnywidgetModalController(generation, self.shell)
        self.output = FigureMessageOutput(
            self.shell,
            markdown_payload=self.host_adapter.markdown_payload,
        )
        self.info_markdown_widgets: dict[tuple[object, ...], object] = {}
        self.shell.on_msg(self._handle_message)
        self._root = self.host_adapter.root_for_display(
            self.shell,
            hosted_widgets=(plot_widget,),
        )

    @property
    def root(self) -> object:
        """Return the displayed shell widget."""

        return self._root

    @property
    def layout(self) -> object:
        """Return the shell object used by compatibility accessors."""

        return self

    def set_runtime_widgets(self, widgets: tuple[object, ...]) -> None:
        """Publish hidden helper widgets that must share the shell DOM root."""

        self.shell.set_runtime_widgets(widgets)
        if hasattr(self._root, "hosted_widgets"):
            self._root.hosted_widgets = (self.plot_widget, *widgets)

    def reconcile_controls(self, layout: tuple[ControlLayoutItem, ...]) -> None:
        """Publish parameter control layout payloads."""

        self.controls.reconcile(layout)

    def sync_control_values(self, values: tuple[SliderValueItem, ...]) -> None:
        """Publish parameter value payloads."""

        self.controls.sync_values(values)

    def reconcile_legend(self, items: tuple[LegendItem, ...]) -> None:
        """Publish legend row payloads."""

        self.legend.reconcile(items)

    def set_info(self, snapshots: tuple[InfoCardSnapshot, ...]) -> None:
        """Publish authored info card payloads."""

        wanted_keys = self._info_markdown_widget_keys(snapshots)
        for key in tuple(self.info_markdown_widgets):
            if key not in wanted_keys:
                widget = self.info_markdown_widgets.pop(key)
                close = getattr(widget, "close", None)
                if callable(close):
                    close()

        info_widgets = self._info_markdown_widgets(snapshots)
        widget_indices = {
            key: index
            for index, (key, _widget) in enumerate(info_widgets)
        }
        self.shell.set_info(
            tuple(
                self._info_payload(snapshot, widget_indices)
                for snapshot in snapshots
            ),
            tuple(widget for _key, widget in info_widgets),
        )

    def set_status(self, message: str | None, *, kind: str = "info") -> None:
        """Publish a status message."""

        self.shell.set_status(message, kind=kind)

    def retire(self, message: str) -> None:
        """Mark the shell disconnected and append a trusted notice."""

        self.controls.disable()
        self.shell.set_disconnected(message)
        self.output.append_stdout(f"{message}\n")

    def dispose(self) -> None:
        """Release backend-owned callbacks and widget state."""

        self.controls.dispose()
        self.legend.dispose()
        self.modal.dispose()
        self.output.clear()
        for widget in self.info_markdown_widgets.values():
            close = getattr(widget, "close", None)
            if callable(close):
                close()
        self.info_markdown_widgets.clear()
        close = getattr(self.shell, "close", None)
        if callable(close):
            close()

    def _info_payload(
        self,
        snapshot: InfoCardSnapshot,
        widget_indices: dict[tuple[object, ...], int],
    ) -> dict[str, object]:
        """Return one info card frontend payload."""

        payload = {
            "card_id": snapshot.card_id,
            "name": snapshot.name,
            "title_markdown": snapshot.title_markdown,
            "markdown": snapshot.markdown,
            "segments": tuple(
                self._info_segment_payload(
                    snapshot.card_id,
                    segment,
                    widget_indices,
                )
                for segment in snapshot.segments
            ),
            "title_markdown_payload": self.host_adapter.markdown_payload(
                f"**{snapshot.title_markdown}**"
                if snapshot.title_markdown
                else "",
            ),
            "markdown_payload": self.host_adapter.markdown_payload(snapshot.markdown),
            "error": snapshot.error,
        }
        title_key = (snapshot.card_id, "title")
        if title_key in widget_indices:
            payload["title_widget_index"] = widget_indices[title_key]
        return payload

    def _info_segment_payload(
        self,
        card_id: int,
        segment: object,
        widget_indices: dict[tuple[object, ...], int],
    ) -> dict[str, object]:
        """Return a frontend payload for one info card body segment."""

        text = str(getattr(segment, "text"))
        kind = str(getattr(segment, "kind"))
        index = int(getattr(segment, "index"))
        payload: dict[str, object] = {
            "index": index,
            "kind": kind,
            "text": text,
            "markdown_payload": self.host_adapter.markdown_payload(text),
        }
        segment_key = (card_id, "segment", index)
        if segment_key in widget_indices:
            payload["widget_index"] = widget_indices[segment_key]
        return payload

    def _info_markdown_widget_keys(
        self,
        snapshots: tuple[InfoCardSnapshot, ...],
    ) -> set[tuple[object, ...]]:
        """Return native Markdown widget keys needed by current info cards."""

        if not self.host_adapter.native_markdown_widgets:
            return set()

        wanted: set[tuple[object, ...]] = set()
        for snapshot in snapshots:
            if snapshot.title_markdown:
                wanted.add((snapshot.card_id, "title"))
            for segment in snapshot.segments:
                wanted.add((snapshot.card_id, "segment", segment.index))
        return wanted

    def _info_markdown_widgets(
        self,
        snapshots: tuple[InfoCardSnapshot, ...],
    ) -> tuple[tuple[tuple[object, ...], object], ...]:
        """Return native Markdown output widgets for Jupyter info cards."""

        if not self.host_adapter.native_markdown_widgets:
            return ()

        ordered = []
        for snapshot in snapshots:
            if snapshot.title_markdown:
                title_key = (snapshot.card_id, "title")
                title_widget = self._info_markdown_widget(
                    title_key,
                    f"**{snapshot.title_markdown}**",
                )
                ordered.append((title_key, title_widget))

            for segment in snapshot.segments:
                key = (snapshot.card_id, "segment", segment.index)
                widget = self._info_markdown_widget(key, segment.text)
                ordered.append((key, widget))
        return tuple(ordered)

    def _info_markdown_widget(
        self,
        key: tuple[object, ...],
        markdown: str,
    ) -> object:
        """Return a stable native Markdown output for one static info fragment."""

        widget = self.info_markdown_widgets.get(key)
        if widget is None:
            import ipywidgets as ipywidgets

            widget = ipywidgets.Output()
            self.info_markdown_widgets[key] = widget
        if getattr(widget, "_mt_markdown", None) != markdown:
            widget.outputs = _markdown_outputs(markdown)
            widget._mt_markdown = markdown
        return widget

    def _handle_message(
        self,
        widget: object,
        content: dict[str, object],
        buffers: object,
    ) -> None:
        """Route one browser-originated custom message to Python model state."""

        if content.get("generation_id") != self.generation.generation_id:
            return
        message_type = content.get("type")
        if message_type == messages.OPEN_PARAMETER_SETTINGS:
            self.modal.open_parameter(
                node_id=content.get("node_id"),
                symbol_text=content.get("symbol"),
            )
        elif message_type == messages.TOGGLE_PLOT_VISIBILITY:
            self.legend.toggle_visibility(content.get("node_id"))
        elif message_type == messages.TOGGLE_PLOT_SOUND:
            self.legend.toggle_sound(content.get("node_id"))
        elif message_type == messages.OPEN_PLOT_SETTINGS:
            self.modal.open_plot(node_id=content.get("node_id"))
        elif message_type == messages.MODAL_FIELD_CHANGED:
            self.modal.update_field(content.get("field_id"), content.get("value"))
        elif message_type == messages.MODAL_APPLY:
            self.modal.apply()
        elif message_type in {messages.MODAL_CANCEL, messages.MODAL_CLOSE}:
            if self.generation.accepts_frontend_events():
                self.modal.close()


class MarimoFrontendBackend(AnywidgetFrontendBackend):
    """Compatibility wrapper for callers that construct the Marimo backend."""

    def __init__(self, generation: FigureDisplayGeneration, plot_widget: object) -> None:
        """Create the anywidget frontend with the Marimo host adapter."""

        from .host import MarimoHostAdapter

        super().__init__(
            generation,
            plot_widget,
            host_adapter=MarimoHostAdapter(),
        )


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
