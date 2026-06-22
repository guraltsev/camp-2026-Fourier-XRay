"""Synchronize legend rows for the anywidget plotting backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from .markdown import render_markdown_payload

if TYPE_CHECKING:
    from ..display import FigureDisplayGeneration
    from ..model import LegendItem

__all__ = ["AnywidgetLegend"]


class AnywidgetLegend:
    """Own legend payloads and guarded frontend event handling."""

    def __init__(
        self,
        generation: FigureDisplayGeneration,
        shell: object,
        *,
        markdown_payload: Callable[[str], dict[str, str]] | None = None,
        native_markdown_labels: bool = False,
    ) -> None:
        """Create legend synchronization for one generation."""

        self.generation = generation
        self.shell = shell
        self.markdown_payload = markdown_payload or render_markdown_payload
        self.native_markdown_labels = bool(native_markdown_labels)
        self.items: dict[int, LegendItem] = {}
        self.label_widgets: dict[int, object] = {}

    def reconcile(self, items: tuple[LegendItem, ...]) -> None:
        """Publish ordered legend row payloads."""

        self.items = {item.node_id: item for item in items}
        wanted_ids = {item.node_id for item in items}
        for node_id in tuple(self.label_widgets):
            if node_id not in wanted_ids:
                widget = self.label_widgets.pop(node_id)
                close = getattr(widget, "close", None)
                if callable(close):
                    close()

        label_widgets = self._native_label_widgets(items)
        label_indices = {
            node_id: index
            for index, (node_id, _widget) in enumerate(label_widgets)
        }
        self.shell.set_legend(
            tuple(self._legend_payload(item, label_indices) for item in items),
            tuple(widget for _node_id, widget in label_widgets),
        )

    def toggle_visibility(self, node_id: object) -> None:
        """Toggle the target plot's visibility when the event is live."""

        node = self._node_for_event(node_id)
        if node is not None:
            node.toggle_visible()

    def toggle_sound(self, node_id: object) -> None:
        """Toggle the target curve's sound playback when available."""

        node = self._node_for_event(node_id)
        if node is None:
            return
        handle = self.generation.figure._handle_for_node(node)
        sound = getattr(handle, "sound", None)
        if sound is None or not sound.enabled:
            return
        if sound.state().status == "playing":
            sound.pause()
        else:
            sound.resume()

    def dispose(self) -> None:
        """Release cached legend payload state."""

        self.items.clear()
        for widget in self.label_widgets.values():
            close = getattr(widget, "close", None)
            if callable(close):
                close()
        self.label_widgets.clear()

    def _legend_payload(
        self,
        item: LegendItem,
        label_indices: dict[int, int],
    ) -> dict[str, object]:
        """Return one frontend legend row payload."""

        payload = {
            "node_id": item.node_id,
            "label_markdown": item.label_markdown,
            "label_payload": self.markdown_payload(item.label_markdown),
            "visible": item.visible,
            "marker": asdict(item.marker),
            "sound_playable": item.sound_playable,
            "sound_enabled": item.sound_enabled,
            "sound_playing": item.sound_playing,
            "sound_status": item.sound_status,
        }
        if item.node_id in label_indices:
            payload["label_widget_index"] = label_indices[item.node_id]
        return payload

    def _native_label_widgets(
        self,
        items: tuple[LegendItem, ...],
    ) -> tuple[tuple[int, object], ...]:
        """Return native Markdown output widgets for Jupyter legend labels."""

        if not self.native_markdown_labels:
            return ()

        ordered = []
        for item in items:
            widget = self.label_widgets.get(item.node_id)
            if widget is None:
                import ipywidgets as ipywidgets

                widget = ipywidgets.Output()
                self.label_widgets[item.node_id] = widget
            if getattr(widget, "_mt_label_markdown", None) != item.label_markdown:
                widget.outputs = _markdown_outputs(item.label_markdown)
                widget._mt_label_markdown = item.label_markdown
            ordered.append((item.node_id, widget))
        return tuple(ordered)

    def _node_for_event(self, node_id: object) -> object | None:
        """Return a live plot node for a frontend event."""

        if not self.generation.accepts_frontend_events():
            return None
        try:
            wanted = int(node_id)
        except (TypeError, ValueError):
            return None
        for node in self.generation.figure.plots:
            if node.id == wanted:
                return node
        return None


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
