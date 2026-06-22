"""Reconcile toolkit-owned legend widgets for plotted figures."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from .display import FigureDisplayGeneration
    from .model import LegendItem, LegendMarker


def reconcile_legend(
    generation: FigureDisplayGeneration,
    items: tuple[LegendItem, ...],
) -> None:
    """Reconcile legend row widgets while preserving surviving row identity."""

    wanted_ids = {item.node_id for item in items}

    # Remove vanished plots first so observer lifetimes match the visible row
    # registry before the next ordered children tuple is published.
    for node_id in tuple(generation._legend_widgets):
        if node_id not in wanted_ids:
            _dispose_legend_row(generation, node_id)

    rows = []
    for item in items:
        row = generation._legend_widgets.get(item.node_id)
        if row is None:
            row = _create_legend_row(generation, item)
        else:
            _update_legend_row(row, item)
        rows.append(row)

    generation.layout.set_legend(tuple(rows))


def _create_legend_row(
    generation: FigureDisplayGeneration,
    item: LegendItem,
) -> object:
    """Create one legend row and attach its guarded click callback."""

    import ipywidgets as ipywidgets

    marker_style = ipywidgets.HTML(layout=ipywidgets.Layout(display="none"))
    button = ipywidgets.Button(
        description="",
        tooltip="Toggle plot visibility",
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
    marker_class = f"mt-plot__legend-marker-{generation.generation_id}-{item.node_id}"
    add_class = getattr(button, "add_class", None)
    if add_class is not None:
        add_class("mt-plot__legend-marker-button")
        add_class(marker_class)
    label = ipywidgets.Output(
        layout=ipywidgets.Layout(
            flex="1 1 auto",
            margin="0",
            min_width="0",
            overflow="visible",
        )
    )
    edit_button = ipywidgets.Button(
        description="",
        icon="gear",
        tooltip="Edit plot settings",
        layout=ipywidgets.Layout(
            border="0",
            flex="0 0 1.15rem",
            height="1.15rem",
            margin="0",
            overflow="hidden",
            padding="0",
            width="1.15rem",
        ),
        style={"button_color": "transparent"},
    )
    sound_style = ipywidgets.HTML(layout=ipywidgets.Layout(display="none"))
    sound_button = ipywidgets.Button(
        description="",
        tooltip="Play or pause this curve as sound",
        layout=ipywidgets.Layout(
            border="0",
            display="none",
            flex="0 0 1.15rem",
            height="1.15rem",
            margin="0",
            padding="0",
            width="1.15rem",
        ),
        style={"button_color": "transparent"},
    )
    sound_class = f"mt-plot__legend-sound-{generation.generation_id}-{item.node_id}"
    add_sound_class = getattr(sound_button, "add_class", None)
    if add_sound_class is not None:
        add_sound_class("mt-plot__legend-sound-button")
        add_sound_class(sound_class)
    row = ipywidgets.HBox(
        [marker_style, button, sound_style, sound_button, label, edit_button],
        layout=ipywidgets.Layout(
            align_items="center",
            grid_gap="0.45rem",
            overflow="visible",
            width="100%",
        ),
    )
    row._mt_marker_style = marker_style
    row._mt_marker_class = marker_class
    row._mt_button = button
    row._mt_sound_style = sound_style
    row._mt_sound_button = sound_button
    row._mt_sound_class = sound_class
    row._mt_label = label
    row._mt_edit_button = edit_button
    row._mt_label_markdown = None
    row._mt_node_id = item.node_id
    add_edit_class = getattr(edit_button, "add_class", None)
    if add_edit_class is not None:
        add_edit_class("mt-plot__legend-edit-button")
    _update_legend_row(row, item)

    def _toggle_visibility(_button: object, *, node_id: int = item.node_id) -> None:
        if not generation.accepts_frontend_events():
            return
        node = _plot_node_for_id(generation, node_id)
        if node is not None:
            node.toggle_visible()

    def _toggle_sound(_button: object, *, node_id: int = item.node_id) -> None:
        if not generation.accepts_frontend_events():
            return
        node = _plot_node_for_id(generation, node_id)
        if node is None:
            return
        handle = generation.figure._handle_for_node(node)
        sound = getattr(handle, "sound", None)
        if sound is None or not sound.enabled:
            return
        state = sound.state()
        if state.status == "playing":
            sound.pause()
        else:
            sound.resume()

    def _open_plot_settings(_button: object, *, node_id: int = item.node_id) -> None:
        if not generation.accepts_frontend_events():
            return
        node = _plot_node_for_id(generation, node_id)
        if node is None:
            return
        from .editors import PlotStylePanel

        generation.modal.open(PlotStylePanel(generation.figure, node))

    button.on_click(_toggle_visibility)
    sound_button.on_click(_toggle_sound)
    edit_button.on_click(_open_plot_settings)
    generation._legend_widgets[item.node_id] = row
    generation._legend_observers[item.node_id] = (button, _toggle_visibility)
    generation._legend_sound_observers[item.node_id] = (sound_button, _toggle_sound)
    generation._legend_edit_observers[item.node_id] = (edit_button, _open_plot_settings)
    return row


def _update_legend_row(row: object, item: LegendItem) -> None:
    """Update a legend row in place from one snapshot item."""

    button = getattr(row, "_mt_button", None)
    marker_style = getattr(row, "_mt_marker_style", None)
    marker_class = getattr(row, "_mt_marker_class", None)
    if button is not None and marker_style is not None and marker_class is not None:
        _update_marker_button(button, marker_style, marker_class, item.marker)

    sound_style = getattr(row, "_mt_sound_style", None)
    sound_button = getattr(row, "_mt_sound_button", None)
    sound_class = getattr(row, "_mt_sound_class", None)
    if sound_style is not None and sound_button is not None and sound_class is not None:
        _update_sound_button(sound_button, sound_style, sound_class, item)

    label = getattr(row, "_mt_label", None)
    if label is not None and getattr(row, "_mt_label_markdown", None) != item.label_markdown:
        label.outputs = (
            {
                "output_type": "display_data",
                "data": {"text/markdown": item.label_markdown},
                "metadata": {},
            },
        )
        row._mt_label_markdown = item.label_markdown

    opacity = "1" if item.visible else "0.55"
    row.layout.opacity = opacity


def _dispose_legend_row(generation: FigureDisplayGeneration, node_id: int) -> None:
    """Remove one legend row observer and registry entries exactly once."""

    generation._legend_widgets.pop(node_id, None)
    observer = generation._legend_observers.pop(node_id, None)
    sound_observer = generation._legend_sound_observers.pop(node_id, None)
    edit_observer = generation._legend_edit_observers.pop(node_id, None)
    if observer is not None:
        button, callback = observer
        button.on_click(callback, remove=True)
    if sound_observer is not None:
        sound_button, sound_callback = sound_observer
        sound_button.on_click(sound_callback, remove=True)
    if edit_observer is not None:
        edit_button, edit_callback = edit_observer
        edit_button.on_click(edit_callback, remove=True)


def _update_marker_button(
    button: object,
    marker_style: object,
    marker_class: str,
    marker: LegendMarker,
) -> None:
    """Apply an SVG circle background to the clickable marker button."""

    button.description = ""
    button.style.button_color = "transparent"
    button.layout.border = "0"
    button.layout.overflow = "hidden"
    button.layout.margin = "0"
    button.layout.padding = "0"
    marker_style.value = _marker_style_html(marker_class, marker)


def _marker_style_html(marker_class: str, marker: LegendMarker) -> str:
    """Return a scoped CSS rule that paints one marker button with SVG."""

    svg_url = _marker_svg_data_url(marker)
    class_name = escape(marker_class, quote=True)
    opacity = min(1.0, max(0.0, float(marker.opacity)))
    return (
        "<style>"
        f".{class_name} {{"
        f"background-image: url('{svg_url}') !important;"
        "background-position: center !important;"
        "background-repeat: no-repeat !important;"
        "background-size: 1rem 1rem !important;"
        "border: 0 !important;"
        "box-shadow: none !important;"
        "min-width: 1.15rem !important;"
        "outline: 0 !important;"
        "overflow: hidden !important;"
        f"opacity: {opacity:.2f} !important;"
        "}}"
        f".{class_name}:active,"
        f".{class_name}:focus,"
        f".{class_name}:focus-visible,"
        f".{class_name}.mod-active {{"
        "border: 0 !important;"
        "box-shadow: none !important;"
        "outline: 0 !important;"
        "}}"
        f".{class_name}:hover {{ filter: brightness(0.96); }}"
        "</style>"
    )


def _update_sound_button(
    button: object,
    sound_style: object,
    sound_class: str,
    item: LegendItem,
) -> None:
    """Show or hide the legend sound button for one playable curve."""

    button.description = ""
    button.tooltip = "Pause sound" if item.sound_playing else "Play sound"
    button.style.button_color = "transparent"
    button.layout.border = "0"
    button.layout.display = (
        "block" if item.sound_playable and item.sound_enabled else "none"
    )
    button.layout.overflow = "hidden"
    button.layout.margin = "0"
    button.layout.padding = "0"
    button.layout.width = "1.15rem"
    button.layout.height = "1.15rem"
    sound_style.value = _sound_style_html(sound_class, item.sound_status)


def _sound_style_html(sound_class: str, status: str) -> str:
    """Return a scoped CSS rule that paints one sound button with SVG."""

    svg_url = _sound_svg_data_url(status)
    class_name = escape(sound_class, quote=True)
    return (
        "<style>"
        f".{class_name} {{"
        f"background-image: url('{svg_url}') !important;"
        "background-position: center !important;"
        "background-repeat: no-repeat !important;"
        "background-size: 0.95rem 0.95rem !important;"
        "border: 0 !important;"
        "box-shadow: none !important;"
        "min-width: 1.15rem !important;"
        "outline: 0 !important;"
        "overflow: hidden !important;"
        "}}"
        f".{class_name}:active,"
        f".{class_name}:focus,"
        f".{class_name}:focus-visible,"
        f".{class_name}.mod-active {{"
        "border: 0 !important;"
        "box-shadow: none !important;"
        "outline: 0 !important;"
        "}}"
        f".{class_name}:hover {{ filter: brightness(0.92); }}"
        "</style>"
    )


def _sound_svg_data_url(status: str) -> str:
    """Return a compact data URL for one sound-control SVG."""

    fill = "#0f172a" if status == "playing" else "#64748b"
    speaker = (
        f"<path d='M2 6.2h2.6L8.5 3v10L4.6 9.8H2z' "
        f"fill='{escape(fill, quote=True)}'/>"
    )
    pause = (
        "<rect x='5.0' y='4.0' width='2.0' height='8.0' "
        "rx='0.35' fill='currentColor'/>"
        "<rect x='9.0' y='4.0' width='2.0' height='8.0' "
        "rx='0.35' fill='currentColor'/>"
    )
    waves = (
        "<path d='M11 5.2c1.1 1.5 1.1 4.1 0 5.6' "
        "fill='none' stroke='currentColor' stroke-width='1.2' "
        "stroke-linecap='round'/>"
        "<path d='M13 3.4c1.8 2.5 1.8 6.7 0 9.2' "
        "fill='none' stroke='currentColor' stroke-width='1.1' "
        "stroke-linecap='round'/>"
    )
    body = pause if status == "paused" else speaker
    badge = waves if status == "playing" else ""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' "
        f"style='color:{escape(fill, quote=True)}'>"
        f"{body}"
        f"{badge}"
        "</svg>"
    )
    return "data:image/svg+xml," + quote(svg, safe="")


def _marker_svg_data_url(marker: LegendMarker) -> str:
    """Return a compact data URL for one SVG circle marker."""

    fill = marker.fill_color or "transparent"
    stroke = marker.border_color or marker.fill_color or "#64748b"
    stroke_width = min(3.0, max(1.0, float(marker.border_width)))
    dasharray = _svg_dasharray(marker.border_dash)
    dash_attribute = f" stroke-dasharray='{dasharray}'" if dasharray else ""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        f"<circle cx='8' cy='8' r='5.5' fill='{escape(fill, quote=True)}' "
        f"stroke='{escape(stroke, quote=True)}' "
        f"stroke-width='{stroke_width:.2f}' "
        f"{dash_attribute}/>"
        "</svg>"
    )
    return "data:image/svg+xml," + quote(svg, safe="")


def _svg_dasharray(dash: str) -> str:
    """Map Plotly dash names to simple SVG stroke dash arrays."""

    if dash == "dot":
        return "1.2 2.0"
    if dash in {"dash", "dashdot", "longdash", "longdashdot"}:
        return "3.0 2.0"
    return ""


def _plot_node_for_id(generation: FigureDisplayGeneration, node_id: int) -> object | None:
    """Return the current plot node matching a legend row id."""

    for node in generation.figure.plots:
        if node.id == node_id:
            return node
    return None
