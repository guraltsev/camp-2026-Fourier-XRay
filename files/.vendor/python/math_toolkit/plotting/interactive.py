"""Compatibility exports for the public notebook-first plotting commands."""

from __future__ import annotations

from .api import (
    contour_plot,
    current_figure,
    domain_plot,
    figure,
    get_plot,
    info,
    list_plot,
    parametric_plot,
    plot,
    set_current_figure,
    temperature_plot,
)
from .figure import FigureHandle, InfoHandle, PlotHandle, ViewHandle

__all__ = [
    "FigureHandle",
    "InfoHandle",
    "PlotHandle",
    "ViewHandle",
    "contour_plot",
    "current_figure",
    "domain_plot",
    "figure",
    "get_plot",
    "info",
    "list_plot",
    "parametric_plot",
    "plot",
    "set_current_figure",
    "temperature_plot",
]
