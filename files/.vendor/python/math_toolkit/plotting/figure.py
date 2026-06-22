"""Compatibility exports for notebook figures and related plotting errors."""

from __future__ import annotations

from .errors import (
    AudioPlaybackError,
    FigureNotFoundError,
    InfoNotFoundError,
    PlotNotFoundError,
    PlottingError,
    ViewNotFoundError,
)
from .audio import AudioPlaybackState
from .model import (
    CurvePlotHandle,
    FigureHandle,
    FigureView,
    InfoHandle,
    LegendItem,
    LegendMarker,
    PLOT_NAMED_COLOR_HEX,
    PLOT_NAMED_COLORS,
    PlotHandle,
    PlotStyle,
    ViewHandle,
)

__all__ = [
    "AudioPlaybackError",
    "AudioPlaybackState",
    "CurvePlotHandle",
    "FigureHandle",
    "FigureView",
    "InfoHandle",
    "LegendItem",
    "LegendMarker",
    "PLOT_NAMED_COLOR_HEX",
    "PLOT_NAMED_COLORS",
    "PlotHandle",
    "PlotStyle",
    "ViewHandle",
    "PlottingError",
    "FigureNotFoundError",
    "InfoNotFoundError",
    "PlotNotFoundError",
    "ViewNotFoundError",
]
