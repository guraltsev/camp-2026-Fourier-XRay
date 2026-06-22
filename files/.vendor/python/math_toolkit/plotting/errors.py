"""Define public exceptions raised by notebook plotting."""

from __future__ import annotations


class PlottingError(Exception):
    """Base class for plotting-layer errors."""


class FigureNotFoundError(PlottingError):
    """Report a requested named figure that does not exist."""


class PlotNotFoundError(PlottingError):
    """Report a requested plot that does not exist in a figure."""


class InfoNotFoundError(PlottingError):
    """Report a requested info card that does not exist in a figure."""


class ViewNotFoundError(PlottingError):
    """Report a requested view that does not exist in a figure."""


class PlotSpecError(PlottingError):
    """Report invalid public plotting arguments."""


class PlotCompilationError(PlottingError):
    """Report expressions that cannot cross into numeric plotting."""


class PlotShapeError(PlottingError):
    """Report numeric outputs that are not phase 1 scalar curves."""


class AudioPlaybackError(PlottingError):
    """Report invalid audio playback requests or audio backend failures."""
