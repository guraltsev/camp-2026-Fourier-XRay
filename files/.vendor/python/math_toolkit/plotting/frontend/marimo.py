"""Expose Marimo host bindings for the anywidget plotting frontend."""

from __future__ import annotations

from .backend import MarimoFrontendBackend
from .host import MarimoAnywidgetRoot, running_in_marimo

__all__ = ["MarimoAnywidgetRoot", "MarimoFrontendBackend", "running_in_marimo"]
