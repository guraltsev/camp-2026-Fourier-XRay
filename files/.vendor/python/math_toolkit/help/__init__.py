"""Coordinate the public runtime help API."""

from __future__ import annotations

from . import topics as _topics
from ..util import jupyter_document_link
from .runtime import Help

help = Help

__all__ = ["Help", "help", "jupyter_document_link"]
