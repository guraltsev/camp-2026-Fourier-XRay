"""Route toolkit-authored rich text to an active display sink when present."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "ToolkitDisplayContext",
    "display_html",
    "display_markdown",
    "push_display_sink",
]


class ToolkitDisplaySink(Protocol):
    """Accept trusted toolkit display fragments."""

    def append_markdown(self, markdown: str) -> None:
        """Append a Markdown fragment to the sink."""

    def append_html(self, html: str) -> None:
        """Append a trusted HTML fragment to the sink."""


_DISPLAY_SINK_STACK: ContextVar[tuple[ToolkitDisplaySink, ...]] = ContextVar(
    "math_toolkit_display_sink_stack",
    default=(),
)


@dataclass
class ToolkitDisplayContext:
    """Manage one active toolkit display sink for a ``with`` block."""

    sink: ToolkitDisplaySink
    _token: object | None = None

    def __enter__(self) -> ToolkitDisplayContext:
        """Push the sink onto the current context-local display stack."""

        stack = _DISPLAY_SINK_STACK.get()
        self._token = _DISPLAY_SINK_STACK.set((*stack, self.sink))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Restore the previous display stack."""

        if self._token is not None:
            _DISPLAY_SINK_STACK.reset(self._token)
            self._token = None


def push_display_sink(sink: ToolkitDisplaySink) -> ToolkitDisplayContext:
    """Return a context manager that routes toolkit output to ``sink``."""

    return ToolkitDisplayContext(sink)


def display_markdown(markdown: str) -> bool:
    """Display Markdown through the active toolkit sink when one exists."""

    stack = _DISPLAY_SINK_STACK.get()
    if not stack:
        return False
    stack[-1].append_markdown(markdown)
    return True


def display_html(html: str) -> bool:
    """Display trusted HTML through the active toolkit sink when one exists."""

    stack = _DISPLAY_SINK_STACK.get()
    if not stack:
        return False
    stack[-1].append_html(html)
    return True
