"""Provide the lightweight figure output sink for the anywidget backend."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stdout
import io

from math_toolkit.display_context import push_display_sink

from .markdown import html_block, render_markdown_payload, stdout_block

__all__ = ["FigureMessageOutput"]


class FigureMessageOutput:
    """Capture stdout and toolkit display fragments for one figure generation."""

    def __init__(
        self,
        shell: object,
        *,
        markdown_payload: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """Create an output sink attached to an anywidget shell."""

        self.shell = shell
        self.markdown_payload = markdown_payload or render_markdown_payload
        self.items: list[dict[str, str]] = []
        self._stdout_buffer: io.StringIO | None = None
        self._stdout_context: object | None = None
        self._display_context: object | None = None

    def __enter__(self) -> FigureMessageOutput:
        """Capture stdout and toolkit Markdown while the context is active."""

        self._stdout_buffer = io.StringIO()
        self._stdout_context = redirect_stdout(self._stdout_buffer)
        self._display_context = push_display_sink(self)
        self._stdout_context.__enter__()
        self._display_context.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Flush captured stdout and restore display routing."""

        try:
            if self._display_context is not None:
                self._display_context.__exit__(exc_type, exc, traceback)
            if self._stdout_context is not None:
                self._stdout_context.__exit__(exc_type, exc, traceback)
        finally:
            if self._stdout_buffer is not None:
                text = self._stdout_buffer.getvalue()
                if text:
                    self.append_stdout(text)
            self._stdout_buffer = None
            self._stdout_context = None
            self._display_context = None

    def append_stdout(self, text: str) -> None:
        """Append a plain stdout text block."""

        self._append(stdout_block(text))

    def append_markdown(self, markdown: str) -> None:
        """Append a Markdown display block."""

        self._append(self.markdown_payload(markdown))

    def append_html(self, html: str) -> None:
        """Append a trusted HTML display block."""

        self._append(html_block(html))

    def clear(self) -> None:
        """Clear all output blocks from the sink and shell."""

        self.items.clear()
        set_output = getattr(self.shell, "set_output", None)
        if callable(set_output):
            set_output(tuple(self.items))

    def _append(self, item: dict[str, str]) -> None:
        """Store one block and publish the immutable output payload."""

        self.items.append(item)
        set_output = getattr(self.shell, "set_output", None)
        if callable(set_output):
            set_output(tuple(self.items))
