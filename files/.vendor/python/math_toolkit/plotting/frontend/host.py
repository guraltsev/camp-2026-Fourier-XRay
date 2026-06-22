"""Describe host-specific anywidget display policies."""

from __future__ import annotations

from dataclasses import dataclass

from .markdown import render_markdown_payload

__all__ = [
    "AnywidgetHostAdapter",
    "JupyterHostAdapter",
    "MarimoAnywidgetRoot",
    "MarimoHostAdapter",
    "create_host_adapter",
    "running_in_marimo",
]


@dataclass(frozen=True)
class AnywidgetHostAdapter:
    """Provide display and frontend policy values for one widget host."""

    host_name: str = "jupyter"
    root_class: str = "mt-host-jupyter"

    @property
    def child_mount_policy(self) -> dict[str, object]:
        """Return frontend policy for nested child widget mounting."""

        return {
            "hosted_node_lookup": False,
            "hosted_node_timeout_ms": 0,
            "widget_manager_view": True,
            "local_anywidget_view": True,
        }

    @property
    def markdown_policy(self) -> dict[str, object]:
        """Return frontend policy for toolkit Markdown rendering."""

        return {
            "renderer": "host",
            "fallback": "none",
        }

    def markdown_payload(self, markdown: str) -> dict[str, str]:
        """Return a frontend Markdown payload for this host."""

        return render_markdown_payload(markdown)

    @property
    def native_markdown_widgets(self) -> bool:
        """Return whether Markdown fragments should mount native outputs."""

        return True

    def root_for_display(
        self,
        widget: object,
        *,
        hosted_widgets: tuple[object, ...] = (),
    ) -> object:
        """Return the object that should be displayed for the host."""

        return widget


class JupyterHostAdapter(AnywidgetHostAdapter):
    """Use the ordinary Jupyter widget manager for nested child views."""


@dataclass(frozen=True)
class MarimoHostAdapter(AnywidgetHostAdapter):
    """Use marimo's separately hosted anywidget nodes when needed."""

    host_name: str = "marimo"
    root_class: str = "mt-host-marimo"

    @property
    def child_mount_policy(self) -> dict[str, object]:
        """Return frontend policy for Marimo hosted-node lookup."""

        return {
            "hosted_node_lookup": True,
            "hosted_node_timeout_ms": 3000,
            "widget_manager_view": True,
            "local_anywidget_view": True,
        }

    def markdown_payload(self, markdown: str) -> dict[str, str]:
        """Return Markdown with Marimo-rendered HTML when Marimo provides it."""

        text = str(markdown)
        return render_markdown_payload(
            text,
            rendered_html=_render_marimo_markdown_html(text),
        )

    @property
    def native_markdown_widgets(self) -> bool:
        """Return whether Markdown fragments should mount native outputs."""

        return False

    def root_for_display(
        self,
        widget: object,
        *,
        hosted_widgets: tuple[object, ...] = (),
    ) -> object:
        """Return a marimo MIME wrapper that hosts child widgets separately."""

        return MarimoAnywidgetRoot(widget, hosted_widgets=hosted_widgets)


def create_host_adapter() -> AnywidgetHostAdapter:
    """Return the active anywidget host adapter."""

    if running_in_marimo():
        return MarimoHostAdapter()
    return JupyterHostAdapter()


def running_in_marimo() -> bool:
    """Return whether the current Python execution belongs to a marimo app."""

    try:
        from marimo._runtime.context import get_context
    except Exception:
        return False

    try:
        get_context()
    except Exception:
        return False
    return True


class MarimoAnywidgetRoot:
    """Expose one anywidget as a marimo MIME object without changing its model."""

    def __init__(self, widget: object, *, hosted_widgets: tuple[object, ...]) -> None:
        """Create a marimo display wrapper around a toolkit widget."""

        self.widget = widget
        self.hosted_widgets = hosted_widgets

    def _mime_(self) -> tuple[str, object]:
        """Return marimo's MIME payload for the wrapped anywidget."""

        import marimo as mo

        return mo.vstack(
            [
                mo.ui.anywidget(self.widget),
                *(mo.ui.anywidget(widget) for widget in self.hosted_widgets),
            ]
        )._mime_()


def _render_marimo_markdown_html(markdown: str) -> str:
    """Return Marimo-rendered Markdown HTML when available."""

    try:
        from marimo._output.md import md as marimo_md
    except Exception:
        return ""

    try:
        return str(marimo_md(markdown).text)
    except Exception:
        return ""
