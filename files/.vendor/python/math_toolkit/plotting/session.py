"""Maintain process-local plotting figures and route ``plot(...)`` calls."""

from __future__ import annotations

from .errors import FigureNotFoundError, PlotSpecError
from .layout import FigureLayout
from .model import FigureHandle


class PlottingSession:
    """Store the process-local notebook plotting session."""

    def __init__(self) -> None:
        """Create an empty plotting session."""

        self.current: FigureHandle | None = None
        self.figure_context_stack: list[FigureHandle] = []
        self.named_figures: dict[str, FigureHandle] = {}
        self.unnamed_figures: list[FigureHandle] = []

    def figure(
        self,
        target: str | FigureHandle | None = None,
        *,
        layout: type[FigureLayout] | None = None,
        layout_options: dict[str, object] | None = None,
        backend: str | None = None,
        new: bool = False,
    ) -> FigureHandle:
        """Return or create a figure handle without changing current routing."""

        if isinstance(target, FigureHandle):
            if new:
                raise PlotSpecError("figure(..., new=True) requires a name or no target.")
            if layout is not None:
                target.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                target.set_layout_options(layout_options)
            if backend is not None:
                target.set_default_backend(backend)
            return target
        if target is None:
            handle = FigureHandle(backend=backend)
            if layout is not None:
                handle.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                handle.set_layout_options(layout_options)
            self.unnamed_figures.append(handle)
            return handle
        if isinstance(target, str):
            handle = None if new else self.named_figures.get(target)
            if handle is None:
                old = self.named_figures.get(target)
                if old is not None:
                    old._set_manager_name(None)
                    self._track_unnamed(old)
                handle = FigureHandle(name=target, backend=backend)
                self.named_figures[target] = handle
            if layout is not None:
                handle.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                handle.set_layout_options(layout_options)
            if backend is not None:
                handle.set_default_backend(backend)
            return handle
        raise PlotSpecError("figure(...) expects no argument, a name, or a FigureHandle.")

    def set_current_figure(
        self,
        target: str | FigureHandle | None = None,
        *,
        layout: type[FigureLayout] | None = None,
        layout_options: dict[str, object] | None = None,
        backend: str | None = None,
        only_existing: bool = False,
    ) -> FigureHandle:
        """Set and return the persistent current figure."""

        if isinstance(target, FigureHandle):
            if layout is not None:
                target.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                target.set_layout_options(layout_options)
            if backend is not None:
                target.set_default_backend(backend)
            self.current = target
            return target
        if target is None:
            handle = FigureHandle(backend=backend)
            if layout is not None:
                handle.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                handle.set_layout_options(layout_options)
            self.unnamed_figures.append(handle)
            self.current = handle
            return handle
        if isinstance(target, str):
            handle = self.named_figures.get(target)
            if handle is None:
                if only_existing:
                    raise FigureNotFoundError(
                        f"No figure named {target!r} exists. Use figure({target!r}) "
                        "to create it, or call set_current_figure("
                        f"{target!r}) without only_existing=True."
                    )
                handle = FigureHandle(name=target, backend=backend)
                self.named_figures[target] = handle
            if layout is not None:
                handle.set_layout(layout, layout_options=layout_options)
            elif layout_options is not None:
                handle.set_layout_options(layout_options)
            if backend is not None:
                handle.set_default_backend(backend)
            self.current = handle
            return handle
        raise PlotSpecError(
            "set_current_figure(...) expects no argument, a name, or a FigureHandle."
        )

    def current_figure(self) -> FigureHandle:
        """Return the persistent current figure, creating it lazily."""

        if self.current is None:
            self.current = self.set_current_figure()
        return self.current

    def resolve_plot_figure(
        self,
        explicit: str | FigureHandle | None = None,
    ) -> FigureHandle:
        """Resolve the target figure for one public ``plot(...)`` call."""

        if explicit is not None:
            return self.figure(explicit)
        if self.figure_context_stack:
            return self.figure_context_stack[-1]
        return self.current_figure()

    def resolve_existing_figure(
        self,
        explicit: str | FigureHandle | None = None,
    ) -> FigureHandle:
        """Resolve a figure for lookup without creating missing named figures."""

        if isinstance(explicit, FigureHandle):
            return explicit
        if explicit is None:
            if self.figure_context_stack:
                return self.figure_context_stack[-1]
            return self.current_figure()
        if isinstance(explicit, str):
            handle = self.named_figures.get(explicit)
            if handle is None:
                raise FigureNotFoundError(
                    f"No figure named {explicit!r} exists. Use figure({explicit!r}) "
                    "to create it before looking up plots."
                )
            return handle
        raise PlotSpecError(
            "get_plot(..., figure=...) expects a name or a FigureHandle."
        )

    def push_figure(self, handle: FigureHandle) -> None:
        """Push a temporary context routing figure."""

        self.figure_context_stack.append(handle)

    def pop_figure(self, handle: FigureHandle) -> None:
        """Pop a temporary context routing figure."""

        if not self.figure_context_stack or self.figure_context_stack[-1] is not handle:
            raise RuntimeError("Figure context stack is out of order.")
        self.figure_context_stack.pop()

    def close(self) -> None:
        """Dispose every figure owned by this session."""

        seen = set()
        for handle in [*self.unnamed_figures, *self.named_figures.values()]:
            if handle.id in seen:
                continue
            seen.add(handle.id)
            handle.close()
        self.current = None
        self.figure_context_stack.clear()
        self.named_figures.clear()
        self.unnamed_figures.clear()

    def _track_unnamed(self, handle: FigureHandle) -> None:
        """Track a detached figure for later session cleanup."""

        if all(existing is not handle for existing in self.unnamed_figures):
            self.unnamed_figures.append(handle)


_SESSION: PlottingSession | None = None


def get_session() -> PlottingSession:
    """Return the process-local plotting session."""

    global _SESSION
    if _SESSION is None:
        _SESSION = PlottingSession()
    return _SESSION


def _reset_session() -> None:
    """Reset process-local plotting state for tests and development."""

    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
    _SESSION = PlottingSession()
