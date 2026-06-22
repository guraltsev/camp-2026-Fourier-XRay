"""Expose the public notebook-first plotting commands."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import overload

from .layout import FigureLayout
from .model import FigureHandle, InfoHandle, PlotHandle, ViewHandle
from .session import get_session
from .specs import OMITTED


# Keep this wrapper docstring aligned with ``FigureHandle.plot``; only this
# global command should mention context-aware figure routing.
def plot(
    expr: object,
    domain: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
    samples: object = OMITTED,
) -> PlotHandle:
    """Plot or update one sampled two-dimensional function curve.

    The top-level command is context aware: ``figure=`` selects a named or
    explicit figure, an active ``with figure(...):`` block receives the plot
    when present, and otherwise the current figure is used.

    Parameters
    ----------
    expr : object
        SymPy-compatible scalar expression describing ``y = f(x)``. Symbols
        not used as the independent variable become figure-owned parameters.
    domain : object, optional
        Independent variable symbol such as ``x`` for view-aware sampling, or
        a finite interval tuple such as ``(x, -10, 10)``. Named updates may
        omit it to preserve the existing domain.
    name : str, optional
        Plot identity within the target figure. Reusing a name updates the
        existing curve in place when it is already a curve plot.
    label : object, optional
        Legend display label. It does not define plot identity.
    figure : str | FigureHandle | None, optional
        Per-call figure route as a figure name or ``FigureHandle``.
    style : object, optional
        Line style dictionary supporting ``color``, ``width``, ``opacity``,
        ``visible``, and ``dash``.
    samples : object, optional
        Number of sample points to use for the curve.

    Returns
    -------
    PlotHandle
        Handle for updating style, label, parameters, audio, and plot removal.

    Raises
    ------
    PlotSpecError
        Raised when the expression, domain, style, samples, or update target
        is invalid for a curve plot.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import plot
    >>> x = sympy.Symbol("x")
    >>> handle = plot(sympy.sin(x), x, name="sine")
    >>> handle.name
    'sine'

    Named update:

    >>> updated = plot(sympy.cos(x), name="sine", label="cosine")
    >>> updated.name
    'sine'
    """

    target = get_session().resolve_plot_figure(figure)
    return target.plot(
        expr,
        domain,
        name=name,
        label=label,
        style=style,
        samples=samples,
    )


def list_plot(
    source: object,
    index: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
) -> PlotHandle:
    """Plot or update one finite set of discrete points.

    Parameters
    ----------
    source : object
        Numeric values, explicit ``(x, y)`` values, or an expression/callable
        evaluated at integer index values.
    index : object, optional
        SymPy index symbol for viewport-inferred sampling, or
        ``(n, min, max[, step])`` for a half-open integer range. Value inputs
        do not accept an index argument.
    name : str, optional
        Plot identity within the target figure for in-place updates.
    label : object, optional
        Legend and control display label. It does not define identity.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.
    style : object, optional
        Small style dictionary supporting color, width, opacity, visible, dash.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, and removal updates.
    """

    target = get_session().resolve_plot_figure(figure)
    return target.list_plot(
        source,
        index,
        name=name,
        label=label,
        style=style,
    )


def temperature_plot(
    expr: object,
    x_domain: object = OMITTED,
    y_domain: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
    samples: object = OMITTED,
) -> PlotHandle:
    """Plot or update one sampled scalar field as a heatmap.

    Parameters
    ----------
    expr : object
        SymPy-compatible scalar expression describing ``z = f(x, y)``.
    x_domain, y_domain : object, optional
        Domain symbols or finite domain tuples. Named plot updates may omit
        them after the plot exists.
    name : str, optional
        Plot identity within the target figure for in-place updates.
    label : object, optional
        Legend and control display label. It does not define identity.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.
    style : object, optional
        Style dictionary supporting colorscale, opacity, visible, showscale,
        zmin, zmax, and zsmooth.
    samples : object, optional
        Integer sample count or ``(x_samples, y_samples)`` tuple.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, and removal updates.
    """

    target = get_session().resolve_plot_figure(figure)
    return target.temperature_plot(
        expr,
        x_domain,
        y_domain,
        name=name,
        label=label,
        style=style,
        samples=samples,
    )


def contour_plot(
    expr: object,
    x_domain: object = OMITTED,
    y_domain: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
    samples: object = OMITTED,
) -> PlotHandle:
    """Plot or update one sampled scalar field as contour lines.

    Parameters
    ----------
    expr : object
        SymPy-compatible scalar expression describing ``z = f(x, y)``.
    x_domain, y_domain : object, optional
        Domain symbols or finite domain tuples. Named plot updates may omit
        them after the plot exists.
    name : str, optional
        Plot identity within the target figure for in-place updates.
    label : object, optional
        Legend and control display label. It does not define identity.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.
    style : object, optional
        Style dictionary supporting colorscale, opacity, visible, showscale,
        zmin, zmax, contour_color, contour_width, and line_smoothing.
    samples : object, optional
        Integer sample count or ``(x_samples, y_samples)`` tuple.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, and removal updates.
    """

    target = get_session().resolve_plot_figure(figure)
    return target.contour_plot(
        expr,
        x_domain,
        y_domain,
        name=name,
        label=label,
        style=style,
        samples=samples,
    )


def domain_plot(
    condition: object,
    x_domain: object = OMITTED,
    y_domain: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
    samples: object = OMITTED,
    boundary: bool = True,
) -> PlotHandle:
    """Plot or update one filled Boolean or signed domain.

    Parameters
    ----------
    condition : object
        Symbolic Boolean condition, signed scalar expression, or finite system
        of conditions and signed expressions.
    x_domain, y_domain : object, optional
        Domain symbols or finite domain tuples. Named plot updates may omit
        them after the plot exists.
    name : str, optional
        Plot identity within the target figure for in-place updates.
    label : object, optional
        Legend and control display label. It does not define identity.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.
    style : object, optional
        Nested style dictionary with ``domain`` and ``boundary`` sections.
        Domain fill supports ``zsmooth`` and boundary supports ``smoothing``.
    samples : object, optional
        Integer sample count or ``(x_samples, y_samples)`` tuple.
    boundary : bool, optional
        Whether the boundary contour trace is visible.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, and removal updates.
    """

    target = get_session().resolve_plot_figure(figure)
    return target.domain_plot(
        condition,
        x_domain,
        y_domain,
        name=name,
        label=label,
        style=style,
        samples=samples,
        boundary=boundary,
    )


def parametric_plot(
    exprs: object,
    parameter_domain: object = OMITTED,
    *,
    name: str | None = None,
    label: object = OMITTED,
    figure: str | FigureHandle | None = None,
    style: object = OMITTED,
    samples: object = OMITTED,
) -> PlotHandle:
    """Plot or update one two-dimensional parametric curve.

    Parameters
    ----------
    exprs : object
        Two coordinate expressions supplied as a sequence or SymPy Matrix.
    parameter_domain : object, optional
        Explicit interval tuple such as ``(t, 0, 2*pi)``. Named plot updates
        may omit it after the plot exists.
    name : str, optional
        Plot identity within the target figure for in-place updates.
    label : object, optional
        Legend and control display label. It does not define identity.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.
    style : object, optional
        Small style dictionary supporting color, width, opacity, visible, dash.
    samples : object, optional
        Sample-count override for the parameter interval.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, and removal updates.
    """

    target = get_session().resolve_plot_figure(figure)
    return target.parametric_plot(
        exprs,
        parameter_domain,
        name=name,
        label=label,
        style=style,
        samples=samples,
    )


def get_plot(
    name: object = None,
    *,
    figure: str | FigureHandle | None = None,
) -> PlotHandle:
    """Return the latest plot handle or a named plot handle.

    Parameters
    ----------
    name : object, optional
        Plot identity within the selected figure. When omitted or ``None``, the
        latest plot in the figure is returned.
    figure : object, optional
        Figure route as a name or ``FigureHandle``. When omitted, lookup uses
        the current figure or active figure context.

    Returns
    -------
    PlotHandle
        Lightweight handle for style, parameter, display, and removal updates.

    Examples
    --------
    Latest plot:

    >>> import sympy
    >>> from math_toolkit import get_plot, plot
    >>> x = sympy.Symbol("x")
    >>> plot(sympy.sin(x), (x, -1, 1), name="sine").name
    'sine'
    >>> get_plot().name
    'sine'

    Named plot:

    >>> get_plot("sine").name
    'sine'
    """

    target = get_session().resolve_existing_figure(figure)
    return target.get_plot(name)


def info(
    *fragments: object,
    name: str | None = None,
    title: object = OMITTED,
    params: object = OMITTED,
    figure: str | FigureHandle | None = None,
) -> InfoHandle:
    """Create or update one Markdown info card on a figure.

    Parameters
    ----------
    *fragments : object
        Markdown strings, SymPy expressions, or callables accepting the target
        figure.
    name : str, optional
        Info card identity within the target figure for in-place updates.
    title : object, optional
        Optional Markdown title. Omit it on named updates to preserve the
        previous title.
    params : object, optional
        Parameter slider specs supplied as ``{symbol: value}`` or dictionaries.
    figure : object, optional
        Per-call figure route as a name or ``FigureHandle``.

    Returns
    -------
    InfoHandle
        Lightweight handle for the created or updated info card.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import figure, info
    >>> a = sympy.Symbol("a")
    >>> fig = figure("notes")
    >>> info("a = ", a, figure=fig).figure is fig
    True
    """

    target = get_session().resolve_plot_figure(figure)
    return target.info(
        *fragments,
        name=name,
        title=title,
        params=params,
    )


@overload
def figure(
    target: None = None,
    *,
    layout: type[FigureLayout] | None = None,
    layout_options: dict[str, object] | None = None,
    backend: str | None = None,
    new: bool = False,
) -> FigureHandle: ...


@overload
def figure(
    target: str,
    *,
    layout: type[FigureLayout] | None = None,
    layout_options: dict[str, object] | None = None,
    backend: str | None = None,
    new: bool = False,
) -> FigureHandle: ...


@overload
def figure(
    target: FigureHandle,
    *,
    layout: type[FigureLayout] | None = None,
    layout_options: dict[str, object] | None = None,
    backend: str | None = None,
    new: bool = False,
) -> FigureHandle: ...


def figure(
    target: str | FigureHandle | None = None,
    *,
    layout: type[FigureLayout] | None = None,
    layout_options: dict[str, object] | None = None,
    backend: str | None = None,
    new: bool = False,
) -> FigureHandle:
    """Return or create a figure handle without changing current routing.

    Call ``figure(...)`` when you want a durable plotting container that later
    ``plot(...)`` calls can target explicitly. Named figures are reused by
    name, unnamed figures create a fresh handle each time, and passing an
    existing ``FigureHandle`` returns that same handle. Supplying ``layout=``
    stores the layout class that future display generations should use for the
    selected figure, ``layout_options=`` stores keyword options passed to each
    future layout constructor, and ``backend=`` stores the display backend used
    when ``show()`` or implicit notebook display does not receive an explicit
    backend.

    Parameters
    ----------
    target : object, optional
        Figure selector. Omit it or pass ``None`` to create a new unnamed
        figure, pass a string to create or retrieve a named figure, or pass an
        existing ``FigureHandle`` to reuse it directly.
    layout : object, optional
        Layout class for future display generations of the resolved figure.
        This should be a plotting layout class such as a ``FigureLayout``
        subclass, not a prebuilt layout instance.
    layout_options : dict, optional
        Keyword options passed to the layout constructor after the generated
        layout parts. Meaningful keys are layout-dependent.
    backend : {'anywidget', 'ipywidgets', 'jupyter', 'widget'} or None, optional
        Default display backend for future generations of this figure. ``None``
        selects the package default, currently ``"anywidget"``. The aliases
        ``"ipywidgets"``, ``"jupyter"``, and ``"widget"`` select the legacy
        built-in ipywidgets backend.
    new : bool, optional
        When true with a name, create a fresh named figure even if that name is
        already registered. The previous handle is detached from the name and
        remains usable only through existing Python references.

    Returns
    -------
    FigureHandle
        Durable figure handle that owns plots, views, display generations, and
        future layout selection.

    Raises
    ------
    PlotSpecError
        Raised when ``target`` is not omitted, ``None``, a string, or a
        ``FigureHandle``; when ``new=True`` is supplied with an existing
        ``FigureHandle``; or when ``layout`` is not a valid layout class.

    Notes
    -----
    ``figure(...)`` does not make the returned handle current. Use
    ``set_current_figure(...)`` when you want later unnamed plotting commands
    to route there automatically.

    Default display uses the figure's stored backend. A figure created with
    ``figure("legacy", backend="ipywidgets")`` will use the legacy backend for
    ``fig.show()`` until a later ``figure(fig, backend=...)`` or
    ``figure("legacy", backend=...)`` call changes that default.

    Repeated calls such as ``figure("demo")`` return the same durable named
    handle, which makes them useful for notebook cells that are rerun out of
    order.

    A figure name is a manager lookup key. If ``fig.name`` is a string, then
    ``figure(fig.name) is fig``. Replacing a named figure with
    ``figure("demo", new=True)`` makes the previous handle unnamed.

    Examples
    --------
    Basic usage:

    >>> fig = figure("demo")
    >>> fig.name
    'demo'

    Reusing a named figure:

    >>> same_fig = figure("demo")
    >>> same_fig is fig
    True

    Replacing a named figure:

    >>> old = figure("replace-me")
    >>> fresh = figure("replace-me", new=True)
    >>> old.name is None
    True
    >>> fresh.name
    'replace-me'

    Creating a fresh unnamed figure:

    >>> unnamed = figure()
    >>> unnamed.name is None
    True

    Registering a custom layout class:

    >>> class ExampleLayout:
    ...     def __init__(self, parts):
    ...         self.parts = parts
    ...     def build(self):
    ...         import ipywidgets as ipywidgets
    ...         return ipywidgets.VBox([self.parts.plot, self.parts.controls, self.parts.status])
    >>> styled = figure("styled", layout=ExampleLayout)
    >>> styled.layout_style is ExampleLayout
    True

    Selecting a default backend:

    >>> legacy = figure("legacy", backend="ipywidgets")
    >>> legacy.default_backend
    'ipywidgets'

    See Also
    --------
    current_figure : <function> Return the persistent current figure, creating it lazily.
    set_current_figure : <function> Change the figure targeted by unnamed plotting calls.
    plot : <function> Add or update plots on a selected figure.
    FigureHandle : <class> Durable figure object returned by this factory.
    """

    return get_session().figure(
        target,
        layout=layout,
        layout_options=layout_options,
        backend=backend,
        new=new,
    )


def current_figure() -> FigureHandle:
    """Return the persistent current figure, creating one lazily if needed."""

    return get_session().current_figure()


def set_current_figure(
    target: str | FigureHandle | None = None,
    *,
    layout: type[FigureLayout] | None = None,
    layout_options: dict[str, object] | None = None,
    backend: str | None = None,
    only_existing: bool = False,
) -> FigureHandle:
    """Set and return the persistent current figure.

    Call ``set_current_figure(...)`` when you want later unnamed plotting
    commands such as ``plot(...)`` and ``current_figure().view.current`` to route to one
    durable figure automatically. Passing a name reuses or creates a named
    figure, passing ``None`` creates a fresh unnamed current figure, and
    passing an existing ``FigureHandle`` makes that handle current directly.
    Supplying ``layout=`` stores the layout class that future display
    generations should use for the selected figure. Supplying
    ``layout_options=`` stores keyword options passed to each future layout
    constructor, and ``backend=`` stores the default display backend for future
    generations of the selected figure.

    Parameters
    ----------
    target : str | FigureHandle | None, optional
        Figure selector. Omit it or pass ``None`` to create and select a new
        unnamed figure, pass a string to select a named figure, or pass an
        existing ``FigureHandle`` to make it current directly.
    layout : type[FigureLayout] | None, optional
        Layout class for future display generations of the selected figure.
        This should be a plotting layout class such as a ``FigureLayout``
        subclass, not a prebuilt layout instance.
    layout_options : dict, optional
        Keyword options passed to the layout constructor after the generated
        layout parts. Meaningful keys are layout-dependent.
    backend : {'anywidget', 'ipywidgets', 'jupyter', 'widget'} or None, optional
        Default display backend for future generations of this figure. ``None``
        selects the package default, currently ``"anywidget"``.
    only_existing : bool, optional
        When true, a missing named figure raises instead of being created.

    Returns
    -------
    FigureHandle
        The figure that is now current for unnamed plotting and view commands.

    Raises
    ------
    FigureNotFoundError
        Raised when ``target`` is a missing figure name and
        ``only_existing=True``.
    PlotSpecError
        Raised when ``target`` is not ``None``, a string, or a
        ``FigureHandle``, or when ``layout`` is not a valid layout class.

    Notes
    -----
    The current figure is process-local session state. Once set, unnamed calls
    such as ``plot(expr)`` route to this figure until another routing action
    changes the current target or a temporary figure context overrides it.

    Examples
    --------
    Basic usage:

    >>> fig = set_current_figure("demo")
    >>> current_figure() is fig
    True

    Reusing an existing named figure:

    >>> same_fig = set_current_figure("demo")
    >>> same_fig is fig
    True

    Selecting a current figure and storing its layout class:

    >>> class ExampleLayout:
    ...     def __init__(self, parts):
    ...         self.parts = parts
    ...     def build(self):
    ...         import ipywidgets as ipywidgets
    ...         return ipywidgets.VBox([self.parts.plot, self.parts.controls, self.parts.status])
    >>> styled = set_current_figure("styled", layout=ExampleLayout)
    >>> styled.layout_style is ExampleLayout
    True

    Selecting a specific existing handle:

    >>> other = figure("other")
    >>> set_current_figure(other) is other
    True

    Creating a fresh unnamed current figure:

    >>> unnamed = set_current_figure()
    >>> unnamed.name is None
    True

    See Also
    --------
    figure : <function> Return or create a figure without changing current routing.
    current_figure : <function> Return the persistent current figure, creating it lazily.
    plot : <function> Add or update plots on the current or explicitly selected figure.
    FigureHandle : <class> Durable figure object used for routing and display.
    """

    return get_session().set_current_figure(
        target,
        layout=layout,
        layout_options=layout_options,
        backend=backend,
        only_existing=only_existing,
    )


figure._mt_help = {
    "path": PurePosixPath("library/figure"),
    "anchor": None,
    "label": "figure",
}
plot._mt_help = {
    "path": PurePosixPath("library/plot"),
    "anchor": None,
    "label": "plot",
}
list_plot._mt_help = {
    "path": PurePosixPath("library/list_plot"),
    "anchor": None,
    "label": "list_plot",
}
temperature_plot._mt_help = {
    "path": PurePosixPath("library/temperature_plot"),
    "anchor": None,
    "label": "temperature_plot",
}
contour_plot._mt_help = {
    "path": PurePosixPath("library/contour_plot"),
    "anchor": None,
    "label": "contour_plot",
}
domain_plot._mt_help = {
    "path": PurePosixPath("library/domain_plot"),
    "anchor": None,
    "label": "domain_plot",
}
parametric_plot._mt_help = {
    "path": PurePosixPath("library/parametric_plot"),
    "anchor": None,
    "label": "parametric_plot",
}
get_plot._mt_help = {
    "path": PurePosixPath("library/plot"),
    "anchor": None,
    "label": "plot",
}
info._mt_help = {
    "path": PurePosixPath("library/figure"),
    "anchor": None,
    "label": "info",
}


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
