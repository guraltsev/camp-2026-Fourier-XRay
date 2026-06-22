"""Normalize public plotting arguments into concrete plot specifications."""

from __future__ import annotations

class PlotSpecError(Exception):
    """Report invalid public plotting arguments."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
import re
from typing import Any

import numpy as np
import sympy
from sympy.logic.boolalg import Boolean

from math_toolkit.num import autodetect_runtime_args, indexed_runtime_parameter_info

from .errors import PlotSpecError

OMITTED = object()

PLOT_KIND_CURVE = "curve"
PLOT_KIND_TEMPERATURE = "temperature"
PLOT_KIND_CONTOUR = "contour"
PLOT_KIND_DOMAIN = "domain"
PLOT_KIND_PARAMETRIC = "parametric"
PLOT_KIND_LIST = "list"

TRACE_ROLE_MAIN = "main"
TRACE_ROLE_DOMAIN_FILL = "domain-fill"
TRACE_ROLE_DOMAIN_BOUNDARY = "domain-boundary"

# This is the initial visible y range for unbounded Cartesian plots. The x
# range is derived from the current layout policy so the first view is wider
# than it is tall on the toolkit's default figure shape.
DEFAULT_VIEW_Y_HALF_WIDTH = 1.7
DEFAULT_LAYOUT_WIDTH = 700.0
DEFAULT_LAYOUT_HEIGHT = 450.0
_DIGIT_RUN = re.compile(r"\d+")


@dataclass(frozen=True)
class AxisDomain:
    """Describe a declared mathematical axis domain."""

    symbol: sympy.Symbol
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class AxisView:
    """Describe the currently visible range for one axis."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class CurveView:
    """Describe the declared x-domain and active view for one real curve."""

    x_domain: AxisDomain
    x_view: AxisView
    samples: int

    @property
    def domain_symbol(self) -> sympy.Symbol:
        """Return the independent variable for backward-compatible callers."""

        return self.x_domain.symbol

    @property
    def x_min(self) -> float:
        """Return the declared minimum or active visible minimum."""

        return self.x_domain.minimum if self.x_domain.minimum is not None else self.x_view.minimum

    @property
    def x_max(self) -> float:
        """Return the declared maximum or active visible maximum."""

        return self.x_domain.maximum if self.x_domain.maximum is not None else self.x_view.maximum

    @property
    def view_x_min(self) -> float:
        """Return the active visible minimum."""

        return self.x_view.minimum

    @property
    def view_x_max(self) -> float:
        """Return the active visible maximum."""

        return self.x_view.maximum


@dataclass(frozen=True)
class CartesianView2D:
    """Describe declared domains and active views for a Cartesian field."""

    x_domain: AxisDomain
    y_domain: AxisDomain
    x_view: AxisView
    y_view: AxisView
    x_samples: int
    y_samples: int


@dataclass(frozen=True)
class ParametricView:
    """Describe the declared parameter interval for a parametric curve."""

    parameter_symbol: sympy.Symbol
    minimum: float
    maximum: float
    samples: int


@dataclass(frozen=True)
class ListView:
    """Describe the integer index domain for one discrete list plot."""

    index_symbol: sympy.Symbol | None = None
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1
    x_view: AxisView | None = None

    @property
    def inferred(self) -> bool:
        """Return whether this list plot samples from the active x viewport."""

        return (
            self.index_symbol is not None
            and self.minimum is None
            and self.maximum is None
        )


@dataclass(frozen=True)
class ListPlotSpec:
    """Describe a normalized discrete list-plot source."""

    source_kind: str
    source: object


@dataclass(frozen=True)
class ParameterMetadata:
    """Describe the visible slider metadata for one parameter."""

    minimum: float
    maximum: float
    step: float
    label: str | None = None


@dataclass(frozen=True)
class ParameterSpec:
    """Describe one active plot parameter and its slider state."""

    symbol: sympy.Basic
    value: float
    metadata: ParameterMetadata


@dataclass(frozen=True)
class DomainConditionSpec:
    """Describe one Boolean condition or signed expression in a domain system."""

    expression: object
    role: str


def normalize_expression(expr: object) -> object:
    """Return a SymPy-compatible expression while preserving matrix objects."""

    if isinstance(expr, sympy.MatrixBase | sympy.Basic):
        return expr
    try:
        return sympy.sympify(expr)
    except sympy.SympifyError:
        return expr


def default_initial_2d_view(
    *,
    layout_width: float = DEFAULT_LAYOUT_WIDTH,
    layout_height: float = DEFAULT_LAYOUT_HEIGHT,
) -> tuple[AxisView, AxisView]:
    """Return the default visible Cartesian view for unbounded plot domains."""

    height = layout_height if layout_height > 0 else DEFAULT_LAYOUT_HEIGHT
    ratio = layout_width / height if layout_width > 0 else DEFAULT_LAYOUT_WIDTH / height
    y_half_width = DEFAULT_VIEW_Y_HALF_WIDTH
    x_half_width = y_half_width * ratio
    return (
        AxisView(minimum=-x_half_width, maximum=x_half_width),
        AxisView(minimum=-y_half_width, maximum=y_half_width),
    )


def normalize_domain(
    domain: object = OMITTED,
    *,
    samples: object = OMITTED,
    existing: CurveView | None = None,
) -> CurveView:
    """Normalize the public single-axis curve domain grammar."""

    if domain is OMITTED or domain is None:
        if existing is None:
            raise PlotSpecError(
                "plot(...) needs an explicit x variable, such as plot(expr, x) "
                "or plot(expr, (x, -10, 10)). Plot variables are never "
                "inferred from expression free symbols."
            )
        if samples is OMITTED or samples is None:
            return existing
        return replace(existing, samples=normalize_sample_count(samples))

    x_default_view, _ = default_initial_2d_view()
    x_domain = normalize_axis_domain(
        domain,
        plotter="plot",
        axis_name="x",
    )
    sample_count = 400 if samples is OMITTED or samples is None else normalize_sample_count(samples)
    return CurveView(
        x_domain=x_domain,
        x_view=_initial_axis_view(x_domain, x_default_view),
        samples=sample_count,
    )


def normalize_cartesian_view(
    x_domain: object = OMITTED,
    y_domain: object = OMITTED,
    *,
    samples: object = OMITTED,
    existing: CartesianView2D | None = None,
    plotter: str,
) -> CartesianView2D:
    """Normalize the public two-axis Cartesian domain grammar."""

    if x_domain is OMITTED or y_domain is OMITTED or x_domain is None or y_domain is None:
        if existing is None:
            raise PlotSpecError(
                f"{plotter}(...) needs explicit x and y variables, such as "
                f"{plotter}(expr, x, y) or "
                f"{plotter}(expr, (x, -2, 2), (y, -2, 2)). Plot variables "
                "are never inferred from expression free symbols."
            )
        if samples is OMITTED or samples is None:
            return existing
        x_samples, y_samples = normalize_grid_sample_count(samples)
        return replace(existing, x_samples=x_samples, y_samples=y_samples)

    default_x_view, default_y_view = default_initial_2d_view()
    normalized_x = normalize_axis_domain(
        x_domain,
        plotter=plotter,
        axis_name="x",
    )
    normalized_y = normalize_axis_domain(
        y_domain,
        plotter=plotter,
        axis_name="y",
    )
    x_samples, y_samples = (
        (150, 150)
        if samples is OMITTED or samples is None
        else normalize_grid_sample_count(samples)
    )
    return CartesianView2D(
        x_domain=normalized_x,
        y_domain=normalized_y,
        x_view=_initial_axis_view(normalized_x, default_x_view),
        y_view=_initial_axis_view(normalized_y, default_y_view),
        x_samples=x_samples,
        y_samples=y_samples,
    )


def normalize_axis_domain(
    domain: object,
    *,
    plotter: str,
    axis_name: str,
) -> AxisDomain:
    """Normalize one public axis domain as a symbol or finite tuple."""

    if isinstance(domain, sympy.Symbol):
        return AxisDomain(symbol=domain)

    if not isinstance(domain, tuple):
        raise PlotSpecError(
            f"{plotter}(...) expects the {axis_name}-domain as a symbol or "
            f"({axis_name}, minimum, maximum)."
        )
    if len(domain) == 4:
        raise PlotSpecError(
            "Plot domain tuples use only (symbol, minimum, maximum). Use "
            "samples=... for sample counts. Step-size domains are not "
            "supported."
        )
    if len(domain) != 3:
        raise PlotSpecError(
            f"{plotter}(...) expects the {axis_name}-domain as a symbol or "
            f"({axis_name}, minimum, maximum)."
        )

    symbol = domain[0]
    if not isinstance(symbol, sympy.Symbol):
        raise PlotSpecError(f"The {axis_name}-domain variable must be a SymPy symbol.")

    return AxisDomain(
        symbol=symbol,
        minimum=_finite_float(domain[1], f"{axis_name}-domain minimum"),
        maximum=_finite_float(domain[2], f"{axis_name}-domain maximum"),
    )


def normalize_parametric_view(
    parameter_domain: object = OMITTED,
    *,
    samples: object = OMITTED,
    existing: ParametricView | None = None,
) -> ParametricView:
    """Normalize the explicit parameter interval for a parametric curve."""

    if parameter_domain is OMITTED or parameter_domain is None:
        if existing is None:
            raise PlotSpecError(
                "parametric_plot(...) needs an explicit parameter interval, "
                "such as parametric_plot((cos(t), sin(t)), (t, 0, 2*pi))."
            )
        if samples is OMITTED or samples is None:
            return existing
        return replace(existing, samples=normalize_sample_count(samples))

    if not isinstance(parameter_domain, tuple) or len(parameter_domain) != 3:
        raise PlotSpecError(
            "parametric_plot(...) needs an explicit parameter interval, "
            "such as parametric_plot((cos(t), sin(t)), (t, 0, 2*pi))."
        )
    symbol = parameter_domain[0]
    if not isinstance(symbol, sympy.Symbol):
        raise PlotSpecError("The parametric_plot(...) parameter must be a SymPy symbol.")

    return ParametricView(
        parameter_symbol=symbol,
        minimum=_finite_float(parameter_domain[1], "parameter minimum"),
        maximum=_finite_float(parameter_domain[2], "parameter maximum"),
        samples=1000
        if samples is OMITTED or samples is None
        else normalize_sample_count(samples),
    )


def normalize_parametric_expressions(exprs: object) -> tuple[object, object]:
    """Normalize public parametric coordinates into exactly two expressions."""

    if isinstance(exprs, sympy.MatrixBase):
        items = tuple(exprs)
    elif isinstance(exprs, Sequence) and not isinstance(exprs, str | bytes):
        items = tuple(exprs)
    else:
        raise PlotSpecError(
            "parametric_plot(...) supports exactly two real coordinate "
            "expressions in this phase."
        )

    if len(items) != 2:
        raise PlotSpecError(
            "parametric_plot(...) supports exactly two real coordinate "
            "expressions in this phase."
        )
    return (normalize_expression(items[0]), normalize_expression(items[1]))


def normalize_list_plot_spec(
    source: object,
    index: object = OMITTED,
) -> tuple[ListPlotSpec, ListView]:
    """Normalize the public ``list_plot`` source and integer index grammar."""

    if index is not OMITTED and _is_mixed_list_plot_domain(index):
        raise PlotSpecError(
            "list_plot(...) index spec must be (n, min, max) or "
            "(n, min, max, step)."
        )

    # Value arrays own their domain. They intentionally reject symbolic index
    # arguments so callers cannot accidentally ask for expression resampling.
    value_spec = _normalize_list_plot_values(source)
    if value_spec is not None:
        if index is not OMITTED and index is not None:
            raise PlotSpecError(
                "list_plot(values, n) is invalid; value inputs use their own "
                "integer indices."
            )
        view = _list_value_view(value_spec)
        return value_spec, view

    if index is OMITTED or index is None:
        raise PlotSpecError(
            "list_plot(...) expression sources need an index symbol or index spec."
        )

    view = _normalize_list_index(index)
    if _is_expression_pair_source(source):
        items = tuple(source) if not isinstance(source, sympy.MatrixBase) else tuple(source)
        return (
            ListPlotSpec(
                source_kind=(
                    "expr_pair_inferred" if view.inferred else "expr_pair_explicit"
                ),
                source=(normalize_expression(items[0]), normalize_expression(items[1])),
            ),
            view,
        )

    return (
        ListPlotSpec(
            source_kind="expr_inferred" if view.inferred else "expr_explicit",
            source=normalize_expression(source),
        ),
        view,
    )


def normalize_domain_conditions(condition: object) -> tuple[DomainConditionSpec, ...]:
    """Normalize a public domain condition or system into typed items."""

    if isinstance(condition, bool):
        raise PlotSpecError(
            "domain_plot(...) received a plain Python bool. Use symbolic "
            "comparisons such as x @Eq@ y, x < y, and symbolic And/Or "
            "conditions."
        )

    if isinstance(condition, sympy.MatrixBase):
        raw_items = tuple(condition)
    elif (
        isinstance(condition, Sequence)
        and not isinstance(condition, str | bytes)
        and not isinstance(condition, sympy.Basic)
    ):
        raw_items = tuple(condition)
    else:
        raw_items = (condition,)

    if not raw_items:
        raise PlotSpecError("domain_plot(...) needs at least one condition.")

    items: list[DomainConditionSpec] = []
    for raw_item in raw_items:
        if isinstance(raw_item, bool):
            raise PlotSpecError(
                "domain_plot(...) received a plain Python bool. Use symbolic "
                "comparisons such as x @Eq@ y, x < y, and symbolic And/Or "
                "conditions."
            )
        expression = normalize_expression(raw_item)
        role = "boolean" if isinstance(expression, Boolean) else "signed"
        items.append(DomainConditionSpec(expression=expression, role=role))
    return tuple(items)


def normalize_parameter_specs(
    expr: object,
    independent_symbols: sympy.Symbol | Sequence[sympy.Symbol],
    params: object = OMITTED,
    *,
    existing: Mapping[sympy.Basic, ParameterSpec] | None = None,
) -> dict[sympy.Basic, ParameterSpec]:
    """Normalize explicit or autodetected parameter slider specs."""

    expression = normalize_expression(expr)
    current = {} if existing is None else dict(existing)
    if isinstance(independent_symbols, sympy.Symbol):
        independent = frozenset({independent_symbols})
    else:
        independent = frozenset(independent_symbols)

    # Ask Num which runtime arguments the expression needs, then apply the only
    # plotting-specific indexed rule: concrete indexed parameters are scalar UI
    # controls that pack back into Num's array argument.
    num_symbols = _num_parameter_symbols(expression, independent)
    indexed_bases = {
        symbol
        for symbol in num_symbols
        if isinstance(symbol, sympy.IndexedBase)
    }
    indexed_infos = _indexed_runtime_parameter_infos(expression)
    indexed_symbols = _ordered_unique(
        entry
        for info in indexed_infos
        if _indexed_base_in_set(info.base, indexed_bases)
        for entry in info.entries
    )
    indexed_symbols.extend(
        symbol
        for symbol in current
        if isinstance(symbol, sympy.Indexed)
        and _indexed_symbol_matches_base(symbol, indexed_bases)
        and symbol not in indexed_symbols
    )
    if params is not OMITTED and isinstance(params, Mapping):
        indexed_symbols.extend(
            symbol
            for symbol in params
            if isinstance(symbol, sympy.Indexed)
            and _indexed_symbol_matches_base(symbol, indexed_bases)
            and symbol not in indexed_symbols
        )
    indexed_base_names = {_indexed_base_name(symbol) for symbol in indexed_symbols}
    scalar_symbols = [
        symbol
        for symbol in num_symbols
        if isinstance(symbol, sympy.Symbol)
        and str(symbol) not in indexed_base_names
    ]
    free_symbols = set(scalar_symbols).union(indexed_symbols)
    _raise_scalar_indexed_parameter_conflict(free_symbols, current)

    explicit_values: dict[sympy.Basic, object] = {}
    if params is not OMITTED and params is not None:
        if not isinstance(params, Mapping):
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )
        for symbol, raw_spec in params.items():
            if not isinstance(symbol, (sympy.Symbol, sympy.Indexed)):
                raise PlotSpecError(
                    "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                    'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
                )
            if symbol in free_symbols:
                explicit_values[symbol] = raw_spec

    # Existing live parameters keep their visible order. New user-supplied
    # parameters are appended in mapping order, and remaining autodetected
    # parameters follow Num's call signature.
    symbol_order = _ordered_unique(
        symbol
        for symbol in current
        if symbol in free_symbols
    )
    symbol_order.extend(
        symbol
        for symbol in explicit_values
        if symbol in free_symbols and symbol not in symbol_order
    )
    symbol_order.extend(
        symbol
        for symbol in tuple(scalar_symbols) + tuple(indexed_symbols)
        if symbol in free_symbols and symbol not in symbol_order
    )

    # Preserve existing metadata for surviving symbols, then apply any explicit
    # fields the user provided in this call.
    normalized: dict[sympy.Basic, ParameterSpec] = {}
    for symbol in symbol_order:
        base = current.get(symbol)
        if base is None:
            base = default_parameter_spec(symbol, expression, independent)
        if symbol in explicit_values:
            base = _apply_parameter_update(base, explicit_values[symbol])
        normalized[symbol] = base

    return normalized


def normalize_style(
    style: object = OMITTED,
    *,
    existing: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Normalize the small curve-style dictionary."""

    return normalize_line_style(style, existing=existing, plotter="plot")


def normalize_line_style(
    style: object = OMITTED,
    *,
    existing: Mapping[str, object] | None = None,
    plotter: str,
) -> dict[str, object] | None:
    """Normalize a public line style dictionary."""

    if style is OMITTED or style is None:
        return None
    if not isinstance(style, Mapping):
        raise PlotSpecError("Plot style must be a dictionary of supported style keys.")

    allowed = {"color", "width", "opacity", "visible", "dash"}
    normalized = {} if existing is None else dict(existing)

    for key, value in style.items():
        if key not in allowed:
            raise PlotSpecError(
                f"Unknown {plotter} style key {key!r}. Supported keys are "
                "color, width, opacity, visible, and dash."
            )
        _assign_common_style_value(normalized, key, value)
    return normalized


def normalize_field_style(
    style: object = OMITTED,
    *,
    existing: Mapping[str, object] | None = None,
    plotter: str,
) -> dict[str, object] | None:
    """Normalize heatmap and contour style dictionaries."""

    if style is OMITTED or style is None:
        return None
    if not isinstance(style, Mapping):
        raise PlotSpecError("Plot style must be a dictionary of supported style keys.")

    allowed = {"colorscale", "opacity", "visible", "showscale", "zmin", "zmax"}
    if plotter == "temperature_plot":
        allowed = allowed | {"zsmooth"}
    if plotter == "contour_plot":
        allowed = allowed | {
            "contour_color",
            "contour_width",
            "line_smoothing",
        }
    normalized = {} if existing is None else dict(existing)

    for key, value in style.items():
        if key not in allowed:
            supported = ", ".join(sorted(allowed))
            raise PlotSpecError(
                f"Unknown {plotter} style key {key!r}. Supported keys are {supported}."
            )
        if key == "opacity":
            opacity = _finite_float(value, "style opacity")
            if opacity < 0 or opacity > 1:
                raise PlotSpecError("Plot style opacity must be between 0 and 1.")
            normalized[key] = opacity
        elif key == "visible" or key == "showscale":
            if not isinstance(value, bool):
                raise PlotSpecError(f"Plot style {key} must be True or False.")
            normalized[key] = value
        elif key == "zmin" or key == "zmax":
            normalized[key] = _finite_float(value, f"style {key}")
        elif key == "contour_width":
            normalized[key] = _positive_float(value, "style contour width")
        elif key == "line_smoothing":
            normalized[key] = _smoothing_float(value, "style line smoothing")
        elif key == "zsmooth":
            if value is not False and value not in {"fast", "best"}:
                raise PlotSpecError(
                    "Heatmap zsmooth style must be False, 'fast', or 'best'."
                )
            normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def normalize_domain_style(
    style: object = OMITTED,
    *,
    existing: Mapping[str, object] | None = None,
    boundary: bool = True,
) -> dict[str, object] | None:
    """Normalize the fill and boundary style dictionaries for ``domain_plot``."""

    if style is OMITTED or style is None:
        if existing is not None:
            normalized = _copy_domain_style(existing)
            normalized["boundary"]["visible"] = (
                bool(normalized["boundary"].get("visible", True)) and boundary
            )
            return normalized
        return _default_domain_style(boundary=boundary)
    if not isinstance(style, Mapping):
        raise PlotSpecError("Plot style must be a dictionary of supported style keys.")

    normalized = (
        _default_domain_style(boundary=boundary)
        if existing is None
        else _copy_domain_style(existing)
    )
    normalized["boundary"] = {
        **normalized["boundary"],  # type: ignore[index]
        "visible": bool(normalized["boundary"].get("visible", True)) and boundary,  # type: ignore[union-attr]
    }

    for key, value in style.items():
        if key == "domain":
            if not isinstance(value, Mapping):
                raise PlotSpecError("domain_plot(...) style['domain'] must be a dictionary.")
            _merge_domain_fill_style(normalized["domain"], value)
        elif key == "boundary":
            if not isinstance(value, Mapping):
                raise PlotSpecError(
                    "domain_plot(...) style['boundary'] must be a dictionary."
                )
            _merge_domain_boundary_style(normalized["boundary"], value)
        elif key in {"color", "opacity", "visible"}:
            _merge_domain_fill_style(normalized["domain"], {key: value})
        else:
            raise PlotSpecError(
                f"Unknown domain_plot style key {key!r}. Supported keys are "
                "domain, boundary, color, opacity, and visible."
            )
    return normalized


def expression_free_symbols(expr: object) -> set[sympy.Symbol]:
    """Return scalar symbols used by an expression or finite expression system."""

    if isinstance(expr, ListPlotSpec):
        return expression_free_symbols(expr.source)
    if isinstance(expr, DomainConditionSpec):
        return expression_free_symbols(expr.expression)
    if isinstance(expr, sympy.MatrixBase):
        return {
            symbol
            for item in expr
            for symbol in expression_free_symbols(item)
        }
    if isinstance(expr, Sequence) and not isinstance(expr, str | bytes | sympy.Basic):
        return {
            symbol
            for item in expr
            for symbol in expression_free_symbols(item)
        }
    return set(getattr(expr, "free_symbols", set()))


def _num_parameter_symbols(
    expr: object,
    independent_symbols: frozenset[sympy.Symbol],
) -> tuple[sympy.Basic, ...]:
    """Return Num-discovered runtime parameters excluding plot axes."""

    if isinstance(expr, ListPlotSpec):
        return _num_parameter_symbols(expr.source, independent_symbols)
    if isinstance(expr, DomainConditionSpec):
        return _num_parameter_symbols(expr.expression, independent_symbols)
    if isinstance(expr, sympy.MatrixBase):
        return _ordered_unique(
            symbol
            for item in expr
            for symbol in _num_parameter_symbols(item, independent_symbols)
        )
    if isinstance(expr, Sequence) and not isinstance(expr, str | bytes | sympy.Basic):
        return _ordered_unique(
            symbol
            for item in expr
            for symbol in _num_parameter_symbols(item, independent_symbols)
        )
    if not isinstance(expr, sympy.Basic):
        return ()

    return tuple(
        spec.symbol
        for spec in autodetect_runtime_args(expr, var=(), autodetect_vars=True)
        if spec.symbol not in independent_symbols
        and not isinstance(spec.symbol, sympy.Idx)
    )


def _ordered_unique(items: Iterable[sympy.Basic]) -> list[sympy.Basic]:
    """Return unique items in their first-seen iteration order."""

    ordered: list[sympy.Basic] = []
    for item in items:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _indexed_runtime_parameter_infos(expr: object) -> tuple[object, ...]:
    """Return Num indexed metadata from expressions inside plot specs."""

    if isinstance(expr, ListPlotSpec):
        return _indexed_runtime_parameter_infos(expr.source)
    if isinstance(expr, DomainConditionSpec):
        return _indexed_runtime_parameter_infos(expr.expression)
    if isinstance(expr, sympy.MatrixBase):
        return tuple(
            info
            for item in expr
            for info in _indexed_runtime_parameter_infos(item)
        )
    if isinstance(expr, Sequence) and not isinstance(expr, str | bytes | sympy.Basic):
        return tuple(
            info
            for item in expr
            for info in _indexed_runtime_parameter_infos(item)
        )
    if not isinstance(expr, sympy.Basic):
        return ()
    return indexed_runtime_parameter_info(expr)


def indexed_parameter_symbols(
    expr: object,
    independent_symbols: sympy.Symbol | Sequence[sympy.Symbol] = (),
) -> set[sympy.Indexed]:
    """Return concrete indexed entries that should become visible sliders."""

    if isinstance(independent_symbols, sympy.Symbol):
        independent = frozenset({independent_symbols})
    else:
        independent = frozenset(independent_symbols)

    if isinstance(expr, ListPlotSpec):
        return indexed_parameter_symbols(expr.source, independent)
    if isinstance(expr, DomainConditionSpec):
        return indexed_parameter_symbols(expr.expression, independent)
    if isinstance(expr, sympy.MatrixBase):
        return {
            symbol
            for item in expr
            for symbol in indexed_parameter_symbols(item, independent)
        }
    if isinstance(expr, Sequence) and not isinstance(expr, str | bytes | sympy.Basic):
        return {
            symbol
            for item in expr
            for symbol in indexed_parameter_symbols(item, independent)
        }
    if not isinstance(expr, sympy.Basic):
        return set()

    return {
        entry
        for info in _indexed_runtime_parameter_infos(expr)
        for entry in info.entries
        if all(index not in independent for index in entry.indices)
    }


def _raise_scalar_indexed_parameter_conflict(
    symbols: set[sympy.Basic],
    existing: Mapping[sympy.Basic, ParameterSpec],
) -> None:
    """Reject simultaneous scalar and indexed meanings for one parameter base."""

    scalar_names = {
        str(symbol)
        for symbol in set(symbols).union(existing)
        if isinstance(symbol, sympy.Symbol)
    }
    indexed_names = {
        _indexed_base_name(symbol)
        for symbol in set(symbols).union(existing)
        if isinstance(symbol, sympy.Indexed)
    }
    conflicts = scalar_names & indexed_names
    if conflicts:
        name = sorted(conflicts)[0]
        raise PlotSpecError(
            f"Parameter {name!r} is already used as indexed entries like "
            f"{name}[i], so it cannot also be used as a scalar parameter."
        )


def _indexed_base_name(symbol: sympy.Indexed) -> str:
    """Return the visible base name for one indexed parameter."""

    return str(symbol.base)


def _indexed_symbol_matches_base(
    symbol: sympy.Indexed,
    bases: set[sympy.IndexedBase],
) -> bool:
    """Return whether an indexed slider can supply one Num array base."""

    return _indexed_base_in_set(symbol.base, bases)


def _indexed_base_in_set(
    base: sympy.IndexedBase,
    bases: set[sympy.IndexedBase],
) -> bool:
    """Return whether a base is represented in a discovered base set."""

    return any(str(base) == str(candidate) for candidate in bases)


def sort_symbols(symbols: object) -> tuple[sympy.Basic, ...]:
    """Return symbols sorted by visible name and stable string fallback."""

    return tuple(
        sorted(
            symbols,
            key=lambda symbol: (
                _natural_text_sort_key(getattr(symbol, "name", str(symbol))),
                _natural_text_sort_key(str(symbol)),
            ),
        )
    )


def _natural_text_sort_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Return a key that compares digit runs by numeric value."""

    pieces: list[tuple[int, int | str]] = []
    position = 0
    for match in _DIGIT_RUN.finditer(text):
        if match.start() > position:
            pieces.append((1, text[position : match.start()]))
        pieces.append((0, int(match.group())))
        position = match.end()
    if position < len(text):
        pieces.append((1, text[position:]))
    return tuple(pieces)


def default_parameter_spec(
    symbol: sympy.Basic,
    expr: object,
    independent_symbols: sympy.Symbol | Sequence[sympy.Symbol],
) -> ParameterSpec:
    """Return conservative default slider state for an autodetected parameter."""

    if isinstance(independent_symbols, sympy.Symbol):
        independent = frozenset({independent_symbols})
    else:
        independent = frozenset(independent_symbols)

    if _is_coefficient_like(symbol, expr, independent):
        return ParameterSpec(
            symbol=symbol,
            value=1.0,
            metadata=ParameterMetadata(minimum=0.0, maximum=2.0, step=0.01),
        )
    return ParameterSpec(
        symbol=symbol,
        value=0.0,
        metadata=ParameterMetadata(minimum=-1.0, maximum=1.0, step=0.01),
    )


def normalize_sample_count(value: object) -> int:
    """Return a validated one-dimensional sample count."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise PlotSpecError("Plot sample counts must be integers greater than 1.")
    if value <= 1:
        raise PlotSpecError("Plot sample counts must be integers greater than 1.")
    return value


def normalize_grid_sample_count(value: object) -> tuple[int, int]:
    """Return validated x and y sample counts for grid-based plots."""

    if isinstance(value, bool):
        raise PlotSpecError(
            "Field sample counts must be an integer or a length-two tuple of "
            "integers greater than 1."
        )
    if isinstance(value, int):
        count = normalize_sample_count(value)
        _guard_grid_sample_count(count, count)
        return count, count
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and not isinstance(value[0], bool)
        and not isinstance(value[1], bool)
        and isinstance(value[0], int)
        and isinstance(value[1], int)
    ):
        x_count = normalize_sample_count(value[0])
        y_count = normalize_sample_count(value[1])
        _guard_grid_sample_count(x_count, y_count)
        return x_count, y_count
    raise PlotSpecError(
        "Field sample counts must be an integer or a length-two tuple of "
        "integers greater than 1."
    )


def _initial_axis_view(domain: AxisDomain, default_view: AxisView) -> AxisView:
    """Return an initial active view from finite bounds or the default policy."""

    if domain.minimum is not None and domain.maximum is not None:
        return AxisView(minimum=domain.minimum, maximum=domain.maximum)
    return default_view


def _normalize_list_plot_values(source: object) -> ListPlotSpec | None:
    """Return a value-list spec when ``source`` is finite numeric data."""

    if isinstance(source, str | bytes | sympy.Basic | sympy.MatrixBase) or callable(source):
        return None
    try:
        array = np.asarray(source, dtype=float)
    except (TypeError, ValueError):
        return None

    if array.ndim == 0:
        return None
    if np.iscomplexobj(np.asarray(source)):
        raise PlotSpecError("list_plot(...) expected real numeric values.")
    if not np.all(np.isfinite(array)):
        raise PlotSpecError("list_plot(...) expected finite numeric values.")

    if array.ndim == 1:
        return ListPlotSpec(source_kind="value_y", source=tuple(float(v) for v in array))
    if array.ndim == 2 and array.shape[1] == 2:
        return ListPlotSpec(
            source_kind="value_xy",
            source=tuple((float(x), float(y)) for x, y in array),
        )
    if array.ndim == 2:
        raise PlotSpecError(
            "list_plot(...) 2D values must contain exactly two coordinates per point."
        )
    raise PlotSpecError("list_plot(...) expected values or an expression source.")


def _list_value_view(spec: ListPlotSpec) -> ListView:
    """Return a fixed-domain list view for normalized value inputs."""

    if spec.source_kind == "value_y":
        count = len(spec.source)  # type: ignore[arg-type]
        if count == 0:
            return ListView()
        return ListView(minimum=0, maximum=count, step=1)

    points = tuple(spec.source)  # type: ignore[arg-type]
    if not points:
        return ListView()
    x_values = [point[0] for point in points]
    return ListView(
        minimum=int(np.floor(min(x_values))),
        maximum=int(np.ceil(max(x_values))) + 1,
        step=1,
    )


def _normalize_list_index(index: object) -> ListView:
    """Return a validated integer-index view for expression list plots."""

    default_x_view, _ = default_initial_2d_view()
    if isinstance(index, sympy.Symbol):
        return ListView(index_symbol=index, x_view=default_x_view)
    if not isinstance(index, tuple) or len(index) not in {3, 4}:
        raise PlotSpecError(
            "list_plot(...) index spec must be (n, min, max) or "
            "(n, min, max, step)."
        )

    symbol = index[0]
    if not isinstance(symbol, sympy.Symbol):
        raise PlotSpecError("list_plot(...) index variable must be a SymPy symbol.")
    minimum = _integral_int(index[1], "index minimum")
    maximum = _integral_int(index[2], "index maximum")
    step = 1 if len(index) == 3 else _integral_int(index[3], "index step")
    if step == 0:
        raise PlotSpecError("list_plot(...) index step must be a nonzero integer.")
    return ListView(index_symbol=symbol, minimum=minimum, maximum=maximum, step=step)


def _integral_int(value: object, label: str) -> int:
    """Return ``value`` as an exact integer for list-plot index bounds."""

    if isinstance(value, bool):
        raise PlotSpecError("list_plot(...) index bounds must be integral.")
    try:
        sympified = sympy.sympify(value)
    except sympy.SympifyError as exc:
        raise PlotSpecError("list_plot(...) index bounds must be integral.") from exc
    if not bool(getattr(sympified, "is_integer", False)):
        raise PlotSpecError("list_plot(...) index bounds must be integral.")
    try:
        return int(sympified)
    except TypeError as exc:
        raise PlotSpecError(f"list_plot(...) {label} must be integral.") from exc


def _is_expression_pair_source(source: object) -> bool:
    """Return whether ``source`` is an explicit two-expression coordinate pair."""

    if isinstance(source, sympy.MatrixBase):
        return len(tuple(source)) == 2
    return (
        isinstance(source, Sequence)
        and not isinstance(source, str | bytes | sympy.Basic)
        and len(source) == 2
    )


def _is_mixed_list_plot_domain(index: object) -> bool:
    """Return whether an index argument uses a known invalid mixed spelling."""

    return (
        isinstance(index, tuple)
        and len(index) == 2
        and not isinstance(index[0], sympy.Symbol)
    )


def _apply_parameter_update(
    base: ParameterSpec,
    raw_spec: object,
) -> ParameterSpec:
    """Apply one public parameter value or metadata dictionary."""

    if isinstance(raw_spec, Mapping):
        unknown = set(raw_spec) - {"value", "min", "max", "step", "label"}
        if unknown:
            raise PlotSpecError(
                "Parameter specs must be supplied through fig.parameters({symbol: value}) "
                'or fig.parameters({symbol: {"value": ..., "min": ..., "max": ...}}).'
            )

        value = base.value
        metadata = base.metadata
        provided = set(raw_spec)

        if "value" in raw_spec:
            value = _finite_float(raw_spec["value"], "parameter value")
        if "min" in raw_spec:
            metadata = replace(
                metadata,
                minimum=_finite_float(raw_spec["min"], "parameter minimum"),
            )
        if "max" in raw_spec:
            metadata = replace(
                metadata,
                maximum=_finite_float(raw_spec["max"], "parameter maximum"),
            )
        if "step" in raw_spec:
            metadata = replace(
                metadata,
                step=_positive_float(raw_spec["step"], "parameter step"),
            )
        if "label" in raw_spec:
            label = raw_spec["label"]
            metadata = replace(metadata, label=None if label is None else str(label))
        return _validated_parameter_spec(base.symbol, value, metadata, provided)

    value = _finite_float(raw_spec, "parameter value")
    return _validated_parameter_spec(base.symbol, value, base.metadata, {"value"})


def _validated_parameter_spec(
    symbol: sympy.Basic,
    value: float,
    metadata: ParameterMetadata,
    explicitly_provided: set[str],
) -> ParameterSpec:
    """Return a parameter spec after slider range validation."""

    minimum = metadata.minimum
    maximum = metadata.maximum

    # If the user only gave a value, expand the default range enough for the
    # slider to represent that value without surprising widget construction
    # failures.
    if "min" not in explicitly_provided and value < minimum:
        minimum = value
    if "max" not in explicitly_provided and value > maximum:
        maximum = value

    if minimum >= maximum:
        raise PlotSpecError("Parameter slider minimum must be less than maximum.")
    if value < minimum or value > maximum:
        raise PlotSpecError("Parameter value must lie within its slider range.")

    return ParameterSpec(
        symbol=symbol,
        value=value,
        metadata=ParameterMetadata(
            minimum=minimum,
            maximum=maximum,
            step=metadata.step,
            label=metadata.label,
        ),
    )


def _is_coefficient_like(
    symbol: sympy.Basic,
    expr: object,
    independent_symbols: frozenset[sympy.Symbol],
) -> bool:
    """Return whether ``symbol`` is a direct multiplicative coefficient."""

    if isinstance(expr, ListPlotSpec):
        expr = expr.source
    expressions = (
        tuple(expr)
        if isinstance(expr, Sequence) and not isinstance(expr, str | bytes | sympy.Basic)
        else (expr,)
    )
    for expression in expressions:
        if not isinstance(expression, sympy.Basic):
            continue
        for part in sympy.preorder_traversal(expression):
            if not isinstance(part, sympy.Mul):
                continue
            if symbol not in part.args:
                continue
            if any(
                arg != symbol and any(arg.has(independent) for independent in independent_symbols)
                for arg in part.args
            ):
                return True
    return False


def _assign_common_style_value(
    normalized: dict[str, object],
    key: str,
    value: object,
) -> None:
    """Assign one common style value after public validation."""

    if key == "width":
        normalized[key] = _positive_float(value, "style width")
    elif key == "opacity":
        opacity = _finite_float(value, "style opacity")
        if opacity < 0 or opacity > 1:
            raise PlotSpecError("Plot style opacity must be between 0 and 1.")
        normalized[key] = opacity
    elif key == "visible":
        if not isinstance(value, bool):
            raise PlotSpecError("Plot style visible must be True or False.")
        normalized[key] = value
    else:
        normalized[key] = value


def _default_domain_style(*, boundary: bool) -> dict[str, object]:
    """Return the default nested domain style dictionary."""

    color = "royalblue"
    return {
        "domain": {
            "color": color,
            "opacity": 0.25,
            "visible": True,
            "zsmooth": "best",
        },
        "boundary": {
            "color": color,
            "width": 2.0,
            "dash": "solid",
            "visible": boundary,
            "smoothing": 1.0,
        },
    }


def _copy_domain_style(style: Mapping[str, object]) -> dict[str, object]:
    """Return a shallow copy of a normalized nested domain style."""

    domain = style.get("domain", {})
    boundary = style.get("boundary", {})
    return {
        "domain": dict(domain) if isinstance(domain, Mapping) else {},
        "boundary": dict(boundary) if isinstance(boundary, Mapping) else {},
    }


def _merge_domain_fill_style(
    target: dict[str, object],
    update: Mapping[str, object],
) -> None:
    """Merge and validate public fill style keys."""

    for key, value in update.items():
        if key not in {"color", "opacity", "visible", "zsmooth"}:
            raise PlotSpecError(
                f"Unknown domain fill style key {key!r}. Supported keys are "
                "color, opacity, visible, and zsmooth."
            )
        if key == "opacity":
            opacity = _finite_float(value, "domain fill opacity")
            if opacity < 0 or opacity > 1:
                raise PlotSpecError("Domain fill opacity must be between 0 and 1.")
            target[key] = opacity
        elif key == "zsmooth":
            if value is not False and value not in {"fast", "best"}:
                raise PlotSpecError(
                    "Domain fill zsmooth style must be False, 'fast', or 'best'."
                )
            target[key] = value
        elif key == "visible":
            if not isinstance(value, bool):
                raise PlotSpecError("Domain fill visible must be True or False.")
            target[key] = value
        else:
            target[key] = value


def _merge_domain_boundary_style(
    target: dict[str, object],
    update: Mapping[str, object],
) -> None:
    """Merge and validate public boundary style keys."""

    for key, value in update.items():
        if key not in {"color", "width", "dash", "visible", "smoothing"}:
            raise PlotSpecError(
                f"Unknown domain boundary style key {key!r}. Supported keys are "
                "color, width, dash, visible, and smoothing."
            )
        if key == "width":
            target[key] = _positive_float(value, "domain boundary width")
        elif key == "smoothing":
            target[key] = _smoothing_float(value, "domain boundary smoothing")
        elif key == "visible":
            if not isinstance(value, bool):
                raise PlotSpecError("Domain boundary visible must be True or False.")
            target[key] = value
        else:
            target[key] = value


def _guard_grid_sample_count(x_samples: int, y_samples: int) -> None:
    """Keep phase-one field grids within a conservative notebook size."""

    if x_samples * y_samples > 250_000:
        raise PlotSpecError(
            "Field sample grids are limited to 250000 points in this phase. "
            "Use a smaller samples= integer or (x_samples, y_samples) tuple."
        )


def _finite_float(value: object, label: str) -> float:
    """Return ``value`` as a finite float or raise a plotting spec error."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise PlotSpecError(f"{label.capitalize()} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise PlotSpecError(f"{label.capitalize()} must be a finite real scalar.")
    return result


def _positive_float(value: object, label: str) -> float:
    """Return a finite positive float or raise a plotting spec error."""

    result = _finite_float(value, label)
    if result <= 0:
        raise PlotSpecError(f"{label.capitalize()} must be positive.")
    return result


def _smoothing_float(value: object, label: str) -> float:
    """Return a Plotly-compatible contour smoothing amount."""

    result = _finite_float(value, label)
    if result < 0 or result > 1.3:
        raise PlotSpecError(f"{label.capitalize()} must be between 0 and 1.3.")
    return result
