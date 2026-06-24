"""Compile symbolic plot expressions through ``Num`` and sample them."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import sympy

from math_toolkit.num import Num

from .errors import PlotCompilationError, PlotShapeError
from .specs import (
    CartesianView2D,
    CurveView,
    DomainConditionSpec,
    ListPlotSpec,
    ListView,
    ParametricView,
)


@dataclass(frozen=True)
class SampleSignature:
    """Describe the state that determines one sampled plot."""

    expression: object
    view: object
    parameter_symbols: tuple[sympy.Basic, ...]
    parameter_values: tuple[float, ...]


@dataclass(frozen=True)
class FieldSample:
    """Describe sampled scalar-field grid data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class DomainSample:
    """Describe sampled filled-domain and boundary grid data."""

    x: np.ndarray
    y: np.ndarray
    fill_z: np.ndarray
    boundary_z: np.ndarray
    boundary_level: float


def compile_numeric_curve(
    expression: object,
    domain_symbol: sympy.Symbol,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> object:
    """Compile a scalar curve using explicit domain-first ``Num`` arguments."""

    numeric_symbols, pack = _numeric_parameter_binding(parameter_symbols)
    try:
        numeric = expression >> Num(var=(domain_symbol, *numeric_symbols))
    except Exception as exc:
        raise PlotCompilationError(
            "plot(...) could not compile the expression numerically. Use "
            "UnholdAll before plotting held definitions, or simplify the "
            "expression before plotting."
        ) from exc

    if not callable(numeric):
        raise PlotCompilationError(
            "plot(...) could not compile the expression as a function of the "
            "declared domain."
        )
    return _wrap_packed_numeric(numeric, pack, leading_count=1)


def compile_numeric_field(
    expression: object,
    x_symbol: sympy.Symbol,
    y_symbol: sympy.Symbol,
    parameter_symbols: tuple[sympy.Basic, ...],
    *,
    plotter: str,
) -> object:
    """Compile a scalar field using explicit x, y, parameter ``Num`` arguments."""

    numeric_symbols, pack = _numeric_parameter_binding(parameter_symbols)
    try:
        numeric = expression >> Num(var=(x_symbol, y_symbol, *numeric_symbols))
    except Exception as exc:
        raise PlotCompilationError(
            f"{plotter}(...) could not compile the expression numerically."
        ) from exc

    if not callable(numeric):
        raise PlotCompilationError(
            f"{plotter}(...) could not compile the expression as a function of "
            "the declared x and y variables."
        )
    return _wrap_packed_numeric(numeric, pack, leading_count=2)


def compile_numeric_domain(
    items: tuple[DomainConditionSpec, ...],
    x_symbol: sympy.Symbol,
    y_symbol: sympy.Symbol,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> tuple[object, ...]:
    """Compile each domain item as an explicit x, y numeric function."""

    compiled = []
    for item in items:
        item_slider_symbols = tuple(
            symbol
            for symbol in parameter_symbols
            if _symbol_needed_by_expression(symbol, item.expression)
        )
        item_numeric_symbols, item_pack = _numeric_parameter_binding(item_slider_symbols)
        item_symbols = tuple(
            symbol
            for symbol in (x_symbol, y_symbol, *item_numeric_symbols)
            if _symbol_needed_by_expression(symbol, item.expression)
        )
        try:
            if item_symbols:
                numeric = item.expression >> Num(var=item_symbols)
            else:
                numeric = item.expression >> Num()
        except Exception as exc:
            raise PlotCompilationError(
                "domain_plot(...) could not compile a condition numerically."
            ) from exc
        if not callable(numeric):
            raise PlotCompilationError(
                "domain_plot(...) could not compile a condition as a function "
                "of the declared x and y variables."
            )
        leading_count = int(x_symbol in item_symbols) + int(y_symbol in item_symbols)
        numeric = _wrap_packed_numeric(numeric, item_pack, leading_count=leading_count)
        slider_declared_symbols = (x_symbol, y_symbol, *item_slider_symbols)
        symbol_positions = tuple(
            slider_declared_symbols.index(symbol)
            for symbol in (x_symbol, y_symbol, *item_slider_symbols)
            if symbol in item_symbols or symbol in item_slider_symbols
        )

        def _invoke(
            *args: object,
            _numeric: object = numeric,
            _positions: tuple[int, ...] = symbol_positions,
        ) -> object:
            if not _positions:
                return _numeric()
            selected_args = tuple(args[position] for position in _positions)
            return _numeric(*selected_args)

        compiled.append(_invoke)
    return tuple(compiled)


def compile_numeric_parametric(
    expressions: tuple[object, object],
    parameter_symbol: sympy.Symbol,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> tuple[object, object]:
    """Compile two parametric coordinates using explicit ``Num`` arguments."""

    compiled = []
    numeric_symbols, pack = _numeric_parameter_binding(parameter_symbols)
    for expression in expressions:
        try:
            numeric = expression >> Num(var=(parameter_symbol, *numeric_symbols))
        except Exception as exc:
            raise PlotCompilationError(
                "parametric_plot(...) could not compile the coordinate "
                "expressions numerically."
            ) from exc
        if not callable(numeric):
            raise PlotCompilationError(
                "parametric_plot(...) could not compile the coordinates as "
                "functions of the declared parameter."
            )
        compiled.append(_wrap_packed_numeric(numeric, pack, leading_count=1))
    return (compiled[0], compiled[1])


def compile_numeric_list(
    spec: ListPlotSpec,
    index_symbol: sympy.Symbol | None,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> object:
    """Compile a list-plot expression source for integer-index sampling."""

    if spec.source_kind in {"value_y", "value_xy"}:
        return None
    if index_symbol is None:
        raise PlotCompilationError(
            "list_plot(...) expression sources need an index symbol or index spec."
        )

    if spec.source_kind.startswith("expr_pair"):
        sources = tuple(spec.source)  # type: ignore[arg-type]
    else:
        sources = (spec.source,)

    compiled = []
    numeric_symbols, pack = _numeric_parameter_binding(parameter_symbols)
    for source in sources:
        if callable(source):
            compiled.append(source)
            continue
        try:
            numeric = source >> Num(var=(index_symbol, *numeric_symbols))
        except Exception as exc:
            raise PlotCompilationError(
                "list_plot(...) could not compile the expression numerically."
            ) from exc
        if not callable(numeric):
            raise PlotCompilationError(
                "list_plot(...) could not compile the expression as a function "
                "of the declared integer index."
            )
        compiled.append(_wrap_packed_numeric(numeric, pack, leading_count=1))
    return tuple(compiled)


def compile_numeric_info(
    expression: object,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> object:
    """Compile a scalar info expression using figure parameter arguments."""

    # Reuse the plot parameter binding so concrete indexed sliders are packed
    # into the array argument shape that Num expects.
    numeric_symbols, pack = _numeric_parameter_binding(parameter_symbols)
    try:
        info_hints = {
            "Integrator": "Sampled",
            "sample_count": 4097,
            "sample_chunk_size": 1024,
        }
        numeric = (
            expression >> Num(var=numeric_symbols, Integral=info_hints)
            if numeric_symbols
            else expression >> Num(Integral=info_hints)
        )
    except Exception as exc:
        raise PlotCompilationError(
            "fig.info(...) could not compile a symbolic fragment."
        ) from exc

    if not callable(numeric):
        raise PlotCompilationError(
            "fig.info(...) symbolic fragments must be callable after compilation."
        )
    return _wrap_packed_numeric(numeric, pack, leading_count=0)


def expression_parameter_symbols(
    expression: object,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> tuple[sympy.Basic, ...]:
    """Return parameter symbols needed to evaluate one symbolic expression."""

    return tuple(
        symbol
        for symbol in parameter_symbols
        if _symbol_needed_by_expression(symbol, expression)
    )


def uncovered_expression_symbols(
    expression: object,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> tuple[sympy.Basic, ...]:
    """Return free expression symbols not represented by parameter symbols."""

    return tuple(
        symbol
        for symbol in getattr(expression, "free_symbols", set())
        if not _free_symbol_covered_by_parameters(symbol, parameter_symbols)
    )


def _numeric_parameter_binding(
    parameter_symbols: tuple[sympy.Basic, ...],
) -> tuple[tuple[sympy.Basic, ...], Callable[[tuple[object, ...]], tuple[object, ...]]]:
    """Return compiled ``Num`` symbols and a value packer for indexed sliders."""

    if not any(isinstance(symbol, sympy.Indexed) for symbol in parameter_symbols):
        return parameter_symbols, lambda values: values

    numeric_symbols: list[sympy.Basic] = []
    indexed_groups: dict[sympy.IndexedBase, list[tuple[sympy.Indexed, tuple[int, ...]]]] = {}
    seen_bases: set[sympy.IndexedBase] = set()

    for symbol in parameter_symbols:
        if not isinstance(symbol, sympy.Indexed):
            numeric_symbols.append(symbol)
            continue

        indices = _concrete_indices(symbol)
        if indices is None:
            raise PlotCompilationError(
                "Indexed plot parameters must use concrete nonnegative integer indices."
            )
        base = symbol.base
        indexed_groups.setdefault(base, []).append((symbol, indices))
        if base not in seen_bases:
            numeric_symbols.append(base)
            seen_bases.add(base)

    def pack(values: tuple[object, ...]) -> tuple[object, ...]:
        value_by_symbol = dict(zip(parameter_symbols, values, strict=True))
        packed_values: list[object] = []
        emitted_bases: set[sympy.IndexedBase] = set()
        for symbol in parameter_symbols:
            if isinstance(symbol, sympy.Indexed):
                base = symbol.base
                if base not in emitted_bases:
                    packed_values.append(_packed_indexed_array(indexed_groups[base], value_by_symbol))
                    emitted_bases.add(base)
                continue
            packed_values.append(value_by_symbol[symbol])
        return tuple(packed_values)

    return tuple(numeric_symbols), pack


def _wrap_packed_numeric(
    numeric: object,
    pack: Callable[[tuple[object, ...]], tuple[object, ...]],
    *,
    leading_count: int,
) -> object:
    """Expose a slider-shaped callable around a compiled ``Num`` function."""

    def invoke(*args: object) -> object:
        if leading_count == 0:
            return numeric(*pack(tuple(args)))
        if len(args) < leading_count:
            return numeric()
        leading_values = args[:leading_count]
        parameter_values = args[leading_count:]
        return numeric(*leading_values, *pack(tuple(parameter_values)))

    return invoke


def _packed_indexed_array(
    entries: Sequence[tuple[sympy.Indexed, tuple[int, ...]]],
    value_by_symbol: Mapping[sympy.Basic, object],
) -> np.ndarray:
    """Return one dense coefficient array from concrete indexed slider values."""

    rank = len(entries[0][1])
    shape = tuple(max(indices[axis] for _symbol, indices in entries) + 1 for axis in range(rank))
    array = np.zeros(shape, dtype=float)
    for symbol, indices in entries:
        array[indices] = float(value_by_symbol[symbol])
    return array


def _concrete_indices(symbol: sympy.Indexed) -> tuple[int, ...] | None:
    """Return concrete nonnegative integer indices for an Indexed symbol."""

    indices: list[int] = []
    for index in symbol.indices:
        if not bool(getattr(index, "is_integer", False)):
            return None
        try:
            value = int(index)
        except TypeError:
            return None
        if value < 0:
            return None
        indices.append(value)
    return tuple(indices)


def _symbol_needed_by_expression(symbol: sympy.Basic, expression: object) -> bool:
    """Return whether a slider or compiled symbol appears in an expression."""

    free_symbols = getattr(expression, "free_symbols", set())
    if symbol in free_symbols:
        return True
    if isinstance(symbol, sympy.Indexed):
        for item in free_symbols:
            if not isinstance(item, sympy.Indexed):
                continue
            if not _same_indexed_base(item.base, symbol.base):
                continue
            if _concrete_indices(item) is None:
                return True
        return False
    if isinstance(symbol, sympy.IndexedBase):
        for item in free_symbols:
            if isinstance(item, sympy.Indexed) and _same_indexed_base(item.base, symbol):
                return True
    return False


def _free_symbol_covered_by_parameters(
    free_symbol: sympy.Basic,
    parameter_symbols: tuple[sympy.Basic, ...],
) -> bool:
    """Return whether a free symbol is represented by selected parameters."""

    if free_symbol in parameter_symbols:
        return True
    for symbol in parameter_symbols:
        if not isinstance(symbol, sympy.Indexed):
            continue
        if isinstance(free_symbol, sympy.Indexed) and _same_indexed_base(
            free_symbol.base,
            symbol.base,
        ):
            return True
        if _same_indexed_base(free_symbol, symbol.base):
            return True
    return False


def _same_indexed_base(left: object, right: object) -> bool:
    """Return whether two visible indexed bases describe the same array."""

    try:
        if left == right:
            return True
    except Exception:
        pass
    return str(left) == str(right)


def sample_curve(numeric: object, signature: SampleSignature) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly sampled x and y arrays for one curve signature."""

    view = signature.view
    if not isinstance(view, CurveView):
        raise TypeError("sample_curve(...) expects a CurveView signature.")
    x_values = _sample_axis_values(
        view.x_domain.minimum,
        view.x_domain.maximum,
        view.x_view.minimum,
        view.x_view.maximum,
        view.samples,
    )

    try:
        raw_y = numeric(x_values, *signature.parameter_values)
    except Exception as exc:
        raise PlotShapeError(
            "plot(...) supports real scalar curves y = f(x). Use "
            "parametric_plot(...) for two-coordinate parametric curves and "
            "temperature_plot(...) or contour_plot(...) for scalar fields."
        ) from exc

    y_values = _real_array_like(
        raw_y,
        x_values.shape,
        message=(
            "plot(...) supports real scalar curves y = f(x). Use "
            "parametric_plot(...) for two-coordinate parametric curves and "
            "temperature_plot(...) or contour_plot(...) for scalar fields."
        ),
    )
    return x_values, y_values


def sample_scalar_field(numeric: object, signature: SampleSignature) -> FieldSample:
    """Return Plotly-oriented x, y, and z grid data for one scalar field."""

    view = signature.view
    if not isinstance(view, CartesianView2D):
        raise TypeError("sample_scalar_field(...) expects a CartesianView2D signature.")
    x_values, y_values, x_grid, y_grid = _sample_cartesian_grid(view)
    if x_values.size == 0 or y_values.size == 0:
        return FieldSample(
            x=x_values,
            y=y_values,
            z=np.empty((y_values.size, x_values.size), dtype=float),
        )

    try:
        raw_z = numeric(x_grid, y_grid, *signature.parameter_values)
    except Exception as exc:
        raise PlotShapeError(
            "Scalar field plots require a real scalar expression over the "
            "declared x and y variables."
        ) from exc

    # Plotly heatmap and contour traces expect z rows to align with y values
    # and z columns to align with x values. ``meshgrid`` with default indexing
    # produces exactly that shape: (len(y), len(x)).
    z_values = _real_array_like(
        raw_z,
        x_grid.shape,
        message=(
            "Scalar field plots require a real scalar expression over the "
            "declared x and y variables."
        ),
    )
    return FieldSample(x=x_values, y=y_values, z=z_values)


def sample_domain(
    compiled_items: tuple[object, ...],
    signature: SampleSignature,
) -> DomainSample:
    """Return sampled fill and boundary grids for a Boolean or signed domain."""

    items = signature.expression
    view = signature.view
    if not isinstance(view, CartesianView2D):
        raise TypeError("sample_domain(...) expects a CartesianView2D signature.")
    if not isinstance(items, tuple):
        raise TypeError("sample_domain(...) expects domain item specs.")

    x_values, y_values, x_grid, y_grid = _sample_cartesian_grid(view)
    empty_shape = (y_values.size, x_values.size)
    if x_values.size == 0 or y_values.size == 0:
        return DomainSample(
            x=x_values,
            y=y_values,
            fill_z=np.empty(empty_shape, dtype=float),
            boundary_z=np.empty(empty_shape, dtype=float),
            boundary_level=0.5,
        )

    combined_mask = np.ones(x_grid.shape, dtype=bool)
    signed_values_for_boundary: np.ndarray | None = None

    for item, numeric in zip(items, compiled_items, strict=True):
        try:
            raw_values = numeric(x_grid, y_grid, *signature.parameter_values)
        except Exception as exc:
            raise PlotShapeError(
                "domain_plot(...) signed expressions must evaluate to real "
                "scalar grid values. Boolean conditions are also accepted."
            ) from exc

        if not isinstance(item, DomainConditionSpec):
            raise TypeError("sample_domain(...) expects DomainConditionSpec items.")
        if item.role == "boolean":
            item_mask = _boolean_array_like(raw_values, x_grid.shape)
        else:
            signed_values = _real_array_like(
                raw_values,
                x_grid.shape,
                message=(
                    "domain_plot(...) signed expressions must evaluate to real "
                    "scalar grid values. Boolean conditions are also accepted."
                ),
            )
            finite_values = np.isfinite(signed_values)
            item_mask = finite_values & (signed_values > 0)
            if len(items) == 1:
                signed_values_for_boundary = signed_values
        combined_mask &= item_mask

    fill_z = np.where(combined_mask, 1.0, np.nan)
    if signed_values_for_boundary is not None:
        boundary_z = signed_values_for_boundary
        boundary_level = 0.0
    else:
        # Phase one draws the boundary of the combined mask. This is intentionally
        # coarse for systems, but it gives one stable trace without inventing a
        # boundary algebra for every constituent condition.
        boundary_z = combined_mask.astype(float)
        boundary_level = 0.5

    return DomainSample(
        x=x_values,
        y=y_values,
        fill_z=fill_z,
        boundary_z=boundary_z,
        boundary_level=boundary_level,
    )


def sample_parametric(
    numerics: tuple[object, object],
    signature: SampleSignature,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly sampled x and y arrays for a parametric curve."""

    view = signature.view
    if not isinstance(view, ParametricView):
        raise TypeError("sample_parametric(...) expects a ParametricView signature.")

    t_values = np.linspace(view.minimum, view.maximum, view.samples)
    coordinates = []
    for numeric in numerics:
        try:
            raw_values = numeric(t_values, *signature.parameter_values)
        except Exception as exc:
            raise PlotShapeError(
                "parametric_plot(...) supports exactly two real coordinate "
                "expressions in this phase."
            ) from exc
        coordinates.append(
            _real_array_like(
                raw_values,
                t_values.shape,
                message=(
                    "parametric_plot(...) supports exactly two real coordinate "
                    "expressions in this phase."
                ),
            )
        )
    return coordinates[0], coordinates[1]


def sample_list_plot(
    compiled: object,
    signature: SampleSignature,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y arrays for one discrete list plot."""

    spec = signature.expression
    view = signature.view
    if not isinstance(spec, ListPlotSpec) or not isinstance(view, ListView):
        raise TypeError("sample_list_plot(...) expects a list-plot signature.")

    if spec.source_kind == "value_y":
        y_values = _real_vector(spec.source, "list_plot(...) expected real numeric values.")
        x_values = np.arange(y_values.size, dtype=float)
        return x_values, y_values
    if spec.source_kind == "value_xy":
        points = np.asarray(spec.source, dtype=float)
        if points.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise PlotShapeError(
                "list_plot(...) 2D values must contain exactly two coordinates per point."
            )
        return points[:, 0].astype(float), points[:, 1].astype(float)

    n_values = _integer_index_values(view)
    if spec.source_kind.startswith("expr_pair"):
        numerics = tuple(compiled)  # type: ignore[arg-type]
        if len(numerics) != 2:
            raise PlotShapeError(
                "list_plot(...) expression output must be scalar or exactly two coordinates."
            )
        return (
            _evaluate_list_coordinate(numerics[0], n_values, signature.parameter_values),
            _evaluate_list_coordinate(numerics[1], n_values, signature.parameter_values),
        )

    (numeric,) = tuple(compiled)  # type: ignore[arg-type]
    raw_values = _invoke_list_numeric(numeric, n_values, signature.parameter_values)
    return _coordinates_from_list_output(raw_values, n_values)


def _sample_cartesian_grid(
    view: CartesianView2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one-dimensional axes and Plotly-oriented meshgrid arrays."""

    x_values = _sample_axis_values(
        view.x_domain.minimum,
        view.x_domain.maximum,
        view.x_view.minimum,
        view.x_view.maximum,
        view.x_samples,
    )
    y_values = _sample_axis_values(
        view.y_domain.minimum,
        view.y_domain.maximum,
        view.y_view.minimum,
        view.y_view.maximum,
        view.y_samples,
    )
    if x_values.size == 0 or y_values.size == 0:
        return (
            x_values,
            y_values,
            np.empty((y_values.size, x_values.size), dtype=float),
            np.empty((y_values.size, x_values.size), dtype=float),
        )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_values, y_values, x_grid, y_grid


def _sample_axis_values(
    domain_minimum: float | None,
    domain_maximum: float | None,
    view_minimum: float,
    view_maximum: float,
    samples: int,
) -> np.ndarray:
    """Return axis samples from the active view clipped to finite domain bounds."""

    view_low = min(view_minimum, view_maximum)
    view_high = max(view_minimum, view_maximum)
    domain_low = -np.inf if domain_minimum is None else min(domain_minimum, domain_maximum)
    domain_high = np.inf if domain_maximum is None else max(domain_minimum, domain_maximum)
    sample_low = max(domain_low, view_low)
    sample_high = min(domain_high, view_high)

    if sample_low > sample_high:
        return np.empty(0, dtype=float)
    if sample_low == sample_high:
        return np.asarray([sample_low], dtype=float)

    # Preserve the visible axis direction so manually reversed ranges continue
    # to produce trace data in the same orientation as the active view.
    if view_minimum <= view_maximum:
        return np.linspace(sample_low, sample_high, samples)
    return np.linspace(sample_high, sample_low, samples)


def _integer_index_values(view: ListView) -> np.ndarray:
    """Return the integer sample values for a normalized list view."""

    if view.inferred:
        if view.x_view is None:
            return np.empty(0, dtype=int)
        start = int(np.ceil(min(view.x_view.minimum, view.x_view.maximum)))
        stop = int(np.floor(max(view.x_view.minimum, view.x_view.maximum))) + 1
        if view.x_view.minimum <= view.x_view.maximum:
            return np.arange(start, stop, dtype=int)
        return np.arange(stop - 1, start - 1, -1, dtype=int)
    if view.minimum is None or view.maximum is None:
        return np.empty(0, dtype=int)
    return np.asarray(range(view.minimum, view.maximum, view.step), dtype=int)


def _evaluate_list_coordinate(
    numeric: object,
    n_values: np.ndarray,
    parameter_values: tuple[float, ...],
) -> np.ndarray:
    """Evaluate one scalar coordinate over integer list indices."""

    raw_values = _invoke_list_numeric(numeric, n_values, parameter_values)
    return _real_array_like(
        raw_values,
        n_values.shape,
        message=(
            "list_plot(...) expression output must be scalar or exactly two "
            "coordinates."
        ),
    )


def _invoke_list_numeric(
    numeric: object,
    n_values: np.ndarray,
    parameter_values: tuple[float, ...],
) -> object:
    """Invoke a compiled list expression, falling back to per-index calls."""

    try:
        return numeric(n_values, *parameter_values)  # type: ignore[misc]
    except Exception:
        values = []
        for index in n_values.tolist():
            try:
                values.append(numeric(int(index), *parameter_values))  # type: ignore[misc]
            except TypeError:
                values.append(numeric(int(index)))  # type: ignore[misc]
        return values


def _coordinates_from_list_output(
    raw_values: object,
    n_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinates from scalar or two-coordinate outputs."""

    array = np.asarray(raw_values)
    if array.ndim == 2 and array.shape == (n_values.size, 2):
        if np.iscomplexobj(array):
            raise PlotShapeError(
                "list_plot(...) expression output must be scalar or exactly two coordinates."
            )
        return array[:, 0].astype(float), array[:, 1].astype(float)
    if array.ndim == 2 and array.shape == (2, n_values.size):
        if np.iscomplexobj(array):
            raise PlotShapeError(
                "list_plot(...) expression output must be scalar or exactly two coordinates."
            )
        return array[0, :].astype(float), array[1, :].astype(float)
    y_values = _real_array_like(
        raw_values,
        n_values.shape,
        message=(
            "list_plot(...) expression output must be scalar or exactly two "
            "coordinates."
        ),
    )
    return n_values.astype(float), y_values


def _real_vector(raw_values: object, message: str) -> np.ndarray:
    """Return a one-dimensional real float vector."""

    array = np.asarray(raw_values)
    if np.iscomplexobj(array):
        raise PlotShapeError(message)
    try:
        result = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise PlotShapeError(message) from exc
    if result.ndim != 1:
        raise PlotShapeError(message)
    if not np.all(np.isfinite(result)):
        raise PlotShapeError(message)
    return result


def _real_array_like(raw_values: object, shape: tuple[int, ...], *, message: str) -> np.ndarray:
    """Return real float values broadcastable to ``shape``."""

    array = np.asarray(raw_values)
    if np.iscomplexobj(array):
        raise PlotShapeError(message)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, shape).astype(float)
    except (TypeError, ValueError) as exc:
        raise PlotShapeError(message) from exc


def _boolean_array_like(raw_values: object, shape: tuple[int, ...]) -> np.ndarray:
    """Return Boolean values broadcastable to ``shape``."""

    array = np.asarray(raw_values)
    if array.shape == ():
        return np.full(shape, bool(array), dtype=bool)
    try:
        return np.broadcast_to(array, shape).astype(bool)
    except (TypeError, ValueError) as exc:
        raise PlotShapeError(
            "domain_plot(...) signed expressions must evaluate to real scalar "
            "grid values. Boolean conditions are also accepted."
        ) from exc
