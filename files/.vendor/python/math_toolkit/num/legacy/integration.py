"""Integrate numeric and symbolic functions with SciPy cubature."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cubature
import sympy

from ...sympy_extensions.sympy_ifdsl import If
from .compile import compile_num
from .diagnostics import NumArgumentError
from .functions import NumFunction


@dataclass(frozen=True)
class _CubatureRectangle:
    """Store normalized cubature bounds and optional coordinate symbols."""

    symbols: tuple[sympy.Symbol | None, ...]
    lower: np.ndarray
    upper: np.ndarray


def integrate(
    function: object,
    ranges: object,
    *,
    domain_func: object | None = None,
    args: tuple[object, ...] = (),
    rule: str = "gk21",
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_subdivisions: int = 10000,
    workers: int | Callable[..., object] = 1,
    points: object | None = None,
) -> object:
    """Integrate a vectorized function over a rectangular cubature domain.

    Parameters
    ----------
    function : object
        Symbolic expression, compiled ``NumFunction``, or vectorized Python
        callable to integrate.
    ranges : object
        Rectangular bounds. Use ``[(x, a, b), (y, c, d)]`` for symbolic
        integrands or ``[(a, b), (c, d)]`` for purely numeric integrands.
    domain_func : object, optional
        Symbolic expression or vectorized Python callable that is negative
        inside the desired domain and positive outside it.
    args : tuple[object, ...], optional
        Extra positional arguments passed after the coordinate arrays.
    rule : str, optional
        Cubature rule passed to :func:`scipy.integrate.cubature`.
    rtol : float, optional
        Relative integration tolerance.
    atol : float, optional
        Absolute integration tolerance.
    max_subdivisions : int, optional
        Maximum cubature subdivisions.
    workers : int or callable, optional
        Worker configuration passed to SciPy.
    points : object, optional
        Points passed to SciPy for avoiding singularities.

    Returns
    -------
    object
        SciPy ``CubatureResult`` containing the integral estimate and error.
    """

    rectangle = _normalize_rectangle(ranges)
    coordinate_symbols = _coordinate_symbols_for(
        rectangle,
        function=function,
        domain_func=domain_func,
    )
    callback = _cubature_callback(
        function,
        domain_func=domain_func,
        coordinate_symbols=coordinate_symbols,
        args=args,
    )
    return cubature(
        callback,
        rectangle.lower,
        rectangle.upper,
        rule=rule,
        rtol=rtol,
        atol=atol,
        max_subdivisions=max_subdivisions,
        workers=workers,
        points=points,
    )


def _normalize_rectangle(ranges: object) -> _CubatureRectangle:
    """Return coordinate symbols and numeric endpoints from public ranges."""

    entries = _range_entries(ranges)
    symbols: list[sympy.Symbol | None] = []
    lower: list[float] = []
    upper: list[float] = []

    for entry in entries:
        symbol, left, right = _normalize_range_entry(entry)
        symbols.append(symbol)
        lower.append(float(left))
        upper.append(float(right))

    if not lower:
        raise NumArgumentError("Num.Integrate requires at least one interval.")
    return _CubatureRectangle(
        symbols=tuple(symbols),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


def _range_entries(ranges: object) -> tuple[object, ...]:
    """Return interval entries while accepting a convenient one-dimensional form."""

    if not isinstance(ranges, Sequence) or isinstance(ranges, str | bytes):
        raise NumArgumentError(
            "Num.Integrate ranges must be an interval or a sequence of intervals."
        )
    if len(ranges) == 2 and not _looks_like_nested_intervals(ranges):
        return (ranges,)
    if len(ranges) == 3 and isinstance(ranges[0], sympy.Symbol):
        return (ranges,)
    return tuple(ranges)


def _looks_like_nested_intervals(ranges: Sequence[object]) -> bool:
    """Return whether a length-two value already looks like two intervals."""

    return all(
        isinstance(entry, Sequence)
        and not isinstance(entry, str | bytes)
        and len(entry) in {2, 3}
        for entry in ranges
    )


def _normalize_range_entry(
    entry: object,
) -> tuple[sympy.Symbol | None, object, object]:
    """Return ``(symbol, lower, upper)`` for one interval entry."""

    if not isinstance(entry, Sequence) or isinstance(entry, str | bytes):
        raise NumArgumentError(
            "Each Num.Integrate interval must be (lower, upper), "
            "(symbol, lower, upper), or (symbol, (lower, upper))."
        )

    if len(entry) == 2:
        first, second = entry
        if isinstance(first, sympy.Symbol):
            if (
                not isinstance(second, Sequence)
                or isinstance(second, str | bytes)
                or len(second) != 2
            ):
                raise NumArgumentError(
                    "Symbolic Num.Integrate intervals must include two bounds."
                )
            lower, upper = second
            return first, lower, upper
        return None, first, second

    if len(entry) == 3 and isinstance(entry[0], sympy.Symbol):
        symbol, lower, upper = entry
        return symbol, lower, upper

    raise NumArgumentError(
        "Each Num.Integrate interval must be (lower, upper), "
        "(symbol, lower, upper), or (symbol, (lower, upper))."
    )


def _coordinate_symbols_for(
    rectangle: _CubatureRectangle,
    *,
    function: object,
    domain_func: object | None,
) -> tuple[sympy.Symbol, ...] | None:
    """Return coordinate symbols when symbolic expressions need compilation."""

    explicit_symbols = tuple(symbol for symbol in rectangle.symbols if symbol is not None)
    if len(explicit_symbols) != len(rectangle.symbols) and explicit_symbols:
        raise NumArgumentError(
            "Num.Integrate ranges must either all name coordinate symbols or "
            "all omit them."
        )

    if explicit_symbols:
        return explicit_symbols

    symbolic_symbols = sorted(
        _free_symbols(function) | _free_symbols(domain_func),
        key=lambda symbol: symbol.name,
    )
    if not symbolic_symbols:
        return None
    if len(symbolic_symbols) != len(rectangle.symbols):
        raise NumArgumentError(
            "Symbolic Num.Integrate inputs need one coordinate symbol per "
            "range, such as [(x, -1, 1), (y, -1, 1)]."
        )
    return tuple(symbolic_symbols)


def _cubature_callback(
    function: object,
    *,
    domain_func: object | None,
    coordinate_symbols: tuple[sympy.Symbol, ...] | None,
    args: tuple[object, ...],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a SciPy cubature callback with sample-first output shape."""

    symbolic_domain = _is_symbolic(domain_func)
    if coordinate_symbols is not None and _is_symbolic(function):
        function = _compile_symbolic_integrand(
            function,
            domain_func=domain_func,
            coordinate_symbols=coordinate_symbols,
        )
        if symbolic_domain:
            domain_func = None
    elif coordinate_symbols is not None and _is_symbolic(domain_func):
        domain_func = compile_num(domain_func, args=coordinate_symbols)

    if not callable(function):
        constant = sympy.sympify(function)
        if coordinate_symbols is None:
            function = lambda *unused_args: constant
        else:
            function = compile_num(constant, args=coordinate_symbols)

    if domain_func is not None and not callable(domain_func):
        domain_func = sympy.sympify(domain_func)
        if coordinate_symbols is None:
            constant_domain = domain_func
            domain_func = lambda *unused_args: constant_domain
        else:
            domain_func = compile_num(domain_func, args=coordinate_symbols)

    def callback(points: np.ndarray) -> np.ndarray:
        values = _evaluate_vectorized(function, points, args)
        if domain_func is None:
            return values

        domain_values = _evaluate_vectorized(domain_func, points, args)
        mask = np.asarray(domain_values) < 0
        while mask.ndim < values.ndim:
            mask = mask[..., np.newaxis]
        return np.where(mask, values, 0)

    return callback


def _compile_symbolic_integrand(
    function: object,
    *,
    domain_func: object | None,
    coordinate_symbols: tuple[sympy.Symbol, ...],
) -> NumFunction:
    """Compile a symbolic integrand, including a symbolic domain mask when possible."""

    expression = function
    if domain_func is not None and _is_symbolic(domain_func):
        indicator = If(sympy.sympify(domain_func) < 0).Then(1).Otherwise(0)
        expression = sympy.sympify(function) * indicator
    return compile_num(expression, args=coordinate_symbols)


def _evaluate_vectorized(
    function: Callable[..., object],
    points: np.ndarray,
    args: tuple[object, ...],
) -> np.ndarray:
    """Evaluate a vectorized callable using one coordinate array per dimension."""

    coordinate_columns = tuple(points[:, axis] for axis in range(points.shape[1]))
    values = function(*coordinate_columns, *args)
    return _sample_first_array(values, points.shape[0])


def _sample_first_array(value: object, sample_count: int) -> np.ndarray:
    """Return an array whose leading axis has one entry per cubature sample."""

    array = np.asarray(value)
    if array.shape == ():
        return np.full((sample_count,), array)
    if array.shape[0] == sample_count:
        return array
    return np.broadcast_to(array, (sample_count,) + array.shape)


def _is_symbolic(value: object) -> bool:
    """Return whether ``value`` is a symbolic expression accepted by ``Num``."""

    return isinstance(value, sympy.Basic | sympy.MatrixBase)


def _free_symbols(value: object | None) -> set[sympy.Symbol]:
    """Return free symbols from symbolic inputs only."""

    if value is None or callable(value):
        return set()
    if isinstance(value, sympy.MatrixBase):
        symbols: set[sympy.Symbol] = set()
        for entry in value:
            symbols.update(entry.free_symbols)
        return symbols
    if isinstance(value, sympy.Basic):
        return set(value.free_symbols)
    return set()
