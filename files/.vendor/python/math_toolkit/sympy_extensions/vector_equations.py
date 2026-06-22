"""Convert between matrix-shaped equalities and scalar equation systems.

Use ``VectorEquation2SystemOfEquations(...)`` when a calculation is most
readable as one vector or matrix equation, but the next symbolic operation
needs the corresponding scalar component equations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sympy

__all__ = [
    "SystemOfEquations2VectorEquation",
    "VectorEquation2SystemOfEquations",
]


def SystemOfEquations2VectorEquation(
    system: Iterable[sympy.Equality],
    *,
    shape: tuple[int, int] | None = None,
) -> sympy.Equality:
    """Return one matrix-shaped equality from scalar component equations.

    Parameters
    ----------
    system : Iterable[sympy.Equality]
        Scalar equalities ordered row by row through the desired matrix
        entries.
    shape : tuple[int, int] | None, default=None
        Optional matrix shape for both sides. When omitted, the equations
        become a column matrix with one row per scalar equation.

    Returns
    -------
    sympy.Equality
        Unevaluated equality between the left-hand and right-hand matrices.

    Raises
    ------
    TypeError
        If ``system`` is not an iterable of SymPy equalities, or ``shape`` is
        not a pair of concrete integer dimensions.
    ValueError
        If ``shape`` does not contain exactly enough entries for ``system``.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import SystemOfEquations2VectorEquation
    >>> x, y = sympy.symbols("x y")
    >>> equations = [sympy.Eq(x, 1), sympy.Eq(y, 2)]
    >>> SystemOfEquations2VectorEquation(equations)
    Eq(Matrix([
    [x],
    [y]]), Matrix([
    [1],
    [2]]))
    """

    equations = _scalar_equations(system)
    rows, cols = _system_shape(shape, len(equations))
    left_entries = [equation.lhs for equation in equations]
    right_entries = [equation.rhs for equation in equations]

    return sympy.Eq(
        sympy.Matrix(rows, cols, left_entries),
        sympy.Matrix(rows, cols, right_entries),
        evaluate=False,
    )


def VectorEquation2SystemOfEquations(equation: sympy.Equality) -> list[sympy.Equality]:
    """Return scalar component equations from a matrix-shaped equality.

    Parameters
    ----------
    equation : sympy.Equality
        Equality whose left and right sides have the same finite matrix shape.

    Returns
    -------
    list[sympy.Equality]
        Scalar equalities ordered row by row through the matrix entries.

    Raises
    ------
    TypeError
        If ``equation`` is not a SymPy equality, either side is not
        matrix-shaped, or the matrix shape is not finite and concrete.
    ValueError
        If the two sides do not have the same shape.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import VectorEquation2SystemOfEquations
    >>> x, y = sympy.symbols("x y")
    >>> equation = sympy.Eq(
    ...     sympy.Matrix([x, x + 1]),
    ...     sympy.Matrix([y, 0]),
    ... )
    >>> VectorEquation2SystemOfEquations(equation)
    [Eq(x, y), Eq(x + 1, 0)]
    """

    if not isinstance(equation, sympy.Equality):
        raise TypeError(
            "VectorEquation2SystemOfEquations expects a SymPy Equality."
        )

    lhs_shape = _finite_matrix_shape(equation.lhs)
    rhs_shape = _finite_matrix_shape(equation.rhs)
    if lhs_shape is None or rhs_shape is None:
        raise TypeError(
            "VectorEquation2SystemOfEquations expects both sides to be "
            "matrix-shaped."
        )
    if lhs_shape != rhs_shape:
        raise ValueError(
            "VectorEquation2SystemOfEquations requires both sides to have "
            f"the same shape, got {lhs_shape} and {rhs_shape}."
        )

    # Preserve component equations even when one entry pair is visibly equal.
    # Downstream tools can still simplify or discard tautologies explicitly.
    rows, cols = lhs_shape
    scalar_equations: list[sympy.Equality] = []
    for row in range(rows):
        for col in range(cols):
            scalar_equations.append(
                sympy.Eq(
                    equation.lhs[row, col],
                    equation.rhs[row, col],
                    evaluate=False,
                )
            )
    return scalar_equations


def _finite_matrix_shape(obj: Any) -> tuple[int, int] | None:
    """Return a concrete two-dimensional matrix shape when one is available."""

    shape = getattr(obj, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        return None

    concrete_shape: list[int] = []
    for dimension in shape:
        try:
            concrete_dimension = int(dimension)
        except TypeError as exc:
            raise TypeError(
                "VectorEquation2SystemOfEquations requires finite concrete "
                f"matrix dimensions, got {shape}."
            ) from exc
        concrete_shape.append(concrete_dimension)
    return (concrete_shape[0], concrete_shape[1])


def _scalar_equations(system: Iterable[sympy.Equality]) -> list[sympy.Equality]:
    """Return a concrete list after validating scalar equations."""

    if isinstance(system, sympy.Equality):
        raise TypeError(
            "SystemOfEquations2VectorEquation expects an iterable of SymPy "
            "equalities, not a single equality."
        )
    try:
        equations = list(system)
    except TypeError as exc:
        raise TypeError(
            "SystemOfEquations2VectorEquation expects an iterable of SymPy "
            "equalities."
        ) from exc

    for equation in equations:
        if not isinstance(equation, sympy.Equality):
            raise TypeError(
                "SystemOfEquations2VectorEquation expects every system entry "
                "to be a SymPy Equality."
            )
    return equations


def _system_shape(
    shape: tuple[int, int] | None,
    entry_count: int,
) -> tuple[int, int]:
    """Return the output shape after checking it matches the system length."""

    if shape is None:
        return (entry_count, 1)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError(
            "SystemOfEquations2VectorEquation shape must be a "
            "(rows, columns) tuple."
        )

    rows, cols = _concrete_dimension(shape[0], shape), _concrete_dimension(
        shape[1],
        shape,
    )
    if rows * cols != entry_count:
        raise ValueError(
            "SystemOfEquations2VectorEquation shape must contain exactly "
            f"{entry_count} entries, got {shape}."
        )
    return rows, cols


def _concrete_dimension(dimension: Any, shape: tuple[Any, Any]) -> int:
    """Return one nonnegative concrete dimension."""

    try:
        concrete_dimension = int(dimension)
    except TypeError as exc:
        raise TypeError(
            "SystemOfEquations2VectorEquation shape requires concrete integer "
            f"dimensions, got {shape}."
        ) from exc

    if concrete_dimension < 0:
        raise ValueError(
            "SystemOfEquations2VectorEquation shape dimensions must be "
            f"nonnegative, got {shape}."
        )
    return concrete_dimension
