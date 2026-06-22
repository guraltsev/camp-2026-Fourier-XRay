"""Provide uppercase symbolic pipe operators with explicit, unambiguous payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import sympy

from .core import PipeOp, pipeop

__all__ = [
    "Diff",
    "DoIt",
    "Evalf",
    "Expand",
    "Replace",
    "Series",
    "Simplify",
    "Subs",
    "Taylor",
]

_PIPEOPS_DOC = {
    "path": PurePosixPath("library/PipeOps"),
    "anchor": None,
    "label": "PipeOps",
}

_DOIT_DOC = {
    "path": PurePosixPath("library/DoIt"),
    "anchor": None,
    "label": "DoIt",
}


def _expand(expr: Any) -> Any:
    """Return the SymPy expansion of ``expr``."""

    return sympy.expand(expr)


def _simplify(expr: Any) -> Any:
    """Return the SymPy simplification of ``expr``."""

    return sympy.simplify(expr)


def _subs(expr: Any, replacements: Any) -> Any:
    """Apply one substitution payload to ``expr``."""

    return expr.subs(replacements)


def _replace(expr: Any, replacements: Mapping[Any, Any]) -> Any:
    """Apply an exact-structure replacement map to ``expr``."""

    return expr.xreplace(replacements)


def _evalf(expr: Any, options: Any) -> Any:
    """Evaluate ``expr`` numerically with one explicit option payload."""

    if isinstance(options, Mapping):
        return expr.evalf(**dict(options))

    if isinstance(options, Sequence) and not isinstance(options, str | bytes):
        return expr.evalf(*options)

    return expr.evalf(options)


def _doit(expr: Any, **hints: Any) -> Any:
    """Evaluate pending symbolic operations in an expression.

    ``DoIt`` is a pipe-aware wrapper around ``expr.doit(**hints)``. It is
    useful after building expressions that intentionally contain unevaluated
    SymPy operations, such as ``Integral``, ``Sum``, ``Product``, ``Limit``, or
    ``Derivative``. Direct calls evaluate the expression immediately, while
    curried calls such as ``DoIt(deep=False)`` capture SymPy ``doit`` hints for
    later pipeline execution.

    Parameters
    ----------
    expr : object
        SymPy expression or expression-like object that provides a ``doit``
        method.
    **hints : object
        Keyword hints forwarded unchanged to ``expr.doit``. Common hints include
        ``deep=False`` when only the outer pending operation should be
        evaluated.

    Returns
    -------
    object
        The result returned by ``expr.doit(**hints)``.

    Notes
    -----
    ``DoIt`` performs SymPy's explicit pending-operation evaluation. It does
    not substitute values, simplify algebra, expand products, or force numeric
    evaluation beyond what the target object's ``doit`` implementation chooses.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import DoIt
    >>> x = sympy.Symbol("x")
    >>> DoIt(sympy.Sum(x, (x, 1, 3)))
    6

    Pipeline usage:

    >>> sympy.Integral(x, (x, 0, 1)) >> DoIt
    1/2

    Hints:

    >>> expr = 2 * sympy.Integral(x, x)
    >>> expr >> DoIt(deep=False)
    2*Integral(x, x)
    """

    return expr.doit(**hints)


def _diff(expr: Any, specification: Any) -> Any:
    """Differentiate ``expr`` using one explicit differentiation specification."""

    if (
        isinstance(specification, Sequence)
        and not isinstance(specification, str | bytes)
        and specification
        and all(isinstance(item, Sequence) for item in specification)
    ):
        return sympy.diff(expr, *specification)
    return sympy.diff(expr, specification)


def _series(expr: Any, specification: Any) -> Any:
    """Expand ``expr`` into a series using one explicit expansion specification."""

    if isinstance(specification, Mapping):
        variable = specification["x"]
        x0 = specification.get("x0", 0)
        order = specification["n"]
        direction = specification.get("dir", "+")
        return sympy.series(expr, variable, x0, order, dir=direction)

    if isinstance(specification, Sequence) and not isinstance(
        specification, str | bytes
    ):
        return sympy.series(expr, *specification)

    return sympy.series(expr, specification)


Expand = pipeop(_expand, name="Expand")
Simplify = pipeop(_simplify, name="Simplify")
Subs = pipeop(_subs, name="Subs")
Replace = pipeop(_replace, name="Replace")
Evalf = pipeop(_evalf, name="Evalf")
DoIt = pipeop(_doit, name="DoIt")
Diff = pipeop(_diff, name="Diff")
Series = pipeop(_series, name="Series")
Taylor = pipeop(_series, name="Taylor")

for _name in __all__:
    _operator = locals()[_name]
    _operator._mt_help = _PIPEOPS_DOC
    _operator.fn._mt_help = _PIPEOPS_DOC

DoIt._mt_help = _DOIT_DOC
DoIt.fn._mt_help = _DOIT_DOC
