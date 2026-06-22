"""Build SymPy ``Piecewise`` expressions with fluent conditional syntax.

The public ``If(...)`` builder creates ordinary SymPy ``Piecewise``
expressions while keeping notebook definitions compact. Infix operators such as
    ``Eq``, ``And``, and ``Or`` provide symbolic condition syntax for places where
    Python operators would otherwise produce plain booleans or rely on precedence
    that is easy to misread.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

import sympy

__all__ = [
    "InfixAt",
    "Eq",
    "Ne",
    "And",
    "Or",
    "If",
    "IfBuilder",
    "Otherwise",
    "Else",
]

# Runtime help topic shared by the builder, infix operators, and branch helpers.
_PIECEWISE_IF_DOC = {
    "path": PurePosixPath("library/If"),
    "anchor": None,
    "label": "If",
}

# Private sentinel for an infix operator or builder state that is waiting for
# the next public DSL call.
_MISSING = object()


class InfixAt:
    """Adapt a two-argument callable to Python's matrix-multiply infix form.

    ``InfixAt`` supports notation such as ``x @Eq@ y``. Python parses that
    expression as ``(x @ Eq) @ y``, so the first operation stores the left
    operand and the second operation calls the wrapped function.

    Parameters
    ----------
    func : Callable[[Any, Any], Any]
        Function that receives the left and right operands.
    name : str | None, default=None
        Name used in representations and error messages. When omitted, the
        wrapped callable name is used when available.

    Methods
    -------
    __rmatmul__
        Store the left operand from ``left @ operator``.
    __matmul__
        Apply the wrapped callable to the stored left operand and the right
        operand.

    Raises
    ------
    TypeError
        If ``func`` is not callable.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions.sympy_ifdsl import InfixAt
    >>> x = sympy.Symbol("x")
    >>> LessThanOrEqual = InfixAt(sympy.Le, name="Le")
    >>> x @LessThanOrEqual@ 2
    x <= 2
    """

    __slots__ = ("_func", "_name", "_left")

    def __init__(
        self,
        func: Callable[[Any, Any], Any],
        *,
        name: str | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError("InfixAt func must be callable.")
        self._func = func
        self._name = name or getattr(func, "__name__", func.__class__.__name__)
        self._left: Any = _MISSING

    def __rmatmul__(self, left: Any) -> "InfixAt":
        """Return a partial operator that has captured the left operand.

        Parameters
        ----------
        left : Any
            Left operand from ``left @ operator``.

        Returns
        -------
        InfixAt
            Operator waiting for the right operand.

        Raises
        ------
        TypeError
            If the operator already has a captured left operand.
        """

        if self._left is not _MISSING:
            raise TypeError(f"Incomplete infix expression near {self!r}.")

        partial = type(self)(self._func, name=self._name)
        partial._left = left
        return partial

    def __matmul__(self, right: Any) -> Any:
        """Apply the wrapped callable to the captured left operand and ``right``.

        Parameters
        ----------
        right : Any
            Right operand from ``operator @ right``.

        Returns
        -------
        Any
            Result returned by the wrapped callable.

        Raises
        ------
        TypeError
            If no left operand has been captured.
        """

        if self._left is _MISSING:
            raise TypeError(f"Use as: left @{self._name}@ right.")
        return self._func(self._left, right)

    def __repr__(self) -> str:
        """Return an interactive representation of the operator state."""

        if self._left is _MISSING:
            return f"<{self._name}: use as left @{self._name}@ right>"
        return f"<{self._name}: waiting for right operand>"


InfixAt._mt_help = _PIECEWISE_IF_DOC

# Symbolic relation and Boolean-composition operators for ``@...@`` syntax.
Eq = InfixAt(sympy.Eq, name="Eq")
Ne = InfixAt(sympy.Ne, name="Ne")
And = InfixAt(sympy.And, name="And")
Or = InfixAt(sympy.Or, name="Or")


def If(cond: Any, *, evaluate: bool = True) -> "IfBuilder":
    """Start a chained ``Piecewise`` expression.

    Parameters
    ----------
    cond : Any
        First symbolic condition. Plain Python ``bool`` values are rejected
        because they usually indicate accidental use of Python equality.
    evaluate : bool, default=True
        Whether SymPy should evaluate and canonicalize the final
        ``Piecewise`` expression.

    Returns
    -------
    IfBuilder
        Builder waiting for ``Then(expr)``.

    Raises
    ------
    TypeError
        If ``cond`` is a plain Python ``bool``.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions.sympy_ifdsl import If
    >>> x = sympy.Symbol("x", real=True)
    >>> If(x < 0).Then(-x).Otherwise(x)
    Piecewise((-x, x < 0), (x, True))
    >>> If(x < 0).Then(-1).Otherwise()
    Piecewise((-1, x < 0))
    """

    return IfBuilder(cond, evaluate=evaluate)


class IfBuilder:
    """Build a chained SymPy ``Piecewise`` expression from ordered conditions.

    Each ``If(condition).Then(expr)`` pair contributes one branch. Branches are
    tried in order, matching the usual ``if`` / ``elif`` / ``else`` model, and
    closing the builder returns a real SymPy ``Piecewise`` expression, except
    for ``Otherwise(None)`` concrete misses, which return ``None``.

    Parameters
    ----------
    cond : Any
        First symbolic condition in the branch chain.
    evaluate : bool, default=True
        Whether SymPy should evaluate and canonicalize the final
        ``Piecewise`` expression.

    Methods
    -------
    Then
        Attach an expression to the most recent condition.
    If
        Start the next conditional branch.
    Otherwise
        Close the chain with an optional default expression.
    Done
        Close the chain without a default expression.

    Raises
    ------
    TypeError
        If ``cond`` is a plain Python ``bool``.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions.sympy_ifdsl import Eq, If
    >>> x = sympy.Symbol("x", real=True)
    >>> abs_x = If(x < 0).Then(-x).If(x @Eq@ 0).Then(0).Otherwise(x)
    >>> abs_x.subs(x, -3)
    3
    >>> abs_x.subs(x, 0)
    0
    >>> If(x < 0).Then(-1).If(x > 0).Then(1).Otherwise()
    Piecewise((-1, x < 0), (1, x > 0))
    """

    def __init__(self, cond: Any, *, evaluate: bool = True) -> None:
        self._pairs: list[tuple[Any, Any]] = []
        self._pending = _normalize_condition(cond)
        self._evaluate = evaluate
        self._closed = False

    def Then(self, expr: Any) -> "IfBuilder":
        """Attach an expression to the most recent condition.

        Parameters
        ----------
        expr : Any
            SymPy-compatible expression to use when the pending condition is
            true.

        Returns
        -------
        IfBuilder
            The same builder, ready for ``If(...)``, ``Otherwise(...)``, or
            ``Done()``.

        Raises
        ------
        RuntimeError
            If the builder has already been closed.
        SyntaxError
            If there is no pending condition.
        """

        self._check_open()
        if self._pending is _MISSING:
            raise SyntaxError("Then(expr) must follow If(condition).")
        self._pairs.append((expr, self._pending))
        self._pending = _MISSING
        return self

    def If(self, cond: Any) -> "IfBuilder":
        """Start another conditional branch in the chain.

        Parameters
        ----------
        cond : Any
            Symbolic condition for the next branch.

        Returns
        -------
        IfBuilder
            The same builder, waiting for ``Then(expr)``.

        Raises
        ------
        RuntimeError
            If the builder has already been closed.
        SyntaxError
            If the previous condition has not received a ``Then(expr)``.
        TypeError
            If ``cond`` is a plain Python ``bool``.
        """

        self._check_open()
        if self._pending is not _MISSING:
            raise SyntaxError("Missing Then(expr) before the next If(condition).")
        self._pending = _normalize_condition(cond)
        return self

    def Otherwise(self, expr: Any = _MISSING) -> sympy.Piecewise | None:
        """Close the chain with an optional default branch.

        Parameters
        ----------
        expr : Any, optional
            SymPy-compatible expression used when no earlier condition is true.
            Omit this argument to close the chain without a default branch.
            Pass ``None`` to return ``None`` only when all earlier conditions
            are concrete falsehoods.

        Returns
        -------
        sympy.Piecewise | None
            Final piecewise expression, or ``None`` when ``expr`` is ``None``
            and no earlier branch can match.

        Raises
        ------
        RuntimeError
            If the builder has already been closed.
        SyntaxError
            If the most recent condition has not received a ``Then(expr)``.
        """

        self._check_open()
        if self._pending is not _MISSING:
            raise SyntaxError("Missing Then(expr) before closing the If expression.")
        if expr is _MISSING:
            return self.Done()
        if expr is None:
            self._closed = True
            return _piecewise_or_none(self._pairs, evaluate=self._evaluate)
        self._pairs.append((expr, sympy.S.true))
        self._closed = True
        return sympy.Piecewise(*self._pairs, evaluate=self._evaluate)

    def Done(self) -> sympy.Piecewise:
        """Close the chain without a default branch.

        Returns
        -------
        sympy.Piecewise
            Final piecewise expression with no default branch.

        Raises
        ------
        RuntimeError
            If the builder has already been closed.
        SyntaxError
            If the most recent condition has not received a ``Then(expr)``.
        """

        self._check_open()
        if self._pending is not _MISSING:
            raise SyntaxError("Missing Then(expr) before closing the If expression.")
        self._closed = True
        return sympy.Piecewise(*self._pairs, evaluate=self._evaluate)

    # Lowercase aliases keep the fluent API usable in ordinary Python style.
    then = Then
    otherwise = Otherwise
    done = Done

    def _check_open(self) -> None:
        """Raise when the builder has already returned a ``Piecewise``."""

        if self._closed:
            raise RuntimeError("This If expression is already closed.")


IfBuilder._mt_help = _PIECEWISE_IF_DOC


def Otherwise(expr: Any) -> tuple[Any, Any]:
    """Return a default branch tuple for a raw ``sympy.Piecewise`` call.

    Parameters
    ----------
    expr : Any
        SymPy-compatible default expression.

    Returns
    -------
    tuple[Any, Any]
        ``(expr, sympy.S.true)`` for use as the final ``Piecewise`` branch.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions.sympy_ifdsl import Otherwise
    >>> x = sympy.Symbol("x", real=True)
    >>> sympy.Piecewise((x, x > 0), Otherwise(0))
    Piecewise((x, x > 0), (0, True))
    """

    return (expr, sympy.S.true)


Else = Otherwise

If._mt_help = _PIECEWISE_IF_DOC
Otherwise._mt_help = _PIECEWISE_IF_DOC


def _piecewise_or_none(
    pairs: list[tuple[Any, Any]],
    *,
    evaluate: bool,
) -> sympy.Piecewise | None:
    """Return a ``Piecewise`` expression or ``None`` for concrete misses."""

    # ``None`` is not a SymPy expression, so it cannot be stored as a
    # ``Piecewise`` branch. When the earlier conditions decide the result
    # concretely, return that value or ``None`` directly.
    undecided_pairs: list[tuple[Any, Any]] = []
    for expr, cond in pairs:
        if cond is sympy.S.true:
            return expr
        if cond is sympy.S.false:
            continue
        undecided_pairs.append((expr, cond))

    if not undecided_pairs:
        return None
    return sympy.Piecewise(*undecided_pairs, evaluate=evaluate)


def _normalize_condition(cond: Any) -> Any:
    """Return a symbolic condition after rejecting plain Python booleans."""

    if isinstance(cond, bool):
        raise TypeError(
            "If(condition) received a plain Python bool. "
            "For symbolic equality, use x @Eq@ y, not x == y. "
            "For the default branch, use .Otherwise(expr)."
        )
    return cond
