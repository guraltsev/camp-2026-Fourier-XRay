"""Hold symbolic expressions behind explicit definition boundaries.

Use ``Hold(...)`` to preserve a formula or named definition as opaque notation,
``Unhold(...)`` to expose one held boundary, and ``UnholdAll(...)`` to expose
held boundaries recursively. Future versions may add protected-subtree controls
for recursive unholding; this module currently expands every reachable held
boundary.
"""

from __future__ import annotations

from typing import Any

import sympy

from .free_symbols import UniversalFreeSymbols

__all__ = [
    "HeldExpression",
    "Hold",
    "Unhold",
    "UnholdAll",
]


class HeldExpression(sympy.AtomicExpr):
    """Store an anonymous held formula as an atomic SymPy expression.

    Parameters
    ----------
    body : Any
        SymPy-compatible expression to keep behind the held boundary.

    Methods
    -------
    _hold
        Return this object unchanged.
    _unhold
        Return the held body.

    Notes
    -----
    Ordinary SymPy traversal treats ``HeldExpression`` as an atom. Its
    ``free_symbols`` set is universal: calculus routines should treat the held
    body as opaque notation that may depend on any differentiation variable.
    """

    is_commutative = True

    def __new__(cls, body: Any) -> "HeldExpression":
        """Create an atomic held expression wrapper."""

        obj = sympy.AtomicExpr.__new__(cls)
        obj._math_toolkit_held_body = body
        return obj

    @property
    def body(self) -> Any:
        """Return the formula stored behind this held boundary."""

        return self._math_toolkit_held_body

    @property
    def free_symbols(self) -> UniversalFreeSymbols:
        """Return an unknown free-symbol set for the opaque held body."""

        return UniversalFreeSymbols()

    def has(self, *patterns: Any) -> bool:
        """Return whether a pattern is visible at the held boundary."""

        return any(pattern == self for pattern in patterns)

    def is_constant(self, *wrt: Any, **flags: Any) -> bool:
        """Return constancy from the held body's unknown dependencies."""

        if not wrt:
            return not bool(self.free_symbols)

        return not any(_held_body_may_depend_on(self, item) for item in wrt)

    def _eval_derivative_n_times(self, symbol: Any, count: Any) -> Any:
        """Keep dependent held derivatives unevaluated."""

        if count == 0:
            return self
        if _held_body_may_depend_on(self, symbol):
            return sympy.Derivative(self, (symbol, count), evaluate=False)
        return sympy.S.Zero

    def _hold(self) -> "HeldExpression":
        """Return this held expression unchanged."""

        return self

    def _unhold(self) -> Any:
        """Return the formula stored behind this held boundary."""

        return self._math_toolkit_held_body

    def _eval_subs(self, old: Any, new: Any) -> Any:
        """Substitute only the wrapper itself, not the stored body."""

        if old == self:
            return new
        return None

    def _hashable_content(self) -> tuple[Any, ...]:
        """Return equality and hashing content for this held wrapper."""

        return (_hashable_held_content(self._math_toolkit_held_body),)

    def _sympystr(self, printer: Any) -> str:
        """Return a readable held-expression display form."""

        return f"Hold({printer._print(self._math_toolkit_held_body)})"

    def _latex(self, printer: Any) -> str:
        """Render the held body in LaTeX while keeping it semantically held."""

        return printer._print(self._math_toolkit_held_body)


def _hold(obj: Any) -> Any:
    """Return ``obj`` behind a held boundary.

    Parameters
    ----------
    obj : Any
        Object to hold. Objects may customize holding through a callable
        private ``_hold`` member.

    Returns
    -------
    Any
        Custom held object returned by ``obj._hold()``, or a generic
        ``HeldExpression`` wrapper.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit import Hold, Unhold
    >>> x = sympy.Symbol("x")
    >>> held = Hold((x + 1)**2)
    >>> Unhold(held)
    (x + 1)**2
    """

    custom = getattr(obj, "_hold", None)
    if callable(custom):
        return custom()
    return HeldExpression(obj)


def _unhold(
    obj: Any,
    *,
    recursive: bool = False,
    order: str = "top",
    max_steps: int = 100,
) -> Any:
    """Expose one or more held boundaries.

    Parameters
    ----------
    obj : Any
        Object that may contain held boundaries.
    recursive : bool, default=False
        Whether to walk through the expression and keep unholding until the
        result stabilizes.
    order : {"top", "depth"}, default="top"
        Recursive traversal order. ``"top"`` exposes outer boundaries before
        traversing into newly exposed bodies. ``"depth"`` traverses children
        before exposing the current object.
    max_steps : int, default=100
        Maximum number of changing recursive passes.

    Returns
    -------
    Any
        Object after one held boundary, or all reachable held boundaries, have
        been exposed.

    Raises
    ------
    ValueError
        If ``order`` or ``max_steps`` is invalid.
    RecursionError
        If recursive unholding does not stabilize within ``max_steps``.
    """

    if order not in {"top", "depth"}:
        raise ValueError("Unhold order must be 'top' or 'depth'.")
    if max_steps < 1:
        raise ValueError("Unhold max_steps must be at least one.")

    if not recursive:
        return _unhold_one(obj)

    # Repeat shallow recursive passes so a body exposed by one pass can be
    # traversed by the next pass. This makes cycle handling explicit.
    result = obj
    for _ in range(max_steps):
        opened = _unhold_pass(result, order=order)
        if _same_value(opened, result):
            return opened
        result = opened

    raise RecursionError(
        f"unholding did not stabilize after {max_steps} recursive passes."
    )


def _unhold_all(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Expose all reachable held boundaries in ``obj``.

    Parameters
    ----------
    obj : Any
        Object that may contain held boundaries.
    *args : Any
        Positional arguments forwarded to ``Unhold``.
    **kwargs : Any
        Keyword arguments forwarded to ``Unhold``.

    Returns
    -------
    Any
        Result of ``Unhold(obj, recursive=True, ...)``.
    """

    return _unhold(obj, *args, recursive=True, **kwargs)


class _HoldOperator:
    """Callable and pipeline-capable hold operator."""

    __name__ = "Hold"
    __qualname__ = "Hold"
    __doc__ = _hold.__doc__

    def __call__(self, obj: Any) -> Any:
        """Return ``obj`` behind a held boundary."""

        return _hold(obj)

    def __rrshift__(self, obj: Any) -> Any:
        """Hold the object on the left of ``>>``."""

        return _hold(obj)

    def __repr__(self) -> str:
        """Return an interactive representation of the wrapper."""

        return "<Hold: use as Hold(expr) or expr >> Hold>"


class _UnholdOperator:
    """Callable and pipeline-capable unhold operator."""

    __name__ = "Unhold"
    __qualname__ = "Unhold"
    __doc__ = _unhold.__doc__

    def __call__(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Expose one or more held boundaries in ``obj``."""

        return _unhold(obj, *args, **kwargs)

    def __rrshift__(self, obj: Any) -> Any:
        """Unhold the object on the left of ``>>`` once."""

        return _unhold(obj)

    def __repr__(self) -> str:
        """Return an interactive representation of the wrapper."""

        return "<Unhold: use as Unhold(expr) or expr >> Unhold>"


class _UnholdAllOperator:
    """Callable and pipeline-capable recursive unhold operator."""

    __name__ = "UnholdAll"
    __qualname__ = "UnholdAll"
    __doc__ = _unhold_all.__doc__

    def __call__(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Expose every reachable held boundary in ``obj``."""

        return _unhold_all(obj, *args, **kwargs)

    def __rrshift__(self, obj: Any) -> Any:
        """Unhold every reachable boundary in the object on the left of ``>>``."""

        return _unhold_all(obj)

    def __repr__(self) -> str:
        """Return an interactive representation of the wrapper."""

        return "<UnholdAll: use as UnholdAll(expr) or expr >> UnholdAll>"


Hold = _HoldOperator()
Unhold = _UnholdOperator()
UnholdAll = _UnholdAllOperator()


def _unhold_one(obj: Any) -> Any:
    """Return one exposed held boundary when ``obj`` provides the protocol."""

    custom = getattr(obj, "_unhold", None)
    if callable(custom):
        return custom()
    return obj


def _unhold_pass(obj: Any, *, order: str) -> Any:
    """Return one recursive unholding pass through ``obj``."""

    if order == "top":
        opened = _unhold_one(obj)
        if not _same_value(opened, obj):
            return opened
        return _unhold_children(obj, order=order)

    opened_children = _unhold_children(obj, order=order)
    if not _same_value(opened_children, obj):
        return opened_children
    return _unhold_one(obj)


def _unhold_children(obj: Any, *, order: str) -> Any:
    """Return ``obj`` rebuilt after unholding through traversable children."""

    if isinstance(obj, sympy.MatrixBase):
        return obj.applyfunc(lambda item: _unhold_pass(item, order=order))

    if isinstance(obj, sympy.Basic) and obj.args:
        new_args = tuple(_unhold_pass(arg, order=order) for arg in obj.args)
        if all(_same_value(new, old) for new, old in zip(new_args, obj.args)):
            return obj
        try:
            return obj.func(*new_args)
        except Exception:
            return obj

    if isinstance(obj, tuple):
        return tuple(_unhold_pass(item, order=order) for item in obj)
    if isinstance(obj, list):
        return [_unhold_pass(item, order=order) for item in obj]
    return obj


def _same_value(left: Any, right: Any) -> bool:
    """Return whether two values should stop an unholding iteration."""

    try:
        return bool(left == right)
    except Exception:
        return left is right


def _hashable_held_content(value: Any) -> Any:
    """Return stable equality content for arbitrary held bodies."""

    if isinstance(value, sympy.MatrixBase):
        return sympy.ImmutableMatrix(value)
    if isinstance(value, list):
        return tuple(_hashable_held_content(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_held_content(item) for item in value)
    try:
        hash(value)
    except Exception:
        return repr(value)
    return value


def _held_body_may_depend_on(held: HeldExpression, variable: Any) -> bool:
    """Return whether ``held`` may depend on a calculus variable."""

    try:
        candidate = sympy.sympify(variable)
    except Exception:
        return False

    visible_symbols = held.free_symbols
    if isinstance(candidate, sympy.Symbol):
        return candidate in visible_symbols
    return bool(visible_symbols & set(getattr(candidate, "free_symbols", set())))
