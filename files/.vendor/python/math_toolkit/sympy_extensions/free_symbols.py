"""Preserve extensible ``free_symbols`` set semantics in SymPy expressions."""

from __future__ import annotations

from typing import Any

import sympy
from sympy.core.basic import Basic

__all__ = [
    "UniversalFreeSymbols",
    "patch",
]

# Patch state keeps package-profile imports idempotent. The captured property is
# available for diagnostics and for future compatibility checks.
_PATCHED = False
_ORIGINAL_BASIC_FREE_SYMBOLS = Basic.free_symbols


class UniversalFreeSymbols(set):
    """Represent an unknown set of free symbolic dependencies.

    ``UniversalFreeSymbols`` is a ``set`` subclass because SymPy's derivative
    constructor requires ``free_symbols`` to return a ``set`` instance. It
    models an expression that may depend on any free symbolic variable, while
    preserving ordinary set algebra where finite sets interact with the
    universal dependency set.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions.free_symbols import UniversalFreeSymbols
    >>> x = sympy.Symbol("x")
    >>> y = sympy.Symbol("y")
    >>> symbols = UniversalFreeSymbols()
    >>> x in symbols
    True
    >>> symbols | {x}
    UniversalFreeSymbols()
    >>> symbols & {x, y} == {x, y}
    True
    >>> {x, y} - symbols == set()
    True
    """

    def __contains__(self, item: object) -> bool:
        """Return whether ``item`` may be a free symbolic dependency."""

        return isinstance(item, sympy.Basic) and bool(item.free_symbols)

    def __bool__(self) -> bool:
        """Return ``True`` because the dependency set is not empty."""

        return True

    def __len__(self) -> int:
        """Return a conservative non-singleton size for ambiguity checks."""

        return 2

    def __repr__(self) -> str:
        """Return the constructor-style representation."""

        return "UniversalFreeSymbols()"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is also a universal free-symbol set."""

        return isinstance(other, UniversalFreeSymbols)

    def __ne__(self, other: object) -> bool:
        """Return whether ``other`` is not a universal free-symbol set."""

        return not self == other

    def copy(self) -> "UniversalFreeSymbols":
        """Return an independent universal free-symbol set."""

        return type(self)()

    def union(self, *others: Any) -> "UniversalFreeSymbols":
        """Return the universal dependency set."""

        return type(self)()

    def __or__(self, other: Any) -> "UniversalFreeSymbols":
        """Return the universal dependency set."""

        return self.union(other)

    def __ror__(self, other: Any) -> "UniversalFreeSymbols":
        """Return the universal dependency set."""

        return self.union(other)

    def intersection(self, *others: Any) -> set[Any] | "UniversalFreeSymbols":
        """Return finite information preserved by intersecting with universal."""

        if not others:
            return type(self)()

        result: set[Any] | None = None
        for other in others:
            if isinstance(other, UniversalFreeSymbols):
                continue
            if result is None:
                result = set(other)
                continue
            result.intersection_update(other)
        if result is None:
            return type(self)()
        return result

    def __and__(self, other: Any) -> set[Any] | "UniversalFreeSymbols":
        """Return the finite set selected by the other operand."""

        return self.intersection(other)

    def __rand__(self, other: Any) -> set[Any] | "UniversalFreeSymbols":
        """Return the finite set selected by the other operand."""

        return self.intersection(other)

    def difference(self, *others: Any) -> "UniversalFreeSymbols":
        """Return universal dependencies after removing finite information."""

        return type(self)()

    def __sub__(self, other: Any) -> "UniversalFreeSymbols":
        """Return universal dependencies after removing finite information."""

        return self.difference(other)

    def __rsub__(self, other: Any) -> set[Any]:
        """Return the empty finite set after subtracting universal."""

        return set()

    def symmetric_difference(self, other: Any) -> "UniversalFreeSymbols":
        """Return universal dependencies for symmetric difference."""

        return type(self)()

    def __xor__(self, other: Any) -> "UniversalFreeSymbols":
        """Return universal dependencies for symmetric difference."""

        return self.symmetric_difference(other)

    def __rxor__(self, other: Any) -> "UniversalFreeSymbols":
        """Return universal dependencies for symmetric difference."""

        return self.symmetric_difference(other)

    def issubset(self, other: Any) -> bool:
        """Return whether universal dependencies fit inside ``other``."""

        return isinstance(other, UniversalFreeSymbols)

    def __le__(self, other: Any) -> bool:
        """Return whether universal dependencies fit inside ``other``."""

        return self.issubset(other)

    def __lt__(self, other: Any) -> bool:
        """Return whether universal dependencies are a proper subset."""

        return False

    def issuperset(self, other: Any) -> bool:
        """Return ``True`` because universal contains every finite dependency."""

        return True

    def __ge__(self, other: Any) -> bool:
        """Return ``True`` because universal contains finite dependencies."""

        return self.issuperset(other)

    def __gt__(self, other: Any) -> bool:
        """Return whether universal strictly contains ``other``."""

        return not isinstance(other, UniversalFreeSymbols)

    def isdisjoint(self, other: Any) -> bool:
        """Return whether ``other`` has no possible dependency symbols."""

        return not bool(other)

    def update(self, *others: Any) -> None:
        """Keep the set universal after in-place union."""

        return None

    def difference_update(self, *others: Any) -> None:
        """Keep the set universal after removing finite information."""

        return None

    def symmetric_difference_update(self, other: Any) -> None:
        """Keep the set universal after in-place symmetric difference."""

        return None

    def intersection_update(self, *others: Any) -> None:
        """Keep the set universal after in-place intersection."""

        return None

    def add(self, element: Any) -> None:
        """Keep the set universal after adding one finite dependency."""

        return None

    def discard(self, element: Any) -> None:
        """Keep the set universal after discarding finite information."""

        return None

    def remove(self, element: Any) -> None:
        """Keep the set universal after removing finite information."""

        return None

    def clear(self) -> None:
        """Keep the set universal when callers try to clear finite contents."""

        return None


def _combine_free_symbol_sets(args: tuple[Basic, ...]) -> set[Basic]:
    """Return the union of child free-symbol sets with subclass dispatch."""

    # Keep SymPy's optimized exact-set behavior for ordinary expressions. Once
    # a subclass appears, use the binary operator protocol so specialized set
    # algebra can participate.
    symbol_sets = tuple(arg.free_symbols for arg in args)
    if all(type(symbols) is set for symbols in symbol_sets):
        return set().union(*symbol_sets)

    result: set[Basic] = set()
    for symbols in symbol_sets:
        if type(result) is set and type(symbols) is set:
            result.update(symbols)
            continue
        try:
            result = result | symbols
        except TypeError:
            result.update(symbols)
    return result


@property
def _basic_free_symbols(self: Basic) -> set[Basic]:
    """Return child free symbols while preserving specialized set semantics."""

    return _combine_free_symbol_sets(self.args)


def patch() -> None:
    """Install extensible ``Basic.free_symbols`` aggregation.

    Returns
    -------
    None

    Notes
    -----
    SymPy's default ``Basic.free_symbols`` combines child sets with
    ``set().union(...)``, which is fast for plain finite sets but erases set
    subclass behavior. This patch keeps that fast path for exact built-in sets
    and uses ``|`` dispatch only when a subclass participates.

    Examples
    --------
    Basic usage:

    >>> from math_toolkit.sympy_extensions import free_symbols
    >>> free_symbols.patch()
    >>> isinstance(free_symbols.UniversalFreeSymbols(), set)
    True
    """

    global _PATCHED
    if _PATCHED:
        return

    # Replace only the generic child aggregation. Classes with their own
    # ``free_symbols`` properties keep their specialized SymPy behavior.
    Basic.free_symbols = _basic_free_symbols
    _PATCHED = True
