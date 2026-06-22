"""Enable structural symbol indexing and explicit symbol intent getters."""

from __future__ import annotations

import re
from numbers import Integral
from typing import Any

import sympy

__all__ = ["patch"]

# String indices must be atomic names; structural indices should be real SymPy
# objects.
_ATOMIC_INDEX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Patch state keeps package-profile imports idempotent.
_PATCHED = False
_ORIGINAL_INDEXEDBASE_GETITEM = sympy.IndexedBase.__getitem__

# Keep implicit x[i] semantics separate from the explicit x.indexedbase intent
# getter.
_SYMBOL_INDEXED_BASE_CACHE: dict[sympy.Symbol, sympy.IndexedBase] = {}
_SYMBOL_NAME_INDEXED_BASE_CACHE: dict[str, sympy.IndexedBase] = {}


def _symbol_symbol(self: sympy.Symbol) -> sympy.Symbol:
    """Return the receiving symbol for the explicit atomic-symbol view."""

    return self


def _symbol_indexedbase(self: sympy.Symbol) -> sympy.IndexedBase:
    """Return the cached indexed-base view for a symbol name."""

    base = _SYMBOL_NAME_INDEXED_BASE_CACHE.get(self.name)
    if base is None:
        base = sympy.IndexedBase(self.name)
        _SYMBOL_NAME_INDEXED_BASE_CACHE[self.name] = base
    return base


def _indexedbase_getitem(
    self: sympy.IndexedBase,
    indices: Any,
    **kw_args: Any,
) -> sympy.Indexed:
    """Treat a single Symbol index as scalar after Symbol gains __getitem__."""

    if isinstance(indices, sympy.Symbol):
        return _ORIGINAL_INDEXEDBASE_GETITEM(self, (indices,), **kw_args)
    return _ORIGINAL_INDEXEDBASE_GETITEM(self, indices, **kw_args)


def _symbol_getitem(self: sympy.Symbol, key: Any) -> sympy.Indexed:
    """Build a structural Indexed object from a symbol and validated indices."""

    # Validate and convert each index in order.
    values = key if isinstance(key, tuple) else (key,)
    indices: list[sympy.Basic] = []
    for value in values:
        if isinstance(value, slice):
            raise ValueError("Slices are not supported as symbolic indices.")
        if isinstance(value, sympy.Basic):
            indices.append(value)
            continue
        if isinstance(value, Integral) and not isinstance(value, bool):
            indices.append(sympy.Integer(int(value)))
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text or not _ATOMIC_INDEX_RE.fullmatch(text):
                raise ValueError(
                    f"Invalid index string {value!r}. "
                    "Use plain atomic names like 'i', or pass a structural "
                    "SymPy object."
                )
            indices.append(sympy.Symbol(text))
            continue
        indices.append(sympy.sympify(value))

    # Cache the implicit indexed base per Symbol object so repeated indexing
    # preserves ordinary SymPy equality and display behavior.
    base = _SYMBOL_INDEXED_BASE_CACHE.get(self)
    if base is None:
        base = sympy.IndexedBase(self)
        _SYMBOL_INDEXED_BASE_CACHE[self] = base
    return base[tuple(indices)]


def patch() -> None:
    """Install structural indexing and Symbol intent getters.

    Returns
    -------
    None

    Notes
    -----
    ``Symbol.symbol`` returns the symbol unchanged. ``Symbol.indexedbase``
    returns an ``IndexedBase`` built from the symbol name, making the intent to
    treat an atomic name as an indexed base explicit.

    Adding ``Symbol.__getitem__`` makes SymPy symbols look sequence-like to
    SymPy's ordinary ``IndexedBase.__getitem__`` implementation. This patch
    also treats a single ``Symbol`` index as scalar there, so explicit
    ``x.indexedbase[i]`` notation stays finite and structural.

    Patched symbols declare ``_iterable = False`` because the indexing syntax is
    structural notation, not a finite sequence for recursive SymPy algorithms.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions import symbol_indexing
    >>> symbol_indexing.patch()
    >>> x = sympy.Symbol("x")
    >>> i = sympy.Symbol("i")
    >>> str(x[i])
    'x[i]'
    >>> x.symbol is x
    True
    >>> str(x.indexedbase[i])
    'x[i]'
    """

    global _PATCHED
    if _PATCHED:
        return

    # Install the symbol-facing notation and keep explicit IndexedBase indexing
    # compatible after Symbol becomes subscriptable.
    sympy.Symbol.__getitem__ = _symbol_getitem
    sympy.Symbol.symbol = property(
        _symbol_symbol,
        doc="Return this symbol unchanged.",
    )
    sympy.Symbol.indexedbase = property(
        _symbol_indexedbase,
        doc="Return an IndexedBase built from this symbol's name.",
    )
    sympy.Symbol._iterable = False
    sympy.IndexedBase.__getitem__ = _indexedbase_getitem
    _PATCHED = True
