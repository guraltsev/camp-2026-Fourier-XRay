"""
# TODO write docstring
Build SciPy/NumPy numerical callables for symbolic expressions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy


def build_backend_callable(
    arg_symbols: tuple[object, ...],
    expr: object,
    *,
    custom_impls: dict[str, object] | None = None,
) -> Callable[..., Any]:
    """Return a callable using the default SciPy/NumPy backend chain."""

    if custom_impls is None:
        custom_impls = {}
    return sympy.lambdify(
        arg_symbols,
        expr,
        modules=[custom_impls, "scipy", "numpy"],
    )
