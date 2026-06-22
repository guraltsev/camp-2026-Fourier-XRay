"""Evaluate ``Num`` outputs with sample axes before mathematical axes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sympy

from .backend import build_backend_callable


def build_output_evaluator(
    expr: object,
    arg_symbols: tuple[object, ...],
    *,
    custom_impls: dict[str, object] | None = None,
) -> tuple[object, Callable[[tuple[object, ...]], object]]:
    """Return raw callables and a shape-normalizing evaluator."""

    if isinstance(expr, sympy.MatrixBase):
        return _build_matrix_output_evaluator(
            expr,
            arg_symbols,
            custom_impls=custom_impls,
        )

    raw = build_backend_callable(arg_symbols, expr, custom_impls=custom_impls)

    def evaluate(values: tuple[object, ...]) -> object:
        return raw(*values)

    return raw, evaluate


def _build_matrix_output_evaluator(
    expr: sympy.MatrixBase,
    arg_symbols: tuple[object, ...],
    *,
    custom_impls: dict[str, object] | None,
) -> tuple[object, Callable[[tuple[object, ...]], object]]:
    """Return entrywise callables for sample-first matrix output."""

    rows, cols = expr.shape
    entries = tuple(expr[row, col] for row in range(rows) for col in range(cols))
    raw = tuple(
        build_backend_callable(arg_symbols, entry, custom_impls=custom_impls)
        for entry in entries
    )

    if rows == 1:
        math_shape = (cols,)
    elif cols == 1:
        math_shape = (rows,)
    else:
        math_shape = (rows, cols)

    def evaluate(values: tuple[object, ...]) -> object:
        evaluated = [np.asarray(callable_entry(*values)) for callable_entry in raw]
        if not evaluated:
            return np.empty(math_shape)

        # Broadcast every entry over the same sample axes, then append the
        # mathematical vector or matrix axes at the end.
        broadcasted = np.broadcast_arrays(*evaluated)
        sample_shape = broadcasted[0].shape
        stacked = np.stack(broadcasted, axis=-1)
        return stacked.reshape(sample_shape + math_shape)

    return raw, evaluate
