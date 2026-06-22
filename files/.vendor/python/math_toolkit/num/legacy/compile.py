"""
# TODO write docstring
Compile symbolic expressions into numeric values or ``NumFunction`` objects."""

from __future__ import annotations

from typing import Any

import sympy

from .diagnostics import (
    NumDiagnostic,
    WarningPolicy,
    apply_warning_policy,
    NumCompilationError,
)
from .discovery import (
    discover_roles,
    lambdify_symbols_for_specs,
    merge_explicit_and_discovered_roles,
)
from .functions import (
    NumEvaluationContext,
    NumFunction,
    collect_implemented_functions,
    implemented_modules,
    make_call_cache,
)
from .shapes import build_output_evaluator
from .specs import NumArgSpec, normalize_arg_specs
from ...util import validate_option

def compile_num(
    expr: object,
    *,
    args: tuple[Any, ...] = (),
    warnings: str = "collect",
    cache: object = False,
) -> object:
    """Compile an expression as a numeric value or ``NumFunction``."""

    warning_policy = validate_option(warnings,  WarningPolicy)
    expression = sympy.sympify(expr)

    explicit_specs = normalize_arg_specs(args)
    implemented_functions = collect_implemented_functions(expression)

    discovered_roles = discover_roles(expression)
    final_specs = merge_explicit_and_discovered_roles(
        explicit_specs,
        discovered_roles,
    )
    diagnostics = _collect_diagnostics(expression)
    apply_warning_policy(diagnostics, warning_policy)

    # Evaluate concrete symbolic expressions directly before we build a SciPy
    # lambdify callable. This keeps ``expr >> Num`` cheap for scalar constants
    # and avoids unnecessary backend setup when no numeric arguments exist.
    if not final_specs and not implemented_functions:
        try:
            evaluated = expression.doit()
            if isinstance(evaluated, sympy.MatrixBase):
                return _concrete_matrix_value(evaluated)
            return _concrete_scalar_value(evaluated)
        except Exception:
            pass

    arg_symbols = lambdify_symbols_for_specs(final_specs, discovered_roles)
    context = NumEvaluationContext() if implemented_functions else None
    custom_impls = (
        implemented_modules(implemented_functions, context)
        if context is not None
        else None
    )
    raw, evaluator = build_output_evaluator(
        expression,
        arg_symbols,
        custom_impls=custom_impls,
    )

    if not final_specs:
        try:
            return evaluator(())
        except Exception as exc:
            raise NumCompilationError(
                f"Num could not evaluate {expression} as a numeric value."
    ) from exc

    execution_plan = _execution_plan_for(implemented_functions)
    performance_notes = _performance_notes_for(implemented_functions)

    return NumFunction(
        symbolic_origin=expression,
        args=final_specs,
        backend="scipy/numpy",
        raw=raw,
        warnings=diagnostics,
        _evaluator=evaluator,
        execution_plan=execution_plan,
        vectorized=True,
        performance_notes=performance_notes,
        _context=context,
        _call_cache=make_call_cache(cache),
        _implemented_functions=implemented_functions,
    )



def _collect_diagnostics(expr: object) -> tuple[NumDiagnostic, ...]:
    """Return compile-time diagnostics for valid but notable expressions."""

    diagnostics: list[NumDiagnostic] = []
    if isinstance(expr, sympy.MatrixBase):
        diagnostics.append(
            NumDiagnostic(
                code="matrix-output-entrywise",
                message=(
                    "Matrix output is compiled entrywise so sample axes appear "
                    "before mathematical output axes."
                ),
                expr=expr,
            )
        )

    for part in _diagnostic_parts(expr):
        if isinstance(part, sympy.Sum) and part.has(sympy.Indexed):
            diagnostics.append(
                NumDiagnostic(
                    code="indexed-sum-python-loop",
                    message=(
                        "Finite indexed Sum compiled through SymPy's generated "
                        "loop; vectorized lowering is not implemented yet."
                    ),
                    expr=part,
                )
            )
    return tuple(diagnostics)


def _execution_plan_for(implemented_functions: tuple[object, ...]) -> str:
    """Return a public execution-plan label for a compiled expression."""

    if not implemented_functions:
        return "direct"
    if any(component.required_parameters for component in implemented_functions):
        return "solver-backed-lazy"
    return "solver-backed-ready"


def _performance_notes_for(implemented_functions: tuple[object, ...]) -> tuple[str, ...]:
    """Return performance notes for solver-backed numeric functions."""

    if not implemented_functions:
        return ()

    parameters: set[sympy.Symbol] = set()
    for component in implemented_functions:
        parameters.update(component.required_parameters)
    if not parameters:
        return (
            "Uses an implemented symbolic solution and evaluates it through "
            "Num over the independent variable.",
        )

    joined = ", ".join(symbol.name for symbol in sorted(parameters, key=lambda item: item.name))
    return (
        "Solves lazily when concrete parameter values are supplied.",
        f"Varying parameter(s) {joined} creates distinct solver runs.",
        "Solver results are cached by concrete parameter tuple.",
    )


def _diagnostic_parts(expr: object) -> tuple[object, ...]:
    """Return expression nodes inspected for diagnostics."""

    roots: tuple[object, ...]
    if isinstance(expr, sympy.MatrixBase):
        roots = tuple(expr[row, col] for row in range(expr.rows) for col in range(expr.cols))
    else:
        roots = (expr,)

    parts: list[object] = []
    for root in roots:
        if isinstance(root, sympy.Basic):
            parts.extend(sympy.preorder_traversal(root))
    return tuple(parts)


def _concrete_scalar_value(expr: object) -> object:
    """Return a Python or NumPy scalar for a concrete symbolic expression."""

    evaluated = sympy.N(expr)
    if isinstance(evaluated, sympy.Basic) and evaluated.free_symbols:
        raise ValueError("Concrete scalar evaluation requires no free symbols.")
    return complex(evaluated) if getattr(evaluated, "is_complex", False) and not getattr(evaluated, "is_real", False) else float(evaluated)


def _concrete_matrix_value(expr: sympy.MatrixBase) -> object:
    """Return a NumPy array for a concrete symbolic matrix expression."""

    import numpy as np

    rows, cols = expr.shape
    values = [
        _concrete_scalar_value(expr[row, col])
        for row in range(rows)
        for col in range(cols)
    ]
    array = np.array(values)
    if rows == 1:
        return array.reshape(cols)
    if cols == 1:
        return array.reshape(rows)
    return array.reshape(rows, cols)
