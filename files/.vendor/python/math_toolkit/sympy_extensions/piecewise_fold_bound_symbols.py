"""Patch SymPy ``piecewise_fold`` to preserve binder-local conditions.

SymPy 1.14 folds ``Piecewise`` expressions out of ``Integral`` and ``Sum``
constructors even when a branch condition mentions a variable bound by that
constructor. This compatibility patch keeps those branch conditions inside the
binder while preserving ordinary folding for external conditions.
"""

from __future__ import annotations

__all__ = ["patch"]

from typing import Any


def patch() -> None:
    """Install a binder-aware ``piecewise_fold`` wrapper into SymPy."""

    import sympy
    import sympy.concrete.expr_with_limits as expr_with_limits
    import sympy.functions.elementary.piecewise as piecewise_module

    original = getattr(
        piecewise_module.piecewise_fold,
        "_math_toolkit_original_piecewise_fold",
        piecewise_module.piecewise_fold,
    )

    def binder_aware_piecewise_fold(expr: Any, evaluate: bool | None = True) -> Any:
        """Fold piecewise expressions without leaking bound variables."""

        if not isinstance(expr, sympy.Basic) or not expr.has(sympy.Piecewise):
            return expr

        # Mask branch conditionals whose conditions mention a symbol bound by
        # the expression being folded. Folding may still act on safe siblings.
        bound_symbols = set(getattr(expr, "bound_symbols", ()))
        replacements: dict[sympy.Basic, sympy.Dummy] = {}
        if bound_symbols:
            for piecewise in expr.atoms(sympy.Piecewise):
                if any(condition.free_symbols & bound_symbols for _, condition in piecewise.args):
                    replacements[piecewise] = sympy.Dummy()

        if replacements:
            masked = expr.xreplace(replacements)
            folded = original(masked, evaluate=evaluate)
            return folded.xreplace({dummy: piecewise for piecewise, dummy in replacements.items()})

        return original(expr, evaluate=evaluate)

    binder_aware_piecewise_fold._math_toolkit_original_piecewise_fold = original  # type: ignore[attr-defined]

    # SymPy exposes and imports ``piecewise_fold`` in several places. Patch the
    # direct constructor reference so ``Integral`` and ``Sum`` use the wrapper.
    piecewise_module.piecewise_fold = binder_aware_piecewise_fold
    sympy.piecewise_fold = binder_aware_piecewise_fold
    expr_with_limits.piecewise_fold = binder_aware_piecewise_fold
