"""Expose SymPy extension submodules for notebook-oriented symbolic work.

The root :mod:`math_toolkit` package profile imports selected extension
modules and calls their ``patch()`` functions. Import submodules explicitly
when direct access to helper classes or functions is needed.
"""

from __future__ import annotations

__all__ = []


# SymPy compatibility patches
from . import piecewise_fold_bound_symbols

piecewise_fold_bound_symbols.patch()

__all__ += ["piecewise_fold_bound_symbols"]


# Indexed-function syntax
from . import function_indexing
from .function_indexing import (
    AppliedIndexedUndef,
    IndexedFunction,
    IndexedFunctionBase,
    IndexedFunctionHead,
    IndexedUndefinedFunction,
)

__all__ += [
    "function_indexing",
    "AppliedIndexedUndef",
    "IndexedFunction",
    "IndexedFunctionBase",
    "IndexedFunctionHead",
    "IndexedUndefinedFunction",
]


# Named symbolic definitions
from . import eqdef
from .eqdef import EqDef

__all__ += ["eqdef", "EqDef"]

from . import named_functions
from .named_functions import NamedFunction

__all__ += ["named_functions", "NamedFunction"]


# Symbolic weighted Lebesgue norms
from . import lp_norms
from .lp_norms import EssentialSupremum, L1Norm, L2Norm, LinftyNorm, LpNorm

__all__ += [
    "lp_norms",
    "EssentialSupremum",
    "L1Norm",
    "L2Norm",
    "LinftyNorm",
    "LpNorm",
]


# Held expressions and conditional notation
from . import holding
from .holding import HeldExpression, Hold, Unhold, UnholdAll

__all__ += ["holding", "HeldExpression", "Hold", "Unhold", "UnholdAll"]

from . import sympy_ifdsl
from .sympy_ifdsl import And, Eq, If, Or, Otherwise

__all__ += ["sympy_ifdsl", "And", "Eq", "If", "Or", "Otherwise"]


# Structural vector-equation adapters
from . import vector_equations
from .vector_equations import (
    SystemOfEquations2VectorEquation,
    VectorEquation2SystemOfEquations,
)

__all__ += [
    "vector_equations",
    "SystemOfEquations2VectorEquation",
    "VectorEquation2SystemOfEquations",
]


# Support modules that callers may patch explicitly.
from . import free_symbols
from .free_symbols import UniversalFreeSymbols

__all__ += ["free_symbols", "UniversalFreeSymbols"]
