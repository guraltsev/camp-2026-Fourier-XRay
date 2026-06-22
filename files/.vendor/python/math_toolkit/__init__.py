"""Provide notebook-oriented symbolic tools and runtime documentation bindings.

Importing :mod:`math_toolkit` registers bundled documentation topics and
installs the package profile selected below. Use
``math_toolkit.notebook.activate()`` to populate a notebook namespace, and use
``NamedFunction``, ``EqDef``, ``Hold``, ``Unhold``, ``Num``, ``IVP``, ``Solve``,
``VectorEquation2SystemOfEquations``, and
``SystemOfEquations2VectorEquation`` to author symbolic notation and cross
explicitly into numeric execution.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
import platform

__all__ = []
__notebook__ = []


@contextmanager
def _patched_platform_machine() -> Iterator[None]:
    """Temporarily replace ``platform.machine()`` with a non-WMI Windows fallback.

    On this Windows environment, SciPy's eager import path can reach NumPy test
    helpers that call ``platform.machine()`` while module top-level code is still
    running. Python's standard Windows implementation may answer that query
    through WMI, and the WMI call can block long enough to stall
    ``import math_toolkit``.

    This context manager narrows the workaround to import-time code that only
    needs a stable architecture label. It prefers the normal implementation on
    non-Windows platforms and restores ``platform.machine`` immediately after
    the guarded import completes.
    """

    # Non-Windows platforms do not use the problematic WMI-backed path.
    if os.name != "nt":
        yield
        return

    # Windows publishes the process architecture directly in the environment,
    # which is sufficient for SciPy's import-time platform checks.
    machine_name = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get(
        "PROCESSOR_ARCHITECTURE"
    )

    # Fall back to the standard library behavior when the environment does not
    # expose an architecture label we can trust.
    if not machine_name:
        yield
        return

    # Replace the query only for the guarded import so unrelated callers still
    # observe the ordinary standard-library function after package import.
    original_machine = platform.machine
    platform.machine = lambda: machine_name
    try:
        yield
    finally:
        platform.machine = original_machine


# Basic scientific stack
with _patched_platform_machine():
    import sympy
    import numpy
    import scipy

    # Publish the base scientific stack first so every later export can rely on
    # the same top-level aliases that notebook activation injects.
    __all__ += ["sympy", "numpy", "scipy"]
    __notebook__ += [("sp", sympy), ("np", numpy), ("sci", scipy),
                    ("sympy", sympy), ("numpy", numpy), ("scipy", scipy)]

    # Help system
    # Importing ``help`` registers the core documentation topics.
    # TODO: hide this behind activate
    from . import help
    from .help import Help

    __all__ += ["help", "Help"]
    __notebook__ += ["Help"]

    # Notebook convenience functions
    from . import notebook
    from .notebook import activate, reset

    __all__ += ["notebook", "activate", "reset"]
    __notebook__ += ["activate", "reset"]
    # PipeOps - piping syntax
    from . import pipeops
    from .pipeops import CLEAR, pipeop
    from .pipeops import common_pipeops as _common_pipeops

    __all__ += ["pipeops", "pipeop", "CLEAR"]
    __notebook__ += ["pipeop", "CLEAR"]

    # Publish the curated symbolic pipe operators from the dedicated submodule.
    for _pipeop_name in _common_pipeops.__all__:
        globals()[_pipeop_name] = getattr(_common_pipeops, _pipeop_name)
        __all__.append(_pipeop_name)
        __notebook__.append(_pipeop_name)

    from .md_print import md

    __all__ += ["md"]
    __notebook__ += ["md"]

    from .util import time_it

    __all__ += ["time_it"]
    __notebook__ += ["time_it"]

    from . import sympy_extensions

    __all__ += ["sympy_extensions"]

    from .sympy_extensions import EqDef

    __all__ += ["EqDef"]
    __notebook__ += ["EqDef"]

    from .sympy_extensions import (
        AppliedIndexedUndef,
        IndexedFunctionBase,
        IndexedFunction,
        IndexedFunctionHead,
        IndexedUndefinedFunction,
    )

    __all__ += [
        "AppliedIndexedUndef",
        "IndexedFunctionBase",
        "IndexedFunction",
        "IndexedFunctionHead",
        "IndexedUndefinedFunction",
    ]

    __notebook__ += [
        "AppliedIndexedUndef",
        "IndexedFunctionBase",
        "IndexedFunction",
        "IndexedFunctionHead",
        "IndexedUndefinedFunction",
    ]

    from .sympy_extensions import NamedFunction

    __all__ += ["NamedFunction"]
    __notebook__ += ["NamedFunction"]

    from .sympy_extensions import (
        EssentialSupremum,
        L1Norm,
        L2Norm,
        LinftyNorm,
        LpNorm,
    )

    __all__ += ["EssentialSupremum", "L1Norm", "L2Norm", "LinftyNorm", "LpNorm"]
    __notebook__ += [
        "EssentialSupremum",
        "L1Norm",
        "L2Norm",
        "LinftyNorm",
        "LpNorm",
    ]

    from .sympy_extensions import UniversalFreeSymbols

    __all__ += ["UniversalFreeSymbols"]
    __notebook__ += ["UniversalFreeSymbols"]

    # SymPy notation helpers
    from .sympy_extensions import (
        And,
        Eq,
        HeldExpression,
        Hold,
        If,
        Or,
        Otherwise,
        SystemOfEquations2VectorEquation,
        Unhold,
        UnholdAll,
        VectorEquation2SystemOfEquations,
    )

    __all__ += [
        "HeldExpression",
        "Hold",
        "Unhold",
        "UnholdAll",
        "If",
        "Eq",
        "And",
        "Or",
        "Otherwise",
        "SystemOfEquations2VectorEquation",
        "VectorEquation2SystemOfEquations",
    ]
    __notebook__ += [
        "HeldExpression",
        "Hold",
        "Unhold",
        "UnholdAll",
        "If",
        "Eq",
        "And",
        "Or",
        "Otherwise",
        "SystemOfEquations2VectorEquation",
        "VectorEquation2SystemOfEquations",
    ]

    # Modules imported in this package profile attach their own documentation.
    from . import num
    from .num import (
        compile,
        ImplementedFunction,
        LoweredCallable,
        Num,
        NumFunction,
    )
    __all__ += ["num", "compile", "ImplementedFunction", "LoweredCallable", "NumFunction", "Num"]
    __notebook__ += ["compile", "ImplementedFunction", "LoweredCallable", "NumFunction", "Num"]

    # Root import exposes the curated plotting API that notebook activation
    # reuses when it injects the package profile into a live namespace.
    from . import plotting
    # Plotting helpers
    from .plotting import (
        contour_plot,
        current_figure,
        get_session,
        domain_plot,
        figure,
        get_plot,
        info,
        list_plot,
        parametric_plot,
        plot,
        set_current_figure,
        temperature_plot,
    )

    __all__ += [
        "plotting",
        "contour_plot",
        "current_figure",
        "get_session",
        "domain_plot",
        "figure",
        "get_plot",
        "info",
        "list_plot",
        "parametric_plot",
        "plot",
        "set_current_figure",
        "temperature_plot",
    ]
    __notebook__ += [
        "contour_plot",
        "current_figure",
        "get_session",
        "domain_plot",
        "figure",
        "get_plot",
        "info",
        "list_plot",
        "parametric_plot",
        "plot",
        "set_current_figure",
        "temperature_plot",
    ]

    # Notebook activation also publishes the standard default symbol set.
    from .notebook.default_symbols import default_symbols

    __notebook__.append(default_symbols)
    del default_symbols
