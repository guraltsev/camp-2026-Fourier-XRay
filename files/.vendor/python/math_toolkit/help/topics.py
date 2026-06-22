"""Attach declared notebook documentation topics to runtime object types.

Importing this module registers ``_mt_help`` metadata on SymPy and toolkit
classes so ``math_toolkit.help.help(...)`` can render topic links without a
runtime registry or filesystem lookup.
"""

from __future__ import annotations

from pathlib import PurePosixPath
import sys

import sympy
from sympy.core.function import FunctionClass

# Documentation topics are assigned directly to the runtime types they describe.
sys.modules["math_toolkit"]._mt_help = {
    "path": PurePosixPath("index"),
    "anchor": None,
    "label": "math_toolkit documentation",
}
sympy.Symbol._mt_help = {
    "path": PurePosixPath("library/Symbol"),
    "anchor": None,
    "label": "Symbol",
}
sympy.Expr._mt_help = {
    "path": PurePosixPath("library/Expression"),
    "anchor": None,
    "label": "Expression",
}
sympy.Basic.rewrite._mt_help = {
    "path": PurePosixPath("library/rewrite"),
    "anchor": None,
    "label": "rewrite",
}
sympy.Indexed._mt_help = {
    "path": PurePosixPath("library/Indexed"),
    "anchor": None,
    "label": "Indexed",
}
FunctionClass._mt_help = {
    "path": PurePosixPath("library/Function"),
    "anchor": None,
    "label": "Function",
}
