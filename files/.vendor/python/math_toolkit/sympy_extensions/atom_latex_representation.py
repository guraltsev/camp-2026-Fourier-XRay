"""Render plain atomic SymPy symbol names as readable LaTeX."""

from __future__ import annotations

import re
from typing import Any

import sympy

__all__ = ["atom_name_to_latex", "patch"]

# Simple atom names render either directly or through ``\mathrm``.
_SIMPLE_ATOM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")

# Identifiers may contain inert underscores that become presentation subscripts.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Greek names use canonical LaTeX command forms when SymPy stores them as atoms.
_GREEK_NAME_TO_LATEX: dict[str, str] = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "varepsilon": r"\varepsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "vartheta": r"\vartheta",
    "iota": r"\iota",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "omicron": "o",
    "pi": r"\pi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "upsilon": r"\upsilon",
    "phi": r"\phi",
    "varphi": r"\varphi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
    "Gamma": r"\Gamma",
    "Delta": r"\Delta",
    "Theta": r"\Theta",
    "Lambda": r"\Lambda",
    "Xi": r"\Xi",
    "Pi": r"\Pi",
    "Sigma": r"\Sigma",
    "Upsilon": r"\Upsilon",
    "Phi": r"\Phi",
    "Psi": r"\Psi",
    "Omega": r"\Omega",
}

# Patch state keeps package-profile imports idempotent. The captured renderer
# remains the fallback for names this module does not own.
_PATCHED = False
_ORIGINAL_SYMBOL_LATEX = getattr(sympy.Symbol, "_latex", None)



def _escape_identifier_subscript(text: str) -> str:
    """Return a literal subscript body for identifier text after an underscore."""

    return text.replace("_", r"\_")



def atom_name_to_latex(name: str) -> str:
    """Render one plain atomic symbol name as a LaTeX fragment.

    Parameters
    ----------
    name : str
        Atomic symbol name to render.

    Returns
    -------
    str
        LaTeX text for ``name`` using direct Greek forms, ``\\mathrm`` for
        multi-letter atomic names, and grouped presentation subscripts for
        inert underscores.

    Examples
    --------
    Basic usage:

    >>> from math_toolkit.sympy_extensions import atom_latex_representation
    >>> atom_latex_representation.atom_name_to_latex("theta")
    '\\\\theta'
    >>> atom_latex_representation.atom_name_to_latex("speed")
    '\\\\mathrm{speed}'
    >>> atom_latex_representation.atom_name_to_latex("x_2")
    'x_{2}'
    """

    text = str(name)

    # Treat underscores as inert presentation subscripts only for valid atomic
    # identifiers; structural indexing is represented elsewhere.
    if "_" in text and _IDENTIFIER_RE.fullmatch(text):
        base, subscript = text.split("_", 1)
        return f"{atom_name_to_latex(base)}_{{{_escape_identifier_subscript(subscript)}}}"

    # Prefer named mathematical glyphs, then plain one-letter symbols, then a
    # roman word form for multi-letter atomic identifiers.
    if text in _GREEK_NAME_TO_LATEX:
        return _GREEK_NAME_TO_LATEX[text]
    if len(text) == 1 and text.isalpha():
        return text
    if _SIMPLE_ATOM_RE.fullmatch(text):
        return rf"\mathrm{{{text}}}"
    return text



def _symbol_latex(self: sympy.Symbol, printer: Any) -> str:
    """Render owned atomic symbol names and delegate all other names to SymPy."""

    name = self.name

    # Use custom rendering only for atomic names this patch owns.
    if (
        name in _GREEK_NAME_TO_LATEX
        or (len(name) == 1 and name.isalpha())
        or bool(_IDENTIFIER_RE.fullmatch(name))
    ):
        return atom_name_to_latex(name)
    if _ORIGINAL_SYMBOL_LATEX is not None:
        return _ORIGINAL_SYMBOL_LATEX(self, printer)
    custom_name = printer._settings["symbol_names"].get(self)
    if custom_name is not None:
        return custom_name
    return printer._deal_with_super_sub(name, style="plain")



def patch() -> None:
    """Install atom-level Symbol LaTeX rendering.

    Returns
    -------
    None

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions import atom_latex_representation
    >>> atom_latex_representation.patch()
    >>> sympy.latex(sympy.Symbol("speed"))
    '\\\\mathrm{speed}'
    >>> sympy.latex(sympy.Symbol("x_2"))
    'x_{2}'
    """

    global _PATCHED
    if _PATCHED:
        return

    # Replace only Symbol rendering; indexed and applied objects keep their own
    # printer hooks.
    sympy.Symbol._latex = _symbol_latex
    _PATCHED = True
