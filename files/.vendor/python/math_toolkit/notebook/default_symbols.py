"""Construct the default symbolic names injected into notebook namespaces."""

from __future__ import annotations

__all__ = ["default_symbols"]


def default_symbols() -> list[tuple[str, object]]:
    """Return the default symbolic bindings used by notebook activation."""

    import sympy

    bindings: list[tuple[str, object]] = []

    # Common symbols are intentionally ordinary atomic names; structural
    # indexing remains explicit through ``[]`` after activation.
    for name in (
        "a",
        "b",
        "c",
        "d",
        "e",
        "l",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "A",
        "B",
        "C",
        "D",
        "J",
        "K",
        "L",
        "M",
        "N",
        "P",
        "R",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ):
        bindings.append((name, sympy.Symbol(name)))

    # Notebook index variables default to integer assumptions because they are
    # most often used as discrete indices.
    for name in ("i", "j", "k", "m", "n"):
        bindings.append((name, sympy.Symbol(name, integer=True)))

    # Function heads mirror the compact single-letter symbols used in examples.
    for name in ("f", "g", "h", "F", "G", "H"):
        bindings.append((name, sympy.Function(name)))

    # Python keywords get safe binding names while the underlying SymPy atom
    # keeps the mathematical name.
    for binding_name, atomic_name in (
        ("alpha", "alpha"),
        ("delta", "delta"),
        ("epsilon", "epsilon"),
        ("varepsilon", "varepsilon"),
        ("eta", "eta"),
        ("theta", "theta"),
        ("vartheta", "vartheta"),
        ("iota", "iota"),
        ("kappa", "kappa"),
        ("lambda_", "lambda"),
        ("mu", "mu"),
        ("nu", "nu"),
        ("xi", "xi"),
        ("omicron", "omicron"),
        ("rho", "rho"),
        ("sigma", "sigma"),
        ("tau", "tau"),
        ("upsilon", "upsilon"),
        ("phi", "phi"),
        ("varphi", "varphi"),
        ("chi", "chi"),
        ("psi", "psi"),
        ("omega", "omega"),
    ):
        bindings.append((binding_name, sympy.Symbol(atomic_name)))

    return bindings
