"""Dispatch symbolic numerical problems to their declared solvers."""

from __future__ import annotations

from .diagnostics import NumArgumentError


def Solve(problem: object, **kwargs: object) -> object:
    """Dispatch a symbolic problem object to its private solver.

    Parameters
    ----------
    problem : object
        Symbolic problem object with a private ``_solver`` callable.
    **kwargs : object
        Solver-specific options forwarded without interpretation.

    Returns
    -------
    object
        Solver-specific Python result container.

    Raises
    ------
    NumArgumentError
        If ``problem`` does not declare a solver.

    Examples
    --------
    Basic usage:

    >>> from math_toolkit import IVP, Solve
    >>> import sympy
    >>> t, y = sympy.symbols("t y")
    >>> solution = Solve(IVP([y], unknowns=[y], domain=(t, 0, 1), initial_data={y: 1}))
    >>> len(solution)
    1
    """

    solver = getattr(problem, "_solver", None)
    if solver is None:
        raise NumArgumentError(
            "Solve expects a symbolic problem object with a declared solver."
        )
    return solver(problem, **kwargs)
