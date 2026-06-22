"""Provide local helpers for the pure-trig Fourier matching game worksheet."""

from __future__ import annotations

import random

from math_toolkit import ImplementedFunction, Num
from sympy import Float, Symbol, pi, sin

__all__ = ["create_mystery_sine_function"]


def create_mystery_sine_function(*, modes):
    """Return a randomly generated pure-sine target function."""

    x = Symbol("x")
    mode_numbers = tuple(range(1, modes + 1))
    coefficient_choices = tuple(i / 10 for i in range(-10, 11))

    # Regenerate only the zero target, so the matching game always has a
    # visible signal.
    target_coefficients = tuple(
        random.choice(coefficient_choices) for _ in mode_numbers
    )
    while all(coefficient == 0 for coefficient in target_coefficients):
        target_coefficients = tuple(
            random.choice(coefficient_choices) for _ in mode_numbers
        )

    target_expr = sum(
        Float(coefficient) * sin(2 * pi * mode * x)
        for mode, coefficient in zip(mode_numbers, target_coefficients)
    )
    return ImplementedFunction("f", target_expr >> Num(var=x))
