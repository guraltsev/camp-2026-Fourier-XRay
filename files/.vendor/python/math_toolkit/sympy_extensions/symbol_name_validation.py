"""Validate atomic names for SymPy symbols and undefined functions."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import sympy
from sympy.core.function import AppliedUndef, Function, UndefinedFunction

__all__ = ["validate_name", "patch"]

# Atomic names stay plain so structural notation remains explicit.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Patch state keeps package-profile imports idempotent. Captured constructors
# remain the delegation targets for the patched constructor wrappers.
_PATCHED = False
_ORIGINAL_SYMBOL_NEW = sympy.Symbol.__new__
_ORIGINAL_FUNCTION_NEW = sympy.Function.__new__
_ORIGINAL_UNDEFINEDFUNCTION_NEW = UndefinedFunction.__new__

# SymPy creates implementation-only symbols with display names that are not
# valid toolkit atomic identifiers, such as limit directions and unit labels.
_SYMPY_PACKAGE_ROOT = Path(sympy.__file__).resolve().parent


def validate_name(name: str) -> str:
    """Validate that a SymPy name is a plain atomic identifier.

    Parameters
    ----------
    name : str
        Candidate symbol or undefined-function name.

    Returns
    -------
    str
        The validated name text.

    Raises
    ------
    ValueError
        If ``name`` is not a plain identifier using letters, digits, and
        underscores, with a leading letter.

    Examples
    --------
    Basic usage:

    >>> from math_toolkit.sympy_extensions import symbol_name_validation
    >>> symbol_name_validation.validate_name("Force")
    'Force'
    >>> symbol_name_validation.validate_name("x_1")
    'x_1'

    Edge cases:

    >>> symbol_name_validation.validate_name("Force[t]")
    Traceback (most recent call last):
    ...
    ValueError: Invalid atomic name 'Force[t]'. Use names that start with a letter and contain only letters, digits, and underscores. Represent indexing structurally with [] instead of encoding it into the name.
    """

    text = str(name)
    if not _NAME_RE.fullmatch(text):
        raise ValueError(
            f"Invalid atomic name {text!r}. Use names that start with a letter "
            "and contain only letters, digits, and underscores. Represent "
            "indexing structurally with [] instead of encoding it into the name."
        )
    return text


def _called_from_sympy_internal() -> bool:
    """Return whether the patched constructor was invoked by SymPy internals."""

    frame = inspect.currentframe()
    caller_frame = None
    try:
        if frame is None or frame.f_back is None:
            return False
        caller_frame = frame.f_back.f_back
        if caller_frame is None:
            return False
        filename = caller_frame.f_code.co_filename
        try:
            return Path(filename).resolve().is_relative_to(_SYMPY_PACKAGE_ROOT)
        except OSError:
            return False
    finally:
        del caller_frame
        del frame



def _validated_symbol_new(
    cls: type[sympy.Symbol],
    name: Any,
    **assumptions: Any,
) -> sympy.Symbol:
    """Validate string names before delegating to SymPy's Symbol constructor."""

    if isinstance(name, str) and not _called_from_sympy_internal():
        validate_name(name)
    return _ORIGINAL_SYMBOL_NEW(cls, name, **assumptions)



def _validated_function_new(cls: type[Function], *args: Any, **options: Any):
    """Validate undefined-function names before SymPy creates the head."""

    if cls is Function and args and not _called_from_sympy_internal():
        name = args[0]
        if isinstance(name, sympy.Symbol):
            validate_name(name.name)
        elif isinstance(name, str):
            validate_name(name)
    return _ORIGINAL_FUNCTION_NEW(cls, *args, **options)



def _validated_undefinedfunction_new(
    mcl: type[UndefinedFunction],
    name: Any,
    bases: tuple[type[AppliedUndef], ...] = (AppliedUndef,),
    __dict__: dict[str, Any] | None = None,
    **kwargs: Any,
) -> type[AppliedUndef]:
    """Validate class-level undefined-function names created by SymPy."""

    if _called_from_sympy_internal():
        return _ORIGINAL_UNDEFINEDFUNCTION_NEW(
            mcl,
            name,
            bases=bases,
            __dict__=__dict__,
            **kwargs,
        )

    if isinstance(name, sympy.Symbol):
        validate_name(name.name)
    elif isinstance(name, str):
        validate_name(name)
    return _ORIGINAL_UNDEFINEDFUNCTION_NEW(mcl, name, bases=bases, __dict__=__dict__, **kwargs)



def patch() -> None:
    """Install atomic-name validation for SymPy constructors.

    Returns
    -------
    None

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> from math_toolkit.sympy_extensions import symbol_name_validation
    >>> symbol_name_validation.patch()
    >>> str(sympy.Symbol("x"))
    'x'
    >>> str(sympy.Symbol("x_1"))
    'x_1'

    Edge cases:

    >>> sympy.Symbol("Force[t]")
    Traceback (most recent call last):
    ...
    ValueError: Invalid atomic name 'Force[t]'. Use names that start with a letter and contain only letters, digits, and underscores. Represent indexing structurally with [] instead of encoding it into the name.
    """

    global _PATCHED
    if _PATCHED:
        return

    # SymPy lazily creates a few internal unit abbreviations with display names
    # that are not user-facing atomic identifiers. Let SymPy build that table
    # under its original constructors before user validation takes ownership of
    # new Symbol and Function calls.
    importlib.import_module("sympy.physics.units")

    # Replace the constructors together so symbols and function heads share the
    # same atomic-name contract.
    sympy.Symbol.__new__ = _validated_symbol_new
    sympy.Function.__new__ = _validated_function_new
    UndefinedFunction.__new__ = _validated_undefinedfunction_new
    _PATCHED = True
