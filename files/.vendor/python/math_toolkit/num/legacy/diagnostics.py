"""Define diagnostics and warning policies for ``Num`` compilation."""

from __future__ import annotations

from dataclasses import dataclass
import warnings as python_warnings
from enum import StrEnum

from ...util import validate_option


class NumError(Exception):
    """Base class for numerical-layer errors."""


class NumRoleError(NumError):
    """Report incompatible symbolic roles during numericalization."""


class NumArgumentError(NumError):
    """Report invalid ``Num`` argument specifications or call arguments."""


class NumCompilationError(NumError):
    """Report expressions that cannot be compiled for numeric execution."""


class NumSolverError(NumError):
    """Report numerical solver failures through toolkit language."""


class NumHeldExpressionError(NumCompilationError):
    """Report held notation passed across the numerical boundary."""


class NumUnsupportedExpressionError(NumCompilationError):
    """Report symbolic forms with no MVP numeric execution semantics."""


@dataclass(frozen=True)
class NumDiagnostic:
    """Describe a non-fatal numerical compilation diagnostic.

    Parameters
    ----------
    code : str
        Stable machine-readable diagnostic identifier.
    message : str
        Human-readable explanation.
    expr : object
        Symbolic expression or subexpression associated with the diagnostic.
    """

    code: str
    message: str
    expr: object


class NumCompileWarning(UserWarning):
    """Warning category emitted by ``Num(warnings="show")``."""

# Policy for dealing with warnings when compiling functions


class WarningPolicy(StrEnum):
    COLLECT = "collect"
    SHOW = "show"
    ERROR="error"



def validate_warning_policy(policy: str) -> WarningPolicy:
    """Return a supported warning policy or raise a clear error."""
    return validate_option(policy, WarningPolicy)


def apply_warning_policy(
    diagnostics: tuple[NumDiagnostic, ...],
    policy: WarningPolicy,
) -> None:
    """Apply the requested compile-time diagnostic policy."""

    if not diagnostics or policy == "collect":
        return

    if policy == "show":
        for diagnostic in diagnostics:
            python_warnings.warn(
                diagnostic.message,
                NumCompileWarning,
                stacklevel=3,
            )
        return

    joined = "; ".join(
        f"{diagnostic.code}: {diagnostic.message}"
        for diagnostic in diagnostics
    )
    error = NumCompilationError(f"Num compilation produced diagnostics: {joined}")
    error.diagnostics = diagnostics
    raise error
