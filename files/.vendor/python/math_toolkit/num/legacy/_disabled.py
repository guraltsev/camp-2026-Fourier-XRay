"""Expose the public ``num`` surface as disabled rewrite placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


_NUM_DISABLED_MESSAGE = (
    "Num is intentionally not implemented during the rewrite."
)


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
    """Report symbolic forms with no numeric execution semantics."""


class NumNotImplementedError(NumError, NotImplementedError):
    """Report that the public ``num`` surface is disabled during the rewrite."""


@dataclass(frozen=True)
class NumDiagnostic:
    """Describe a non-fatal numerical compilation diagnostic."""

    code: str
    message: str
    expr: object


class NumCompileWarning(UserWarning):
    """Warning category reserved for future numeric compilation warnings."""


def _raise_num_not_implemented() -> None:
    """Raise the shared transition error for disabled numeric features."""

    raise NumNotImplementedError(_NUM_DISABLED_MESSAGE)


class _DisabledNumOperator:
    """Represent the disabled public ``Num`` boundary during the rewrite."""

    _mt_help = {
        "path": PurePosixPath("library/Num"),
        "anchor": None,
        "label": "Num",
    }

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Reject ``Num(...)`` while the rewrite surface is disabled."""

        _raise_num_not_implemented()

    def __rrshift__(self, expr: object) -> object:
        """Reject ``expr >> Num`` while the rewrite surface is disabled."""

        _raise_num_not_implemented()

    def __repr__(self) -> str:
        """Return an interactive representation of the disabled boundary."""

        return (
            "<Num disabled: the public numeric boundary is being rewritten>"
        )

    @property
    def Scalar(self) -> object:
        """Reject access to scalar role helpers while disabled."""

        _raise_num_not_implemented()

    @property
    def Index(self) -> object:
        """Reject access to index role helpers while disabled."""

        _raise_num_not_implemented()

    def Array(self, rank: int, element_type: object | None = None) -> object:
        """Reject access to array role helpers while disabled."""

        _raise_num_not_implemented()

    def Integrate(self, *args: object, **kwargs: object) -> object:
        """Reject access to integration helpers while disabled."""

        _raise_num_not_implemented()


class NumFunction:
    """Placeholder for the disabled numeric callable surface."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject numeric callable construction while disabled."""

        _raise_num_not_implemented()


class ImplementedFunction:
    """Placeholder for the disabled implemented-function surface."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject implemented-function construction while disabled."""

        _raise_num_not_implemented()


class IVP:
    """Placeholder for the disabled IVP surface."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject IVP construction while disabled."""

        _raise_num_not_implemented()


def Solve(problem: object, **kwargs: object) -> object:
    """Reject solver dispatch while disabled."""

    _raise_num_not_implemented()


def SolveIVP(*args: object, **kwargs: object) -> object:
    """Reject IVP solving while disabled."""

    _raise_num_not_implemented()


Num = _DisabledNumOperator()


__all__ = [
    "IVP",
    "ImplementedFunction",
    "Num",
    "NumFunction",
    "Solve",
    "SolveIVP",
    "NumArgumentError",
    "NumCompilationError",
    "NumCompileWarning",
    "NumDiagnostic",
    "NumError",
    "NumHeldExpressionError",
    "NumNotImplementedError",
    "NumRoleError",
    "NumSolverError",
    "NumUnsupportedExpressionError",
]
