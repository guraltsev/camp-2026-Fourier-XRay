"""Represent numeric argument roles accepted by ``Num``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class ScalarType:
    """Represent a floating-point or array-broadcast scalar argument."""

    def __repr__(self) -> str:
        """Return the public constructor-style role name."""

        return "Num.Scalar"


@dataclass(frozen=True)
class IndexType:
    """Represent an integer-valued index argument."""

    def __repr__(self) -> str:
        """Return the public constructor-style role name."""

        return "Num.Index"


NumericElementType: TypeAlias = ScalarType | IndexType


@dataclass(frozen=True)
class ArrayType:
    """Represent a numeric array-like argument.

    Parameters
    ----------
    rank : int
        Positive number of indices required to access one array element.
    element_type : ScalarType | IndexType
        Role of the elements stored in the array.
    """

    rank: int
    element_type: NumericElementType

    def __post_init__(self) -> None:
        """Validate public array role construction."""

        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise TypeError("Num.Array(rank) requires a positive integer rank.")
        if self.rank < 1:
            raise ValueError("Num.Array(rank) requires a positive integer rank.")
        if not isinstance(self.element_type, ScalarType | IndexType):
            raise TypeError(
                "Num.Array(rank, element_type) supports Num.Scalar or "
                "Num.Index elements in the MVP."
            )

    def __repr__(self) -> str:
        """Return the public constructor-style role name."""

        if isinstance(self.element_type, ScalarType):
            return f"Num.Array({self.rank})"
        return f"Num.Array({self.rank}, {self.element_type!r})"


SCALAR = ScalarType()
INDEX = IndexType()


def is_numeric_arg_type(value: object) -> bool:
    """Return whether ``value`` is one of the supported public role objects."""

    return isinstance(value, ScalarType | IndexType | ArrayType)
