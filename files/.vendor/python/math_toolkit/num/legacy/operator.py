"""Expose the public ``Num`` pipeline operator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .compile import compile_num
from .diagnostics import validate_warning_policy
from .integration import integrate
from .types import INDEX, SCALAR, ArrayType, IndexType, ScalarType


@dataclass(frozen=True)
class NumStage:
    """Represent a configured ``Num(...)`` pipeline stage."""

    args: tuple[Any, ...]
    warnings: object
    cache: object

    def __post_init__(self) -> None:
        """Validate options when the stage is created."""

        validate_warning_policy(self.warnings)

    def __rrshift__(self, expr: object) -> object:
        """Compile the expression on the left of ``>>``."""

        return compile_num(
            expr,
            args=self.args,
            warnings=self.warnings,
            cache=self.cache,
        )

    def __repr__(self) -> str:
        """Return an interactive representation of the configured stage."""

        arguments = [repr(arg) for arg in self.args]
        if self.warnings != "collect":
            arguments.append(f"warnings={self.warnings!r}")
        if self.cache is not False:
            arguments.append(f"cache={self.cache!r}")
        return f"Num({', '.join(arguments)})"


class NumOperator:
    """Compile symbolic expressions into numeric values or callables."""

    Scalar: ScalarType = SCALAR
    Index: IndexType = INDEX
    _mt_help = {
        "path": PurePosixPath("library/Num"),
        "anchor": None,
        "label": "Num",
    }

    def Array(self, rank: int, element_type: object | None = None) -> ArrayType:
        """Return an array numeric role with scalar elements by default."""

        if element_type is None:
            element_type = self.Scalar
        return ArrayType(rank, element_type)

    def Integrate(
        self,
        function: object,
        ranges: object,
        *,
        domain_func: object | None = None,
        args: tuple[object, ...] = (),
        rule: str = "gk21",
        rtol: float = 1e-8,
        atol: float = 0.0,
        max_subdivisions: int = 10000,
        workers: int | object = 1,
        points: object | None = None,
    ) -> object:
        """Integrate a numeric or symbolic function with SciPy cubature.

        Parameters
        ----------
        function : object
            Symbolic expression, compiled ``NumFunction``, or vectorized
            callable to integrate.
        ranges : object
            Rectangular bounds such as ``[(x, -1, 1), (y, -1, 1)]``.
        domain_func : object, optional
            Numeric or symbolic function that is negative on the desired
            domain and positive outside it.
        args : tuple[object, ...], optional
            Extra positional values passed after the coordinate arrays.
        rule : str, optional
            Cubature rule passed to SciPy.
        rtol : float, optional
            Relative integration tolerance.
        atol : float, optional
            Absolute integration tolerance.
        max_subdivisions : int, optional
            Maximum cubature subdivisions.
        workers : int or object, optional
            Worker configuration passed to SciPy.
        points : object, optional
            Points passed to SciPy for avoiding singularities.

        Returns
        -------
        object
            SciPy ``CubatureResult`` with ``estimate`` and ``error`` fields.
        """

        return integrate(
            function,
            ranges,
            domain_func=domain_func,
            args=args,
            rule=rule,
            rtol=rtol,
            atol=atol,
            max_subdivisions=max_subdivisions,
            workers=workers,
            points=points,
        )

    def __call__(
        self,
        *args: Any,
        warnings: object = "collect",
        cache: object = False,
    ) -> NumStage:
        """Return a configured ``Num`` pipeline stage."""

        return NumStage(args=args, warnings=warnings, cache=cache)

    def __rrshift__(self, expr: object) -> object:
        """Compile the expression on the left of ``>>`` with default options."""

        return compile_num(expr, args=(), warnings="collect", cache=False)

    def __repr__(self) -> str:
        """Return an interactive representation of the operator."""

        return "<Num: use as expr >> Num or expr >> Num(var=...)>"


Num = NumOperator()
