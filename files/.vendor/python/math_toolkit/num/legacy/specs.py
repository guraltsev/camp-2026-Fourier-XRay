"""Normalize public ``Num`` argument specifications."""

from __future__ import annotations

from dataclasses import dataclass
import keyword
from typing import Any

import sympy

from .diagnostics import NumArgumentError
from .types import ArrayType, IndexType, ScalarType, SCALAR, is_numeric_arg_type


@dataclass(frozen=True)
class NumArgSpec:
    """Describe one argument accepted by a ``NumFunction``.

    Parameters
    ----------
    symbol : object
        Symbolic object represented by the numeric argument.
    name : str
        External keyword name accepted by the numeric callable.
    type : ScalarType | IndexType | ArrayType
        Numeric role assigned to the argument.
    """

    symbol: object
    name: str
    type: ScalarType | IndexType | ArrayType


def normalize_arg_specs(args: tuple[Any, ...]) -> tuple[NumArgSpec, ...]:
    """Return validated public argument specifications in call order."""

    specs = tuple(_normalize_one_arg_spec(arg) for arg in args)

    # External names form the Python keyword surface of NumFunction calls.
    seen_names: dict[str, NumArgSpec] = {}
    for spec in specs:
        previous = seen_names.get(spec.name)
        if previous is not None:
            raise NumArgumentError(
                f"Num arguments use duplicate external name {spec.name!r}."
            )
        seen_names[spec.name] = spec
    return specs


def visible_name(symbol: object) -> str:
    """Return the public symbolic base name used for role conflict checks."""

    if isinstance(symbol, sympy.Indexed):
        return visible_name(symbol.base)
    if isinstance(symbol, sympy.IndexedBase):
        label = symbol.label
        if isinstance(label, sympy.Symbol):
            return label.name
        return str(label)
    if isinstance(symbol, sympy.Symbol):
        return symbol.name
    return str(symbol)


def make_indexed_base(symbol: object) -> sympy.IndexedBase:
    """Return an ``IndexedBase`` matching a public array spec symbol."""

    if isinstance(symbol, sympy.IndexedBase):
        return symbol
    if isinstance(symbol, sympy.Symbol):
        return sympy.IndexedBase(symbol)
    raise NumArgumentError(
        f"Num array argument {symbol!r} must be a Symbol or IndexedBase."
    )


def _normalize_one_arg_spec(raw: Any) -> NumArgSpec:
    """Return one validated ``NumArgSpec`` from the public grammar."""

    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        symbol, options = raw
        return _normalize_options_spec(symbol, options)
    if isinstance(raw, tuple):
        raise NumArgumentError(
            "Num argument tuples must have the form (symbol, options_dict)."
        )
    return _make_spec(raw, name=None, arg_type=SCALAR)


def _normalize_options_spec(symbol: object, options: dict[str, Any]) -> NumArgSpec:
    """Return one spec from a symbol and public options dictionary."""

    unknown = sorted(set(options) - {"name", "type"})
    if unknown:
        joined = ", ".join(unknown)
        raise NumArgumentError(f"Unknown Num argument option(s): {joined}.")

    name = options.get("name")
    arg_type = options.get("type", SCALAR)
    return _make_spec(symbol, name=name, arg_type=arg_type)


def _make_spec(
    symbol: object,
    *,
    name: object | None,
    arg_type: object,
) -> NumArgSpec:
    """Validate one symbolic argument and return its public spec."""

    if isinstance(symbol, sympy.Indexed):
        raise NumArgumentError(
            "Num array arguments use the visible base, such as x, not x[i]."
        )
    if not isinstance(symbol, sympy.Symbol | sympy.IndexedBase):
        raise NumArgumentError(
            f"Num argument {symbol!r} must be a Symbol or IndexedBase."
        )
    if not is_numeric_arg_type(arg_type):
        raise NumArgumentError(
            f"Num argument {visible_name(symbol)} uses unsupported type "
            f"{arg_type!r}. MVP Num supports Num.Scalar, Num.Index, and "
            "Num.Array(rank)."
        )

    external_name = visible_name(symbol) if name is None else name
    if not isinstance(external_name, str):
        raise NumArgumentError("Num argument name option must be a string.")
    if not external_name.isidentifier() or keyword.iskeyword(external_name):
        raise NumArgumentError(
            f"Num argument name {external_name!r} must be a valid Python identifier."
        )

    return NumArgSpec(symbol=symbol, name=external_name, type=arg_type)
