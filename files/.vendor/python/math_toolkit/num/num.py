from __future__ import annotations

"""Provide the public Num entrypoint and argument autodetection helpers.

This module wraps the numeric compiler surface to provide a convenience
boundary for compiling expressions, along with lightweight symbol-name
sanitation and argument autodetection.
"""

import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import sympy

from .Compiler import NumCompiler
from .legacy._disabled import (
    NumArgumentError,
    NumNotImplementedError,
    NumUnsupportedExpressionError,
)
from .numfunction_implementedfunction import NumArgSpec
from ..pipeops import pipeop

__all__ = [
    "compile",
    "IndexedRuntimeParameterInfo",
    "Num",
    "NumArgSpec",
    "NumArgumentError",
    "NumCompiler",
    "NumNotImplementedError",
    "NumUnsupportedExpressionError",
    "autodetect_args",
    "autodetect_runtime_args",
    "indexed_runtime_parameter_info",
    "sanitize_symbol_name",
]

# Keep the legacy error prefix stable while documenting the new IndexedBase path.
_VAR_TYPE_ERROR = "var must be a Symbol, tuple, dict, or None; IndexedBase is also supported"
_DEFAULT_INDEXED_PARAMETER_COUNT = 3
_DIGIT_RUN = re.compile(r"\d+")


@dataclass(frozen=True)
class IndexedRuntimeParameterInfo:
    """Describe concrete indexed UI entries for one Num array argument."""

    base: sympy.IndexedBase
    entries: tuple[sympy.Indexed, ...]
    complete: bool


def _compile_num(
    expr: sympy.Basic,
    *,
    var: sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None = None,
    autodetect_vars: bool = True,
    **compile_hints: object,
):
    """Compile or immediately evaluate a public numeric expression.

    Parameters
    ----------
    expr : sympy.Basic
        The symbolic expression to compile or evaluate.
    var : sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None
        Explicitly declared symbols or mapping of symbols to Python names.
    autodetect_vars : bool
        Whether to automatically detect and include unbound free symbols.
    **compile_hints : object
        Additional hints to pass directly to the compiler.

    Returns
    -------
    object
        The compiled numeric function, or an evaluated scalar/array if no
        variables are present.

    Raises
    ------
    NumArgumentError
        If unbound variables are detected when autodetection is disabled,
        or if hints are malformed.
    NumUnsupportedExpressionError
        If the expression contains unsupported elements like derivatives.
    """
    # Normalize the input expression to ensure we are working with a
    # standard SymPy object.
    expression = sympy.sympify(expr)

    # Establish the explicit symbols provided by the user to check against
    # the expression's free symbols.
    explicit_specs = _normalize_explicit_var_specs(var)
    explicit_symbols = {spec.symbol for spec in explicit_specs}

    # Enforce variable binding constraints when the user disables
    # automatic discovery.
    if not autodetect_vars:
        unbound = expression.free_symbols - explicit_symbols
        if unbound:
            raise NumArgumentError("Unbound variables detected")

    # Reject unsupported mathematical operations that the numeric compiler
    # cannot currently lower.
    if expression.has(sympy.Derivative):
        raise NumUnsupportedExpressionError("Unsupported expression: Derivative")

    # Matrix constants are already concrete table-shaped values; scalar
    # constants continue through the compiler so symbolic numbers such as pi
    # are lowered through the same numeric path as larger expressions.
    if not expression.free_symbols and var is None and isinstance(expression, sympy.MatrixBase):
        return np.array(expression.tolist(), dtype=float)

    # Use the shared Num argument discoverer so indexed arrays, reductions, and
    # ordinary scalar expressions all follow one binding policy.
    arg_specs = autodetect_runtime_args(
        expression,
        var=var,
        autodetect_vars=autodetect_vars,
    )

    # Merge the top-level compile_hints kwargs into a flat dictionary, handling
    # the legacy `compile_hints={...}` nested mapping pattern if it exists.
    hints = dict(compile_hints)
    nested_hints = hints.pop("compile_hints", None)
    if nested_hints is not None:
        if not isinstance(nested_hints, Mapping):
            raise NumArgumentError("compile_hints must be a mapping when provided.")
        hints = {**dict(nested_hints), **hints}

    # Delegate to the default compiler instance to produce the executable function.
    compiler = NumCompiler()
    compiled = compiler.compile(expression=expression, arg_specs=arg_specs, hints=hints)
    if not expression.free_symbols and var is None:
        return compiled()
    return compiled


Num = pipeop(_compile_num, name="Num")


def compile(
    expr: sympy.Basic,
    *,
    arg_specs: tuple[NumArgSpec, ...] = (),
    hints: Mapping[str, object] | None = None,
):
    """Compile an expression with the default numeric compiler instance.

    Parameters
    ----------
    expr : sympy.Basic
        The symbolic expression to compile.
    arg_specs : tuple[NumArgSpec, ...]
        The specification of arguments the resulting function will accept.
    hints : Mapping[str, object] | None
        Optional hints to configure compiler behavior.

    Returns
    -------
    object
        The compiled numeric function.
    """
    # Construct a fresh compiler and immediately compile the expression using
    # the provided explicit argument definitions.
    compiler = NumCompiler()
    return compiler.compile(expression=expr, arg_specs=arg_specs, hints=hints)


def autodetect_args(
    expr: sympy.Basic,
    var: sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None = None,
    autodetect_vars: bool = True,
) -> tuple[NumArgSpec, ...]:
    """Return argument specs for a symbolic expression and optional explicit vars.

    Parameters
    ----------
    expr : sympy.Basic
        The expression to analyze for free symbols.
    var : sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None
        Explicitly declared symbols to include in the specification.
    autodetect_vars : bool
        Whether to automatically detect missing variables from the expression.

    Returns
    -------
    tuple[NumArgSpec, ...]
        The ordered sequence of argument specifications.

    Raises
    ------
    NumNotImplementedError
        If the expression contains indexed symbols.
    NumArgumentError
        If free symbols exist but autodetection is disabled and no vars are provided,
        or if symbol names collide.
    """
    # Ensure the input is a valid SymPy expression before querying its symbols.
    expression = sympy.sympify(expr)
    free_symbols = expression.free_symbols

    # Reject indexed objects here; they require specialized discovery via
    # the `_autodetect_indexed_args` fallback in `Num`.
    if expression.has(sympy.IndexedBase) or expression.has(sympy.Indexed):
        raise NumNotImplementedError("Indexed symbols are not supported")

    # Validate that distinct Symbol objects do not map to the same raw name,
    # which would cause collisions during Python code generation.
    names: dict[str, sympy.Symbol] = {}
    for symbol in free_symbols:
        existing = names.get(symbol.name)
        if existing is not None and existing != symbol:
            raise NumArgumentError("Name collision detected")
        names[symbol.name] = symbol

    # Normalize user-provided variables into canonical argument specifications.
    explicit_specs = _normalize_explicit_var_specs(var)
    explicit_symbols = {spec.symbol for spec in explicit_specs}

    # Reject expressions that require variable bindings if the caller disabled
    # autodetection and provided no explicit variables.
    if not autodetect_vars and var is None and free_symbols:
        raise NumArgumentError(
            "Free symbols present but autodetection disabled and no var provided"
        )

    # Return only the explicitly requested variables if autodetection is off.
    if not autodetect_vars:
        return explicit_specs

    # Discover any remaining free symbols, sanitize their names for Python,
    # and append them in alphabetical order.
    autodetected_specs = [
        NumArgSpec(symbol=symbol, name=sanitize_symbol_name(symbol))
        for symbol in free_symbols
        if symbol not in explicit_symbols
    ]
    autodetected_specs.sort(key=lambda spec: _natural_text_sort_key(spec.name))
    return explicit_specs + tuple(autodetected_specs)


def autodetect_runtime_args(
    expr: sympy.Basic,
    var: sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None = None,
    autodetect_vars: bool = True,
) -> tuple[NumArgSpec, ...]:
    """Return Num argument specs using the same discovery policy as ``Num``."""

    expression = sympy.sympify(expr)
    needs_indexed = (
        expression.has(sympy.IndexedBase, sympy.Indexed, sympy.Sum, sympy.Product)
        or any(isinstance(symbol, sympy.Idx) for symbol in expression.free_symbols)
    )
    if needs_indexed:
        return _autodetect_indexed_args(
            expression,
            var=var,
            autodetect_vars=autodetect_vars,
        )
    return autodetect_args(
        expression,
        var=var,
        autodetect_vars=autodetect_vars,
    )


def indexed_runtime_parameter_info(
    expr: sympy.Basic,
    *,
    default_count: int = _DEFAULT_INDEXED_PARAMETER_COUNT,
) -> tuple[IndexedRuntimeParameterInfo, ...]:
    """Return concrete indexed entries represented by Num array arguments."""

    expression = sympy.sympify(expr)
    if default_count < 1:
        raise NumArgumentError("default_count must be a positive integer.")

    infos: list[IndexedRuntimeParameterInfo] = []
    for spec in autodetect_runtime_args(expression, var=(), autodetect_vars=True):
        if not isinstance(spec.symbol, sympy.IndexedBase):
            continue
        entries, complete = _indexed_entries_for_base(
            expression,
            spec.symbol,
            default_count=default_count,
        )
        infos.append(
            IndexedRuntimeParameterInfo(
                base=spec.symbol,
                entries=tuple(sorted(entries, key=_visible_parameter_sort_key)),
                complete=complete,
            )
        )
    return tuple(infos)


def _indexed_entries_for_base(
    expression: sympy.Basic,
    base: sympy.IndexedBase,
    *,
    default_count: int,
) -> tuple[set[sympy.Indexed], bool]:
    """Return visible indexed entries and whether discovery was exhaustive."""

    entries = _indexed_entries_from_node(expression, base, {})
    if entries is not None:
        return entries, True

    rank = _indexed_rank_for_base(expression, base)
    default_entries = {
        base[
            tuple(
                sympy.Integer(index if axis == 0 else 0)
                for axis in range(rank)
            )
        ]
        for index in range(default_count)
    }
    return default_entries, False


def _indexed_entries_from_node(
    node: sympy.Basic,
    base: sympy.IndexedBase,
    bound_values: Mapping[sympy.Basic, tuple[int, ...]],
) -> set[sympy.Indexed] | None:
    """Collect concrete indexed entries for one base or return unknown."""

    if isinstance(node, (sympy.Sum, sympy.Product)):
        next_bound_values = dict(bound_values)
        for limit in node.limits:
            if len(limit) != 3 or not isinstance(limit[0], (sympy.Symbol, sympy.Idx)):
                continue
            values = _finite_integer_values(limit[1], limit[2])
            if values is None:
                return None
            next_bound_values[limit[0]] = values
        return _indexed_entries_from_node(node.function, base, next_bound_values)

    if isinstance(node, sympy.Indexed) and str(node.base) == str(base):
        return _concrete_indexed_variants(node, bound_values)

    discovered: set[sympy.Indexed] = set()
    for child in getattr(node, "args", ()):
        if not isinstance(child, sympy.Basic):
            continue
        child_entries = _indexed_entries_from_node(child, base, bound_values)
        if child_entries is None:
            return None
        discovered.update(child_entries)
    return discovered


def _concrete_indexed_variants(
    node: sympy.Indexed,
    bound_values: Mapping[sympy.Basic, tuple[int, ...]],
) -> set[sympy.Indexed] | None:
    """Return concrete indexed entries represented by one indexed read."""

    choices = []
    for index in node.indices:
        if index in bound_values:
            choices.append(bound_values[index])
            continue
        value = _finite_integer_value(index)
        if value is None or value < 0:
            return None
        choices.append((value,))
    return {
        node.base[tuple(sympy.Integer(value) for value in values)]
        for values in product(*choices)
    }


def _indexed_rank_for_base(expression: sympy.Basic, base: sympy.IndexedBase) -> int:
    """Return the first observed rank for one indexed base."""

    for node in sympy.preorder_traversal(expression):
        if isinstance(node, sympy.Indexed) and str(node.base) == str(base):
            return max(1, len(node.indices))
    return 1


def _finite_integer_values(minimum: object, maximum: object) -> tuple[int, ...] | None:
    """Return inclusive nonnegative integer bounds when known."""

    lower = _finite_integer_value(minimum)
    upper = _finite_integer_value(maximum)
    if lower is None or upper is None or lower < 0 or upper < lower:
        return None
    return tuple(range(lower, upper + 1))


def _finite_integer_value(value: object) -> int | None:
    """Return an exact integer value or ``None`` when symbolic."""

    try:
        sympified = sympy.sympify(value)
    except sympy.SympifyError:
        return None
    if not bool(getattr(sympified, "is_integer", False)):
        return None
    try:
        return int(sympified)
    except TypeError:
        return None


# Translate LaTeX-style math symbols into pure ASCII identifiers for Python kwargs.
_GREEK_NAME_MAP = {
    "\\alpha": "alpha",
    "\\beta": "beta",
    "\\gamma": "gamma",
    "\\delta": "delta",
    "\\epsilon": "epsilon",
    "\\zeta": "zeta",
    "\\eta": "eta",
    "\\theta": "theta",
    "\\iota": "iota",
    "\\kappa": "kappa",
    "\\lambda": "lambda_",
    "\\mu": "mu",
    "\\nu": "nu",
    "\\xi": "xi",
    "\\pi": "pi",
    "\\rho": "rho",
    "\\sigma": "sigma",
    "\\tau": "tau",
    "\\phi": "phi",
    "\\chi": "chi",
    "\\psi": "psi",
    "\\omega": "omega",
}


def sanitize_symbol_name(symbol: sympy.Symbol | sympy.IndexedBase) -> str:
    """Return a Python-safe external keyword name for one SymPy symbol.

    Parameters
    ----------
    symbol : sympy.Symbol
        The mathematical symbol to sanitize.

    Returns
    -------
    str
        A valid Python identifier corresponding to the symbol.

    Raises
    ------
    NumArgumentError
        If the input is not a Symbol, or if the name cannot be transformed
        into a valid Python identifier.
    """
    # Enforce that only strict SymPy symbols are passed to avoid crashing
    # on unexpected expression nodes.
    if not isinstance(symbol, (sympy.Symbol, sympy.IndexedBase)):
        raise NumArgumentError("Only SymPy Symbol or IndexedBase values can become Num arguments.")

    # Translate known Greek LaTeX strings into English words, and strip
    # leading slashes for unknown LaTeX commands.
    raw_name = symbol.name if isinstance(symbol, sympy.Symbol) else str(symbol)
    sanitized = _GREEK_NAME_MAP.get(raw_name, raw_name.removeprefix("\\"))

    # Append a trailing underscore to avoid clashing with reserved keywords.
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_"

    # Reject names that still aren't valid identifiers, guiding the user
    # to provide an explicit string mapping instead.
    if not sanitized.isidentifier():
        raise NumArgumentError(
            f"Symbol {raw_name!r} has an invalid Python identifier. "
            "Pass var={symbol: 'name'} to provide an explicit keyword name."
        )

    return sanitized


def _normalize_explicit_var_specs(
    var: sympy.Symbol | sympy.IndexedBase | Sequence[sympy.Symbol | sympy.IndexedBase] | Mapping[sympy.Symbol | sympy.IndexedBase, str] | None,
) -> tuple[NumArgSpec, ...]:
    """Return explicit ``NumArgSpec`` values in the caller's declared order.

    Parameters
    ----------
    var : sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None
        The user's declared explicit variables.

    Returns
    -------
    tuple[NumArgSpec, ...]
        Normalized specifications.

    Raises
    ------
    NumArgumentError
        If the variables are provided in an unsupported or malformed structure.
    """
    # Handle the empty case immediately.
    if var is None:
        return ()

    # Wrap a single symbol into a one-element tuple to simplify downstream processing.
    if isinstance(var, (sympy.Symbol, sympy.IndexedBase)):
        return (NumArgSpec(symbol=var, name=sanitize_symbol_name(var)),)

    # Process dictionary mappings, ensuring the user provided valid Python strings
    # as their explicit identifier choices.
    if isinstance(var, Mapping):
        specs: list[NumArgSpec] = []
        for symbol, name in var.items():
            if not isinstance(symbol, (sympy.Symbol, sympy.IndexedBase)) or not isinstance(name, str):
                raise NumArgumentError("var dict entries must map Symbol or IndexedBase values to string names")
            if not name.isidentifier():
                raise NumArgumentError(
                    f"Symbol {str(symbol)!r} maps to invalid Python identifier {name!r}."
                )
            specs.append(NumArgSpec(symbol=symbol, name=name))
        return tuple(specs)

    # Process sequential inputs, applying the default sanitizer to each symbol.
    if isinstance(var, tuple) or isinstance(var, list):
        specs = []
        for symbol in var:
            if not isinstance(symbol, (sympy.Symbol, sympy.IndexedBase)):
                raise NumArgumentError(_VAR_TYPE_ERROR)
            specs.append(NumArgSpec(symbol=symbol, name=sanitize_symbol_name(symbol)))
        return tuple(specs)

    # Reject completely invalid input types.
    raise NumArgumentError(_VAR_TYPE_ERROR)


def _autodetect_indexed_args(
    expr: sympy.Basic,
    *,
    var: sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None,
    autodetect_vars: bool,
) -> tuple[NumArgSpec, ...]:
    """Return indexed-aware argument specs for the public ``Num`` wrapper.

    Parameters
    ----------
    expr : sympy.Basic
        The expression to analyze.
    var : sympy.Symbol | Sequence[sympy.Symbol] | Mapping[sympy.Symbol, str] | None
        Optional explicitly declared variables.
    autodetect_vars : bool
        Whether to dynamically discover variables.

    Returns
    -------
    tuple[NumArgSpec, ...]
        Argument specifications including discovered array bases and scalars.
    """
    # Retrieve explicitly defined variables first, establishing the baseline.
    explicit_specs = _normalize_explicit_var_specs(var)

    # If discovery is off, strip out index symbols (like `i` or `j`) since
    # they shouldn't be exposed as external parameters.
    if not autodetect_vars:
        return tuple(
            spec for spec in explicit_specs if not isinstance(spec.symbol, sympy.Idx)
        )

    # Traverse the expression tree to locate `Indexed` nodes, tracking their
    # base objects (the arrays) to expose as array arguments.
    explicit_symbols = {spec.symbol for spec in explicit_specs}
    discovered_arrays: dict[str, sympy.IndexedBase] = {}
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Indexed):
            discovered_arrays.setdefault(str(node.base), node.base)

    # Collect any standard symbols that are not array bases, index iterators,
    # or explicitly user-provided, treating them as external scalar parameters.
    discovered_scalars = [
        symbol
        for symbol in sorted(expr.free_symbols, key=_visible_parameter_sort_key)
        if isinstance(symbol, sympy.Symbol)
        and not isinstance(symbol, sympy.Idx)
        and symbol not in explicit_symbols
    ]

    # Construct final specification lists for discovered arrays and scalars.
    discovered_array_specs = [
        NumArgSpec(symbol=base, name=str(base))
        for _, base in sorted(
            discovered_arrays.items(),
            key=lambda item: _natural_text_sort_key(item[0]),
        )
        if base not in explicit_symbols
    ]
    discovered_scalar_specs = [
        NumArgSpec(symbol=symbol, name=sanitize_symbol_name(symbol))
        for symbol in discovered_scalars
        if str(symbol) not in discovered_arrays
    ]

    # Combine explicit variables and sorted discovered variables into one signature.
    return explicit_specs + tuple(
        sorted(
            discovered_array_specs + discovered_scalar_specs,
            key=lambda spec: _natural_text_sort_key(spec.name),
        )
    )


def _visible_parameter_sort_key(
    symbol: sympy.Basic,
) -> tuple[tuple[int, int | str], ...]:
    """Return a natural visible-name key for autodetected parameters."""

    return _natural_text_sort_key(str(symbol))


def _natural_text_sort_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Return a key that compares digit runs by numeric value."""

    pieces: list[tuple[int, int | str]] = []
    position = 0
    for match in _DIGIT_RUN.finditer(text):
        if match.start() > position:
            pieces.append((1, text[position : match.start()]))
        pieces.append((0, int(match.group())))
        position = match.end()
    if position < len(text):
        pieces.append((1, text[position:]))
    return tuple(pieces)
