"""
#TODO Write docstring
#TODO Move this to argument_autodiscovery
Discover scalar, index, and array roles in symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy
from sympy.core.function import AppliedUndef

from .diagnostics import (
    NumHeldExpressionError,
    NumRoleError,
    NumUnsupportedExpressionError,
)
from .functions import implemented_function_from_application
from .specs import NumArgSpec, make_indexed_base, visible_name
from ...sympy_extensions.holding import HeldExpression
from .types import INDEX, SCALAR, ArrayType, IndexType, ScalarType


@dataclass(frozen=True)
class DiscoveredRole:
    """Store one numeric role discovered from expression structure."""

    name: str
    symbol: object
    type: ScalarType | IndexType | ArrayType
    evidence: object
    lambdify_symbol: object


def discover_roles(expr: object) -> tuple[DiscoveredRole, ...]:
    """Return numeric roles discovered from a symbolic expression."""

    collector = _RoleCollector()
    for part in _expression_parts(expr):
        collector.visit(part, bound_symbols=frozenset(), index_context=False)
    return collector.finalize()


def merge_explicit_and_discovered_roles(
    explicit_specs: tuple[NumArgSpec, ...],
    discovered_roles: tuple[DiscoveredRole, ...],
) -> tuple[NumArgSpec, ...]:
    """Return final argument specs after validating explicit role choices."""

    roles_by_name = {role.name: role for role in discovered_roles}
    explicit_by_visible_name: set[str] = set()
    final_specs: list[NumArgSpec] = []

    # Explicit arguments own the leading call order, but their role cannot
    # contradict the expression's structural evidence.
    for spec in explicit_specs:
        name = visible_name(spec.symbol)
        role = roles_by_name.get(name)
        if role is not None and role.type != spec.type:
            raise NumRoleError(
                f"Num argument {name} uses {spec.type!r}, but the expression "
                f"uses {role.evidence} as {role.type!r}."
            )
        final_specs.append(spec)
        explicit_by_visible_name.add(name)

    # Any roles the user did not name explicitly remain part of the numeric
    # function and are appended deterministically.
    for role in sorted(discovered_roles, key=lambda item: item.name):
        if role.name in explicit_by_visible_name:
            continue
        final_specs.append(
            NumArgSpec(symbol=role.symbol, name=role.name, type=role.type)
        )

    return tuple(final_specs)


def lambdify_symbols_for_specs(
    specs: tuple[NumArgSpec, ...],
    discovered_roles: tuple[DiscoveredRole, ...],
) -> tuple[object, ...]:
    """Return backend argument symbols matching public spec order."""

    roles_by_name = {role.name: role for role in discovered_roles}
    arg_symbols: list[object] = []
    for spec in specs:
        role = roles_by_name.get(visible_name(spec.symbol))
        if isinstance(spec.type, ArrayType):
            if role is not None:
                arg_symbols.append(role.lambdify_symbol)
            else:
                arg_symbols.append(make_indexed_base(spec.symbol))
            continue
        arg_symbols.append(sympy.sympify(spec.symbol))
    return tuple(arg_symbols)


@dataclass(frozen=True)
class _CandidateRole:
    """Internal role evidence collected before conflict resolution."""

    name: str
    symbol: object
    type: ScalarType | IndexType | ArrayType
    evidence: object
    lambdify_symbol: object


class _RoleCollector:
    """Collect numeric role evidence from a real expression tree walk."""

    def __init__(self) -> None:
        self._arrays: dict[str, list[_CandidateRole]] = {}
        self._indices: dict[str, list[_CandidateRole]] = {}
        self._scalars: dict[str, list[_CandidateRole]] = {}
        self._bound_names: set[str] = set()

    def visit(
        self,
        value: object,
        *,
        bound_symbols: frozenset[sympy.Symbol],
        index_context: bool,
    ) -> None:
        """Visit one expression node and update collected role candidates."""

        implemented_function = implemented_function_from_application(value)
        if implemented_function is not None:
            for parameter in implemented_function.required_parameters:
                self._record_scalar_symbol(parameter)
            for argument in value.args:
                self.visit(
                    argument,
                    bound_symbols=bound_symbols,
                    index_context=index_context,
            )
            return

        self._raise_for_held_or_opaque(value)

        if isinstance(value, sympy.Indexed):
            element_type = INDEX if index_context else SCALAR
            self._record_indexed(value, element_type=element_type)
            for index in value.indices:
                self.visit(
                    index,
                    bound_symbols=bound_symbols,
                    index_context=True,
                )
            return

        if isinstance(value, sympy.IndexedBase):
            raise NumRoleError(
                f"Num cannot infer an array role from bare IndexedBase {value}. "
                "Use indexed access such as x[i] or pass an explicit "
                "Num.Array(...) argument."
            )

        if isinstance(value, sympy.Symbol):
            if value in bound_symbols:
                return
            if index_context:
                self._record_index_symbol(value)
            else:
                self._record_scalar_symbol(value)
            return

        if not isinstance(value, sympy.Basic):
            try:
                sympified = sympy.sympify(value)
            except sympy.SympifyError as exc:
                raise NumUnsupportedExpressionError(
                    f"Num cannot compile unsupported expression {value!r}."
                ) from exc
            if sympified is not value:
                self.visit(
                    sympified,
                    bound_symbols=bound_symbols,
                    index_context=index_context,
                )
            return

        local_bound = self._bound_symbols_from(value)
        next_bound = bound_symbols | local_bound
        for symbol in local_bound:
            self._bound_names.add(visible_name(symbol))

        for child in value.args:
            self.visit(
                child,
                bound_symbols=next_bound,
                index_context=index_context,
            )

    def finalize(self) -> tuple[DiscoveredRole, ...]:
        """Return conflict-checked discovered roles."""

        array_roles = self._resolve_same_named_candidates(self._arrays)
        index_roles = self._resolve_same_named_candidates(self._indices)
        scalar_roles = self._resolve_same_named_candidates(self._scalars)

        # Index roles are integer scalar arguments, so ordinary arithmetic uses
        # of the same symbol are allowed to ride along with the index role.
        for name in list(scalar_roles):
            if name in index_roles:
                scalar_roles.pop(name)

        for name, array_role in array_roles.items():
            scalar_role = scalar_roles.get(name)
            if scalar_role is not None:
                raise NumRoleError(
                    f"Num found conflicting numeric roles for {name!r}: "
                    f"scalar use {scalar_role.evidence} and array use "
                    f"{array_role.evidence}. Pass explicit Num arguments or "
                    "rename one role before numericalization."
                )
            index_role = index_roles.get(name)
            if index_role is not None:
                raise NumRoleError(
                    f"Num found conflicting numeric roles for {name!r}: "
                    f"index use {index_role.evidence} and array use "
                    f"{array_role.evidence}. Pass explicit Num arguments or "
                    "rename one role before numericalization."
                )

        for name, index_role in index_roles.items():
            if name in self._bound_names:
                raise NumRoleError(
                    f"Num found index symbol {name!r} used as both a bound "
                    f"dummy and an external index role near {index_role.evidence}."
                )

        roles: list[DiscoveredRole] = []
        roles.extend(array_roles.values())
        roles.extend(index_roles.values())
        roles.extend(scalar_roles.values())
        return tuple(sorted(roles, key=lambda item: item.name))

    def _record_indexed(
        self,
        indexed: sympy.Indexed,
        *,
        element_type: IndexType | object,
    ) -> None:
        """Record one indexed array access."""

        base = indexed.base
        name = visible_name(base)
        public_symbol = _public_symbol_for_indexed_base(base)
        role_type = ArrayType(len(indexed.indices), element_type)
        self._arrays.setdefault(name, []).append(
            _CandidateRole(
                name=name,
                symbol=public_symbol,
                type=role_type,
                evidence=indexed,
                lambdify_symbol=base,
            )
        )

    def _record_index_symbol(self, symbol: sympy.Symbol) -> None:
        """Record a free index argument."""

        name = visible_name(symbol)
        self._indices.setdefault(name, []).append(
            _CandidateRole(
                name=name,
                symbol=symbol,
                type=INDEX,
                evidence=symbol,
                lambdify_symbol=symbol,
            )
        )

    def _record_scalar_symbol(self, symbol: sympy.Symbol) -> None:
        """Record an ordinary scalar argument candidate."""

        name = visible_name(symbol)
        self._scalars.setdefault(name, []).append(
            _CandidateRole(
                name=name,
                symbol=symbol,
                type=SCALAR,
                evidence=symbol,
                lambdify_symbol=symbol,
            )
        )

    def _resolve_same_named_candidates(
        self,
        candidates_by_name: dict[str, list[_CandidateRole]],
    ) -> dict[str, DiscoveredRole]:
        """Return one role per name, raising when evidence disagrees."""

        resolved: dict[str, DiscoveredRole] = {}
        for name, candidates in candidates_by_name.items():
            first = candidates[0]
            for candidate in candidates[1:]:
                if candidate.type != first.type:
                    raise NumRoleError(
                        f"Num found conflicting numeric roles for {name!r}: "
                        f"use {first.evidence} as {first.type!r} and "
                        f"use {candidate.evidence} as {candidate.type!r}."
                    )
            resolved[name] = DiscoveredRole(
                name=first.name,
                symbol=first.symbol,
                type=first.type,
                evidence=first.evidence,
                lambdify_symbol=first.lambdify_symbol,
            )
        return resolved

    def _bound_symbols_from(
        self,
        value: sympy.Basic,
    ) -> frozenset[sympy.Symbol]:
        """Return local bound symbols declared by a SymPy node."""

        raw_bound = getattr(value, "bound_symbols", ())
        if not raw_bound:
            return frozenset()
        return frozenset(
            symbol for symbol in raw_bound if isinstance(symbol, sympy.Symbol)
        )

    def _raise_for_held_or_opaque(self, value: object) -> None:
        """Reject symbolic boundaries that ``Num`` does not cross implicitly."""

        if isinstance(value, HeldExpression) or _is_held_named_expression(value):
            raise NumHeldExpressionError(
                f"Num cannot compile held notation {value}. Use expr >> "
                "UnholdAll >> Num when you want to expose held definitions first."
            )

        if _is_opaque_named_expression(value):
            raise NumUnsupportedExpressionError(
                f"Num cannot compile the opaque named expression {value}. "
                "Expose a definition with UnholdAll before Num when one exists."
            )

        if isinstance(value, sympy.Function):
            if getattr(value.func, "_math_toolkit_held", False):
                raise NumHeldExpressionError(
                    f"Num cannot compile held notation {value}. Use expr >> "
                    "UnholdAll >> Num when you want to expose held definitions first."
                )
            if isinstance(value, AppliedUndef):
                if hasattr(value.func, "_mt_indices") or hasattr(
                    value.func,
                    "_math_toolkit_indices",
                ):
                    raise NumUnsupportedExpressionError(
                        f"Num cannot compile the opaque function call {value}. "
                        "Expose a definition with UnholdAll before Num, or "
                        "provide an explicit numeric implementation when that "
                        "API exists."
                    )
                raise NumUnsupportedExpressionError(
                    f"Num cannot compile the opaque function call {value}. "
                    "Expose a definition with UnholdAll before Num, or provide "
                    "an explicit numeric implementation when that API exists."
                )
            if hasattr(value.func, "_math_toolkit_definitions"):
                raise NumUnsupportedExpressionError(
                    f"Num cannot compile the opaque function call {value}. "
                    "Expose a definition with UnholdAll before Num, or provide "
                    "an explicit numeric implementation when that API exists."
                )


def _expression_parts(expr: object) -> tuple[object, ...]:
    """Return traversable expression parts while preserving matrix entries."""

    if isinstance(expr, sympy.MatrixBase):
        return tuple(expr[row, col] for row in range(expr.rows) for col in range(expr.cols))
    return (expr,)


def _public_symbol_for_indexed_base(base: sympy.IndexedBase) -> object:
    """Return the public symbol represented by an indexed base."""

    label = base.label
    if isinstance(label, sympy.Symbol):
        return label
    return sympy.Symbol(str(label))


def _is_held_named_expression(value: object) -> bool:
    """Return whether ``value`` is a held named-expression boundary."""

    family = getattr(value, "_math_toolkit_family", None)
    if family is not None and getattr(family, "_math_toolkit_held", False):
        return True
    return bool(
        getattr(value, "_math_toolkit_held", False)
        and hasattr(value, "_math_toolkit_definitions")
    )


def _is_opaque_named_expression(value: object) -> bool:
    """Return whether ``value`` is an opaque named expression."""

    return hasattr(value, "_math_toolkit_family")
