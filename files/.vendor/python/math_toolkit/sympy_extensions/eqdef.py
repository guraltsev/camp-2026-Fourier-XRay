"""Create symbolic named SymPy function classes from Python definitions.

Use ``NamedFunction`` as a decorator when a symbolic expression should be
authored once and then called as a custom SymPy ``Function`` class. Use
``EqDef`` with symbol or indexed-symbol left operands to author named
expressions. Hold generated results explicitly with ``Hold(...)`` or ``>> Hold``
when compact notation should remain visible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
from numbers import Integral
import operator
from pathlib import PurePosixPath
from typing import Any

import sympy
from sympy.matrices.expressions.matexpr import MatrixElement, MatrixSymbol
from sympy.core.function import FunctionClass

from ..sympy_extensions.atom_latex_representation import atom_name_to_latex
from ..sympy_extensions.function_indexing import (
    IndexedFunctionBase,
    IndexedUndefinedFunction,
)
from ..sympy_extensions.named_functions import NamedFunction
from ..sympy_extensions.symbol_name_validation import validate_name
__all__ = ["EqDef"]


# Shared contracts and metadata

# Only fixed positional symbolic signatures can round-trip through SymPy's
# function application model.
_POSITIONAL_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
}

# Private sentinel for infix operators and optional public configuration.
_MISSING = object()

# Definition dispatch keys keep the definition table plain: ordinary argument
# arity first, then index count. ``None`` means the general indexed fallback.
_GENERAL_INDEX_ARITY = None

LatexSpec = str | Callable[..., str]

# EqDef dispatch first looks for this private toolkit protocol on the left
# operand. The fallback resolver below keeps old behavior for classes without
# the hook.
_EQDEF_LEFT_HOOK = "__mt_eqdef__"
_EQDEF_NOT_IMPLEMENTED = object()

# EqDef authoring syntax

def _make_eqdef_operator(
    *,
    args: Any = _MISSING,
    indices: Any = None,
    indexes: Any = None,
    indexed: bool | None = None,
    name: str | None = None,
    latex: LatexSpec | None = None,
) -> "_EqDefinitionOperator":
    """Create an infix operator that defines symbolic named notation.

    ``EqDef`` is pure authoring syntax. The preferred form puts the symbolic
    pattern being defined on the left, such as ``Function("F")(x)`` or
    ``A[i]``. The explicit compatibility form puts only the head on the left
    and supplies placeholders through ``args=`` and ``indices=``. The infix
    expression returns a generated ``NamedFunction`` class for function
    patterns, or a named expression object or family for symbol and
    indexed-base patterns. Ordinary Python assignment is responsible for
    binding that returned object in the caller's namespace.

    Parameters
    ----------
    args : object | tuple[object, ...], optional
        Formal call-argument placeholders for the explicit compatibility form.
        Prefer putting an applied function pattern on the left.
    indices : object | tuple[object, ...] | None, default=None
        Formal bracket-index placeholders for the explicit compatibility form.
        Prefer putting an indexed pattern on the left.
    indexes : object | tuple[object, ...] | None, default=None
        Alias for ``indices``.
    indexed : bool | None, default=None
        Whether to create an indexed object even when no bracket placeholders
        are supplied.
    name : str | None, default=None
        Optional generated name. When omitted, the left pattern supplies the
        name.
    latex : str | Callable[..., str] | None, default=None
        Optional LaTeX representation passed to the generated object. Template
        strings may use ``#0`` for the normalized generated name, and callables
        receive rendered LaTeX argument strings matching the generated call
        signature.
    Returns
    -------
    _EqDefinitionOperator
        Infix operator used as ``Function("F")(x) @EqDef@ expression``.

    Raises
    ------
    TypeError
        If placeholders or options have invalid types.
    ValueError
        If placeholder names are invalid, duplicated, or conflict with the
        indexed ``idx`` implementation parameter.

    Examples
    --------
    Pattern-left function form:

    >>> import sympy
    >>> import math_toolkit.notebook
    >>> from math_toolkit import EqDef
    >>> math_toolkit.notebook.activate(namespace={})
    >>> x = sympy.Symbol("x")
    >>> F = sympy.Function("F")(x) @EqDef@ (x + 1)
    >>> F(x)
    x + 1

    Explicit argument form:

    >>> G = sympy.Function("G") @ EqDef(args=x) @ (x + 2)
    >>> G(x)
    x + 2

    Arity dispatch with pattern-left syntax:

    >>> y = sympy.Symbol("y")
    >>> F = F(x, y) @EqDef@ (x + y)
    >>> F(x, y)
    x + y

    Arity dispatch with explicit placeholders:

    >>> G = G @ EqDef(args=(x, y)) @ (x * y)
    >>> G(x, y)
    x*y

    Indexed placeholders from the left pattern:

    >>> n = sympy.Symbol("n")
    >>> B = sympy.Function("B")[n](x) @EqDef@ (x**n)
    >>> B[3](x)
    x**3

    Indexed placeholders from explicit options:

    >>> C = sympy.Function("C") @ EqDef(args=x, indices=n) @ (x + n)
    >>> C[3](x)
    x + 3

    Named expression form:

    >>> A = sympy.Symbol("A") @EqDef@ (x + 1)
    >>> A
    x + 1

    Indexed named expression form:

    >>> S = sympy.Symbol("S")[n] @EqDef@ (n**2)
    >>> S[3]
    9
    """

    # Validate public options before turning shorthand placeholders into SymPy
    # objects, so option-type errors remain direct.
    if indexed is not None and not isinstance(indexed, bool):
        raise TypeError("EqDef indexed option must be a bool or None.")
    if name is not None and not isinstance(name, str):
        raise TypeError("EqDef name option must be a string or None.")
    if indices is not None and indexes is not None:
        raise ValueError("EqDef accepts either indices= or indexes=, not both.")
    _validate_latex_option(latex, "EqDef latex")

    # Normalize the explicit compatibility form. The preferred pattern-left
    # form will fill these values later from the captured left operand.
    args_were_supplied = args is not _MISSING
    arg_placeholders = (
        ()
        if args is _MISSING
        else _normalize_eqdef_placeholders(args, "args")
    )
    if indexes is not None:
        indices = indexes
    index_placeholders = _normalize_eqdef_placeholders(indices, "indices")
    is_indexed = indices is not None if indexed is None else indexed

    # Keep the public options internally consistent before constructing the
    # partial infix operator that will later receive the left operand.
    if indices is not None and is_indexed is False:
        raise ValueError("EqDef indices require indexed=True or indexed=None.")
    _validate_eqdef_placeholders(
        arg_placeholders,
        index_placeholders,
        indexed=is_indexed,
    )

    # Return a reusable immutable operator so configured and bare EqDef forms
    # share the same implementation.
    return _EqDefinitionOperator(
        args=arg_placeholders,
        indices=index_placeholders,
        indexed=is_indexed,
        args_were_supplied=args_were_supplied,
        name=name,
        latex=latex,
    )


class _EqDefFactory:
    """Callable factory and default infix operator for ``EqDef`` definitions."""

    _mt_help = {
        "path": PurePosixPath("library/NamedFunction"),
        "anchor": None,
        "label": "NamedFunction",
    }

    def __call__(
        self,
        *,
        args: Any = _MISSING,
        indices: Any = None,
        indexes: Any = None,
        indexed: bool | None = None,
        name: str | None = None,
        latex: LatexSpec | None = None,
    ) -> "_EqDefinitionOperator":
        """Return a configured ``EqDef`` infix operator."""

        return _make_eqdef_operator(
            args=args,
            indices=indices,
            indexes=indexes,
            indexed=indexed,
            name=name,
            latex=latex,
        )

    def __rmatmul__(self, left: Any) -> "_EqDefinitionOperator":
        """Capture the left operand for the default ``@ EqDef @`` form."""

        return _make_eqdef_operator().__rmatmul__(left)

    def __matmul__(self, definition: Any) -> Any:
        """Create a no-left named definition from explicit ``EqDef`` options."""

        return _make_eqdef_operator().__matmul__(definition)

    def __repr__(self) -> str:
        """Return the interactive representation for the default operator."""

        return "<EqDef: use as pattern @ EqDef @ definition>"


# Present the singleton factory as the public EqDef object while keeping the
# callable factory's user-facing documentation on the object users inspect.
_EqDefFactory.__doc__ = _make_eqdef_operator.__doc__
EqDef = _EqDefFactory()


@dataclass(frozen=True)
class _EqDefinitionOperator:
    """Infix operator for expression-authored ``NamedFunction`` definitions."""

    _mt_help = {
        "path": PurePosixPath("library/NamedFunction"),
        "anchor": None,
        "label": "NamedFunction",
    }

    args: tuple[sympy.Symbol, ...]
    indices: tuple[sympy.Symbol, ...]
    indexed: bool
    args_were_supplied: bool
    name: str | None
    latex: LatexSpec | None
    left: Any = _MISSING

    def __rmatmul__(self, left: Any) -> "_EqDefinitionOperator":
        """Return a partial operator that has captured the function head."""

        if self.left is not _MISSING:
            raise TypeError(f"Incomplete EqDef expression near {self!r}.")
        return type(self)(
            args=self.args,
            indices=self.indices,
            indexed=self.indexed,
            args_were_supplied=self.args_were_supplied,
            name=self.name,
            latex=self.latex,
            left=left,
        )

    def __matmul__(self, definition: Any) -> Any:
        """Create the named symbolic object from the captured left operand."""

        return _build_eqdef_definition(
            self.left,
            definition,
            args=self.args,
            indices=self.indices,
            indexed=self.indexed,
            args_were_supplied=self.args_were_supplied,
            name=self.name,
            latex=self.latex,
        )

    def __repr__(self) -> str:
        """Return an interactive representation of the operator state."""

        if self.left is _MISSING:
            return '<EqDef: use as Function("F") @ EqDef(...) @ definition>'
        return "<EqDef: waiting for definition expression>"


@dataclass(frozen=True)
class _EqDefBuildContext:
    """Private build request passed from ``EqDef`` to left-operand hooks."""

    definition: Any
    args: tuple[sympy.Symbol, ...]
    indices: tuple[sympy.Symbol, ...]
    indexed: bool
    args_were_supplied: bool
    name: str | None
    latex: LatexSpec | None

    def build_named_function(
        self,
        left: Any,
        *,
        args: tuple[sympy.Symbol, ...] | None = None,
        indices: tuple[sympy.Symbol, ...] | None = None,
        indexed: bool | None = None,
    ) -> sympy.FunctionClass:
        """Build the ``NamedFunction`` result for a resolved function left side."""

        return _build_eqdef_named_function(
            left,
            self.definition,
            args=self.args if args is None else args,
            indices=self.indices if indices is None else indices,
            indexed=self.indexed if indexed is None else indexed,
            name=self.name,
            latex=self.latex,
        )

    def build_named_expression(
        self,
        left: Any,
    ) -> "_NamedExpression | _NamedExpressionFamily":
        """Build the named-expression result for a resolved expression left side."""

        return _build_eqdef_named_expression(
            left,
            self.definition,
            args=self.args,
            indices=self.indices,
            indexed=self.indexed,
            args_were_supplied=self.args_were_supplied,
            name=self.name,
            latex=self.latex,
            hold=False,
        )

    def build_fallback(self, left: Any) -> Any:
        """Build an ``EqDef`` result using the built-in left-side resolver."""

        # Resolve function-shaped left operands first. The fallback keeps the
        # historical core behavior for left classes that do not publish the
        # private hook.
        function_left = _resolve_eqdef_function_left(
            left,
            args=self.args,
            args_were_supplied=self.args_were_supplied,
            indices=self.indices,
            indexed=self.indexed,
        )
        if function_left is not None:
            function_head, function_args, function_indices, function_indexed = (
                function_left
            )
            return self.build_named_function(
                function_head,
                args=function_args,
                indices=function_indices,
                indexed=function_indexed,
            )

        # Only function-shaped left operands can consume call-argument
        # placeholders. Everything else is a named-expression definition.
        if self.args_were_supplied:
            raise ValueError(
                "EqDef args can be used only with function-head left operands."
            )
        return self.build_named_expression(left)


# EqDef generated-callable bridge

class _EqDefinitionCallable:
    """Callable symbolic source generated by ``EqDef``."""

    def __init__(
        self,
        *,
        function_name: str,
        definition: Any,
        args: tuple[sympy.Symbol, ...],
        indices: tuple[sympy.Symbol, ...],
        indexed: bool,
    ) -> None:
        self.__name__ = function_name
        self.__module__ = __name__
        self.__signature__ = _make_eqdef_signature(args, indexed=indexed)
        self._definition = definition
        self._args = args
        self._indices = indices
        self._indexed = indexed
        self._mt_help = {
            "path": PurePosixPath("library/NamedFunction"),
            "anchor": None,
            "label": "NamedFunction",
        }

    def __call__(self, *call_args: Any) -> sympy.Basic:
        """Return the authored expression with formal placeholders replaced."""

        if len(call_args) != len(self._args) + int(self._indexed):
            raise TypeError(
                f"{self.__name__} expected {len(self._args)} call arguments"
                f"{' plus indices' if self._indexed else ''}."
            )

        # Map ordinary formal arguments directly to the values supplied in the
        # applied SymPy function expression.
        replacements: dict[sympy.Basic, Any] = {
            placeholder: value
            for placeholder, value in zip(self._args, call_args[: len(self._args)])
        }

        # Indexed EqDef definitions bind explicit placeholders to the bracket
        # indices. The private ``idx`` tuple remains hidden from user syntax.
        if self._indexed and self._indices:
            index_values = tuple(call_args[-1])
            if len(index_values) != len(self._indices):
                raise ValueError(
                    f"{self.__name__} expected {len(self._indices)} indices, "
                    f"got {len(index_values)}."
                )
            replacements.update(
                {
                    placeholder: value
                    for placeholder, value in zip(self._indices, index_values)
                }
            )

        # Use exact structural replacement so formal symbols stand for
        # placeholders rather than algebraic substitution patterns.
        definition = self._definition
        if not isinstance(definition, sympy.Basic):
            definition = sympy.sympify(definition)
        return definition.xreplace(replacements)


# Named-expression runtime objects

@dataclass(frozen=True)
class _NamedExpressionDefinition:
    """Replacement rule for one named expression index arity."""

    definition: Any
    indices: tuple[sympy.Symbol, ...]


@dataclass(eq=False)
class _NamedExpressionFamily:
    """Indexable family returned by indexed symbolic ``EqDef`` definitions."""

    _iterable = False

    _mt_name: str
    _mt_latex: LatexSpec
    _mt_latex_name: str
    _mt_definitions: dict[int | None, _NamedExpressionDefinition]
    _mt_held: bool = False

    def __post_init__(self) -> None:
        """Attach the shared documentation topic to the family."""

        self._mt_help = {
            "path": PurePosixPath("library/NamedExpression"),
            "anchor": None,
            "label": "NamedExpression",
        }

    def __getitem__(self, key: Any) -> Any:
        """Return an indexed named expression from structural indices."""

        indices = _normalize_index_tuple(key)
        if self._mt_held:
            return _NamedExpression(self, indices)
        expanded = _evaluate_named_expression(self, indices)
        if expanded is _EQDEF_NOT_IMPLEMENTED:
            return _NamedExpression(self, indices)
        return expanded

    def _hold(self) -> "_NamedExpressionFamily":
        """Return a held family with the same definitions and display options."""

        if self._mt_held:
            return self
        return _clone_named_expression_family(self, hold=True)

    def _unhold(self) -> "_NamedExpressionFamily":
        """Return an auto-expanding family with the same definitions."""

        if not self._mt_held:
            return self
        return _clone_named_expression_family(self, hold=False)

    def _sympystr(self, printer: Any) -> str:
        """Return the plain-text rendering for this indexed-expression family."""

        return self._mt_name

    def _latex(self, printer: Any) -> str:
        """Return the LaTeX rendering for this indexed-expression family."""

        latex_spec = self._mt_latex
        if callable(latex_spec):
            return self._mt_latex_name
        if _latex_template_has_marker(latex_spec):
            return _render_latex_template(
                latex_spec,
                name=self._mt_latex_name,
                args=(),
                indices=(),
                indexed=False,
                printer=printer,
            )
        return latex_spec

    def __str__(self) -> str:
        """Return the user-facing plain-text form of the family."""

        return sympy.sstr(self)

    def __repr__(self) -> str:
        """Return a debug representation matching the user-facing form."""

        return str(self)


class _NamedExpression(sympy.AtomicExpr):
    """Opaque symbolic expression with an optional indexed family label."""

    is_commutative = True

    def __new__(
        cls,
        family: _NamedExpressionFamily,
        indices: tuple[sympy.Basic, ...] = (),
    ) -> "_NamedExpression":
        """Create a named expression application for a family and indices."""

        obj = sympy.AtomicExpr.__new__(cls)
        obj._math_toolkit_family = family
        obj._math_toolkit_indices = indices
        obj._mt_help = {
            "path": PurePosixPath("library/NamedExpression"),
            "anchor": None,
            "label": "NamedExpression",
        }
        return obj

    def _hashable_content(self) -> tuple[Any, ...]:
        """Return identity content for SymPy equality and hashing."""

        return (self._math_toolkit_family, self._math_toolkit_indices)

    @property
    def free_symbols(self) -> set[sympy.Basic]:
        """Return this opaque expression and any symbols in its indices."""

        result: set[sympy.Basic] = {self}
        for index in self._math_toolkit_indices:
            result.update(index.free_symbols)
        return result

    def _eval_subs(self, old: Any, new: Any):
        """Substitute through structural indices while keeping the head opaque."""

        if old == self:
            return new

        # Preserve indexed named-expression structure when a substitution
        # changes one of the structural index values.
        new_indices = tuple(
            index.subs(old, new) for index in self._math_toolkit_indices
        )
        if new_indices != self._math_toolkit_indices:
            return type(self)(self._math_toolkit_family, new_indices)
        return None

    def _unhold(self) -> Any:
        """Expose this held named expression's authored definition."""

        expanded = _evaluate_named_expression(
            self._math_toolkit_family,
            self._math_toolkit_indices,
        )
        if expanded is _EQDEF_NOT_IMPLEMENTED:
            return self
        return expanded

    def _hold(self) -> "_NamedExpression":
        """Return this held named expression unchanged."""

        return self

    def _sympystr(self, printer: Any) -> str:
        """Return the plain-text rendering for this named expression."""

        family = self._math_toolkit_family
        if self._math_toolkit_indices:
            return _render_indexed_head_sympystr(
                family._mt_name,
                self._math_toolkit_indices,
                printer,
            )
        return family._mt_name

    def _latex(self, printer: Any) -> str:
        """Return the LaTeX rendering for this named expression."""

        family = self._math_toolkit_family
        return _render_named_expression_latex(
            family,
            self._math_toolkit_indices,
            printer,
        )


# EqDef construction and left-side resolution

def _build_eqdef_definition(
    left: Any,
    definition: Any,
    *,
    args: tuple[sympy.Symbol, ...],
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
    args_were_supplied: bool,
    name: str | None,
    latex: LatexSpec | None,
) -> Any:
    """Return the named object defined by one ``EqDef`` expression."""

    if isinstance(left, list | tuple):
        return _build_eqdef_sequence_definition(
            left,
            definition,
            args=args,
            indices=indices,
            indexed=indexed,
            args_were_supplied=args_were_supplied,
            name=name,
            latex=latex,
        )
    if hasattr(left, "body") and type(left).__name__ == "HeldExpression":
        raise TypeError(
            "EqDef no longer accepts held left operands. Hold the completed "
            "definition result instead."
        )

    if left is _MISSING and name is None:
        raise ValueError(
            "EqDef without a left operand requires name=. Use Lambda for "
            "anonymous functions and Hold(...) for anonymous held formulas."
        )

    context = _EqDefBuildContext(
        definition=definition,
        args=args,
        indices=indices,
        indexed=indexed,
        args_were_supplied=args_were_supplied,
        name=name,
        latex=latex,
    )

    if left is _MISSING:
        return _build_eqdef_no_left(context)

    # Let runtime types own their left-side EqDef behavior. The built-in
    # resolver remains as a fallback for unhooked classes and invalid operands.
    handled = _dispatch_eqdef_left_hook(left, context)
    if handled is not _EQDEF_NOT_IMPLEMENTED:
        return handled
    return context.build_fallback(left)


def _build_eqdef_no_left(context: _EqDefBuildContext) -> Any:
    """Build an explicit ``EqDef(name=...) @ body`` definition."""

    if context.name is None:
        raise ValueError("EqDef without a left operand requires name=.")

    # Supplying args explicitly asks for a function definition, even when the
    # placeholder list is empty for a zero-argument function.
    if context.args_were_supplied:
        return context.build_named_function(sympy.Function(context.name))
    return context.build_named_expression(sympy.Symbol(context.name))


def _build_eqdef_sequence_definition(
    left: list[Any] | tuple[Any, ...],
    definition: Any,
    *,
    args: tuple[sympy.Symbol, ...],
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
    args_were_supplied: bool,
    name: str | None,
    latex: LatexSpec | None,
) -> tuple[Any, ...]:
    """Build flat sequence ``EqDef`` unpacking results."""

    if any(isinstance(item, list | tuple) for item in left):
        raise ValueError("EqDef sequence unpacking supports only flat left patterns.")
    if name is not None:
        raise ValueError("EqDef sequence unpacking cannot combine with name=.")

    if isinstance(definition, sympy.Tuple):
        values = tuple(definition)
    elif isinstance(definition, Sequence) and not isinstance(definition, str | bytes):
        values = tuple(definition)
    else:
        raise TypeError(
            "EqDef sequence unpacking requires a finite sequence right operand."
        )
    if len(values) != len(left):
        raise ValueError(
            f"EqDef sequence unpacking expected {len(left)} values, "
            f"got {len(values)}."
        )

    # Build each named definition through the same public options, preserving
    # order and letting the normal single-left validation report bad patterns.
    results: list[Any] = []
    for item, value in zip(left, values):
        results.append(
            _build_eqdef_definition(
                item,
                value,
                args=args,
                indices=indices,
                indexed=indexed,
                args_were_supplied=args_were_supplied,
                name=None,
                latex=latex,
            )
        )
    return tuple(results)


def _dispatch_eqdef_left_hook(left: Any, context: _EqDefBuildContext) -> Any:
    """Return a hook-built result, or the EqDef hook sentinel when absent."""

    # Generated SymPy function heads are class objects, so attribute lookup on
    # the class itself can accidentally find application-level hooks stored in
    # the class dict. Ask the metaclass for class hooks and ordinary instances
    # for bound-instance hooks so both surfaces dispatch correctly.
    if isinstance(left, FunctionClass):
        hook = getattr(type(left), _EQDEF_LEFT_HOOK, None)
    else:
        hook = getattr(left, _EQDEF_LEFT_HOOK, None)
    if not callable(hook):
        return _EQDEF_NOT_IMPLEMENTED

    # SymPy function heads are class objects whose metaclass can expose the
    # hook as an unbound function. Ordinary instances receive a bound method.
    if getattr(hook, "__self__", None) is None:
        result = hook(left, context)
    else:
        result = hook(context)
    if result is NotImplemented:
        return _EQDEF_NOT_IMPLEMENTED
    return result


def _resolve_eqdef_function_left(
    left: Any,
    *,
    args: tuple[sympy.Symbol, ...],
    args_were_supplied: bool,
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
) -> tuple[
    sympy.FunctionClass,
    tuple[sympy.Symbol, ...],
    tuple[sympy.Symbol, ...],
    bool,
] | None:
    """Return function-head metadata captured from an ``EqDef`` left side."""

    if isinstance(left, FunctionClass):
        return left, args, indices, indexed

    if isinstance(left, IndexedFunctionBase):
        return left.function_head, args, indices, indexed

    if isinstance(left, sympy.Function):
        function_head = left.func
        call_args = left.args
        index_placeholders = indices
        indexed_from_left = False

        # Indexed undefined heads from ``Function("F")[i](x)`` carry their
        # structural placeholders on the head rather than in a private final
        # call argument. Recover the public base head and placeholder tuple.
        if isinstance(function_head, IndexedUndefinedFunction):
            if indices:
                raise ValueError(
                    "EqDef indexed left operands already supply index "
                    "placeholders; omit indices=."
                )
            index_placeholders = _normalize_eqdef_left_index_placeholders(
                function_head.indices
            )
            function_head = function_head.base_function
            indexed_from_left = True

        if args_were_supplied:
            raise ValueError(
                "EqDef applied left operands already supply call argument "
                "placeholders; omit args=."
            )

        arg_placeholders = _normalize_eqdef_left_arg_placeholders(call_args)
        is_indexed = indexed or indexed_from_left or bool(index_placeholders)
        _validate_eqdef_placeholders(
            arg_placeholders,
            index_placeholders,
            indexed=is_indexed,
        )
        return (
            function_head,
            arg_placeholders,
            index_placeholders,
            is_indexed,
        )

    return None


def _build_eqdef_named_expression(
    left: Any,
    definition: Any,
    *,
    args: tuple[sympy.Symbol, ...],
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
    args_were_supplied: bool,
    name: str | None,
    latex: LatexSpec | None,
    hold: bool,
) -> _NamedExpression | _NamedExpressionFamily:
    """Return the named expression or indexed-expression family for ``EqDef``."""

    # Named expressions have no call arguments; reject that syntax before
    # resolving the more permissive symbol and IndexedBase left operands.
    if args_were_supplied:
        raise ValueError(
            "EqDef args can be used only with function-head left operands."
        )

    # Lower concrete matrix element patterns directly to a realized matrix.
    matrix = _build_eqdef_concrete_matrix(
        left,
        definition,
        indices=indices,
    )
    if matrix is not _EQDEF_NOT_IMPLEMENTED:
        return matrix

    # Resolve every accepted expression-left shape to one family and the index
    # placeholders that select the definition case being added or cleared.
    family, index_placeholders, returns_family, existing = (
        _resolve_eqdef_expression_left(
            left,
            indices=indices,
            indexed=indexed,
            name_override=name,
            latex=latex,
            hold=hold,
        )
    )
    definition_key = (
        _GENERAL_INDEX_ARITY
        if indexed and not index_placeholders
        else len(index_placeholders)
    )

    # ``None`` means removal for an existing definition case. It is not a
    # constructor for an empty named expression.
    if definition is None:
        if not existing:
            raise ValueError(
                "EqDef(... ) @ None can clear only an existing named expression "
                "definition."
            )
        family._mt_definitions.pop(definition_key, None)
        if returns_family:
            return family
        return _NamedExpression(family)

    # Reject duplicate exact dispatch surfaces. Specific index counts may
    # coexist with and override a general indexed fallback.
    if definition_key in family._mt_definitions:
        label = _named_expression_definition_label(definition_key)
        raise ValueError(
            f"Named expression {family._mt_name!r} already has a "
            f"definition for {label}."
        )

    family._mt_definitions[definition_key] = _NamedExpressionDefinition(
        definition=definition,
        indices=index_placeholders,
    )
    if returns_family:
        return family
    if not hold:
        return definition
    return _NamedExpression(family)


def _build_eqdef_named_function(
    left: Any,
    definition: Any,
    *,
    args: tuple[sympy.Symbol, ...],
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
    name: str | None,
    latex: LatexSpec | None,
) -> sympy.FunctionClass:
    """Return the generated ``NamedFunction`` class for an ``EqDef`` expression."""

    if not isinstance(left, FunctionClass):
        raise TypeError(
            "EqDef left operand must be a SymPy function head, such as "
            'Function("F").'
        )

    if definition is None:
        raise ValueError(
            "EqDef(... ) @ None can clear only an existing NamedFunction definition."
        )

    function_name = validate_name(name) if name is not None else validate_name(left.__name__)
    symbolic_callable = _EqDefinitionCallable(
        function_name=function_name,
        definition=definition,
        args=args,
        indices=indices,
        indexed=indexed,
    )
    return NamedFunction(
        symbolic_callable,
        name=function_name,
        latex=latex,
        index_count=len(indices) if indices else None,
    )


# EqDef named-expression left resolution

def _build_eqdef_concrete_matrix(
    left: Any,
    definition: Any,
    *,
    indices: tuple[sympy.Symbol, ...],
) -> Any:
    """Return a concrete matrix for a ``MatrixSymbol(...)[i, j]`` left operand."""

    if not isinstance(left, MatrixElement):
        return _EQDEF_NOT_IMPLEMENTED

    base, *matrix_indices = left.args
    if not isinstance(base, MatrixSymbol):
        return _EQDEF_NOT_IMPLEMENTED
    if definition is None:
        raise ValueError(
            "EqDef(... ) @ None can clear only an existing named expression "
            "definition."
        )
    if indices:
        raise ValueError(
            "EqDef indexed left operands already supply index placeholders; "
            "omit indices=."
        )

    # Validate the placeholder indices and require concrete matrix dimensions
    # before we realize the symbolic entry rule into a dense matrix.
    row_placeholder, col_placeholder = _normalize_eqdef_left_index_placeholders(
        tuple(matrix_indices)
    )
    row_count = _coerce_eqdef_matrix_dimension(base.shape[0])
    col_count = _coerce_eqdef_matrix_dimension(base.shape[1])

    # Fill every concrete matrix slot by substituting the row and column
    # placeholders with the coordinates requested by the Matrix constructor.
    body = definition if isinstance(definition, sympy.Basic) else sympy.sympify(definition)
    return sympy.Matrix(
        row_count,
        col_count,
        lambda row, col: body.xreplace(
            {
                row_placeholder: sympy.Integer(row),
                col_placeholder: sympy.Integer(col),
            }
        ),
    )


def _resolve_eqdef_expression_left(
    left: Any,
    *,
    indices: tuple[sympy.Symbol, ...],
    indexed: bool,
    name_override: str | None,
    latex: LatexSpec | None,
    hold: bool,
) -> tuple[
    _NamedExpressionFamily,
    tuple[sympy.Symbol, ...],
    bool,
    bool,
]:
    """Return expression-family metadata captured from an ``EqDef`` left side."""

    # Existing named-expression objects can receive additional index-count
    # definitions. Indexed applications use their indices as placeholders.
    if isinstance(left, _NamedExpression):
        family = left._math_toolkit_family
        if not family._mt_held and hold:
            family = family._hold()
        _validate_existing_named_expression_options(
            family,
            name_override=name_override,
            latex=latex,
        )
        if indices and left._math_toolkit_indices:
            raise ValueError(
                "EqDef indexed left operands already supply index placeholders; "
                "omit indices=."
            )
        index_placeholders = indices or _normalize_eqdef_left_index_placeholders(
            left._math_toolkit_indices
        )
        return family, index_placeholders, bool(index_placeholders or indexed), True

    # Families are binders, so defining a concrete case through them requires
    # explicit placeholders unless the user asks for a general indexed fallback.
    if isinstance(left, _NamedExpressionFamily):
        if hold and not left._mt_held:
            left = left._hold()
        _validate_existing_named_expression_options(
            left,
            name_override=name_override,
            latex=latex,
        )
        if not indices and not indexed:
            raise ValueError(
                "EqDef on an indexed expression family requires indices= "
                "or indexed=True."
            )
        return left, indices, True, True

    if isinstance(left, sympy.Indexed):
        if indices:
            raise ValueError(
                "EqDef indexed left operands already supply index placeholders; "
                "omit indices=."
            )
        function_name = _resolve_eqdef_indexed_base_name(
            left.base,
            name_override=name_override,
        )
        index_placeholders = _normalize_eqdef_left_index_placeholders(left.indices)
        family = _make_named_expression_family(function_name, latex=latex, hold=hold)
        return family, index_placeholders, True, False

    if isinstance(left, sympy.IndexedBase):
        function_name = _resolve_eqdef_indexed_base_name(
            left,
            name_override=name_override,
        )
        if not indices and not indexed:
            raise ValueError(
                "EqDef on an IndexedBase requires indices= or indexed=True."
            )
        family = _make_named_expression_family(function_name, latex=latex, hold=hold)
        return family, indices, True, False

    if isinstance(left, sympy.Symbol):
        function_name = (
            validate_name(name_override)
            if name_override is not None
            else validate_name(left.name)
        )
        family = _make_named_expression_family(function_name, latex=latex, hold=hold)
        return family, indices, bool(indices or indexed), False

    raise TypeError(
        "EqDef left operand must be a SymPy function head, symbol, indexed "
        "symbol, or indexed base."
    )


def _coerce_eqdef_matrix_dimension(value: Any) -> int:
    """Return one concrete matrix dimension or raise a helpful EqDef error."""

    # Prefer true integer-like objects so symbolic names such as ``n`` do not
    # quietly pass through as underspecified concrete matrix sizes.
    try:
        return operator.index(value)
    except TypeError:
        pass

    sympified = sympy.sympify(value)
    if sympified.is_number and sympified.is_integer:
        return int(sympified)

    raise ValueError(
        "Cannot create concrete matrix with unspecified dimensions."
    )


def _make_named_expression_family(
    name: str,
    *,
    latex: LatexSpec | None,
    hold: bool = False,
) -> _NamedExpressionFamily:
    """Return an empty named-expression family for a canonical name."""

    latex_name = atom_name_to_latex(name) if latex is None else latex
    normalized_latex_name = atom_name_to_latex(name)
    return _NamedExpressionFamily(
        _mt_name=name,
        _mt_latex=latex_name,
        _mt_latex_name=normalized_latex_name,
        _mt_definitions={},
        _mt_held=hold,
    )


def _clone_named_expression_family(
    family: _NamedExpressionFamily,
    *,
    hold: bool,
) -> _NamedExpressionFamily:
    """Return a named-expression family with the same definitions and mode."""

    return _NamedExpressionFamily(
        _mt_name=family._mt_name,
        _mt_latex=family._mt_latex,
        _mt_latex_name=family._mt_latex_name,
        _mt_definitions=dict(family._mt_definitions),
        _mt_held=hold,
    )


def _validate_existing_named_expression_options(
    family: _NamedExpressionFamily,
    *,
    name_override: str | None,
    latex: LatexSpec | None,
) -> None:
    """Validate options supplied while extending an existing named expression."""

    if (
        name_override is not None
        and validate_name(name_override) != family._mt_name
    ):
        raise ValueError(
            f"Cannot add definition named {name_override!r} to named "
            f"expression {family._mt_name!r}."
        )
    if latex is not None and latex != family._mt_latex:
        raise ValueError(
            f"Named expression {family._mt_name!r} already has a "
            "different LaTeX representation."
        )


def _resolve_eqdef_indexed_base_name(
    base: sympy.IndexedBase,
    *,
    name_override: str | None,
) -> str:
    """Return the generated expression name for an indexed base."""

    if name_override is not None:
        return validate_name(name_override)
    label = getattr(base, "label", None)
    if isinstance(label, sympy.Symbol):
        return validate_name(label.name)
    return validate_name(str(label))


# EqDef left hook installation

def _eqdef_function_left_hook(left: Any, context: _EqDefBuildContext) -> Any:
    """Build an ``EqDef`` result for function-like left operands."""

    resolved = _resolve_eqdef_function_left(
        left,
        args=context.args,
        args_were_supplied=context.args_were_supplied,
        indices=context.indices,
        indexed=context.indexed,
    )
    if resolved is None:
        return NotImplemented

    function_head, args, indices, indexed = resolved
    return context.build_named_function(
        function_head,
        args=args,
        indices=indices,
        indexed=indexed,
    )


def _eqdef_named_expression_left_hook(
    left: Any,
    context: _EqDefBuildContext,
) -> _NamedExpression | _NamedExpressionFamily:
    """Build an ``EqDef`` result for expression-like left operands."""

    return context.build_named_expression(left)


def _install_eqdef_left_hooks() -> None:
    """Attach private ``EqDef`` hooks to the built-in left operand classes."""

    sympy.Symbol.__mt_eqdef__ = _eqdef_named_expression_left_hook
    sympy.Indexed.__mt_eqdef__ = _eqdef_named_expression_left_hook
    sympy.IndexedBase.__mt_eqdef__ = _eqdef_named_expression_left_hook
    _NamedExpression.__mt_eqdef__ = _eqdef_named_expression_left_hook
    _NamedExpressionFamily.__mt_eqdef__ = _eqdef_named_expression_left_hook


_install_eqdef_left_hooks()


def _normalize_eqdef_left_index_placeholders(
    values: tuple[Any, ...],
) -> tuple[sympy.Symbol, ...]:
    """Return symbolic placeholders captured from an indexed left operand."""

    placeholders: list[sympy.Symbol] = []
    for value in values:
        if not isinstance(value, sympy.Symbol):
            raise TypeError(
                "EqDef indexed left operands must use Symbol index placeholders."
            )
        validate_name(value.name)
        placeholders.append(value)
    _validate_eqdef_placeholders((), tuple(placeholders), indexed=True)
    return tuple(placeholders)


def _normalize_eqdef_left_arg_placeholders(
    values: tuple[Any, ...],
) -> tuple[sympy.Symbol, ...]:
    """Return symbolic placeholders captured from an applied function operand."""

    placeholders: list[sympy.Symbol] = []
    for value in values:
        if not isinstance(value, sympy.Symbol):
            raise TypeError(
                "EqDef applied left operands must use Symbol argument "
                "placeholders."
            )
        validate_name(value.name)
        placeholders.append(value)
    _validate_eqdef_placeholders(tuple(placeholders), (), indexed=False)
    return tuple(placeholders)


def _named_expression_definition_label(index_count: int | None) -> str:
    """Return a readable named-expression dispatch label for errors."""

    if index_count == 0:
        return "no indices"
    if index_count is _GENERAL_INDEX_ARITY:
        return "indices"
    return f"{index_count} indices"


def _normalize_eqdef_placeholders(
    value: Any,
    option_name: str,
) -> tuple[sympy.Symbol, ...]:
    """Return normalized symbolic placeholders for an ``EqDef`` option."""

    if value is None:
        return ()

    # Accept the common single-placeholder form ``args=x`` while preserving
    # tuple/list forms for multiple placeholders.
    values = tuple(value) if isinstance(value, list | tuple) else (value,)
    placeholders: list[sympy.Symbol] = []
    for item in values:
        if isinstance(item, sympy.Symbol):
            validate_name(item.name)
            placeholders.append(item)
            continue
        if isinstance(item, str):
            placeholders.append(sympy.Symbol(validate_name(item.strip())))
            continue
        raise TypeError(
            f"EqDef {option_name} placeholders must be SymPy Symbols or strings."
        )
    return tuple(placeholders)


def _validate_eqdef_placeholders(
    args: tuple[sympy.Symbol, ...],
    indices: tuple[sympy.Symbol, ...],
    *,
    indexed: bool,
) -> None:
    """Validate placeholder names before building an ``EqDef`` callable."""

    # Signature names and substitution placeholders must be unambiguous across
    # both call arguments and bracket-index placeholders.
    seen_names: set[str] = set()
    for placeholder in (*args, *indices):
        if placeholder.name in seen_names:
            raise ValueError(
                f"EqDef placeholder {placeholder.name!r} is duplicated."
            )
        seen_names.add(placeholder.name)

    # Indexed ``NamedFunction`` definitions reserve a final implementation
    # parameter named ``idx``; EqDef users bind index placeholders explicitly.
    if indexed and "idx" in seen_names:
        raise ValueError(
            "EqDef indexed definitions reserve the placeholder name 'idx'. "
            "Use an explicit index placeholder such as n, i, or j."
        )


def _make_eqdef_signature(
    args: tuple[sympy.Symbol, ...],
    *,
    indexed: bool,
) -> inspect.Signature:
    """Return the callable signature presented to ``NamedFunction``."""

    parameters = [
        inspect.Parameter(
            placeholder.name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for placeholder in args
    ]
    if indexed:
        parameters.append(
            inspect.Parameter("idx", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
    return inspect.Signature(parameters)


def _normalize_index_tuple(key: Any) -> tuple[sympy.Basic, ...]:
    """Return a nonempty tuple of structural indices from subscript syntax."""

    values = key if isinstance(key, tuple) else (key,)

    # Normalize user-facing subscript values in one pass, so the accepted
    # scalar cases and the rejected slice case are visible at the call boundary.
    indices: list[sympy.Basic] = []
    for value in values:
        if isinstance(value, slice):
            raise ValueError("Slices are not supported as NamedFunction indices.")
        if isinstance(value, sympy.Basic):
            indices.append(value)
            continue
        if isinstance(value, Integral) and not isinstance(value, bool):
            indices.append(sympy.Integer(int(value)))
            continue
        if isinstance(value, str):
            text = value.strip()
            validate_name(text)
            indices.append(sympy.Symbol(text))
            continue
        sympified = sympy.sympify(value)
        if isinstance(sympified, list | tuple):
            indices.append(sympy.Tuple(*sympified))
            continue
        indices.append(sympified)

    if not indices:
        raise ValueError("Indexed NamedFunction heads require at least one index.")
    return tuple(indices)


# Definition dispatch

def _evaluate_named_expression(
    family: _NamedExpressionFamily,
    indices: tuple[Any, ...],
) -> Any:
    """Return an authored named-expression body or the EqDef sentinel."""

    definitions = family._mt_definitions
    definition = definitions.get(len(indices))
    if definition is None:
        definition = definitions.get(_GENERAL_INDEX_ARITY)
    if definition is None:
        return _EQDEF_NOT_IMPLEMENTED

    # Replace formal index placeholders with the structural indices from this
    # expression application.
    replacements: dict[sympy.Basic, Any] = {}
    if definition.indices:
        if len(indices) != len(definition.indices):
            return _EQDEF_NOT_IMPLEMENTED
        replacements = {
            placeholder: value
            for placeholder, value in zip(definition.indices, indices)
        }

    body = definition.definition
    if not isinstance(body, sympy.Basic):
        body = sympy.sympify(body)
    return body.xreplace(replacements)


def _render_named_expression_latex(
    family: _NamedExpressionFamily,
    indices: tuple[Any, ...],
    printer: Any,
) -> str:
    """Return the LaTeX rendering for a named expression."""

    latex_spec = family._mt_latex
    if callable(latex_spec):
        return _render_latex_callable(
            latex_spec,
            args=(),
            indices=indices,
            indexed=bool(indices),
            printer=printer,
            label=f"Named expression {family._mt_name!r} latex",
        )
    if indices:
        if _latex_template_has_marker(latex_spec):
            return _render_latex_template(
                latex_spec,
                name=family._mt_latex_name,
                args=(),
                indices=indices,
                indexed=True,
                printer=printer,
            )
        base = (
            f"{{{latex_spec}}}"
            if "_" in family._mt_name
            else latex_spec
        )
        rendered_indices = ", ".join(printer._print(item) for item in indices)
        return f"{base}_{{{rendered_indices}}}"
    if _latex_template_has_marker(latex_spec):
        return _render_latex_template(
            latex_spec,
            name=family._mt_latex_name,
            args=(),
            indices=(),
            indexed=False,
            printer=printer,
        )
    return latex_spec


def _render_indexed_head_sympystr(
    name: str,
    indices: tuple[Any, ...],
    printer: Any,
) -> str:
    """Return the plain-text rendering for an indexed named-expression head."""

    rendered_indices = ", ".join(printer._print(item) for item in indices)
    return f"{name}[{rendered_indices}]"


def _validate_latex_option(latex: LatexSpec | None, label: str) -> None:
    """Validate a public ``latex=`` option before storing it."""

    if latex is not None and not isinstance(latex, str) and not callable(latex):
        raise TypeError(f"{label} option must be a string, callable, or None.")


def _latex_template_has_marker(template: str) -> bool:
    """Return whether a LaTeX value contains an unescaped template marker."""

    for index, char in enumerate(template):
        if char == "#" and not _is_escaped_latex_template_hash(template, index):
            return True
    return False


def _is_escaped_latex_template_hash(template: str, index: int) -> bool:
    """Return whether a hash character is escaped by an odd backslash run."""

    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and template[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _render_latex_template(
    template: str,
    *,
    name: str,
    args: tuple[Any, ...],
    indices: tuple[Any, ...],
    indexed: bool,
    printer: Any,
) -> str:
    """Return a LaTeX template with argument and index slots substituted."""

    rendered: list[str] = []
    cursor = 0

    # Parse template markers directly so named-expression rendering supports
    # the same visible ``#0`` / ``#{idx:*}`` contract as NamedFunction.
    while cursor < len(template):
        char = template[cursor]
        if char != "#" or _is_escaped_latex_template_hash(template, cursor):
            rendered.append(char)
            cursor += 1
            continue

        if cursor + 1 >= len(template):
            raise ValueError("Invalid EqDef LaTeX template placeholder '#'.")
        marker = template[cursor + 1]

        if marker == "*":
            rendered.append(
                _render_latex_template_slot(
                    "arg",
                    "*",
                    args,
                    indices,
                    indexed,
                    printer,
                )
            )
            cursor += 2
            continue

        if marker.isdigit():
            end = cursor + 2
            while end < len(template) and template[end].isdigit():
                end += 1
            selector = template[cursor + 1 : end]
            if selector == "0":
                rendered.append(name)
                cursor = end
                continue
            if len(selector) != 1:
                raise ValueError(
                    "EqDef LaTeX template aliases support only #1 through #9; "
                    "use #{arg:10} for multi-digit arguments."
                )
            rendered.append(
                _render_latex_template_slot(
                    "arg",
                    selector,
                    args,
                    indices,
                    indexed,
                    printer,
                )
            )
            cursor = end
            continue

        if marker == "{":
            end = template.find("}", cursor + 2)
            if end == -1:
                raise ValueError(
                    "Invalid EqDef LaTeX template placeholder; expected a "
                    "closing '}'."
                )
            placeholder = template[cursor + 2 : end].strip()
            if ":" not in placeholder:
                raise ValueError(
                    "EqDef LaTeX template placeholders must use #{arg:n}, "
                    "#{arg:*}, #{idx:n}, or #{idx:*}."
                )
            kind, selector = (part.strip() for part in placeholder.split(":", 1))
            rendered.append(
                _render_latex_template_slot(
                    kind,
                    selector,
                    args,
                    indices,
                    indexed,
                    printer,
                )
            )
            cursor = end + 1
            continue

        raise ValueError(
            f"Invalid EqDef LaTeX template placeholder near '#{marker}'."
        )

    return "".join(rendered)


def _render_latex_callable(
    renderer: Callable[..., str],
    *,
    args: tuple[Any, ...],
    indices: tuple[Any, ...],
    indexed: bool,
    printer: Any,
    label: str,
) -> str:
    """Return LaTeX text from a callable ``latex=`` option."""

    # Render symbolic values before invoking the user callback so its contract
    # stays purely textual and matches NamedFunction's public behavior.
    rendered_args = tuple(printer._print(arg) for arg in args)
    call_args: tuple[Any, ...] = rendered_args
    if indexed:
        rendered_indices = tuple(printer._print(index) for index in indices)
        call_args = (*rendered_args, rendered_indices)

    try:
        result = renderer(*call_args)
    except TypeError as exc:
        raise TypeError(
            f"{label} callable must accept the rendered call arguments"
            + (" plus a final idx tuple." if indexed else ".")
        ) from exc
    if not isinstance(result, str):
        raise TypeError(f"{label} callable must return a LaTeX string.")
    return result


def _render_latex_template_slot(
    kind: str,
    selector: str,
    args: tuple[Any, ...],
    indices: tuple[Any, ...],
    indexed: bool,
    printer: Any,
) -> str:
    """Return one rendered argument or index slot from a LaTeX template."""

    if kind == "arg":
        values = args
        label = "call argument"
    elif kind == "idx":
        if not indexed:
            raise ValueError(
                "EqDef LaTeX template uses an index placeholder, but the "
                "rendered object is not indexed."
            )
        values = indices
        label = "index"
    else:
        raise ValueError(
            "EqDef LaTeX template placeholders must use 'arg' or 'idx'."
        )

    if selector == "*":
        return ", ".join(printer._print(value) for value in values)

    try:
        position = int(selector)
    except ValueError as exc:
        raise ValueError(
            "EqDef LaTeX template selectors must be positive integers or '*'."
        ) from exc
    if position < 1:
        raise ValueError(
            "EqDef LaTeX template selectors are one-based; use 1 or greater."
        )
    if position > len(values):
        raise ValueError(
            f"EqDef LaTeX template references {label} {position}, but only "
            f"{len(values)} are available."
        )
    return printer._print(values[position - 1])
