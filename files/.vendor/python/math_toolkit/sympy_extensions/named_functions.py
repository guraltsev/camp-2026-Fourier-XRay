"""Create symbolic named SymPy function classes from Python definitions.

Every NamedFunction has a fixed number of ordinary arguments and a fixed
number of structural indices, determined directly from its Python signature.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import PurePosixPath
from typing import Any

import sympy
from sympy.core.function import FunctionClass

from .atom_latex_representation import atom_name_to_latex

__all__ = ["NamedFunction", "IndexedFunctionHead"]

LatexSpec = str | Callable[[Any], str]
_NAMED_FUNCTION_DOC = {
    "path": PurePosixPath("library/NamedFunction"),
    "anchor": None,
    "label": "NamedFunction",
}


def NamedFunction(
    obj: Any = None,
    *,
    name: str | None = None,
    latex: LatexSpec | None = None,
    index_count: int | None = None,
):
    """Decorator to turn a Python function into a symbolic SymPy function class.

    Parameters
    ----------
    obj : Any, optional
        Callable to convert into a symbolic SymPy function class.
    name : str | None, optional
        Public function name to expose instead of the decorated callable's
        ``__name__``.
    latex : str | Callable[[Any], str] | None, optional
        LaTeX renderer for held symbolic applications. A string is used as the
        rendered function head. A callable receives the rendered held object
        and must return a LaTeX string.
    index_count : int | None, optional
        Fixed number of structural indices for internal builders such as
        ``EqDef``.

    Returns
    -------
    sympy.FunctionClass
        Generated SymPy function class with fixed call arity.

    Raises
    ------
    TypeError
        If the decorated object is not callable, or if ``latex`` is not a
        string, callable, or ``None``.
    ValueError
        If the generated function name is missing or invalid.
    """
    def decorator(target: Any) -> sympy.FunctionClass:
        if inspect.isclass(target) or not callable(target):
            raise TypeError("NamedFunction can decorate only callables.")
        _validate_latex_option(latex)
        
        func_name = name or getattr(target, "__name__", None)
        if not func_name:
            raise ValueError("NamedFunction objects must provide a name.")

        sig = inspect.signature(target)
        parameters = list(sig.parameters.values())
        
        is_indexed = len(parameters) > 0 and parameters[-1].name == "idx"
        
        if is_indexed:
            arg_params = parameters[:-1]
            inferred_index_count = index_count
            if inferred_index_count is None:
                annotation = parameters[-1].annotation
                inferred_index_count = annotation if isinstance(annotation, int) else 1
        else:
            arg_params = parameters
            inferred_index_count = 0

        for p in arg_params:
            if p.kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
                raise ValueError("NamedFunction parameters must be standard positional arguments.")
            if p.default is not inspect.Parameter.empty:
                raise ValueError("NamedFunction parameters must not define defaults.")

        arity = len(arg_params)
        arg_symbols = tuple(sympy.Symbol(p.name) for p in arg_params)
        idx_symbols = tuple(sympy.Symbol(f"i{k}") for k in range(inferred_index_count)) if is_indexed else ()

        if is_indexed:
            definition = target(*arg_symbols, idx_symbols)
        else:
            definition = target(*arg_symbols)

        class_dict = {
            "__module__": getattr(target, "__module__", __name__),
            "_mt_help": getattr(target, "_mt_help", _NAMED_FUNCTION_DOC),
            "_mt_name": func_name,
            "_mt_latex_name": atom_name_to_latex(func_name),
            "_mt_latex": latex,
            "_arity": arity,
            "_index_count": inferred_index_count,
            "_user_signature": inspect.Signature(arg_params),
            "_arg_symbols": arg_symbols,
            "_idx_symbols": idx_symbols,
            "_definition": sympy.sympify(definition),
            "_is_held": False,
            "_iterable": False,
            "_latex": _named_function_latex,
            "__doc__": getattr(target, "__doc__", None),
            "_sympystr": _named_function_sympystr,
            "matches": _named_function_matches,
            "_unhold": _UnholdDescriptor(),
        }

        return _NamedFunctionMeta(func_name, (sympy.Function,), class_dict)

    if obj is None:
        return decorator
    return decorator(obj)


NamedFunction._mt_help = _NAMED_FUNCTION_DOC


class _NamedFunctionMeta(FunctionClass):
    """Metaclass for generated symbolic fixed-signature functions."""

    @property
    def __signature__(cls) -> inspect.Signature:
        """Return the public callable signature for non-indexed function classes."""

        if cls._index_count > 0:
            noun = "index" if cls._index_count == 1 else "indices"
            raise TypeError(
                f"{cls._mt_name} is an indexed function head and requires "
                f"{cls._index_count} {noun} "
                f"(via {cls._mt_name}[i1, i2, ...]) before it becomes callable."
            )
        return cls._user_signature

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._index_count > 0 and args and isinstance(args[-1], _IndexPayload):
            indices = args[-1].args
            call_args = args[:-1]
        else:
            indices = ()
            call_args = args

        if len(call_args) != cls._arity:
            raise TypeError(f"{cls.__name__} expects exactly {cls._arity} arguments, got {len(call_args)}.")
        
        if cls._index_count > 0 and len(indices) != cls._index_count:
            raise ValueError(f"{cls.__name__} expects exactly {cls._index_count} indices, got {len(indices)}.")

        if cls._is_held:
            return super().__call__(*args, **kwargs)

        replacements = dict(zip(cls._arg_symbols, call_args))
        if cls._index_count > 0:
            replacements.update(zip(cls._idx_symbols, indices))
        return cls._definition.xreplace(replacements)

    def __getitem__(cls, key: Any) -> IndexedFunctionHead:
        if cls._index_count == 0:
            raise TypeError(f"Function {cls.__name__} does not accept indices.")
        indices = key if isinstance(key, tuple) else (key,)
        if len(indices) != cls._index_count:
            raise ValueError(f"{cls.__name__} expects exactly {cls._index_count} indices, got {len(indices)}.")
        return IndexedFunctionHead(cls, *[sympy.sympify(i) for i in indices])

    def _hold(cls) -> sympy.FunctionClass:
        if cls._is_held:
            return cls
        return _clone_class(cls, held=True)


class _UnholdDescriptor:
    """Safely maps holding.py's zero-argument getattr() requirements to the correct execution."""
    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            if not getattr(owner, "_is_held", False):
                return lambda: owner
            return lambda: _clone_class(owner, held=False)
        
        def unhold_instance():
            if not getattr(instance.func, "_is_held", False):
                return instance
            # Retrieve the autoevaluating variant and feed the args back into it
            unheld_cls = instance.func._unhold()
            return unheld_cls(*instance.args)
            
        return unhold_instance


class IndexedFunctionHead(sympy.Basic):
    """An unapplied function head representing an indexed function binding."""
    _iterable = False

    def __new__(cls, base_function: sympy.FunctionClass, *indices: sympy.Basic) -> IndexedFunctionHead:
        obj = sympy.Basic.__new__(cls, base_function, *indices)
        obj._mt_help = getattr(base_function, "_mt_help", _NAMED_FUNCTION_DOC)
        return obj

    @property
    def base_function(self) -> sympy.FunctionClass:
        return self.args[0]

    @property
    def indices(self) -> tuple[sympy.Basic, ...]:
        return self.args[1:]

    @property
    def __signature__(self) -> inspect.Signature:
        """Return the ordinary argument signature for this indexed callable head."""

        return self.base_function._user_signature

    def __call__(self, *call_args: Any) -> sympy.Function:
        return self.base_function(*call_args, _IndexPayload(*self.indices))

    def _hold(self) -> IndexedFunctionHead:
        return IndexedFunctionHead(self.base_function._hold(), *self.indices)

    def _unhold(self) -> IndexedFunctionHead:
        return IndexedFunctionHead(self.base_function._unhold(), *self.indices)

    def _sympystr(self, printer: Any) -> str:
        idx_str = ", ".join(printer._print(i) for i in self.indices)
        return f"{self.base_function._mt_name}[{idx_str}]"

    def _latex(self, printer: Any) -> str:
        idx_latex = ", ".join(printer._print(i) for i in self.indices)
        dots = ", ".join(r"\cdot" for _ in range(self.base_function._arity))
        return f"{self.base_function._mt_name}_{{{idx_latex}}}({dots})"


class _IndexPayload(sympy.Basic):
    """Internal marker to transport structural indices inside SymPy args safely."""
    def __new__(cls, *indices: sympy.Basic) -> _IndexPayload:
        return sympy.Basic.__new__(cls, *indices)


def _clone_class(cls: _NamedFunctionMeta, held: bool) -> _NamedFunctionMeta:
    """Clones the metaclass definitions explicitly preserving the held state."""
    class_dict = {
        "__module__": cls.__module__,
        "_mt_help": getattr(cls, "_mt_help", _NAMED_FUNCTION_DOC),
        "_mt_name": cls._mt_name,
        "_mt_latex_name": cls._mt_latex_name,
        "_mt_latex": cls._mt_latex,
        "_arity": cls._arity,
        "_index_count": cls._index_count,
        "_user_signature": cls._user_signature,
        "_arg_symbols": cls._arg_symbols,
        "_idx_symbols": cls._idx_symbols,
        "_definition": cls._definition,
        "_is_held": held,
        "_iterable": False,
        "_latex": _named_function_latex,
        "__doc__": getattr(cls, "__doc__", None),
        "_sympystr": _named_function_sympystr,
        "matches": _named_function_matches,
        "_unhold": _UnholdDescriptor(),
    }
    return _NamedFunctionMeta(cls.__name__, (sympy.Function,), class_dict)


def _named_function_sympystr(self: sympy.Function, printer: Any) -> str:
    if self.args and isinstance(self.args[-1], _IndexPayload):
        indices = self.args[-1].args
        args = self.args[:-1]
        idx_str = ", ".join(printer._print(i) for i in indices)
        arg_str = ", ".join(printer._print(a) for a in args)
        return f"{self.func._mt_name}[{idx_str}]({arg_str})"
    
    arg_str = ", ".join(printer._print(a) for a in self.args)
    return f"{self.func._mt_name}({arg_str})"


def _named_function_latex(self: sympy.Function, printer: Any) -> str:
    has_indices = self.args and isinstance(self.args[-1], _IndexPayload)
    indices = self.args[-1].args if has_indices else ()
    args = self.args[:-1] if has_indices else self.args

    latex_spec = self.func._mt_latex
    if callable(latex_spec):
        return _render_latex_callable(latex_spec, self)
    if isinstance(latex_spec, str):
        return _render_named_function_head(
            latex_spec,
            args=args,
            indices=indices,
            indexed=bool(has_indices),
            printer=printer,
        )

    return _render_named_function_head(
        self.func._mt_latex_name,
        args=args,
        indices=indices,
        indexed=bool(has_indices),
        printer=printer,
    )


def _named_function_matches(
    self: sympy.Function,
    expr: Any,
    repl_dict: dict[Any, Any] | None = None,
    old: bool = False,
) -> dict[Any, Any] | None:
    """Match indexed NamedFunction calls across hidden indices and call args."""

    expr = sympy.sympify(expr)
    if not isinstance(expr, sympy.Function):
        return None

    self_args, self_indices = _split_named_function_application(self)
    expr_args, expr_indices = _split_named_function_application(expr)

    # Fall back to SymPy's ordinary matcher unless both calls share one
    # generated NamedFunction family and index layout.
    if (
        not hasattr(self.func, "_mt_name")
        or not hasattr(expr.func, "_mt_name")
        or self.func._mt_name != expr.func._mt_name
        or self.func._arity != expr.func._arity
        or self.func._index_count != expr.func._index_count
        or self.func._is_held != expr.func._is_held
        or len(self_indices) != len(expr_indices)
    ):
        return sympy.Basic.matches(self, expr, repl_dict=repl_dict, old=old)

    replacements = {} if repl_dict is None else repl_dict.copy()
    if self == expr:
        return replacements

    # Bind structural indices before visible call arguments so a failed index
    # match exits early and argument wildcards never hide the mismatch.
    replacements = _match_named_function_sequence(
        self_indices,
        expr_indices,
        replacements,
        old=old,
    )
    if replacements is None:
        return None

    if len(self_args) != len(expr_args):
        return None
    return _match_named_function_sequence(
        self_args,
        expr_args,
        replacements,
        old=old,
    )


def _split_named_function_application(
    value: sympy.Function,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return ordinary call arguments and structural indices for a call."""

    if value.args and isinstance(value.args[-1], _IndexPayload):
        return value.args[:-1], value.args[-1].args
    return value.args, ()


def _match_named_function_sequence(
    patterns: tuple[Any, ...],
    values: tuple[Any, ...],
    replacements: dict[Any, Any],
    *,
    old: bool,
) -> dict[Any, Any] | None:
    """Return wildcard bindings after matching two same-length sequences."""

    current = replacements
    for pattern, value in zip(patterns, values):
        if pattern == value:
            continue
        current = pattern.xreplace(current).matches(value, current, old=old)
        if current is None:
            return None
    return current


def _render_named_function_head(
    head_latex: str,
    *,
    args: tuple[Any, ...],
    indices: tuple[Any, ...],
    indexed: bool,
    printer: Any,
) -> str:
    """Return default function-application LaTeX from a rendered head label."""

    wrapped_head = f"{{{head_latex}}}"

    if not args and not indexed:
        return wrapped_head

    idx_latex = ""
    if indexed:
        rendered_indices = ", ".join(printer._print(item) for item in indices)
        idx_latex = f"_{{{rendered_indices}}}"
    arg_latex = ", ".join(printer._print(item) for item in args)
    return f"{wrapped_head}{idx_latex}({arg_latex})"


def _validate_latex_option(latex: LatexSpec | None) -> None:
    """Validate the public ``latex=`` option for ``NamedFunction``."""

    if latex is not None and not isinstance(latex, str) and not callable(latex):
        raise TypeError("NamedFunction latex option must be a string, callable, or None.")


def _render_latex_callable(renderer: Callable[[Any], str], value: Any) -> str:
    """Return LaTeX text from a callable ``latex=`` option."""

    try:
        result = renderer(value)
    except TypeError as exc:
        raise TypeError(
            "NamedFunction latex callable must accept the rendered object."
        ) from exc
    if not isinstance(result, str):
        raise TypeError("NamedFunction latex callable must return a LaTeX string.")
    return result
