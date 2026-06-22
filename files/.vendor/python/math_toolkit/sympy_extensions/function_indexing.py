from __future__ import annotations

from typing import Any

import sympy
from sympy import Basic, Tuple, sympify
from sympy.core.function import AppliedUndef, FunctionClass

__all__ = [
    "AppliedIndexedUndef",
    "IndexedFunctionBase",
    "IndexedFunction",
    "IndexedFunctionHead",
    "IndexedUndefinedFunction",
    "patch",
]

_PATCHED = False


class IndexedFunctionHead(Basic):
    """Represent an undefined function head with bound structural indices."""

    _iterable = False

    def __new__(
        cls,
        base_function: FunctionClass,
        *indices: Any,
    ) -> IndexedFunctionHead:
        sympy_indices = tuple(sympify(index) for index in indices)
        return Basic.__new__(cls, base_function, *sympy_indices)

    @property
    def base_function(self) -> FunctionClass:
        """Return the underlying unindexed SymPy function head."""

        return self.args[0]

    @property
    def indices(self) -> tuple[Basic, ...]:
        """Return the structural index tuple bound to this head."""

        return self.args[1:]

    def __call__(self, *call_args: Any) -> AppliedIndexedUndef:
        """Apply the indexed head to ordinary call arguments."""

        return AppliedIndexedUndef(self, *call_args)

    def __getitem__(self, key: Any) -> IndexedFunctionHead:
        """Append additional structural indices to this head."""

        new_indices = key if isinstance(key, tuple) else (key,)
        return IndexedFunctionHead(self.base_function, *self.indices, *new_indices)

    def __repr__(self) -> str:
        return self._sympystr(sympy.printing.str.StrPrinter())

    def __str__(self) -> str:
        return repr(self)

    def _sympystr(self, printer: Any) -> str:
        """Return SymPy's plain-text printer form for this head."""

        index_text = ",".join(printer.doprint(index) for index in self.indices)
        return f"{self.base_function.__name__}[{index_text}]"

    def _latex(self, printer: Any) -> str:
        """Return SymPy's LaTeX printer form for this head."""

        index_text = ",".join(printer._print(index) for index in self.indices)
        return rf"{printer._print(self.base_function)}_{{{index_text}}}"

    def compare(self, other: Any) -> int:
        """Compare indexed heads without ordering raw FunctionClass objects."""

        if self is other:
            return 0
        if not isinstance(other, IndexedFunctionHead):
            return super().compare(other)

        name_cmp = _compare_scalars(
            self.base_function.__name__,
            other.base_function.__name__,
        )
        if name_cmp:
            return name_cmp
        return _compare_basic_sequences(self.indices, other.indices)

    def _xreplace(self, rule: dict[Any, Any]) -> tuple[Any, bool]:
        """Apply structural replacements to the base head and its indices."""

        if self in rule:
            return rule[self], True

        new_base = rule.get(self.base_function, self.base_function)
        new_indices = []
        changed = new_base is not self.base_function

        # Visit structural indices so callers can rewrite placeholders and
        # whole index expressions with ordinary SymPy replacement tools.
        for index in self.indices:
            replaced_index, index_changed = index._xreplace(rule)
            new_indices.append(replaced_index)
            changed = changed or index_changed

        if not changed:
            return self, False

        return _coerce_head_result(new_base, tuple(new_indices), original=self), True

    def _subs(self, old: Any, new: Any, **hints: Any) -> Any:
        """Apply substitution to the base head and structural indices."""

        if self == old:
            return new

        new_base = new if self.base_function == old else self.base_function
        new_indices = []
        changed = new_base is not self.base_function

        for index in self.indices:
            replaced_index = index._subs(old, new, **hints)
            new_indices.append(replaced_index)
            changed = changed or replaced_index != index

        if not changed:
            return self

        return _coerce_head_result(new_base, tuple(new_indices), original=self)


class AppliedIndexedUndef(AppliedUndef):
    """Represent an application of an indexed undefined-function head."""

    _iterable = False

    def __new__(cls, head: IndexedFunctionHead | FunctionClass, *args: Any) -> AppliedIndexedUndef:
        if isinstance(head, IndexedFunctionHead):
            indexed_head = head
        else:
            indexed_head = IndexedFunctionHead(head)

        call_args = tuple(sympify(arg) for arg in args)
        obj = Basic.__new__(cls, *call_args)
        obj._indexed_head = indexed_head
        return obj

    @property
    def func(self) -> IndexedFunctionHead:
        """Return the symbolic indexed head used to build this call."""

        return self._indexed_head

    @property
    def base_function(self) -> FunctionClass:
        """Return the underlying unindexed SymPy function head."""

        return self.func.base_function

    @property
    def base(self) -> FunctionClass:
        """Alias for base_function to match the instance-based guide."""

        return self.base_function

    @property
    def indices(self) -> tuple[Basic, ...]:
        """Return the structural index tuple bound to this call."""

        return self.func.indices

    @property
    def call_args(self) -> tuple[Basic, ...]:
        """Return the ordinary call-argument tuple."""

        return self.args

    @property
    def free_symbols(self) -> set[Basic]:
        """Return free symbols from both the head indices and call arguments."""

        symbols = set(super().free_symbols)
        for index in self.indices:
            symbols.update(index.free_symbols)
        return symbols

    def _hashable_content(self) -> tuple[Any, ...]:
        """Include the indexed head in structural identity and hashing."""

        return (self.func, *self.args)

    def __iter__(self) -> None:
        """Reject sequence-style iteration over applied indexed functions."""

        raise TypeError(f"{self} is not iterable")

    def __repr__(self) -> str:
        return self._sympystr(sympy.printing.str.StrPrinter())

    def __str__(self) -> str:
        return repr(self)

    def _sympystr(self, printer: Any) -> str:
        """Return SymPy's plain-text printer form for this expression."""

        index_text = ",".join(printer.doprint(index) for index in self.indices)
        arg_text = ",".join(printer.doprint(arg) for arg in self.args)
        if index_text:
            return f"{self.base_function.__name__}[{index_text}]({arg_text})"
        return f"{self.base_function.__name__}({arg_text})"

    def _latex(self, printer: Any) -> str:
        """Return SymPy's LaTeX printer form for this expression."""

        arg_text = ",".join(printer._print(arg) for arg in self.args)
        base_text = printer._print(self.base_function)
        if self.indices:
            index_text = ",".join(printer._print(index) for index in self.indices)
            return rf"{base_text}_{{{index_text}}}\!\left({arg_text}\right)"
        return rf"{base_text}\!\left({arg_text}\right)"

    def compare(self, other: Any) -> int:
        """Compare indexed calls without ordering raw FunctionClass objects."""

        if self is other:
            return 0
        if not isinstance(other, AppliedIndexedUndef):
            return super().compare(other)

        head_cmp = self.func.compare(other.func)
        if head_cmp:
            return head_cmp
        return _compare_basic_sequences(self.args, other.args)

    def matches(
        self,
        pattern: Any,
        repl_dict: dict[Any, Any] | None = None,
        old: bool = False,
    ) -> dict[Any, Any] | None:
        """Match indexed heads, structural indices, and call arguments."""

        repl_dict = {} if repl_dict is None else repl_dict.copy()

        if isinstance(pattern, AppliedIndexedUndef):
            if self.base_function != pattern.base_function:
                return None
            if len(self.indices) != len(pattern.indices):
                return None
            if len(self.args) != len(pattern.args):
                return None

            for self_index, pattern_index in zip(self.indices, pattern.indices):
                repl_dict = self_index.matches(pattern_index, repl_dict, old=old)
                if repl_dict is None:
                    return None

            for self_arg, pattern_arg in zip(self.args, pattern.args):
                repl_dict = self_arg.matches(pattern_arg, repl_dict, old=old)
                if repl_dict is None:
                    return None
            return repl_dict

        from sympy import WildFunction

        if hasattr(pattern, "func") and isinstance(pattern.func, WildFunction):
            if len(self.args) != len(pattern.args):
                return None

            wild_func = pattern.func
            if wild_func in repl_dict and repl_dict[wild_func] != self.func:
                return None
            repl_dict[wild_func] = self.func

            for pattern_arg, self_arg in zip(pattern.args, self.args):
                repl_dict = pattern_arg.matches(self_arg, repl_dict, old=old)
                if repl_dict is None:
                    return None
            return repl_dict

        return super().matches(pattern, repl_dict, old=old)

    def has(self, *patterns: Any) -> bool:
        """Return whether the call or its head indices contain a pattern."""

        if self.base_function in patterns or self.func in patterns:
            return True
        if super().has(*patterns):
            return True
        for index in self.indices:
            if index in patterns:
                return True
            if index.has(*patterns):
                return True
        return False

    def _xreplace(self, rule: dict[Any, Any]) -> tuple[Any, bool]:
        """Apply structural replacements to the call head and arguments."""

        if self in rule:
            return rule[self], True

        new_head, head_changed = self.func._xreplace(rule)
        new_args = []
        args_changed = False

        # Rebuild call arguments through SymPy's replacement protocol so larger
        # expressions can replace payloads without corrupting indexed-head state.
        for arg in self.args:
            replaced_arg, arg_changed = arg._xreplace(rule)
            new_args.append(replaced_arg)
            args_changed = args_changed or arg_changed

        if not head_changed and not args_changed:
            return self, False

        rebuilt = _coerce_application_result(new_head, tuple(new_args), original=self)
        return rebuilt, True

    def _subs(self, old: Any, new: Any, **hints: Any) -> Any:
        """Apply substitution to the call head, indices, and ordinary arguments."""

        if self == old:
            return new

        new_head = self.func._subs(old, new, **hints)
        new_args = []
        changed = new_head != self.func

        for arg in self.args:
            replaced_arg = arg._subs(old, new, **hints)
            new_args.append(replaced_arg)
            changed = changed or replaced_arg != arg

        if not changed:
            return self

        return _coerce_application_result(new_head, tuple(new_args), original=self)


class IndexedFunctionBase:
    """Declare an indexable undefined-function base explicitly."""

    def __init__(self, name: Any, **assumptions: Any) -> None:
        if isinstance(name, FunctionClass):
            self.function_head = name
        elif isinstance(name, IndexedFunctionBase):
            self.function_head = name.function_head
        else:
            self.function_head = sympy.Function(str(name), **assumptions)

    @property
    def base_function(self) -> FunctionClass:
        """Return the unindexed function head."""

        return self.function_head

    def __getitem__(self, key: Any) -> IndexedFunctionHead:
        """Return a symbolic head that captures the provided indices."""

        indices = key if isinstance(key, tuple) else (key,)
        return IndexedFunctionHead(self.function_head, *indices)

    def __call__(self, *args: Any) -> Any:
        """Apply the plain unindexed function head."""

        return self.function_head(*args)

    def __iter__(self) -> None:
        """Reject sequence-style iteration over explicit function bases."""

        raise TypeError(f"{self} is not iterable")

    def __repr__(self) -> str:
        return self.function_head.__name__

    def __str__(self) -> str:
        return repr(self)

    def _sympystr(self, printer: Any) -> str:
        return printer.doprint(self.function_head)

    def _latex(self, printer: Any) -> str:
        return printer._print(self.function_head)

    def _sympy_(self) -> FunctionClass:
        """Coerce explicit bases to their underlying SymPy function head."""

        return self.function_head


IndexedFunction = IndexedFunctionBase
IndexedUndefinedFunction = IndexedFunctionHead

from pathlib import PurePosixPath
IndexedFunctionHead._mt_help = {
    "path": PurePosixPath("library/Indexed"),
    "anchor": None,
    "label": "Indexed",
}
AppliedIndexedUndef._mt_help = {
    "path": PurePosixPath("library/Expression"),
    "anchor": None,
    "label": "Expression",
}
IndexedFunctionBase._mt_help = {
    "path": PurePosixPath("library/Indexed"),
    "anchor": None,
    "label": "Indexed",
}


def patch() -> None:
    """Install indexed-head bracket syntax on standard SymPy undefined functions."""

    global _PATCHED
    if _PATCHED:
        return

    from sympy.core.function import UndefinedFunction

    def _undefined_function_getitem(self: FunctionClass, key: Any) -> IndexedFunctionHead:
        indices = key if isinstance(key, tuple) else (key,)
        return IndexedFunctionHead(self, *indices)

    UndefinedFunction.__getitem__ = _undefined_function_getitem
    UndefinedFunction._iterable = False

    _PATCHED = True


def _coerce_head_result(
    base_or_head: Any,
    indices: tuple[Basic, ...],
    *,
    original: IndexedFunctionHead,
) -> IndexedFunctionHead | Any:
    """Normalize head replacements back into an indexed symbolic head."""

    if isinstance(base_or_head, IndexedFunctionHead):
        return IndexedFunctionHead(
            base_or_head.base_function,
            *base_or_head.indices,
            *indices,
        )
    if isinstance(base_or_head, FunctionClass):
        return IndexedFunctionHead(base_or_head, *indices)
    if base_or_head == original.base_function:
        return IndexedFunctionHead(original.base_function, *indices)
    return base_or_head


def _coerce_application_result(
    head_or_base: Any,
    args: tuple[Basic, ...],
    *,
    original: AppliedIndexedUndef,
) -> Any:
    """Normalize call replacements back into a callable indexed application."""

    if isinstance(head_or_base, IndexedFunctionHead):
        return head_or_base(*args)
    if isinstance(head_or_base, FunctionClass):
        return IndexedFunctionHead(head_or_base, *original.indices)(*args)
    if head_or_base == original.base_function:
        return IndexedFunctionHead(original.base_function, *original.indices)(*args)
    if callable(head_or_base):
        return head_or_base(*args)
    return head_or_base


def _compare_basic_sequences(left: tuple[Basic, ...], right: tuple[Basic, ...]) -> int:
    """Compare two Basic tuples elementwise."""

    length_cmp = _compare_scalars(len(left), len(right))
    if length_cmp:
        return length_cmp
    for left_item, right_item in zip(left, right):
        item_cmp = left_item.compare(right_item)
        if item_cmp:
            return item_cmp
    return 0


def _compare_scalars(left: Any, right: Any) -> int:
    """Return the sign of a simple scalar comparison."""

    return (left > right) - (left < right)
