"""Build immutable unary pipeline operators around canonical argument slots.

``PipeOp`` compiles a callable signature into separate positional-only,
positional-or-keyword, keyword-only, var-positional, and var-keyword
containers. Base projections such as ``.p(...)`` and ``.c(...)`` update those
containers for both direct calls and pipeline execution, while
``.p_pipe(...)`` and ``.c_pipe(...)`` override only the pipeline-execution
mirrors. ``Pipeline`` composes ``PipeOp`` stages into flat, immutable unary
execution chains.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
import inspect
from types import MethodType
from typing import Any, overload


class _Sentinel:
    """Represent a slot marker that is compared by identity."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


CLEAR = _Sentinel("CLEAR")
_EMPTY = _Sentinel("_EMPTY")
_INPUT = _Sentinel("_INPUT")

__all__ = ["CLEAR", "PipeOp", "Pipeline", "pipeop"]


class _ExecutionPlan:
    """Describe whether a direct call can execute immediately."""

    def __init__(self, ready: bool, args: list[Any], kwargs: dict[str, Any]) -> None:
        self.ready = ready
        self.args = args
        self.kwargs = kwargs


class Pipeline:
    """Store an immutable sequence of executable ``PipeOp`` stages.

    Parameters
    ----------
    stages : Iterable[Any], default=()
        Objects to normalize into a flat pipeline stage list.

    Examples
    --------
    Basic usage:

    >>> def inc(x):
    ...     return x + 1
    >>> def double(x):
    ...     return x * 2
    >>> pipe = Pipeline([PipeOp(inc), PipeOp(double)])
    >>> pipe(3)
    8
    """

    def __init__(self, stages: Iterable[Any] = ()) -> None:
        normalized: list[PipeOp] = []

        # Flatten nested pipelines so execution stays a simple linear pass.
        for item in stages:
            if isinstance(item, Pipeline):
                normalized.extend(item._stages)
                continue

            if isinstance(item, PipeOp):
                normalized.append(item)
                continue

            if callable(item):
                normalized.append(PipeOp(item))
                continue

            raise TypeError(f"unsupported pipeline stage type: {type(item).__name__}")

        self._stages = tuple(normalized)

    def __iter__(self) -> Iterator[PipeOp]:
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    @overload
    def __getitem__(self, index: int) -> PipeOp: ...

    @overload
    def __getitem__(self, index: slice) -> Pipeline: ...

    def __getitem__(self, index: int | slice) -> PipeOp | Pipeline:
        if isinstance(index, slice):
            return Pipeline(self._stages[index])
        return self._stages[index]

    def __bool__(self) -> bool:
        return bool(self._stages)

    def __call__(self, value: Any) -> Any:
        result = value

        # Execute each stage through its pipeline mirror state.
        for stage in self._stages:
            result = stage._execute_pipe(result)

        return result

    def __rrshift__(self, value: Any) -> Any:
        return self(value)

    def __rshift__(self, other: Any) -> Pipeline:
        # Known composition targets are handled directly and preserve flatness.
        if isinstance(other, Pipeline):
            return Pipeline((*self._stages, *other._stages))

        if isinstance(other, PipeOp):
            return Pipeline((*self._stages, other))

        if callable(other) and not _has_custom_rrshift(other):
            return Pipeline((*self._stages, PipeOp(other)))

        return NotImplemented

    def __repr__(self) -> str:
        names = ", ".join(stage.__qualname__ for stage in self._stages)
        return f"Pipeline([{names}])"


class PipeOp:
    """Wrap a callable in an immutable unary pipeline operator.

    Parameters
    ----------
    fn : callable
        Callable to compile into canonical ``PipeOp`` slot state.
    input : int | str, default=0
        Selector identifying which slot receives the piped value.
    name : str | None, default=None
        Optional public name override for the operator.

    Notes
    -----
    ``PipeOp`` intentionally omits ``__wrapped__``. Introspection is driven by
    the operator's dynamic signature and current configuration state instead.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | PipeOp,
        *,
        input: int | str = 0,
        name: str | None = None,
    ) -> None:
        wrapped = fn.fn if isinstance(fn, PipeOp) else fn
        if not callable(wrapped):
            raise TypeError(f"PipeOp expected a callable, got {type(fn).__name__}")

        self._fn = wrapped
        self._input_selector = input
        self._name_override = name
        self._signature = inspect.signature(wrapped)

        # Split the callable signature into the canonical slot containers.
        self._po_names: list[str] = []
        self._pk_names: list[str] = []
        self._ko_names: list[str] = []
        self._vp_name: str | None = None
        self._vk_name: str | None = None
        self._po: list[Any] = []
        self._pk: OrderedDict[str, Any] = OrderedDict()
        self._ko: OrderedDict[str, Any] = OrderedDict()
        self._vp: list[Any] | None = None
        self._vk: dict[str, Any] | None = None
        self._compile_signature()
        self._select_input(input)
        self._sync_pipe_mirrors()

        # Mirror the callable metadata while still allowing a public name override.
        self.__annotations__ = dict(getattr(wrapped, "__annotations__", {}))
        self.__module__ = getattr(wrapped, "__module__", __name__)
        public_name = getattr(wrapped, "__name__", type(wrapped).__name__)
        public_qualname = getattr(wrapped, "__qualname__", public_name)
        if name is not None:
            public_name = name
            public_qualname = name
        self.__name__ = public_name
        self.__qualname__ = public_qualname

    @property
    def fn(self) -> Callable[..., Any]:
        """Return the wrapped callable object."""

        return self._fn

    @property
    def __signature__(self) -> inspect.Signature:
        """Return the current required direct-call surface."""

        parameters: list[inspect.Parameter] = []

        # Only slots that still need caller input remain in the public signature.
        for name in self._po_names:
            value = self._po[self._po_names.index(name)]
            if value in (_EMPTY, _INPUT):
                param = self._signature.parameters[name]
                parameters.append(param.replace(default=inspect.Parameter.empty))

        for name in self._pk_names:
            value = self._pk[name]
            if value in (_EMPTY, _INPUT):
                param = self._signature.parameters[name]
                parameters.append(param.replace(default=inspect.Parameter.empty))

        for name in self._ko_names:
            value = self._ko[name]
            if value in (_EMPTY, _INPUT):
                param = self._signature.parameters[name]
                parameters.append(param.replace(default=inspect.Parameter.empty))

        if self._vp_name is not None:
            parameters.append(self._signature.parameters[self._vp_name])

        if self._vk_name is not None:
            parameters.append(self._signature.parameters[self._vk_name])

        return inspect.Signature(parameters)

    @property
    def __doc__(self) -> str:
        """Return the wrapped docstring plus current configuration diagnostics."""

        original = inspect.getdoc(self._fn) or ""
        lines = [original] if original else []
        lines.append("pipeops.PipeOp configuration")
        lines.append(f"base _po={self._po!r}")
        lines.append(f"base _pk={self._pk!r}")
        lines.append(f"base _ko={self._ko!r}")
        lines.append(f"base _vp={self._vp!r}")
        lines.append(f"base _vk={self._vk!r}")
        lines.append(f"pipe _po={self._po_pipe!r}")
        lines.append(f"pipe _pk={self._pk_pipe!r}")
        lines.append(f"pipe _ko={self._ko_pipe!r}")
        lines.append(f"pipe _vp={self._vp_pipe!r}")
        lines.append(f"pipe _vk={self._vk_pipe!r}")
        return "\n\n".join(lines[:1] + ["\n".join(lines[1:])]) if original else "\n".join(lines)

    def __repr__(self) -> str:
        return f"pipeops.PipeOp({getattr(self._fn, '__qualname__', type(self._fn).__name__)})"

    def __get__(self, instance: Any, owner: type[Any] | None = None) -> PipeOp:
        if instance is None:
            return self

        # Rebind the callable first, then reapply the user-facing configuration.
        descriptor_get = getattr(self._fn, "__get__", None)
        if descriptor_get is None:
            bound_fn = MethodType(self._fn, instance)
        else:
            bound_fn = descriptor_get(instance, owner)

        rebound = PipeOp(bound_fn, input=self._input_selector, name=self._name_override)

        # Keyword-addressed configurations survive descriptor binding directly.
        keyword_base: dict[str, Any] = {}
        keyword_pipe: dict[str, Any] = {}
        for name, value in self._pk.items():
            if name == self._pk_names[0] and self._signature.parameters[name].name not in rebound._pk:
                continue
            if name in rebound._pk and value not in (_EMPTY, _INPUT):
                keyword_base[name] = value
                if self._pk_pipe[name] != value:
                    keyword_pipe[name] = self._pk_pipe[name]
        for name, value in self._ko.items():
            if name in rebound._ko and value not in (_EMPTY, _INPUT):
                keyword_base[name] = value
                if self._ko_pipe[name] != value:
                    keyword_pipe[name] = self._ko_pipe[name]
        if self._vk is not None:
            for name, value in self._vk.items():
                if value is _INPUT:
                    continue
                keyword_base[name] = value
                if self._vk_pipe is not None and self._vk_pipe.get(name) != value:
                    keyword_pipe[name] = self._vk_pipe[name]

        if keyword_base:
            rebound = rebound.c(**keyword_base)
        if keyword_pipe:
            rebound = rebound.c_pipe(**keyword_pipe)
        return rebound

    def __getattr__(self, name: str) -> Any:
        if name == "__wrapped__":
            raise AttributeError(name)
        return getattr(self._fn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "fn":
            raise AttributeError("fn is read-only")

        if name.startswith("_") or name in {
            "__annotations__",
            "__module__",
            "__name__",
            "__qualname__",
        }:
            object.__setattr__(self, name, value)
            return

        setattr(self._fn, name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_") or name in {
            "__annotations__",
            "__module__",
            "__name__",
            "__qualname__",
        }:
            object.__delattr__(self, name)
            return

        delattr(self._fn, name)

    def __rshift__(self, other: Any) -> Pipeline:
        return Pipeline((self,)).__rshift__(other)

    def __rrshift__(self, value: Any) -> Any:
        return self._execute_pipe(value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plan = self._build_direct_execution_plan(args, kwargs)
        if plan.ready:
            return self._fn(*plan.args, **plan.kwargs)

        # Calls that target varargs or arbitrary varkw input remain execution-only.
        if self._input_is_var_input():
            raise TypeError("direct call is incomplete")

        if not args and not kwargs:
            raise TypeError("direct call is incomplete")

        if self._currying_targets_input(kwargs):
            raise TypeError("currying may not configure the pipe input slot")

        curried = self

        # Positional currying fills only open non-input fixed positional slots.
        if args:
            open_indices = self._open_currying_indices()
            if len(args) > len(open_indices):
                raise TypeError("too many positional values for currying")
            mapping = {index: value for index, value in zip(open_indices, args)}
            curried = curried.p(mapping)

        # Keyword currying reuses the same validation as ordinary keyword projection.
        if kwargs:
            curried = curried.c(**kwargs)

        return curried

    def p(self, values: Mapping[int, Any] | Any) -> PipeOp:
        """Project positional defaults into the canonical base state."""

        configured = self._clone()

        # Mapping mode targets explicit canonical positional indices.
        if isinstance(values, Mapping):
            for index, value in values.items():
                configured._apply_base_positional_update(index, value)
        else:
            sequence = _coerce_projection_sequence(values)
            targets = configured._base_sequence_targets(len(sequence))
            for index, value in zip(targets, sequence):
                configured._apply_base_positional_update(index, value)

        configured._sync_pipe_mirrors()
        return configured

    def c(self, **kwargs: Any) -> PipeOp:
        """Project keyword defaults into the canonical base state."""

        configured = self._clone()

        # Base keyword projection updates the authoritative containers first.
        for name, value in kwargs.items():
            configured._apply_base_keyword_update(name, value)

        configured._sync_pipe_mirrors()
        return configured

    def p_pipe(self, values: Mapping[int, Any] | Any) -> PipeOp:
        """Override only pipeline-execution positional mirrors."""

        configured = self._clone()

        # Pipeline-only positional overrides may only touch concrete baselines.
        if isinstance(values, Mapping):
            for index, value in values.items():
                configured._apply_pipe_positional_update(index, value)
        else:
            sequence = _coerce_projection_sequence(values)
            targets = configured._pipe_sequence_targets(len(sequence))
            for index, value in zip(targets, sequence):
                configured._apply_pipe_positional_update(index, value)

        return configured

    def c_pipe(self, **kwargs: Any) -> PipeOp:
        """Override only pipeline-execution keyword mirrors."""

        configured = self._clone()

        # Pipeline-only keyword overrides are strictly override-only.
        for name, value in kwargs.items():
            configured._apply_pipe_keyword_update(name, value)

        return configured

    def _compile_signature(self) -> None:
        """Split the wrapped signature into the canonical slot containers."""

        for parameter in self._signature.parameters.values():
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                self._po_names.append(parameter.name)
                self._po.append(_parameter_default(parameter))
                continue

            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                self._pk_names.append(parameter.name)
                self._pk[parameter.name] = _parameter_default(parameter)
                continue

            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                self._ko_names.append(parameter.name)
                self._ko[parameter.name] = _parameter_default(parameter)
                continue

            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                self._vp_name = parameter.name
                self._vp = []
                continue

            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                self._vk_name = parameter.name
                self._vk = {}

    def _select_input(self, selector: int | str) -> None:
        """Place the unique ``_INPUT`` marker into the selected canonical slot."""

        if isinstance(selector, bool):
            raise TypeError("input selector must be an int or str")

        if isinstance(selector, int):
            if selector < -1:
                raise IndexError("input selector below -1 is invalid")

            if selector == -1:
                if self._vp is None:
                    raise TypeError("input=-1 requires a var-positional collector")
                self._vp.append(_INPUT)
                return

            refs = self._fixed_slot_refs()
            if selector >= len(refs):
                if not refs and self._vp is None and self._vk is None:
                    raise TypeError("callable has no slot available for pipeline input")
                raise IndexError("input selector is outside the fixed argument range")
            domain, key = refs[selector]
            self._set_slot_value(domain, key, _INPUT)
            return

        if isinstance(selector, str):
            if selector in self._po_names:
                raise TypeError("positional-only parameters cannot be selected by name")
            if selector in self._pk:
                self._pk[selector] = _INPUT
                return
            if selector in self._ko:
                self._ko[selector] = _INPUT
                return
            if self._vk is not None:
                self._vk[selector] = _INPUT
                return
            raise TypeError(f"unknown input selector name: {selector}")

        raise TypeError("input selector must be an int or str")

    def _clone(self) -> PipeOp:
        """Return a deep copy of the operator state."""

        clone = PipeOp(self._fn, input=self._input_selector, name=self._name_override)
        clone._po = list(self._po)
        clone._pk = OrderedDict(self._pk)
        clone._ko = OrderedDict(self._ko)
        clone._vp = None if self._vp is None else list(self._vp)
        clone._vk = None if self._vk is None else dict(self._vk)
        clone._po_pipe = list(self._po_pipe)
        clone._pk_pipe = OrderedDict(self._pk_pipe)
        clone._ko_pipe = OrderedDict(self._ko_pipe)
        clone._vp_pipe = None if self._vp_pipe is None else list(self._vp_pipe)
        clone._vk_pipe = None if self._vk_pipe is None else dict(self._vk_pipe)
        return clone

    def _sync_pipe_mirrors(self) -> None:
        """Copy the authoritative base state into the execution mirrors."""

        self._po_pipe = list(self._po)
        self._pk_pipe = OrderedDict(self._pk)
        self._ko_pipe = OrderedDict(self._ko)
        self._vp_pipe = None if self._vp is None else list(self._vp)
        self._vk_pipe = None if self._vk is None else dict(self._vk)

    def _fixed_slot_refs(self) -> list[tuple[str, int | str]]:
        """Return fixed-slot references in signature order."""

        refs: list[tuple[str, int | str]] = []
        refs.extend(("po", index) for index in range(len(self._po)))
        refs.extend(("pk", name) for name in self._pk_names)
        refs.extend(("ko", name) for name in self._ko_names)
        return refs

    def _base_sequence_targets(self, count: int) -> list[int]:
        """Return positional indices targeted by base sequence projection."""

        targets: list[int] = []
        fixed_count = self._fixed_positional_count()

        # Sequence projection skips only the reserved pipe-input slot.
        for index in range(fixed_count):
            if self._get_base_positional_value(index) is not _INPUT:
                targets.append(index)

        if self._vp is not None:
            for offset, value in enumerate(self._vp):
                if value is not _INPUT:
                    targets.append(fixed_count + offset)

        while len(targets) < count:
            if self._vp is None:
                raise TypeError("too many positional projection values")
            targets.append(fixed_count + len(self._vp) + (len(targets) - len(self._current_sequence_targets())))

        return targets[:count]

    def _current_sequence_targets(self) -> list[int]:
        """Return the existing non-input positional indices."""

        targets: list[int] = []
        fixed_count = self._fixed_positional_count()
        for index in range(fixed_count):
            if self._get_base_positional_value(index) is not _INPUT:
                targets.append(index)
        if self._vp is not None:
            for offset, value in enumerate(self._vp):
                if value is not _INPUT:
                    targets.append(fixed_count + offset)
        return targets

    def _pipe_sequence_targets(self, count: int) -> list[int]:
        """Return positional indices that may receive pipeline-only overrides."""

        targets: list[int] = []
        fixed_count = self._fixed_positional_count()

        # Override-only sequence projection includes only concrete baselines.
        for index in range(fixed_count):
            value = self._get_base_positional_value(index)
            if value not in (_EMPTY, _INPUT):
                targets.append(index)

        if self._vp is not None:
            for offset, value in enumerate(self._vp):
                if value is not _INPUT:
                    targets.append(fixed_count + offset)

        if count > len(targets):
            raise TypeError("too many pipeline-only positional overrides")

        return targets[:count]

    def _apply_base_positional_update(self, index: Any, value: Any) -> None:
        """Apply one base positional update against the canonical index space."""

        absolute = _validate_projection_index(index)
        fixed_count = self._fixed_positional_count()

        if absolute < fixed_count:
            domain, key = self._fixed_positional_ref(absolute)
            current = self._get_slot_value(domain, key)
            if current is _INPUT and domain != "po":
                raise TypeError("cannot overwrite the pipe input slot")
            self._set_slot_value(domain, key, _EMPTY if value is CLEAR else value)
            return

        if self._vp is None:
            raise IndexError("var-positional projection requires *args")

        offset = absolute - fixed_count
        if offset > len(self._vp):
            raise IndexError("var-positional projection may not create holes")

        if offset == len(self._vp):
            if value is CLEAR:
                raise IndexError("cannot clear a non-existent var-positional slot")
            self._vp.append(value)
            return

        current = self._vp[offset]
        if current is _INPUT:
            raise TypeError("cannot overwrite the pipe input slot")
        if value is CLEAR:
            del self._vp[offset]
            return
        self._vp[offset] = value

    def _apply_base_keyword_update(self, name: str, value: Any) -> None:
        """Apply one base keyword update against the canonical name domains."""

        if name in self._po_names:
            raise TypeError("positional-only parameters are not keyword-addressable")

        if name in self._pk:
            if self._pk[name] is _INPUT:
                raise TypeError("cannot overwrite the pipe input slot")
            self._pk[name] = _EMPTY if value is CLEAR else value
            return

        if name in self._ko:
            if self._ko[name] is _INPUT:
                raise TypeError("cannot overwrite the pipe input slot")
            self._ko[name] = _EMPTY if value is CLEAR else value
            return

        if self._vk is None:
            raise TypeError(f"unknown keyword projection target: {name}")

        current = self._vk.get(name)
        if current is _INPUT:
            raise TypeError("cannot overwrite the pipe input slot")
        if value is CLEAR:
            self._vk.pop(name, None)
            return
        self._vk[name] = value

    def _apply_pipe_positional_update(self, index: Any, value: Any) -> None:
        """Apply one pipeline-only positional override."""

        absolute = _validate_projection_index(index)
        fixed_count = self._fixed_positional_count()

        if absolute < fixed_count:
            domain, key = self._fixed_positional_ref(absolute)
            base = self._get_slot_value(domain, key)
            if base is _INPUT:
                raise TypeError("cannot override the pipe input slot")
            if base is _EMPTY:
                raise TypeError("pipeline-only overrides require a concrete baseline")
            self._set_pipe_slot_value(domain, key, base if value is CLEAR else value)
            return

        if self._vp is None:
            raise IndexError("pipeline-only var-positional override requires *args")

        offset = absolute - fixed_count
        if offset >= len(self._vp):
            raise IndexError("pipeline-only positional overrides may not extend *args")

        base = self._vp[offset]
        if base is _INPUT:
            raise TypeError("cannot override the pipe input slot")
        self._vp_pipe[offset] = base if value is CLEAR else value

    def _apply_pipe_keyword_update(self, name: str, value: Any) -> None:
        """Apply one pipeline-only keyword override."""

        if name in self._po_names:
            raise TypeError("positional-only parameters are not keyword-addressable")

        if name in self._pk:
            base = self._pk[name]
            if base is _INPUT:
                raise TypeError("cannot override the pipe input slot")
            if base is _EMPTY:
                raise TypeError("pipeline-only overrides require a concrete baseline")
            self._pk_pipe[name] = base if value is CLEAR else value
            return

        if name in self._ko:
            base = self._ko[name]
            if base is _INPUT:
                raise TypeError("cannot override the pipe input slot")
            if base is _EMPTY:
                raise TypeError("pipeline-only overrides require a concrete baseline")
            self._ko_pipe[name] = base if value is CLEAR else value
            return

        if self._vk is None or name not in self._vk:
            raise TypeError(f"unknown pipeline-only keyword override target: {name}")

        base = self._vk[name]
        if base is _INPUT:
            raise TypeError("cannot override the pipe input slot")
        self._vk_pipe[name] = base if value is CLEAR else value

    def _build_direct_execution_plan(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _ExecutionPlan:
        """Return either an executable direct-call plan or an incomplete plan."""

        remaining_args = list(args)
        remaining_kwargs = dict(kwargs)
        final_args: list[Any] = []
        final_kwargs: dict[str, Any] = {}
        missing_required = False

        # Merge direct-call inputs with configured defaults while preserving Python rules.
        for parameter in self._signature.parameters.values():
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                slot = self._po[self._po_names.index(parameter.name)]
                if remaining_args:
                    final_args.append(remaining_args.pop(0))
                    continue
                if parameter.name in remaining_kwargs:
                    raise TypeError("positional-only parameter passed by keyword")
                if slot not in (_EMPTY, _INPUT):
                    final_args.append(slot)
                    continue
                missing_required = True
                continue

            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                slot = self._pk[parameter.name]
                if remaining_args:
                    if parameter.name in remaining_kwargs:
                        raise TypeError("multiple values for the same parameter")
                    final_args.append(remaining_args.pop(0))
                    continue
                if parameter.name in remaining_kwargs:
                    final_kwargs[parameter.name] = remaining_kwargs.pop(parameter.name)
                    continue
                if slot not in (_EMPTY, _INPUT):
                    final_kwargs[parameter.name] = slot
                    continue
                missing_required = True
                continue

            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                final_args.extend(remaining_args)
                remaining_args.clear()
                if self._vp is not None:
                    for value in self._vp:
                        if value is not _INPUT:
                            final_args.append(value)
                continue

            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                slot = self._ko[parameter.name]
                if parameter.name in remaining_kwargs:
                    final_kwargs[parameter.name] = remaining_kwargs.pop(parameter.name)
                    continue
                if slot not in (_EMPTY, _INPUT):
                    final_kwargs[parameter.name] = slot
                    continue
                missing_required = True
                continue

            if self._vk is not None:
                for key, value in self._vk.items():
                    if value is _INPUT:
                        continue
                    if key not in remaining_kwargs:
                        final_kwargs[key] = value
                final_kwargs.update(remaining_kwargs)
                remaining_kwargs.clear()

        if remaining_args:
            raise TypeError("too many positional arguments")
        if remaining_kwargs:
            raise TypeError(f"unexpected keyword argument {next(iter(remaining_kwargs))!r}")

        if missing_required:
            return _ExecutionPlan(False, final_args, final_kwargs)

        self._signature.bind(*final_args, **final_kwargs)
        return _ExecutionPlan(True, final_args, final_kwargs)

    def _open_currying_indices(self) -> list[int]:
        """Return fixed positional indices that remain open for currying."""

        indices: list[int] = []
        fixed_count = self._fixed_positional_count()

        # Positional currying fills only open non-input fixed positional slots.
        for index in range(fixed_count):
            value = self._get_base_positional_value(index)
            if value is _EMPTY:
                indices.append(index)

        return indices

    def _currying_targets_input(self, kwargs: Mapping[str, Any]) -> bool:
        """Report whether incomplete keyword currying tries to configure input."""

        if isinstance(self._input_selector, str):
            return self._input_selector in kwargs
        return False

    def _input_is_var_input(self) -> bool:
        """Report whether the pipe input lives in ``*args`` or arbitrary ``**kwargs``."""

        if self._vp is not None and any(value is _INPUT for value in self._vp):
            return True
        if self._vk is not None and any(value is _INPUT for value in self._vk.values()):
            return True
        return False

    def _execute_pipe(self, value: Any) -> Any:
        """Execute the wrapped callable through the pipeline mirror state."""

        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        # Materialize the full call from the pipeline mirrors without mutating them.
        for slot in self._po_pipe:
            if slot is _EMPTY:
                raise TypeError("pipeline execution requires all fixed slots to be concrete")
            args.append(value if slot is _INPUT else slot)

        for name in self._pk_names:
            slot = self._pk_pipe[name]
            if slot is _EMPTY:
                raise TypeError("pipeline execution requires all fixed slots to be concrete")
            args.append(value if slot is _INPUT else slot)

        if self._vp_pipe is not None:
            for slot in self._vp_pipe:
                args.append(value if slot is _INPUT else slot)

        for name in self._ko_names:
            slot = self._ko_pipe[name]
            if slot is _EMPTY:
                raise TypeError("pipeline execution requires all fixed slots to be concrete")
            kwargs[name] = value if slot is _INPUT else slot

        if self._vk_pipe is not None:
            for key, slot in self._vk_pipe.items():
                kwargs[key] = value if slot is _INPUT else slot

        self._signature.bind(*args, **kwargs)
        return self._fn(*args, **kwargs)

    def _fixed_positional_count(self) -> int:
        """Return the size of the fixed positional projection domain."""

        return len(self._po) + len(self._pk)

    def _fixed_positional_ref(self, absolute: int) -> tuple[str, int | str]:
        """Return the fixed-slot reference for one absolute positional index."""

        if absolute < len(self._po):
            return "po", absolute
        return "pk", self._pk_names[absolute - len(self._po)]

    def _get_base_positional_value(self, absolute: int) -> Any:
        """Return the base value for one absolute positional index."""

        domain, key = self._fixed_positional_ref(absolute)
        return self._get_slot_value(domain, key)

    def _get_slot_value(self, domain: str, key: int | str) -> Any:
        """Return a value from one canonical base container."""

        if domain == "po":
            return self._po[key]
        if domain == "pk":
            return self._pk[key]
        if domain == "ko":
            return self._ko[key]
        raise KeyError(domain)

    def _set_slot_value(self, domain: str, key: int | str, value: Any) -> None:
        """Write a value into one canonical base container."""

        if domain == "po":
            self._po[key] = value
            return
        if domain == "pk":
            self._pk[key] = value
            return
        if domain == "ko":
            self._ko[key] = value
            return
        raise KeyError(domain)

    def _set_pipe_slot_value(self, domain: str, key: int | str, value: Any) -> None:
        """Write a value into one canonical pipeline mirror container."""

        if domain == "po":
            self._po_pipe[key] = value
            return
        if domain == "pk":
            self._pk_pipe[key] = value
            return
        if domain == "ko":
            self._ko_pipe[key] = value
            return
        raise KeyError(domain)


def pipeop(
    fn: Callable[..., Any] | PipeOp | None = None,
    /,
    *,
    p: Mapping[int, Any] | Any | None = None,
    c: Mapping[str, Any] | None = None,
    p_pipe: Mapping[int, Any] | Any | None = None,
    c_pipe: Mapping[str, Any] | None = None,
    input: int | str = 0,
    name: str | None = None,
) -> PipeOp | Callable[[Callable[..., Any]], PipeOp]:
    """Wrap a callable in ``PipeOp`` and apply the requested projections.

    Parameters
    ----------
    fn : callable or PipeOp or None, default=None
        Callable to wrap immediately, or ``None`` for decorator form.
    p : mapping or sequence-like, optional
        Base positional projection passed to ``PipeOp.p(...)``.
    c : Mapping[str, Any], optional
        Base keyword projection passed to ``PipeOp.c(...)``.
    p_pipe : mapping or sequence-like, optional
        Pipeline-only positional overrides passed to ``PipeOp.p_pipe(...)``.
    c_pipe : Mapping[str, Any], optional
        Pipeline-only keyword overrides passed to ``PipeOp.c_pipe(...)``.
    input : int | str, default=0
        Selector identifying the pipe-input slot.
    name : str | None, default=None
        Optional public name override.

    Returns
    -------
    PipeOp or callable
        Immediate operator instance or deferred decorator.
    """

    def decorate(target: Callable[..., Any] | PipeOp) -> PipeOp:
        source = target if isinstance(target, PipeOp) else None
        selector = source._input_selector if source is not None and input == 0 else input
        op = PipeOp(source.fn if source is not None else target, input=selector, name=name)
        if p is not None:
            op = op.p(p)
        if c is not None:
            op = op.c(**dict(c))
        if p_pipe is not None:
            op = op.p_pipe(p_pipe)
        if c_pipe is not None:
            op = op.c_pipe(**dict(c_pipe))
        return op

    if fn is None:
        return decorate
    return decorate(fn)


def _parameter_default(parameter: inspect.Parameter) -> Any:
    """Return the canonical base value for one declared parameter."""

    if parameter.default is inspect.Parameter.empty:
        return _EMPTY
    return parameter.default


def _coerce_projection_sequence(values: Any) -> list[Any]:
    """Normalize a sequence-like projection source and reject string-like objects."""

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("string-like values are not positional projection sequences")
    try:
        return list(values)
    except TypeError as exc:
        raise TypeError("projection value must be a mapping or sequence-like object") from exc


def _validate_projection_index(index: Any) -> int:
    """Validate one explicit projection index."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("projection indices must be non-negative integers")
    if index < 0:
        raise IndexError("projection indices must be non-negative")
    return index


def _has_custom_rrshift(obj: Any) -> bool:
    """Report whether Python should defer ``>>`` to the right-hand object."""

    return getattr(type(obj), "__rrshift__", None) is not None
