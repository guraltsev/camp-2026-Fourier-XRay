"""Represent public numeric callables and implemented symbolic functions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import count

import numpy as np
import sympy

from .diagnostics import NumDiagnostic, NumArgumentError
from .specs import NumArgSpec

_IMPLEMENTED_FUNCTION_COUNTER = count()


@dataclass
class _CallCache:
    """Store optional public ``NumFunction`` call results."""

    enabled: bool
    maxsize: int | None
    values: OrderedDict[tuple[object, ...], object] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get_or_compute(
        self,
        values: tuple[object, ...],
        compute: Callable[[], object],
    ) -> object:
        """Return a cached result when public call caching is enabled."""

        if not self.enabled:
            return compute()

        key = _hashable_values(values)
        if key is None:
            self.misses += 1
            return compute()

        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]

        self.misses += 1
        result = compute()
        self.values[key] = result
        self.values.move_to_end(key)
        if self.maxsize is not None:
            while len(self.values) > self.maxsize:
                self.values.popitem(last=False)
        return result

    def info(self) -> dict[str, object]:
        """Return public cache counters without exposing cached values."""

        return {
            "enabled": self.enabled,
            "size": len(self.values),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> None:
        """Remove stored results and reset cache counters."""

        self.values.clear()
        self.hits = 0
        self.misses = 0


class NumEvaluationContext:
    """Expose bound ``NumFunction`` arguments to implemented callables."""

    def __init__(self) -> None:
        self._bound: dict[object, object] | None = None

    def bind(
        self,
        specs: tuple[NumArgSpec, ...],
        values: tuple[object, ...],
    ) -> None:
        """Make one public call's arguments visible during evaluation."""

        bound: dict[object, object] = {}
        for spec, value in zip(specs, values):
            bound[spec.symbol] = value
            bound[spec.name] = value
            if isinstance(spec.symbol, sympy.Symbol):
                bound[spec.symbol.name] = value
        self._bound = bound

    def clear(self) -> None:
        """Leave the active public-call context."""

        self._bound = None

    def parameter_values(
        self,
        parameters: tuple[sympy.Symbol, ...],
    ) -> dict[sympy.Symbol, object]:
        """Return scalar parameter values for a solver-backed function."""

        if self._bound is None:
            raise NumArgumentError(
                "Implemented functions can only be evaluated through Num."
            )

        values: dict[sympy.Symbol, object] = {}
        for parameter in parameters:
            if parameter in self._bound:
                value = self._bound[parameter]
            elif parameter.name in self._bound:
                value = self._bound[parameter.name]
            else:
                raise NumArgumentError(
                    f"Missing numeric value for solver parameter `{parameter}`."
                )

            array = np.asarray(value)
            if array.shape != ():
                raise NumArgumentError(
                    f"Parameter `{parameter}` received an array. This would "
                    "require many separate solver runs. Use an explicit sweep "
                    "API or loop intentionally."
                )
            values[parameter] = array.item()
        return values


@dataclass(frozen=True)
class NumFunction:
    """Represent a reusable numeric callable produced by ``Num``.

    Parameters
    ----------
    symbolic_origin : object
        Original symbolic expression compiled for numeric execution.
    args : tuple[NumArgSpec, ...]
        Numeric argument specs in call order.
    backend : str
        Public backend description.
    raw : object
        Underlying generated callable or callable bundle.
    warnings : tuple[NumDiagnostic, ...]
        Compile-time diagnostics collected for this function.

    Methods
    -------
    __call__
        Evaluate the numeric function with positional and keyword arguments.
    cache_info
        Return public cache counters for call and solver caches.
    clear_cache
        Clear private operational caches owned by this function.
    explain
        Return a short human-readable execution summary.
    """

    symbolic_origin: object
    args: tuple[NumArgSpec, ...]
    backend: str
    raw: object = field(compare=False, hash=False)
    warnings: tuple[NumDiagnostic, ...]
    _evaluator: Callable[[tuple[object, ...]], object] = field(
        repr=False,
        compare=False,
        hash=False,
    )
    execution_plan: str = "direct"
    vectorized: bool = True
    performance_notes: tuple[str, ...] = ()
    _context: NumEvaluationContext | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )
    _call_cache: _CallCache = field(
        default_factory=lambda: _CallCache(enabled=False, maxsize=None),
        repr=False,
        compare=False,
        hash=False,
    )
    _implemented_functions: tuple["ImplementedFunction", ...] = field(
        default=(),
        repr=False,
        compare=False,
        hash=False,
    )

    @property
    def num_role(self) -> str:
        """Return the symbolic/numeric boundary role name."""

        return "num-function"

    @property
    def expr(self) -> object:
        """Return the expression compiled by this numeric function."""

        return self.symbolic_origin

    @property
    def arguments(self) -> tuple[NumArgSpec, ...]:
        """Return public argument specifications in call order."""

        return self.args

    @property
    def diagnostics(self) -> tuple[NumDiagnostic, ...]:
        """Return compile-time diagnostics collected for this function."""

        return self.warnings

    def __call__(self, *values: object, **kwargs: object) -> object:
        """Evaluate the numeric function with bound arguments."""

        if len(values) > len(self.args):
            raise TypeError(
                f"{self!r} expected at most {len(self.args)} positional "
                f"arguments, got {len(values)}."
            )

        bound: dict[str, object] = {}
        for spec, value in zip(self.args, values):
            bound[spec.name] = value

        argument_names = {spec.name for spec in self.args}
        for name, value in kwargs.items():
            if name not in argument_names:
                raise TypeError(f"{self!r} got an unknown argument {name!r}.")
            if name in bound:
                raise TypeError(f"{self!r} got multiple values for argument {name!r}.")
            bound[name] = value

        missing = [spec.name for spec in self.args if spec.name not in bound]
        if missing:
            joined = ", ".join(missing)
            raise TypeError(f"{self!r} missing required argument(s): {joined}.")

        ordered_values = tuple(bound[spec.name] for spec in self.args)

        def compute() -> object:
            if self._context is None:
                return self._evaluator(ordered_values)
            self._context.bind(self.args, ordered_values)
            try:
                return self._evaluator(ordered_values)
            finally:
                self._context.clear()

        return self._call_cache.get_or_compute(ordered_values, compute)

    def __repr__(self) -> str:
        """Return a concise notebook-facing representation."""

        names = ", ".join(spec.name for spec in self.args)
        return f"NumFunction[{names}]"

    def cache_info(self) -> dict[str, object]:
        """Return public information about private operational caches."""

        solver_caches: dict[str, object] = {}
        seen_branches: set[int] = set()
        for component in self._implemented_functions:
            branch = component.branch
            identity = id(branch)
            if identity in seen_branches:
                continue
            seen_branches.add(identity)
            solver_caches[branch.cache_label] = branch.cache_info()

        return {
            "call_cache": self._call_cache.info(),
            "solver_caches": solver_caches,
        }

    def clear_cache(self) -> None:
        """Clear private call and solver caches."""

        self._call_cache.clear()
        seen_branches: set[int] = set()
        for component in self._implemented_functions:
            branch = component.branch
            identity = id(branch)
            if identity in seen_branches:
                continue
            seen_branches.add(identity)
            branch.clear_cache()

    def explain(self) -> str:
        """Return a short explanation of the numeric execution plan."""

        lines = [
            f"num_role: {self.num_role}",
            f"execution_plan: {self.execution_plan}",
            f"backend: {self.backend}",
        ]
        if self.performance_notes:
            lines.append("performance_notes:")
            lines.extend(f"- {note}" for note in self.performance_notes)
        if self.diagnostics:
            lines.append("diagnostics:")
            lines.extend(
                f"- {diagnostic.code}: {diagnostic.message}"
                for diagnostic in self.diagnostics
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ImplementedFunction:
    """Represent a symbolic function with an attached numeric implementation.

    Parameters
    ----------
    display_name : str
        Human-readable function name used for symbolic applications.
    branch : object
        Implemented solution branch that owns component evaluation.
    component_key : object
        Public key used to select the component from the branch.
    component_index : int
        Positional component index in the numeric solution vector.

    Methods
    -------
    __call__
        Apply the symbolic function to its independent variable.
    subs
        Return a specialized implemented function after symbolic substitution.
    explain
        Return a short human-readable implementation summary.
    """

    display_name: str
    branch: object
    component_key: object
    component_index: int
    _head: object = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        """Create a unique SymPy function head for this implementation."""

        suffix = next(_IMPLEMENTED_FUNCTION_COUNTER)
        safe_name = _safe_function_name(self.display_name)
        head = sympy.Function(f"{safe_name}_implemented_{suffix}")
        head._mt_implemented_function = self
        object.__setattr__(self, "_head", head)

    @property
    def num_role(self) -> str:
        """Return the symbolic/numeric boundary role name."""

        return "implemented-function"

    @property
    def symbolic_origin(self) -> object:
        """Return the symbolic problem that produced this function."""

        return self.branch.symbolic_origin

    @property
    def implementation_status(self) -> str:
        """Return whether the component is ready or parameterized."""

        return self.branch.implementation_status

    @property
    def required_parameters(self) -> tuple[sympy.Symbol, ...]:
        """Return unresolved solver parameters in deterministic order."""

        return self.branch.required_parameters

    @property
    def independent_variables(self) -> tuple[sympy.Symbol, ...]:
        """Return independent variables accepted by this function."""

        return self.branch.independent_variables

    @property
    def domain(self) -> tuple[object, object, object]:
        """Return the independent-variable domain of the component."""

        return self.branch.domain

    @property
    def vectorized(self) -> bool:
        """Return whether the implementation accepts array sample inputs."""

        return True

    @property
    def diagnostics(self) -> tuple[object, ...]:
        """Return diagnostics attached to the owning branch."""

        return self.branch.diagnostics

    def __call__(self, variable: object) -> object:
        """Return a symbolic application of the implemented function."""

        return self._head(variable)

    def subs(self, *args: object, **kwargs: object) -> "ImplementedFunction":
        """Return this component after substituting in the owning branch."""

        return self.branch.subs(*args, **kwargs)[self.component_key]

    def xreplace(self, rule: Mapping[object, object]) -> "ImplementedFunction":
        """Return this component after exact replacement in the branch."""

        return self.branch.xreplace(rule)[self.component_key]

    def evaluate(
        self,
        function_args: tuple[object, ...],
        parameter_values: dict[sympy.Symbol, object],
    ) -> object:
        """Evaluate this component through the owning solution branch."""

        if len(function_args) != 1:
            raise NumArgumentError(
                "Implemented IVP functions accept exactly one independent "
                "variable argument."
            )
        return self.branch.evaluate_component(
            self.component_index,
            function_args[0],
            parameter_values,
        )

    def explain(self) -> str:
        """Return a short explanation of this implemented function."""

        return (
            f"num_role: {self.num_role}\n"
            f"implementation_status: {self.implementation_status}\n"
            f"required_parameters: {self.required_parameters}"
        )

    def __repr__(self) -> str:
        """Return a concise component representation."""

        return f"ImplementedFunction[{self.display_name}]"


def make_call_cache(cache: object) -> _CallCache:
    """Return a private call cache from the public ``Num(cache=...)`` option."""

    if cache is False or cache is None:
        return _CallCache(enabled=False, maxsize=None)
    if cache is True:
        return _CallCache(enabled=True, maxsize=None)
    if isinstance(cache, dict):
        unknown = sorted(set(cache) - {"maxsize"})
        if unknown:
            joined = ", ".join(unknown)
            raise NumArgumentError(f"Unknown Num cache option(s): {joined}.")
        maxsize = cache.get("maxsize")
        if maxsize is not None:
            if not isinstance(maxsize, int) or isinstance(maxsize, bool):
                raise NumArgumentError("Num cache maxsize must be an integer.")
            if maxsize < 1:
                raise NumArgumentError("Num cache maxsize must be positive.")
        return _CallCache(enabled=True, maxsize=maxsize)
    raise NumArgumentError("Num cache must be False, True, or {'maxsize': n}.")


def implemented_function_from_application(value: object) -> ImplementedFunction | None:
    """Return the implemented function represented by a SymPy application."""

    if not isinstance(value, sympy.Function):
        return None
    component = getattr(value.func, "_mt_implemented_function", None)
    if isinstance(component, ImplementedFunction):
        return component
    return None


def collect_implemented_functions(expr: object) -> tuple[ImplementedFunction, ...]:
    """Return implemented functions found in a symbolic expression."""

    functions: list[ImplementedFunction] = []
    seen: set[int] = set()
    for part in _expression_parts(expr):
        if not isinstance(part, sympy.Basic):
            continue
        for node in sympy.preorder_traversal(part):
            component = implemented_function_from_application(node)
            if component is None:
                continue
            identity = id(component)
            if identity in seen:
                continue
            seen.add(identity)
            functions.append(component)
    return tuple(functions)


def implemented_modules(
    functions: tuple[ImplementedFunction, ...],
    context: NumEvaluationContext,
) -> dict[str, object]:
    """Return lambdify module bindings for implemented symbolic functions."""

    modules: dict[str, object] = {}
    for component in functions:
        modules[component._head.__name__] = _implemented_callable(component, context)
    return modules


def _implemented_callable(
    component: ImplementedFunction,
    context: NumEvaluationContext,
) -> Callable[..., object]:
    """Return a lambdify callback for one implemented component."""

    def evaluate(*function_args: object) -> object:
        parameter_values = context.parameter_values(component.required_parameters)
        return component.evaluate(function_args, parameter_values)

    return evaluate


def _expression_parts(expr: object) -> tuple[object, ...]:
    """Return traversable expression parts while preserving matrix entries."""

    if isinstance(expr, sympy.MatrixBase):
        return tuple(expr[row, col] for row in range(expr.rows) for col in range(expr.cols))
    return (expr,)


def _hashable_values(values: tuple[object, ...]) -> tuple[object, ...] | None:
    """Return a hashable call key, or ``None`` for uncacheable values."""

    key: list[object] = []
    for value in values:
        array = np.asarray(value)
        if array.shape != ():
            return None
        item = array.item()
        try:
            hash(item)
        except TypeError:
            return None
        key.append(item)
    return tuple(key)


def _safe_function_name(name: str) -> str:
    """Return a Python identifier suitable for SymPy lambdify output."""

    characters = [character if character.isalnum() else "_" for character in name]
    safe = "".join(characters).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"f_{safe}"
    return safe
