"""Provide public numerical-function boundary primitives.

These primitives manage the conversion between symbolic SymPy expressions
and executable numeric functions, supporting NumPy broadcasting and
compiled evaluation targets.
"""

from __future__ import annotations

__all__ = [
    "CompilationMetadata",
    "ImplementedFunction",
    "LastExecutionMetadata",
    "NumArgSpec",
    "NumEvaluationError",
    "NumFunction",
    "ShapeParseError",
    "ShapeSpec",
]

import importlib
import inspect
import sys
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import sympy
from sympy.core.function import AppliedUndef

from .legacy._disabled import NumArgumentError, NumError, NumNotImplementedError


class NumFunction:
    """Execute keyword-driven numerical evaluations for a compiled expression.

    Parameters
    ----------
    *args : Any
        Positional arguments providing the signature and target callable.
    metadata : CompilationMetadata | None, default=None
        Compilation characteristics and receipt data for the target.
    hints : Mapping[str, Any] | None, default=None
        Runtime execution hints to guide the evaluation backend.
    arg_specs : tuple[NumArgSpec, ...], default=()
        Argument specifications defining the parameter contract.
    used_symbols : frozenset[sympy.Symbol] | None, default=None
        SymPy symbols explicitly utilized by the target logic.
    internal_target : bool, default=False
        Whether the target callable expects an internal environment mapping.

    Attributes
    ----------
    args : tuple[NumArgSpec, ...]
        The public argument signature sequence.
    raw_evaluator : Callable[..., Any]
        The callable target used by this boundary object.
    last_execution_metadata : LastExecutionMetadata
        Track diagnostics and error bounds captured during the latest execution.
    """

    def __init__(
        self,
        *args: Any,
        metadata: CompilationMetadata | None = None,
        hints: Mapping[str, Any] | None = None,
        arg_specs: tuple[NumArgSpec, ...] = (),
        used_symbols: frozenset[sympy.Symbol] | None = None,
        internal_target: bool = False,
    ) -> None:
        # Differentiate between positional tuple definitions and standard arguments
        # to support legacy and modern compiled invocation strategies.
        if args and isinstance(args[0], tuple):
            if len(args) < 2 or not callable(args[1]):
                raise TypeError("NumFunction requires argument specs and a callable.")
            self.arg_specs = tuple(args[0])
            self._target = args[1]
            self.metadata = args[2] if len(args) > 2 else metadata or CompilationMetadata()
            self._positional_target = True
            self._internal_target = False
            self._used_symbols = frozenset(spec.symbol for spec in self.arg_specs)
        else:
            if not args or not callable(args[0]):
                raise TypeError("NumFunction target must be callable.")
            self._target = args[0]
            self.arg_specs = tuple(arg_specs)
            self.metadata = metadata or CompilationMetadata()
            self._positional_target = False
            self._internal_target = bool(internal_target)
            self._used_symbols = frozenset(used_symbols or ())

        # Establish execution state boundaries and runtime caches.
        self.runtime_hints = dict(hints or {})
        self.last_execution_metadata = LastExecutionMetadata()
        self._symbol_to_name = {spec.symbol: spec.name for spec in self.arg_specs}
        self._name_to_symbol = {spec.name: spec.symbol for spec in self.arg_specs}
        self._expected_keys = frozenset(self._name_to_symbol)
        
        # Build the callable public interface.
        self._signature = self._build_public_signature()
        self._public_call = self._build_public_call()

    @property
    def args(self) -> tuple[NumArgSpec, ...]:
        return self.arg_specs

    @property
    def __signature__(self) -> inspect.Signature:
        return self._signature

    @property
    def raw_evaluator(self) -> Callable[..., Any]:
        return self._target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._public_call(*args, **kwargs)

    def __repr__(self) -> str:
        names = ", ".join(spec.name for spec in self.arg_specs)
        return f"NumFunction[{names}]"

    def _build_public_signature(self) -> inspect.Signature:
        parameters = [
            inspect.Parameter(
                spec.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for spec in self.arg_specs
        ]
        return inspect.Signature(parameters)

    def _build_public_call(self) -> Callable[..., Any]:
        # Enforce Python binding rules transparently by wrapping the internal invocation.
        def public_call(*args: Any, **kwargs: Any) -> Any:
            bound_args = self._signature.bind(*args, **kwargs)
            return self._invoke_bound_arguments(bound_args)

        public_call.__signature__ = self._signature  # type: ignore[attr-defined]
        return public_call

    def _invoke_bound_arguments(self, bound_args: inspect.BoundArguments) -> Any:
        symbolic_expression = getattr(self, "symbolic_expression", None)
        
        # Delegate to symbolic evaluation if any bound argument acts as a list 
        # containing SymPy basic types.
        has_symbolic_list = any(
            isinstance(val, list) and any(isinstance(item, sympy.Basic) for item in val)
            for val in bound_args.arguments.values()
        )
        if symbolic_expression is not None and has_symbolic_list:
            return _evaluate_symbolic_list_call(
                symbolic_expression,
                self.arg_specs,
                bound_args.arguments,
            )

        # Coerce public keyword arguments down to runtime arrays or numeric formats 
        # required by internal evaluation targets.
        spec_by_name = {spec.name: spec for spec in self.arg_specs}
        public_kwargs = {
            name: _coerce_runtime_value(value, spec_by_name[name].symbol)
            for name, value in bound_args.arguments.items()
            if name in spec_by_name
        }
        env: dict[sympy.Symbol, Any] = {
            self._name_to_symbol[name]: value
            for name, value in public_kwargs.items()
            if name in self._name_to_symbol
        }
        
        # Reset execution metadata for the current run before invoking the target.
        execution_metadata = LastExecutionMetadata()
        self.last_execution_metadata = execution_metadata

        # Execute the underlying numeric evaluation target safely.
        try:
            if self._internal_target:
                result = self._target(env, execution_metadata)
            elif self._positional_target:
                payload = tuple(public_kwargs[spec.name] for spec in self.arg_specs)
                result = self._target(*payload)
            else:
                result = self._target(**public_kwargs)
        except NumEvaluationError:
            raise
        except NumArgumentError:
            raise
        except Exception as exc:
            msg = (
                "The compiled numerical function could not be evaluated for the supplied inputs. "
                "Please check that argument values are real numeric scalars or NumPy arrays "
                f"with compatible shapes. Details: {exc}"
            )
            raise NumEvaluationError(msg) from exc

        # When resolving constant results against vectorized inputs, ensure the
        # output is broadcast accordingly.
        if self._internal_target and not self._used_symbols and public_kwargs:
            result = _broadcast_constant_to_kwargs(result, public_kwargs)
            
        return _normalize_numeric_result(result)


class ImplementedFunction:
    """Carry a raw numerical implementation behind a passive SymPy function head.

    Parameters
    ----------
    display_name : str | Callable[..., Any]
        The string name of the function or the callable itself if no name is provided.
    raw_implementation : Callable[..., Any] | None, default=None
        The underlying executable target.
    required_parameters : Iterable[str], default=()
        A sequence of parameter names necessary to invoke this function.
    shape_spec : ShapeSpec | Any | None, default=None
        Descriptor governing the multidimensional inputs and outputs.

    Raises
    ------
    TypeError
        If a valid callable implementation is not provided.
    """

    def __init__(
        self,
        display_name: str | Callable[..., Any],
        raw_implementation: Callable[..., Any] | None = None,
        required_parameters: Iterable[str] = (),
        shape_spec: ShapeSpec | Any | None = None,
    ) -> None:
        # Determine the user-facing name and backing callable depending on whether 
        # display_name behaves as a decorator target.
        if callable(display_name) and raw_implementation is None:
            raw_implementation = display_name
            display = getattr(display_name, "__name__", "ImplementedFunction")
        else:
            display = str(display_name)
            
        if raw_implementation is None or not callable(raw_implementation):
            raise TypeError("ImplementedFunction requires a callable implementation.")
            
        self.display_name = display
        self.raw_implementation = raw_implementation
        self.required_parameters = tuple(required_parameters)
        
        if isinstance(shape_spec, ShapeSpec):
            self.shape_spec = shape_spec
        else:
            self.shape_spec = ShapeSpec.scalar() if shape_spec is None else ShapeSpec(shape_spec)
            
        # Validate the public display name through the active SymPy constructor,
        # then create a unique function class for this instance. SymPy caches
        # ``Function("name")`` heads by display name, which would make two
        # implemented functions with the same label indistinguishable.
        sympy.Function(display)
        self._head = type(display, (AppliedUndef,), {})
        # SymPy expressions preserve this head structurally, so ``G(x).func``
        # can recover the implementation during ``Num`` compilation. The
        # marker is private by convention, not a hard immutability boundary:
        # Python code could still reassign the attribute on the function head.
        self._head._mt_implemented_function = self

    def __call__(self, *args: Any) -> Any:
        return self._head(*args)


# ---------------------------------------------------------------------------
# Supporting Primitives and Types
# ---------------------------------------------------------------------------

class ShapeParseError(ValueError):
    """Raised when NumPy rejects a generalized ufunc signature string."""


class NumEvaluationError(NumError):
    """Raised when numeric evaluation fails at runtime."""


@dataclass(frozen=True)
class ShapeSpec:
    """Represent an immutable numerical shape or gufunc signature descriptor.

    Parameters
    ----------
    dims : Any, default=()
        A generalized ufunc signature string, an integer dimension, or an
        iterable of dimensions.

    Raises
    ------
    ShapeParseError
        If the signature string cannot be evaluated by the NumPy parser.
    ValueError
        If provided non-string dimensions are not non-negative integers or None.
    """

    dims: tuple[Any, ...] = ()

    def __init__(self, dims: Any = ()) -> None:
        # Gufunc signatures preserve the blueprint-facing parsing contract while
        # also exposing a scalar shape for compiler metadata.
        if isinstance(dims, str):
            try:
                inputs, outputs = _parse_signature_with_numpy(dims)
            except ValueError as exc:
                raise ShapeParseError(f"Failed parsing signature: {exc}") from exc
                
            input_shapes = tuple(tuple(shape) for shape in inputs)
            output_shapes = tuple(tuple(shape) for shape in outputs)
            is_scalar_only = all(
                len(shape) == 0 for shape in input_shapes + output_shapes
            )
            object.__setattr__(self, "signature", dims)
            object.__setattr__(self, "input_shapes", input_shapes)
            object.__setattr__(self, "output_shapes", output_shapes)
            object.__setattr__(self, "is_scalar_only", is_scalar_only)
            object.__setattr__(self, "dims", ())
            return

        # Plain dimension tuples support the compiler's lightweight runtime
        # shape metadata without introducing a second public shape class.
        if dims is None:
            normalized: tuple[Any, ...] = ()
        elif isinstance(dims, int):
            normalized = (dims,)
        else:
            normalized = tuple(dims)
            
        # Verify shape constraints immediately to ensure subsequent logic can 
        # rely on sanitized shape lengths and bounds.
        for dim in normalized:
            if isinstance(dim, int) and dim < 0:
                raise ValueError("Shape dimensions must be non-negative.")
            if dim is not None and not isinstance(dim, (int, str)):
                raise ValueError("Shape dimensions must be integers, strings, or None.")
                
        object.__setattr__(self, "dims", normalized)
        object.__setattr__(self, "signature", None)
        object.__setattr__(self, "input_shapes", ())
        object.__setattr__(self, "output_shapes", ())
        object.__setattr__(self, "is_scalar_only", len(normalized) == 0)

    @property
    def shape(self) -> tuple[Any, ...]:
        return self.dims

    @classmethod
    def scalar(cls) -> "ShapeSpec":
        """Create a shape descriptor representing a scalar value."""
        return cls(())

    @classmethod
    def from_value(cls, value: Any) -> "ShapeSpec":
        """Extract a shape descriptor from a concrete numeric or array object."""
        return cls(np.asarray(value).shape)

    @classmethod
    def broadcast(cls, *specs: "ShapeSpec") -> "ShapeSpec":
        """Compute the broadcasted shape from concrete shape specifications.

        Parameters
        ----------
        *specs : ShapeSpec
            A variable number of shape specifications to combine.

        Returns
        -------
        ShapeSpec
            The unified broadcast shape.

        Raises
        ------
        ValueError
            If any spec contains an unknown dynamic dimension.
        """
        if not specs:
            return cls.scalar()
            
        concrete: list[tuple[int, ...]] = []
        for spec in specs:
            if any(not isinstance(dim, int) for dim in spec.dims):
                raise ValueError(
                    "Cannot broadcast ShapeSpec values that contain unknown dimensions."
                )
            concrete.append(tuple(spec.dims))
        return cls(np.broadcast_shapes(*concrete))


@dataclass(frozen=True)
class NumArgSpec:
    """Define one strict keyword signature entry for a compiled function.

    Parameters
    ----------
    symbol : sympy.Basic
        The source symbolic representation mapped to this argument.
    name : str
        The public parameter name exposed to Python function callers.
    shape : ShapeSpec | Iterable[Any] | None, default=None
        The expected numeric layout or signature constraint.

    Raises
    ------
    TypeError
        If the symbol is not a valid SymPy representation.
    ValueError
        If the name is empty or not a string.
    """

    symbol: sympy.Basic
    name: str
    shape: ShapeSpec | None = None

    def __init__(
        self,
        symbol: sympy.Basic,
        name: str,
        shape: ShapeSpec | Iterable[Any] | None = None,
    ) -> None:
        if not isinstance(symbol, (sympy.Symbol, sympy.Idx, sympy.IndexedBase)):
            raise TypeError(
                "NumArgSpec.symbol must be a SymPy Symbol, Idx, or IndexedBase."
            )
        if not isinstance(name, str) or not name:
            raise ValueError("NumArgSpec.name must be a non-empty string.")
            
        if shape is None:
            normalized_shape = None
        elif isinstance(shape, ShapeSpec):
            normalized_shape = shape
        else:
            normalized_shape = ShapeSpec(shape)
            
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", normalized_shape)

    @property
    def core_shape(self) -> tuple[Any, ...]:
        if self.shape is None:
            return ()
        return self.shape.shape


@dataclass
class CompilationMetadata:
    """Store compilation characteristics and optional backend receipt data.

    Parameters
    ----------
    backend_name : str | None, default=None
        The numerical library generating the target execution plan.
    execution_plan : str | None, default=None
        Diagnostics string detailing internal compilation sequences.
    shape_spec : ShapeSpec | Any | None, default=None
        The overarching shape returned by the compiled function.
    diagnostics : Iterable[Any], default=()
        Additional auxiliary logs from the compilation backend.
    is_compile_time_constant : bool, default=False
        Whether the evaluation is entirely cached and static.
    uses_runtime_numerical_integration : bool, default=False
        Indicates if the function utilizes numerical solvers at execution.
    compile_time_integrals_evaluated : int, default=0
        Number of integral steps unrolled prior to execution.
    runtime_integrals : int, default=0
        Number of dynamic integral evaluations executed during invocation.
    """

    is_compile_time_constant: bool = False
    uses_runtime_numerical_integration: bool = False
    compile_time_integrals_evaluated: int = 0
    runtime_integrals: int = 0
    backend_name: str | None = None
    execution_plan: str | None = None
    shape_spec: ShapeSpec = field(default_factory=ShapeSpec.scalar)
    diagnostics: tuple[Any, ...] = ()

    def __init__(
        self,
        backend_name: str | None = None,
        execution_plan: str | None = None,
        shape_spec: ShapeSpec | Any | None = None,
        diagnostics: Iterable[Any] = (),
        *,
        is_compile_time_constant: bool = False,
        uses_runtime_numerical_integration: bool = False,
        compile_time_integrals_evaluated: int = 0,
        runtime_integrals: int = 0,
    ) -> None:
        self.is_compile_time_constant = is_compile_time_constant
        self.uses_runtime_numerical_integration = uses_runtime_numerical_integration
        self.compile_time_integrals_evaluated = compile_time_integrals_evaluated
        self.runtime_integrals = runtime_integrals
        self.backend_name = backend_name
        self.execution_plan = execution_plan
        
        if shape_spec is None:
            self.shape_spec = ShapeSpec.scalar()
        elif isinstance(shape_spec, ShapeSpec):
            self.shape_spec = shape_spec
        else:
            self.shape_spec = ShapeSpec(shape_spec)
            
        self.diagnostics = tuple(diagnostics)


@dataclass
class LastExecutionMetadata:
    """Track diagnostics and error bounds captured during the latest execution."""

    integration_error_bound: float | np.ndarray | None = None
    integration_error_bounds: list[float] = field(default_factory=list)
    integration_call_count: int = 0
    runtime_logs: list[dict[str, Any]] = field(default_factory=list)

    def record_integration_error(self, error: Any) -> None:
        """Log a new integration error boundary and update aggregate diagnostics."""
        try:
            error_float = float(np.asarray(error))
        # ISSUE: Catching generic Exception suppresses systemic errors or interrupts.
        except Exception:
            return
            
        if error_float < 0:
            error_float = abs(error_float)
            
        self.integration_error_bounds.append(error_float)
        self.integration_call_count += 1
        
        if self.integration_error_bound is None:
            self.integration_error_bound = error_float
            return
            
        try:
            self.integration_error_bound = max(
                float(self.integration_error_bound), error_float
            )
        # ISSUE: Catching generic Exception suppresses systemic errors or interrupts.
        except Exception:
            self.integration_error_bound = error_float

    def record_runtime_log(self, category: str, message: str, **details: Any) -> None:
        """Append a runtime diagnostic entry for the current evaluation."""
        entry = {"category": category, "message": message}
        entry.update(details)
        self.runtime_logs.append(entry)


# ---------------------------------------------------------------------------
# Low-Level Evaluation Helpers
# ---------------------------------------------------------------------------

def _normalize_numeric_result(result: Any) -> Any:
    """Normalize internal results into Python scalars or NumPy arrays."""
    if isinstance(result, np.ndarray):
        if result.shape == ():
            return float(result)
        return result
        
    if isinstance(result, np.generic):
        return result
        
    if isinstance(result, (int, float, bool)):
        return float(result) if not isinstance(result, bool) else bool(result)
        
    if isinstance(result, sympy.Basic):
        try:
            return float(result)
        # ISSUE: Catching generic Exception masks symbolic conversion failures.
        except Exception:
            return result
            
    array = np.asarray(result)
    if array.shape == () and np.issubdtype(array.dtype, np.number):
        return float(array)
    return result


def _coerce_runtime_value(value: Any, symbol: sympy.Basic) -> Any:
    """Convert public values into the numeric type promised by one symbol."""
    if isinstance(value, sympy.Basic) and not value.free_symbols:
        try:
            value = float(value)
        # ISSUE: Catching generic Exception masks float coercion faults.
        except Exception:
            return value
    if isinstance(value, list):
        value = np.asarray(value)
    if _symbol_requires_integer(symbol):
        return _coerce_integer_runtime_value(value)
    return value


def _symbol_requires_integer(symbol: sympy.Basic) -> bool:
    """Return whether a public argument symbol declares integer values."""
    return isinstance(symbol, sympy.Idx) or getattr(symbol, "is_integer", False) is True


def _coerce_integer_runtime_value(value: Any) -> Any:
    """Round numeric runtime values for arguments declared with integer symbols."""
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if array.shape == ():
        try:
            return int(round(float(array)))
        except Exception:
            return value
    try:
        return np.rint(array).astype(int)
    except Exception:
        return value


def _broadcast_constant_to_kwargs(result: Any, kwargs: Mapping[str, Any]) -> Any:
    """Propagate scalar constants across layout dimensions of keyword arguments."""
    arrays = [np.asarray(value) for value in kwargs.values() if np.asarray(value).shape != ()]
    if not arrays:
        return result
        
    # Standardize result shapes against the combined broadcast dimensions of inputs.
    try:
        shape = np.broadcast_shapes(*[array.shape for array in arrays])
    except ValueError as exc:
        raise NumArgumentError(
            "Unused argument values could not be broadcast together. "
            "Please use NumPy-compatible shapes."
        ) from exc
        
    scalar = np.asarray(result)
    if scalar.shape != ():
        return result
        
    return np.full(shape, float(scalar), dtype=float)


def _evaluate_symbolic_list_call(
    expression: sympy.Basic,
    arg_specs: tuple[NumArgSpec, ...],
    arguments: Mapping[str, Any],
) -> list[sympy.Basic]:
    """Evaluate a compiled scalar expression entrywise over symbolic Python lists."""
    # Ensure ordered iteration mirrors public invocation arguments.
    ordered_values = [
        arguments[spec.name]
        for spec in arg_specs
        if spec.name in arguments
    ]
    arrays = [np.asarray(value, dtype=object) for value in ordered_values]
    broadcasted = np.broadcast_arrays(*arrays) if arrays else []
    
    if not broadcasted:
        return [expression]

    # Map the scalar blueprint sequentially across indices in the broadcast object.
    result: list[sympy.Basic] = []
    for index in np.ndindex(broadcasted[0].shape):
        substitutions = {}
        for spec, array in zip(arg_specs, broadcasted):
            item = array[index]
            substitutions[spec.symbol] = item.item() if hasattr(item, "item") else item
            
        result.append(
            expression.xreplace(
                {
                    symbol: sympy.sympify(value)
                    for symbol, value in substitutions.items()
                }
            )
        )
    return result


def _parse_signature_with_numpy(signature: str) -> tuple[Any, Any]:
    """Call NumPy's private gufunc parser through the compatibility path."""
    module = importlib.import_module("numpy.lib.function_base")
    parser = getattr(module, "_parse_gufunc_signature")
    return parser(signature)


def _install_numpy_parser_compatibility() -> None:
    """Expose NumPy's gufunc parser through the legacy path when necessary."""
    try:
        importlib.import_module("numpy.lib.function_base")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        modern_module = importlib.import_module("numpy.lib._function_base_impl")
        parser = getattr(modern_module, "_parse_gufunc_signature")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover
        raise ImportError(
            "Could not import NumPy's generalized ufunc signature parser."
        ) from exc

    compat_module = types.ModuleType("numpy.lib.function_base")
    compat_module._parse_gufunc_signature = parser  # type: ignore[attr-defined]
    sys.modules["numpy.lib.function_base"] = compat_module

    numpy_lib = importlib.import_module("numpy.lib")
    setattr(numpy_lib, "function_base", compat_module)


# Register backward-compatible NumPy signature parser for ShapeSpec behavior.
_install_numpy_parser_compatibility()
