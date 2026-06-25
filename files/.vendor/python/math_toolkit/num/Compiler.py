"""Compile declared SymPy expressions into numerical functions.

This module provides the core compilation infrastructure to transform symbolic
expressions into callable objects that support numpy broadcasting and runtime
numerical integration via SciPy.
"""
from __future__ import annotations

__all__ = [
    "NumCompiler",
    "NumUnsupportedExpressionError",
    "NumArgumentError",
    "NumEvaluationError",
    "CompilationContext",
    "ImplementedFunction",
    "LoweredCallable",
    "PinnedExpression",
]

import contextlib
import contextvars
import copy
import inspect
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import sympy

# ISSUE: Broad exception catch masks potential environment or syntax errors in optional dependencies.
try:  # SciPy is an optional runtime dependency until an Integral is evaluated.
    import scipy.integrate as _scipy_integrate
except Exception:  # pragma: no cover - exercised only in environments sans SciPy.
    _scipy_integrate = None

from .legacy._disabled import (
    NumArgumentError,
    NumUnsupportedExpressionError,
)
from .numfunction_implementedfunction import (
    CompilationMetadata,
    ImplementedFunction,
    LastExecutionMetadata,
    NumArgSpec,
    NumEvaluationError,
    NumFunction,
)

# ---------------------------------------------------------------------------
# Core Public Compiler & Evaluator Entry Points
# ---------------------------------------------------------------------------


class NumCompiler:
    """Compile SymPy expressions into keyword-driven numerical function evaluators.

    A compiler instance stores long-lived configuration such as custom extension
    registries, while runtime settings are provided during individual compilation
    tasks.

    Parameters
    ----------
    extensions : Mapping[Any, Any] | None, default=None
        Optional custom compiler extension registry copied at construction time.

    Attributes
    ----------
    extensions : Mapping[Any, Any]
        Immutable read-only view of the compiler's extension registry.
    """

    def __init__(self, extensions: Mapping[Any, Any] | None = None) -> None:
        # Construct the baseline extension registry and merge any user-provided hooks.
        merged_extensions = _default_num_extensions()
        merged_extensions.update(dict(extensions or {}))
        self._extensions = _safe_deepcopy(merged_extensions)

    @property
    def extensions(self) -> Mapping[Any, Any]:
        return _FrozenExtensions(self._extensions)

    def compile(
        self,
        expression: sympy.Basic,
        arg_specs: tuple[NumArgSpec, ...] = (),
        hints: Mapping[str, Any] | None = None,
    ) -> NumFunction:
        """Compile a symbolic expression into a keyword-driven numerical function.

        Parameters
        ----------
        expression : sympy.Basic
            SymPy expression or sympifiable object to compile.
        arg_specs : tuple[NumArgSpec, ...], default=()
            Public keyword signature specification for the resulting function.
        hints : Mapping[str, Any] | None, default=None
            Advisory execution and optimization settings for this compilation.

        Returns
        -------
        NumFunction
            Callable numeric function wrapper that executes the compiled expression.

        Raises
        ------
        TypeError
            If the expression cannot be parsed into a SymPy expression tree.
        NumArgumentError
            If argument specifications do not align with the expression free symbols.
        """
        # Standardize argument rules and initialize the compilation configuration map.
        specs = tuple(arg_specs or ())
        self._validate_arg_specs(specs)
        hints_dict = _normalize_compiler_hints(hints or {})

        # Normalize the primary input value into a proper SymPy expression tree node.
        try:
            expr = sympy.sympify(expression)
        except Exception:
            # ISSUE: Catch-all exception block hides specific sympify parse failures.
            expr = expression

        # Fallback to extensions if the root object remains unrecognized as a basic expression.
        if not isinstance(expr, sympy.Basic):
            extension_result = self._compile_root_extension(
                expression=expression,
                arg_specs=specs,
                hints=hints_dict,
            )
            if extension_result is not None:
                return extension_result
            raise TypeError("expression must be a SymPy expression or sympifiable value.")

        # Validate array-rank consistency before choosing either the modern
        # generated path or the legacy free-index execution plan.
        if expr.has(sympy.Indexed):
            _validate_indexed_base_shapes(expr)

        # Delegate free-index array outputs to the dedicated execution path.
        # Finite reductions with indexed reads continue through the generated
        # compiler so their reduction axis can stay vectorized.
        if _requires_indexed_execution(expr, specs):
            return _compile_indexed_expression(expr, specs, hints_dict)

        # Establish which free symbols actively require keyword arguments for evaluation.
        expression_free_symbols = _runtime_free_symbols(expr)
        spec_symbols = {spec.symbol for spec in specs}
        # Validate that no free symbols are left undefined by the public signature.
        missing = expression_free_symbols - spec_symbols
        if missing:
            names = ", ".join(sorted(str(sym) for sym in missing))
            raise NumArgumentError(
                f"Missing argument specification for symbol(s): {names}. "
                "Add each missing free symbol to arg_specs."
            )

        # Verify that every free symbol is accounted for by the public signature.
        # Extra explicit parameters are allowed so callers can keep a stable arity.
        missing = expression_free_symbols - spec_symbols
        if missing:
            names = ", ".join(sorted(str(sym) for sym in missing))
            raise NumArgumentError(
                f"Missing argument specification for symbol(s): {names}. "
                "Add each missing free symbol to arg_specs."
            )

        # Initialize the global state tracker for this compilation traversal.
        metadata = CompilationMetadata(backend_name="sympy_lambdify")
        lowering_state = _LoweringState(_CompileLogSink())
        ctx = CompilationContext(
            compiler=self,
            pinned_expression=PinnedExpression(root_expression=expr, coordinate_path=()),
            arg_specs=specs,
            hints=hints_dict,
            lowering_state=lowering_state,
        )

        # Execute the recursive compilation routine to generate the execution plan.
        compiled = self._compile_node(ctx, metadata)
        metadata.diagnostics = tuple(lowering_state.log_sink.compile_logs)
        metadata.uses_runtime_numerical_integration = compiled.has_runtime_integral
        metadata.is_compile_time_constant = not compiled.used_symbols and not compiled.has_runtime_integral

        # Construct and return the wrapper that safely executes the compiled blocks.
        compiled_function = NumFunction(
            compiled.evaluator,
            metadata=metadata,
            hints=hints_dict,
            arg_specs=specs,
            used_symbols=compiled.used_symbols,
            internal_target=True,
        )
        compiled_function.symbolic_expression = expr
        return compiled_function

    @staticmethod
    def _validate_arg_specs(specs: tuple[NumArgSpec, ...]) -> None:
        """Validate structural uniqueness of parameter rules within argument specifications."""
        names: set[str] = set()
        symbols: set[sympy.Symbol] = set()
        
        # Verify that each signature requirement provides unique names and symbols.
        for spec in specs:
            if not hasattr(spec, "symbol") or not hasattr(spec, "name"):
                raise TypeError("arg_specs must contain objects with symbol and name fields.")
            if spec.name in names:
                raise NumArgumentError(f"Duplicate argument name '{spec.name}' in arg_specs.")
            if spec.symbol in symbols:
                raise NumArgumentError(f"Duplicate argument symbol '{spec.symbol}' in arg_specs.")
            names.add(spec.name)
            symbols.add(spec.symbol)

    def _compile_node(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Evaluate a generic tree node and return its compilation execution strategy."""
        node = ctx.current_node

        # Finite reductions still own a local environment for their bound
        # variables. Their expression and bound children are compiled through
        # the generated path, so nested runtime operators keep seeing those
        # local values without requiring them as public arguments.
        if isinstance(node, (sympy.Sum, sympy.Product)):
            return self._compile_symbolic_reduction(ctx, metadata)

        return self._compile_generated_expression(ctx, metadata)

    def _compile_generated_expression(
        self,
        ctx: CompilationContext,
        metadata: CompilationMetadata,
    ) -> _CompiledNode:
        """Compile one expression subtree into a generated NumPy callable."""
        state = _ensure_lowering_state(ctx)
        module_map: dict[str, Callable[..., Any]] = {}
        lowered = self._lower_expression(ctx, metadata, module_map)
        argument_symbols = _ordered_symbols(lowered.expression.free_symbols, ctx.arg_specs)

        # Lambdify owns ordinary arithmetic code generation. The local module
        # mapping contains the explicit Python-call boundaries inserted during
        # lowering for runtime operators and custom escape hatches.
        try:
            generated = sympy.lambdify(
                argument_symbols,
                lowered.expression,
                modules=[module_map, "numpy"],
                cse=_generated_cse_option(ctx.hints),
            )
        except Exception as exc:
            raise self._unsupported(ctx.current_node) from exc

        def eval_generated(
            env: dict[sympy.Symbol, Any],
            exec_meta: LastExecutionMetadata,
        ) -> Any:
            values = []
            for symbol in argument_symbols:
                if symbol not in env:
                    raise NumArgumentError(f"Missing value for symbol '{symbol}'.")
                values.append(env[symbol])

            with state.log_sink.activate(exec_meta):
                try:
                    return generated(*values)
                except (NumArgumentError, NumEvaluationError):
                    raise
                except Exception as exc:
                    raise NumEvaluationError(_friendly_runtime_message(exc)) from exc

        return _CompiledNode(
            eval_generated,
            used_symbols=frozenset(argument_symbols),
            has_runtime_integral=lowered.has_runtime_integral,
            direct_callable=generated,
            direct_symbols=argument_symbols,
        )

    def _lower_expression(
        self,
        ctx: CompilationContext,
        metadata: CompilationMetadata,
        module_map: dict[str, Callable[..., Any]],
    ) -> _LoweredExpression:
        """Lower runtime boundaries and leave ordinary SymPy math for lambdify."""
        node = ctx.current_node
        state = _ensure_lowering_state(ctx)

        # Local compile hooks have first claim over their subtree. A claimed
        # subtree becomes one generated function call backed by the returned
        # vectorized Python callable.
        local_escape = _get_local_escape_hatch(node)
        if local_escape is not None:
            try:
                result = local_escape(ctx)
            except TypeError as exc:
                raise NumUnsupportedExpressionError(
                    "I ran into a part of your expression I don't know how to convert: "
                    f"{type(node).__name__}. You didn't do anything wrong! Its _mt_compile method "
                    "could not be called with the expected CompilationContext."
                ) from exc
            boundary = _coerce_lowering_result(
                result,
                ctx,
                state,
                source=f"{type(node).__name__}._mt_compile",
            )
            expression = _register_lowered_callable(
                boundary.lowered,
                state,
                module_map,
                category="compile",
            )
            return _LoweredExpression(expression, boundary.has_runtime_integral)

        # Registered extensions get the next deterministic first-match claim.
        extension = _match_lowering_extension(self._extensions, ctx)
        if extension is not None:
            extension_name, handler = extension
            result = handler(ctx)
            boundary = _coerce_lowering_result(
                result,
                ctx,
                state,
                source=f"{extension_name} extension",
            )
            expression = _register_lowered_callable(
                boundary.lowered,
                state,
                module_map,
                category="extension",
                extension_name=extension_name,
            )
            return _LoweredExpression(expression, boundary.has_runtime_integral)

        # Runtime operators compile their children and return a vectorized
        # placeholder callable over external runtime parameters.
        if isinstance(node, sympy.Integral):
            lowered = self._lower_runtime_integral(ctx, metadata)
            expression = _register_lowered_callable(
                lowered,
                state,
                module_map,
                category="extension",
                extension_name="Integral",
            )
            return _LoweredExpression(expression, has_runtime_integral=True)

        if isinstance(node, sympy.Indexed):
            return self._lower_indexed_lookup(ctx, metadata, module_map)

        if isinstance(node, (sympy.Sum, sympy.Product)):
            compiled = self._compile_symbolic_reduction(ctx, metadata)
            args = _ordered_symbols(compiled.used_symbols, ctx.arg_specs)

            def evaluate_reduction(*values: Any) -> Any:
                exec_meta = state.log_sink.current_metadata()
                if exec_meta is None:
                    exec_meta = LastExecutionMetadata()
                env = dict(zip(args, values))
                return compiled.evaluator(env, exec_meta)

            expression = _register_lowered_callable(
                LoweredCallable(callable=evaluate_reduction, args=args),
                state,
                module_map,
                category="extension",
                extension_name="Sum" if isinstance(node, sympy.Sum) else "Product",
            )
            return _LoweredExpression(expression, compiled.has_runtime_integral)

        # ImplementedFunction calls are explicit Python-call boundaries. Their
        # display names never participate in generated placeholder names.
        implemented = getattr(getattr(node, "func", None), "_mt_implemented_function", None)
        if isinstance(node, sympy.Function) and isinstance(implemented, ImplementedFunction):
            lowered_children = [
                self._lower_expression(ctx.with_descend(i), metadata, module_map)
                for i in range(len(node.args))
            ]
            child_expressions = tuple(child.expression for child in lowered_children)
            cached_call = _cached_vectorized_callable(implemented.raw_implementation)

            def call_implemented_function(*values: Any) -> Any:
                return cached_call(*values)

            expression = _register_lowered_callable(
                LoweredCallable(callable=call_implemented_function, args=child_expressions),
                state,
                module_map,
                category="implementedfunction",
            )
            return _LoweredExpression(
                expression,
                any(child.has_runtime_integral for child in lowered_children),
            )

        # Hard unsupported calculus constructs should fail before lambdify can
        # produce a less instructive runtime error.
        if isinstance(node, (sympy.Derivative, sympy.Limit)):
            raise self._unsupported(node)

        if isinstance(node, sympy.Piecewise):
            pairs = []
            has_runtime = False
            for pair_index, (_expr, _cond) in enumerate(node.args):
                pair_ctx = ctx.with_descend(pair_index)
                lowered_expr = self._lower_expression(pair_ctx.with_descend(0), metadata, module_map)
                lowered_cond = self._lower_expression(pair_ctx.with_descend(1), metadata, module_map)
                pairs.append((lowered_expr.expression, lowered_cond.expression))
                has_runtime = (
                    has_runtime
                    or lowered_expr.has_runtime_integral
                    or lowered_cond.has_runtime_integral
                )
            return _LoweredExpression(sympy.Piecewise(*pairs), has_runtime)

        if isinstance(node, sympy.Function) and node.func not in _NUMPY_FUNCTIONS:
            raise self._unsupported(node)

        # Ordinary atoms pass through unchanged when lambdify knows how to
        # render them; unknown custom atoms require an explicit boundary.
        if not getattr(node, "args", ()):
            if (
                isinstance(node, (sympy.Symbol, sympy.Number, sympy.NumberSymbol))
                or isinstance(node, sympy.Idx)
                or isinstance(node, sympy.logic.boolalg.BooleanAtom)
                or _is_positive_infinity(node)
                or _is_negative_infinity(node)
            ):
                return _LoweredExpression(node, has_runtime_integral=False)
            raise self._unsupported(node)

        lowered_children = [
            self._lower_expression(ctx.with_descend(i), metadata, module_map)
            for i in range(len(node.args))
        ]
        try:
            rebuilt = node.func(*(child.expression for child in lowered_children))
        except Exception as exc:
            raise self._unsupported(node) from exc
        return _LoweredExpression(
            rebuilt,
            any(child.has_runtime_integral for child in lowered_children),
        )

    def _lower_indexed_lookup(
        self,
        ctx: CompilationContext,
        metadata: CompilationMetadata,
        module_map: dict[str, Callable[..., Any]],
    ) -> _LoweredExpression:
        """Lower one ``IndexedBase`` read to a vectorized NumPy lookup."""
        node = ctx.current_node
        if not isinstance(node, sympy.Indexed):
            raise self._unsupported(node)

        # Lower index expressions through the same machinery as ordinary
        # arithmetic, then make the base array an explicit runtime argument.
        lowered_indices = [
            self._lower_expression(ctx.with_descend(index + 1), metadata, module_map)
            for index in range(len(node.indices))
        ]
        args = (node.base, *(item.expression for item in lowered_indices))

        expression = _register_lowered_callable(
            LoweredCallable(callable=_vectorized_indexed_lookup, args=args),
            state=_ensure_lowering_state(ctx),
            module_map=module_map,
            category="extension",
            extension_name="Indexed",
        )
        return _LoweredExpression(
            expression,
            any(item.has_runtime_integral for item in lowered_indices),
        )

    def _lower_runtime_integral(
        self,
        ctx: CompilationContext,
        metadata: CompilationMetadata,
    ) -> LoweredCallable:
        """Lower a SymPy integral into a vectorized SciPy runtime callable."""
        node: sympy.Integral = ctx.current_node  # type: ignore[assignment]
        if _scipy_integrate is None:
            raise NumUnsupportedExpressionError(
                "I ran into a part of your expression I don't know how to convert: Integral. "
                "You didn't do anything wrong! Runtime integration requires SciPy to be installed."
            )

        metadata.runtime_integrals += 1
        state = _ensure_lowering_state(ctx)
        integrand_node = self._compile_node(ctx.with_descend(0), metadata)
        integrand_function = _compile_integral_integrand(node, integrand_node)

        # Bounds are compiled once here. Their evaluators are reused for every
        # scalar parameter point and never rebuilt inside SciPy callbacks.
        compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]] = []
        integration_symbols: set[sympy.Symbol] = set()
        child_used_symbols: set[sympy.Symbol] = set(integrand_node.used_symbols)
        for limit_index, limit in enumerate(node.limits, start=1):
            if len(limit) != 3:
                raise self._unsupported(node)
            var, _lower_expr, _upper_expr = limit
            if not isinstance(var, sympy.Symbol):
                raise self._unsupported(node)
            limit_ctx = ctx.with_descend(limit_index)
            lower_node = self._compile_node(limit_ctx.with_descend(1), metadata)
            upper_node = self._compile_node(limit_ctx.with_descend(2), metadata)
            compiled_limits.append((var, lower_node, upper_node))
            integration_symbols.add(var)
            child_used_symbols.update(lower_node.used_symbols)
            child_used_symbols.update(upper_node.used_symbols)

        external_symbols = frozenset(child_used_symbols - integration_symbols)
        args = _ordered_symbols(external_symbols, ctx.arg_specs)
        runtime_symbols = frozenset(args)
        hints = dict(ctx.hints)
        symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}
        scalar_cache: dict[tuple[tuple[str, Any], ...], float] = {}
        vector_cache: dict[tuple[tuple[str, Any], ...], np.ndarray] = {}
        policy = _integral_imprecision_policy(hints)

        def integrate_scalar_point(
            point_env: dict[sympy.Symbol, Any],
            exec_meta: LastExecutionMetadata,
        ) -> float:
            cache_key = _runtime_integral_cache_key(runtime_symbols, point_env)
            if cache_key is not None and cache_key in scalar_cache:
                return scalar_cache[cache_key]
            if _integral_integrator(hints) == "Sampled":
                value = _sample_integral_at_parameter_point(
                    compiled_limits,
                    integrand_function,
                    integrand_node,
                    point_env,
                    exec_meta,
                    hints,
                )
            else:
                try:
                    value = _integrate_at_parameter_point(
                        compiled_limits,
                        integrand_function,
                        point_env,
                        exec_meta,
                        hints,
                        symbol_to_name,
                        runtime_symbols,
                    )
                except NumEvaluationError as exc:
                    try:
                        value = _sample_integral_at_parameter_point(
                            compiled_limits,
                            integrand_function,
                            integrand_node,
                            point_env,
                            exec_meta,
                            hints,
                        )
                    except NumEvaluationError:
                        if policy == "Raise":
                            raise exc
                        _record_integral_failure(exec_meta, policy, exc, point_env, symbol_to_name, runtime_symbols)
                        value = np.nan
            if cache_key is not None:
                scalar_cache[cache_key] = float(value)
            return float(value)

        def evaluate_integral(*values: Any) -> Any:
            if len(values) != len(args):
                raise NumArgumentError("Runtime integral received the wrong number of arguments.")
            exec_meta = state.log_sink.current_metadata()
            if exec_meta is None:
                exec_meta = LastExecutionMetadata()
            env = dict(zip(args, values))
            try:
                shape_symbols, broadcast_shape, broadcasted_values = (
                    _runtime_parameter_broadcast_values(args, env, node)
                )
            except ValueError as exc:
                names = ", ".join(
                    symbol_to_name.get(sym, str(sym)) for sym in args if sym in env
                )
                raise NumArgumentError(
                    f"Arguments for integral could not be broadcast together ({names}). "
                    "Please use NumPy-compatible shapes."
                ) from exc

            if not shape_symbols:
                point_env = dict(env)
                return integrate_scalar_point(point_env, exec_meta)

            vector_cache_key = _runtime_integral_vector_cache_key(args, env)
            if vector_cache_key is not None and vector_cache_key in vector_cache:
                return vector_cache[vector_cache_key].copy()

            indexed_ranks = _indexed_base_ranks(node)
            if _integral_integrator(hints) == "Sampled":
                sampled_result = _sample_integral_at_parameter_grid(
                    compiled_limits,
                    integrand_node,
                    env,
                    exec_meta,
                    hints,
                    broadcast_shape,
                    broadcasted_values,
                )
                _seed_scalar_integral_cache_from_broadcast_result(
                    runtime_symbols,
                    env,
                    broadcasted_values,
                    indexed_ranks,
                    sampled_result,
                    scalar_cache,
                )
                if vector_cache_key is not None:
                    _store_limited_vector_integral_cache(
                        vector_cache,
                        vector_cache_key,
                        sampled_result,
                    )
                return sampled_result

            scalar_cached_result = _evaluate_from_scalar_integral_cache(
                runtime_symbols,
                env,
                broadcasted_values,
                broadcast_shape,
                indexed_ranks,
                scalar_cache,
                integrate_scalar_point,
                exec_meta,
            )
            if scalar_cached_result is not None:
                return scalar_cached_result

            vectorized_result = None
            if _integral_vector_valued_enabled(hints):
                vectorized_result = _integrate_vectorized_1d_with_cubature(
                    compiled_limits,
                    integrand_node,
                    env,
                    exec_meta,
                    hints,
                    symbol_to_name,
                    runtime_symbols,
                    shape_symbols,
                    broadcast_shape,
                    broadcasted_values,
                )
                if vectorized_result is None:
                    vectorized_result = _integrate_vectorized_1d_with_quad_vec(
                        compiled_limits,
                        integrand_node,
                        env,
                        exec_meta,
                        hints,
                        symbol_to_name,
                        runtime_symbols,
                        shape_symbols,
                        broadcast_shape,
                        broadcasted_values,
                    )
            if vectorized_result is not None:
                _seed_scalar_integral_cache_from_broadcast_result(
                    runtime_symbols,
                    env,
                    broadcasted_values,
                    indexed_ranks,
                    vectorized_result,
                    scalar_cache,
                )
                if vector_cache_key is not None:
                    _store_limited_vector_integral_cache(
                        vector_cache,
                        vector_cache_key,
                        vectorized_result,
                    )
                return vectorized_result

            result = np.empty(broadcast_shape, dtype=float)
            for index in np.ndindex(result.shape):
                point_env = _runtime_integral_scalar_point_env(
                    env,
                    broadcasted_values,
                    indexed_ranks,
                    index,
                )
                result[index] = integrate_scalar_point(point_env, exec_meta)
            if vector_cache_key is not None:
                _store_limited_vector_integral_cache(vector_cache, vector_cache_key, result)
            return result

        return LoweredCallable(callable=evaluate_integral, args=args)

    def _compile_extension_node(self, ctx: CompilationContext) -> _CompiledNode | None:
        """Route node compilation out to registry-level extensions if matching rules exist."""
        node = ctx.current_node
        entry = _lookup_extension_entry(self._extensions, node)
        if entry is None:
            return None

        handler = entry.get("handler") if isinstance(entry, Mapping) else entry
        if not callable(handler):
            return None
        result = handler(ctx)
        
        # Package raw function configurations back into managed nodes.
        if isinstance(result, ImplementedFunction):
            symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}
            used_symbols = frozenset(symbol_to_name)

            def eval_extension(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
                kwargs = {name: env[sym] for sym, name in symbol_to_name.items() if sym in env}
                return result.raw_implementation(**kwargs)

            return _CompiledNode(eval_extension, used_symbols=used_symbols)
            
        if isinstance(result, NumFunction):
            used = getattr(result, "_used_symbols", frozenset())

            def eval_num_function(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
                kwargs = {spec.name: env[spec.symbol] for spec in result.arg_specs if spec.symbol in env}
                value = result(**kwargs)
                if result.last_execution_metadata.integration_error_bound is not None:
                    exec_meta.record_integration_error(result.last_execution_metadata.integration_error_bound)
                return value

            return _CompiledNode(
                eval_num_function,
                used_symbols=frozenset(used),
                has_runtime_integral=result.metadata.uses_runtime_numerical_integration,
            )
            
        if callable(result):
            symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}
            used_symbols = frozenset(symbol_to_name)

            def eval_callable(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
                kwargs = {name: env[sym] for sym, name in symbol_to_name.items() if sym in env}
                return result(**kwargs)

            return _CompiledNode(eval_callable, used_symbols=used_symbols)
            
        return None

    def _compile_root_extension(
        self,
        *,
        expression: Any,
        arg_specs: tuple[NumArgSpec, ...],
        hints: Mapping[str, Any],
    ) -> NumFunction | None:
        """Give registered extensions one chance to handle non-Basic root objects."""
        entry = _lookup_extension_entry(self._extensions, expression)
        if entry is None:
            return None

        handler = entry.get("handler") if isinstance(entry, Mapping) else entry
        if not callable(handler):
            return None
            
        result = handler(expression)
        if isinstance(result, NumFunction):
            return result
        if callable(result):
            return NumFunction(
                result,
                metadata=CompilationMetadata(),
                hints=dict(hints),
                arg_specs=arg_specs,
                used_symbols=frozenset(spec.symbol for spec in arg_specs),
                internal_target=False,
            )
        return None

    def _compile_function(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Map standard functional signatures over to concrete target NumPy callables."""
        node = ctx.current_node
        implemented = getattr(node.func, "_mt_implemented_function", None)
        if isinstance(implemented, ImplementedFunction):
            children = [self._compile_node(ctx.with_descend(i), metadata) for i in range(len(node.args))]
            cached_call = _cached_vectorized_callable(implemented.raw_implementation)

            def eval_implemented_function(
                env: dict[sympy.Symbol, Any],
                exec_meta: LastExecutionMetadata,
            ) -> Any:
                args = [child.evaluator(env, exec_meta) for child in children]
                return cached_call(*args)

            return _merge_node(eval_implemented_function, children)

        np_func = _NUMPY_FUNCTIONS.get(node.func)
        if np_func is None:
            raise self._unsupported(node)
            
        children = [self._compile_node(ctx.with_descend(i), metadata) for i in range(len(node.args))]

        def eval_function(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            args = [child.evaluator(env, exec_meta) for child in children]
            return np_func(*args)

        return _merge_node(eval_function, children)

    def _compile_relation(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Establish runtime comparison logic execution paths between expression sides."""
        node = ctx.current_node
        lhs = self._compile_node(ctx.with_descend(0), metadata)
        rhs = self._compile_node(ctx.with_descend(1), metadata)

        def eval_relation(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            left = lhs.evaluator(env, exec_meta)
            right = rhs.evaluator(env, exec_meta)
            if isinstance(node, sympy.StrictLessThan):
                return np.less(left, right)
            if isinstance(node, sympy.LessThan):
                return np.less_equal(left, right)
            if isinstance(node, sympy.StrictGreaterThan):
                return np.greater(left, right)
            if isinstance(node, sympy.GreaterThan):
                return np.greater_equal(left, right)
            if isinstance(node, sympy.Equality):
                return np.equal(left, right)
            return np.not_equal(left, right)

        return _merge_node(eval_relation, [lhs, rhs])

    def _compile_piecewise(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Formulate vectorized ternary conditional loops to evaluate piecewise expressions."""
        node = ctx.current_node
        expr_nodes: list[_CompiledNode] = []
        cond_nodes: list[_CompiledNode] = []
        
        # Traverse sub-conditions to map them into executable pair segments.
        for i, (_expr, _cond) in enumerate(node.args):
            pair_ctx = ctx.with_descend(i)
            expr_nodes.append(self._compile_node(pair_ctx.with_descend(0), metadata))
            cond_nodes.append(self._compile_node(pair_ctx.with_descend(1), metadata))

        def eval_piecewise(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            result = None
            for expr_node, cond_node in reversed(list(zip(expr_nodes, cond_nodes))):
                expr_value = expr_node.evaluator(env, exec_meta)
                cond_value = cond_node.evaluator(env, exec_meta)
                if result is None:
                    result = expr_value
                else:
                    result = np.where(cond_value, expr_value, result)
            return result

        return _merge_node(eval_piecewise, expr_nodes + cond_nodes)

    def _compile_boolean_operation(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Compile composite boolean predicates used by piecewise conditions."""
        node = ctx.current_node
        children = [self._compile_node(ctx.with_descend(i), metadata) for i in range(len(node.args))]

        # Convert SymPy's boolean tree into NumPy-compatible predicates so
        # piecewise conditions work for both scalar and broadcasted inputs.
        def eval_boolean(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            values = [np.asarray(child.evaluator(env, exec_meta), dtype=bool) for child in children]
            if isinstance(node, sympy.And):
                result = values[0]
                for value in values[1:]:
                    result = np.logical_and(result, value)
                return result
            if isinstance(node, sympy.Or):
                result = values[0]
                for value in values[1:]:
                    result = np.logical_or(result, value)
                return result
            return np.logical_not(values[0])

        return _merge_node(eval_boolean, children)

    def _compile_integral(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Route definite integrals to the numerical runtime integration path."""
        return self._compile_generated_expression(ctx, metadata)

    def _compile_symbolic_reduction(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Compile finite ``Sum`` and ``Product`` expressions as numeric loops."""
        node = ctx.current_node
        if not isinstance(node, (sympy.Sum, sympy.Product)):
            raise self._unsupported(node)

        expression_node = self._compile_node(ctx.with_descend(0), metadata)
        compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]] = []
        bound_symbols: set[sympy.Symbol] = set()
        child_nodes = [expression_node]

        # Compile bounds through the normal expression compiler. The bound
        # variable is added to the local environment only while evaluating terms.
        for limit_index, limit in enumerate(node.limits, start=1):
            if len(limit) != 3:
                raise self._unsupported(node)
            var = limit[0]
            if not isinstance(var, (sympy.Symbol, sympy.Idx)):
                raise self._unsupported(node)
            limit_ctx = ctx.with_descend(limit_index)
            lower_node = self._compile_node(limit_ctx.with_descend(1), metadata)
            upper_node = self._compile_node(limit_ctx.with_descend(2), metadata)
            compiled_limits.append((var, lower_node, upper_node))
            bound_symbols.add(var)
            child_nodes.extend([lower_node, upper_node])

        operator = "sum" if isinstance(node, sympy.Sum) else "product"
        indexed_value_symbols = _indexed_value_symbols(node)

        def eval_reduction(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            if len(compiled_limits) == 1:
                var, lower_node, upper_node = compiled_limits[0]
                lower = _integer_bound(lower_node.evaluator(env, exec_meta))
                upper = _integer_bound(upper_node.evaluator(env, exec_meta))
                return _evaluate_single_axis_reduction(
                    expression_node,
                    operator,
                    var,
                    lower,
                    upper,
                    env,
                    exec_meta,
                    indexed_value_symbols,
                )

            def evaluate_limit(
                limit_index: int,
                local_env: dict[sympy.Symbol, Any],
            ) -> Any:
                if limit_index >= len(compiled_limits):
                    return expression_node.evaluator(local_env, exec_meta)

                var, lower_node, upper_node = compiled_limits[limit_index]
                lower = _integer_bound(lower_node.evaluator(local_env, exec_meta))
                upper = _integer_bound(upper_node.evaluator(local_env, exec_meta))

                total = 0 if operator == "sum" else 1
                for position in range(lower, upper + 1):
                    next_env = dict(local_env)
                    next_env[var] = position
                    value = evaluate_limit(limit_index + 1, next_env)
                    if operator == "sum":
                        total = np.asarray(total) + np.asarray(value)
                    else:
                        total = np.asarray(total) * np.asarray(value)
                return total

            return evaluate_limit(0, dict(env))

        compiled = _merge_node(eval_reduction, child_nodes)
        return _CompiledNode(
            compiled.evaluator,
            used_symbols=frozenset(sym for sym in compiled.used_symbols if sym not in bound_symbols),
            has_runtime_integral=compiled.has_runtime_integral,
        )

    def _compile_runtime_integral(self, ctx: CompilationContext, metadata: CompilationMetadata) -> _CompiledNode:
        """Compile an expression's integration nodes into structural runtime solvers."""
        node: sympy.Integral = ctx.current_node  # type: ignore[assignment]
        if _scipy_integrate is None:
            raise NumUnsupportedExpressionError(
                "I ran into a part of your expression I don't know how to convert: Integral. "
                "You didn't do anything wrong! Runtime integration requires SciPy to be installed."
            )

        # Parse and compile the integrand along with its corresponding bounds.
        integrand_node = self._compile_node(ctx.with_descend(0), metadata)
        integrand_function = _compile_integral_integrand(node, integrand_node)
        compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]] = []
        for limit_index, limit in enumerate(node.limits, start=1):
            if len(limit) != 3:
                raise self._unsupported(node)
            var, lower_expr, upper_expr = limit
            if not isinstance(var, sympy.Symbol):
                raise self._unsupported(node)
            limit_ctx = ctx.with_descend(limit_index)
            lower_node = self._compile_node(limit_ctx.with_descend(1), metadata)
            upper_node = self._compile_node(limit_ctx.with_descend(2), metadata)
            compiled_limits.append((var, lower_node, upper_node))

        # Aggregate the active variables and symbols required for parameter verification.
        runtime_free_symbols = frozenset(sym for sym in node.free_symbols)
        top_symbols = frozenset(spec.symbol for spec in ctx.arg_specs)
        used_symbols = frozenset(sym for sym in runtime_free_symbols if sym in top_symbols)
        child_used = integrand_node.used_symbols
        for _, lower_node, upper_node in compiled_limits:
            child_used |= lower_node.used_symbols | upper_node.used_symbols
        used_symbols |= frozenset(sym for sym in child_used if sym in top_symbols)

        hints = dict(ctx.hints)
        symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}
        scalar_cache: dict[tuple[tuple[str, Any], ...], float] = {}

        # Define the dynamic runtime evaluator block that executes multidimensional integration loops.
        def eval_integral(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
            shape_symbols = [sym for sym in runtime_free_symbols if sym in env and _value_has_shape(env[sym])]
            try:
                if shape_symbols:
                    arrays = np.broadcast_arrays(*[np.asarray(env[sym]) for sym in shape_symbols])
                    broadcast_shape = arrays[0].shape
                else:
                    arrays = []
                    broadcast_shape = ()
            except ValueError as exc:
                names = ", ".join(symbol_to_name.get(sym, str(sym)) for sym in shape_symbols)
                raise NumArgumentError(
                    f"Arguments for integral could not be broadcast together ({names}). "
                    "Please use NumPy-compatible shapes."
                ) from exc

            # Execute one numerical integration when no public argument needs broadcasting.
            if broadcast_shape == ():
                point_env = dict(env)
                for sym in shape_symbols:
                    point_env[sym] = _to_python_scalar(np.asarray(env[sym]))
                cache_key = _runtime_integral_cache_key(runtime_free_symbols, point_env)
                if cache_key in scalar_cache:
                    return scalar_cache[cache_key]
                value = _integrate_at_parameter_point(
                    compiled_limits,
                    integrand_function,
                    point_env,
                    exec_meta,
                    hints,
                    symbol_to_name,
                    used_symbols,
                )
                scalar_cache[cache_key] = value
                return value

            # Loop through individual coordinate slices to construct a broadcasted output grid.
            result = np.empty(broadcast_shape, dtype=float)
            broadcasted_values = dict(zip(shape_symbols, arrays))
            for index in np.ndindex(broadcast_shape):
                point_env = dict(env)
                for sym, array in broadcasted_values.items():
                    point_env[sym] = _to_python_scalar(array[index])
                cache_key = _runtime_integral_cache_key(runtime_free_symbols, point_env)
                if cache_key not in scalar_cache:
                    scalar_cache[cache_key] = _integrate_at_parameter_point(
                        compiled_limits,
                        integrand_function,
                        point_env,
                        exec_meta,
                        hints,
                        symbol_to_name,
                        used_symbols,
                    )
                result[index] = scalar_cache[cache_key]
            return result

        return _CompiledNode(
            eval_integral,
            used_symbols=used_symbols,
            has_runtime_integral=True,
        )

    @staticmethod
    def _unsupported(node: sympy.Basic) -> NumUnsupportedExpressionError:
        """Construct user-facing diagnostics when an unconvertible node type is hit."""
        return NumUnsupportedExpressionError(
            "I ran into a part of your expression I don't know how to convert: "
            f"{type(node).__name__}. You didn't do anything wrong! "
            "Try simplifying the expression, replacing that construct with standard SymPy functions, "
            "or adding a local _mt_compile implementation for it."
        )


@dataclass(frozen=True)
class PinnedExpression:
    """Coordinate pointing to a specific location within a SymPy expression tree.

    Parameters
    ----------
    root_expression : sympy.Basic
        The foundational base expression tree.
    coordinate_path : tuple[int, ...], default=()
        The zero-indexed child path sequence leading to the target subexpression.

    Attributes
    ----------
    root_expression : sympy.Basic
        The foundational base expression tree.
    coordinate_path : tuple[int, ...]
        The tracking path sequence leading to the targeted node.
    node : sympy.Basic
        The precise subexpression element located at the coordinate path.
    """

    root_expression: sympy.Basic
    coordinate_path: tuple[int, ...] = ()

    def __init__(self, root_expression: sympy.Basic, coordinate_path: tuple[int, ...] = ()) -> None:
        path = tuple(coordinate_path)
        cursor = root_expression
        for index in path:
            if not isinstance(index, int):
                raise TypeError("PinnedExpression coordinate indices must be integers.")
            if index < 0 or index >= len(cursor.args):
                raise IndexError("Coordinate path points outside the expression tree.")
            cursor = cursor.args[index]
            
        object.__setattr__(self, "root_expression", root_expression)
        object.__setattr__(self, "coordinate_path", path)
        object.__setattr__(self, "_node", cursor)

    @property
    def node(self) -> sympy.Basic:
        return self._node

    def is_root(self) -> bool:
        """Determine if the coordinate path references the root expression node."""
        return len(self.coordinate_path) == 0

    def get_parent_node(self) -> sympy.Basic | None:
        """Retrieve the immediate structural parent of the current targeted node."""
        if self.is_root():
            return None
        cursor = self.root_expression
        for index in self.coordinate_path[:-1]:
            cursor = cursor.args[index]
        return cursor

    def descend(self, child_index: int) -> "PinnedExpression":
        """Navigate downwards into a specific child index of the current node."""
        if not isinstance(child_index, int):
            raise TypeError("Child index must be an integer.")
        if child_index < 0 or child_index >= len(self.node.args):
            raise IndexError("Target child index out of bounds for current expression node layout.")
            
        return PinnedExpression(
            root_expression=self.root_expression,
            coordinate_path=self.coordinate_path + (child_index,),
        )


@dataclass(frozen=True)
class CompilationContext:
    """Encapsulate the state traversal parameters passed during expression walks.

    Parameters
    ----------
    compiler : NumCompiler
        The compiler instance coordinating the current code generation task.
    pinned_expression : PinnedExpression | None, default=None
        The localized positioning coordinate inside the expression tree.
    arg_specs : tuple[NumArgSpec, ...], default=()
        The complete signature configuration rules for the compilation.
    hints : Mapping[str, Any] | None, default=None
        Advisory optimization flags and evaluation context modifiers.
    root_expression : sympy.Basic | None, default=None
        Fallback initializer when no explicit pinned expression is supplied.

    Attributes
    ----------
    compiler : NumCompiler
        The active compiler coordinator instance.
    pinned_expression : PinnedExpression
        The localized positioning coordinate within the expression tree.
    arg_specs : tuple[NumArgSpec, ...]
        The global keyword signature configuration rules.
    hints : Mapping[str, Any]
        Advisory optimization flags and execution settings.
    mt_compile_log : Any | None
        Compiler-owned logging sink for compile-time and runtime diagnostics.
    current_node : sympy.Basic
        The specific expression node being evaluated at this step.
    root_expression : sympy.Basic
        The root expression tree bound to this compilation task.
    """

    compiler: "NumCompiler"
    pinned_expression: PinnedExpression
    arg_specs: tuple[NumArgSpec, ...] = ()
    hints: Mapping[str, Any] = field(default_factory=dict)
    lowering_state: "_LoweringState | None" = None
    mt_compile_log: "_CompileLogSink | None" = None

    def __init__(
        self,
        compiler: "NumCompiler",
        pinned_expression: PinnedExpression | None = None,
        arg_specs: tuple[NumArgSpec, ...] = (),
        hints: Mapping[str, Any] | None = None,
        lowering_state: "_LoweringState | None" = None,
        *,
        root_expression: sympy.Basic | None = None,
    ) -> None:
        if pinned_expression is None:
            if root_expression is None:
                raise TypeError("CompilationContext requires pinned_expression or root_expression.")
            pinned_expression = PinnedExpression(root_expression=root_expression, coordinate_path=())
            
        object.__setattr__(self, "compiler", compiler)
        object.__setattr__(self, "pinned_expression", pinned_expression)
        object.__setattr__(self, "arg_specs", tuple(arg_specs))
        object.__setattr__(self, "hints", dict(hints or {}))
        object.__setattr__(self, "lowering_state", lowering_state)
        object.__setattr__(
            self,
            "mt_compile_log",
            lowering_state.log_sink if lowering_state is not None else None,
        )

    @property
    def current_node(self) -> sympy.Basic:
        return self.pinned_expression.node

    @property
    def root_expression(self) -> sympy.Basic:
        return self.pinned_expression.root_expression

    def with_descend(self, child_index: int) -> "CompilationContext":
        """Create a child compilation context descending into the specified index."""
        return CompilationContext(
            compiler=self.compiler,
            pinned_expression=self.pinned_expression.descend(child_index),
            arg_specs=self.arg_specs,
            hints=self.hints,
            lowering_state=self.lowering_state,
        )


# ---------------------------------------------------------------------------
# Private Internal Compiler Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredCallable:
    """Represent an explicit generated-call boundary backed by Python code.

    Parameters
    ----------
    callable : Callable[..., Any]
        Vectorized runtime implementation for the generated placeholder.
    args : tuple[sympy.Basic, ...], default=()
        Symbolic generated-call signature owned by the lowering provider.
    """

    callable: Callable[..., Any]
    args: tuple[sympy.Basic, ...] = ()

    def __init__(
        self,
        callable: Callable[..., Any],
        args: tuple[sympy.Basic, ...] = (),
    ) -> None:
        if not hasattr(callable, "__call__"):
            raise TypeError("LoweredCallable.callable must be callable.")
        normalized_args = tuple(args or ())
        for arg in normalized_args:
            if not isinstance(arg, sympy.Basic):
                raise TypeError("LoweredCallable.args must contain SymPy expressions.")
        object.__setattr__(self, "callable", callable)
        object.__setattr__(self, "args", normalized_args)


@dataclass(frozen=True)
class _LoweredExpression:
    """Carry a lowered SymPy expression and runtime-operator marker."""

    expression: sympy.Basic
    has_runtime_integral: bool = False


@dataclass(frozen=True)
class _BoundaryLowering:
    """Carry a lowered callable with metadata propagated from compatibility wrappers."""

    lowered: LoweredCallable
    has_runtime_integral: bool = False


class _CompileLogSink:
    """Scope runtime metadata records to one compiled function."""

    def __init__(self) -> None:
        self.compile_logs: list[dict[str, Any]] = []
        self._active_metadata: contextvars.ContextVar[LastExecutionMetadata | None] = (
            contextvars.ContextVar(f"mt_compile_log_{id(self)}", default=None)
        )

    @contextlib.contextmanager
    def activate(self, metadata: LastExecutionMetadata) -> Iterator[None]:
        """Set the active runtime metadata while generated code executes."""
        token = self._active_metadata.set(metadata)
        try:
            yield
        finally:
            self._active_metadata.reset(token)

    def current_metadata(self) -> LastExecutionMetadata | None:
        """Return the metadata object for the current generated evaluation."""
        return self._active_metadata.get()

    def record_compile_log(self, category: str, message: str, **details: Any) -> None:
        """Store a compile-time diagnostic record."""
        entry = {"category": category, "message": message}
        entry.update(details)
        self.compile_logs.append(entry)


class _LoweringState:
    """Allocate generated names and share the runtime log sink."""

    def __init__(self, log_sink: _CompileLogSink) -> None:
        self.log_sink = log_sink
        self._next_generated_id = 0

    def next_name(self, category: str, extension_name: str | None = None) -> str:
        """Return one collision-free generated placeholder function name."""
        self._next_generated_id += 1
        suffix = f"{self._next_generated_id:03d}"
        if category == "compile":
            return f"_mt_num_compile_{suffix}"
        if category == "implementedfunction":
            return f"_mt_num_implementedfunction_{suffix}"
        if category == "extension":
            fragment = _valid_extension_fragment(extension_name or "Extension")
            return f"_mt_num_{fragment}_{suffix}"
        raise ValueError(f"Unknown generated callable category: {category!r}.")


@dataclass(frozen=True)
class _CompiledNode:
    """Internal package container holding structural sub-expression evaluators."""

    evaluator: Callable[[dict[sympy.Symbol, Any], LastExecutionMetadata], Any]
    used_symbols: frozenset[sympy.Symbol] = frozenset()
    has_runtime_integral: bool = False
    direct_callable: Callable[..., Any] | None = None
    direct_symbols: tuple[sympy.Symbol, ...] = ()


class _FrozenExtensions(Mapping):
    """Read-only dictionary mapping wrapper ensuring deep defensive copies."""

    def __init__(self, data: Mapping[Any, Any]):
        self._data = _safe_deepcopy(dict(data))

    def __getitem__(self, key: Any) -> Any:
        return _safe_deepcopy(self._data[key])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def clear(self) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def pop(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def popitem(self) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def setdefault(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("NumCompiler extensions are immutable after compiler creation.")


# ---------------------------------------------------------------------------
# Indexed-expression execution support
# ---------------------------------------------------------------------------


def _default_num_extensions() -> dict[Any, Any]:
    """Return the compiler's built-in extension registry entries."""
    from math_toolkit.sympy_extensions.function_indexing import (
        AppliedIndexedUndef,
        IndexedFunctionBase,
        IndexedFunctionHead,
    )
    from math_toolkit.sympy_extensions.lp_norms import (
        EssentialSupremum,
        FiniteLpNorm,
        LinftyNorm,
    )

    return {
        FiniteLpNorm: {"name": "LpNorm", "compile": _compile_finite_lp_norm},
        LinftyNorm: {"name": "LinftyNorm", "compile": _compile_infinity_norm},
        EssentialSupremum: {
            "name": "EssentialSupremum",
            "compile": _compile_essential_supremum,
        },
        IndexedFunctionBase: {"handler": _reject_indexed_function_extension},
        IndexedFunctionHead: {"handler": _reject_indexed_function_extension},
        AppliedIndexedUndef: {"handler": _reject_indexed_function_extension},
    }


def _lookup_extension_entry(extensions: Mapping[Any, Any], value: Any) -> Any | None:
    """Return the first registered extension entry matching one runtime value."""
    for registered_type, candidate in extensions.items():
        try:
            if isinstance(value, registered_type):
                return candidate
        except TypeError:
            if type(value) is registered_type:
                return candidate
    return None


def _reject_indexed_function_extension(value: Any) -> Any:
    """Reject indexed-function forms through extension infrastructure."""
    if isinstance(value, CompilationContext):
        value = value.current_node
    raise NotImplementedError(
        f"{type(value).__name__} compilation is not implemented."
    )


def _compile_finite_lp_norm(ctx: CompilationContext) -> NumFunction:
    """Compile a finite ``Lp`` norm through its explicit integral rewrite."""
    node = ctx.current_node
    try:
        integral_expression = node.rewrite(sympy.Integral)
    except NotImplementedError as exc:
        raise NumUnsupportedExpressionError(
            "I ran into a part of your expression I don't know how to convert: LpNorm. "
            "Only interval and rectangular product domains are supported numerically."
        ) from exc

    return ctx.compiler.compile(
        integral_expression,
        arg_specs=ctx.arg_specs,
        hints=ctx.hints,
    )


def _compile_infinity_norm(ctx: CompilationContext) -> LoweredCallable:
    """Compile an ``L-infinity`` norm as a numerical supremum calculation."""
    from math_toolkit.sympy_extensions.lp_norms import EssentialSupremum

    node = ctx.current_node
    supremum = EssentialSupremum(
        sympy.Abs(node.expr),
        (node.variables, node.domain),
        weight=node.weight,
    )
    return _compile_essential_supremum(ctx, supremum=supremum)


def _compile_essential_supremum(
    ctx: CompilationContext,
    *,
    supremum: Any | None = None,
) -> LoweredCallable:
    """Compile a formal essential supremum over a bounded rectangular domain."""
    node = supremum if supremum is not None else ctx.current_node
    if getattr(node, "weight", sympy.S.One) != sympy.S.One:
        raise NumUnsupportedExpressionError(
            "I ran into a part of your expression I don't know how to convert: LinftyNorm. "
            "Weighted infinity norms are not supported numerically yet."
        )

    domain_bounds = _infinity_norm_domain_bounds(node.variables, node.domain)
    objective_arg_specs = _infinity_norm_objective_arg_specs(node.variables, ctx.arg_specs)
    objective = ctx.compiler.compile(
        node.expr,
        arg_specs=objective_arg_specs,
        hints=ctx.hints,
    )
    bound_functions = tuple(
        (
            ctx.compiler.compile(
                lower,
                arg_specs=_arg_specs_for_runtime_symbols(
                    _runtime_free_symbols(sympy.sympify(lower)),
                    ctx.arg_specs,
                ),
                hints=ctx.hints,
            ),
            ctx.compiler.compile(
                upper,
                arg_specs=_arg_specs_for_runtime_symbols(
                    _runtime_free_symbols(sympy.sympify(upper)),
                    ctx.arg_specs,
                ),
                hints=ctx.hints,
            ),
        )
        for lower, upper in domain_bounds
    )

    external_symbols = frozenset(_runtime_free_symbols(node))
    args = _ordered_symbols(external_symbols, ctx.arg_specs)
    symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}
    bound_names = [spec.name for spec in objective_arg_specs[: len(node.variables)]]

    def evaluate_supremum(*values: Any) -> Any:
        if len(values) != len(args):
            raise NumArgumentError("Infinity norm received the wrong number of arguments.")

        env = dict(zip(args, values))
        try:
            shape_symbols, broadcast_shape, broadcasted_values = (
                _runtime_parameter_broadcast_values(args, env, node)
            )
        except ValueError as exc:
            names = ", ".join(
                symbol_to_name.get(sym, str(sym)) for sym in args if sym in env
            )
            raise NumArgumentError(
                f"Arguments for infinity norm could not be broadcast together ({names}). "
                "Please use NumPy-compatible shapes."
            ) from exc

        if not shape_symbols:
            return _evaluate_infinity_norm_scalar(
                objective,
                bound_functions,
                bound_names,
                env,
                symbol_to_name,
                ctx.hints,
            )

        result = np.empty(broadcast_shape, dtype=float)
        indexed_ranks = _indexed_base_ranks(node)
        for index in np.ndindex(result.shape):
            point_env = dict(env)
            for symbol, array in broadcasted_values.items():
                if symbol in indexed_ranks:
                    point_env[symbol] = _indexed_parameter_axis_slice(
                        array,
                        indexed_ranks[symbol],
                        index,
                    )
                else:
                    point_env[symbol] = _to_python_scalar(array[index])
            result[index] = _evaluate_infinity_norm_scalar(
                objective,
                bound_functions,
                bound_names,
                point_env,
                symbol_to_name,
                ctx.hints,
            )
        return result

    return LoweredCallable(callable=evaluate_supremum, args=args)


def _infinity_norm_domain_bounds(
    variables: tuple[sympy.Symbol, ...],
    domain: sympy.Basic,
) -> tuple[tuple[sympy.Basic, sympy.Basic], ...]:
    """Return rectangular finite bounds accepted by numerical infinity norms."""
    if len(variables) == 1 and isinstance(domain, sympy.Interval):
        _reject_unbounded_infinity_norm_interval(domain)
        return ((domain.start, domain.end),)

    if isinstance(domain, sympy.ProductSet):
        factors = tuple(domain.sets)
        if len(factors) != len(variables):
            raise NumUnsupportedExpressionError(
                "Infinity norm domain dimension does not match its variables."
            )
        bounds: list[tuple[sympy.Basic, sympy.Basic]] = []
        for factor in factors:
            if not isinstance(factor, sympy.Interval):
                raise NumUnsupportedExpressionError(
                    "Numerical infinity norms require interval or rectangular product domains."
                )
            _reject_unbounded_infinity_norm_interval(factor)
            bounds.append((factor.start, factor.end))
        return tuple(bounds)

    raise NumUnsupportedExpressionError(
        "Numerical infinity norms require interval or rectangular product domains."
    )


def _reject_unbounded_infinity_norm_interval(interval: sympy.Interval) -> None:
    """Reject intervals whose endpoints cannot define a finite optimizer box."""
    if interval.start.is_infinite or interval.end.is_infinite:
        raise NumUnsupportedExpressionError(
            "Numerical infinity norms require a bounded interval or rectangular product domain."
        )


def _infinity_norm_objective_arg_specs(
    variables: tuple[sympy.Symbol, ...],
    arg_specs: tuple[NumArgSpec, ...],
) -> tuple[NumArgSpec, ...]:
    """Return nested objective arguments with bound variables first."""
    used_names = {spec.name for spec in arg_specs}
    bound_specs: list[NumArgSpec] = []
    for variable in variables:
        base_name = str(variable)
        name = base_name
        suffix = 0
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        used_names.add(name)
        bound_specs.append(NumArgSpec(variable, name))
    return tuple(bound_specs) + tuple(arg_specs)


def _evaluate_infinity_norm_scalar(
    objective: NumFunction,
    bound_functions: tuple[tuple[NumFunction, NumFunction], ...],
    bound_names: list[str],
    env: Mapping[sympy.Symbol, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    hints: Mapping[str, Any],
) -> float:
    """Evaluate one scalar-parameter numerical infinity norm by uniform sampling."""
    external_kwargs = {
        symbol_to_name[symbol]: value
        for symbol, value in env.items()
        if symbol in symbol_to_name
    }
    bounds = tuple(
        (
            _finite_float(lower(**_num_function_kwargs(lower, external_kwargs))),
            _finite_float(upper(**_num_function_kwargs(upper, external_kwargs))),
        )
        for lower, upper in bound_functions
    )
    for lower, upper in bounds:
        if upper < lower:
            raise NumEvaluationError("Infinity norm domain bounds must be ordered.")

    return _evaluate_infinity_norm_grid(
        objective,
        bound_names,
        external_kwargs,
        bounds,
        _infinity_norm_sample_count(hints),
    )


def _evaluate_infinity_norm_grid(
    objective: NumFunction,
    bound_names: list[str],
    external_kwargs: Mapping[str, Any],
    bounds: tuple[tuple[float, float], ...],
    sample_count: int,
) -> float:
    """Return the maximum over a uniform grid of the bounded domain."""

    if len(bounds) == 1:
        lower, upper = bounds[0]
        samples = np.linspace(lower, upper, sample_count, dtype=float)
        kwargs = dict(external_kwargs)
        kwargs[bound_names[0]] = samples
        values = np.asarray(objective(**kwargs), dtype=float)
        if values.shape == ():
            return _finite_float(values)
        if not np.all(np.isfinite(values)):
            raise NumEvaluationError("The numerical integrand produced a non-finite value.")
        return float(np.max(values))

    axis_count = _infinity_norm_axis_sample_count(sample_count, len(bounds))
    axes = [
        np.linspace(lower, upper, axis_count, dtype=float)
        for lower, upper in bounds
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    kwargs = dict(external_kwargs)
    for name, grid in zip(bound_names, grids, strict=True):
        kwargs[name] = grid
    values = np.asarray(objective(**kwargs), dtype=float)
    if values.shape == ():
        return _finite_float(values)
    if not np.all(np.isfinite(values)):
        raise NumEvaluationError("The numerical integrand produced a non-finite value.")
    return float(np.max(values))


def _infinity_norm_sample_count(hints: Mapping[str, Any]) -> int:
    """Return the dense-sample count used for one-dimensional infinity norms."""

    raw_value = hints.get("LinftyNormSamples", hints.get("infinity_norm_samples", 65537))
    try:
        sample_count = int(raw_value)
    except Exception:
        sample_count = 65537
    sample_count = max(sample_count, 3)
    if sample_count % 2 == 0:
        sample_count += 1
    return sample_count


def _infinity_norm_axis_sample_count(sample_count: int, dimensions: int) -> int:
    """Return the per-axis sample count for a multidimensional uniform grid."""

    axis_count = int(np.floor(sample_count ** (1 / max(dimensions, 1))))
    axis_count = max(axis_count, 3)
    if axis_count % 2 == 0:
        axis_count += 1
    return axis_count


def _num_function_kwargs(function: NumFunction, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return keyword arguments accepted by one compiled Num function."""

    return {
        spec.name: values[spec.name]
        for spec in function.arg_specs
        if spec.name in values
    }


def _requires_indexed_execution(
    expression: sympy.Basic,
    specs: tuple[NumArgSpec, ...],
) -> bool:
    """Return whether an expression needs the indexed fallback compiler."""
    free_index_symbols = {
        symbol
        for symbol in expression.free_symbols
        if isinstance(symbol, sympy.Idx)
    }
    if free_index_symbols:
        return True
    if any(isinstance(symbol, sympy.Idx) for symbol in expression.free_symbols):
        return True
    return any(isinstance(spec.symbol, sympy.Idx) for spec in specs)


def _runtime_free_symbols(expression: sympy.Basic) -> set[sympy.Basic]:
    """Return public runtime symbols required to evaluate one expression."""
    bound_symbols = _reduction_bound_symbols(expression)
    indexed_bases = {
        node.base
        for node in sympy.preorder_traversal(expression)
        if isinstance(node, sympy.Indexed)
    }
    indexed_base_labels = {
        _indexed_base_label(base)
        for base in indexed_bases
    }
    required: set[sympy.Basic] = set()
    for symbol in getattr(expression, "free_symbols", set()):
        if isinstance(symbol, sympy.Indexed):
            required.add(symbol.base)
            for index in symbol.indices:
                required.update(
                    item
                    for item in getattr(index, "free_symbols", set())
                    if item not in bound_symbols
                )
            continue
        if symbol in indexed_base_labels:
            continue
        if symbol not in bound_symbols:
            required.add(symbol)
    required.update(indexed_bases)
    return required


def _indexed_value_symbols(expression: sympy.Basic) -> frozenset[sympy.Basic]:
    """Return runtime symbols that are consumed as indexed arrays."""

    traversed_bases = {
        node.base
        for node in sympy.preorder_traversal(expression)
        if isinstance(node, sympy.Indexed)
    }
    free_bases = {
        symbol.base
        for symbol in getattr(expression, "free_symbols", set())
        if isinstance(symbol, sympy.Indexed)
    }
    return frozenset(traversed_bases | free_bases)


def _indexed_base_ranks(expression: sympy.Basic) -> dict[sympy.IndexedBase, int]:
    """Return the explicit symbolic rank used by every indexed base."""

    ranks: dict[sympy.IndexedBase, int] = {}
    for node in sympy.preorder_traversal(expression):
        if isinstance(node, sympy.Indexed):
            ranks.setdefault(node.base, len(node.indices))
    return ranks


def _runtime_parameter_broadcast_values(
    args: tuple[sympy.Symbol, ...],
    env: Mapping[sympy.Symbol, Any],
    expression: sympy.Basic,
) -> tuple[list[sympy.Symbol], tuple[int, ...], dict[sympy.Symbol, np.ndarray]]:
    """Return broadcast axes for runtime parameters without consuming index axes."""

    indexed_ranks = _indexed_base_ranks(expression)
    shapes: list[tuple[int, ...]] = []
    shape_symbols: list[sympy.Symbol] = []

    # Indexed bases reserve their leading axes for explicit symbolic indexing.
    # Only trailing axes can participate in automatic parameter-family
    # broadcasting.
    for symbol in args:
        if symbol not in env:
            continue
        shape = _runtime_parameter_broadcast_shape(
            symbol,
            env[symbol],
            indexed_ranks,
        )
        if not shape:
            continue
        shape_symbols.append(symbol)
        shapes.append(shape)

    if not shapes:
        return [], (), {}

    broadcast_shape = tuple(np.broadcast_shapes(*shapes))
    broadcasted_values: dict[sympy.Symbol, np.ndarray] = {}
    for symbol in shape_symbols:
        array = np.asarray(env[symbol])
        if symbol in indexed_ranks:
            rank = indexed_ranks[symbol]
            if array.ndim <= rank:
                continue
            target_shape = tuple(array.shape[:rank]) + broadcast_shape
            broadcasted_values[symbol] = np.broadcast_to(array, target_shape)
        else:
            broadcasted_values[symbol] = np.broadcast_to(array, broadcast_shape)

    return shape_symbols, broadcast_shape, broadcasted_values


def _runtime_parameter_broadcast_shape(
    symbol: sympy.Symbol,
    value: Any,
    indexed_ranks: Mapping[sympy.IndexedBase, int],
) -> tuple[int, ...]:
    """Return automatic vectorization axes for one runtime argument."""

    array = np.asarray(value)
    if array.shape == ():
        return ()
    if symbol not in indexed_ranks:
        return tuple(array.shape)

    rank = indexed_ranks[symbol]
    if array.ndim <= rank:
        return ()
    return tuple(array.shape[rank:])


def _indexed_parameter_axis_slice(
    value: Any,
    explicit_rank: int,
    parameter_index: tuple[int, ...],
) -> Any:
    """Select one parameter-family point while preserving explicit index axes."""

    array = np.asarray(value)
    if array.ndim <= explicit_rank:
        return array
    return array[(slice(None),) * explicit_rank + parameter_index]


def _reduction_bound_symbols(expression: sympy.Basic) -> set[sympy.Basic]:
    """Return local symbols bound by finite reductions in an expression."""
    bound: set[sympy.Basic] = set()
    for node in sympy.preorder_traversal(expression):
        if not isinstance(node, (sympy.Sum, sympy.Product)):
            continue
        for limit in node.limits:
            if len(limit) >= 1 and isinstance(limit[0], (sympy.Symbol, sympy.Idx)):
                bound.add(limit[0])
    return bound


def _indexed_base_label(base: sympy.IndexedBase) -> sympy.Basic:
    """Return the symbolic label SymPy exposes for one indexed base."""
    label = getattr(base, "label", None)
    return label if isinstance(label, sympy.Basic) else sympy.Symbol(str(base))


def _arg_specs_for_runtime_symbols(
    symbols: set[sympy.Basic] | frozenset[sympy.Basic],
    arg_specs: tuple[NumArgSpec, ...],
) -> tuple[NumArgSpec, ...]:
    """Return explicit argument specs required by a runtime symbol set."""

    selected: list[NumArgSpec] = []
    for spec in arg_specs:
        if spec.symbol in symbols:
            selected.append(spec)
            continue
        if isinstance(spec.symbol, sympy.IndexedBase):
            label = _indexed_base_label(spec.symbol)
            if label in symbols:
                selected.append(spec)
    return tuple(selected)


def _compile_indexed_expression(
    expression: sympy.Basic,
    specs: tuple[NumArgSpec, ...],
    hints: Mapping[str, Any],
) -> Callable[..., Any]:
    """Compile indexed expressions through a direct symbolic evaluation plan."""
    _validate_indexed_base_shapes(expression)
    plan = _IndexedExecutionPlan(expression=expression, arg_specs=specs, hints=dict(hints))
    return plan.build_callable()


def _validate_indexed_base_shapes(expression: sympy.Basic) -> None:
    """Reject inconsistent indexed ranks for the same symbolic array base."""
    shapes_by_base: dict[sympy.IndexedBase, tuple[int, sympy.Indexed]] = {}
    for node in sympy.preorder_traversal(expression):
        if not isinstance(node, sympy.Indexed):
            continue

        rank = len(node.indices)
        previous = shapes_by_base.setdefault(node.base, (rank, node))
        previous_rank, previous_node = previous
        if previous_rank != rank:
            raise NumArgumentError(
                f"Indexed base '{node.base}' is used with incompatible ranks: "
                f"saw rank {previous_rank} as {previous_node} and rank {rank} as {node}. "
                "Use one indexed shape per IndexedBase in a single Num expression."
            )


def _vectorized_indexed_lookup(array: Any, *indices: Any) -> Any:
    """Return zero-padded NumPy indexing for scalar or broadcasted indices."""
    values = np.asarray(array)
    if not indices:
        return values

    # Broadcast all index expressions together so a reduction axis and ordinary
    # sample axes combine through NumPy rather than through Python loops.
    try:
        index_arrays = np.broadcast_arrays(
            *(np.asarray(index, dtype=int) for index in indices)
        )
    except ValueError as exc:
        raise NumArgumentError(
            "Indexed argument indices could not be broadcast together. "
            "Please use NumPy-compatible shapes."
        ) from exc

    # Missing or empty dimensions mean every requested entry is out of range.
    # Match the indexed fallback's zero-padding contract for sequence math.
    result_shape = index_arrays[0].shape if index_arrays else ()
    if values.ndim < len(index_arrays):
        return np.zeros(result_shape, dtype=float)
    if any(values.shape[axis] <= 0 for axis in range(len(index_arrays))):
        return np.zeros(result_shape, dtype=float)

    valid = np.ones(result_shape, dtype=bool)
    clipped_indices = []
    for axis, index_array in enumerate(index_arrays):
        valid = np.logical_and(valid, index_array >= 0)
        valid = np.logical_and(valid, index_array < values.shape[axis])
        clipped_indices.append(np.clip(index_array, 0, values.shape[axis] - 1))

    gathered = values[tuple(clipped_indices)]
    if values.ndim > len(index_arrays):
        trailing_shape = tuple(values.shape[len(index_arrays):])
        valid = valid.reshape(valid.shape + (1,) * len(trailing_shape))
    result = np.where(valid, gathered, 0)

    # A reduction or integration callback may insert singleton placeholders for
    # parameter axes before the lookup. Once the indexed value contributes its
    # own trailing parameter axes, those placeholders should collapse.
    if values.ndim > len(index_arrays):
        trailing_shape = tuple(values.shape[len(index_arrays):])
        if (
            trailing_shape
            and len(result_shape) >= len(trailing_shape)
            and all(axis == 1 for axis in result_shape[-len(trailing_shape):])
        ):
            target_shape = result_shape[:-len(trailing_shape)] + trailing_shape
            result = np.reshape(result, target_shape)
    return result


@dataclass(frozen=True)
class _IndexedExecutionPlan:
    """Carry the public binding and runtime evaluation rules for indexed expressions."""

    expression: sympy.Basic
    arg_specs: tuple[NumArgSpec, ...]
    hints: Mapping[str, Any]

    def build_callable(self) -> Callable[..., Any]:
        """Build a small dynamic callable whose runtime delegates into this plan."""
        public_specs = self.public_arg_specs
        arg_names = [spec.name for spec in public_specs]
        signature_parts = list(arg_names)
        
        if signature_parts:
            signature_parts.append("*")
        else:
            signature_parts.append("*")
            
        signature_parts.append("numfun_array_length=None")
        signature = ", ".join(signature_parts)
        payload = ", ".join(arg_names)
        helper_call = (
            f"_helper({payload}, numfun_array_length=numfun_array_length)"
            if payload
            else "_helper(numfun_array_length=numfun_array_length)"
        )
        source = (
            f"def compiled({signature}):\n"
            f"    return {helper_call}\n"
        )
        namespace = {"_helper": self._invoke}
        exec(source, namespace)
        
        compiled = namespace["compiled"]
        compiled.__name__ = "compiled_indexed_expression"
        compiled.arg_specs = self.arg_specs
        compiled.indexed_execution = True
        return compiled

    @property
    def free_index_symbols(self) -> tuple[sympy.Idx, ...]:
        """Return free index variables that determine vector output length."""
        symbols = [
            symbol
            for symbol in sorted(self.expression.free_symbols, key=str)
            if isinstance(symbol, sympy.Idx)
        ]
        return tuple(symbols)

    @property
    def public_arg_specs(self) -> tuple[NumArgSpec, ...]:
        """Return public call arguments, excluding free index placeholders."""
        index_symbols = set(self.free_index_symbols)
        return tuple(
            spec for spec in self.arg_specs if spec.symbol not in index_symbols
        )

    def _invoke(self, *args: Any, numfun_array_length: Mapping[str, Any] | None = None) -> Any:
        """Evaluate the indexed expression for one public call."""
        array_length = dict(numfun_array_length or {})
        env = {
            spec.symbol: value
            for spec, value in zip(self.public_arg_specs, args)
        }

        # Indexed expressions support one free vector index in the current public contract.
        free_indices = self.free_index_symbols
        if len(free_indices) > 1:
            raise NumArgumentError(
                "Indexed compilation currently supports at most one free index."
            )

        if free_indices:
            free_index = free_indices[0]
            length = _resolve_output_length(free_index, env, array_length)
            values = [
                self._evaluate_scalar(
                    env,
                    index_env={free_index: position},
                    preserve_symbolic=False,
                )
                for position in range(length)
            ]
            return _coerce_indexed_vector_result(values)

        shaped_names = [
            spec.name
            for spec in self.public_arg_specs
            if _value_has_shape(env.get(spec.symbol))
        ]
        if not shaped_names:
            return self._evaluate_scalar(env, index_env={}, preserve_symbolic=False)

        preserve_symbolic = any(
            _needs_symbolic_list_fallback(env[spec.symbol])
            for spec in self.public_arg_specs
            if spec.symbol in env
        )
        return _evaluate_shaped_scalar_expression(
            self,
            env,
            preserve_symbolic=preserve_symbolic,
        )

    def _evaluate_scalar(
        self,
        env: Mapping[sympy.Basic, Any],
        *,
        index_env: Mapping[sympy.Basic, Any],
        preserve_symbolic: bool,
    ) -> Any:
        """Evaluate the stored expression for one scalar environment."""
        merged_env = dict(env)
        merged_env.update(index_env)
        return _evaluate_indexed_node(
            self.expression,
            merged_env,
            preserve_symbolic=preserve_symbolic,
        )


def _resolve_output_length(
    free_index: sympy.Idx,
    env: Mapping[sympy.Basic, Any],
    numfun_array_length: Mapping[str, Any],
) -> int:
    """Resolve the public output length for a free index."""
    explicit = numfun_array_length.get(str(free_index))
    if explicit is not None:
        return max(int(explicit), 0)

    lengths = [
        len(value)
        for symbol, value in env.items()
        if isinstance(symbol, sympy.IndexedBase) and hasattr(value, "__len__")
    ]
    if not lengths:
        raise NumArgumentError(
            f"Missing numfun_array_length for free index {free_index}."
        )
    return max(lengths)


def _evaluate_shaped_scalar_expression(
    plan: _IndexedExecutionPlan,
    env: Mapping[sympy.Basic, Any],
    *,
    preserve_symbolic: bool,
) -> Any:
    """Broadcast shaped public arguments and evaluate the scalar expression entrywise."""
    # Keep concrete indexed bases intact while reading terms such as ``a[0]``.
    # Other shaped scalar parameters may still broadcast around those lookups.
    indexed_access_bases = {
        node.base
        for node in sympy.preorder_traversal(plan.expression)
        if isinstance(node, sympy.Indexed)
    }
    shaped_symbols = [
        spec.symbol
        for spec in plan.public_arg_specs
        if spec.symbol in env
        and spec.symbol not in indexed_access_bases
        and _value_has_shape(env[spec.symbol])
    ]
    arrays = [np.asarray(env[symbol], dtype=object if preserve_symbolic else None) for symbol in shaped_symbols]
    
    try:
        broadcasted = np.broadcast_arrays(*arrays) if arrays else []
    except ValueError as exc:
        raise NumArgumentError(
            "Shaped arguments could not be broadcast together. Please use NumPy-compatible shapes."
        ) from exc
        
    if not broadcasted:
        return plan._evaluate_scalar(env, index_env={}, preserve_symbolic=preserve_symbolic)

    result: list[Any] = []
    for index in np.ndindex(broadcasted[0].shape):
        scalar_env = dict(env)
        for symbol, array in zip(shaped_symbols, broadcasted):
            scalar_env[symbol] = array[index].item() if hasattr(array[index], "item") else array[index]
        result.append(
            plan._evaluate_scalar(
                scalar_env,
                index_env={},
                preserve_symbolic=preserve_symbolic,
            )
        )

    if preserve_symbolic:
        return result
    return np.asarray(result, dtype=float).reshape(broadcasted[0].shape)


def _evaluate_indexed_node(
    node: Any,
    env: Mapping[sympy.Basic, Any],
    *,
    preserve_symbolic: bool,
) -> Any:
    """Evaluate one symbolic node under indexed-runtime rules."""
    if isinstance(node, sympy.Indexed):
        sequence = env.get(node.base)
        if sequence is None:
            raise NumArgumentError(f"Missing value for indexed base '{node.base}'.")
        index_value = _evaluate_indexed_node(node.indices[0], env, preserve_symbolic=False)
        position = int(sympy.Integer(index_value))
        if position < 0 or position >= len(sequence):
            return 0
        return sequence[position]

    if isinstance(node, sympy.Symbol):
        if node not in env:
            raise NumArgumentError(f"Missing value for symbol '{node}'.")
        return env[node]

    if isinstance(node, sympy.Idx):
        if node not in env:
            raise NumArgumentError(f"Missing value for index '{node}'.")
        return env[node]

    if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational, sympy.NumberSymbol)):
        return node if preserve_symbolic else _constant_expression_to_float(node)

    if isinstance(node, sympy.Add):
        total = _evaluate_indexed_node(node.args[0], env, preserve_symbolic=preserve_symbolic)
        for child in node.args[1:]:
            total = total + _evaluate_indexed_node(child, env, preserve_symbolic=preserve_symbolic)
        return total

    if isinstance(node, sympy.Mul):
        product = _evaluate_indexed_node(node.args[0], env, preserve_symbolic=preserve_symbolic)
        for child in node.args[1:]:
            product = product * _evaluate_indexed_node(child, env, preserve_symbolic=preserve_symbolic)
        return product

    if isinstance(node, sympy.Pow):
        base = _evaluate_indexed_node(node.args[0], env, preserve_symbolic=preserve_symbolic)
        exponent = _evaluate_indexed_node(node.args[1], env, preserve_symbolic=preserve_symbolic)
        return base ** exponent

    if isinstance(node, sympy.Sum):
        return _evaluate_symbolic_reduction(
            node,
            env,
            preserve_symbolic=preserve_symbolic,
            operator="sum",
        )

    if isinstance(node, sympy.Product):
        return _evaluate_symbolic_reduction(
            node,
            env,
            preserve_symbolic=preserve_symbolic,
            operator="product",
        )

    if isinstance(node, sympy.Integral):
        substituted = _substitute_runtime_values(node, env)
        evaluated = substituted.doit()
        return evaluated if preserve_symbolic else _constant_expression_to_float(evaluated)

    if isinstance(node, sympy.Function):
        values = [
            _evaluate_indexed_node(child, env, preserve_symbolic=preserve_symbolic)
            for child in node.args
        ]
        evaluated = node.func(*values)
        if preserve_symbolic:
            return evaluated
        return _constant_expression_to_float(evaluated)

    if isinstance(node, sympy.Basic):
        substituted = _substitute_runtime_values(node, env)
        if preserve_symbolic:
            return substituted
        if isinstance(substituted, sympy.Basic) and substituted.free_symbols:
            raise NumEvaluationError("Indexed expression left unresolved symbols after substitution.")
        return _constant_expression_to_float(substituted)

    return node


def _evaluate_symbolic_reduction(
    node: sympy.Basic,
    env: Mapping[sympy.Basic, Any],
    *,
    preserve_symbolic: bool,
    operator: str,
) -> Any:
    """Evaluate ``Sum`` or ``Product`` with integer runtime bounds."""
    expression = node.args[0]
    accumulator = 0 if operator == "sum" else 1
    
    for limit in node.limits:
        bound_symbol, lower, upper = limit
        lower_value = int(sympy.Integer(_evaluate_indexed_node(lower, env, preserve_symbolic=False)))
        upper_value = int(sympy.Integer(_evaluate_indexed_node(upper, env, preserve_symbolic=False)))
        values = []
        for position in range(lower_value, upper_value + 1):
            local_env = dict(env)
            local_env[bound_symbol] = position
            values.append(
                _evaluate_indexed_node(
                    expression,
                    local_env,
                    preserve_symbolic=preserve_symbolic,
                )
            )
        if operator == "sum":
            accumulator = sum(values, accumulator)
        else:
            for value in values:
                accumulator *= value
                
    return accumulator


def _substitute_runtime_values(node: sympy.Basic, env: Mapping[sympy.Basic, Any]) -> sympy.Basic:
    """Substitute runtime scalar values into a symbolic node."""
    substitutions = {
        symbol: sympy.sympify(value)
        for symbol, value in env.items()
        if isinstance(symbol, (sympy.Symbol, sympy.Idx))
    }
    return node.xreplace(substitutions)


def _coerce_indexed_vector_result(values: list[Any]) -> Any:
    """Normalize a vector result using numeric arrays when possible."""
    if all(_is_numeric_like_value(value) for value in values):
        return np.asarray([_constant_expression_to_float(value) for value in values], dtype=float)
    return values


def _is_numeric_like_value(value: Any) -> bool:
    """Return whether a scalar output should live in a numeric NumPy array."""
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, sympy.Basic):
        return bool(getattr(value, "is_number", False))
    return False


def _needs_symbolic_list_fallback(value: Any) -> bool:
    """Return whether a plain Python sequence should stay in symbolic list mode."""
    if not isinstance(value, list):
        return False
    return any(isinstance(item, sympy.Basic) for item in value)


def _is_plain_numeric_entry(value: Any) -> bool:
    """Return whether a sequence entry is safely coercible to float."""
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, sympy.Basic):
        return bool(value.is_number and value.is_real and value.is_finite)
    return False


# ---------------------------------------------------------------------------
# Internal Runtime Integration Support Code
# ---------------------------------------------------------------------------


def _compile_integral_integrand(
    node: sympy.Integral,
    integrand_node: _CompiledNode,
) -> Callable[[Mapping[sympy.Symbol, Any], LastExecutionMetadata], float]:
    """Return the numerical callable used by SciPy for an integral body."""
    for limit in node.limits:
        if len(limit) != 3:
            return _compiled_node_integrand(integrand_node)
        var = limit[0]
        if not isinstance(var, sympy.Symbol):
            return _compiled_node_integrand(integrand_node)
    if integrand_node.direct_callable is not None and not integrand_node.has_runtime_integral:
        return _direct_generated_integrand(integrand_node)
    return _compiled_node_integrand(integrand_node)


def _direct_generated_integrand(
    integrand_node: _CompiledNode,
) -> Callable[[Mapping[sympy.Symbol, Any], LastExecutionMetadata], float]:
    """Wrap a generated callable as a scalar SciPy integrand callback."""
    generated = integrand_node.direct_callable
    argument_symbols = integrand_node.direct_symbols
    if generated is None:
        return _compiled_node_integrand(integrand_node)

    def evaluate(
        local_env: Mapping[sympy.Symbol, Any],
        exec_meta: LastExecutionMetadata,
    ) -> float:
        values = []
        for symbol in argument_symbols:
            if symbol not in local_env:
                raise NumArgumentError(f"Missing value for symbol '{symbol}'.")
            values.append(local_env[symbol])
        try:
            return _finite_float(generated(*values))
        except (NumArgumentError, NumEvaluationError):
            raise
        except Exception as exc:
            raise NumEvaluationError(_friendly_runtime_message(exc)) from exc

    return evaluate


def _compiled_node_integrand(
    integrand_node: _CompiledNode,
) -> Callable[[Mapping[sympy.Symbol, Any], LastExecutionMetadata], float]:
    """Wrap a compiled expression node as a scalar SciPy integrand callback."""
    def evaluate(
        local_env: Mapping[sympy.Symbol, Any],
        exec_meta: LastExecutionMetadata,
    ) -> float:
        return _finite_float(integrand_node.evaluator(dict(local_env), exec_meta))

    return evaluate


def _integrate_at_parameter_point(
    compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]],
    integrand_function: Callable[[Mapping[sympy.Symbol, Any], LastExecutionMetadata], float],
    env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    hints: Mapping[str, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    used_symbols: frozenset[sympy.Symbol],
) -> float:
    """Execute numerical integration for one broadcasted parameter coordinate."""
    epsabs, epsrel = _integration_tolerances(hints)
    integrator = _integral_integrator(hints)

    def evaluate_integrand(local_env: dict[sympy.Symbol, Any]) -> float:
        """Evaluate the precompiled numerical integrand callable."""
        return integrand_function(local_env, exec_meta)

    def integrate_with_cubature_1d(local_env: dict[sympy.Symbol, Any]) -> tuple[float, float]:
        """Perform higher-order multidimensional cubature when single-dimensional limits apply."""
        var, lower_node, upper_node = compiled_limits[0]
        lower = _bound_to_float(lower_node.evaluator(local_env, exec_meta))
        upper = _bound_to_float(upper_node.evaluator(local_env, exec_meta))
        if np.isnan(lower) or np.isnan(upper):
            raise NumEvaluationError(
                "The numerical integral does not converge because one of its bounds evaluated to NaN. "
                + _parameter_suffix(local_env, symbol_to_name, used_symbols)
            )

        def cubature_integrand(points: np.ndarray) -> np.ndarray:
            point_array = np.asarray(points, dtype=float)
            values = np.empty(point_array.shape[0], dtype=float)
            for row_index, row in enumerate(point_array):
                local_env[var] = float(row[0])
                values[row_index] = evaluate_integrand(local_env)
            return values

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = _scipy_integrate.cubature(  # type: ignore[union-attr]
                    cubature_integrand,
                    np.array([lower], dtype=float),
                    np.array([upper], dtype=float),
                    rtol=epsrel,
                    atol=epsabs,
                    max_subdivisions=int(_integral_runtime_hint(hints, "cubature_max_subdivisions", 10000)),
                )
                value = float(np.asarray(result.estimate))
                error = float(np.asarray(result.error))
                status = getattr(result, "status", "converged")
            except NumEvaluationError as exc:
                raise NumEvaluationError(
                    "The numerical integral does not converge or diverges for the supplied parameter values. "
                    + _parameter_suffix(local_env, symbol_to_name, used_symbols)
                    + f" Details: {exc}"
                ) from exc
            except Exception as exc:
                raise NumEvaluationError(
                    "The numerical integral does not converge for the supplied parameter values. "
                    + _parameter_suffix(local_env, symbol_to_name, used_symbols)
                ) from exc

        warning_text = " ".join(str(w.message) for w in caught)
        bad_warning = any(isinstance(w.message, RuntimeWarning) for w in caught)
        if status != "converged" or bad_warning or not np.isfinite(value) or not np.isfinite(error):
            message = (
                "The numerical integral does not converge or diverges for the supplied parameter values. "
                + _parameter_suffix(local_env, symbol_to_name, used_symbols)
            )
            if warning_text:
                message += f" Integrator note: {warning_text}"
            raise NumEvaluationError(message)

        error = float(abs(error))
        exec_meta.record_integration_error(error)
        return float(value), error

    def integrate_dimension(dim_index: int, local_env: dict[sympy.Symbol, Any]) -> tuple[float, float]:
        """Coordinate individual integration steps over recursive nested dimensions."""
        if (
            dim_index == 0
            and len(compiled_limits) == 1
            and integrator in {"Auto", "Cubature"}
            and hasattr(_scipy_integrate, "cubature")
        ):
            return integrate_with_cubature_1d(local_env)

        if dim_index >= len(compiled_limits):
            return evaluate_integrand(local_env), 0.0

        var, lower_node, upper_node = compiled_limits[dim_index]
        lower = _bound_to_float(lower_node.evaluator(local_env, exec_meta))
        upper = _bound_to_float(upper_node.evaluator(local_env, exec_meta))
        if np.isnan(lower) or np.isnan(upper):
            raise NumEvaluationError(
                "The numerical integral does not converge because one of its bounds evaluated to NaN. "
                + _parameter_suffix(local_env, symbol_to_name, used_symbols)
            )

        nested_error_max = 0.0

        def scalar_integrand(t: float) -> float:
            nonlocal nested_error_max
            next_env = dict(local_env)
            next_env[var] = float(t)
            inner_value, inner_error = integrate_dimension(dim_index + 1, next_env)
            nested_error_max = max(nested_error_max, inner_error)
            return _finite_float(inner_value)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value, error = _scipy_integrate.quad(  # type: ignore[union-attr]
                    scalar_integrand,
                    lower,
                    upper,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=int(_integral_runtime_hint(hints, "integration_limit", 100)),
                )
            except NumEvaluationError as exc:
                raise NumEvaluationError(
                    "The numerical integral does not converge or diverges for the supplied parameter values. "
                    + _parameter_suffix(local_env, symbol_to_name, used_symbols)
                    + f" Details: {exc}"
                ) from exc
            except Exception as exc:
                raise NumEvaluationError(
                    "The numerical integral does not converge for the supplied parameter values. "
                    + _parameter_suffix(local_env, symbol_to_name, used_symbols)
                ) from exc

        warning_text = " ".join(str(w.message) for w in caught)
        bad_warning = any(
            isinstance(w.message, (_scipy_integrate.IntegrationWarning, RuntimeWarning))  # type: ignore[union-attr]
            for w in caught
        )
        if bad_warning or not np.isfinite(value) or not np.isfinite(error):
            message = (
                "The numerical integral does not converge or diverges for the supplied parameter values. "
                + _parameter_suffix(local_env, symbol_to_name, used_symbols)
            )
            if warning_text:
                message += f" Integrator note: {warning_text}"
            raise NumEvaluationError(message)

        error = max(float(abs(error)), float(nested_error_max))
        exec_meta.record_integration_error(error)
        return float(value), error

    value, error = integrate_dimension(0, dict(env))
    return float(value)


def _sample_integral_at_parameter_point(
    compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]],
    integrand_function: Callable[[Mapping[sympy.Symbol, Any], LastExecutionMetadata], float],
    integrand_node: _CompiledNode,
    env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    hints: Mapping[str, Any],
) -> float:
    """Approximate a finite 1D integral by dense sampling after adaptive failure."""

    if len(compiled_limits) != 1:
        raise NumEvaluationError("Sampled integral fallback supports one-dimensional finite integrals.")

    var, lower_node, upper_node = compiled_limits[0]
    lower = _bound_to_float(lower_node.evaluator(env, exec_meta))
    upper = _bound_to_float(upper_node.evaluator(env, exec_meta))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise NumEvaluationError("Sampled integral fallback requires finite bounds.")

    sample_count = _integral_sample_count(hints)
    points = np.linspace(lower, upper, sample_count, dtype=float)
    local_env = dict(env)
    try:
        values = _evaluate_sampled_integral_grid(
            integrand_node,
            local_env,
            exec_meta,
            var,
            points,
            points.shape,
            hints,
        )
    except Exception:
        values = np.empty(points.shape, dtype=float)
        for index, point in enumerate(points):
            local_env[var] = float(point)
            values[index] = integrand_function(local_env, exec_meta)
    if not np.all(np.isfinite(values)):
        raise NumEvaluationError("The sampled integral fallback produced a non-finite value.")
    return _sampled_integral_value(values, points)


def _sample_integral_at_parameter_grid(
    compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]],
    integrand_node: _CompiledNode,
    env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    hints: Mapping[str, Any],
    broadcast_shape: tuple[int, ...],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
) -> np.ndarray:
    """Approximate a 1D integral for a whole broadcasted parameter grid."""
    if len(compiled_limits) != 1:
        raise NumEvaluationError("Sampled integral fallback supports one-dimensional finite integrals.")

    var, lower_node, upper_node = compiled_limits[0]
    lower = _bound_to_float(lower_node.evaluator(env, exec_meta))
    upper = _bound_to_float(upper_node.evaluator(env, exec_meta))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise NumEvaluationError("Sampled integral fallback requires finite bounds.")

    points = np.linspace(lower, upper, _integral_sample_count(hints), dtype=float)
    local_env = dict(env)
    for symbol, array in broadcasted_values.items():
        if isinstance(symbol, sympy.IndexedBase):
            local_env[symbol] = np.asarray(array)
        else:
            local_env[symbol] = np.asarray(array).reshape((1,) + broadcast_shape)

    expected_shape = (points.size,) + broadcast_shape
    values = _evaluate_sampled_integral_grid(
        integrand_node,
        local_env,
        exec_meta,
        var,
        points,
        expected_shape,
        hints,
    )
    if not np.all(np.isfinite(values)):
        raise NumEvaluationError("The sampled integral fallback produced a non-finite value.")
    return _sampled_integral_value_axis0(values, points)


def _evaluate_sampled_integral_grid(
    integrand_node: _CompiledNode,
    local_env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    var: sympy.Symbol,
    points: np.ndarray,
    expected_shape: tuple[int, ...],
    hints: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate sampled-integral values in bounded chunks."""
    chunk_size = _integral_sample_chunk_size(hints)
    if points.size <= chunk_size:
        local_env[var] = points.reshape((points.size,) + (1,) * (len(expected_shape) - 1))
        return np.broadcast_to(
            _evaluate_integrand_grid(integrand_node, local_env, exec_meta, expected_shape),
            expected_shape,
        )

    values = np.empty(expected_shape, dtype=float)
    for start in range(0, points.size, chunk_size):
        stop = min(start + chunk_size, points.size)
        point_chunk = points[start:stop]
        chunk_shape = (point_chunk.size,) + expected_shape[1:]
        local_env[var] = point_chunk.reshape((point_chunk.size,) + (1,) * (len(expected_shape) - 1))
        values[start:stop] = np.broadcast_to(
            _evaluate_integrand_grid(integrand_node, local_env, exec_meta, chunk_shape),
            chunk_shape,
        )
    return values


def _sampled_integral_value(values: np.ndarray, points: np.ndarray) -> float:
    """Return a sampled 1D integral using Simpson's rule when possible."""

    if values.ndim == 1 and values.shape == points.shape and values.size >= 3 and values.size % 2 == 1:
        step = (float(points[-1]) - float(points[0])) / float(values.size - 1)
        total = values[0] + values[-1]
        total = total + 4.0 * np.sum(values[1:-1:2])
        total = total + 2.0 * np.sum(values[2:-1:2])
        return float(step * total / 3.0)
    return float(np.trapezoid(values, points))


def _sampled_integral_value_axis0(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return sampled 1D integrals over axis 0 using Simpson's rule when possible."""
    if values.ndim >= 1 and values.shape[0] == points.size and values.shape[0] >= 3 and values.shape[0] % 2 == 1:
        step = (float(points[-1]) - float(points[0])) / float(values.shape[0] - 1)
        total = values[0] + values[-1]
        total = total + 4.0 * np.sum(values[1:-1:2], axis=0)
        total = total + 2.0 * np.sum(values[2:-1:2], axis=0)
        return np.asarray(step * total / 3.0, dtype=float)
    return np.asarray(np.trapezoid(values, points, axis=0), dtype=float)


def _evaluate_integrand_grid(
    integrand_node: _CompiledNode,
    local_env: Mapping[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Evaluate an integral body on a vector of sample points without scalar coercion."""

    generated = integrand_node.direct_callable
    argument_symbols = integrand_node.direct_symbols
    if generated is not None:
        values = []
        for symbol in argument_symbols:
            if symbol not in local_env:
                raise NumArgumentError(f"Missing value for symbol '{symbol}'.")
            values.append(local_env[symbol])
        result = generated(*values)
    else:
        result = integrand_node.evaluator(dict(local_env), exec_meta)
    return np.broadcast_to(np.asarray(result, dtype=float), expected_shape)


def _integral_sample_count(hints: Mapping[str, Any]) -> int:
    """Return the dense-sample count used after adaptive integral failures."""

    raw_value = _integral_runtime_hint(hints, "sample_count", None)
    if raw_value is None:
        raw_value = hints.get("IntegralSamples", hints.get("integral_samples", 65537))
    try:
        sample_count = int(raw_value)
    except Exception:
        sample_count = 65537
    sample_count = max(sample_count, 3)
    if sample_count % 2 == 0:
        sample_count += 1
    return sample_count


def _integral_sample_chunk_size(hints: Mapping[str, Any]) -> int:
    """Return the maximum sampled-integral grid points evaluated at once."""
    raw_value = _integral_runtime_hint(hints, "sample_chunk_size", None)
    if raw_value is None:
        raw_value = hints.get("IntegralSampleChunkSize", hints.get("integral_sample_chunk_size", 4096))
    try:
        chunk_size = int(raw_value)
    except Exception:
        chunk_size = 4096
    return max(chunk_size, 1)


def _integrate_vectorized_1d_with_quad_vec(
    compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]],
    integrand_node: _CompiledNode,
    env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    hints: Mapping[str, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    used_symbols: frozenset[sympy.Symbol],
    shape_symbols: list[sympy.Symbol],
    broadcast_shape: tuple[int, ...],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
) -> np.ndarray | None:
    """Integrate one vector-valued 1D generated integrand with SciPy ``quad_vec``."""
    if (
        len(compiled_limits) != 1
        or integrand_node.direct_callable is None
        or integrand_node.has_runtime_integral
        or not hasattr(_scipy_integrate, "quad_vec")
        or _integral_integrator(hints) != "Quad"
    ):
        return None

    var, lower_node, upper_node = compiled_limits[0]
    shaped_symbol_set = set(shape_symbols)
    if lower_node.used_symbols & shaped_symbol_set or upper_node.used_symbols & shaped_symbol_set:
        return None

    try:
        lower = _bound_to_float(lower_node.evaluator(env, exec_meta))
        upper = _bound_to_float(upper_node.evaluator(env, exec_meta))
    except NumEvaluationError:
        return None
    if np.isnan(lower) or np.isnan(upper):
        return None

    generated = integrand_node.direct_callable
    argument_symbols = integrand_node.direct_symbols
    epsabs, epsrel = _integration_tolerances(hints)

    def vector_integrand(value: float) -> np.ndarray:
        local_env = dict(env)
        local_env[var] = float(value)
        for symbol, array in broadcasted_values.items():
            local_env[symbol] = array
        args = []
        for symbol in argument_symbols:
            if symbol not in local_env:
                raise NumArgumentError(f"Missing value for symbol '{symbol}'.")
            args.append(local_env[symbol])
        result = np.asarray(generated(*args), dtype=float)
        if result.shape == ():
            result = np.full(broadcast_shape, float(result), dtype=float)
        if result.shape != broadcast_shape:
            result = np.broadcast_to(result, broadcast_shape)
        if not np.all(np.isfinite(result)):
            raise NumEvaluationError("The numerical integrand produced a non-finite value.")
        return result

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = _scipy_integrate.quad_vec(  # type: ignore[union-attr]
                vector_integrand,
                lower,
                upper,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=int(_integral_runtime_hint(hints, "integration_limit", 100)),
            )
        except (NumArgumentError, NumEvaluationError, ValueError, TypeError):
            return None
        except Exception:
            return None

    result = np.asarray(value, dtype=float)
    warning_text = " ".join(str(w.message) for w in caught)
    bad_warning = any(
        isinstance(w.message, (_scipy_integrate.IntegrationWarning, RuntimeWarning))  # type: ignore[union-attr]
        for w in caught
    )
    if bad_warning or result.shape != broadcast_shape or not np.all(np.isfinite(result)):
        if warning_text:
            exec_meta.record_runtime_log(
                "Integral",
                "Vectorized integration fell back after an integrator warning.",
                parameters=_parameter_suffix(env, symbol_to_name, used_symbols),
                note=warning_text,
            )
        return None

    try:
        error_float = float(np.asarray(error))
    except Exception:
        error_float = 0.0
    exec_meta.record_integration_error(abs(error_float))
    return result


def _integrate_vectorized_1d_with_cubature(
    compiled_limits: list[tuple[sympy.Symbol, _CompiledNode, _CompiledNode]],
    integrand_node: _CompiledNode,
    env: dict[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    hints: Mapping[str, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    used_symbols: frozenset[sympy.Symbol],
    shape_symbols: list[sympy.Symbol],
    broadcast_shape: tuple[int, ...],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
) -> np.ndarray | None:
    """Integrate one vector-valued 1D generated integrand with SciPy ``cubature``."""
    if (
        len(compiled_limits) != 1
        or integrand_node.direct_callable is None
        or integrand_node.has_runtime_integral
        or not hasattr(_scipy_integrate, "cubature")
        or _integral_integrator(hints) not in {"Auto", "Cubature"}
    ):
        return None

    var, lower_node, upper_node = compiled_limits[0]
    shaped_symbol_set = set(shape_symbols)
    if lower_node.used_symbols & shaped_symbol_set or upper_node.used_symbols & shaped_symbol_set:
        return None

    try:
        lower = _bound_to_float(lower_node.evaluator(env, exec_meta))
        upper = _bound_to_float(upper_node.evaluator(env, exec_meta))
    except NumEvaluationError:
        return None
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None

    generated = integrand_node.direct_callable
    argument_symbols = integrand_node.direct_symbols
    epsabs, epsrel = _integration_tolerances(hints)

    def cubature_integrand(points: np.ndarray) -> np.ndarray:
        point_array = np.asarray(points, dtype=float)
        point_count = int(point_array.shape[0])
        local_env = dict(env)
        local_env[var] = point_array[:, 0].reshape((point_count,) + (1,) * len(broadcast_shape))
        for symbol, array in broadcasted_values.items():
            if isinstance(symbol, sympy.IndexedBase):
                local_env[symbol] = np.asarray(array)
            else:
                local_env[symbol] = np.asarray(array).reshape((1,) + broadcast_shape)

        args = []
        for symbol in argument_symbols:
            if symbol not in local_env:
                raise NumArgumentError(f"Missing value for symbol '{symbol}'.")
            args.append(local_env[symbol])

        result = np.asarray(generated(*args), dtype=float)
        expected_shape = (point_count,) + broadcast_shape
        if result.shape == ():
            result = np.full(expected_shape, float(result), dtype=float)
        elif result.shape == broadcast_shape:
            result = np.broadcast_to(result, expected_shape)
        elif result.shape != expected_shape:
            result = np.broadcast_to(result, expected_shape)
        if not np.all(np.isfinite(result)):
            raise NumEvaluationError("The numerical integrand produced a non-finite value.")
        return result

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _scipy_integrate.cubature(  # type: ignore[union-attr]
                cubature_integrand,
                np.array([lower], dtype=float),
                np.array([upper], dtype=float),
                rtol=epsrel,
                atol=epsabs,
                max_subdivisions=int(_integral_runtime_hint(hints, "cubature_max_subdivisions", 10000)),
            )
        except (NumArgumentError, NumEvaluationError, ValueError, TypeError):
            return None
        except Exception:
            return None

    value = np.asarray(result.estimate, dtype=float)
    error = np.asarray(result.error, dtype=float)
    status = getattr(result, "status", "converged")
    warning_text = " ".join(str(w.message) for w in caught)
    bad_warning = any(isinstance(w.message, RuntimeWarning) for w in caught)
    if (
        status != "converged"
        or bad_warning
        or value.shape != broadcast_shape
        or not np.all(np.isfinite(value))
        or not np.all(np.isfinite(error))
    ):
        if warning_text:
            exec_meta.record_runtime_log(
                "Integral",
                "Vectorized cubature fell back after an integrator warning.",
                parameters=_parameter_suffix(env, symbol_to_name, used_symbols),
                note=warning_text,
            )
        return None

    exec_meta.record_integration_error(float(np.max(np.abs(error))) if error.size else 0.0)
    return value


def _integration_tolerances(hints: Mapping[str, Any]) -> tuple[float, float]:
    """Calculate specific absolute and relative errors using runtime execution hints."""
    target = _integral_runtime_hint(hints, "target_precision", None)
    if target is None:
        return 1.49e-8, 1.49e-8
    try:
        tol = float(target)
    except Exception:
        return 1.49e-8, 1.49e-8
    if tol <= 0 or not np.isfinite(tol):
        return 1.49e-8, 1.49e-8
    return max(tol, 1e-14), max(tol, 1e-14)


def _bound_to_float(value: Any) -> float:
    """Normalize general expression bound entries to their real float equivalents."""
    if _is_positive_infinity(value):
        return np.inf
    if _is_negative_infinity(value):
        return -np.inf
    return _finite_or_infinite_float(value)


def _integer_bound(value: Any) -> int:
    """Normalize finite symbolic-reduction bounds to integer loop endpoints."""
    numeric = _as_scalar_float(value)
    if not float(numeric).is_integer():
        raise NumEvaluationError("Finite sums and products require integer bounds.")
    return int(numeric)


def _evaluate_single_axis_reduction(
    expression_node: _CompiledNode,
    operator: str,
    var: sympy.Basic,
    lower: int,
    upper: int,
    env: Mapping[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
    indexed_value_symbols: frozenset[sympy.Basic],
) -> Any:
    """Evaluate one finite reduction using a vectorized bound axis when possible."""
    count = upper - lower + 1
    if count <= 0:
        return 0 if operator == "sum" else 1

    # If the body does not use the bound variable, evaluate it once and apply
    # the reduction count directly. This preserves the old scalar-loop result
    # while avoiding repeated generated calls for array-valued bodies.
    if var not in expression_node.used_symbols:
        value = np.asarray(expression_node.evaluator(dict(env), exec_meta))
        if operator == "sum":
            return value * count
        return value ** count

    # Insert the bound values as a leading axis. Existing array inputs keep
    # their broadcast shape after that leading reduction axis.
    local_env = dict(env)
    broadcast_shape = _reduction_broadcast_shape(
        expression_node.used_symbols,
        var,
        local_env,
        indexed_value_symbols,
    )
    bound_shape = (count,) + (1,) * len(broadcast_shape)
    local_env[var] = np.arange(lower, upper + 1).reshape(bound_shape)

    try:
        values = np.asarray(expression_node.evaluator(local_env, exec_meta))
    except (TypeError, ValueError, NumArgumentError, NumEvaluationError):
        return _evaluate_single_axis_reduction_scalar_loop(
            expression_node,
            operator,
            var,
            lower,
            upper,
            env,
            exec_meta,
        )

    # Custom lowerings are allowed to be conservative. If a body collapses the
    # inserted axis, fall back to the scalar loop so behavior stays unchanged.
    if values.shape == () or values.shape[0] != count:
        return _evaluate_single_axis_reduction_scalar_loop(
            expression_node,
            operator,
            var,
            lower,
            upper,
            env,
            exec_meta,
        )

    if operator == "sum":
        return np.sum(values, axis=0)
    return np.prod(values, axis=0)


def _evaluate_single_axis_reduction_scalar_loop(
    expression_node: _CompiledNode,
    operator: str,
    var: sympy.Basic,
    lower: int,
    upper: int,
    env: Mapping[sympy.Symbol, Any],
    exec_meta: LastExecutionMetadata,
) -> Any:
    """Evaluate one finite reduction with the original scalar local-variable loop."""
    total = 0 if operator == "sum" else 1
    for position in range(lower, upper + 1):
        local_env = dict(env)
        local_env[var] = position
        value = expression_node.evaluator(local_env, exec_meta)
        if operator == "sum":
            total = np.asarray(total) + np.asarray(value)
        else:
            total = np.asarray(total) * np.asarray(value)
    return total


def _reduction_broadcast_shape(
    used_symbols: frozenset[sympy.Symbol],
    bound_symbol: sympy.Basic,
    env: Mapping[sympy.Symbol, Any],
    indexed_value_symbols: frozenset[sympy.Basic],
) -> tuple[int, ...]:
    """Return the broadcast shape of non-bound runtime values used by a reduction body."""
    shapes = [
        np.asarray(env[symbol]).shape
        for symbol in sorted(used_symbols, key=str)
        if symbol != bound_symbol
        and not isinstance(symbol, sympy.IndexedBase)
        and symbol not in indexed_value_symbols
        and symbol in env
        and _value_has_shape(env[symbol])
    ]
    if not shapes:
        return ()
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise NumArgumentError(
            "Arguments for finite reduction could not be broadcast together. "
            "Please use NumPy-compatible shapes."
        ) from exc


def _runtime_integral_cache_key(
    runtime_free_symbols: frozenset[sympy.Symbol],
    env: Mapping[sympy.Symbol, Any],
) -> tuple[tuple[str, Any], ...] | None:
    """Return a stable scalar parameter key for one runtime integral call."""
    pieces: list[tuple[str, Any]] = []
    total_size = 0
    for symbol in sorted(runtime_free_symbols, key=str):
        if symbol not in env:
            continue
        array = np.asarray(env[symbol])
        if array.shape == ():
            value = _to_python_scalar(array)
            if isinstance(value, np.generic):
                value = value.item()
            pieces.append((str(symbol), ("scalar", value)))
            continue
        if array.dtype == object:
            return None
        total_size += int(array.size)
        if total_size > 4096:
            return None
        contiguous = np.ascontiguousarray(array)
        pieces.append(
            (
                str(symbol),
                (
                    "array",
                    tuple(contiguous.shape),
                    str(contiguous.dtype),
                    contiguous.tobytes(),
                ),
            )
        )
    return tuple(pieces)


def _runtime_integral_vector_cache_key(
    runtime_symbols: tuple[sympy.Symbol, ...],
    env: Mapping[sympy.Symbol, Any],
) -> tuple[tuple[str, Any], ...] | None:
    """Return a stable key for a vectorized integral placeholder call."""
    pieces: list[tuple[str, Any]] = []
    total_size = 0
    for symbol in runtime_symbols:
        if symbol not in env:
            continue
        array = np.asarray(env[symbol])
        if array.shape == ():
            value = _to_python_scalar(array)
            if isinstance(value, np.generic):
                value = value.item()
            pieces.append((str(symbol), ("scalar", value)))
            continue
        if array.dtype == object:
            return None
        total_size += int(array.size)
        if total_size > 4096:
            return None
        contiguous = np.ascontiguousarray(array)
        pieces.append(
            (
                str(symbol),
                (
                    "array",
                    tuple(contiguous.shape),
                    str(contiguous.dtype),
                    contiguous.tobytes(),
                ),
            )
        )
    return tuple(pieces)


def _evaluate_from_scalar_integral_cache(
    runtime_symbols: frozenset[sympy.Symbol],
    env: Mapping[sympy.Symbol, Any],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
    broadcast_shape: tuple[int, ...],
    indexed_ranks: Mapping[sympy.Basic, int],
    scalar_cache: dict[tuple[tuple[str, Any], ...], float],
    integrate_scalar_point: Callable[[dict[sympy.Symbol, Any], LastExecutionMetadata], float],
    exec_meta: LastExecutionMetadata,
) -> np.ndarray | None:
    """Return a broadcast result from scalar cache entries when that is cheaper."""
    if _broadcast_point_count(broadcast_shape) > 4096:
        return None

    cached_values: dict[tuple[int, ...], float] = {}
    missing_points: list[
        tuple[tuple[int, ...], dict[sympy.Symbol, Any], tuple[tuple[str, Any], ...]]
    ] = []

    # A first vectorized call should still use the vector-valued integrator.
    # Once a nearby prefix is cached, only a small number of new coefficients
    # need scalar work during slider steps.
    for index in np.ndindex(broadcast_shape):
        point_env = _runtime_integral_scalar_point_env(
            env,
            broadcasted_values,
            indexed_ranks,
            index,
        )
        cache_key = _runtime_integral_cache_key(runtime_symbols, point_env)
        if cache_key is None:
            return None
        if cache_key in scalar_cache:
            cached_values[index] = scalar_cache[cache_key]
        else:
            missing_points.append((index, point_env, cache_key))

    if not cached_values:
        return None
    if missing_points and len(missing_points) > 8:
        return None

    result = np.empty(broadcast_shape, dtype=float)
    for index, value in cached_values.items():
        result[index] = value
    for index, point_env, cache_key in missing_points:
        value = integrate_scalar_point(point_env, exec_meta)
        scalar_cache[cache_key] = value
        result[index] = value
    return result


def _seed_scalar_integral_cache_from_broadcast_result(
    runtime_symbols: frozenset[sympy.Symbol],
    env: Mapping[sympy.Symbol, Any],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
    indexed_ranks: Mapping[sympy.Basic, int],
    result: Any,
    scalar_cache: dict[tuple[tuple[str, Any], ...], float],
) -> None:
    """Populate scalar integral cache entries from a vector-valued result."""
    values = np.asarray(result, dtype=float)
    if values.shape == ():
        return
    if _broadcast_point_count(values.shape) > 4096:
        return

    for index in np.ndindex(values.shape):
        point_env = _runtime_integral_scalar_point_env(
            env,
            broadcasted_values,
            indexed_ranks,
            index,
        )
        cache_key = _runtime_integral_cache_key(runtime_symbols, point_env)
        if cache_key is not None:
            scalar_cache[cache_key] = float(values[index])


def _broadcast_point_count(shape: tuple[int, ...]) -> int:
    """Return the number of scalar broadcast points represented by a shape."""
    if not shape:
        return 1
    return int(np.prod(shape, dtype=int))


def _runtime_integral_scalar_point_env(
    env: Mapping[sympy.Symbol, Any],
    broadcasted_values: Mapping[sympy.Symbol, np.ndarray],
    indexed_ranks: Mapping[sympy.Basic, int],
    index: tuple[int, ...],
) -> dict[sympy.Symbol, Any]:
    """Return the scalar environment for one broadcasted integral point."""
    point_env = dict(env)
    for symbol, array in broadcasted_values.items():
        if symbol in indexed_ranks:
            point_env[symbol] = _indexed_parameter_axis_slice(
                array,
                indexed_ranks[symbol],
                index,
            )
        else:
            point_env[symbol] = _to_python_scalar(array[index])
    return point_env


def _cached_vectorized_callable(
    target: Callable[..., Any],
    *,
    max_entries: int = 32,
) -> Callable[..., Any]:
    """Wrap one vectorized numeric boundary with a bounded value cache."""
    cache: dict[tuple[tuple[int, Any], ...], Any] = {}

    def call(*values: Any) -> Any:
        key = _runtime_value_sequence_cache_key(values)
        if key is not None and key in cache:
            return _copy_cached_runtime_value(cache[key])

        result = target(*values)
        if key is not None:
            cached_result = _snapshot_cacheable_runtime_value(result)
            if cached_result is not None:
                _store_limited_runtime_value_cache(cache, key, cached_result, max_entries)
        return result

    return call


def _runtime_value_sequence_cache_key(
    values: tuple[Any, ...],
) -> tuple[tuple[int, Any], ...] | None:
    """Return a stable key for small numeric scalar and array call arguments."""
    pieces: list[tuple[int, Any]] = []
    total_size = 0
    for index, value in enumerate(values):
        try:
            array = np.asarray(value)
        except Exception:
            return None

        if array.shape == ():
            scalar = _to_python_scalar(array)
            if isinstance(scalar, np.generic):
                scalar = scalar.item()
            try:
                hash(scalar)
            except TypeError:
                return None
            pieces.append((index, ("scalar", type(scalar).__name__, scalar)))
            continue

        if array.dtype == object:
            return None
        total_size += int(array.size)
        if total_size > 4096:
            return None
        contiguous = np.ascontiguousarray(array)
        pieces.append(
            (
                index,
                (
                    "array",
                    tuple(contiguous.shape),
                    str(contiguous.dtype),
                    contiguous.tobytes(),
                ),
            )
        )
    return tuple(pieces)


def _snapshot_cacheable_runtime_value(value: Any) -> Any | None:
    """Copy a numeric runtime result into cache storage when it is safe to reuse."""
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.dtype == object:
        return None
    if array.shape == ():
        scalar = _to_python_scalar(array)
        if isinstance(scalar, np.generic):
            return scalar.item()
        return scalar
    return np.array(array, copy=True)


def _copy_cached_runtime_value(value: Any) -> Any:
    """Return a cached runtime value without exposing mutable cache storage."""
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _store_limited_runtime_value_cache(
    cache: dict[tuple[tuple[int, Any], ...], Any],
    key: tuple[tuple[int, Any], ...],
    result: Any,
    max_entries: int,
) -> None:
    """Store one runtime boundary result while bounding cache growth."""
    if len(cache) >= max_entries:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[key] = result


def _store_limited_vector_integral_cache(
    cache: dict[tuple[tuple[str, Any], ...], np.ndarray],
    key: tuple[tuple[str, Any], ...],
    result: np.ndarray,
) -> None:
    """Store one vectorized integral result while bounding cache growth."""
    if len(cache) >= 8:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[key] = np.asarray(result, dtype=float).copy()


def _finite_float(value: Any) -> float:
    """Verify that a scalar numeric outcome is finite and return it as a float."""
    numeric = _as_scalar_float(value)
    if not np.isfinite(numeric):
        raise NumEvaluationError("The numerical integrand produced a non-finite value.")
    return numeric


def _finite_or_infinite_float(value: Any) -> float:
    """Verify scalar numerical limits allowing standard infinite bounds."""
    return _as_scalar_float(value, allow_infinite=True)


def _as_scalar_float(value: Any, *, allow_infinite: bool = False) -> float:
    """Verify shape signatures and parse input values into singular scalar floats."""
    if _is_positive_infinity(value):
        return np.inf
    if _is_negative_infinity(value):
        return -np.inf
    array = np.asarray(value)
    if array.shape != ():
        if array.size == 1:
            array = array.reshape(())
        else:
            raise NumEvaluationError("Expected a scalar numeric value during integration.")
    try:
        result = float(array)
    except Exception as exc:
        raise NumEvaluationError("Expected a real numeric scalar during integration.") from exc
    if not allow_infinite and not np.isfinite(result):
        raise NumEvaluationError("The numerical calculation produced a non-finite value.")
    if allow_infinite and np.isnan(result):
        raise NumEvaluationError("The numerical calculation produced NaN.")
    return result


def _parameter_suffix(
    env: Mapping[sympy.Symbol, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    used_symbols: frozenset[sympy.Symbol],
) -> str:
    """Construct localized user-friendly parameter error tracking strings."""
    pieces = []
    for sym in sorted(used_symbols, key=str):
        if sym not in env:
            continue
        name = symbol_to_name.get(sym, str(sym))
        value = env[sym]
        try:
            scalar = _to_python_scalar(value)
            pieces.append(f"{name} = {scalar!r}")
        except Exception:
            pieces.append(f"{name} = {value!r}")
    if not pieces:
        return ""
    return "Parameters: " + ", ".join(pieces) + "."


# ---------------------------------------------------------------------------
# Generated-expression lowering helpers
# ---------------------------------------------------------------------------


def _normalize_compiler_hints(hints: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize public compiler hints accepted by built-in compiler components."""
    normalized: dict[str, Any] = {}
    for key, value in dict(hints or {}).items():
        if not isinstance(key, str):
            normalized[key] = value
            continue
        compact_key = key.replace("_", "").lower()
        if "." in key:
            raise NumArgumentError(
                "Compiler hints use nested dictionaries; use "
                "{'Integral': {'ImprecisionError': ...}} instead of dotted keys."
            )
        if compact_key == "integral":
            normalized["Integral"] = _normalize_integral_hints(value)
        elif compact_key == "targetprecision":
            normalized["target_precision"] = value
        elif compact_key == "integrationlimit":
            normalized["integration_limit"] = value
        elif compact_key == "cubaturemaxsubdivisions":
            normalized["cubature_max_subdivisions"] = value
        elif compact_key == "integrator":
            normalized["Integrator"] = _normalize_integral_integrator(value)
        elif compact_key in {
            "vectorvalued",
            "vectorvaluedintegration",
            "vectorizedintegration",
            "vectorizedintegrals",
        }:
            normalized["VectorValued"] = _normalize_bool_mode(value, "Integral VectorValued")
        elif compact_key in {"cse", "commonsubexpressionelimination"}:
            normalized["CSE"] = _normalize_cse_hint(value)
        else:
            normalized[key] = value
    return normalized


def _normalize_integral_hints(value: Any) -> dict[str, Any]:
    """Normalize nested integral-owned hint keys and string mode values."""
    if not isinstance(value, Mapping):
        raise NumArgumentError("Integral hints must be provided as a nested mapping.")

    normalized: dict[str, Any] = {}
    for option_key, option_value in value.items():
        if not isinstance(option_key, str):
            normalized[option_key] = option_value
            continue
        compact_key = option_key.replace("_", "").lower()
        if compact_key == "imprecisionerror":
            if not isinstance(option_value, str):
                raise NumArgumentError("Integral ImprecisionError must be a string mode.")
            mode_lookup = {
                "nan": "Nan",
                "ignore": "Ignore",
                "raise": "Raise",
            }
            mode = mode_lookup.get(option_value.lower())
            if mode is None:
                raise NumArgumentError(
                    "Integral ImprecisionError must be one of 'Nan', 'Ignore', or 'Raise'."
                )
            normalized["ImprecisionError"] = mode
        elif compact_key == "integrationlimit":
            normalized["integration_limit"] = option_value
        elif compact_key == "cubaturemaxsubdivisions":
            normalized["cubature_max_subdivisions"] = option_value
        elif compact_key == "targetprecision":
            normalized["target_precision"] = option_value
        elif compact_key == "integrator":
            normalized["Integrator"] = _normalize_integral_integrator(option_value)
        elif compact_key in {
            "vectorvalued",
            "vectorvaluedintegration",
            "vectorizedintegration",
            "vectorizedintegrals",
        }:
            normalized["VectorValued"] = _normalize_bool_mode(option_value, "Integral VectorValued")
        else:
            normalized[option_key] = option_value
    return normalized


def _normalize_integral_integrator(value: Any) -> str:
    """Normalize the integral integrator selection mode."""
    if not isinstance(value, str):
        raise NumArgumentError("Integral Integrator must be a string mode.")
    mode = {
        "quad": "Quad",
        "cubature": "Cubature",
        "auto": "Auto",
        "sampled": "Sampled",
        "sample": "Sampled",
        "samples": "Sampled",
    }.get(value.lower())
    if mode is None:
        raise NumArgumentError(
            "Integral Integrator must be one of 'Quad', 'Cubature', 'Sampled', or 'Auto'."
        )
    return mode


def _normalize_cse_hint(value: Any) -> bool | Callable[..., Any]:
    """Normalize the generated-expression common-subexpression hint."""
    if isinstance(value, bool):
        return value
    if callable(value):
        return value
    if isinstance(value, str):
        mode = value.lower()
        if mode in {"true", "yes", "on", "enable", "enabled"}:
            return True
        if mode in {"false", "no", "off", "disable", "disabled"}:
            return False
    raise NumArgumentError(
        "CSE must be a boolean, a callable, or one of 'On'/'Off'."
    )


def _normalize_bool_mode(value: Any, label: str) -> bool:
    """Normalize a public boolean optimization mode."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        mode = value.lower()
        if mode in {"true", "yes", "on", "enable", "enabled"}:
            return True
        if mode in {"false", "no", "off", "disable", "disabled"}:
            return False
    raise NumArgumentError(f"{label} must be a boolean or one of 'On'/'Off'.")


def _integral_imprecision_policy(hints: Mapping[str, Any]) -> str:
    """Return the normalized runtime integral imprecision policy."""
    integral_hints = hints.get("Integral", {})
    if not isinstance(integral_hints, Mapping):
        return "Nan"
    policy = integral_hints.get("ImprecisionError", "Nan")
    if isinstance(policy, str):
        mode = {"nan": "Nan", "ignore": "Ignore", "raise": "Raise"}.get(policy.lower())
        if mode is not None:
            return mode
    raise NumArgumentError(
        "Integral ImprecisionError must be one of 'Nan', 'Ignore', or 'Raise'."
    )


def _integral_runtime_hint(hints: Mapping[str, Any], key: str, default: Any) -> Any:
    """Return one integral-owned runtime hint with compatibility fallback."""
    integral_hints = hints.get("Integral", {})
    if isinstance(integral_hints, Mapping) and key in integral_hints:
        return integral_hints[key]
    return hints.get(key, default)


def _integral_integrator(hints: Mapping[str, Any]) -> str:
    """Return the normalized numerical integration backend mode."""
    integral_hints = hints.get("Integral", {})
    if isinstance(integral_hints, Mapping) and "Integrator" in integral_hints:
        return _normalize_integral_integrator(integral_hints["Integrator"])
    if "Integrator" in hints:
        return _normalize_integral_integrator(hints["Integrator"])
    return "Quad"


def _integral_vector_valued_enabled(hints: Mapping[str, Any]) -> bool:
    """Return whether vector-valued integral family evaluation is enabled."""
    integral_hints = hints.get("Integral", {})
    if isinstance(integral_hints, Mapping) and "VectorValued" in integral_hints:
        return _normalize_bool_mode(integral_hints["VectorValued"], "Integral VectorValued")
    if "VectorValued" in hints:
        return _normalize_bool_mode(hints["VectorValued"], "Integral VectorValued")
    return True


def _generated_cse_option(hints: Mapping[str, Any]) -> bool | Callable[..., Any]:
    """Return the normalized lambdify CSE option for generated expressions."""
    if "CSE" not in hints:
        return True
    return _normalize_cse_hint(hints["CSE"])


def _record_integral_failure(
    exec_meta: LastExecutionMetadata,
    policy: str,
    exc: Exception,
    env: Mapping[sympy.Symbol, Any],
    symbol_to_name: Mapping[sympy.Symbol, str],
    used_symbols: frozenset[sympy.Symbol],
) -> None:
    """Record one pointwise runtime integration failure on execution metadata."""
    message = str(exc)
    if hasattr(exec_meta, "record_runtime_log"):
        exec_meta.record_runtime_log(
            "Integral",
            message,
            policy=policy,
            parameters=_parameter_suffix(env, symbol_to_name, used_symbols),
        )


def _ensure_lowering_state(ctx: CompilationContext) -> _LoweringState:
    """Return the shared lowering state or create a compatibility-local one."""
    if ctx.lowering_state is not None:
        return ctx.lowering_state
    return _LoweringState(_CompileLogSink())


def _ordered_symbols(
    symbols: Any,
    arg_specs: tuple[NumArgSpec, ...],
) -> tuple[sympy.Symbol, ...]:
    """Return generated callable arguments in explicit public then local order."""
    symbol_set = set(symbols)
    ordered = []
    consumed: set[sympy.Basic] = set()
    for spec in arg_specs:
        if spec.symbol in symbol_set:
            ordered.append(spec.symbol)
            consumed.add(spec.symbol)
            continue
        if isinstance(spec.symbol, sympy.IndexedBase):
            label = _indexed_base_label(spec.symbol)
            if label in symbol_set:
                ordered.append(spec.symbol)
                consumed.add(label)
    remaining = [
        symbol
        for symbol in sorted(symbol_set, key=str)
        if symbol not in ordered and symbol not in consumed
    ]
    return tuple(ordered + remaining)


def _register_lowered_callable(
    lowered: LoweredCallable,
    state: _LoweringState,
    module_map: dict[str, Callable[..., Any]],
    *,
    category: str,
    extension_name: str | None = None,
) -> sympy.Basic:
    """Insert one generated placeholder call and register its implementation."""
    name = state.next_name(category, extension_name)
    module_map[name] = lowered.callable
    return _generated_function_head(name)(*lowered.args)


def _generated_function_head(name: str) -> Any:
    """Create a compiler-internal undefined function without public name validation."""
    try:
        from sympy.core.function import AppliedUndef, UndefinedFunction
        from math_toolkit.sympy_extensions import symbol_name_validation

        return symbol_name_validation._ORIGINAL_UNDEFINEDFUNCTION_NEW(
            UndefinedFunction,
            name,
            bases=(AppliedUndef,),
            __dict__={},
        )
    except Exception:
        return sympy.Function(name)


def _coerce_lowering_result(
    result: Any,
    ctx: CompilationContext,
    state: _LoweringState,
    *,
    source: str,
) -> _BoundaryLowering:
    """Normalize modern and compatibility extension results to LoweredCallable."""
    if isinstance(result, LoweredCallable):
        return _BoundaryLowering(result, has_runtime_integral=False)

    if isinstance(result, ImplementedFunction):
        args = tuple(spec.symbol for spec in ctx.arg_specs)
        symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}

        def call_implemented_escape(*values: Any) -> Any:
            kwargs = {
                symbol_to_name[symbol]: value
                for symbol, value in zip(args, values)
            }
            return result.raw_implementation(**kwargs)

        state.log_sink.record_compile_log(
            "compatibility",
            f"{source} returned ImplementedFunction; use LoweredCallable for new code.",
        )
        return _BoundaryLowering(
            LoweredCallable(callable=call_implemented_escape, args=args),
            has_runtime_integral=False,
        )

    if isinstance(result, NumFunction):
        args = tuple(spec.symbol for spec in result.arg_specs)

        def call_num_function(*values: Any) -> Any:
            kwargs = {
                spec.name: value
                for spec, value in zip(result.arg_specs, values)
            }
            value = result(**kwargs)
            active_metadata = state.log_sink.current_metadata()
            if active_metadata is not None:
                error = result.last_execution_metadata.integration_error_bound
                if error is not None:
                    active_metadata.record_integration_error(error)
                for entry in getattr(result.last_execution_metadata, "runtime_logs", []):
                    active_metadata.runtime_logs.append(dict(entry))
            return value

        return _BoundaryLowering(
            LoweredCallable(callable=call_num_function, args=args),
            has_runtime_integral=result.metadata.uses_runtime_numerical_integration,
        )

    if callable(result):
        args = tuple(spec.symbol for spec in ctx.arg_specs)
        symbol_to_name = {spec.symbol: spec.name for spec in ctx.arg_specs}

        def call_raw_callable(*values: Any) -> Any:
            kwargs = {
                symbol_to_name[symbol]: value
                for symbol, value in zip(args, values)
            }
            return result(**kwargs)

        state.log_sink.record_compile_log(
            "compatibility",
            f"{source} returned a raw callable; use LoweredCallable for new code.",
        )
        return _BoundaryLowering(
            LoweredCallable(callable=call_raw_callable, args=args),
            has_runtime_integral=False,
        )

    raise NumUnsupportedExpressionError(
        "I ran into a part of your expression I don't know how to convert: "
        f"{source}. You didn't do anything wrong! Compiler extensions and _mt_compile "
        "methods must return a LoweredCallable."
    )


def _match_lowering_extension(
    extensions: Mapping[Any, Any],
    ctx: CompilationContext,
) -> tuple[str, Callable[[CompilationContext], Any]] | None:
    """Return the first extension that claims the current node."""
    node = ctx.current_node
    for registered_type, entry in extensions.items():
        try:
            type_matches = isinstance(node, registered_type)
        except TypeError:
            type_matches = type(node) is registered_type
        if not type_matches:
            continue

        if isinstance(entry, Mapping):
            condition = entry.get("condition")
            if callable(condition) and not _extension_condition_matches(condition, node, ctx):
                continue
            handler = entry.get("compile", entry.get("handler"))
            extension_name = str(entry.get("name") or getattr(registered_type, "__name__", "Extension"))
        else:
            handler = entry
            extension_name = getattr(registered_type, "__name__", "Extension")
        if not callable(handler):
            continue
        return _valid_extension_fragment(extension_name), handler
    return None


def _extension_condition_matches(
    condition: Callable[..., Any],
    node: sympy.Basic,
    ctx: CompilationContext,
) -> bool:
    """Evaluate an extension condition against the current node and context."""
    try:
        signature = inspect.signature(condition)
    except (TypeError, ValueError):
        return bool(condition(ctx))

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
    ]
    if len(positional) >= 2:
        return bool(condition(node, ctx))
    return bool(condition(ctx))


def _valid_extension_fragment(name: str) -> str:
    """Return a valid generated-name extension fragment without underscores."""
    fragment = "".join(ch for ch in str(name) if ch.isalnum())
    if not fragment:
        fragment = "Extension"
    if fragment[0].isdigit():
        fragment = f"X{fragment}"
    return fragment


# ---------------------------------------------------------------------------
# General Internal Helpers
# ---------------------------------------------------------------------------


_NUMPY_FUNCTIONS: dict[Any, Callable[..., Any]] = {
    sympy.sin: np.sin,
    sympy.cos: np.cos,
    sympy.tan: np.tan,
    sympy.asin: np.arcsin,
    sympy.acos: np.arccos,
    sympy.atan: np.arctan,
    sympy.sinh: np.sinh,
    sympy.cosh: np.cosh,
    sympy.tanh: np.tanh,
    sympy.exp: np.exp,
    sympy.log: np.log,
    sympy.sqrt: np.sqrt,
    sympy.Abs: np.abs,
    sympy.floor: np.floor,
    sympy.ceiling: np.ceil,
    sympy.sign: np.sign,
}

def _merge_node(evaluator: Callable[[dict[sympy.Symbol, Any], LastExecutionMetadata], Any], children: list[_CompiledNode]) -> _CompiledNode:
    """Combine a group of separate child node specs into a unified execution track."""
    used: set[sympy.Symbol] = set()
    has_runtime = False
    for child in children:
        used.update(child.used_symbols)
        has_runtime = has_runtime or child.has_runtime_integral
    return _CompiledNode(evaluator, used_symbols=frozenset(used), has_runtime_integral=has_runtime)


def _constant_node(value: Any) -> _CompiledNode:
    """Build an internal evaluation block targeting static invariant constants."""
    def eval_constant(env: dict[sympy.Symbol, Any], exec_meta: LastExecutionMetadata) -> Any:
        return value

    return _CompiledNode(eval_constant, used_symbols=frozenset(), has_runtime_integral=False)


def _safe_deepcopy(value: Any) -> Any:
    """Execute deep defensive object copying while bypassing uncopyable entities."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _is_numeric_atom(node: Any) -> bool:
    """Verify if an expression element is a member of the SymPy concrete numeric hierarchy."""
    return isinstance(node, (sympy.Number, sympy.NumberSymbol))


def _sympy_to_float(node: Any) -> float:
    """Evaluate complex or basic symbolic entries directly into float scalars."""
    if _is_positive_infinity(node):
        return np.inf
    if _is_negative_infinity(node):
        return -np.inf
    if node == sympy.pi:
        return float(np.pi)
    if node == sympy.E:
        return float(np.e)
    if node == sympy.EulerGamma:
        return float(np.euler_gamma)
    if node == sympy.GoldenRatio:
        return float((1.0 + np.sqrt(5.0)) / 2.0)
    if node == sympy.Catalan:
        return 0.915965594177219
    if isinstance(node, sympy.Integer):
        return float(int(node))
    if isinstance(node, sympy.Rational):
        return float(int(node.p)) / float(int(node.q))
    if isinstance(node, sympy.Float):
        return float(node)
    if isinstance(node, sympy.NumberSymbol):
        raise TypeError(f"Unsupported symbolic numeric constant: {node!s}.")
    return float(node)


def _constant_expression_to_float(value: Any) -> float:
    """Evaluate a numeric expression using only explicit compiler mappings."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if _is_positive_infinity(value):
        return np.inf
    if _is_negative_infinity(value):
        return -np.inf
    if _is_numeric_atom(value):
        return _sympy_to_float(value)

    if isinstance(value, sympy.Add):
        total = 0.0
        for child in value.args:
            total += _constant_expression_to_float(child)
        return total

    if isinstance(value, sympy.Mul):
        product = 1.0
        for child in value.args:
            product *= _constant_expression_to_float(child)
        return product

    if isinstance(value, sympy.Pow):
        base = _constant_expression_to_float(value.args[0])
        exponent = _constant_expression_to_float(value.args[1])
        return float(np.power(base, exponent))

    if isinstance(value, sympy.Function):
        np_func = _NUMPY_FUNCTIONS.get(value.func)
        if np_func is None:
            raise TypeError(f"Unsupported constant function: {value.func!s}.")
        args = [_constant_expression_to_float(child) for child in value.args]
        return _as_scalar_float(np_func(*args))

    if isinstance(value, sympy.Basic):
        if value.free_symbols:
            raise TypeError("Cannot convert symbolic expression with free symbols to float.")
        raise TypeError(f"Unsupported constant expression: {value!s}.")

    return float(value)


def _is_positive_infinity(value: Any) -> bool:
    """Evaluate if a value matches symbolic or core mathematical positive infinity definitions."""
    if value is sympy.oo:
        return True
    try:
        if value == sympy.oo:
            return True
    except Exception:
        pass
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return array.shape == () and bool(array == np.inf)


def _is_negative_infinity(value: Any) -> bool:
    """Evaluate if a value matches symbolic or core mathematical negative infinity definitions."""
    if value is -sympy.oo:
        return True
    try:
        if value == -sympy.oo:
            return True
    except Exception:
        pass
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return array.shape == () and bool(array == -np.inf)


def _evaluate_constant_integral(node: sympy.Integral) -> float:
    """Evaluate a purely static constant integral symbolically or via precision float calculation."""
    evaluated = node.doit()
    if isinstance(evaluated, sympy.Integral):
        raise ValueError("Integral did not resolve to a numeric value.")
    if getattr(evaluated, "has", lambda *_: False)(sympy.Integral):
        raise ValueError("Integral did not resolve to a numeric value.")
    return _constant_expression_to_float(evaluated)


def _normalize_numeric_result(result: Any) -> Any:
    """Normalize internal evaluation products into standard Python float types or array definitions."""
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
        except Exception:
            return result
            
    array = np.asarray(result)
    if array.shape == () and np.issubdtype(array.dtype, np.number):
        return float(array)
    return result


def _broadcast_constant_to_kwargs(result: Any, kwargs: Mapping[str, Any]) -> Any:
    """Propagate single constant values across layout dimensions of input keyword arguments."""
    arrays = [np.asarray(value) for value in kwargs.values() if np.asarray(value).shape != ()]
    if not arrays:
        return result
        
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


def _value_has_shape(value: Any) -> bool:
    """Determine if a provided argument exhibits an evaluation shape dimension."""
    return np.asarray(value).shape != ()


def _to_python_scalar(value: Any) -> Any:
    """Extract singular scalar elements out of zero-dimensional array grids."""
    array = np.asarray(value)
    if array.shape == ():
        try:
            return array.item()
        except Exception:
            return float(array)
            
    if array.size == 1:
        return array.reshape(()).item()
    raise ValueError("not scalar")


def _friendly_runtime_message(exc: Exception) -> str:
    """Format and build explanatory diagnostic messaging for evaluation execution failures."""
    return (
        "The compiled numerical function could not be evaluated for the supplied inputs. "
        "Please check that argument values are real numeric scalars or NumPy arrays with compatible shapes. "
        f"Details: {exc}"
    )


def _get_local_escape_hatch(node: Any) -> Optional[Callable[[CompilationContext], ImplementedFunction]]:
    """Locate customized compile escape methods declared directly on individual expression nodes."""
    method = getattr(node, "_mt_compile", None)
    if method is None:
        return None
        
    # ISSUE: Redundant logic testing callable sequentially.
    if "_mt_compile" not in type(node).__dict__ and not callable(method):
        return None
    if not callable(method):
        return None
        
    return method


def _contains_local_escape_hatch(expr: sympy.Basic) -> bool:
    """Scan the expression pre-order traversal layout for custom compilation escape hatches."""
    for node in sympy.preorder_traversal(expr):
        if _get_local_escape_hatch(node) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Global Builtin Registration Shim
# ---------------------------------------------------------------------------


# ISSUE: Catch-all exception allows silent failure of the builtin shim initialization loop.
try:  # pragma: no cover - compatibility shim only.
    import builtins as _builtins

    if not hasattr(_builtins, "PinnedExpression"):
        _builtins.PinnedExpression = PinnedExpression
    if not hasattr(_builtins, "NumEvaluationError"):
        _builtins.NumEvaluationError = NumEvaluationError
except Exception:
    pass
