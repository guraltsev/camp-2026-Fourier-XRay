"""Expose the curated public ``num`` surface during the numeric rewrite."""

from __future__ import annotations

__all__ = []
__notebook__ = []

from .num import (
    IndexedRuntimeParameterInfo,
    compile,
    Num,
    NumArgumentError,
    NumNotImplementedError,
    NumUnsupportedExpressionError,
    autodetect_args,
    autodetect_runtime_args,
    indexed_runtime_parameter_info,
    sanitize_symbol_name,
)
from .numfunction_implementedfunction import (
    CompilationMetadata,
    ImplementedFunction,
    LastExecutionMetadata,
    NumArgSpec,
    NumEvaluationError,
    NumFunction,
    ShapeParseError,
    ShapeSpec,
)
from .Compiler import (
    CompilationContext,
    LoweredCallable,
    NumCompiler,
    NumUnsupportedExpressionError,
    PinnedExpression,
)

__all__ += [
    "compile",
    "IndexedRuntimeParameterInfo",
    "Num",
    "NumArgumentError",
    "NumNotImplementedError",
    "NumUnsupportedExpressionError",
    "autodetect_args",
    "autodetect_runtime_args",
    "indexed_runtime_parameter_info",
    "sanitize_symbol_name",
    "NumFunction",
    "ImplementedFunction",
    "CompilationMetadata",
    "NumArgSpec",
    "ShapeParseError",
    "ShapeSpec",
    "CompilationContext",
    "LoweredCallable",
    "LastExecutionMetadata",
    "NumCompiler",
    "NumEvaluationError",
    "PinnedExpression",
]

__notebook__ += [
    "compile",
    "Num",
    "NumFunction",
    "ImplementedFunction",
    "LoweredCallable",
]
