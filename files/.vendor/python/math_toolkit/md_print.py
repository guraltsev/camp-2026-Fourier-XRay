"""Render small markdown documents with inline mathematical objects.

The public entry point is ``md(...)``. It displays rendered Markdown by
default and returns rendered text by default when display is suppressed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from html import escape as html_escape
import importlib.util
import inspect
import sys
from types import FrameType
from typing import Any, Literal

from lark import Lark, Transformer, Token
from lark.exceptions import VisitError
import sympy

from .pipeops import pipeop

__all__ = ["md"]


# ==============================================================================
# Public Entrypoint
# ==============================================================================

def md(
    content: Any,
    namespace: Mapping[str, Any] | None = None,
    *,
    display: bool = True,
    Return: Literal["markdown", "original"] | None = None,
) -> Any:
    """Display markdown text rendered from a string or object.

    String input is treated as a Markdown template. Non-string input is
    treated as a single insertion, so ``md(obj)`` behaves like rendering
    ``{{ obj }}`` in prose. Insertions use Python expressions in double braces,
    evaluated in the caller's namespace by default. SymPy expressions render
    as LaTeX. Inside existing markdown math delimiters, such as ``$...$`` or
    ``$$...$$``, they render as raw LaTeX fragments. In ordinary prose, they
    render as inline math.

    Use command tags such as ``{table(header=["A", "B"])}{{ data }}`` for structured
    Markdown fragments. Table output shows at most 20 data rows by default;
    pass ``max_rows=0`` to render all rows. The command name is resolved by
    ``md``, while the data expression and option values are evaluated in
    the caller's namespace. Plain single braces remain literal Markdown or
    LaTeX text.

    Parameters
    ----------
    content : Any
        Markdown template text or one object to render as a single insertion.
    namespace : Mapping[str, Any] | None, default=None
        Optional expression namespace. When omitted, the caller's locals and
        globals are used.
    display : bool, default=True
        Whether to display the rendered Markdown through IPython's display
        hook when available.
    Return : {'markdown', 'original'} | None, default=None
        Whether to return the rendered Markdown text or the original provided
        object. When omitted, ``md`` returns rendered Markdown when
        ``display=False`` and otherwise returns ``None``.

    Returns
    -------
    str | object | None
        The rendered Markdown text when ``display=False`` or when ``Return``
        is ``"markdown"``, the original provided object when ``Return`` is
        ``"original"``, and otherwise ``None``.
    """
    if Return not in (None, "markdown", "original"):
        raise ValueError("md Return must be None, 'markdown', or 'original'.")

    # Treat non-string input as a one-slot template so users can pass objects
    # directly and get the same rendering pipeline as explicit insertions.
    if isinstance(content, str):
        source = content
        effective_namespace = namespace
    else:
        source = "{{ __md_value__ }}"
        effective_namespace = dict(namespace or {})
        effective_namespace["__md_value__"] = content

    # Inspect the call stack to capture the caller's execution environment
    # when an explicit evaluation namespace is not provided.
    frame = inspect.currentframe()
    caller_frame: FrameType | None = None
    try:
        caller_frame = _resolve_md_caller_frame(frame)
        globals_map, locals_map = _build_evaluation_scope(
            effective_namespace,
            caller_frame,
        )
    finally:
        del caller_frame
        del frame

    # Process the template string and render any matched moustache tags.
    rendered = _MarkdownText(_render_template(source, globals_map, locals_map))
    if display:
        _display_markdown_text(rendered)

    # Return either the rendered markdown or the original caller-provided
    # object, without coupling the result contract to display side effects.
    if Return == "markdown":
        return rendered
    if Return == "original":
        return content
    if not display:
        return rendered
    return None


# ==============================================================================
# Template Parsing and Evaluation (Lark)
# ==============================================================================

GRAMMAR = r"""
start: (text | directive_tag | expr_tag_escaped | expr_tag_standard)*

# Text fallback catches any arbitrary text outside of the template tags
text: TEXT_CHAR+
TEXT_CHAR: /[^{]/ | /\{[^{E]/ | /\{E[^{]/ | /\{$/ | /\{E$/

# =============================================================================
# 1. PURE EXPRESSION TAG HOOKS
# =============================================================================
expr_tag_standard: "{{" WS expr_no_double_close [WS] "}}"
expr_tag_escaped: "{E{" [WS] expr_escaped_content [WS] "}"

# =============================================================================
# 2. COMMAND DIRECTIVE TAG HOOKS
# =============================================================================
directive_tag: directive_head [WS] directive_body

directive_head: "{" [WS] IDENTIFIER [WS] "(" [WS] kwargs_std [WS] ")" [WS] "}" -> head_std
              | "{E" [WS] IDENTIFIER [WS] "(" [WS] kwargs_esc [WS] ")" [WS] "}" -> head_esc

directive_body: "{{" WS expr_no_double_close [WS] "}}"  -> body_std
              | "{E{" [WS] expr_escaped_content [WS] "}"  -> body_esc

# =============================================================================
# 3. CONTEXT-ISOLATED EXPRESSION LEXING RULES
# =============================================================================

kwargs_std: item_kwargs_std*
kwargs_esc: item_escaped_content*

expr_no_double_close: item_no_double_close+
?item_no_double_close: OTHER_TEXT | WS | STRING | COMMA | LPAREN | RPAREN | LBRACK | RBRACK | LBRACE | COLON | EQUALS | single_brace_close
?item_kwargs_std: OTHER_TEXT | WS | STRING | COMMA | LPAREN | RPAREN | LBRACK | RBRACK | LBRACE | COLON | EQUALS | single_brace_close_in_directive

expr_escaped_content: item_escaped_content+
?item_escaped_content: OTHER_TEXT | WS | STRING | COMMA | LPAREN | RPAREN | LBRACK | RBRACK | LBRACE | COLON | EQUALS | ESCAPED_BRACE | ESCAPED_BACKSLASH

# =============================================================================
# 4. TERMINAL DEFINITIONS
# =============================================================================
OTHER_TEXT: /[^{}\[\]()\'\",:= \t\n\r]+/
WS: /[ \t\n\r]+/
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
COMMA: ","
LPAREN: "("
RPAREN: ")"
LBRACK: "["
RBRACK: "]"
LBRACE: "{"
COLON: ":"
EQUALS: "="

# Standard Python String literal configuration prevents internal string leaks
STRING: /\"([^\"\\]|\\.)*\"/ | /\'([^\'\\]|\\.)*\'/

# Symmetrical Escapes
ESCAPED_BRACE: "\\}"
ESCAPED_BACKSLASH: "\\\\"
single_brace_close: /}(?!})/
single_brace_close_in_directive: "}" ~ 1
"""


class _TemplateTransformer(Transformer):
    """Lark transformer to evaluate expressions and render markdown fragments.
    
    It natively restores escaped structural tokens exclusively within the escaped 
    token pathways, preventing brittle global regex replacements across arbitrary 
    code strings.
    """

    def __init__(self, globals_map: Mapping[str, Any], locals_map: Mapping[str, Any]):
        self.globals_map = globals_map
        self.locals_map = locals_map
        self.state = _MarkdownState()

    def start(self, children):
        # Assemble all processed text chunks and tag replacements into one document.
        return "".join(children)

    def text(self, children):
        # Feed plain prose text to the markdown state tracker and pass it through.
        val = "".join(children)
        self.state.feed(val)
        return val

    @staticmethod
    def _is_structural_ws(node: object) -> bool:
        """Return whether ``node`` is grammar-only whitespace."""
        return isinstance(node, str) and not node.strip()

    def _non_structural_nodes(self, children: Sequence[object]) -> list[object]:
        """Return children with grammar-only whitespace removed."""
        return [
            child
            for child in children
            if child is not None and not self._is_structural_ws(child)
        ]

    def expr_tag_standard(self, children):
        # Extract the expression while omitting surrounding structural WS tokens
        expression = next(iter(self._non_structural_nodes(children)), "")
        return self._render_expr(expression)

    def expr_tag_escaped(self, children):
        expression = next(iter(self._non_structural_nodes(children)), "")
        return self._render_expr(expression)

    def _render_expr(self, expression: str):
        if not expression.strip():
            raise ValueError("Empty md insertion in markdown template.")
        
        value = _evaluate_expression(expression, self.globals_map, self.locals_map)
        replacement = _render_value(value, math_mode=self.state.in_math)
        self.state.feed(replacement)
        return replacement

    def head_std(self, children):
        nodes = self._non_structural_nodes(children)
        name = str(nodes[0])
        kwargs_expr = str(nodes[1]) if len(nodes) > 1 else ""
        return name, kwargs_expr

    def head_esc(self, children):
        nodes = self._non_structural_nodes(children)
        name = str(nodes[0])
        kwargs_expr = str(nodes[1]) if len(nodes) > 1 else ""
        return name, kwargs_expr

    def body_std(self, children):
        return next(iter(self._non_structural_nodes(children)), "")

    def body_esc(self, children):
        return next(iter(self._non_structural_nodes(children)), "")

    def directive_tag(self, children):
        nodes = self._non_structural_nodes(children)

        name, kwargs_expr = nodes[0]
        content_expr = str(nodes[1])

        if name != "table":
            raise ValueError(f"Unknown md command '{name}'. Available commands: table.")

        # Guard against nested tags to keep rendering predictable and secure.
        if "{{" in content_expr or "}}" in content_expr:
            raise ValueError(
                "Nested md insertions are not allowed; command subjects "
                "and option values are already Python expressions."
            )

        data = _evaluate_expression(content_expr, self.globals_map, self.locals_map)

        kwargs = {}
        if "{{" in kwargs_expr or "}}" in kwargs_expr:
            raise ValueError(
                "Nested md insertions are not allowed; command subjects "
                "and option values are already Python expressions."
            )
        
        if kwargs_expr.strip():
            # Evaluate using standard python dict construct syntax to handle keywords
            dict_expr = f"dict({kwargs_expr})"
            kwargs = _evaluate_expression(dict_expr, self.globals_map, self.locals_map)

        if "header" not in kwargs:
            kwargs["header"] = False

        value = _table(data, **kwargs)
        replacement = _render_value(value, math_mode=self.state.in_math)
        self.state.feed(replacement)
        return replacement

    def kwargs_std(self, children):
        return "".join(str(c) for c in children)

    def kwargs_esc(self, children):
        return "".join(str(c) for c in children)

    def expr_no_double_close(self, children):
        return "".join(str(c) for c in children)

    def expr_escaped_content(self, children):
        return "".join(str(c) for c in children)

    def item_no_double_close(self, children):
        return "".join(str(c) for c in children)

    def item_escaped_content(self, children):
        return "".join(str(c) for c in children)

    # Clean AST Restoration for explicitly requested backslash-escapes
    def ESCAPED_BRACE(self, token):
        return "}"

    def ESCAPED_BACKSLASH(self, token):
        return "\\"

    def single_brace_close(self, children):
        return "}"

    def OTHER_TEXT(self, token):
        return str(token)

    def WS(self, token):
        return str(token)

    def STRING(self, token):
        return str(token)

    def IDENTIFIER(self, token):
        return str(token)

    def COMMA(self, token):
        return str(token)

    def LPAREN(self, token):
        return str(token)

    def RPAREN(self, token):
        return str(token)

    def LBRACK(self, token):
        return str(token)

    def RBRACK(self, token):
        return str(token)

    def LBRACE(self, token):
        return str(token)

    def COLON(self, token):
        return str(token)

    def EQUALS(self, token):
        return str(token)


def _render_template(
    template: str,
    globals_map: Mapping[str, Any],
    locals_map: Mapping[str, Any],
) -> str:
    """Render every moustache tag in ``template`` using the provided scope."""
    # Build and execute the Lark parser with an Earley engine to robustly
    # scan for structural tags, directives, and text fragments.
    parser = Lark(GRAMMAR, parser="earley", start="start", keep_all_tokens=False)
    try:
        tree = parser.parse(template)
    except Exception as e:
        raise ValueError("Unmatched parsing boundaries or syntax error in markdown template.") from e

    # Transform the parsed tree into the final rendered markdown string.
    transformer = _TemplateTransformer(globals_map, locals_map)
    try:
        return transformer.transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, Exception):
            raise exc.orig_exc
        raise


def _build_evaluation_scope(
    namespace: Mapping[str, Any] | None,
    caller_frame: FrameType | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the explicit namespace or capture the caller's frame mappings."""
    if namespace is not None:
        return {"__builtins__": __builtins__}, dict(namespace)

    if caller_frame is None:
        fallback = {"__builtins__": __builtins__}
        return fallback, fallback

    return dict(caller_frame.f_globals), dict(caller_frame.f_locals)


def _resolve_md_caller_frame(frame: FrameType | None) -> FrameType | None:
    """Return the first non-pipeops caller frame above ``md``."""
    caller_frame = frame.f_back if frame is not None else None
    while caller_frame is not None:
        module_name = caller_frame.f_globals.get("__name__")
        if module_name not in {
            "math_toolkit.pipeops",
            "math_toolkit.pipeops.core",
            "math_toolkit.pipeops.common_pipeops",
        }:
            return caller_frame
        caller_frame = caller_frame.f_back
    return None


def _evaluate_expression(
    expression: str,
    globals_map: Mapping[str, Any],
    locals_map: Mapping[str, Any],
) -> Any:
    """Evaluate one expression in the caller-controlled scope."""
    code = compile(expression, "<md expression>", "eval")
    return eval(code, dict(globals_map), dict(locals_map))


def _render_value(value: Any, *, math_mode: bool) -> str:
    """Render one evaluated insertion value for a markdown context."""
    if isinstance(value, _MarkdownFragment):
        if math_mode:
            if value.latex is None:
                raise ValueError(
                    "Markdown table fragments cannot be inserted inside math."
                )
            return value.latex
        return value.markdown

    # Use SymPy's specialized LaTeX printer for objects implementing a _latex
    # method or mapping to recognized basic math primitives.
    if _is_sympy_renderable(value):
        latex = sympy.latex(value)
        return latex if math_mode else f"${latex}$"

    text = str(value)
    return text if math_mode else text.replace("|", r"\|")


def _is_sympy_renderable(value: Any) -> bool:
    """Return whether ``value`` should use SymPy's LaTeX printer."""
    if callable(getattr(value, "_latex", None)):
        return True
    return isinstance(
        value,
        (
            sympy.Basic,
            sympy.matrices.matrixbase.MatrixBase,
            sympy.matrices.expressions.matexpr.MatrixExpr,
        ),
    )


# ==============================================================================
# Table Rendering Utilities
# ==============================================================================

def _table(
    data: Any,
    *,
    header: bool | Sequence[Any] = True,
    index: bool = False,
    max_rows: int = 20,
) -> _MarkdownFragment:
    """Return a markdown table fragment for common row-shaped objects."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise TypeError("table max_rows must be a nonnegative integer.")
    if max_rows < 0:
        raise ValueError("table max_rows must be a nonnegative integer.")

    # Standardize data structures from varying data input layouts into rows/headers.
    headers, rows, stream_truncated = _normalize_table(
        data,
        header=header,
        index=index,
        max_rows=max_rows,
    )
    rows = _limit_table_rows(
        rows,
        headers=headers,
        max_rows=max_rows,
        stream_truncated=stream_truncated,
    )
    rendered_rows = [
        [
            (
                cell.markdown
                if isinstance(cell, _MarkdownFragment)
                else (
                    f"${sympy.latex(cell)}$"
                    if _is_sympy_renderable(cell)
                    else str(cell)
                )
            )
            .replace("\n", "<br>")
            .replace("|", r"\|")
            for cell in row
        ]
        for row in rows
    ]
    if not headers:
        return _MarkdownFragment(_render_table_no_header(rendered_rows))

    rendered_headers = [
        (
            cell.markdown
            if isinstance(cell, _MarkdownFragment)
            else (
                f"${sympy.latex(cell)}$"
                if _is_sympy_renderable(cell)
                else str(cell)
            )
        )
        .replace("\n", "<br>")
        .replace("|", r"\|")
        for cell in headers
    ]

    # Construct the Markdown representation of the tabular rows and delimiters.
    lines = [
        "| " + " | ".join(rendered_headers) + " |",
        "| " + " | ".join("---" for _ in rendered_headers) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")

    return _MarkdownFragment("\n".join(lines))


# Expose the table utility on the public callable before wrapping it into a
# PipeOp so attribute forwarding preserves ``md.table(...)``.
md.table = _table  # type: ignore[attr-defined]


def _render_table_no_header(rows: list[list[str]]) -> str:
    """Return a Markdown-compatible HTML table with no visible header."""
    if not rows:
        raise ValueError("headerless table data must contain at least one row.")

    lines = ["<table>", "<tbody>"]
    for row in rows:
        cells = "".join(f"<td>{html_escape(cell)}</td>" for cell in row)
        lines.append(f"<tr>{cells}</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def _normalize_table(
    data: Any,
    *,
    header: bool | Sequence[Any],
    index: bool,
    max_rows: int,
) -> tuple[list[Any], list[list[Any]], bool]:
    """Return headers and rows for a common tabular input."""
    pandas_table = _normalize_pandas_table(data, header=header, index=index)
    if pandas_table is not None:
        headers, rows = pandas_table
        return headers, rows, False

    if isinstance(data, sympy.matrices.matrixbase.MatrixBase):
        headers, rows = _normalize_sequence_table(data.tolist(), header=header)
        return headers, rows, False

    # Attempt standard row materialization for array-like objects offering an explicitly defined .tolist mapping.
    if hasattr(data, "tolist") and not isinstance(data, (str, bytes, bytearray)):
        converted = data.tolist()
        if converted is not data:
            if isinstance(converted, Mapping):
                headers, rows = _normalize_mapping_table(converted, header=header)
                return headers, rows, False
            if _is_nonstring_sequence(converted):
                headers, rows = _normalize_sequence_table(converted, header=header)
                return headers, rows, False

    if isinstance(data, Mapping):
        headers, rows = _normalize_mapping_table(data, header=header)
        return headers, rows, False

    if _is_nonstring_sequence(data):
        headers, rows = _normalize_sequence_table(data, header=header)
        return headers, rows, False

    # Materialize streaming layouts and mapping collections into an iterable collection before running normalizations.
    if _is_nonstring_iterable(data):
        rows_source, stream_truncated = _materialize_iterable_rows(
            data,
            max_rows=max_rows,
        )
        headers, rows = _normalize_sequence_table(rows_source, header=header)
        return headers, rows, stream_truncated

    raise ValueError(
        "table data must be a mapping, a row iterable, or an array-like object."
    )


def _materialize_iterable_rows(
    data: Iterable[Any],
    *,
    max_rows: int,
) -> tuple[list[Any], bool]:
    """Return bounded rows from a non-sequence iterable."""
    if max_rows == 0 or isinstance(data, Sized):
        return list(data), False

    # Extract a restricted prefix length from continuous streams to verify whether the input sequence contains additional elements.
    rows: list[Any] = []
    iterator = iter(data)
    source_limit = max_rows + 2
    for _ in range(source_limit):
        try:
            rows.append(next(iterator))
        except StopIteration:
            return rows, False

    try:
        next(iterator)
    except StopIteration:
        return rows, False
    return rows, True


def _limit_table_rows(
    rows: list[list[Any]],
    *,
    headers: Sequence[Any],
    max_rows: int,
    stream_truncated: bool,
) -> list[list[Any]]:
    """Return rows with an omission marker when output exceeds the row limit."""
    if max_rows == 0 or (len(rows) <= max_rows and not stream_truncated):
        return rows

    width = len(headers) if headers else (max(len(row) for row in rows) if rows else 1)
    if stream_truncated:
        return [*rows[:max_rows], _omitted_table_row(width, None)]

    omitted_count = len(rows) - max_rows
    if max_rows == 1:
        return [*rows[:1], _omitted_table_row(width, omitted_count)]

    # Distribute bounded rows symmetrically around the truncation spacer entry.
    head_count = (max_rows + 1) // 2
    tail_count = max_rows - head_count
    return [
        *rows[:head_count],
        _omitted_table_row(width, omitted_count),
        *rows[-tail_count:],
    ]


def _omitted_table_row(width: int, omitted_count: int | None) -> list[str]:
    """Return a table row that announces omitted data rows."""
    if omitted_count is None:
        label = "... more rows omitted ..."
    else:
        unit = "row" if omitted_count == 1 else "rows"
        label = f"... {omitted_count} {unit} omitted ..."
    return [label, *([""] * (max(width, 1) - 1))]


def _normalize_pandas_table(
    data: Any,
    *,
    header: bool | Sequence[Any],
    index: bool,
) -> tuple[list[Any], list[list[Any]]] | None:
    """Return a table for pandas-like objects without importing pandas."""
    if hasattr(data, "columns") and hasattr(data, "itertuples"):
        columns = list(data.columns)
        rows = [list(row) for row in data.itertuples(index=index, name=None)]
        if index:
            index_name = getattr(getattr(data, "index", None), "name", None) or "index"
            columns = [index_name, *columns]
        return _apply_headers(columns, rows, header)

    if hasattr(data, "items") and hasattr(data, "name") and not isinstance(data, Mapping):
        if index:
            headers = ["index", data.name or "value"]
            rows = [[key, value] for key, value in data.items()]
        else:
            headers = [data.name or "value"]
            rows = [[value] for _, value in data.items()]
        return _apply_headers(headers, rows, header)

    return None


def _normalize_mapping_table(
    data: Mapping[Any, Any], *, header: bool | Sequence[Any]
) -> tuple[list[Any], list[list[Any]]]:
    """Return a table from a mapping as columns or key-value rows."""
    keys = list(data.keys())
    values = list(data.values())

    # Build multi-column tables if mapping elements correspond to completely symmetrical sequence lists.
    if values and all(_is_nonstring_sequence(value) for value in values):
        lengths = [len(value) for value in values]
        if len(set(lengths)) == 1:
            rows = [list(row) for row in zip(*values)]
            return _apply_headers(keys, rows, header)

    rows = [[key, value] for key, value in data.items()]
    default_headers = ["key", "value"]
    return _apply_headers(default_headers, rows, header)


def _normalize_sequence_table(
    data: Sequence[Any], *, header: bool | Sequence[Any]
) -> tuple[list[Any], list[list[Any]]]:
    """Return a table from a sequence of rows or mappings."""
    rows_source = list(data)
    if not rows_source:
        if isinstance(header, Sequence) and not isinstance(header, (str, bytes, bytearray)):
            return list(header), []
        raise ValueError("table data must contain at least one row or explicit headers.")

    # Harmonize list of dictionary keys into unified headers across disparate row inputs.
    if all(isinstance(row, Mapping) for row in rows_source):
        columns: list[Any] = []
        for row in rows_source:
            for key in row:
                if key not in columns:
                    columns.append(key)
        rows = [[row.get(column, "") for column in columns] for row in rows_source]
        return _apply_headers(columns, rows, header)

    if not any(_is_nonstring_sequence(row) for row in rows_source):
        rows = [[item] for item in rows_source]
        return _apply_headers(["value"], rows, header)

    rows = [list(row) if _is_nonstring_sequence(row) else [row] for row in rows_source]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("table rows must all have the same number of columns.")

    if header is True:
        first, *body = rows
        return first, body

    return _apply_headers(
        [str(position) for position in range(len(rows[0]))], rows, header
    )


def _apply_headers(
    default_headers: Sequence[Any],
    rows: list[list[Any]],
    header: bool | Sequence[Any],
) -> tuple[list[Any], list[list[Any]]]:
    """Return headers according to the public ``header`` option."""
    if isinstance(header, bool):
        if not header:
            return [], rows
        headers = list(default_headers)
    elif isinstance(header, Sequence) and not isinstance(header, (str, bytes, bytearray)):
        headers = list(header)
    else:
        raise TypeError("table header must be a bool or a sequence of labels.")

    if rows and any(len(row) != len(headers) for row in rows):
        raise ValueError("table header count must match the row width.")
    return headers, rows


def _is_nonstring_sequence(value: Any) -> bool:
    """Return whether ``value`` behaves like a user data sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_nonstring_iterable(value: Any) -> bool:
    """Return whether ``value`` can be materialized as user table rows."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))


# ==============================================================================
# Markdown State Tracking
# ==============================================================================

class _MarkdownState:
    """Track enough markdown state to distinguish prose from math regions."""

    _MATH_ENVIRONMENTS = frozenset(
        {
            "align",
            "align*",
            "alignat",
            "alignat*",
            "equation",
            "equation*",
            "flalign",
            "flalign*",
            "gather",
            "gather*",
            "multline",
            "multline*",
            "split",
        }
    )

    def __init__(self) -> None:
        self.in_math = False
        self._delimiter: str | None = None
        self._escape = False
        self._tick_run = 0
        self._in_code = False

    def feed(self, text: str) -> None:
        """Update the state with literal output text."""
        cursor = 0
        while cursor < len(text):
            char = text[cursor]

            if self._escape:
                self._escape = False
                cursor += 1
                continue

            # Process inline code blocks bounded by variable backtick runs.
            if char == "`":
                run_end = cursor
                while run_end < len(text) and text[run_end] == "`":
                    run_end += 1
                run_length = run_end - cursor
                if not self._in_code:
                    self._in_code = True
                    self._tick_run = run_length
                elif run_length == self._tick_run:
                    self._in_code = False
                    self._tick_run = 0
                cursor = run_end
                continue

            if self._in_code:
                cursor += 1
                continue

            # Toggle mathematical typesetting parsing modes depending on structural delimiters.
            if self._delimiter is None:
                delimiter = self._opening_delimiter(text, cursor)
                if delimiter is not None:
                    opening, closing = delimiter
                    self.in_math = True
                    self._delimiter = closing
                    cursor += len(opening)
                    continue
            elif text.startswith(self._delimiter, cursor):
                closing = self._delimiter
                self.in_math = False
                self._delimiter = None
                cursor += len(closing)
                continue

            if char == "\\":
                self._escape = True
                cursor += 1
                continue

            cursor += 1

    @staticmethod
    def _opening_delimiter(text: str, cursor: int) -> tuple[str, str] | None:
        """Return the opening and closing math delimiters at ``cursor``."""
        for opening, closing in (
            ("$$", "$$"),
            (r"\[", r"\]"),
            (r"\(", r"\)"),
            ("$", "$"),
        ):
            if text.startswith(opening, cursor):
                return opening, closing
        environment = _MarkdownState._opening_math_environment(text, cursor)
        if environment is not None:
            opening, name = environment
            return opening, rf"\end{{{name}}}"
        return None

    @staticmethod
    def _opening_math_environment(text: str, cursor: int) -> tuple[str, str] | None:
        """Return a LaTeX math environment opening and name at ``cursor``."""
        prefix = r"\begin{"
        if not text.startswith(prefix, cursor):
            return None

        name_start = cursor + len(prefix)
        name_end = text.find("}", name_start)
        if name_end == -1:
            return None

        name = text[name_start:name_end]
        if name not in _MarkdownState._MATH_ENVIRONMENTS:
            return None

        return text[cursor : name_end + 1], name


# ==============================================================================
# IPython Display and Subclass Types
# ==============================================================================

def _display_markdown_text(rendered: _MarkdownText) -> None:
    """Display rendered markdown text through IPython when it is available."""

    try:
        from .display_context import display_markdown
    except ImportError:
        display_markdown = None
    if display_markdown is not None and display_markdown(str(rendered)):
        return None

    if _display_marimo_markdown(str(rendered)):
        return None

    try:
        from IPython.display import display as ipython_display
    except ImportError:
        print(rendered)
        return None

    ipython_display(rendered)
    return None


def _display_marimo_markdown(markdown: str) -> bool:
    """Display Markdown through Marimo's imperative output API when active."""

    try:
        from marimo._runtime.context import get_context
    except Exception:
        return False

    try:
        get_context()
    except Exception:
        return False

    try:
        import marimo as mo
    except Exception:
        return False

    output = getattr(mo, "output", None)
    append = getattr(output, "append", None)
    md = getattr(mo, "md", None)
    if not callable(append) or not callable(md):
        return False

    append(md(markdown))
    return True


class _MarkdownText(str):
    """String subclass with notebook markdown display hooks."""

    def _repr_markdown_(self) -> str:
        """Return the markdown payload used by IPython frontends."""
        return _markdown_for_active_frontend(str(self))

    def _repr_mimebundle_(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> dict[str, str]:
        """Return a minimal display bundle for IPython ``display``."""
        markdown = _markdown_for_active_frontend(str(self))
        data = {
            "text/plain": str(self),
            "text/markdown": markdown,
        }
        if include is not None:
            allowed = set(include)
            data = {key: value for key, value in data.items() if key in allowed}
        if exclude is not None:
            denied = set(exclude)
            data = {key: value for key, value in data.items() if key not in denied}
        return data


def _markdown_for_active_frontend(markdown: str) -> str:
    """Return Markdown adjusted for the active notebook frontend."""
    if _jupyterlab_myst_is_available():
        return _rewrite_display_math_as_myst(markdown)
    return markdown


def _jupyterlab_myst_is_available() -> bool:
    """Return whether the JupyterLab MyST extension is importable."""
    if "jupyterlab_myst" in sys.modules:
        return True

    try:
        return importlib.util.find_spec("jupyterlab_myst") is not None
    except (ImportError, ValueError):
        return False


def _rewrite_display_math_as_myst(markdown: str) -> str:
    """Return Markdown with display math rewritten as MyST math directives."""
    pieces: list[str] = []
    cursor = 0
    inline_code_run = 0
    fence_run = 0
    at_line_start = True

    while cursor < len(markdown):
        char = markdown[cursor]

        # Keep fenced and inline code intact so examples showing dollar signs
        # remain literal Markdown.
        if char == "`":
            run_end = cursor
            while run_end < len(markdown) and markdown[run_end] == "`":
                run_end += 1
            run = run_end - cursor
            if at_line_start and run >= 3:
                fence_run = 0 if fence_run == run else run
            elif fence_run == 0:
                inline_code_run = 0 if inline_code_run == run else run
            pieces.append(markdown[cursor:run_end])
            at_line_start = False
            cursor = run_end
            continue

        if fence_run == 0 and inline_code_run == 0:
            delimiter = None
            if markdown.startswith("$$", cursor):
                delimiter = "$$"
            elif markdown.startswith(r"\[", cursor):
                delimiter = r"\]"

            if delimiter is not None:
                opening = "$$" if delimiter == "$$" else r"\["
                end = markdown.find(delimiter, cursor + len(opening))
                if end != -1:
                    body = markdown[cursor + len(opening):end]
                    pieces.append(_myst_math_directive(body))
                    cursor = end + len(delimiter)
                    at_line_start = False
                    continue

        pieces.append(char)
        cursor += 1
        at_line_start = char == "\n"

    return "".join(pieces)


def _myst_math_directive(body: str) -> str:
    """Return one MyST math directive with equation numbering disabled."""
    math_body = body.strip("\n")
    return f"```{{math}}\n:enumerated: false\n{math_body}\n```"


@dataclass(frozen=True)
class _MarkdownFragment:
    """Internal marker for helpers that already return markdown structure."""

    markdown: str
    latex: str | None = None

    def __str__(self) -> str:
        """Return the markdown representation."""
        return self.markdown


md = pipeop(
    md,
    name="md",
)
