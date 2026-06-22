"""Render notebook-native runtime help and documentation links.

The public ``help(...)`` function reports an object's real runtime type and, if
declared, a link to the static notebook topic that explains how to use it.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ..util import jupyter_document_link
from . import config

__all__ = ["Help"]


def _documentation_notebook_path(path: PurePosixPath | str | None) -> PurePosixPath:
    """Return the extension-bearing notebook path for a documentation topic."""

    # Validate the declared extension-free path before building the URL.
    if not isinstance(path, PurePosixPath):
        raise TypeError(f"Documentation path must be PurePosixPath: {path!r}")
    normalized = path
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or str(normalized) in {"", "."}
    ):
        raise ValueError(f"Invalid documentation path: {path!r}")
    if normalized.suffix == ".ipynb":
        raise ValueError(f"Documentation path must be extension-free: {path!r}")

    return normalized.parent / f"{normalized.name}.ipynb"

def _make_raw_documentation_link(
    doc: Mapping[str, PurePosixPath | str | None],
) -> str:
    """Return a Markdown ``doc:`` link for a declared documentation topic."""

    notebook_path = _documentation_notebook_path(doc["path"]).as_posix()
    anchor = doc["anchor"]
    fragment = ""
    if anchor is not None:
        fragment = f"#{anchor}"

    return f"[{doc['label']}](doc:/{notebook_path}{fragment})"


def _make_documentation_link(
    doc: Mapping[str, PurePosixPath | str | None],
) -> Any:
    """Return an HTML link for a declared documentation topic."""

    content_path = (
        config.DOCUMENTATION_ROOT / _documentation_notebook_path(doc["path"])
    ).as_posix()

    # Package docs must already be present below the configured documentation
    # root. A future locator can make this less manual once it can verify
    # Jupyter content-tree reachability safely.

    return jupyter_document_link(content_path, text=str(doc["label"]))


def _resolve_documentation_topic(
    obj: Any,
) -> Mapping[str, PurePosixPath | str | None] | None:
    """Return declared documentation metadata for an object, if present."""

    # Respect the documented instance-over-type metadata precedence.
    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None:
        if "_mt_help" in instance_dict:
            return instance_dict["_mt_help"]

    obj_type = type(obj)
    declared_type = getattr(obj_type, "_mt_help", None)
    if declared_type is not None:
        return declared_type

    return getattr(obj, "_mt_help", None)


def _make_rendered_help_html(
    runtime_line: str,
    documentation_html: str,
) -> str:
    """Return the rendered help block as HTML."""

    return (
        f"{html.escape(runtime_line, quote=False)}"
        "<br>"
        f"Documentation: {documentation_html}"
    )


def Help(obj: Any, *, raw: bool = False) -> None:
    """Show runtime type and the resolved documentation link.

    Documentation resolution is declaration-based: an instance-level
    ``_mt_help`` entry wins, otherwise the object's type is checked. No registry
    fallback or heuristic topic lookup is performed.

    Parameters
    ----------
    obj : Any
        Object to describe.
    raw : bool, default=False
        Print raw text instead of rendering an HTML help block.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when a declared documentation path is invalid.
    TypeError
        Raised when a declared documentation path is not a POSIX path.

    Notes
    -----
    The printed runtime type and the documentation topic may differ when a
    broad documentation page intentionally covers several runtime types.

    Examples
    --------
    Basic usage:

    >>> import sympy
    >>> x = sympy.Symbol("x")
    >>> help(x, raw=True)
    x: Symbol
    Documentation: [Symbol](doc:/library/Symbol.ipynb)

    See Also
    --------
    activate : <function> Populate a notebook namespace with documented symbolic objects.
    reset : <function> Refresh notebook bindings without changing patch state.
    """

    doc = _resolve_documentation_topic(obj)
    runtime_line = f"{obj}: {type(obj).__name__}"

    if raw:
        # Raw mode keeps output useful in terminals, tests, and notebooks that
        # do not render HTML.
        if doc is None:
            documentation_line = "Documentation: No documentation available"
        else:
            link = _make_raw_documentation_link(doc)
            documentation_line = f"Documentation: {link}"

        print(runtime_line)
        print(documentation_line)
        return None

    from IPython.display import HTML, display

    # Rendered mode mirrors the same two logical lines with an HTML document
    # link when a declared topic exists.
    if doc is None:
        documentation_html = html.escape("No documentation available", quote=False)
    else:
        documentation_html = _make_documentation_link(doc).data

    display(HTML(_make_rendered_help_html(runtime_line, documentation_html)))

    return None


Help._mt_help = {
    "path": PurePosixPath("library/help"),
    "anchor": None,
    "label": "Help",
}
