"""Provide small shared helpers used across notebook-facing modules."""

from __future__ import annotations

import html
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, TypeVar

__all__ = ["jupyter_document_link", "time_it", "validate_option"]

T = TypeVar("T", bound=StrEnum)


def validate_option(value: str, enum: type[T]) -> T:
    """Validate a string option against a ``StrEnum``."""

    try:
        return enum(value)
    except ValueError as e:
        options = ", ".join(repr(x.value) for x in enum)
        raise ValueError(
            f"Invalid {enum.__name__}: {value!r}. "
            f"Expected one of: {options}"
        ) from e


@contextmanager
def time_it(label: str = "Elapsed") -> Iterator[None]:
    """Print elapsed wall-clock time for a block of code.

    Parameters
    ----------
    label : str, default="Elapsed"
        Text printed before the measured duration.

    Yields
    ------
    None
        The managed block does not receive a timing object.

    Notes
    -----
    Timing uses ``time.perf_counter()`` and always prints when the block exits,
    including when the block raises an exception.

    Examples
    --------
    Basic usage:

    >>> with time_it("quick calculation"):  # doctest: +ELLIPSIS
    ...     _ = sum(range(10))
    quick calculation: ... seconds
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.6f} seconds")


time_it._mt_help = {
    "path": PurePosixPath("library/time_it"),
    "anchor": None,
    "label": "time_it",
}


def _html_attrs(attrs: Mapping[str, str]) -> str:
    """Return escaped HTML attributes in insertion order."""

    return " ".join(
        f'{name}="{html.escape(value, quote=True)}"' for name, value in attrs.items()
    )


def jupyter_document_link(
    path: str,
    text: str | None = None,
    mode: str | None = None,
    activate: bool = True,
    factory: str | None = None,
    debug: bool = False,
) -> Any:
    """Create an HTML link that opens a document in Jupyter frontends.

    In JupyterLab, normal clicks are routed through the command-linker and open
    the document with ``docmanager:open``. In classic Notebook, or when
    ``mode="browser_tab"`` is requested, the click handler opens the matching
    frontend URL in a browser tab.

    Parameters
    ----------
    path : str
        Path to the document within the Jupyter server's content tree. Leading
        slashes are stripped before Jupyter URLs and command arguments are
        built.
    text : str | None, default=None
        Link text. Defaults to the basename of ``path``.
    mode : str | None, default=None
        Optional JupyterLab document open mode. Use ``"browser_tab"`` to open
        the document in a separate browser tab instead of the JupyterLab main
        work area.
    activate : bool, default=True
        Whether JupyterLab should activate the opened document.
    factory : str | None, default=None
        Optional JupyterLab document factory.
    debug : bool, default=False
        Whether to emit browser console logging from the click handler.

    Returns
    -------
    IPython.display.HTML
        Renderable HTML containing a clickable document link.

    Examples
    --------
    Basic usage:

    >>> link = jupyter_document_link("/notebooks/example.ipynb", text="Example")
    >>> "Example" in link.data
    True
    """

    from IPython.display import HTML

    if text is None:
        text = path.split("/")[-1]

    # Jupyter document paths are content-root relative even when callers pass a
    # leading slash for readability.
    clean = path.lstrip("/")
    is_browser_tab = mode == "browser_tab"

    # The command-linker attributes handle normal JupyterLab clicks, while the
    # inline handler provides classic Notebook and browser-tab fallbacks.
    attrs = {
        "href": "",
        "onclick": _jupyter_document_onclick(clean, mode, debug),
        "style": (
            "color: var(--jp-content-link-color); "
            "text-decoration: underline; cursor: pointer;"
        ),
    }

    if not is_browser_tab:
        attrs["data-commandlinker-command"] = "docmanager:open"
        attrs["data-commandlinker-args"] = json.dumps(
            _jupyter_document_command_args(
                clean,
                mode=mode,
                activate=activate,
                factory=factory,
            )
        )

    return HTML(f"<a {_html_attrs(attrs)}>{html.escape(text)}</a>")


def _jupyter_document_command_args(
    path: str,
    *,
    mode: str | None,
    activate: bool,
    factory: str | None,
) -> dict[str, object]:
    """Return JupyterLab ``docmanager:open`` command arguments."""

    args: dict[str, object] = {"path": path}

    # JupyterLab omits default options cleanly but accepts an options object
    # when callers request split modes or inactive opening.
    if mode is not None or activate is not True:
        options: dict[str, object] = {}
        if mode is not None:
            options["mode"] = mode
        if activate is not True:
            options["activate"] = activate
        args["options"] = options

    if factory:
        args["factory"] = factory

    return args


def _jupyter_document_onclick(path: str, mode: str | None, debug: bool) -> str:
    """Return inline fallback JavaScript for Jupyter document links."""

    debug_js = ""
    if debug:
        debug_js = (
            "console.log('jupyter document link click');"
            f"console.log('path:', {json.dumps(path)});"
            f"console.log('mode:', {json.dumps(mode)});"
            "console.log('pathname:', window.location.pathname);"
        )

    # Keep the fallback self-contained because classic Notebook does not load
    # JupyterLab's command-linker extension.
    onclick_js = f"""
{debug_js}
var pathname = window.location.pathname || '';
var isLab = pathname.indexOf('/lab') !== -1;
var clean = {json.dumps(path)};
var labUrl = '/lab/tree/' + clean;
var classicUrl = '/tree/' + clean;
var mode = {json.dumps(mode)};

if (isLab) {{
    if (mode === 'browser_tab') {{
        window.open(labUrl, '_blank');
        return false;
    }}
    return true;
}}

window.open(classicUrl, '_blank');
return false;
"""

    return onclick_js.strip().replace("\n", " ")
