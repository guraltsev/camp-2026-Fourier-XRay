"""Activate the math toolkit notebook namespace.

Call ``activate(...)`` to populate the target namespace with SymPy plus toolkit
convenience bindings. 

Call ``reset(...)`` to reset to the default namespace if your code messed up the bindings

Activation also installs patches to existing global objects. 
"""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Iterable, MutableMapping
from pathlib import Path, PurePosixPath

__all__ = ["activate", "reset"]
_PATCHES_INSTALLED = False


def reset(
    identifiers: str | Iterable[str] | MutableMapping[str, object] | None = None,
    namespace: MutableMapping[str, object] | None = None,
) -> None:
    """Reset the managed notebook bindings in a namespace.

    When ``namespace`` is omitted, the target mapping is resolved the same way
    as :func:`activate`: prefer IPython's ``user_ns`` when available and
    otherwise fall back to the caller's global namespace. When identifiers are
    provided, only those managed bindings are restored.

    Parameters
    ----------
    identifiers : str | Iterable[str] | MutableMapping[str, object] | None, default=None
        Space-separated binding names, iterable of binding names, or an
        existing namespace passed positionally for backward compatibility. When
        omitted, every managed notebook binding is refreshed.
    namespace : MutableMapping[str, object] | None, default=None
        Namespace to update in place. When omitted, the active notebook or the
        caller's module globals are used.

    Returns
    -------
    None

    Examples
    --------
    Basic usage:

    >>> import math_toolkit.notebook as notebook
    >>> x = 1
    >>> x
    1
    >>> notebook.reset()
    >>> x
    x
    >>> x = 1
    >>> F = 2
    >>> notebook.reset("x")
    >>> x
    x
    >>> F
    2

    See Also
    --------
    activate : <function> Populate the managed notebook bindings.
    """

    # Preserve the original ``reset(namespace)`` calling convention while
    # allowing the first argument to name a partial reset request.
    if isinstance(identifiers, MutableMapping):
        if namespace is not None:
            raise TypeError(
                "reset() got both a positional namespace and a namespace keyword."
            )
        target = _resolve_target_namespace(identifiers)
        requested_names = None
    else:
        target = _resolve_target_namespace(namespace)
        requested_names = _normalize_reset_identifiers(identifiers)

    # Install the notebook-specific SymPy behavior before we inject names that
    # rely on indexing, free-symbol, and rendering patches.
    _install_notebook_patches()

    # Mirror the curated root notebook exports onto ``math_toolkit.notebook``
    # only when notebook behavior is actually activated or reset.
    _publish_notebook_module_exports()

    # Rebuild either the whole managed notebook surface or only the requested
    # names, leaving unrelated caller-owned bindings untouched.
    from . import _build_notebook_bindings, inject_notebook_bindings
    if requested_names is None:
        inject_notebook_bindings(target)
    else:
        default_bindings = _build_notebook_bindings()
        missing_names = [
            name for name in requested_names if name not in default_bindings
        ]
        if missing_names:
            joined_names = ", ".join(missing_names)
            raise KeyError(f"Unknown managed notebook binding: {joined_names}")
        for name in requested_names:
            target[name] = default_bindings[name]

    # Initialize SymPy expression printing after the namespace is refreshed.
    setup_sympy_printing()
    return None


def _normalize_reset_identifiers(
    identifiers: str | Iterable[str] | None,
) -> list[str] | None:
    """Return the requested partial reset names, or ``None`` for a full reset."""

    if identifiers is None:
        return None

    if isinstance(identifiers, str):
        return identifiers.split()

    # Convert public iterable inputs once so validation happens before any
    # namespace mutation and so repeated names keep the caller's order.
    requested_names: list[str] = []
    for identifier in identifiers:
        if not isinstance(identifier, str):
            raise TypeError("reset() identifiers must be strings.")
        requested_names.append(identifier)
    return requested_names


def activate(namespace: MutableMapping[str, object] | None = None) -> None:
    """Activate the notebook namespace.

    Parameters
    ----------
    namespace : MutableMapping[str, object] | None, default=None
        Namespace to populate. When omitted, activation targets IPython's
        ``user_ns`` when available and otherwise falls back to the caller's
        global namespace.

    Returns
    -------
    None

    Notes
    -----
    Activation installs the notebook-specific SymPy patch set on first use and
    then refreshes the managed notebook bindings in the target namespace.

    Examples
    --------
    Basic usage:

    >>> import math_toolkit.notebook as notebook
    >>> scope = {}
    >>> notebook.activate(namespace=scope)
    >>> str(scope["x"][scope["i"]])
    'x[i]'

    See Also
    --------
    reset : <function> Rebuild the managed notebook bindings.
    math_toolkit : <package> Package profile that selects enabled extensions.
    """

    # Activation resolves the target namespace and then delegates the actual
    # refresh work to ``reset``.

    #TODO: add guard to only activate ONCE
    target = _resolve_target_namespace(namespace)
    reset(target)
    _run_activation_init_files(target)
    return None

#TODO: FUSE activate and reset documentation
activate._mt_help = {
    "path": PurePosixPath("library/activate"),
    "anchor": None,
    "label": "activate",
}

reset._mt_help = {
    "path": PurePosixPath("library/reset"),
    "anchor": None,
    "label": "reset",
}


def _resolve_target_namespace(
    namespace: MutableMapping[str, object] | None,
) -> MutableMapping[str, object]:
    """Return the namespace that activation or reset should mutate."""

    if namespace is not None:
        return namespace

    # Prefer the live IPython notebook namespace when one is active.
    ipython = sys.modules.get("IPython")
    if ipython is not None:
        get_ipython = getattr(ipython, "get_ipython", None)
        shell = get_ipython() if get_ipython is not None else None
        user_ns = getattr(shell, "user_ns", None)
        if isinstance(user_ns, MutableMapping):
            return user_ns

    # Fall back to the caller's globals outside IPython.
    frame = inspect.currentframe()
    caller_frame = None
    try:
        if frame is not None and frame.f_back is not None:
            caller_frame = frame.f_back.f_back
        if caller_frame is not None:
            return caller_frame.f_globals
        return globals()
    finally:
        del caller_frame
        del frame


def _install_notebook_patches() -> None:
    """Install the monkeypatches required by notebook activation once."""

    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return None

    # Apply the notebook-facing SymPy extensions exactly once per process so
    # repeated activation refreshes stay cheap and stable.
    from ..sympy_extensions import atom_latex_representation
    from ..sympy_extensions import free_symbols
    from ..sympy_extensions import function_indexing
    from ..sympy_extensions import symbol_indexing
    from ..sympy_extensions import symbol_name_validation

    atom_latex_representation.patch()
    free_symbols.patch()
    symbol_indexing.patch()
    function_indexing.patch()
    symbol_name_validation.patch()

    _PATCHES_INSTALLED = True
    return None


def _run_activation_init_files(namespace: MutableMapping[str, object]) -> None:
    """Run document-local activation files when the active path is known."""

    # Resolve the active document before touching the filesystem. Some notebook
    # frontends do not publish a notebook path to the kernel, so activation must
    # remain useful when no document-local init location can be inferred.
    active_path = _resolve_active_document_path(namespace)
    init_directory = active_path.parent if active_path is not None else Path.cwd()

    # Run the shared directory initializer before the document-specific one so
    # a notebook or script can override common setup deliberately.
    init_paths = [init_directory / ".init.py"]
    if active_path is not None:
        init_paths.append(init_directory / f".{active_path.stem}.init.py")
    for init_path in init_paths:
        if init_path.is_file():
            _run_init_file(init_path, namespace)
    return None


def _resolve_active_document_path(
    namespace: MutableMapping[str, object],
) -> Path | None:
    """Return the active script or notebook path if activation can infer it."""

    # Script execution publishes ``__file__`` in the user globals. Treat that as
    # the strongest signal because it is local to the caller's associated file.
    script_file = namespace.get("__file__")
    if isinstance(script_file, str) and script_file:
        return Path(script_file).expanduser().resolve()

    # Common notebook frontends may publish the notebook path in user globals.
    # Classic Jupyter does not guarantee this, so every candidate is optional.
    for notebook_key in (
        "__vsc_ipynb_file__",
        "__notebook_file__",
        "__notebook_path__",
    ):
        notebook_file = namespace.get(notebook_key)
        if isinstance(notebook_file, str) and notebook_file:
            return Path(notebook_file).expanduser().resolve()

    # Some kernels receive the session name through the environment. It is
    # useful only when it actually names a notebook or script file.
    session_name = os.environ.get("JPY_SESSION_NAME")
    if session_name:
        session_path = Path(session_name).expanduser()
        if session_path.suffix in {".ipynb", ".py"}:
            return session_path.resolve()
    return None


def _run_init_file(init_path: Path, namespace: MutableMapping[str, object]) -> None:
    """Execute one activation init file in the target namespace."""

    # Compile with the real filename for useful tracebacks, and preserve the
    # user's ``__file__`` binding so script-mode activation still points to the
    # associated script after setup finishes.
    previous_file = namespace.get("__file__")
    had_previous_file = "__file__" in namespace
    source = init_path.read_text(encoding="utf-8")
    code = compile(source, str(init_path), "exec")
    try:
        namespace["__file__"] = str(init_path)
        exec(code, namespace)
    finally:
        if had_previous_file:
            namespace["__file__"] = previous_file
        else:
            namespace.pop("__file__", None)
    return None


def _publish_notebook_module_exports() -> None:
    """Expose root notebook-facing exports on ``math_toolkit.notebook``."""

    from . import _iter_root_export_entries, _publish_root_exports

    _publish_root_exports(
        {
            binding_name: value
            for binding_name, value in _iter_root_export_entries()
        }
    )
    return None


def setup_sympy_printing() -> None:
    """Initialize SymPy display printing for notebook-style MathJax output."""

    import sympy
    sympy.init_printing(use_latex="mathjax", mul_symbol=r"\,")
    return None
