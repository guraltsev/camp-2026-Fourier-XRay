"""Build and activate the managed math toolkit notebook namespace."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, MutableMapping

import sympy

__all__ = []

NotebookEntry = str | tuple[str, object]
NotebookEntryFactory = Callable[[], Iterable[NotebookEntry]]


def _export(name: str, value: object) -> None:
    """Publish one public notebook attribute and record it in ``__all__``."""

    globals()[name] = value
    __all__.append(name)
    return None


def _add_checked_bindings(
    bindings: dict[str, object],
    additions: Mapping[str, object],
    *,
    allow_sympy_overrides: set[str] | None = None,
) -> None:
    """Add toolkit-owned bindings after checking for SymPy export conflicts."""

    allowed = set() if allow_sympy_overrides is None else allow_sympy_overrides
    overrides = sorted((set(additions) & set(sympy.__all__)) - allowed)
    if overrides:
        joined_names = ", ".join(overrides)
        raise RuntimeError(
            "Notebook convenience bindings must not override SymPy exports: "
            f"{joined_names}"
        )

    bindings.update(additions)


def _plain_display(*objects: object, **kwargs: object) -> None:
    """Display objects in non-IPython contexts with a simple text fallback."""

    for item in objects:
        print(item)
    return None


def _resolve_display() -> object:
    """Return the IPython display callable when it is available."""

    try:
        from IPython.display import display
    except ImportError:
        return _plain_display
    return display


def _normalize_notebook_entry(entry: object) -> tuple[str, object]:
    """Return one notebook export entry as ``(binding_name, value)``."""

    import math_toolkit

    if isinstance(entry, str):
        return entry, getattr(math_toolkit, entry)

    if (
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[0], str)
    ):
        binding_name, value = entry
        return binding_name, value

    raise RuntimeError(
        "math_toolkit.__notebook__ entries must be strings, "
        "two-item ``(name, value)`` tuples, or callables that return them."
    )


def _iter_root_export_entries() -> list[tuple[str, object]]:
    """Return normalized root notebook bindings from ``math_toolkit.__notebook__``."""

    import math_toolkit

    normalized_entries: list[tuple[str, object]] = []
    for entry in getattr(math_toolkit, "__notebook__", []):
        if callable(entry):
            produced_entries = entry()
            for produced_entry in produced_entries:
                normalized_entries.append(_normalize_notebook_entry(produced_entry))
            continue

        normalized_entries.append(_normalize_notebook_entry(entry))
    return normalized_entries


def _build_root_package_bindings() -> dict[str, object]:
    """Return notebook-safe bindings from the curated root package surface."""

    return {
        binding_name: value for binding_name, value in _iter_root_export_entries()
    }


def _build_notebook_bindings() -> dict[str, object]:
    """Build fresh notebook bindings for injection."""

    # Start from SymPy's public namespace, but keep generic exception names out
    # of the notebook scope so symbolic authoring names stay uncluttered.
    bindings = {name: getattr(sympy, name) for name in sympy.__all__}

    # Layer the curated package profile on top of SymPy so notebook activation
    # and root imports expose the same public toolkit objects.
    root_bindings = _build_root_package_bindings()
    _add_checked_bindings(
        bindings,
        root_bindings,
        allow_sympy_overrides=set(root_bindings),
    )

    # ``display`` is an environment helper rather than part of the root package
    # profile, so notebook activation provides it directly.
    _add_checked_bindings(bindings, {"display": _resolve_display()})
    return bindings


def inject_notebook_bindings(
    namespace: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Inject the default notebook bindings into an existing namespace.

    Parameters
    ----------
    namespace : MutableMapping[str, object]
        Mapping to update in place with the full notebook binding surface.

    Returns
    -------
    MutableMapping[str, object]
        The same mapping that was passed in after the notebook bindings have
        been written into it.

    Examples
    --------
    Basic usage:

    >>> from math_toolkit.notebook import inject_notebook_bindings
    >>> scope = {"custom": 123}
    >>> inject_notebook_bindings(scope)["custom"]
    123
    >>> scope["x"].name
    'x'
    """

    namespace.update(_build_notebook_bindings())
    return namespace


def _publish_root_exports(exports: Mapping[str, object]) -> None:
    """Expose root notebook-facing names on ``math_toolkit.notebook`` too."""

    for name, value in exports.items():
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)
    return None


_export("inject_notebook_bindings", inject_notebook_bindings)

# Import the activation module late so namespace construction helpers are
# available without a partially initialized activation import.
activation = importlib.import_module(".activation", __name__)
_export("activate", activation.activate)
_export("reset", activation.reset)
