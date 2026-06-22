SymPy patch modules and explicit symbolic syntax helpers live here.

The root `math_toolkit` package profile imports selected modules from this
package and calls their `patch()` functions. Import a submodule directly when
you need its helper classes or functions.

Enabled patch modules include:

- `free_symbols`: preserves specialized `set` subclass semantics when SymPy's
  default `Basic.free_symbols` aggregates child free-symbol sets.
- `atom_latex_representation`: renders plain atomic symbol names for notebook
  LaTeX.
- `symbol_indexing`: adds structural `x[i]` notation for symbols.
- `function_indexing`: adds structural `F[i](x)` notation for undefined
  functions.
- `symbol_name_validation`: keeps user-created atomic names plain so structure
  stays explicit.
