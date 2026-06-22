# `math_toolkit.num`

The `num` package owns the explicit numerical boundary for the toolkit.

Rewrite status:

- `Num` is intentionally not implemented while the numerical boundary is being
  redesigned.
- Dependent features may currently fail with explicit not-implemented errors.

Use `Num` after symbolic preparation:

```python
f = sin(x) >> Num
f(2)

g = (x[i] + x[i + 1]) >> Num
g([1, 2, 3, 4], 1)
```

Public concepts:

- `Num` compiles an expression into a numeric value or `NumFunction`.
- `NumFunction` binds positional arguments by fixed order and keyword
  arguments by public names.
- `Num.Scalar`, `Num.Index`, and `Num.Array(rank)` describe numeric roles.
- `Num.Integrate(expr, ranges, domain_func=...)` integrates symbolic or
  vectorized numeric functions with SciPy cubature.
- Matrix output follows `sample shape + mathematical output shape`.
- Held notation must be exposed explicitly with `Unhold` or `UnholdAll` before
  numericalization.
- Diagnostics are collected by default and can be shown or raised with
  `warnings="show"` or `warnings="error"`.

Public documentation will be rewritten once the new numerical boundary is
ready.
