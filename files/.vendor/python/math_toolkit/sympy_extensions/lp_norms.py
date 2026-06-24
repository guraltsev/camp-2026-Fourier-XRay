"""Provide symbolic weighted Lebesgue norm notation for SymPy expressions.

Use ``L1Norm``, ``L2Norm``, ``LinftyNorm``, or ``LpNorm(p)`` to build
unevaluated norm expressions with explicit bound-variable domains such as
``(x, a, b)``, repeated interval bindings, or ``((x, y), Omega)`` and
``((x,), S.Reals)`` grouped domain bindings. Finite norms rewrite explicitly to
``Integral`` objects; infinity norms rewrite to the formal
``EssentialSupremum`` object.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import sympy

__all__ = [
    "L1Norm",
    "L2Norm",
    "LinftyNorm",
    "LpNorm",
    "EssentialSupremum",
]


class NormOperand(sympy.Basic):
    """Store the normed expression as a visible SymPy tree child."""

    __slots__ = ()

    def __new__(cls, expr: Any) -> "NormOperand":
        """Create a norm operand component."""

        return sympy.Basic.__new__(cls, sympy.sympify(expr))

    @property
    def expr(self) -> sympy.Basic:
        """Return the expression being normed."""

        return self.args[0]


class BoundDomain(sympy.Basic):
    """Store norm binding variables and their normalized domain."""

    __slots__ = ()

    def __new__(cls, variables: Any, domain: Any) -> "BoundDomain":
        """Create a bound-domain component."""

        normalized_variables = _normalize_variables(variables)
        normalized_domain = _normalize_domain(domain, normalized_variables)
        return sympy.Basic.__new__(
            cls,
            sympy.Tuple(*normalized_variables),
            normalized_domain,
        )

    @property
    def variables(self) -> tuple[sympy.Symbol, ...]:
        """Return the bound variables in integration order."""

        return tuple(self.args[0])

    @property
    def domain(self) -> sympy.Basic:
        """Return the normalized domain."""

        return self.args[1]


class NormWeight(sympy.Basic):
    """Store the density used by a weighted norm."""

    __slots__ = ()

    def __new__(cls, weight: Any) -> "NormWeight":
        """Create a norm-weight component."""

        expr = sympy.sympify(weight)
        if expr.is_nonnegative is False:
            raise ValueError("weight must be a nonnegative density")
        return sympy.Basic.__new__(cls, expr)

    @property
    def expr(self) -> sympy.Basic:
        """Return the weight expression."""

        return self.args[0]


class LebesgueNormBase(sympy.Expr):
    """Represent an unevaluated weighted Lebesgue norm."""

    __slots__ = ()
    is_commutative = True
    _p: sympy.Basic | None = None

    def __new__(
        cls,
        expr: Any,
        *bindings: Any,
        weight: Any = sympy.S.One,
    ) -> sympy.Expr:
        """Create an unevaluated norm expression."""

        # SymPy reconstruction passes the component nodes from ``.args`` back
        # to the constructor. Keep that path separate from public construction.
        if _is_component_form(expr, bindings, weight):
            operand = expr
            binding = bindings[0]
            density = bindings[1]
        else:
            variables, normalized_domain = _normalize_public_binding(*bindings)
            operand = NormOperand(expr)
            binding = BoundDomain(variables, normalized_domain)
            density = NormWeight(weight)

        if operand.expr.is_zero:
            return sympy.S.Zero

        return sympy.Expr.__new__(cls, operand, binding, density)

    @property
    def operand(self) -> NormOperand:
        """Return the expression component."""

        return self.args[0]

    @property
    def binding(self) -> BoundDomain:
        """Return the bound-domain component."""

        return self.args[1]

    @property
    def density(self) -> NormWeight:
        """Return the weight component."""

        return self.args[2]

    @property
    def expr(self) -> sympy.Basic:
        """Return the expression being normed."""

        return self.operand.expr

    @property
    def variables(self) -> tuple[sympy.Symbol, ...]:
        """Return the bound variables."""

        return self.binding.variables

    @property
    def domain(self) -> sympy.Basic:
        """Return the normalized domain."""

        return self.binding.domain

    @property
    def weight(self) -> sympy.Basic:
        """Return the weight density."""

        return self.density.expr

    @property
    def p(self) -> sympy.Basic:
        """Return the Lebesgue exponent."""

        return self._p

    @property
    def bound_symbols(self) -> list[sympy.Symbol]:
        """Return the variables bound by the norm."""

        return list(self.variables)

    @property
    def free_symbols(self) -> set[sympy.Basic]:
        """Return free symbols with norm variables treated as bound."""

        bound = set(self.variables)
        symbols = set(self.expr.free_symbols)
        symbols.update(self.weight.free_symbols)
        symbols.difference_update(bound)
        symbols.update(_domain_free_symbols(self.domain))
        return symbols

    def _eval_subs(self, old: Any, new: Any) -> sympy.Expr | None:
        """Substitute through free parts while preserving bound variables."""

        old = sympy.sympify(old)
        new = sympy.sympify(new)
        if old in self.variables:
            return self

        # Substitute in all user-visible semantic fields. Reconstructing
        # through the public constructor revalidates changed domains and weights.
        return self.func(
            self.expr._subs(old, new),
            *_binding_as_public_args(self.variables, self.domain._subs(old, new)),
            weight=self.weight._subs(old, new),
        )

    def _xreplace(self, rule: dict[Any, Any]) -> tuple[sympy.Expr, bool]:
        """Replace exact nodes with bound-variable renaming support."""

        if self in rule:
            return rule[self], True
        if not rule:
            return self, False

        expr, expr_changed = self.expr._xreplace(rule)
        domain, domain_changed = self.domain._xreplace(rule)
        weight, weight_changed = self.weight._xreplace(rule)

        variables: list[sympy.Symbol] = []
        variables_changed = False
        for variable in self.variables:
            replacement = rule.get(variable, variable)
            if replacement != variable:
                variables_changed = True
            if not isinstance(replacement, sympy.Symbol):
                raise ValueError("bound variables must remain Symbols")
            variables.append(replacement)

        changed = expr_changed or domain_changed or weight_changed or variables_changed
        if not changed:
            return self, False
        return self.func(
            expr,
            *_binding_as_public_args(tuple(variables), domain),
            weight=weight,
        ), True

    def _eval_expand_norm(self, **hints: Any) -> sympy.Basic:
        """Expand the norm to its explicit formal definition."""

        if self.p == sympy.S.Infinity:
            return self.rewrite(EssentialSupremum)
        return self.rewrite(sympy.Integral)

    def doit(self, **hints: Any) -> sympy.Basic:
        """Evaluate the explicit rewrite when SymPy can evaluate it."""

        if self.p == sympy.S.Infinity:
            return self.rewrite(EssentialSupremum).doit(**hints)
        return self.rewrite(sympy.Integral).doit(**hints)

    def _latex(self, printer: Any) -> str:
        """Render compact norm notation in LaTeX."""

        expr_latex = printer._print(self.expr)
        vars_latex = ",".join(printer._print(v) for v in self.variables)
        p_latex = r"\infty" if self.p == sympy.S.Infinity else printer._print(self.p)
        return (
            r"\left\| %s \right\|_{L^{%s}(%s)}"
            % (expr_latex, p_latex, vars_latex)
        )

    def _sympystr(self, printer: Any) -> str:
        """Render a semantic plain-text representation."""

        binding = _format_binding(self.variables, self.domain, printer)
        rendered = (
            f"{self.func.__name__}("
            f"{printer._print(self.expr)}, {binding}"
        )
        if self.weight != sympy.S.One:
            rendered += f", weight={printer._print(self.weight)}"
        return rendered + ")"


class FiniteLpNorm(LebesgueNormBase):
    """Represent an unevaluated finite weighted Lebesgue norm."""

    __slots__ = ()

    def _eval_rewrite_as_Integral(
        self,
        operand: Any,
        binding: Any,
        density: Any,
        **hints: Any,
    ) -> sympy.Basic:
        """Rewrite a finite norm as an explicit integral expression."""

        limits = _domain_to_integral_limits(self.variables, self.domain)
        integrand = sympy.Abs(self.expr) ** self.p * self.weight
        integral = sympy.Integral(integrand, *limits)
        if self.p == sympy.S.One:
            return integral
        return integral ** (sympy.S.One / self.p)


class L1Norm(FiniteLpNorm):
    """Represent an unevaluated weighted ``L^1`` norm."""

    __slots__ = ()
    _p = sympy.S.One


class L2Norm(FiniteLpNorm):
    """Represent an unevaluated weighted ``L^2`` norm."""

    __slots__ = ()
    _p = sympy.S(2)


class LinftyNorm(LebesgueNormBase):
    """Represent an unevaluated weighted ``L^infinity`` norm."""

    __slots__ = ()
    _p = sympy.S.Infinity

    def _eval_rewrite_as_EssentialSupremum(
        self,
        operand: Any,
        binding: Any,
        density: Any,
        **hints: Any,
    ) -> "EssentialSupremum":
        """Rewrite an infinity norm as a formal essential supremum."""

        return EssentialSupremum(
            sympy.Abs(self.expr),
            *_binding_as_public_args(self.variables, self.domain),
            weight=self.weight,
        )


class EssentialSupremum(sympy.Expr):
    """Represent a formal weighted essential supremum over a domain."""

    __slots__ = ()
    is_commutative = True

    def __new__(
        cls,
        expr: Any,
        *bindings: Any,
        weight: Any = sympy.S.One,
    ) -> "EssentialSupremum":
        """Create an unevaluated essential supremum expression."""

        if _is_component_form(expr, bindings, weight):
            operand = expr
            binding = bindings[0]
            density = bindings[1]
        else:
            variables, normalized_domain = _normalize_public_binding(*bindings)
            operand = NormOperand(expr)
            binding = BoundDomain(variables, normalized_domain)
            density = NormWeight(weight)
        return sympy.Expr.__new__(cls, operand, binding, density)

    @property
    def operand(self) -> NormOperand:
        """Return the expression component."""

        return self.args[0]

    @property
    def binding(self) -> BoundDomain:
        """Return the bound-domain component."""

        return self.args[1]

    @property
    def density(self) -> NormWeight:
        """Return the weight component."""

        return self.args[2]

    @property
    def expr(self) -> sympy.Basic:
        """Return the expression inside the essential supremum."""

        return self.operand.expr

    @property
    def variables(self) -> tuple[sympy.Symbol, ...]:
        """Return the bound variables."""

        return self.binding.variables

    @property
    def domain(self) -> sympy.Basic:
        """Return the normalized domain."""

        return self.binding.domain

    @property
    def weight(self) -> sympy.Basic:
        """Return the weight density."""

        return self.density.expr

    @property
    def bound_symbols(self) -> list[sympy.Symbol]:
        """Return the variables bound by the essential supremum."""

        return list(self.variables)

    @property
    def free_symbols(self) -> set[sympy.Basic]:
        """Return free symbols with supremum variables treated as bound."""

        bound = set(self.variables)
        symbols = set(self.expr.free_symbols)
        symbols.update(self.weight.free_symbols)
        symbols.difference_update(bound)
        symbols.update(_domain_free_symbols(self.domain))
        return symbols

    def _sympystr(self, printer: Any) -> str:
        """Render a semantic plain-text representation."""

        binding = _format_binding(self.variables, self.domain, printer)
        rendered = (
            f"EssentialSupremum("
            f"{printer._print(self.expr)}, {binding}"
        )
        if self.weight != sympy.S.One:
            rendered += f", weight={printer._print(self.weight)}"
        return rendered + ")"

    def _latex(self, printer: Any) -> str:
        """Render compact essential-supremum notation in LaTeX."""

        variables = ",".join(printer._print(v) for v in self.variables)
        return (
            r"\operatorname*{ess\,sup}_{%s \in %s} %s"
            % (variables, printer._print(self.domain), printer._print(self.expr))
        )


class LpNormFactory:
    """Return norm classes for validated Lebesgue exponents."""

    _cache: dict[sympy.Basic, type[FiniteLpNorm]] = {}

    def __call__(self, p: Any) -> type[LebesgueNormBase]:
        """Return the norm class for ``p``."""

        normalized_p = _normalize_p(p)
        if normalized_p == sympy.S.One:
            return L1Norm
        if normalized_p == sympy.S(2):
            return L2Norm
        if normalized_p == sympy.S.Infinity:
            return LinftyNorm

        try:
            return self._cache[normalized_p]
        except KeyError:
            pass

        class_name = _private_lp_class_name(normalized_p)
        cls = type(
            class_name,
            (FiniteLpNorm,),
            {
                "__slots__": (),
                "__module__": __name__,
                "_p": normalized_p,
            },
        )
        globals()[class_name] = cls
        self._cache[normalized_p] = cls
        return cls


LpNorm = LpNormFactory()

_LP_NORMS_DOC = {
    "path": PurePosixPath("library/LpNorm"),
    "anchor": None,
    "label": "LpNorm",
}

L1Norm._mt_help = _LP_NORMS_DOC
L2Norm._mt_help = _LP_NORMS_DOC
LinftyNorm._mt_help = _LP_NORMS_DOC
FiniteLpNorm._mt_help = _LP_NORMS_DOC
LebesgueNormBase._mt_help = _LP_NORMS_DOC
EssentialSupremum._mt_help = _LP_NORMS_DOC
LpNorm._mt_help = _LP_NORMS_DOC


def _is_component_form(expr: Any, bindings: tuple[Any, ...], weight: Any) -> bool:
    """Return whether constructor arguments are internal component nodes."""

    return (
        isinstance(expr, NormOperand)
        and len(bindings) == 2
        and isinstance(bindings[0], BoundDomain)
        and isinstance(bindings[1], NormWeight)
        and weight == sympy.S.One
    )


def _normalize_public_binding(*bindings: Any) -> tuple[tuple[sympy.Symbol, ...], sympy.Basic]:
    """Return variables and domain from public norm binding arguments."""

    if not bindings:
        raise ValueError("bound-domain specification must be specified")

    if all(_is_bounded_interval_binding(binding) for binding in bindings):
        if len(bindings) == 1:
            variable, lower, upper = tuple(bindings[0])
            variables = _normalize_variables(variable)
            return variables, _normalize_domain((lower, upper), variables)

        variables = _normalize_variables(tuple(binding[0] for binding in bindings))
        domain_spec = tuple((binding[1], binding[2]) for binding in bindings)
        return variables, _normalize_domain(domain_spec, variables)

    if len(bindings) == 1 and isinstance(bindings[0], tuple | list | sympy.Tuple):
        entries = tuple(bindings[0])

        if entries and all(_is_bounded_interval_binding(entry) for entry in entries):
            variables = _normalize_variables(tuple(entry[0] for entry in entries))
            domain_spec = tuple((entry[1], entry[2]) for entry in entries)
            return variables, _normalize_domain(domain_spec, variables)

        if len(entries) == 2 and _is_grouped_domain_binding(entries):
            variables = _normalize_variables(entries[0])
            return variables, _normalize_domain(entries[1], variables)

    if len(bindings) == 3 and isinstance(bindings[0], sympy.Symbol):
        variable, lower, upper = bindings
        variables = _normalize_variables(variable)
        return variables, _normalize_domain((lower, upper), variables)

    raise ValueError(
        "bound-domain specification must be (x, a, b), "
        "(x, a, b), (y, c, d), ..., ((x, y, z), domain), "
        "or ((x,), domain)"
    )


def _normalize_variables(vars: Any) -> tuple[sympy.Symbol, ...]:
    """Return a validated tuple of bound symbols."""

    if vars is None:
        raise ValueError("vars must be specified")

    if isinstance(vars, sympy.Symbol):
        variables = (vars,)
    elif isinstance(vars, tuple | sympy.Tuple):
        variables = tuple(vars)
    else:
        raise ValueError("vars must be a Symbol or tuple of Symbols")

    if not variables:
        raise ValueError("at least one bound variable is required")
    if any(not isinstance(variable, sympy.Symbol) for variable in variables):
        raise ValueError("all bound variables must be Symbols")
    if len(set(variables)) != len(variables):
        raise ValueError("bound variables must be distinct")
    return variables


def _normalize_domain(domain: Any, variables: tuple[sympy.Symbol, ...]) -> sympy.Basic:
    """Return a normalized domain object for the given variables."""

    if domain is None:
        raise ValueError("domain must be specified")

    # One-dimensional two-tuples are interval shorthand. Multi-dimensional
    # tuple domains must provide one interval or set factor per variable.
    if len(variables) == 1 and _is_interval_spec(domain):
        normalized = sympy.Interval(*domain)
        _reject_bound_symbols_in_domain(normalized, variables)
        return normalized

    if len(variables) > 1 and isinstance(domain, tuple | list | sympy.Tuple):
        if len(domain) != len(variables):
            raise ValueError("domain dimension does not match variables")
        if not all(_is_interval_spec(factor) for factor in domain):
            raise ValueError("domain dimension does not match variables")
        normalized = sympy.ProductSet(
            *(sympy.Interval(*factor) for factor in domain)
        )
        _reject_bound_symbols_in_domain(normalized, variables)
        return normalized

    normalized = sympy.sympify(domain)
    if len(variables) > 1 and isinstance(normalized, sympy.Interval):
        raise ValueError("domain dimension does not match variables")
    if isinstance(normalized, sympy.ProductSet):
        if len(tuple(normalized.sets)) != len(variables):
            raise ValueError("domain dimension does not match variables")
        _reject_bound_symbols_in_domain(normalized, variables)
    elif isinstance(normalized, sympy.Interval):
        _reject_bound_symbols_in_domain(normalized, variables)
    return normalized


def _domain_to_integral_limits(
    variables: tuple[sympy.Symbol, ...],
    domain: sympy.Basic,
) -> tuple[tuple[sympy.Basic, ...], ...]:
    """Return SymPy integration limits for supported domains."""

    if len(variables) == 1:
        variable = variables[0]
        if isinstance(domain, sympy.Interval):
            return ((variable, domain.start, domain.end),)
        if domain == sympy.S.Reals:
            return ((variable, -sympy.oo, sympy.oo),)
        raise NotImplementedError("unsupported one-dimensional domain")

    if isinstance(domain, sympy.ProductSet):
        factors = tuple(domain.sets)
        if len(factors) != len(variables):
            raise ValueError("domain dimension does not match variables")

        limits: list[tuple[sympy.Basic, ...]] = []
        for variable, factor in zip(variables, factors):
            if isinstance(factor, sympy.Interval):
                limits.append((variable, factor.start, factor.end))
            elif factor == sympy.S.Reals:
                limits.append((variable, -sympy.oo, sympy.oo))
            else:
                raise NotImplementedError("unsupported product-domain factor")
        return tuple(limits)

    raise NotImplementedError(
        "only interval and rectangular product domains are supported"
    )


def _normalize_p(p: Any) -> sympy.Basic:
    """Return a validated Lebesgue exponent."""

    normalized = sympy.sympify(p)
    if normalized == sympy.oo:
        return sympy.S.Infinity
    if normalized.is_real is False:
        raise ValueError("p must be real")
    if normalized.is_finite is False:
        raise ValueError("finite Lp norms require finite p")
    if (normalized - 1).is_nonnegative is False:
        raise ValueError("p must satisfy p >= 1")
    if (normalized - 1).is_nonnegative is None:
        raise ValueError("could not prove p >= 1")
    return normalized


def _is_interval_spec(value: Any) -> bool:
    """Return whether ``value`` is a two-endpoint interval shorthand."""

    return isinstance(value, tuple | list | sympy.Tuple) and len(value) == 2


def _is_bounded_interval_binding(value: Any) -> bool:
    """Return whether ``value`` is ``(symbol, lower, upper)`` shorthand."""

    if not isinstance(value, tuple | list | sympy.Tuple) or len(value) != 3:
        return False
    return isinstance(value[0], sympy.Symbol)


def _is_grouped_domain_binding(entries: tuple[Any, ...]) -> bool:
    """Return whether a single tuple uses the grouped ``(variables, domain)`` form."""

    variables = entries[0]
    if isinstance(variables, sympy.Symbol):
        return False
    if not isinstance(variables, tuple | sympy.Tuple):
        return False
    return True


def _reject_bound_symbols_in_domain(
    domain: sympy.Basic,
    variables: tuple[sympy.Symbol, ...],
) -> None:
    """Reject interval endpoints that contain bound variables."""

    bound = set(variables)
    if isinstance(domain, sympy.Interval):
        endpoints = (domain.start, domain.end)
    elif isinstance(domain, sympy.ProductSet):
        endpoints = tuple(
            endpoint
            for factor in domain.sets
            if isinstance(factor, sympy.Interval)
            for endpoint in (factor.start, factor.end)
        )
    else:
        return

    if any(endpoint.free_symbols & bound for endpoint in endpoints):
        raise ValueError("domain endpoints must not contain bound variables")


def _domain_free_symbols(domain: sympy.Basic) -> set[sympy.Basic]:
    """Return symbols contributed by domain parameters."""

    if isinstance(domain, sympy.Interval):
        return set(domain.start.free_symbols) | set(domain.end.free_symbols)
    if isinstance(domain, sympy.ProductSet):
        symbols: set[sympy.Basic] = set()
        for factor in domain.sets:
            symbols.update(_domain_free_symbols(factor))
        return symbols
    return set(domain.free_symbols)


def _format_variables(variables: tuple[sympy.Symbol, ...], printer: Any) -> str:
    """Return a plain-text variable display matching public constructor syntax."""

    if len(variables) == 1:
        return printer._print(variables[0])
    return "(" + ", ".join(printer._print(variable) for variable in variables) + ")"


def _format_variable_tuple(variables: tuple[sympy.Symbol, ...], printer: Any) -> str:
    """Return a plain-text tuple of variables for grouped domain syntax."""

    if len(variables) == 1:
        return f"({printer._print(variables[0])},)"
    return "(" + ", ".join(printer._print(variable) for variable in variables) + ")"


def _format_binding(
    variables: tuple[sympy.Symbol, ...],
    domain: sympy.Basic,
    printer: Any,
) -> str:
    """Return a plain-text bound-domain display matching public syntax."""

    if len(variables) == 1 and isinstance(domain, sympy.Interval):
        variable = printer._print(variables[0])
        return f"({variable}, {printer._print(domain.start)}, {printer._print(domain.end)})"

    if isinstance(domain, sympy.ProductSet):
        factors = tuple(domain.sets)
        if len(factors) == len(variables) and all(
            isinstance(factor, sympy.Interval) for factor in factors
        ):
            bindings = (
                f"({printer._print(variable)}, "
                f"{printer._print(factor.start)}, {printer._print(factor.end)})"
                for variable, factor in zip(variables, factors)
            )
            return ", ".join(bindings)

    return f"({_format_variable_tuple(variables, printer)}, {printer._print(domain)})"


def _binding_as_public_args(
    variables: tuple[sympy.Symbol, ...],
    domain: sympy.Basic,
) -> tuple[Any, ...]:
    """Return constructor binding arguments preserving public notation."""

    if len(variables) == 1 and isinstance(domain, sympy.Interval):
        return ((variables[0], domain.start, domain.end),)

    if isinstance(domain, sympy.ProductSet):
        factors = tuple(domain.sets)
        if len(factors) == len(variables) and all(
            isinstance(factor, sympy.Interval) for factor in factors
        ):
            return tuple(
                (variable, factor.start, factor.end)
                for variable, factor in zip(variables, factors)
            )

    return ((variables, domain),)


def _private_lp_class_name(p: sympy.Basic) -> str:
    """Return a stable internal class name for a finite exponent."""

    rendered = sympy.srepr(p)
    safe = "".join(char if char.isalnum() else "_" for char in rendered)
    return f"_FiniteLpNorm_{safe}"
