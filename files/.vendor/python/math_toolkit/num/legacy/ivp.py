"""Represent and solve first-order initial value problems."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp
import sympy

from ...sympy_extensions.vector_equations import VectorEquation2SystemOfEquations
from .compile import compile_num
from .diagnostics import NumArgumentError, NumSolverError
from .functions import ImplementedFunction


@dataclass(frozen=True)
class _NormalizedIVP:
    """Store a validated first-order IVP system."""

    raw_input: object
    numeric_rhs: Callable[[object, object, object], object] | None
    rhs_entries: tuple[object, ...]
    rhs_function: object | None
    unknowns: tuple[object, ...]
    component_keys: tuple[object, ...]
    state_symbols: tuple[sympy.Symbol, ...]
    domain: tuple[sympy.Symbol, object, object]
    initial_values: tuple[object, ...]
    initial_data: dict[object, object]
    parameters: tuple[sympy.Symbol, ...]
    free_symbols: frozenset[sympy.Symbol]
    selection_mode: str


@dataclass(frozen=True)
class _SolverOptions:
    """Store public SciPy IVP solver options."""

    samples: int | None
    method: str
    rtol: float
    atol: float
    max_step: float


@dataclass(frozen=True)
class _IVPSolveData:
    """Store one concrete IVP solver run."""

    sample_points: np.ndarray
    sample_values: np.ndarray
    dense_solution: Callable[[object], object]
    parameter_values: dict[sympy.Symbol, object]
    raw_result: object


@dataclass(frozen=True, init=False)
class IVP:
    """Represent a symbolic first-order initial value problem.

    Parameters
    ----------
    equations_or_rhs : object
        RHS array, array of pure first-derivative equations, vector equation,
        or numeric callback with signature ``rhs(t_value, state_vector, params)``.
    unknowns : sequence, optional
        State labels for RHS-array and numeric-callback input.
    domain : tuple
        Independent-variable interval ``(variable, start, stop)``.
    initial_data : mapping
        Initial values keyed by state labels or by applied functions at the
        start of the domain.
    parameters : sequence, optional
        Explicit parameter symbols for numeric callbacks or symbolic problems.

    Methods
    -------
    subs
        Return a specialized symbolic IVP.
    xreplace
        Return an exactly replaced symbolic IVP.
    explain
        Return a short human-readable problem summary.
    """

    equations_or_rhs: object
    unknowns: tuple[object, ...]
    domain: tuple[sympy.Symbol, object, object]
    initial_data: dict[object, object]
    parameters: tuple[sympy.Symbol, ...]
    options: Mapping[str, object]
    _system: _NormalizedIVP
    _unknowns_argument: tuple[object, ...] | None
    _explicit_parameters: tuple[sympy.Symbol, ...]
    _parameters_were_explicit: bool
    _solver: ClassVar[object]

    def __init__(
        self,
        equations_or_rhs: object,
        *,
        unknowns: Sequence[object] | None = None,
        domain: object,
        initial_data: Mapping[object, object],
        parameters: Sequence[object] = (),
        **options: object,
    ) -> None:
        """Create an unevaluated symbolic IVP problem."""

        unknowns_argument = None if unknowns is None else tuple(unknowns)
        explicit_parameters = _normalize_parameters(parameters)
        system = _normalize_ivp_system(
            equations_or_rhs,
            unknowns=unknowns_argument,
            domain=domain,
            initial_data=initial_data,
            explicit_parameters=explicit_parameters,
        )

        object.__setattr__(self, "equations_or_rhs", equations_or_rhs)
        object.__setattr__(self, "unknowns", system.unknowns)
        object.__setattr__(self, "domain", system.domain)
        object.__setattr__(self, "initial_data", system.initial_data)
        object.__setattr__(self, "parameters", system.parameters)
        object.__setattr__(self, "options", dict(options))
        object.__setattr__(self, "_system", system)
        object.__setattr__(self, "_unknowns_argument", unknowns_argument)
        object.__setattr__(self, "_explicit_parameters", explicit_parameters)
        object.__setattr__(self, "_parameters_were_explicit", bool(parameters))

    @property
    def num_role(self) -> str:
        """Return the symbolic/numeric boundary role name."""

        return "symbolic-problem"

    @property
    def kind(self) -> str:
        """Return the mathematical problem kind."""

        return "ivp"

    @property
    def free_symbols(self) -> frozenset[sympy.Symbol]:
        """Return unresolved symbols that parameterize the problem."""

        return self._system.free_symbols

    @property
    def independent_variables(self) -> tuple[sympy.Symbol, ...]:
        """Return the independent variables of this IVP."""

        return (self.domain[0],)

    @property
    def diagnostics(self) -> tuple[object, ...]:
        """Return problem diagnostics collected during construction."""

        return ()

    def subs(self, *args: object, **kwargs: object) -> "IVP":
        """Return a specialized IVP after ordinary SymPy substitution."""

        raw_input = _subs_value(self.equations_or_rhs, *args, **kwargs)
        domain = tuple(_subs_value(part, *args, **kwargs) for part in self.domain)
        initial_data = {
            _subs_value(key, *args, **kwargs): _subs_value(value, *args, **kwargs)
            for key, value in self.initial_data.items()
        }
        unknowns = (
            None
            if self._unknowns_argument is None
            else tuple(_subs_value(value, *args, **kwargs) for value in self._unknowns_argument)
        )
        parameters = _remaining_explicit_parameters(
            self._explicit_parameters,
            self._parameters_were_explicit,
            *args,
            **kwargs,
        )
        return IVP(
            raw_input,
            unknowns=unknowns,
            domain=domain,
            initial_data=initial_data,
            parameters=parameters,
            **dict(self.options),
        )

    def xreplace(self, rule: Mapping[object, object]) -> "IVP":
        """Return an IVP after exact SymPy replacement."""

        raw_input = _xreplace_value(self.equations_or_rhs, rule)
        domain = tuple(_xreplace_value(part, rule) for part in self.domain)
        initial_data = {
            _xreplace_value(key, rule): _xreplace_value(value, rule)
            for key, value in self.initial_data.items()
        }
        unknowns = (
            None
            if self._unknowns_argument is None
            else tuple(_xreplace_value(value, rule) for value in self._unknowns_argument)
        )
        parameters = (
            tuple(_xreplace_value(parameter, rule) for parameter in self._explicit_parameters)
            if self._parameters_were_explicit
            else ()
        )
        parameters = tuple(parameter for parameter in parameters if isinstance(parameter, sympy.Symbol))
        return IVP(
            raw_input,
            unknowns=unknowns,
            domain=domain,
            initial_data=initial_data,
            parameters=parameters,
            **dict(self.options),
        )

    def explain(self) -> str:
        """Return a short explanation of the symbolic IVP."""

        return (
            f"num_role: {self.num_role}\n"
            f"kind: {self.kind}\n"
            f"unknowns: {self.unknowns}\n"
            f"domain: {self.domain}\n"
            f"parameters: {self.parameters}"
        )

    def __repr__(self) -> str:
        """Return a concise problem representation."""

        return (
            "IVP("
            f"unknowns={self.unknowns!r}, "
            f"domain={self.domain!r}, "
            f"parameters={self.parameters!r}"
            ")"
        )

    def _latex(self, printer: object) -> str:
        """Return a mathematical LaTeX representation of the IVP."""

        lines = [
            *_ivp_equation_latex_lines(self, printer),
            *_ivp_initial_latex_lines(self, printer),
        ]
        body = r" \\ ".join(lines)
        domain = _ivp_domain_latex_line(self, printer)
        return (
            r"\left\{\begin{array}{l}"
            f"{body}"
            rf"\end{{array}}\right.\quad {domain}"
        )

    def _repr_latex_(self) -> str:
        """Return the notebook LaTeX display payload for this IVP."""

        return f"$${sympy.latex(self)}$$"

    def _rhs_value(
        self,
        t_value: object,
        state_vector: np.ndarray,
        parameter_values: Mapping[sympy.Symbol, object],
    ) -> np.ndarray:
        """Evaluate the normalized RHS vector at one solver point."""

        if self._system.numeric_rhs is not None:
            params = _callback_parameter_dict(parameter_values)
            values = self._system.numeric_rhs(t_value, state_vector, params)
        else:
            if self._system.rhs_function is None:
                raise NumSolverError("IVP has no RHS function to evaluate.")
            arguments = (
                t_value,
                *tuple(state_vector),
                *tuple(parameter_values[parameter] for parameter in self.parameters),
            )
            values = self._system.rhs_function(*arguments)

        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != (len(self.unknowns),):
            raise NumSolverError(
                "IVP RHS returned a vector of length "
                f"{array.shape[0]}, expected {len(self.unknowns)}."
            )
        return array


class IVPSolution:
    """Represent a Python container of IVP solution branches."""

    def __init__(self, branches: Sequence["IVPSolutionBranch"]) -> None:
        self._branches = tuple(branches)

    @property
    def diagnostics(self) -> tuple[object, ...]:
        """Return diagnostics attached to solution branches."""

        diagnostics: list[object] = []
        for branch in self._branches:
            diagnostics.extend(branch.diagnostics)
        return tuple(diagnostics)

    def __getitem__(self, index: int) -> "IVPSolutionBranch":
        """Return one solution branch by position."""

        return self._branches[index]

    def __len__(self) -> int:
        """Return the number of solution branches."""

        return len(self._branches)

    def __iter__(self) -> object:
        """Iterate over solution branches."""

        return iter(self._branches)

    def explain(self) -> str:
        """Return a short explanation of the solution container."""

        return f"IVPSolution with {len(self)} branch(es)."

    def __repr__(self) -> str:
        """Return a concise solution representation."""

        return f"IVPSolution(branches={len(self)})"


class IVPSolutionBranch:
    """Represent one implemented symbolic IVP solution branch."""

    def __init__(
        self,
        ivp: IVP,
        options: _SolverOptions,
        *,
        data: _IVPSolveData | None = None,
    ) -> None:
        self.ivp = ivp
        self.options = options
        self.symbolic_origin = ivp
        self.required_parameters = ivp.parameters
        self.independent_variables = ivp.independent_variables
        self.domain = ivp.domain
        self.output_shape = (len(ivp.unknowns),)
        self.backend = "scipy.integrate.solve_ivp"
        self.discretization = {
            "samples": options.samples,
            "method": options.method,
        }
        self.diagnostics: tuple[object, ...] = ()
        self.cache_label = f"ivp:{id(self):x}"
        self._data = data
        self._cache: dict[tuple[object, ...], _IVPSolveData] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Components are symbolic function heads; numerical evaluation remains
        # behind Num so composed expressions share the same boundary contract.
        self._components: dict[object, ImplementedFunction] = {}
        for index, key in enumerate(ivp._system.component_keys):
            self._components[key] = ImplementedFunction(
                display_name=_component_display_name(key),
                branch=self,
                component_key=key,
                component_index=index,
            )

    @property
    def num_role(self) -> str:
        """Return the symbolic/numeric boundary role name."""

        return "implemented-solution"

    @property
    def implementation_status(self) -> str:
        """Return whether this branch is concrete or parameterized."""

        return "parameterized" if self.required_parameters else "ready"

    @property
    def sample_points(self) -> np.ndarray | None:
        """Return stored sample points when a concrete solve is available."""

        data = self._data
        return None if data is None else data.sample_points

    @property
    def sample_values(self) -> np.ndarray | None:
        """Return stored sample values when a concrete solve is available."""

        data = self._data
        return None if data is None else data.sample_values

    def __getitem__(self, key: object) -> ImplementedFunction:
        """Return an implemented component function by label or position."""

        try:
            return self._components[key]
        except KeyError as exc:
            raise KeyError(f"IVP solution has no component {key!r}.") from exc

    def evaluate_component(
        self,
        component_index: int,
        t_values: object,
        parameter_values: Mapping[sympy.Symbol, object],
    ) -> object:
        """Evaluate one component over scalar or array sample points."""

        data = self._data_for(parameter_values)
        sample_array = np.asarray(t_values, dtype=float)
        scalar_input = sample_array.shape == ()
        flat_times = sample_array.reshape(-1) if not scalar_input else sample_array.reshape(1)

        values = np.asarray(data.dense_solution(flat_times), dtype=float)
        component_values = values[component_index].reshape(flat_times.shape)
        if scalar_input:
            return component_values[0].item()
        return component_values.reshape(sample_array.shape)

    def subs(self, *args: object, **kwargs: object) -> "IVPSolutionBranch":
        """Return this branch after symbolic substitution."""

        return _make_branch(self.ivp.subs(*args, **kwargs), self.options)

    def xreplace(self, rule: Mapping[object, object]) -> "IVPSolutionBranch":
        """Return this branch after exact symbolic replacement."""

        return _make_branch(self.ivp.xreplace(rule), self.options)

    def cache_info(self) -> dict[str, object]:
        """Return public information about cached parameterized solves."""

        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "parameters": tuple(parameter.name for parameter in self.required_parameters),
        }

    def clear_cache(self) -> None:
        """Clear cached parameterized solver results."""

        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def explain(self) -> str:
        """Return a short explanation of this implemented solution branch."""

        return (
            f"num_role: {self.num_role}\n"
            f"implementation_status: {self.implementation_status}\n"
            f"domain: {self.domain}\n"
            f"required_parameters: {self.required_parameters}"
        )

    def _data_for(
        self,
        parameter_values: Mapping[sympy.Symbol, object],
    ) -> _IVPSolveData:
        """Return concrete solver data for a parameter tuple."""

        if not self.required_parameters:
            if self._data is None:
                self._data = _solve_concrete_ivp(self.ivp, self.options, {})
            return self._data

        key = tuple(parameter_values[parameter] for parameter in self.required_parameters)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        data = _solve_concrete_ivp(self.ivp, self.options, parameter_values)
        self._cache[key] = data
        return data


def SolveIVP(
    problem_or_rhs: object,
    *,
    samples: int | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float = np.inf,
    **ivp_kwargs: object,
) -> IVPSolution:
    """Solve an IVP problem or construct one from public IVP arguments.

    Parameters
    ----------
    problem_or_rhs : object
        Existing ``IVP`` problem, or the first argument used to construct one.
    samples : int, optional
        Number of evenly spaced stored samples including both endpoints.
    method : str, optional
        SciPy ``solve_ivp`` method.
    rtol : float, optional
        Relative solver tolerance.
    atol : float, optional
        Absolute solver tolerance.
    max_step : float, optional
        Maximum solver step passed to SciPy.
    **ivp_kwargs : object
        Constructor keywords used only by the convenience form.

    Returns
    -------
    IVPSolution
        Python solution container with one implemented solution branch.
    """

    options = _solver_options(
        samples=samples,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if isinstance(problem_or_rhs, IVP):
        if ivp_kwargs:
            unknown = ", ".join(sorted(ivp_kwargs))
            raise NumArgumentError(
                "SolveIVP received IVP constructor keyword(s) for an existing "
                f"problem: {unknown}."
            )
        branch = _make_branch(problem_or_rhs, options)
        return IVPSolution([branch])

    from .solve import Solve

    problem = IVP(problem_or_rhs, **ivp_kwargs)
    return Solve(
        problem,
        samples=samples,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )


def _make_branch(ivp: IVP, options: _SolverOptions) -> IVPSolutionBranch:
    """Return a branch, solving immediately when the problem is concrete."""

    data = None
    if not ivp.parameters:
        data = _solve_concrete_ivp(ivp, options, {})
    return IVPSolutionBranch(ivp, options, data=data)


def _normalize_ivp_system(
    equations_or_rhs: object,
    *,
    unknowns: tuple[object, ...] | None,
    domain: object,
    initial_data: Mapping[object, object],
    explicit_parameters: tuple[sympy.Symbol, ...],
) -> _NormalizedIVP:
    """Return a normalized first-order IVP system."""

    domain_tuple = _normalize_domain(domain)
    if not isinstance(initial_data, Mapping):
        raise NumArgumentError("IVP initial_data must be a mapping.")

    if _is_numeric_rhs(equations_or_rhs):
        return _normalize_numeric_rhs(
            equations_or_rhs,
            unknowns=unknowns,
            domain=domain_tuple,
            initial_data=initial_data,
            explicit_parameters=explicit_parameters,
        )

    vector_equation = _vector_equation_or_none(equations_or_rhs)
    if vector_equation is not None:
        equations = VectorEquation2SystemOfEquations(vector_equation)
        return _normalize_equation_system(
            equations,
            domain=domain_tuple,
            initial_data=initial_data,
            explicit_parameters=explicit_parameters,
            selection_mode="position",
        )

    equations = _equation_sequence_or_none(equations_or_rhs)
    if equations is not None:
        return _normalize_equation_system(
            equations,
            domain=domain_tuple,
            initial_data=initial_data,
            explicit_parameters=explicit_parameters,
            selection_mode="label",
        )

    return _normalize_rhs_array(
        equations_or_rhs,
        unknowns=unknowns,
        domain=domain_tuple,
        initial_data=initial_data,
        explicit_parameters=explicit_parameters,
    )


def _normalize_rhs_array(
    rhs: object,
    *,
    unknowns: tuple[object, ...] | None,
    domain: tuple[sympy.Symbol, object, object],
    initial_data: Mapping[object, object],
    explicit_parameters: tuple[sympy.Symbol, ...],
) -> _NormalizedIVP:
    """Normalize RHS-array input with explicit state ordering."""

    if unknowns is None:
        raise NumArgumentError("IVP RHS-array input requires unknowns.")
    unknown_tuple = _normalize_unknowns(unknowns)
    rhs_entries = _rhs_entries(rhs)
    if len(rhs_entries) != len(unknown_tuple):
        raise NumArgumentError(
            "IVP RHS array length does not match unknowns: "
            f"{len(rhs_entries)} RHS entries for {len(unknown_tuple)} unknowns."
        )

    state_symbols = _state_symbols(len(unknown_tuple))
    replacements = {
        unknown: state_symbol
        for unknown, state_symbol in zip(unknown_tuple, state_symbols)
        if isinstance(unknown, sympy.Basic)
    }
    normalized_rhs = tuple(sympy.sympify(entry).xreplace(replacements) for entry in rhs_entries)
    initial_values = tuple(
        _initial_value_for_unknown(unknown, initial_data, domain[1])
        for unknown in unknown_tuple
    )
    normalized_initial_data = dict(zip(unknown_tuple, initial_values))
    parameters = _problem_parameters(
        explicit_parameters,
        rhs_entries=rhs_entries,
        unknowns=unknown_tuple,
        domain=domain,
        initial_values=initial_values,
    )
    rhs_function = compile_num(
        sympy.Matrix(normalized_rhs),
        args=(domain[0], *state_symbols, *parameters),
    )
    free_symbols = _problem_free_symbols(
        rhs_entries=rhs_entries,
        unknowns=unknown_tuple,
        domain=domain,
        initial_values=initial_values,
        parameters=parameters,
    )
    return _NormalizedIVP(
        raw_input=rhs,
        numeric_rhs=None,
        rhs_entries=normalized_rhs,
        rhs_function=rhs_function,
        unknowns=unknown_tuple,
        component_keys=unknown_tuple,
        state_symbols=state_symbols,
        domain=domain,
        initial_values=initial_values,
        initial_data=normalized_initial_data,
        parameters=parameters,
        free_symbols=free_symbols,
        selection_mode="label",
    )


def _normalize_equation_system(
    equations: Sequence[sympy.Equality],
    *,
    domain: tuple[sympy.Symbol, object, object],
    initial_data: Mapping[object, object],
    explicit_parameters: tuple[sympy.Symbol, ...],
    selection_mode: str,
) -> _NormalizedIVP:
    """Normalize pure first-derivative equation input."""

    if not equations:
        raise NumArgumentError("IVP equation input requires at least one equation.")

    derivative_data = [
        _pure_derivative_equation(equation, domain[0])
        for equation in equations
    ]
    unknowns = tuple(label for label, _, _ in derivative_data)
    applied_unknowns = tuple(applied for _, applied, _ in derivative_data)
    rhs_entries = tuple(rhs for _, _, rhs in derivative_data)
    state_symbols = _state_symbols(len(unknowns))
    replacements = dict(zip(applied_unknowns, state_symbols))
    normalized_rhs = tuple(sympy.sympify(rhs).xreplace(replacements) for rhs in rhs_entries)
    initial_values = tuple(
        _initial_value_for_unknown(unknown, initial_data, domain[1])
        for unknown in unknowns
    )
    normalized_initial_data = dict(zip(unknowns, initial_values))
    parameters = _problem_parameters(
        explicit_parameters,
        rhs_entries=rhs_entries,
        unknowns=(),
        domain=domain,
        initial_values=initial_values,
    )
    rhs_function = compile_num(
        sympy.Matrix(normalized_rhs),
        args=(domain[0], *state_symbols, *parameters),
    )
    component_keys: tuple[object, ...]
    if selection_mode == "position":
        component_keys = tuple(range(len(unknowns)))
    else:
        component_keys = unknowns
    free_symbols = _problem_free_symbols(
        rhs_entries=rhs_entries,
        unknowns=(),
        domain=domain,
        initial_values=initial_values,
        parameters=parameters,
    )
    return _NormalizedIVP(
        raw_input=tuple(equations),
        numeric_rhs=None,
        rhs_entries=normalized_rhs,
        rhs_function=rhs_function,
        unknowns=unknowns,
        component_keys=component_keys,
        state_symbols=state_symbols,
        domain=domain,
        initial_values=initial_values,
        initial_data=normalized_initial_data,
        parameters=parameters,
        free_symbols=free_symbols,
        selection_mode=selection_mode,
    )


def _normalize_numeric_rhs(
    rhs: object,
    *,
    unknowns: tuple[object, ...] | None,
    domain: tuple[sympy.Symbol, object, object],
    initial_data: Mapping[object, object],
    explicit_parameters: tuple[sympy.Symbol, ...],
) -> _NormalizedIVP:
    """Normalize an already numeric RHS callback."""

    if unknowns is None:
        raise NumArgumentError("Numeric IVP RHS input requires unknowns.")
    unknown_tuple = _normalize_unknowns(unknowns)
    initial_values = tuple(
        _initial_value_for_unknown(unknown, initial_data, domain[1])
        for unknown in unknown_tuple
    )
    normalized_initial_data = dict(zip(unknown_tuple, initial_values))
    free_symbols = _problem_free_symbols(
        rhs_entries=(),
        unknowns=unknown_tuple,
        domain=domain,
        initial_values=initial_values,
        parameters=explicit_parameters,
    )
    return _NormalizedIVP(
        raw_input=rhs,
        numeric_rhs=rhs,
        rhs_entries=(),
        rhs_function=None,
        unknowns=unknown_tuple,
        component_keys=unknown_tuple,
        state_symbols=_state_symbols(len(unknown_tuple)),
        domain=domain,
        initial_values=initial_values,
        initial_data=normalized_initial_data,
        parameters=explicit_parameters,
        free_symbols=free_symbols,
        selection_mode="label",
    )


def _solve_concrete_ivp(
    ivp: IVP,
    options: _SolverOptions,
    parameter_values: Mapping[sympy.Symbol, object],
) -> _IVPSolveData:
    """Run SciPy for one concrete IVP parameter tuple."""

    concrete_parameters = {
        parameter: parameter_values[parameter]
        for parameter in ivp.parameters
    }
    start = _numeric_scalar(ivp.domain[1], concrete_parameters, "domain start")
    stop = _numeric_scalar(ivp.domain[2], concrete_parameters, "domain stop")
    initial_values = np.asarray(
        [
            _numeric_scalar(value, concrete_parameters, "initial data")
            for value in ivp._system.initial_values
        ],
        dtype=float,
    )
    t_eval = None
    if options.samples is not None:
        t_eval = np.linspace(start, stop, options.samples)

    def rhs_callback(t_value: float, state_vector: np.ndarray) -> np.ndarray:
        return ivp._rhs_value(t_value, state_vector, concrete_parameters)

    result = scipy_solve_ivp(
        rhs_callback,
        (start, stop),
        initial_values,
        method=options.method,
        t_eval=t_eval,
        dense_output=True,
        rtol=options.rtol,
        atol=options.atol,
        max_step=options.max_step,
    )
    if not result.success:
        raise NumSolverError(f"IVP solver failed: {result.message}")
    if result.sol is None:
        raise NumSolverError("IVP solver did not return dense output.")

    return _IVPSolveData(
        sample_points=np.asarray(result.t, dtype=float),
        sample_values=np.asarray(result.y, dtype=float),
        dense_solution=result.sol,
        parameter_values=dict(concrete_parameters),
        raw_result=result,
    )


def _solver_options(
    *,
    samples: int | None,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> _SolverOptions:
    """Validate and store public solver options."""

    if samples is not None:
        if not isinstance(samples, int) or isinstance(samples, bool):
            raise NumArgumentError("SolveIVP samples must be an integer.")
        if samples < 2:
            raise NumArgumentError("SolveIVP samples must be at least 2.")
    return _SolverOptions(
        samples=samples,
        method=method,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )


def _normalize_domain(domain: object) -> tuple[sympy.Symbol, object, object]:
    """Return ``(variable, start, stop)`` after checking the public domain."""

    if not _is_sequence(domain) or len(domain) != 3:
        raise NumArgumentError("IVP domain must be (variable, start, stop).")
    variable, start, stop = domain
    if not isinstance(variable, sympy.Symbol):
        raise NumArgumentError("IVP domain variable must be a SymPy Symbol.")
    return variable, start, stop


def _ivp_equation_latex_lines(ivp: IVP, printer: object) -> tuple[str, ...]:
    """Return LaTeX equation lines for an IVP display."""

    raw_input = ivp.equations_or_rhs
    if ivp._system.numeric_rhs is not None:
        variable = _latex_print(ivp.domain[0], printer)
        state = _latex_print(sympy.Matrix(ivp.unknowns), printer)
        return (rf"\frac{{d}}{{d {variable}}}{state} = \operatorname{{rhs}}",)

    vector_equation = _vector_equation_or_none(raw_input)
    if vector_equation is not None:
        return (_latex_print(vector_equation, printer),)

    equations = _equation_sequence_or_none(raw_input)
    if equations is not None:
        return tuple(_latex_print(equation, printer) for equation in equations)

    variable = _latex_print(ivp.domain[0], printer)
    state_entries = [
        _state_display_for_unknown(unknown, ivp.domain[0])
        for unknown in ivp.unknowns
    ]
    display_replacements = {
        unknown: state_entry
        for unknown, state_entry in zip(ivp.unknowns, state_entries)
        if isinstance(unknown, sympy.Basic)
    }
    rhs_entries = tuple(
        _xreplace_value(entry, display_replacements)
        for entry in _rhs_entries(raw_input)
    )
    state = _latex_print(sympy.Matrix(state_entries), printer)
    rhs = _latex_print(sympy.Matrix(rhs_entries), printer)
    return (rf"\frac{{d}}{{d {variable}}}{state} = {rhs}",)


def _ivp_initial_latex_lines(ivp: IVP, printer: object) -> tuple[str, ...]:
    """Return LaTeX initial-condition lines for an IVP display."""

    start = ivp.domain[1]
    if _ivp_uses_vector_initial_latex(ivp):
        left_entries = [
            _initial_display_for_unknown(unknown, start)
            for unknown in ivp.unknowns
        ]
        left = _latex_print(sympy.Matrix(left_entries), printer)
        right = _latex_print(sympy.Matrix(ivp._system.initial_values), printer)
        return (f"{left} = {right}",)

    return tuple(
        (
            f"{_latex_print(_initial_display_for_unknown(unknown, start), printer)}"
            f" = {_latex_print(value, printer)}"
        )
        for unknown, value in zip(ivp.unknowns, ivp._system.initial_values)
    )


def _ivp_uses_vector_initial_latex(ivp: IVP) -> bool:
    """Return whether initial data should render as one vector equation."""

    raw_input = ivp.equations_or_rhs
    if _vector_equation_or_none(raw_input) is not None:
        return True
    if _equation_sequence_or_none(raw_input) is not None:
        return False
    return len(ivp.unknowns) > 1


def _ivp_domain_latex_line(ivp: IVP, printer: object) -> str:
    """Return the LaTeX domain line for an IVP display."""

    variable, start, stop = ivp.domain
    return (
        f"{_latex_print(variable, printer)} \\in "
        rf"\left[{_latex_print(start, printer)}, {_latex_print(stop, printer)}\right]"
    )


def _state_display_for_unknown(unknown: object, variable: object) -> object:
    """Return a display expression for one unknown state component."""

    if isinstance(unknown, sympy.Symbol):
        return sympy.Function(unknown.name)(variable)
    try:
        applied = unknown(variable)
    except TypeError:
        return unknown
    return applied


def _initial_display_for_unknown(unknown: object, start: object) -> object:
    """Return a display expression for one initial condition target."""

    if isinstance(unknown, sympy.Symbol):
        return sympy.Function(unknown.name)(start)
    try:
        applied = unknown(start)
    except TypeError:
        return unknown
    return applied


def _latex_print(value: object, printer: object) -> str:
    """Return a LaTeX fragment through SymPy's active printer."""

    print_method = getattr(printer, "_print", None)
    if callable(print_method):
        return print_method(value)
    return sympy.latex(value)


def _normalize_unknowns(unknowns: Sequence[object]) -> tuple[object, ...]:
    """Return a concrete unknown tuple after public validation."""

    if not _is_sequence(unknowns):
        raise NumArgumentError("IVP unknowns must be a sequence.")
    normalized = tuple(unknowns)
    if not normalized:
        raise NumArgumentError("IVP unknowns must not be empty.")
    return normalized


def _normalize_parameters(parameters: Sequence[object]) -> tuple[sympy.Symbol, ...]:
    """Return explicit parameter symbols in deterministic order."""

    if parameters is None:
        return ()
    if not _is_sequence(parameters):
        raise NumArgumentError("IVP parameters must be a sequence of symbols.")
    normalized: list[sympy.Symbol] = []
    for parameter in parameters:
        if not isinstance(parameter, sympy.Symbol):
            raise NumArgumentError("IVP parameters must be SymPy symbols.")
        normalized.append(parameter)
    return tuple(dict.fromkeys(normalized))


def _rhs_entries(rhs: object) -> tuple[object, ...]:
    """Return vector RHS entries from a public RHS-array value."""

    if isinstance(rhs, sympy.MatrixBase):
        return tuple(rhs[row, col] for row in range(rhs.rows) for col in range(rhs.cols))
    if _is_sequence(rhs):
        return tuple(rhs)
    raise NumArgumentError("IVP RHS-array input must be a sequence or Matrix.")


def _equation_sequence_or_none(value: object) -> tuple[sympy.Equality, ...] | None:
    """Return equation entries when ``value`` is an equation sequence."""

    if isinstance(value, sympy.Equality):
        if _vector_equation_or_none(value) is None:
            return (value,)
        return None
    if not _is_sequence(value):
        return None
    entries = tuple(value)
    if entries and all(isinstance(entry, sympy.Equality) for entry in entries):
        return entries
    return None


def _vector_equation_or_none(value: object) -> sympy.Equality | None:
    """Return a matrix-shaped equality when public input is a vector equation."""

    if not isinstance(value, sympy.Equality):
        return None
    if _finite_matrix_shape(value.lhs) is None or _finite_matrix_shape(value.rhs) is None:
        return None
    return value


def _finite_matrix_shape(value: object) -> tuple[int, int] | None:
    """Return a finite matrix shape when one is available."""

    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        return None
    try:
        return int(shape[0]), int(shape[1])
    except TypeError as exc:
        raise NumArgumentError(
            f"IVP vector equations require finite matrix shapes, got {shape}."
        ) from exc


def _pure_derivative_equation(
    equation: sympy.Equality,
    variable: sympy.Symbol,
) -> tuple[object, sympy.Function, object]:
    """Return ``(label, applied_unknown, rhs)`` for a pure derivative equation."""

    lhs = equation.lhs
    if not isinstance(lhs, sympy.Derivative):
        raise NumArgumentError(
            "IVP equation left side must be a pure first derivative."
        )
    if lhs.variables != (variable,):
        raise NumArgumentError(
            "IVP equation left side must be a pure first derivative with "
            f"respect to {variable}."
        )
    applied = lhs.expr
    if (
        not isinstance(applied, sympy.Function)
        or len(applied.args) != 1
        or applied.args[0] != variable
    ):
        raise NumArgumentError(
            "IVP equation left side must differentiate one unknown function "
            f"applied to {variable}."
        )
    return applied.func, applied, equation.rhs


def _initial_value_for_unknown(
    unknown: object,
    initial_data: Mapping[object, object],
    start: object,
) -> object:
    """Return the initial value for one unknown label."""

    if unknown in initial_data:
        return initial_data[unknown]
    try:
        applied_at_start = unknown(start)
    except TypeError:
        applied_at_start = None
    if applied_at_start is not None and applied_at_start in initial_data:
        return initial_data[applied_at_start]
    raise NumArgumentError(f"IVP initial data is missing unknown {unknown}.")


def _problem_parameters(
    explicit_parameters: tuple[sympy.Symbol, ...],
    *,
    rhs_entries: Sequence[object],
    unknowns: Sequence[object],
    domain: tuple[sympy.Symbol, object, object],
    initial_values: Sequence[object],
) -> tuple[sympy.Symbol, ...]:
    """Return explicit or inferred IVP parameters."""

    if explicit_parameters:
        return explicit_parameters
    return tuple(
        sorted(
            _problem_free_symbols(
                rhs_entries=rhs_entries,
                unknowns=unknowns,
                domain=domain,
                initial_values=initial_values,
                parameters=(),
            ),
            key=lambda symbol: symbol.name,
        )
    )


def _problem_free_symbols(
    *,
    rhs_entries: Sequence[object],
    unknowns: Sequence[object],
    domain: tuple[sympy.Symbol, object, object],
    initial_values: Sequence[object],
    parameters: Sequence[sympy.Symbol],
) -> frozenset[sympy.Symbol]:
    """Return unresolved problem parameters visible through SymPy."""

    symbols: set[sympy.Symbol] = set(parameters)
    for value in (*rhs_entries, domain[1], domain[2], *initial_values):
        symbols.update(_free_symbols(value))
    symbols.discard(domain[0])
    for unknown in unknowns:
        symbols.difference_update(_free_symbols(unknown))
        if isinstance(unknown, sympy.Symbol):
            symbols.discard(unknown)
    return frozenset(symbols)


def _free_symbols(value: object) -> set[sympy.Symbol]:
    """Return free symbols from SymPy-compatible values."""

    if isinstance(value, sympy.MatrixBase):
        symbols: set[sympy.Symbol] = set()
        for entry in value:
            symbols.update(_free_symbols(entry))
        return symbols
    if isinstance(value, sympy.Basic):
        return set(value.free_symbols)
    try:
        sympified = sympy.sympify(value)
    except sympy.SympifyError:
        return set()
    if isinstance(sympified, sympy.Basic):
        return set(sympified.free_symbols)
    return set()


def _state_symbols(count: int) -> tuple[sympy.Symbol, ...]:
    """Return private state placeholders for RHS compilation."""

    return tuple(sympy.Symbol(f"mt_state_{index}") for index in range(count))


def _numeric_scalar(
    value: object,
    parameter_values: Mapping[sympy.Symbol, object],
    description: str,
) -> float:
    """Return a concrete float after substituting parameter values."""

    concrete = value
    if isinstance(concrete, sympy.Basic):
        concrete = concrete.subs(parameter_values)
        if concrete.free_symbols:
            joined = ", ".join(symbol.name for symbol in sorted(concrete.free_symbols, key=lambda item: item.name))
            raise NumArgumentError(
                f"IVP {description} still depends on unresolved parameter(s): {joined}."
            )
    try:
        return float(concrete)
    except (TypeError, ValueError) as exc:
        raise NumArgumentError(f"IVP {description} must be numeric.") from exc


def _is_numeric_rhs(value: object) -> bool:
    """Return whether ``value`` is an already numeric RHS callback."""

    return callable(value) and not isinstance(value, sympy.Basic | sympy.MatrixBase)


def _is_sequence(value: object) -> bool:
    """Return whether ``value`` is a public sequence, excluding strings."""

    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _subs_value(value: object, *args: object, **kwargs: object) -> object:
    """Apply ``subs`` to SymPy-compatible containers and values."""

    if isinstance(value, sympy.MatrixBase | sympy.Basic):
        return value.subs(*args, **kwargs)
    if _is_sequence(value):
        return tuple(_subs_value(part, *args, **kwargs) for part in value)
    return value


def _xreplace_value(value: object, rule: Mapping[object, object]) -> object:
    """Apply ``xreplace`` to SymPy-compatible containers and values."""

    if isinstance(value, sympy.MatrixBase | sympy.Basic):
        return value.xreplace(rule)
    if _is_sequence(value):
        return tuple(_xreplace_value(part, rule) for part in value)
    return value


def _remaining_explicit_parameters(
    parameters: tuple[sympy.Symbol, ...],
    parameters_were_explicit: bool,
    *args: object,
    **kwargs: object,
) -> tuple[sympy.Symbol, ...]:
    """Return explicit parameters that remain symbolic after substitution."""

    if not parameters_were_explicit:
        return ()
    remaining: list[sympy.Symbol] = []
    for parameter in parameters:
        substituted = parameter.subs(*args, **kwargs)
        if isinstance(substituted, sympy.Symbol):
            remaining.append(substituted)
    return tuple(dict.fromkeys(remaining))


def _callback_parameter_dict(
    parameter_values: Mapping[sympy.Symbol, object],
) -> dict[object, object]:
    """Return callback parameters by symbols and names."""

    params: dict[object, object] = {}
    for parameter, value in parameter_values.items():
        params[parameter] = value
        params[parameter.name] = value
    return params


def _component_display_name(key: object) -> str:
    """Return a readable component name from a branch key."""

    if isinstance(key, int):
        return f"component_{key}"
    if isinstance(key, sympy.Symbol):
        return key.name
    name = getattr(key, "__name__", None)
    if isinstance(name, str):
        return name
    return str(key)


IVP._solver = staticmethod(SolveIVP)
