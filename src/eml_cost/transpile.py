"""EML tree → executable code transpiler (Tool 5 of the LLM-native build).

Turn any verified EML tree (a SymPy expression) into executable code
in four targets:

  - ``eml_tree_to_python``  — plain ``math.*`` functions
  - ``eml_tree_to_numpy``   — vectorised ``np.*`` functions
  - ``eml_tree_to_sympy``   — symbolic / differentiable form
  - ``eml_tree_to_c``       — C with ``math.h`` (forward bridge to
    libmonogate.h once the Monogate Extractor product lands)

Each call returns a frozen :class:`TranspileResult` with the function
source, required imports, the deduced free-variable list, the EML
profile (chain order / node count / cost class), and an
auto-generated verification snippet sampled at a deterministic
point.

The transpiler is intentionally **structural**: it does not
canonicalise, inline constants, or attempt to optimise the expression.
Pass the expression through :func:`eml_cost.canonicalize` first if
you want the SuperBEST-routed form.

Honest framing
--------------

This is a thin shell over SymPy's existing code printers
(:mod:`sympy.printing.pycode`, :mod:`sympy.printing.numpy`,
:func:`sympy.ccode`). The value-add is:

  - consistent function-source wrapping with imports;
  - free-symbol auto-detection in alphabetical-by-name order;
  - automatic verification snippet generation;
  - per-target output that matches the conventions agents expect
    in their sandboxes.

Usage
-----

    >>> import sympy as sp
    >>> from eml_cost import eml_tree_to_python, eml_tree_to_numpy
    >>> x, t, omega = sp.symbols("x t omega")
    >>> expr = sp.exp(x) * sp.cos(omega * t)
    >>> r = eml_tree_to_python(expr)
    >>> print(r.imports)
    import math
    >>> print(r.code)
    def f(omega, t, x):
        return math.exp(x)*math.cos(omega*t)
    >>> r.var_names
    ('omega', 't', 'x')
"""
from __future__ import annotations

import math as _math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import sympy as sp
from sympy.printing.pycode import pycode
from sympy.printing.numpy import NumPyPrinter

from .analyze import analyze
from .profile import PfaffianProfile

__all__ = [
    "TranspileResult",
    "eml_tree_to_python",
    "eml_tree_to_numpy",
    "eml_tree_to_sympy",
    "eml_tree_to_c",
]


@dataclass(frozen=True)
class TranspileResult:
    """One transpilation output ready for an agent sandbox."""

    target: str
    """One of ``"python"``, ``"numpy"``, ``"sympy"``, ``"c"``."""

    function_name: str
    """The symbol the generated code defines."""

    var_names: tuple[str, ...]
    """Free symbols of the expression, in argument order."""

    imports: str
    """Block of import / include statements to prepend."""

    code: str
    """The generated function definition (or expression-recreation
    block for the ``"sympy"`` target)."""

    verification: str
    """An assertion snippet that the agent can run to numerically
    verify the transpilation against a deterministic sample point.
    Empty string when the expression has no float-evaluable form
    (e.g. integer-only constants, or expressions that diverge at
    the sample point)."""

    sample_point: dict[str, float] = field(default_factory=dict)
    """The (x_i, value_i) point used to construct ``verification``."""

    expected_value: Optional[float] = None
    """The reference float value at ``sample_point`` (None when
    ``verification`` is empty)."""

    # EML profile of the transpiled expression. Computed once and
    # surfaced here so agents do not have to re-call analyze().
    chain_order: int = 0
    node_count: int = 0
    cost_class: str = ""

    def full_source(self) -> str:
        """Concatenated imports + code + verification, ready to ``exec``."""
        parts = [self.imports.rstrip(), "", self.code.rstrip()]
        if self.verification:
            parts.extend(["", self.verification.rstrip()])
        return "\n".join(p for p in parts if p is not None) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers


def _ordered_var_names(
    expr: sp.Expr,
    var_names: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Return argument names: user override else alphabetical free symbols."""
    if var_names is not None:
        return tuple(var_names)
    syms = sorted(expr.free_symbols, key=lambda s: str(s))
    return tuple(str(s) for s in syms)


def _sample_point(var_names: Sequence[str]) -> dict[str, float]:
    """A deterministic float sample point for verification.

    Picks small positive distinct values per variable. ``x=1.0``,
    next variable ``1.5``, then ``2.0``, etc. Avoids 0 (spurious
    success on multiplicative expressions) and integers shared
    across variables (cancellations).
    """
    return {name: 1.0 + 0.5 * i for i, name in enumerate(var_names)}


def _profile(expr: sp.Expr) -> tuple[int, int, str]:
    """Return ``(chain_order, node_count, cost_class)`` for ``expr``.

    Node count via :func:`sympy.count_ops` (matches the regularizer);
    chain order via :func:`analyze`; cost class via
    :meth:`PfaffianProfile.from_expression`. Best-effort — returns
    zeros / empty string when SymPy can't analyse the expression.
    """
    try:
        chain = int(analyze(expr).pfaffian_r)
    except Exception:  # noqa: BLE001
        chain = 0
    try:
        nodes = int(sp.count_ops(expr))
    except Exception:  # noqa: BLE001
        nodes = 0
    try:
        klass = str(PfaffianProfile.from_expression(expr).cost_class)
    except Exception:  # noqa: BLE001
        klass = ""
    return (chain, nodes, klass)


def _evaluate_at(expr: sp.Expr, point: dict[str, float]) -> Optional[float]:
    """Numerically evaluate ``expr`` at ``point``. None on divergence."""
    if not point:
        try:
            val = float(expr.evalf())
            if not _math.isfinite(val):
                return None
            return val
        except (TypeError, ValueError):
            return None
    subs = {sp.Symbol(name): val for name, val in point.items()}
    try:
        val = float(expr.subs(subs).evalf())
        if not _math.isfinite(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def _format_arg_call(var_names: Sequence[str], point: dict[str, float]) -> str:
    """``f(x=1.0, t=1.5, omega=2.0)`` → positional ``f(2.0, 1.5, 1.0)``.

    var_names is the argument order; render the call with those values
    in that order.
    """
    return ", ".join(f"{point[name]!r}" for name in var_names)


def _isclose_assert(
    function_name: str,
    var_names: Sequence[str],
    point: dict[str, float],
    expected: float,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
    use_numpy: bool = False,
) -> str:
    """Generate an assertion checking ``f(...) ≈ expected`` at the point."""
    args = _format_arg_call(var_names, point)
    if use_numpy:
        return (
            f"import numpy as _np\n"
            f"assert _np.isclose("
            f"{function_name}({args}), {expected!r}, "
            f"atol={abs_tol}, rtol={rel_tol}"
            f"), 'transpile verification failed for {function_name}'"
        )
    return (
        f"import math as _math\n"
        f"assert _math.isclose("
        f"{function_name}({args}), {expected!r}, "
        f"abs_tol={abs_tol}, rel_tol={rel_tol}"
        f"), 'transpile verification failed for {function_name}'"
    )


# ---------------------------------------------------------------------------
# Public API


def eml_tree_to_python(
    expr: sp.Expr,
    *,
    function_name: str = "f",
    var_names: Optional[Sequence[str]] = None,
) -> TranspileResult:
    """Transpile a SymPy expression to a Python function using ``math.*``."""
    args = _ordered_var_names(expr, var_names)
    body = pycode(expr, fully_qualified_modules=True)

    if args:
        signature = f"def {function_name}({', '.join(args)}):"
    else:
        signature = f"def {function_name}():"

    code = f"{signature}\n    return {body}\n"
    imports = "import math"

    point = _sample_point(args)
    expected = _evaluate_at(expr, point)
    if expected is not None:
        verification = _isclose_assert(function_name, args, point, expected)
    else:
        verification = ""

    chain, nodes, klass = _profile(expr)
    return TranspileResult(
        target="python",
        function_name=function_name,
        var_names=args,
        imports=imports,
        code=code,
        verification=verification,
        sample_point=point,
        expected_value=expected,
        chain_order=chain,
        node_count=nodes,
        cost_class=klass,
    )


def eml_tree_to_numpy(
    expr: sp.Expr,
    *,
    function_name: str = "f",
    var_names: Optional[Sequence[str]] = None,
) -> TranspileResult:
    """Transpile to a NumPy-vectorised function (``np.*`` calls).

    The generated function works on scalars OR ndarray inputs of
    matching shape — that's the point of using NumPy.
    """
    args = _ordered_var_names(expr, var_names)
    body = NumPyPrinter().doprint(expr)
    body = re.sub(r"\bnumpy\.", "np.", body)

    if args:
        signature = f"def {function_name}({', '.join(args)}):"
    else:
        signature = f"def {function_name}():"

    code = f"{signature}\n    return {body}\n"
    imports = "import numpy as np"

    point = _sample_point(args)
    expected = _evaluate_at(expr, point)
    if expected is not None:
        verification = _isclose_assert(
            function_name, args, point, expected, use_numpy=True
        )
    else:
        verification = ""

    chain, nodes, klass = _profile(expr)
    return TranspileResult(
        target="numpy",
        function_name=function_name,
        var_names=args,
        imports=imports,
        code=code,
        verification=verification,
        sample_point=point,
        expected_value=expected,
        chain_order=chain,
        node_count=nodes,
        cost_class=klass,
    )


def eml_tree_to_sympy(
    expr: sp.Expr,
    *,
    function_name: str = "expr",
    var_names: Optional[Sequence[str]] = None,
) -> TranspileResult:
    """Transpile to symbolic SymPy code that recreates the expression.

    Unlike the other targets, this output is an expression-recreation
    block (not a callable). It produces a SymPy expression bound to
    ``function_name`` that downstream code can differentiate, integrate,
    series-expand, or substitute. Free symbols are declared as locals
    so the agent can reference them directly (``expr.diff(x)``).
    """
    args = _ordered_var_names(expr, var_names)

    sym_decls = [f"{name} = sp.Symbol({name!r})" for name in args]
    decl_block = "\n".join(sym_decls)

    body = _sympy_natural_form(expr, args, function_name)

    if decl_block:
        code = decl_block + "\n" + body + "\n"
    else:
        code = body + "\n"
    imports = "import sympy as sp"

    point = _sample_point(args)
    expected = _evaluate_at(expr, point)
    if expected is not None:
        subs_args = ", ".join(f"{name}: {point[name]!r}" for name in args)
        verification = (
            f"import math as _math\n"
            f"_val = float({function_name}.subs({{{subs_args}}}).evalf())\n"
            f"assert _math.isclose(_val, {expected!r}, "
            f"abs_tol=1e-9, rel_tol=1e-9), "
            f"'transpile verification failed for {function_name}'"
        )
    else:
        verification = ""

    chain, nodes, klass = _profile(expr)
    return TranspileResult(
        target="sympy",
        function_name=function_name,
        var_names=args,
        imports=imports,
        code=code,
        verification=verification,
        sample_point=point,
        expected_value=expected,
        chain_order=chain,
        node_count=nodes,
        cost_class=klass,
    )


def eml_tree_to_c(
    expr: sp.Expr,
    *,
    function_name: str = "f",
    var_names: Optional[Sequence[str]] = None,
) -> TranspileResult:
    """Transpile to C using ``<math.h>``.

    The future Monogate Extractor product (Product 1) will replace
    ``<math.h>`` calls with libmonogate.h equivalents derived from
    the Lean→C verified extraction layer. For now the generated C
    is plain ISO-C99 and compiles against any standard libm.
    """
    args = _ordered_var_names(expr, var_names)
    body = sp.ccode(expr)

    if args:
        params = ", ".join(f"double {name}" for name in args)
        signature = f"double {function_name}({params})"
    else:
        signature = f"double {function_name}(void)"

    code = (
        f"{signature} {{\n"
        f"    return {body};\n"
        f"}}\n"
    )
    imports = (
        "#include <math.h>\n"
        "/* Future: replace <math.h> with libmonogate.h "
        "when the Monogate Extractor product ships. */"
    )

    point = _sample_point(args)
    expected = _evaluate_at(expr, point)
    if expected is not None:
        # Cannot exec C; emit a comment recording expected value.
        args_repr = ", ".join(f"{name}={point[name]}" for name in args)
        verification = (
            f"/* Reference value at {args_repr or '(void)'}: "
            f"{function_name}() = {expected!r} (within 1e-9 abs / rel) */"
        )
    else:
        verification = ""

    chain, nodes, klass = _profile(expr)
    return TranspileResult(
        target="c",
        function_name=function_name,
        var_names=args,
        imports=imports,
        code=code,
        verification=verification,
        sample_point=point,
        expected_value=expected,
        chain_order=chain,
        node_count=nodes,
        cost_class=klass,
    )


# ---------------------------------------------------------------------------


def _sympy_natural_form(
    expr: sp.Expr,
    locals_: Sequence[str],
    function_name: str,
) -> str:
    """Render ``expr`` as a runnable assignment using natural SymPy syntax.

    Uses ``str(expr)`` (which prints the natural ``exp(x)*cos(omega*t)``
    form rather than the constructor form ``Mul(exp(...), cos(...))``)
    and prefixes function-call tokens with ``sp.`` so the resulting
    line works with only ``import sympy as sp``. Tokens that match
    declared local symbol names are left alone — they refer to the
    Symbol locals declared earlier in the recreation block.
    """
    src = str(expr)
    locals_set = set(locals_)
    pattern = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\(")

    def _replace(m: re.Match[str]) -> str:
        ident = m.group(1)
        if ident in locals_set or ident == "sp":
            return m.group(0)
        return f"sp.{ident}("

    body = pattern.sub(_replace, src)
    return f"{function_name} = {body}"
