"""Top-level :func:`analyze` API and :func:`measure` helper for SymPy.

The result type :class:`AnalyzeResult` is a frozen dataclass exposing every
detector output needed for downstream tools. The :func:`measure` helper is
a SymPy-compatible measure that returns an integer suitable for
``sp.simplify(expr, measure=measure)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import sympy as sp

from .core import (
    eml_depth,
    is_pfaffian_not_eml,
    max_path_r,
    pfaffian_r,
    structural_overhead,
)


__all__ = ["AnalyzeResult", "Corrections", "analyze", "measure"]


@dataclass(frozen=True)
class Corrections:
    """Per-path correction terms in the EML routing model.

    Attributes
    ----------
    c_osc:
        Number of Euler-bypassed oscillations (``sin``/``cos``) along
        the deepest path that are not absorbed by F-family fusion.
    c_composite:
        Composite-operator penalty for unfused operators along the path.
        Currently ``+1`` per ``tan`` occurrence.
    delta_fused:
        Chain stages saved by F-family fusion patterns (LEAd, sigmoid,
        tanh-as-primitive, etc.).
    """

    c_osc: int
    c_composite: int
    delta_fused: int


@dataclass(frozen=True)
class AnalyzeResult:
    """Full Pfaffian profile of a symbolic expression.

    Attributes
    ----------
    expression:
        The SymPy expression that was analyzed.
    pfaffian_r:
        Total Pfaffian chain order (Khovanskii). Sums across parallel
        branches; counts every distinct chain element in the tree.
    max_path_r:
        Pfaffian chain order along the deepest root-to-leaf path. For
        independent-variable products this is dramatically smaller than
        ``pfaffian_r``.
    eml_depth:
        EML routing tree depth (SuperBEST conventions). Reflects how the
        expression actually evaluates — Euler bypass for trig, F-family
        fusion for sigmoid/softplus/tanh.
    structural_overhead:
        Tree-structural depth contributed by Add / Mul / positive-integer
        ``Pow`` nodes along the deepest path.
    corrections:
        Path-restricted correction breakdown (``Corrections`` dataclass).
    predicted_depth:
        ``max_path_r + c_osc + c_composite − delta_fused + structural_overhead``.
    is_pfaffian_not_eml:
        ``True`` if the expression contains any Pfaffian-but-not-EML
        primitive (Bessel, Airy, Lambert W, hypergeometric).
    """

    expression: sp.Basic
    pfaffian_r: int
    max_path_r: int
    eml_depth: int
    structural_overhead: int
    corrections: Corrections
    predicted_depth: int
    is_pfaffian_not_eml: bool


# Mirror the detector-internal `corrections` recursion here so we can
# return path-restricted corrections without depending on undocumented
# private symbols downstream.
def _corrections_along_max_path(expr: sp.Basic) -> tuple[int, int, int]:
    """Walk along the deepest path, accumulating (c_osc, c_composite, delta_fused)."""
    if not isinstance(expr, sp.Basic) or expr.is_Atom:
        return (0, 0, 0)

    # F-family fusion: log(c + exp(g)) and 1/(1+exp(-g))
    if isinstance(expr, sp.log):
        inner = expr.args[0]
        if isinstance(inner, sp.Add) and len(inner.args) == 2:
            a1, a2 = inner.args
            if a1.is_constant() and isinstance(a2, sp.exp):
                sub = _corrections_along_max_path(a2.args[0])
                return (sub[0], sub[1], sub[2] + 1)
            if a2.is_constant() and isinstance(a1, sp.exp):
                sub = _corrections_along_max_path(a1.args[0])
                return (sub[0], sub[1], sub[2] + 1)

    if isinstance(expr, sp.Pow) and expr.args[1] == -1:
        inner = expr.args[0]
        if isinstance(inner, sp.Add) and len(inner.args) == 2:
            for arg in inner.args:
                if isinstance(arg, sp.exp):
                    g = arg.args[0]
                    inner_g = -g if g.could_extract_minus_sign() else g
                    sub = _corrections_along_max_path(inner_g)
                    return (sub[0], sub[1], sub[2] + 1)

    func = expr.func

    if isinstance(expr, (sp.sin, sp.cos)):
        sub = _corrections_along_max_path(expr.args[0])
        return (sub[0] + 1, sub[1], sub[2])

    if isinstance(expr, sp.tan):
        sub = _corrections_along_max_path(expr.args[0])
        return (sub[0], sub[1] + 1, sub[2])  # calibrated: +1 per tan

    if not expr.args:
        return (0, 0, 0)

    best: tuple[int, int, int] = (0, 0, 0)
    best_total = -1
    for arg in expr.args:
        if not isinstance(arg, sp.Basic):
            continue
        sub = _corrections_along_max_path(arg)
        total = sub[0] + sub[1] - sub[2]
        if total > best_total:
            best = sub
            best_total = total
    return best


def analyze(expr: Union[str, sp.Basic]) -> AnalyzeResult:
    """Analyze a symbolic expression and return its full Pfaffian profile.

    Parameters
    ----------
    expr:
        Either a SymPy expression (``sp.Basic``) or a string parseable
        by :func:`sympy.parse_expr`. Strings are passed through SymPy's
        parser; for untrusted input pre-parse with care.

    Returns
    -------
    AnalyzeResult
        Frozen dataclass with ``pfaffian_r``, ``max_path_r``,
        ``eml_depth``, ``structural_overhead``, ``corrections``,
        ``predicted_depth``, and ``is_pfaffian_not_eml``.

    Raises
    ------
    TypeError
        If ``expr`` is neither ``str`` nor ``sp.Basic``.
    ValueError
        If the string cannot be parsed.

    Examples
    --------
    >>> from eml_cost import analyze
    >>> result = analyze("exp(sin(x))")
    >>> result.pfaffian_r
    3
    >>> result.max_path_r
    3
    >>> result.eml_depth
    4
    """
    if isinstance(expr, str):
        try:
            parsed: sp.Basic = sp.parse_expr(expr, evaluate=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not parse expression: {expr!r}") from exc
    elif isinstance(expr, sp.Basic):
        parsed = expr
    else:
        raise TypeError(
            f"analyze expects str or sp.Basic, got {type(expr).__name__}"
        )

    r = pfaffian_r(parsed)
    mr = max_path_r(parsed)
    d = eml_depth(parsed)
    so = structural_overhead(parsed)
    c_osc, c_comp, d_fused = _corrections_along_max_path(parsed)
    cor = Corrections(c_osc=c_osc, c_composite=c_comp, delta_fused=d_fused)
    predicted = mr + c_osc + c_comp - d_fused + so
    non_eml = is_pfaffian_not_eml(parsed)

    return AnalyzeResult(
        expression=parsed,
        pfaffian_r=r,
        max_path_r=mr,
        eml_depth=d,
        structural_overhead=so,
        corrections=cor,
        predicted_depth=predicted,
        is_pfaffian_not_eml=non_eml,
    )


def measure(expr: Any) -> int:
    """SymPy-compatible measure returning the EML predicted depth.

    Lower is better. Suitable as a drop-in for the ``measure=`` argument
    of :func:`sympy.simplify`:

    >>> import sympy as sp
    >>> from eml_cost import measure
    >>> x = sp.Symbol("x", real=True)
    >>> sp.simplify(sp.cos(x)**2 + sp.sin(x)**2, measure=measure)
    1

    For non-``Basic`` inputs returns a large sentinel value (caller
    receives a measure-compatible integer, never raises).
    """
    if not isinstance(expr, sp.Basic):
        return 10**9
    try:
        result = analyze(expr)
        return result.predicted_depth
    except (RecursionError, ValueError, TypeError):
        return 10**9
