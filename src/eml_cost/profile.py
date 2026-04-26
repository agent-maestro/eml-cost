"""PfaffianProfile — rich dataclass wrapping analyze() output with
distance metric, comparison, and serialization helpers.

Created in E-183 (2026-04-26).

Backward compatibility: AnalyzeResult still exists and analyze() still
returns it. PfaffianProfile is a NEW richer wrapper that you can build
via PfaffianProfile.from_expression(expr) or
PfaffianProfile.from_analysis(analyze_result, expr).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import sympy as sp

from .analyze import AnalyzeResult, analyze
from .canonicalize import canonicalize


# Default weights for the cost-class distance metric.
# Justified empirically: chain order (r) is the strongest invariant
# of the Pfaffian classification — different r means structurally
# different mathematical objects. Width is next (parallel chains
# split into independent subproblems). Depth (d) is partly derivable
# from r + corrections, so it carries less independent information.
# Oscillation correction (c) is binary and small.
DEFAULT_WEIGHTS = {
    "r": 4.0,
    "d": 1.0,
    "w": 2.0,
    "c": 1.0,
}


@dataclass(frozen=True)
class PfaffianProfile:
    """Rich Pfaffian profile of a symbolic expression.

    Wraps :class:`AnalyzeResult` with comparison + distance methods.

    Attributes
    ----------
    r : int
        Pfaffian chain order along the deepest path (``max_path_r``).
    degree : int
        EML routing tree depth (``eml_depth``).
    width : int
        Number of independent parallel chains. Approximated from
        Add/Mul disjoint-variable groups; defaults to ``max_path_r``-
        derived heuristic when the analyzer doesn't supply it.
    cost_class : str
        Canonical axes string ``"p{r}-d{degree}-w{width}-c{c_osc}"``.
    oscillatory : bool
        ``True`` if the ``c_osc`` correction was applied (sin/cos in
        the deepest path).
    corrections : dict
        Per-correction values: ``{"c_osc": int, "c_composite": int,
        "delta_fused": int}``.
    expression : str
        Original expression string.
    canonical_form : str
        Canonicalized expression string (output of ``canonicalize``).
    is_pfaffian_not_eml : bool
        ``True`` if the expression contains a Pfaffian-but-not-EML
        primitive (Bessel, Airy, Lambert W, hypergeometric, etc.).
    """

    r: int
    degree: int
    width: int
    cost_class: str
    oscillatory: bool
    corrections: dict
    expression: str
    canonical_form: str
    is_pfaffian_not_eml: bool

    def __repr__(self) -> str:
        return f"PfaffianProfile({self.cost_class})"

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_expression(cls, expr, *, do_canonicalize: bool = True) -> "PfaffianProfile":
        """Build a profile from a sympy expression (or string).

        Parameters
        ----------
        expr : sympy.Basic or str
            The expression to profile.
        do_canonicalize : bool
            If ``True`` (default), runs :func:`canonicalize` first.
        """
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        original_str = str(expr)
        canonical = canonicalize(expr) if do_canonicalize else expr
        canonical_str = str(canonical)
        result = analyze(canonical)
        return cls.from_analysis(result, expression_str=original_str,
                                  canonical_str=canonical_str)

    @classmethod
    def from_analysis(cls, result: AnalyzeResult, *,
                       expression_str: Optional[str] = None,
                       canonical_str: Optional[str] = None) -> "PfaffianProfile":
        """Build a profile from an existing AnalyzeResult."""
        r = result.max_path_r
        d = result.eml_depth
        # Width: best-effort. The AnalyzeResult doesn't expose width
        # directly, but we can derive a stable proxy from the
        # corrections + chain structure.
        width = _estimate_width(result.expression)
        c_osc = result.corrections.c_osc
        cost_class = f"p{r}-d{d}-w{width}-c{c_osc}"
        return cls(
            r=r,
            degree=d,
            width=width,
            cost_class=cost_class,
            oscillatory=(c_osc > 0),
            corrections={
                "c_osc": result.corrections.c_osc,
                "c_composite": result.corrections.c_composite,
                "delta_fused": result.corrections.delta_fused,
            },
            expression=expression_str if expression_str is not None else str(result.expression),
            canonical_form=canonical_str if canonical_str is not None else str(result.expression),
            is_pfaffian_not_eml=result.is_pfaffian_not_eml,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def distance(self, other: "PfaffianProfile",
                 weights: Optional[dict] = None) -> float:
        """Weighted Euclidean distance in (r, d, w, c) coordinate space.

        This IS a metric (proven in tests/test_profile_metric.py):
          d(a, a) = 0
          d(a, b) = d(b, a)
          d(a, c) <= d(a, b) + d(b, c)

        Default weights: r=4, d=1, w=2, c=1 (chain order dominates).
        Pass a ``weights`` dict to customize.
        """
        w = weights or DEFAULT_WEIGHTS
        d_r = self.r - other.r
        d_d = self.degree - other.degree
        d_w = self.width - other.width
        d_c = self.corrections["c_osc"] - other.corrections["c_osc"]
        return math.sqrt(
            w["r"] * d_r * d_r
            + w["d"] * d_d * d_d
            + w["w"] * d_w * d_w
            + w["c"] * d_c * d_c
        )

    def compare(self, other: "PfaffianProfile") -> dict:
        """Per-axis delta + total distance."""
        return {
            "delta_r": self.r - other.r,
            "delta_degree": self.degree - other.degree,
            "delta_width": self.width - other.width,
            "delta_c_osc": self.corrections["c_osc"] - other.corrections["c_osc"],
            "distance": self.distance(other),
            "same_class": self.cost_class == other.cost_class,
        }

    def is_elementary(self) -> bool:
        """``True`` if the expression is EML-expressible (not Pfaffian-not-EML)."""
        return not self.is_pfaffian_not_eml

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-friendly dict representation."""
        return asdict(self)

    def to_row(self) -> list:
        """Flat list for CSV output."""
        return [
            self.expression,
            self.canonical_form,
            self.cost_class,
            self.r,
            self.degree,
            self.width,
            self.corrections["c_osc"],
            self.corrections["c_composite"],
            self.corrections["delta_fused"],
            self.oscillatory,
            self.is_pfaffian_not_eml,
        ]

    @staticmethod
    def csv_header() -> list[str]:
        """Header matching to_row() output order."""
        return [
            "expression",
            "canonical_form",
            "cost_class",
            "r",
            "degree",
            "width",
            "c_osc",
            "c_composite",
            "delta_fused",
            "oscillatory",
            "is_pfaffian_not_eml",
        ]


def _estimate_width(expr: sp.Basic) -> int:
    """Estimate Pfaffian width: count independent parallel chain groups.

    At each Add/Mul node, group children with overlapping free variables
    (chains share variables → same group). Number of independent groups
    = local width. Recurse to find the maximum.
    """
    if not isinstance(expr, sp.Basic) or expr.is_Atom:
        return 0
    func = expr.func
    if func is sp.Add or func is sp.Mul:
        children = list(expr.args)
        # Find children with chain elements (sin, cos, exp, log, sqrt, etc.)
        chain_children = [c for c in children if _has_chain_element(c)]
        if not chain_children:
            return max((_estimate_width(c) for c in children), default=0)
        # Group by overlapping free vars
        groups = []
        for child in chain_children:
            vars_ = frozenset(child.free_symbols)
            merged = []
            for i, g in enumerate(groups):
                if vars_ & g:
                    merged.append(i)
            if not merged:
                groups.append(vars_)
            else:
                new_g = vars_
                for i in merged:
                    new_g = new_g | groups[i]
                kept = [g for i, g in enumerate(groups) if i not in merged]
                kept.append(new_g)
                groups = kept
        local = len(groups)
        return max(local, max((_estimate_width(c) for c in children), default=0))
    if expr.args:
        return max((_estimate_width(a) for a in expr.args
                    if isinstance(a, sp.Basic)), default=0)
    return 0


def _has_chain_element(expr) -> bool:
    """True if expr contains exp/log/sin/cos/tan/sqrt/etc."""
    if not isinstance(expr, sp.Basic):
        return False
    chain_funcs = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.tanh,
                    sp.atan, sp.asin, sp.acos, sp.sinh, sp.cosh)
    if any(isinstance(expr, f) for f in chain_funcs):
        return True
    if isinstance(expr, sp.Pow) and not expr.args[1].is_Integer:
        return True
    return any(_has_chain_element(a) for a in expr.args)
