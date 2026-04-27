"""Decompose a symbolic expression into independent dynamical phenomena.

The heuristic counts:
  - **oscillation modes**: distinct ``sin``/``cos`` arguments (each
    contributes a {sin, cos} chain pair, Pfaffian chain order +2)
  - **decay modes**: distinct ``exp`` arguments (each contributes an
    exp chain element, Pfaffian chain order +1)
  - **static components**: polynomial / algebraic structure
    (no Pfaffian chain contribution)

It then predicts ``r`` via the slope-2 rule
``r ≈ 2 * n_osc + 1 * n_decay`` and compares against the actual
``pfaffian_r`` from :func:`analyze`.

Evidence
--------

E-196 Phase 4f cross-domain study (n=175 paired rows across 12
subdomains, 2026-04-27):

  - **Spearman ρ = +0.890**, p = 5.18e-61
  - **Pearson r = +0.929**, p = 7.91e-77
  - Per-domain ρ ranges 0.46 (olfactory) to 1.00 (crypto).

The slope-2 rule is calibrated for OSCILLATORY expressions where each
mode contributes a {sin, cos} chain pair. It UNDER-predicts for
sqrt-heavy and parallel-Gaussian expressions where Pfaffian's
DAG-summed chain count exceeds the simple per-mode contribution. The
returned ``confidence`` reflects this domain dependence.

Honest framing
--------------

  - **OBSERVATION** tier: ρ=0.890 is empirical, not a theorem.
  - The heuristic does NOT identify *which* domain the expression
    comes from — only the count of modes per category.
  - For PNE expressions (Bessel, Airy, Gamma, etc.) the counter
    abstains via low confidence.

Source
------

  - ``monogate-research/exploration/E196_algorithmic_corpus/
    phase4f_physical_phenomena.py`` — original tagging pipeline.
  - ``monogate-research/exploration/color_science_subdomain/
    phase4f_extended_v5_summary.json`` — ρ=0.890 final fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import sympy as sp

from .analyze import analyze

__all__ = ["DynamicsProfile", "analyze_dynamics"]


@dataclass(frozen=True)
class DynamicsProfile:
    """Output of :func:`analyze_dynamics`.

    Attributes
    ----------
    n_oscillations:
        Count of distinct sin/cos arguments. Each oscillation mode
        contributes a {sin, cos} Pfaffian chain pair.
    n_decays:
        Count of distinct exp arguments. Each decay mode contributes
        a single exp chain element.
    n_static:
        Indicator (0/1) for the presence of non-trivial polynomial /
        algebraic structure. Static components contribute zero to the
        Pfaffian chain order in the slope-2 rule.
    predicted_r:
        ``2 * n_oscillations + 1 * n_decays`` — the slope-2 rule's
        prediction of ``pfaffian_r``.
    actual_r:
        ``pfaffian_r`` returned by :func:`analyze`. Differs from
        ``predicted_r`` when DAG-summing across parallel branches
        exceeds the simple per-mode contribution.
    confidence:
        ``"high"`` for oscillation-heavy expressions or polynomial-
        only expressions (these are the regions where ρ > 0.9 in
        the validation corpus); ``"moderate"`` for decay-heavy or
        mixed; ``"low"`` for PNE expressions (Bessel, Airy, Gamma,
        Lambert W) where the chain-count rule does not apply.
    description:
        Human-readable summary of the mode decomposition.
    domain_fit:
        ``"strong"`` (predicted_r matches actual_r within ±1),
        ``"moderate"`` (within ±3), or ``"weak"`` (off by more).
    honest_note:
        Reminder that this is an OBSERVATION-tier counter and that
        the slope-2 rule is calibrated for oscillation-heavy domains.
    """

    n_oscillations: int
    n_decays: int
    n_static: int
    predicted_r: int
    actual_r: int
    confidence: str
    description: str
    domain_fit: str
    honest_note: str = (
        "OBSERVATION-tier counter. The slope-2 rule "
        "r ≈ 2*n_osc + 1*n_decay is calibrated for oscillation-heavy "
        "expressions (Spearman ρ ≈ 0.890 across 175 expressions in 12 "
        "subdomains). It under-predicts for parallel-Gaussian and "
        "sqrt-heavy expressions where Pfaffian's DAG-summed chain "
        "count exceeds the simple per-mode contribution."
    )


def _collect_distinct_args(
    expr: sp.Basic, op_class: type
) -> set[sp.Expr]:
    """Walk ``expr`` and collect distinct arguments of ``op_class`` calls.

    Two arguments are considered the same if their canonical SymPy
    forms are equal (``a.equals(b)`` is too slow; we use SymPy's
    structural hash via dict deduplication).
    """
    seen: dict[str, sp.Expr] = {}
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, op_class):
            arg = sub.args[0]
            key = sp.srepr(arg)
            if key not in seen:
                seen[key] = arg
    return set(seen.values())


def _has_static_structure(expr: sp.Basic) -> bool:
    """True if ``expr`` contains non-trivial polynomial / algebraic
    structure (Add, Mul with multiple operands, Pow with non-trivial
    exponent, sqrt, log)."""
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Add) and len(sub.args) >= 2:
            return True
        if isinstance(sub, sp.Mul) and len(sub.args) >= 2:
            return True
        if isinstance(sub, sp.Pow):
            base, exponent = sub.args
            if exponent != 1 and not (
                isinstance(exponent, sp.Integer) and exponent == 0
            ):
                return True
        if isinstance(sub, sp.log):
            return True
    return False


def analyze_dynamics(expr: Union[sp.Expr, str]) -> DynamicsProfile:
    """Decompose an expression into oscillation / decay / static modes.

    Parameters
    ----------
    expr:
        A SymPy expression or a string parseable by :func:`sympy.sympify`.

    Returns
    -------
    DynamicsProfile
        The mode decomposition with confidence and description.

    Examples
    --------
    >>> from eml_cost import analyze_dynamics
    >>> p = analyze_dynamics("A * cos(omega * t)")
    >>> p.n_oscillations, p.n_decays
    (1, 0)
    >>> p.predicted_r
    2

    >>> p = analyze_dynamics("A * exp(-zeta * omega * t) * cos(omega_d * t)")
    >>> p.n_oscillations, p.n_decays
    (1, 1)
    >>> p.predicted_r
    3
    """
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)

    osc_args = _collect_distinct_args(expr, sp.cos)
    osc_args |= _collect_distinct_args(expr, sp.sin)
    # Coalesce sin(g) and cos(g) sharing the same g into a single mode.
    coalesced: dict[str, sp.Expr] = {}
    for a in osc_args:
        coalesced[sp.srepr(a)] = a
    n_osc = len(coalesced)

    decay_args = _collect_distinct_args(expr, sp.exp)
    n_decay = len(decay_args)

    n_static = 1 if _has_static_structure(expr) else 0

    actual = analyze(expr)
    actual_r = int(actual.pfaffian_r)

    predicted_r = 2 * n_osc + 1 * n_decay

    if actual.is_pfaffian_not_eml:
        confidence = "low"
    elif n_osc > 0:
        confidence = "high"
    elif n_decay > 0:
        confidence = "moderate"
    else:
        confidence = "high"

    diff = abs(predicted_r - actual_r)
    if diff <= 1:
        domain_fit = "strong"
    elif diff <= 3:
        domain_fit = "moderate"
    else:
        domain_fit = "weak"

    parts: list[str] = []
    if n_osc:
        parts.append(
            f"{n_osc} oscillation mode" + ("" if n_osc == 1 else "s")
        )
    if n_decay:
        parts.append(
            f"{n_decay} decay mode" + ("" if n_decay == 1 else "s")
        )
    if actual.is_pfaffian_not_eml and not (n_osc or n_decay):
        parts.append("Pfaffian-not-elementary primitive")
    elif actual.is_pfaffian_not_eml:
        parts.append("plus Pfaffian-not-elementary primitive")
    elif n_static and not (n_osc or n_decay):
        parts.append("polynomial / algebraic only")
    if not parts:
        parts.append("constant or terminal")
    description = ", ".join(parts) + (
        f" (predicted r={predicted_r}, actual r={actual_r})"
    )

    return DynamicsProfile(
        n_oscillations=n_osc,
        n_decays=n_decay,
        n_static=n_static,
        predicted_r=predicted_r,
        actual_r=actual_r,
        confidence=confidence,
        description=description,
        domain_fit=domain_fit,
    )
