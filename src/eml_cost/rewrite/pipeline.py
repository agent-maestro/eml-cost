"""eml_rewrite.pipeline — high-level rewrite pipeline.

Combines the existing pattern-matching primitives (suggest/best) with
eml-cost canonicalize() + verify_equivalence into a single
``rewrite(expr)`` entry point.

Created in E-187 (2026-04-26).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sympy as sp

from .core import (
    Suggestion,
    best,
    score,
    suggest,
    verify_equivalence,
)


@dataclass(frozen=True)
class CanonicalReport:
    """Report from to_canonical()."""

    original: sp.Basic
    canonical: sp.Basic
    rules_fired: tuple[str, ...]
    cost_before: int
    cost_after: int
    changed: bool

    def __repr__(self) -> str:
        if self.changed:
            return (f"CanonicalReport({self.original!s} -> {self.canonical!s}, "
                    f"cost {self.cost_before} -> {self.cost_after}, "
                    f"rules: {len(self.rules_fired)})")
        return f"CanonicalReport({self.original!s}, no change)"


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of rewrite(expr)."""

    original: sp.Basic
    rewritten: sp.Basic
    cost_before: int
    cost_after: int
    savings_pct: float
    steps: tuple[str, ...]
    suggestions_considered: int
    verified_equivalent: bool

    def __repr__(self) -> str:
        return (f"RewriteResult({self.original!s} -> {self.rewritten!s}, "
                f"cost {self.cost_before} -> {self.cost_after} "
                f"({self.savings_pct:+.0f}%), {len(self.steps)} steps)")


def _canonicalize_via_eml_cost(expr: sp.Basic) -> tuple[sp.Basic, list[str]]:
    """Run eml_cost.canonicalize and report which rules fired."""
    try:
        from eml_cost import canonicalize as _eml_canon
    except ImportError:
        # eml-cost not installed — return identity
        return expr, []

    rules = []
    canonical = _eml_canon(expr)

    # Heuristic rule-firing detection — compare structures.
    if str(canonical) != str(expr):
        # Detect specific rewrites by inspecting the diff
        s_before = str(expr)
        s_after = str(canonical)
        if "tanh" in s_before and "tanh" not in s_after:
            rules.append("tanh -> exp form")
        elif "exp" in s_after and "tanh" in s_before:
            rules.append("tanh -> exp form")
        if "1 - 1/" in s_before or "-1/" in s_before:
            if "1/(1 +" in s_after or "exp(-" in s_after:
                rules.append("complement -> sigmoid form")
        if s_before.count("log") > s_after.count("log"):
            rules.append("logcombine: log(a) - log(b) -> log(a/b)")
        if s_before.count("exp") > s_after.count("exp"):
            rules.append("expcombine: exp(a)*exp(b) -> exp(a+b)")
        if not rules:
            rules.append("structural rewrite (sympy together/factor_terms)")

    return canonical, rules


def to_canonical(expr: sp.Basic) -> CanonicalReport:
    """Rewrite expr to its canonical form for stable cost-class analysis.

    Wraps :func:`eml_cost.canonicalize`. Returns a report documenting
    which rewrites fired and the cost change.

    Examples::

        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> r = to_canonical(1 - 1/(1 + sp.exp(x)))
        >>> r.changed
        True
        >>> r.canonical
        exp(x)/(exp(x) + 1)
    """
    cost_before = score(expr)
    canonical, rules = _canonicalize_via_eml_cost(expr)
    cost_after = score(canonical)
    return CanonicalReport(
        original=expr,
        canonical=canonical,
        rules_fired=tuple(rules),
        cost_before=cost_before,
        cost_after=cost_after,
        changed=str(expr) != str(canonical),
    )


def rewrite(expr: sp.Basic, strategy: str = "optimal") -> RewriteResult:
    """Rewrite a SymPy expression to minimize EML cost.

    Strategies:
        - ``canonical``: apply canonicalize() only.
        - ``optimal`` (default): canonicalize + best() pattern routing.
        - ``aggressive``: canonical + best + suggest()-driven rewrites.

    Returns a :class:`RewriteResult` with original, rewritten, cost
    before/after, steps applied, and numerical equivalence verification.

    Every rewrite is verified for numerical equivalence on random
    sample points; if verification fails, the rewrite is rejected and
    the original is preserved.

    Examples::

        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> r = rewrite(sp.exp(x) / (sp.exp(x) + 1))
        >>> r.cost_before, r.cost_after
        (3, 2)
    """
    if strategy not in ("canonical", "optimal", "aggressive"):
        raise ValueError(
            f"rewrite: unknown strategy {strategy!r}. "
            f"Choose one of: 'canonical', 'optimal', 'aggressive'."
        )

    steps: list[str] = []
    cost_before = score(expr)
    current = expr
    suggestions_considered = 0

    # Step 1: canonicalize
    canon = to_canonical(current)
    if canon.changed:
        # Verify before accepting
        if verify_equivalence(current, canon.canonical):
            current = canon.canonical
            steps.append(f"canonicalize: {len(canon.rules_fired)} rules "
                          f"({', '.join(canon.rules_fired)})")
        else:
            steps.append("canonicalize: rejected (failed verification)")

    if strategy == "canonical":
        cost_after = score(current)
        return RewriteResult(
            original=expr, rewritten=current,
            cost_before=cost_before, cost_after=cost_after,
            savings_pct=_savings(cost_before, cost_after),
            steps=tuple(steps),
            suggestions_considered=0,
            verified_equivalent=verify_equivalence(expr, current) if str(expr) != str(current) else True,
        )

    # Step 2: best() pattern routing
    try:
        b = best(current)
        if str(b) != str(current):
            if verify_equivalence(current, b):
                old_cost = score(current)
                new_cost = score(b)
                if new_cost < old_cost:
                    steps.append(f"best() routing: cost {old_cost} -> {new_cost}")
                    current = b
                else:
                    steps.append(f"best() proposed but not lower-cost (rejected)")
            else:
                steps.append("best() proposed but failed verification (rejected)")
    except Exception as e:
        steps.append(f"best() error (skipped): {type(e).__name__}")

    # Step 3 (aggressive only): consider all suggestions
    if strategy == "aggressive":
        try:
            sugs = suggest(current)
            suggestions_considered = len(sugs)
            for s in sugs:
                cur_cost = score(current)
                rew_cost = score(s.rewritten)
                if rew_cost < cur_cost and verify_equivalence(current, s.rewritten):
                    current = s.rewritten
                    steps.append(f"suggestion '{s.pattern_name}': cost {cur_cost} -> {rew_cost}")
        except Exception as e:
            steps.append(f"suggest() error (skipped): {type(e).__name__}")

    cost_after = score(current)
    return RewriteResult(
        original=expr, rewritten=current,
        cost_before=cost_before, cost_after=cost_after,
        savings_pct=_savings(cost_before, cost_after),
        steps=tuple(steps),
        suggestions_considered=suggestions_considered,
        verified_equivalent=(verify_equivalence(expr, current) if str(expr) != str(current) else True),
    )


def _savings(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return 100.0 * (before - after) / before
