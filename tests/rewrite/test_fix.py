"""Tests for @costlimit_or_fix and CostFixFailed."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost import CostLimitExceeded, analyze
from eml_cost.rewrite import CostFixFailed, costlimit_or_fix
from eml_cost.rewrite.path import Step


def test_under_budget_passes_through():
    """Function returning an under-budget expression is unchanged."""
    @costlimit_or_fix(predicted_depth=5)
    def f(x: sp.Basic) -> sp.Basic:
        return sp.exp(x)   # cost = 1

    x = sp.Symbol("x")
    out = f(x)
    assert out == sp.exp(x)


def test_over_budget_with_known_fix_substitutes_silently():
    """Textbook sigmoid (cost=3) under predicted_depth=2 — must rewrite
    to the canonical sigmoid (cost=2) silently."""
    @costlimit_or_fix(predicted_depth=2)
    def textbook_sigmoid(x: sp.Basic) -> sp.Basic:
        return sp.exp(x) / (1 + sp.exp(x))

    x = sp.Symbol("x")
    out = textbook_sigmoid(x)
    assert out == 1 / (1 + sp.exp(-x))
    # Verify the substitution is symbolically equivalent.
    diff = sp.simplify(out - sp.exp(x) / (1 + sp.exp(x)))
    assert diff == 0
    # Verify the fixed result satisfies the constraint.
    assert analyze(out).predicted_depth <= 2


def test_over_budget_with_no_fix_raises_with_trail():
    """Deeply nested exp/sin chain has no rewrites — must raise
    CostFixFailed (a CostLimitExceeded) with search_trail attached."""
    @costlimit_or_fix(predicted_depth=2, max_search_steps=8)
    def f(x: sp.Basic) -> sp.Basic:
        return sp.exp(sp.exp(sp.exp(sp.sin(x))))   # cost ~6, no patterns match

    x = sp.Symbol("x")
    with pytest.raises(CostFixFailed) as exc_info:
        f(x)

    err = exc_info.value
    assert isinstance(err, CostLimitExceeded)   # subclass relationship
    assert err.axis == "predicted_depth"
    assert err.measured > 2
    assert err.limit == 2
    assert isinstance(err.search_trail, list)
    assert len(err.search_trail) >= 1
    assert isinstance(err.search_trail[0], Step)


def test_non_sympy_return_passes_through():
    """Decorator is safe on functions that conditionally return non-Basic."""
    @costlimit_or_fix(predicted_depth=2)
    def f(toggle: bool) -> object:
        if toggle:
            return "string-result"
        return sp.exp(sp.Symbol("x"))

    assert f(True) == "string-result"


def test_no_axis_raises_at_decoration_time():
    with pytest.raises(ValueError):
        @costlimit_or_fix()                         # type: ignore[call-overload]
        def _f(x: sp.Basic) -> sp.Basic:
            return x


def test_pythagorean_fix_collapses_to_one():
    """sin^2 + cos^2 (cost ~5) → 1 (cost 0) under predicted_depth=2."""
    @costlimit_or_fix(predicted_depth=2)
    def pythagorean(x: sp.Basic) -> sp.Basic:
        return sp.sin(x)**2 + sp.cos(x)**2

    x = sp.Symbol("x")
    out = pythagorean(x)
    assert out == sp.S.One


def test_fix_preserves_equivalence_for_tanh():
    """sinh(x)/cosh(x) (cost=3) → tanh(x) (cost=1) under predicted_depth=2."""
    @costlimit_or_fix(predicted_depth=2)
    def f(x: sp.Basic) -> sp.Basic:
        return sp.sinh(x) / sp.cosh(x)

    x = sp.Symbol("x")
    out = f(x)
    assert out == sp.tanh(x)
    # Substitution is sound:
    diff = sp.simplify(out - sp.sinh(x) / sp.cosh(x))
    assert diff == 0


def test_fix_failure_trail_captures_best_explored_node():
    """When no fix is found, trail's last step is the lowest-cost
    node the search visited (not necessarily the start)."""
    # An expression with a partial-rewrite available but not enough
    # to clear the limit. Use multiplied sigmoid: textbook_sigmoid * x.
    # textbook_sigmoid alone has cost 3; * x raises depth.
    @costlimit_or_fix(predicted_depth=1)   # impossibly tight
    def f(x: sp.Basic) -> sp.Basic:
        return sp.exp(x) / (1 + sp.exp(x)) * sp.sin(x)

    x = sp.Symbol("x")
    with pytest.raises(CostFixFailed) as exc_info:
        f(x)
    trail = exc_info.value.search_trail
    # Trail walks down (or stays) — costs are non-increasing along it.
    costs = [step.cost for step in trail]
    assert costs == sorted(costs, reverse=True) or len(set(costs)) == 1
