"""Stability tests added for the 0.1.0 stable release.

Each test pins a behaviour the user-facing API depends on. If any of
these regresses, downstream consumers (eml-rewrite, eml-cost-torch,
the SymPy ``measure=`` path) silently change predictions.
"""
from __future__ import annotations

import sympy as sp

from eml_cost import (
    PFAFFIAN_NOT_EML_R,
    analyze,
    is_pfaffian_not_eml,
    max_path_r,
    measure,
    pfaffian_r,
)


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


# ---------------------------------------------------------------------------
# Nested compositions
# ---------------------------------------------------------------------------


def test_exp_sin_x_composition_increments_chain_order() -> None:
    """exp(sin(x)) is r=3 (sin contributes r=2 oscillation plus exp)."""
    result = analyze(sp.exp(sp.sin(x)))
    assert result.pfaffian_r == 3
    assert result.max_path_r == 3
    assert result.predicted_depth == 4


def test_log_exp_exp_x_collapses_to_inner_chain() -> None:
    """log(exp(exp(x))) collapses to exp(x); r=1 by SymPy's auto-eval."""
    result = analyze(sp.log(sp.exp(sp.exp(x))))
    assert result.pfaffian_r == 1
    assert result.eml_depth == 1


def test_deeply_nested_three_function_composition() -> None:
    """cos(exp(sin(x))) is r=5 (sin r=2 + exp + cos r=2)."""
    result = analyze(sp.cos(sp.exp(sp.sin(x))))
    assert result.pfaffian_r == 5
    assert result.max_path_r == 5


# ---------------------------------------------------------------------------
# Pfaffian-but-not-EML detection
# ---------------------------------------------------------------------------


def test_besselj_flagged_as_pfaffian_not_eml() -> None:
    expr = sp.besselj(0, x)
    result = analyze(expr)
    assert result.is_pfaffian_not_eml is True
    assert is_pfaffian_not_eml(expr) is True
    assert result.pfaffian_r == PFAFFIAN_NOT_EML_R["besselj"]


def test_airyai_flagged_as_pfaffian_not_eml() -> None:
    expr = sp.airyai(x)
    result = analyze(expr)
    assert result.is_pfaffian_not_eml is True
    assert result.pfaffian_r == PFAFFIAN_NOT_EML_R["airyai"]


def test_lambertw_flagged_as_pfaffian_not_eml() -> None:
    expr = sp.LambertW(x)
    result = analyze(expr)
    assert result.is_pfaffian_not_eml is True
    assert result.pfaffian_r == PFAFFIAN_NOT_EML_R["LambertW"]


# ---------------------------------------------------------------------------
# v5 max-path behaviour on independent-variable products
# ---------------------------------------------------------------------------


def test_independent_variable_product_uses_max_path_not_sum() -> None:
    """sin(x)*cos(y) — sin and cos act on independent variables, so
    max-path r is 2 (the deeper of either factor) while the total
    pfaffian_r sums to 4. The detector reports both."""
    result = analyze(sp.sin(x) * sp.cos(y))
    assert result.max_path_r == 2
    assert result.pfaffian_r == 4
    # max_path < pfaffian_r is the v5 signature for parallel composition.
    assert result.max_path_r < result.pfaffian_r


def test_independent_exp_product_max_path_is_one() -> None:
    """exp(x)*exp(y): each branch r=1; max-path r=1 (independent vars)."""
    result = analyze(sp.exp(x) * sp.exp(y))
    assert result.max_path_r == 1
    assert max_path_r(sp.exp(x) * sp.exp(y)) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_one_has_zero_complexity() -> None:
    result = analyze(sp.S.One)
    assert result.pfaffian_r == 0
    assert result.eml_depth == 0
    assert result.predicted_depth == 0
    assert result.is_pfaffian_not_eml is False


def test_bare_variable_has_zero_complexity() -> None:
    result = analyze(x)
    assert result.pfaffian_r == 0
    assert result.eml_depth == 0
    assert result.predicted_depth == 0


# ---------------------------------------------------------------------------
# Top-level API contract sanity (defensive against future refactors)
# ---------------------------------------------------------------------------


def test_measure_returns_int_for_simple_expression() -> None:
    """measure() is the public SymPy hook and must return a non-negative int."""
    value = measure(sp.sin(x) ** 2 + sp.cos(x) ** 2)
    assert isinstance(value, int)
    assert value >= 0


def test_pfaffian_r_helper_matches_analyze() -> None:
    """The standalone pfaffian_r() helper must agree with analyze().pfaffian_r."""
    expr = sp.exp(sp.sin(x)) + sp.cos(y)
    assert pfaffian_r(expr) == analyze(expr).pfaffian_r
