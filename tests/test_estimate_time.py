"""Tests for the eml_cost.estimate_time compile-time predictor.

Ships in 0.6.0 from session E-191 (regression on cross-domain-3 R v2,
202 expressions x 4 compile-time proxies).
"""
from __future__ import annotations

import math

import pytest
import sympy as sp

from eml_cost import (
    PROXIES,
    TimeEstimate,
    estimate_time,
    model_metadata,
)
from eml_cost.estimate_time import _COEFS  # type: ignore[attr-defined]


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------


def test_proxies_are_the_four_known_passes() -> None:
    assert set(PROXIES) == {"simplify", "factor", "cse", "lambdify"}


def test_estimate_time_all_returns_dict_keyed_by_proxy() -> None:
    result = estimate_time("exp(x)")
    assert isinstance(result, dict)
    assert set(result) == set(PROXIES)
    for proxy, est in result.items():
        assert isinstance(est, TimeEstimate)
        assert est.proxy == proxy


def test_estimate_time_single_proxy_returns_TimeEstimate() -> None:
    result = estimate_time("exp(x)", proxy="simplify")
    assert isinstance(result, TimeEstimate)
    assert result.proxy == "simplify"


def test_estimate_time_accepts_sympy_expr() -> None:
    expr = sp.exp(sp.exp(x)) + sp.sin(x**2)
    result = estimate_time(expr)
    assert isinstance(result, dict)
    assert all(isinstance(r, TimeEstimate) for r in result.values())


def test_estimate_time_unknown_proxy_raises() -> None:
    with pytest.raises(ValueError, match="unknown proxy"):
        estimate_time("x", proxy="solve")


def test_estimate_time_invalid_string_raises() -> None:
    # sympify will reject this with a SympifyError (subclass of ValueError)
    with pytest.raises((sp.SympifyError, ValueError, SyntaxError, TypeError)):
        estimate_time("not a valid expression!! ??")


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("proxy", PROXIES)
def test_predictions_are_positive(proxy: str) -> None:
    est = estimate_time("exp(exp(x))", proxy=proxy)
    assert est.predicted_ms > 0.0
    assert est.ci95[0] >= 0.0
    assert est.ci95[1] > est.ci95[0]


@pytest.mark.parametrize("proxy", PROXIES)
def test_predicted_inside_ci95(proxy: str) -> None:
    est = estimate_time("exp(x) * sin(x**2)", proxy=proxy)
    low, high = est.ci95
    assert low <= est.predicted_ms <= high


@pytest.mark.parametrize("proxy", PROXIES)
def test_log10_ms_consistent_with_predicted_ms(proxy: str) -> None:
    est = estimate_time("(x+1)**5", proxy=proxy)
    assert est.predicted_ms == pytest.approx(10.0**est.log10_ms, rel=1e-9)


def test_features_are_recorded() -> None:
    est = estimate_time("sin(x) + cos(x**2)", proxy="simplify")
    assert set(est.features) == {
        "eml_depth",
        "max_path_r",
        "log_count_ops",
        "log_tree_size",
    }
    assert all(isinstance(v, float) for v in est.features.values())


def test_cv_r2_is_carried_through() -> None:
    for proxy in PROXIES:
        est = estimate_time("x", proxy=proxy)
        assert est.cv_r2 == pytest.approx(_COEFS[proxy]["cv_r2_mean"])


# ---------------------------------------------------------------------------
# Monotonicity / ordering — depth-3 should predict higher simplify time
# than depth-0 polynomial. (Spearman intent of the model.)
# ---------------------------------------------------------------------------


def test_deep_composition_predicts_higher_simplify_than_polynomial() -> None:
    deep = estimate_time("sin(sin(sin(sin(x))))", proxy="simplify")
    shallow = estimate_time("x", proxy="simplify")
    assert deep.predicted_ms > shallow.predicted_ms


def test_larger_tree_predicts_higher_lambdify_than_smaller_tree() -> None:
    big = estimate_time("(x+1)*(x+2)*(x+3)*(x+4)*(x+5)", proxy="lambdify")
    small = estimate_time("x", proxy="lambdify")
    assert big.predicted_ms > small.predicted_ms


def test_rank_order_on_held_corpus_is_sensible() -> None:
    """Tier-level rank sanity check on a held corpus.

    The model legitimately predicts depth-deep / tree-small expressions
    (e.g. ``exp(exp(exp(x)))``) lower than tree-large expressions because
    ``log_tree_size`` carries the dominant simplify-cost weight. So we
    test tiered complexity, not strict element-by-element monotonicity.
    """
    trivial = ["x", "x + 1", "x**2"]
    moderate = ["sin(x) + cos(x)", "exp(sin(x**2))", "(x+1)*(x+2)*(x+3)"]
    heavy = [
        "(x+1)*(x+2)*(x+3)*(x+4)*(x+5)",
        "sin(x)*cos(x)*exp(x)*log(x+1)",
        "(x**2 + 1)*(x**2 + x + 1)*(x**3 + 2*x + 3)",
    ]

    def median_pred(cases: list[str]) -> float:
        ms = sorted(
            estimate_time(s, proxy="simplify").predicted_ms for s in cases
        )
        return ms[len(ms) // 2]

    m_trivial = median_pred(trivial)
    m_moderate = median_pred(moderate)
    m_heavy = median_pred(heavy)

    assert m_moderate > m_trivial, (m_moderate, m_trivial)
    assert m_heavy > m_moderate, (m_heavy, m_moderate)


def test_neighbour_monotone_concordance_above_chance() -> None:
    """Concordance with simplify-cost intuition >= 60% (above 50% chance)."""
    cases = [
        "x",
        "x**2 + 1",
        "(x+1)**3",
        "(x+1)*(x+2)",
        "(x+1)*(x+2)*(x+3)",
        "(x+1)*(x+2)*(x+3)*(x+4)",
        "(x+1)*(x+2)*(x+3)*(x+4)*(x+5)",
    ]
    predictions = [
        estimate_time(s, proxy="simplify").predicted_ms for s in cases
    ]
    concordant = total = 0
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            total += 1
            if predictions[j] > predictions[i]:
                concordant += 1
    assert concordant / total >= 0.7, (
        f"concordance {concordant}/{total} below 0.7; predictions={predictions}"
    )


# ---------------------------------------------------------------------------
# Provenance / metadata
# ---------------------------------------------------------------------------


def test_model_metadata_reports_n_and_provenance() -> None:
    meta = model_metadata()
    assert meta["n"] == 202
    assert meta["session"] == "E-191"
    assert meta["response"] == "log10_milliseconds"
    assert "cross-domain-3" in str(meta["source"])
    assert set(meta["proxies"]) == set(PROXIES)  # type: ignore[arg-type]


def test_coefficients_table_complete_for_all_proxies() -> None:
    expected_keys = {
        "intercept",
        "eml_depth",
        "max_path_r",
        "log_count_ops",
        "log_tree_size",
        "residual_log10_std",
        "cv_r2_mean",
        "cv_mae_mean_log10",
    }
    for proxy in PROXIES:
        assert expected_keys <= set(_COEFS[proxy])
        assert _COEFS[proxy]["residual_log10_std"] > 0.0
        assert -1.0 <= _COEFS[proxy]["cv_r2_mean"] <= 1.0


# ---------------------------------------------------------------------------
# Empirical reasonableness on training-corpus exemplars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_str,proxy,expected_low_ms,expected_high_ms",
    [
        # Simple cases — wide-but-bounded predictions.
        ("x", "lambdify", 0.05, 5.0),
        ("(x+1)**5", "simplify", 0.05, 100.0),
        ("exp(x)", "factor", 0.01, 5.0),
        ("sin(x) + cos(x)", "cse", 0.005, 1.0),
    ],
)
def test_predictions_within_reasonable_bounds(
    expr_str: str, proxy: str, expected_low_ms: float, expected_high_ms: float
) -> None:
    est = estimate_time(expr_str, proxy=proxy)
    assert expected_low_ms <= est.predicted_ms <= expected_high_ms, (
        f"{proxy}({expr_str}) = {est.predicted_ms:.3f} ms "
        f"outside [{expected_low_ms}, {expected_high_ms}]"
    )


def test_residual_std_bounds_match_published_values() -> None:
    """Sigma values from the fit; protect against drift on coefficient updates."""
    expected_max = {
        "simplify": 0.55,   # slack above 0.45
        "factor": 0.20,     # slack above 0.16
        "cse": 0.18,        # slack above 0.14
        "lambdify": 0.12,   # slack above 0.08
    }
    for proxy in PROXIES:
        sigma = _COEFS[proxy]["residual_log10_std"]
        assert sigma <= expected_max[proxy], (
            f"{proxy} residual std {sigma} above expected ceiling "
            f"{expected_max[proxy]}"
        )


def test_ci95_is_asymmetric_in_linear_space() -> None:
    """log-space symmetry => linear-space asymmetry; check it's actually so."""
    est = estimate_time("exp(exp(x)) + sin(x**2)", proxy="simplify")
    low, high = est.ci95
    # Linear-space: high - mean > mean - low
    assert (high - est.predicted_ms) > (est.predicted_ms - low)
    # Log-space symmetry: log10(mean) - log10(low) == log10(high) - log10(mean)
    if low > 0:
        left = est.log10_ms - math.log10(low)
        right = math.log10(high) - est.log10_ms
        assert left == pytest.approx(right, rel=1e-6)
