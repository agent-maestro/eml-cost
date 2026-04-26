"""Tests for the eml_cost.predict_precision_loss runtime predictor.

Ships in 0.7.0 from session E-193 (regression on bench-300-domain
mpmath_max_relerr, 379 valid expressions; held-out CV R^2 = +0.27,
residual log10 std = 0.77).
"""
from __future__ import annotations

import math

import pytest
import sympy as sp

from eml_cost import (
    FLOAT64_EPS,
    PrecisionLossEstimate,
    precision_loss_model_metadata,
    predict_precision_loss,
)
from eml_cost.predict_precision_loss import _COEFS  # type: ignore[attr-defined]


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------


def test_returns_PrecisionLossEstimate() -> None:
    r = predict_precision_loss("exp(x)")
    assert isinstance(r, PrecisionLossEstimate)


def test_accepts_sympy_expr() -> None:
    expr = sp.exp(sp.exp(x)) + sp.sin(x**2)
    r = predict_precision_loss(expr)
    assert isinstance(r, PrecisionLossEstimate)


def test_accepts_string() -> None:
    r = predict_precision_loss("sin(x) + cos(x)")
    assert isinstance(r, PrecisionLossEstimate)


def test_invalid_string_raises() -> None:
    with pytest.raises((sp.SympifyError, ValueError, SyntaxError, TypeError)):
        predict_precision_loss("not a valid expression!! ??")


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


def test_predicted_relerr_positive() -> None:
    r = predict_precision_loss("exp(exp(x))")
    assert r.predicted_max_relerr > 0.0
    assert r.ci95[0] > 0.0
    assert r.ci95[1] > r.ci95[0]


def test_predicted_inside_ci95() -> None:
    r = predict_precision_loss("exp(x) * sin(x**2)")
    low, high = r.ci95
    assert low <= r.predicted_max_relerr <= high


def test_log10_consistent_with_relerr() -> None:
    r = predict_precision_loss("(x+1)**5")
    assert r.predicted_max_relerr == pytest.approx(10.0**r.log10_relerr, rel=1e-9)


def test_features_are_recorded() -> None:
    r = predict_precision_loss("sin(x) + cos(x**2)")
    assert set(r.features) == {
        "eml_depth",
        "max_path_r",
        "log_count_ops",
        "log_tree_size",
    }
    assert all(isinstance(v, float) for v in r.features.values())


def test_cv_r2_is_carried_through() -> None:
    r = predict_precision_loss("x")
    assert r.cv_r2 == pytest.approx(_COEFS["cv_r2_mean"])


def test_digits_lost_nonneg() -> None:
    r = predict_precision_loss("x**2 + 1")
    assert r.predicted_digits_lost >= 0.0


def test_digits_lost_consistent_with_log10_relerr() -> None:
    r = predict_precision_loss("exp(exp(x)) + sin(x**2)")
    expected = max(0.0, r.log10_relerr - math.log10(FLOAT64_EPS))
    assert r.predicted_digits_lost == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Monotonicity / ordering — directional intent of the model.
#
# The marginal coefficient on eml_depth is +0.337 (per fit). Holding
# the other features roughly constant, deeper expressions should
# predict more relative error.
# ---------------------------------------------------------------------------


def test_deeper_composition_predicts_more_loss_than_shallow() -> None:
    """Equal token-count, varying depth — depth term should win."""
    deep = predict_precision_loss("sin(sin(sin(sin(x))))")
    shallow = predict_precision_loss("sin(x) + sin(x) + sin(x) + sin(x)")
    assert deep.predicted_max_relerr > shallow.predicted_max_relerr


def test_larger_expression_predicts_above_eps() -> None:
    """Non-trivial expression should predict measurable (above-eps) loss."""
    r = predict_precision_loss(
        "exp(x) + sin(x**2) + cos(x**3) + log(x + 1) * exp(-x)"
    )
    assert r.predicted_max_relerr > FLOAT64_EPS


def test_neighbour_concordance_on_depth_ladder() -> None:
    """Nested-composition ladder: predicted relerr should rank-order
    monotonically with depth at >= 70% concordance."""
    cases = [
        "x",
        "sin(x)",
        "sin(sin(x))",
        "sin(sin(sin(x)))",
        "sin(sin(sin(sin(x))))",
        "sin(sin(sin(sin(sin(x)))))",
    ]
    predictions = [
        predict_precision_loss(s).predicted_max_relerr for s in cases
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


def test_metadata_reports_n_and_provenance() -> None:
    meta = precision_loss_model_metadata()
    assert meta["n"] == 379
    assert meta["session"] == "E-193"
    assert meta["response"] == "log10_mpmath_max_relerr"
    assert "E193" in str(meta["source"])
    assert meta["relerr_floor"] == pytest.approx(FLOAT64_EPS)


def test_metadata_carries_honest_note() -> None:
    """Honest accuracy framing must travel with the model so consumers
    see it (mirrors the eml-cost-torch 0.5.0 honest_note pattern)."""
    meta = precision_loss_model_metadata()
    assert "honest_note" in meta
    note = str(meta["honest_note"])
    assert "Modest" in note or "modest" in note
    assert "form recommender" in note.lower() or "rewrite" in note.lower()


def test_coefficients_table_complete() -> None:
    expected_keys = {
        "intercept",
        "eml_depth",
        "max_path_r",
        "log_count_ops",
        "log_tree_size",
        "residual_log10_std",
        "cv_r2_mean",
    }
    assert expected_keys <= set(_COEFS)
    assert _COEFS["residual_log10_std"] > 0.0
    assert -1.0 <= _COEFS["cv_r2_mean"] <= 1.0


def test_eml_depth_coefficient_is_positive() -> None:
    """E-193 headline: deeper EML routing -> more precision loss.

    Sign of the coefficient must remain positive; flipping it would
    invert the predicted ordering without anyone noticing.
    """
    assert _COEFS["eml_depth"] > 0.0


def test_residual_std_within_published_band() -> None:
    """Residual log10 std from the fit; protect against drift on
    coefficient updates (~0.77 from the E-193 fit)."""
    assert 0.5 < _COEFS["residual_log10_std"] < 1.2


def test_cv_r2_within_published_band() -> None:
    """Honest framing: CV R^2 ~ 0.27. Catch updates that overstate."""
    cv = _COEFS["cv_r2_mean"]
    assert 0.15 < cv < 0.45


# ---------------------------------------------------------------------------
# CI95 shape
# ---------------------------------------------------------------------------


def test_ci95_is_asymmetric_in_linear_space() -> None:
    """log-space symmetry => linear-space asymmetry; verify."""
    r = predict_precision_loss("exp(exp(x)) + sin(x**2)")
    low, high = r.ci95
    assert (high - r.predicted_max_relerr) > (r.predicted_max_relerr - low)
    if low > 0:
        left = r.log10_relerr - math.log10(low)
        right = math.log10(high) - r.log10_relerr
        assert left == pytest.approx(right, rel=1e-6)


def test_ci95_width_reflects_modest_signal() -> None:
    """CV R^2 ~ 0.27 + sigma 0.77 means CI95 spans ~ factor 30 in
    relerr space. Check the span is at least factor 10 to keep the
    honest-framing intact (a tight CI would imply over-claim)."""
    r = predict_precision_loss("exp(exp(x)) + sin(x**2)")
    low, high = r.ci95
    assert (high / max(low, 1e-300)) > 10.0


# ---------------------------------------------------------------------------
# Empirical reasonableness on training-corpus-style exemplars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_str,low_relerr,high_relerr",
    [
        # Polynomials should land around eps-floor with wide CI.
        ("x", 1e-19, 1e-9),
        ("x**2 + 1", 1e-19, 1e-9),
        # Trig + composition should land somewhere in the corpus range.
        ("sin(x) + cos(x)", 1e-19, 1e-9),
        # Deep composition should land closer to 1e-12 region.
        ("exp(exp(exp(x)))", 1e-17, 1e-7),
    ],
)
def test_predictions_within_reasonable_bounds(
    expr_str: str, low_relerr: float, high_relerr: float
) -> None:
    r = predict_precision_loss(expr_str)
    assert low_relerr <= r.predicted_max_relerr <= high_relerr, (
        f"predict_precision_loss({expr_str!r}) = "
        f"{r.predicted_max_relerr:.2e} outside "
        f"[{low_relerr:.0e}, {high_relerr:.0e}]"
    )
