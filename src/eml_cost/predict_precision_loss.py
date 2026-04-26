"""Float64 numerical precision-loss predictor.

Predicts the magnitude of float64 numerical error vs 50-digit mpmath
ground truth on a SymPy expression, using a regression model fit on
379 expressions across the bench-300-domain corpus (E-193, 2026-04-26).

Empirical basis
---------------
Source: ``monogate-research/exploration/E193_numerical_stability/
corpus_with_stability.csv``.
Features: ``[eml_depth, max_path_r, log10(count_ops+1), log10(tree_size+1)]``
(same pipeline as :func:`eml_cost.estimate_time`).
Response: ``log10(mpmath_max_relerr)``, floored at float64 machine
epsilon (~2.22e-16) for the 50 corpus rows where the float64 evaluation
matched mpmath bit-for-bit at every sampled point.

Validation: 5-fold cross-validation (seed=42). Held-out R^2 = +0.271
+/- 0.060; residual log10 std = 0.772; full-fit R^2 = +0.289.

This is a modest predictor by design — the underlying signal (partial
r = +0.357 controlling for tree size, q = 1.6e-11, E-193) is real but
moderate. CI95 is wide (~factor 30 either way) and the function is
explicit about that. Use it for **rank-ordering candidate expressions
or surfacing high-risk subtrees in a SymPy linter**, not for absolute
precision claims.

Honest limitations
------------------

  - Trained on the bench-300-domain corpus (52 distinct domains).
    Expressions outside this distribution (e.g. expressions involving
    Bessel/Lambert W where mpmath itself takes longer) may project
    poorly onto these features.
  - **Not an algebraically-equivalent-form recommender.** The E-193
    Phase 3 form-sensitivity test showed only 30% best-pick on 10
    rewrite-tests; do NOT use this function to choose between
    `1/(1+exp(-x))` and `tanh(x/2)/2 + 1/2`. It compares expressions
    *cross-corpus*; equivalent-form rewrite advice is out of scope.
  - The 50 bit-for-bit-matching corpus rows are floored to machine
    epsilon. The model can predict values below epsilon (the floor is
    a measurement artifact); callers should clamp at machine epsilon
    when displaying.

Usage
-----

    >>> from eml_cost import predict_precision_loss
    >>> r = predict_precision_loss("exp(exp(x)) + sin(x**2)")
    >>> r.predicted_max_relerr               # doctest: +SKIP
    1.7e-13
    >>> r.predicted_digits_lost              # doctest: +SKIP
    3.05
    >>> r.ci95                               # doctest: +SKIP
    (5.6e-15, 5.4e-12)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Union

import sympy as sp

from .analyze import analyze
from .core import structural_overhead  # noqa: F401  (re-export discipline)

__all__ = [
    "PrecisionLossEstimate",
    "predict_precision_loss",
    "precision_loss_model_metadata",
    "FLOAT64_EPS",
]


# numpy.finfo(float64).eps — the smallest reportable mpmath_max_relerr.
FLOAT64_EPS: float = 2.220446049250313e-16


# Coefficients fit by E-193 (see fit_report.md in monogate-research).
# Response is log10(mpmath_max_relerr). Predict via:
#   yhat = intercept
#        + b_eml_depth      * eml_depth
#        + b_max_path_r     * max_path_r
#        + b_log_count_ops  * log10(count_ops + 1)
#        + b_log_tree_size  * log10(tree_size + 1)
# A 95% prediction interval is yhat +/- 1.96 * residual_log10_std.
_COEFS: dict[str, float] = {
    "intercept": -15.440556729015627,
    "eml_depth": 0.33668813151183796,
    "max_path_r": -0.1249378060199701,
    "log_count_ops": 0.9442895772504774,
    "log_tree_size": -1.3751259909585363,
    "residual_log10_std": 0.7720847093601771,
    "cv_r2_mean": 0.2711510788080426,
    "cv_r2_std": 0.059689422104345385,
    "cv_mae_mean_log10": 0.5788393107752179,
}

_TRAINING_N = 379
_RESPONSE = "log10_mpmath_max_relerr"
_SOURCE = "E193_numerical_stability/corpus_with_stability.csv"


@dataclass(frozen=True)
class PrecisionLossEstimate:
    """Predicted float64 precision loss for a single expression.

    Attributes
    ----------
    predicted_max_relerr:
        Point prediction of ``mpmath_max_relerr``. Smallest meaningful
        value is :data:`FLOAT64_EPS`; predictions below that should be
        interpreted as "at most one ULP".
    predicted_digits_lost:
        Convenience: how many decimal digits of float64 precision are
        predicted to be lost relative to a perfect 16-digit float64
        result, computed as ``log10(predicted_max_relerr) - log10(eps)``.
        Negative values are clamped to 0 (no loss).
    ci95:
        95% prediction interval ``(low_relerr, high_relerr)`` from the
        residual log10-std. Wide interval (~factor 30 either way)
        reflecting the modest CV R^2.
    log10_relerr:
        Raw log10(mpmath_max_relerr) point prediction.
    log10_std:
        Residual log10 standard deviation from the fit (~0.77).
    features:
        The four feature values used; included so callers can inspect
        what drove the estimate.
    cv_r2:
        Held-out 5-fold CV R^2 of the underlying model (~0.27). Carried
        for honest accuracy reporting.
    """

    predicted_max_relerr: float
    predicted_digits_lost: float
    ci95: tuple[float, float]
    log10_relerr: float
    log10_std: float
    features: dict[str, float]
    cv_r2: float


def _featurize(expr: sp.Expr) -> dict[str, float]:
    r = analyze(expr)
    n_ops = int(sp.count_ops(expr))
    n_size = sum(1 for _ in sp.preorder_traversal(expr))
    return {
        "eml_depth": float(r.eml_depth),
        "max_path_r": float(r.max_path_r),
        "log_count_ops": math.log10(n_ops + 1),
        "log_tree_size": math.log10(n_size + 1),
    }


def _digits_lost(log10_relerr: float) -> float:
    """Convert log10(relerr) to "digits of precision lost" for display.

    Float64 caps at ~16 digits (eps ~ 2.22e-16, log10(eps) ~ -15.65).
    A predicted log10(relerr) of -10 means 5.65 decimal digits lost.
    Negative loss (predictions below eps) is clamped to 0.
    """
    log10_eps = math.log10(FLOAT64_EPS)
    return max(0.0, log10_relerr - log10_eps)


def predict_precision_loss(
    expr: Union[sp.Expr, str],
) -> PrecisionLossEstimate:
    """Predict float64 vs mpmath relative error magnitude for ``expr``.

    Parameters
    ----------
    expr:
        SymPy expression or a string parseable by :func:`sympy.sympify`.

    Returns
    -------
    PrecisionLossEstimate
        Point prediction, 95% interval, predicted decimal-digits lost,
        feature inputs, and provenance R^2.

    Raises
    ------
    sympy.SympifyError
        If ``expr`` is a string that cannot be parsed.

    Examples
    --------
    >>> import sympy as sp
    >>> from eml_cost import predict_precision_loss
    >>> r = predict_precision_loss("x**2 + 1")
    >>> isinstance(r.predicted_max_relerr, float)
    True
    >>> r.predicted_max_relerr > 0
    True
    >>> low, high = r.ci95
    >>> low <= r.predicted_max_relerr <= high
    True
    """
    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)

    feats = _featurize(expr)

    log10_relerr = (
        _COEFS["intercept"]
        + _COEFS["eml_depth"] * feats["eml_depth"]
        + _COEFS["max_path_r"] * feats["max_path_r"]
        + _COEFS["log_count_ops"] * feats["log_count_ops"]
        + _COEFS["log_tree_size"] * feats["log_tree_size"]
    )
    sigma = _COEFS["residual_log10_std"]
    predicted = 10.0**log10_relerr
    low = 10.0 ** (log10_relerr - 1.96 * sigma)
    high = 10.0 ** (log10_relerr + 1.96 * sigma)

    return PrecisionLossEstimate(
        predicted_max_relerr=predicted,
        predicted_digits_lost=_digits_lost(log10_relerr),
        ci95=(low, high),
        log10_relerr=log10_relerr,
        log10_std=sigma,
        features=dict(feats),
        cv_r2=_COEFS["cv_r2_mean"],
    )


def precision_loss_model_metadata() -> dict[str, Union[int, float, str, list[str]]]:
    """Return provenance for the shipped precision-loss regression model.

    Useful for tooling that needs to display or audit which model is
    in use.
    """
    return {
        "n": _TRAINING_N,
        "response": _RESPONSE,
        "source": _SOURCE,
        "features": [
            "eml_depth",
            "max_path_r",
            "log_count_ops",
            "log_tree_size",
        ],
        "session": "E-193",
        "cv_r2_mean": _COEFS["cv_r2_mean"],
        "residual_log10_std": _COEFS["residual_log10_std"],
        "relerr_floor": FLOAT64_EPS,
        "headline_partial_r": 0.357,
        "headline_partial_q": 1.6e-11,
        "honest_note": (
            "Modest predictor (CV R^2 ~ 0.27, residual log10 std ~ 0.77). "
            "Use for rank-ordering and high-risk subtree surfacing, NOT "
            "for absolute precision claims. NOT a form recommender — "
            "see E-193 Phase 3 (30% best-pick on form-sensitive rewrite "
            "tests). The form-rewrite recommender was deliberately not "
            "shipped."
        ),
    }
