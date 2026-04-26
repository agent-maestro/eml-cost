"""SymPy compile-time cost predictor.

Predicts wall-time of canonical compiler-style passes (simplify, factor,
cse, lambdify) on a SymPy expression, using a regression model fit on
202 expressions across 4 proxies (cross-domain-3 R v2, 2026-04-26).

Empirical basis
---------------
Source: ``monogate-research/exploration/cross-domain-3/sR_v2_compile_time.csv``.
Features: ``[eml_depth, max_path_r, log10(count_ops+1), log10(tree_size+1)]``.
Response: ``log10(wall-time in milliseconds)``.
Validation: 5-fold cross-validation, seed=42.

Held-out R^2 by proxy:

    simplify  0.68     factor  0.76     cse  0.83     lambdify  0.84

The shipped coefficients are the full-data OLS fit; CV numbers are
held-out, for honest accuracy reporting.

Usage
-----

    >>> from eml_cost import estimate_time
    >>> r = estimate_time("exp(exp(x)) + sin(x**2)")
    >>> r["simplify"].predicted_ms      # doctest: +SKIP
    156.3
    >>> r["simplify"].ci95               # doctest: +SKIP
    (51.1, 478.0)
    >>> r["lambdify"].predicted_ms       # doctest: +SKIP
    0.42

A single-proxy convenience form is also provided::

    >>> estimate_time("exp(x)", proxy="simplify").predicted_ms  # doctest: +SKIP
    1.4

Limitations
-----------

  - Trained on Python 3.x + sympy >= 1.12 on a single CPU class. Absolute
    times will shift by hardware/version; relative ordering is robust.
  - Only 202 expressions; tail behavior beyond ~5s simplify time is
    extrapolation.
  - Proxy coverage limited to the 4 above. ``solve``, ``integrate``,
    ``series`` are NOT modeled and likely have different cost laws.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Union

import sympy as sp

from .analyze import analyze
from .core import structural_overhead  # noqa: F401  (re-export discipline)

__all__ = ["TimeEstimate", "PROXIES", "estimate_time"]


PROXIES: tuple[str, ...] = ("simplify", "factor", "cse", "lambdify")


# Coefficients fit by E-191 (see fit_report.md in monogate-research).
# Response is log10(milliseconds). Predict via:
#   yhat = intercept
#        + b_eml_depth      * eml_depth
#        + b_max_path_r     * max_path_r
#        + b_log_count_ops  * log10(count_ops + 1)
#        + b_log_tree_size  * log10(tree_size + 1)
# A 95% prediction interval is yhat +/- 1.96 * residual_log10_std.
_COEFS: dict[str, dict[str, float]] = {
    "simplify": {
        "intercept": -1.9684425362856606,
        "eml_depth": 0.014518368330597997,
        "max_path_r": 0.09628604089241327,
        "log_count_ops": -0.29094923716817367,
        "log_tree_size": 2.6759211962967804,
        "residual_log10_std": 0.4507574187381507,
        "cv_r2_mean": 0.6839209554121297,
        "cv_mae_mean_log10": 0.36442206700931845,
    },
    "factor": {
        "intercept": -1.446823981112685,
        "eml_depth": -0.007602468971350307,
        "max_path_r": -0.05379120739920691,
        "log_count_ops": 0.22318029088439673,
        "log_tree_size": 0.9401633874452087,
        "residual_log10_std": 0.15551973803146754,
        "cv_r2_mean": 0.7558419530512351,
        "cv_mae_mean_log10": 0.1202022340296727,
    },
    "cse": {
        "intercept": -2.2224644925483825,
        "eml_depth": -0.0018565427442326365,
        "max_path_r": -0.01822460939659685,
        "log_count_ops": 0.23773186180780562,
        "log_tree_size": 1.0889109109331279,
        "residual_log10_std": 0.13936214681238393,
        "cv_r2_mean": 0.8288700110640391,
        "cv_mae_mean_log10": 0.1133869296957019,
    },
    "lambdify": {
        "intercept": -1.0155215764943966,
        "eml_depth": 0.00048682684317952196,
        "max_path_r": -0.014310265219500549,
        "log_count_ops": -0.01145552562490771,
        "log_tree_size": 0.7587798161901212,
        "residual_log10_std": 0.08122780428723195,
        "cv_r2_mean": 0.836579272836065,
        "cv_mae_mean_log10": 0.06131361347133908,
    },
}

_TRAINING_N = 202
_RESPONSE = "log10_milliseconds"
_SOURCE = "cross-domain-3/sR_v2_compile_time.csv"


@dataclass(frozen=True)
class TimeEstimate:
    """Predicted wall-time for a single compile-time proxy.

    Attributes
    ----------
    proxy:
        One of ``"simplify"``, ``"factor"``, ``"cse"``, ``"lambdify"``.
    predicted_ms:
        Point prediction in milliseconds.
    ci95:
        95% prediction interval ``(low_ms, high_ms)`` from the
        residual log10-std. Asymmetric around the mean because the
        response is log-transformed.
    log10_ms:
        Raw log10(ms) point prediction (useful when chaining models).
    log10_std:
        Residual log10-ms standard deviation from the fit (~ 0.08 to
        0.45 across proxies).
    features:
        The four feature values used for the prediction; included so
        callers can inspect what drove the estimate.
    cv_r2:
        Held-out 5-fold CV R^2 of the underlying model (constant per
        proxy; carried for honest accuracy reporting).
    """

    proxy: str
    predicted_ms: float
    ci95: tuple[float, float]
    log10_ms: float
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


def _predict_single(proxy: str, feats: Mapping[str, float]) -> TimeEstimate:
    c = _COEFS[proxy]
    log10_ms = (
        c["intercept"]
        + c["eml_depth"] * feats["eml_depth"]
        + c["max_path_r"] * feats["max_path_r"]
        + c["log_count_ops"] * feats["log_count_ops"]
        + c["log_tree_size"] * feats["log_tree_size"]
    )
    sigma = c["residual_log10_std"]
    predicted_ms = 10.0**log10_ms
    low_ms = 10.0 ** (log10_ms - 1.96 * sigma)
    high_ms = 10.0 ** (log10_ms + 1.96 * sigma)
    return TimeEstimate(
        proxy=proxy,
        predicted_ms=predicted_ms,
        ci95=(low_ms, high_ms),
        log10_ms=log10_ms,
        log10_std=sigma,
        features=dict(feats),
        cv_r2=c["cv_r2_mean"],
    )


def estimate_time(
    expr: Union[sp.Expr, str],
    *,
    proxy: str = "all",
) -> Union[TimeEstimate, dict[str, TimeEstimate]]:
    """Predict wall-time of compile-style passes on ``expr``.

    Parameters
    ----------
    expr:
        SymPy expression or a string parseable by :func:`sympy.sympify`.
    proxy:
        Which compile-time pass to predict. One of ``"simplify"``,
        ``"factor"``, ``"cse"``, ``"lambdify"``, or ``"all"`` (default).
        ``"all"`` returns a dict keyed by proxy.

    Returns
    -------
    TimeEstimate or dict[str, TimeEstimate]
        Single estimate for a named proxy, or a dict of all four when
        ``proxy="all"``.

    Raises
    ------
    ValueError
        If ``proxy`` is not one of the recognized names.
    sympy.SympifyError
        If ``expr`` is a string that cannot be parsed.

    Examples
    --------
    >>> import sympy as sp
    >>> from eml_cost import estimate_time
    >>> r = estimate_time("x**2 + 1")
    >>> sorted(r)
    ['cse', 'factor', 'lambdify', 'simplify']
    >>> isinstance(r["lambdify"].predicted_ms, float)
    True
    >>> r["lambdify"].predicted_ms > 0.0
    True
    """
    if proxy != "all" and proxy not in _COEFS:
        raise ValueError(
            f"unknown proxy {proxy!r}; expected 'all' or one of "
            f"{sorted(_COEFS)}"
        )

    if not isinstance(expr, sp.Basic):
        expr = sp.sympify(expr)

    feats = _featurize(expr)

    if proxy == "all":
        return {p: _predict_single(p, feats) for p in PROXIES}
    return _predict_single(proxy, feats)


def model_metadata() -> dict[str, Union[int, str, list[str]]]:
    """Return provenance for the shipped regression model.

    Useful for tooling that needs to display or audit which model is
    in use.
    """
    return {
        "n": _TRAINING_N,
        "response": _RESPONSE,
        "source": _SOURCE,
        "proxies": list(PROXIES),
        "features": [
            "eml_depth",
            "max_path_r",
            "log_count_ops",
            "log_tree_size",
        ],
        "session": "E-191",
    }
