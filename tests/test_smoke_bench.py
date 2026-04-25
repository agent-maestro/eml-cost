"""Smoke tests: 19 hand-derived rows from the 200-row research bench.

These are the same sanity cases used during detector calibration.
The detector should agree on r and depth (within ±1 for depth, exact for r)
on every row except convention-sensitive cases (sech^2 * tanh).
"""
from __future__ import annotations

import sympy as sp

from eml_cost import analyze


x = sp.Symbol("x", real=True)


# (description, expr, expected_r, expected_depth_min, expected_depth_max)
CASES = [
    ("A1: x*exp(x)",                  x * sp.exp(x),                    1, 2, 3),
    ("B1: exp(exp(exp(x)))",          sp.exp(sp.exp(sp.exp(x))),        3, 3, 3),
    ("B5: sin(sin(x))",               sp.sin(sp.sin(x)),                4, 6, 7),
    ("C1: exp(sin(x))",               sp.exp(sp.sin(x)),                3, 4, 5),
    ("C5: softplus log(1+exp(x))",    sp.log(1 + sp.exp(x)),            2, 1, 1),
    ("C6: tan(x)",                    sp.tan(x),                        1, 4, 4),
    ("E1: x^x = exp(x*log(x))",       sp.exp(x * sp.log(x)),            2, 3, 4),
    ("E5: cosh(x)",                   sp.cosh(x),                       2, 1, 2),
    ("O1: 1s exp(-x)",                sp.exp(-x),                       1, 1, 2),
    ("H1: sinh(x)",                   sp.sinh(x),                       2, 1, 1),
    ("H8: atan(x)",                   sp.atan(x),                       1, 1, 1),
    ("P1: x^10",                      x ** 10,                          0, 1, 1),
    ("DC5: sqrt(exp(sqrt(x)))",       sp.sqrt(sp.exp(sp.sqrt(x))),      3, 3, 4),
    ("SF1: gauss exp(-x^2/2)",        sp.exp(-(x ** 2) / 2),            1, 2, 3),
    ("PH1: T^4",                      x ** 4,                           0, 1, 1),
    ("EE1: RC exp(-t/RC)",            sp.exp(-x),                       1, 1, 2),
    ("AI4: Swish x*sigmoid(x)",       x / (1 + sp.exp(-x)),             1, 2, 3),
    ("Sigmoid: 1/(1+exp(-x))",        1 / (1 + sp.exp(-x)),             1, 1, 2),
]


def test_smoke_bench_pfaffian_r() -> None:
    """All 18 hand-derived rows should match expected pfaffian_r exactly."""
    failures = []
    for desc, expr, exp_r, _dmin, _dmax in CASES:
        result = analyze(expr)
        if result.pfaffian_r != exp_r:
            failures.append(f"{desc}: got r={result.pfaffian_r}, expected {exp_r}")
    assert not failures, "Pfaffian r mismatches:\n  " + "\n  ".join(failures)


def test_smoke_bench_eml_depth_within_range() -> None:
    """All 18 rows should have eml_depth within the calibrated range."""
    failures = []
    for desc, expr, _exp_r, dmin, dmax in CASES:
        result = analyze(expr)
        if not (dmin <= result.eml_depth <= dmax):
            failures.append(
                f"{desc}: got depth={result.eml_depth}, expected [{dmin}, {dmax}]"
            )
    assert not failures, "EML depth out of range:\n  " + "\n  ".join(failures)


def test_smoke_bench_predicted_depth_in_band() -> None:
    """``predicted_depth`` should agree with ``eml_depth`` within +/-2 on these rows."""
    out_of_band = []
    for desc, expr, _exp_r, _dmin, _dmax in CASES:
        result = analyze(expr)
        residual = result.eml_depth - result.predicted_depth
        if abs(residual) > 2:
            out_of_band.append(
                f"{desc}: depth={result.eml_depth}, predicted={result.predicted_depth}, "
                f"residual={residual}"
            )
    assert not out_of_band, "Out-of-band cases:\n  " + "\n  ".join(out_of_band)
