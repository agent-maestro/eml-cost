"""Tests for the 0.3.0 PFAFFIAN_NOT_EML_R extension.

S/R-134 deep research surfaced that 24 named non-elementary SymPy
functions were being silently treated as depth-0 atoms. 0.3.0 adds
20 new entries (digamma → polygamma collapses to one slot) covering
6 families: erf, gamma, exp-integrals, polylog/zeta, elliptic,
Fresnel.

These tests verify that the new entries:
  - are flagged as Pfaffian-not-EML
  - report a non-zero pfaffian_r
  - don't break any existing detector behaviour for elementary inputs
"""
from __future__ import annotations

import sympy as sp

from eml_cost import PFAFFIAN_NOT_EML_R, analyze, fingerprint_axes


x = sp.Symbol("x")
y = sp.Symbol("y")


def test_new_registry_entries_present():
    """Every family from S/R-134 has at least one entry."""
    expected_keys = {
        "erf", "erfc", "erfi", "fresnels", "fresnelc",
        "gamma", "loggamma", "polygamma", "beta",
        "Ei", "li", "Si", "Ci", "Shi", "Chi",
        "polylog", "zeta",
        "elliptic_k", "elliptic_e", "elliptic_f",
    }
    missing = expected_keys - set(PFAFFIAN_NOT_EML_R)
    assert not missing, f"missing entries in PFAFFIAN_NOT_EML_R: {missing}"


def test_erf_is_now_flagged_pfaffian_not_eml():
    """erf was silently depth-0 atom pre-0.3.0; should now be flagged."""
    a = analyze(sp.erf(x))
    assert a.is_pfaffian_not_eml is True
    assert a.pfaffian_r >= 2


def test_gamma_family_is_flagged():
    """gamma, loggamma, polygamma (covers digamma) all flagged."""
    for f, args in [(sp.gamma, (x,)), (sp.loggamma, (x,)),
                    (sp.digamma, (x,)), (sp.polygamma, (1, x)),
                    (sp.beta, (x, y))]:
        expr = f(*args)
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, (
            f"{f.__name__} should be Pfaffian-not-EML"
        )


def test_exp_integrals_are_flagged():
    """Ei, li, Si, Ci, Shi, Chi all flagged."""
    for f in (sp.Ei, sp.li, sp.Si, sp.Ci, sp.Shi, sp.Chi):
        expr = f(x)
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, (
            f"{f.__name__} should be Pfaffian-not-EML"
        )
        assert a.pfaffian_r >= 2


def test_polylog_and_zeta_are_flagged():
    """polylog and zeta — extend exp/log Pfaffian chain."""
    for expr in (sp.polylog(2, x), sp.polylog(3, x),
                 sp.zeta(x), sp.zeta(x, y)):
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True


def test_elliptic_integrals_are_flagged():
    """K, E, F all flagged."""
    for expr in (sp.elliptic_k(x), sp.elliptic_e(x),
                 sp.elliptic_f(y, x)):
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True


def test_elementary_inputs_still_pass_through():
    """Regression: existing elementary inputs still get
    is_pfaffian_not_eml=False. The 0.3.0 expansion shouldn't have
    accidentally tagged anything elementary."""
    elementary_inputs = [
        sp.exp(x), sp.log(x), sp.sin(x), sp.cos(x),
        sp.tan(x), sp.sinh(x), sp.cosh(x), sp.tanh(x),
        sp.exp(sp.sin(x)), sp.log(1 + sp.exp(x)),  # softplus
        x ** 2, 1 / x, sp.sqrt(x),
    ]
    for expr in elementary_inputs:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is False, (
            f"{expr} should NOT be Pfaffian-not-EML"
        )
