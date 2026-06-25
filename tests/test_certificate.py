"""Tests for eml_cost.certificate — the machine-checked non-oscillation cert.

certify_non_oscillation ties a classify_ode EML-finite verdict to the Lean
theorem MachLib.SturmNonOscillation.sturm_no_positive_bump (r >= 0 forbids a
positive arch). certified=True only when r >= 0 is actually verifiable.
"""
from __future__ import annotations

import sympy as sp

from eml_cost import certify_non_oscillation, NonOscillationCertificate

x = sp.Symbol("x", real=True)

_THM = "MachLib.SturmNonOscillation.sturm_no_positive_bump"


def test_hyperbolic_is_certified():
    c = certify_non_oscillation(0, -1)  # y'' - y = 0, r = 1 >= 0
    assert isinstance(c, NonOscillationCertificate)
    assert c.certified is True
    assert c.lean_theorem == _THM
    assert ">=" in c.condition


def test_overdamped_is_certified():
    c = certify_non_oscillation(3, 2)  # r = 1/4 >= 0
    assert c.certified is True
    assert c.lean_theorem == _THM


def test_repeated_root_zero_potential_certified():
    c = certify_non_oscillation(-2, 1)  # r = 0
    assert c.certified is True
    assert c.lean_theorem == _THM


def test_euler_nonneg_c_certified():
    # x²y'' + x y' - y = 0 -> r = 3/(4 x²) >= 0 on (0, oo)
    c = certify_non_oscillation(1 / x, -1 / x**2, "pos")
    assert c.certified is True
    assert c.lean_theorem == _THM


def test_oscillatory_has_no_certificate():
    assert certify_non_oscillation(0, 1) is None        # harmonic, oscillatory
    assert certify_non_oscillation(0, -x, "R") is None  # Airy, oscillatory


def test_candidate_has_no_certificate():
    # modified Bessel I_2: EML-finite? candidate, not EML-finite -> None
    assert certify_non_oscillation(1 / x, -(1 + 4 / x**2), "pos") is None


def test_eml_finite_but_r_negative_is_not_certified():
    # Euler with r = c/x², -1/4 <= c < 0: non-oscillatory but r < 0.
    # Pick indicial disc in [0,1) so r = (disc-1)/(4) /x² is negative but EML-finite.
    # a=1,b=3/16 -> indicial r²+0r+3/16, disc = -3/4 < 0 -> oscillatory (skip).
    # Use a=1, b=3/16 is complex; instead choose real-root Euler with c<0:
    # want disc>=0 and c=(disc-1)/4 <0 -> disc<1. a=1, b=3/16: disc=-3/4 (no).
    # a=1, b=1/8: disc = 0 - 4*(1/8) = -1/2 (<0). Hmm equidim needs (a-1)²-4b>=0.
    # Take a=2, b=3/16: (a-1)²-4b = 1 - 3/4 = 1/4 >=0 (real roots), and
    # normal-form r = ((a²-2a-4b)/4)/x² = ((4-4-3/4)/4)/x² = (-3/16)/x² < 0.
    c = certify_non_oscillation(2 / x, sp.Rational(3, 16) / x**2, "pos")
    assert isinstance(c, NonOscillationCertificate)
    assert c.certified is False           # non-oscillatory but r < 0
    assert c.lean_theorem is None
    assert "not yet Lean-certified" in c.note
