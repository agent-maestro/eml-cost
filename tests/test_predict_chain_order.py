"""Tests for the per-AST-node additivity predictor.

Validates the per-class formulation derived from the
detector-conventions probe (50+ expressions, monogate-research/
exploration/detector-conventions-2026-04-27/) and the asin chain-1
fix that landed in 0.15.1.
"""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import predict_chain_order_via_additivity as predict


x, y, t, m, s = sp.symbols("x y t m s", real=True)


# ─── CLASS 0 — atoms, polynomials, integer-exponent Pow ────────────────

class TestClass0:
    def test_bare_variable(self):
        assert predict(x) == 0

    def test_integer_constant(self):
        assert predict(sp.Integer(7)) == 0

    def test_rational(self):
        assert predict(sp.Rational(3, 4)) == 0

    def test_pi(self):
        assert predict(sp.pi) == 0

    def test_polynomial(self):
        assert predict(x**2 + 3*x + 1) == 0

    def test_integer_power_collapses(self):
        # x**2 → integer Pow → 0
        assert predict(x**2) == 0
        # x**2 * x is canonicalized to x**3 by sympy → 0
        assert predict(x**2 * x) == 0


# ─── CLASS 1 — ln, exp, _TANH_LIKE, non-integer Pow ───────────────────

class TestClass1:
    def test_exp_chain_1(self):
        assert predict(sp.exp(x)) == 1

    def test_ln_chain_1(self):
        assert predict(sp.log(x)) == 1

    def test_exp_plus_ln(self):
        # Two CLASS-1 contributions on parallel branches.
        assert predict(sp.exp(x) + sp.log(x)) == 2

    def test_tan(self):
        assert predict(sp.tan(x)) == 1

    def test_tanh(self):
        assert predict(sp.tanh(x)) == 1

    def test_atan(self):
        assert predict(sp.atan(x)) == 1

    def test_atanh(self):
        assert predict(sp.atanh(x)) == 1

    def test_asinh(self):
        assert predict(sp.asinh(x)) == 1

    def test_acosh(self):
        assert predict(sp.acosh(x)) == 1

    def test_asin_now_chain_1(self):
        # The 0.15.1 fix: asin joins _TANH_LIKE for symmetry with atan.
        assert predict(sp.asin(x)) == 1

    def test_sqrt(self):
        # sqrt → Pow(x, Rational(1,2)) — non-integer exponent, +1.
        assert predict(sp.sqrt(x)) == 1

    def test_cube_root(self):
        assert predict(x**sp.Rational(1, 3)) == 1

    def test_symbolic_exponent(self):
        assert predict(x**s) == 1


# ─── CLASS 2 — sin, cos, sinh, cosh ───────────────────────────────────

class TestClass2:
    def test_sin(self):
        assert predict(sp.sin(x)) == 2

    def test_cos(self):
        assert predict(sp.cos(x)) == 2

    def test_sinh(self):
        assert predict(sp.sinh(x)) == 2

    def test_cosh(self):
        assert predict(sp.cosh(x)) == 2

    def test_sin_polynomial_argument(self):
        # The argument is CLASS 0; the sin contributes 2.
        assert predict(sp.sin(x**2)) == 2


# ─── Compositions ─────────────────────────────────────────────────────

class TestCompositions:
    def test_damped_oscillator(self):
        # sin(x) [CLASS 2] * exp(-x) [CLASS 1] → 2 + 1 = 3.
        assert predict(sp.sin(x) * sp.exp(-x)) == 3

    def test_log_of_sin(self):
        # ln [CLASS 1] over sin [CLASS 2] → 1 + 2 = 3.
        assert predict(sp.log(sp.sin(x))) == 3

    def test_exp_of_log(self):
        # SymPy may auto-simplify exp(log(x)) → x; that's fine for the
        # rule which gives 0 on the simplified form.
        assert predict(sp.exp(sp.log(x))) in (0, 2)

    def test_8_octave_fbm(self):
        # Sum_{k=1..8} sin(2^k * x) — 8 distinct sin nodes, each chain 2.
        fbm = sum(sp.sin(2**k * x) for k in range(1, 9))
        assert predict(fbm) == 16


# ─── PNE primitives ───────────────────────────────────────────────────

class TestPNE:
    def test_gamma(self):
        assert predict(sp.gamma(x)) == 2

    def test_erf(self):
        assert predict(sp.erf(x)) == 2

    def test_lambert_w(self):
        assert predict(sp.LambertW(x)) == 2

    def test_bessel_j(self):
        assert predict(sp.besselj(0, x)) == 3

    def test_airy_ai(self):
        assert predict(sp.airyai(x)) == 3

    def test_zeta(self):
        assert predict(sp.zeta(s)) == 4

    def test_bessel_y_post_0_14(self):
        # 0.14.0 bumped Y_n to 5.
        assert predict(sp.bessely(0, x)) == 5

    def test_pne_pair_additive(self):
        # Γ (chain 2) + erf (chain 2) → 4 if both appear in the AST.
        assert predict(sp.gamma(x) + sp.erf(x)) == 4

    def test_pne_dedup(self):
        # Y_0(x) * Y_0(x) → SymPy collapses to Y_0(x)**2; the integer
        # power is CLASS 0 and the Y_0 is counted once → chain 5.
        assert predict(sp.bessely(0, x)**2) == 5

    def test_distinct_pne_factors(self):
        # Y_0(x) [chain 5] * J_0(x) [chain 3] → distinct primitives,
        # chain order 8.
        expr = sp.bessely(0, x) * sp.besselj(0, x)
        assert predict(expr) == 8


# ─── Cross-check against the existing pfaffian_r (sanity) ─────────────

class TestAgreementWithPfaffianR:
    """The two functions should agree on the canonical primitives.

    This is a regression test: any divergence between the registry-
    based pfaffian_r and the AST-walk additivity rule needs to be
    investigated and surfaced in CHANGELOG.
    """

    @pytest.mark.parametrize("expr", [
        sp.exp(x),
        sp.log(x),
        sp.sin(x),
        sp.cos(x),
        sp.sin(x) * sp.exp(-x),
        sp.gamma(x),
        sp.erf(x),
        sp.besselj(0, x),
        sp.bessely(0, x),
        sp.gamma(x) + sp.erf(x),
        sp.LambertW(x),
        sp.airyai(x),
    ])
    def test_agreement(self, expr):
        from eml_cost import analyze
        a = analyze(expr).pfaffian_r
        p = predict(expr)
        assert a == p, f"divergence on {expr}: pfaffian_r={a}, predict={p}"
