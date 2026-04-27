"""Tests for the 0.9.0 PFAFFIAN_NOT_EML_R extension (S/R-#10).

The 0.3.0 expansion left 36 named SymPy non-elementary functions
treated as depth-0 atoms (verified by direct enumeration of
``sympy.functions.special``). This release adds them with chain
orders derived from each function's defining ODE / closure.

These tests verify that the new entries:

  - exist in the registry,
  - are flagged as Pfaffian-not-EML by ``analyze``,
  - report a non-zero ``pfaffian_r``,
  - don't break detector behaviour for elementary inputs.
"""
from __future__ import annotations

import sympy as sp

from eml_cost import PFAFFIAN_NOT_EML_R, analyze


x = sp.Symbol("x")
y = sp.Symbol("y")
n = sp.Symbol("n", integer=True)


# ---------------------------------------------------------------------------
# Registry-level invariants
# ---------------------------------------------------------------------------


def test_registry_has_at_least_60_entries():
    """The 0.9.0 expansion target was >= 60 distinct entries."""
    assert len(PFAFFIAN_NOT_EML_R) >= 60, (
        f"PFAFFIAN_NOT_EML_R has {len(PFAFFIAN_NOT_EML_R)} entries; "
        f"target was >= 60 after the S/R-#10 expansion."
    )


def test_v090_new_keys_present():
    """All 36 new entries from S/R-#10 are present."""
    new_keys = {
        # Spherical Bessel/Hankel + Marcum
        "jn", "yn", "hn1", "hn2", "marcumq",
        # Erf variants
        "erf2", "erfinv", "erfcinv", "erf2inv",
        # Exponential integrals
        "expint", "E1", "Li",
        # Gamma family extension
        "digamma", "trigamma", "lowergamma", "uppergamma",
        "multigamma", "harmonic",
        "factorial", "factorial2", "subfactorial",
        "RisingFactorial", "FallingFactorial",
        # Elliptic third kind
        "elliptic_pi",
        # Zeta family extension
        "dirichlet_eta", "lerchphi", "stieltjes", "riemann_xi",
        # Hypergeometric
        "meijerg", "appellf1",
        # Mathieu
        "mathieuc", "mathieus", "mathieucprime", "mathieusprime",
        # Spherical harmonics
        "Ynm", "Znm",
    }
    missing = new_keys - set(PFAFFIAN_NOT_EML_R)
    assert not missing, f"missing v0.9.0 entries: {missing}"


def test_all_chain_orders_are_positive_integers():
    """Every registry entry must have an integer chain order >= 2."""
    for name, r in PFAFFIAN_NOT_EML_R.items():
        assert isinstance(r, int), f"{name}: chain order is not int"
        assert r >= 2, f"{name}: chain order {r} < 2"


# ---------------------------------------------------------------------------
# Per-family detector tests
# ---------------------------------------------------------------------------


def test_spherical_bessel_family_flagged():
    """jn, yn, hn1, hn2 — all flagged as PNE."""
    for f in (sp.jn, sp.yn, sp.hn1, sp.hn2):
        a = analyze(f(0, x))
        assert a.is_pfaffian_not_eml is True, f"{f.__name__} should be PNE"
        assert a.pfaffian_r >= 3


def test_marcumq_flagged():
    """Marcum Q is a Bessel-derived integral — PNE."""
    a = analyze(sp.marcumq(1, x, y))
    assert a.is_pfaffian_not_eml is True
    assert a.pfaffian_r >= 3


def test_erf_variants_flagged():
    """erf2 / erfinv / erfcinv / erf2inv — all flagged."""
    cases = [
        sp.erf2(x, y),
        sp.erfinv(x),
        sp.erfcinv(x),
        sp.erf2inv(x, y),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, f"{expr} should be PNE"
        assert a.pfaffian_r >= 2


def test_exp_integrals_extended():
    """expint, E1, Li — flagged."""
    cases = [sp.expint(2, x), sp.E1(x), sp.Li(x)]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True
        assert a.pfaffian_r >= 3


def test_gamma_family_extension():
    """digamma, trigamma, lowergamma, uppergamma — all flagged.

    Note: lowergamma/uppergamma auto-evaluate to elementary forms when
    the first argument is an integer (e.g. lowergamma(2, x) = 1 − (x+1)·exp(−x)),
    so we use a symbolic ``s`` to keep the call unevaluated.
    """
    s = sp.Symbol("s")
    cases = [
        sp.digamma(x), sp.trigamma(x),
        sp.lowergamma(s, x), sp.uppergamma(s, x),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, f"{expr} should be PNE"
        assert a.pfaffian_r >= 2


def test_factorials_and_pochhammer_flagged():
    """factorial / factorial2 / subfactorial / RisingFactorial /
    FallingFactorial — all extend the gamma chain.

    Note: RisingFactorial(x, n) auto-evaluates to a polynomial when ``n``
    is a positive integer literal (e.g. RisingFactorial(x, 2) = x·(x+1)).
    We use a symbolic ``k`` to keep the call unevaluated.
    """
    k = sp.Symbol("k", integer=True, positive=True)
    cases = [
        sp.factorial(x),
        sp.factorial2(x),
        sp.subfactorial(x),
        sp.RisingFactorial(x, k),
        sp.FallingFactorial(x, k),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, f"{expr} should be PNE"
        assert a.pfaffian_r >= 2


def test_multigamma_and_harmonic():
    """multigamma + harmonic — gamma-derived non-elementary."""
    for expr in (sp.multigamma(x, 2), sp.harmonic(x)):
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True
        assert a.pfaffian_r >= 3


def test_elliptic_pi_flagged():
    """Π(n; φ|m) — third kind, completes the elliptic family."""
    a = analyze(sp.elliptic_pi(x, y, sp.Symbol("m")))
    assert a.is_pfaffian_not_eml is True
    assert a.pfaffian_r >= 4


def test_zeta_family_extension():
    """dirichlet_eta, lerchphi, stieltjes, riemann_xi — flagged."""
    cases = [
        sp.dirichlet_eta(x),
        sp.lerchphi(x, 2, sp.Symbol("a")),
        sp.stieltjes(2),
        sp.riemann_xi(x),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True
        assert a.pfaffian_r >= 4


def test_hyper_extended():
    """meijerg + appellf1 — generalised hypergeometric."""
    expr_mj = sp.meijerg([[1], []], [[1], [0]], x)
    expr_ap = sp.appellf1(x, sp.S(1)/2, sp.S(1)/2, sp.S(3)/2, y, x)
    for expr in (expr_mj, expr_ap):
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True


def test_mathieu_family_flagged():
    """mathieuc, mathieus, mathieucprime, mathieusprime — flagged."""
    a_param = sp.Symbol("a")
    q_param = sp.Symbol("q")
    cases = [
        sp.mathieuc(a_param, q_param, x),
        sp.mathieus(a_param, q_param, x),
        sp.mathieucprime(a_param, q_param, x),
        sp.mathieusprime(a_param, q_param, x),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True
        assert a.pfaffian_r >= 4


def test_spherical_harmonics_flagged():
    """Ynm, Znm — flagged via assoc_legendre × Euler."""
    theta = sp.Symbol("theta")
    phi = sp.Symbol("phi")
    cases = [sp.Ynm(2, 1, theta, phi), sp.Znm(2, 1, theta, phi)]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True


# ---------------------------------------------------------------------------
# Regression: elementary inputs still pass through
# ---------------------------------------------------------------------------


def test_v090_does_not_regress_elementary_detection():
    """The new entries shouldn't have accidentally tagged anything elementary."""
    elementary = [
        sp.exp(x), sp.log(x), sp.sin(x), sp.cos(x), sp.tan(x),
        sp.sinh(x), sp.cosh(x), sp.tanh(x),
        sp.exp(sp.sin(x)),
        sp.log(1 + sp.exp(x)),  # softplus
        x ** 2, 1 / x, sp.sqrt(x),
        # sinc(x) = sin(x)/x is elementary by composition
        sp.sinc(x),
        # Polynomials in x (Legendre / Chebyshev / Hermite) — all elementary
        sp.legendre(3, x),
        sp.chebyshevt(4, x),
        sp.hermite(2, x),
    ]
    for expr in elementary:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is False, (
            f"{expr} should NOT be Pfaffian-not-EML"
        )


def test_compositions_still_tag_correctly():
    """A composition containing one PNE primitive should be flagged
    even if the rest of the expression is elementary."""
    pne_compositions = [
        sp.exp(x) + sp.factorial(x),
        sp.sin(x) * sp.elliptic_pi(x, y, sp.Symbol("m")),
        sp.log(1 + sp.harmonic(x)),
        sp.dirichlet_eta(x) ** 2,
    ]
    for expr in pne_compositions:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, (
            f"composition {expr} should be PNE"
        )


# ---------------------------------------------------------------------------
# Adversarial-bench fix: hyper((), (), x) = exp(x) short-circuit
# ---------------------------------------------------------------------------


def test_hyper_empty_args_is_elementary():
    """hyper((), (), x) is the SymPy canonical form of 0F0(;;x) = exp(x).

    SymPy itself doesn't auto-simplify this to exp(x) (verified: .doit()
    is a no-op). The detector short-circuits this case to elementary.
    Surfaced by adversarial bench rows 29 + 30.
    """
    cases = [
        sp.hyper([], [], x),
        sp.hyper([1], [1], x),  # SymPy cancels [1]/[1] → empty/empty
        sp.exp(sp.hyper([], [], x)),  # composition wrapper
        sp.hyper([], [], x) + sp.sin(x),  # additive composition
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is False, (
            f"{expr} should NOT be flagged PNE — it equals exp(x)"
        )


def test_hyper_with_real_params_still_flagged():
    """Non-empty hyper parameter sequences MUST still be flagged PNE.
    The 0F0 short-circuit is narrow and shouldn't open a hole."""
    a_param = sp.Symbol("a")
    b_param = sp.Symbol("b")
    cases = [
        sp.hyper([a_param], [], x),         # 1F0
        sp.hyper([a_param, b_param], [], x),
        sp.hyper([a_param], [b_param], x),  # 1F1 with nontrivial params
        sp.hyper([sp.S(1)/2], [sp.S(3)/2], x),
    ]
    for expr in cases:
        a = analyze(expr)
        assert a.is_pfaffian_not_eml is True, (
            f"{expr} should be flagged PNE — non-trivial hypergeometric"
        )
