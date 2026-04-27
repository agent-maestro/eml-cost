"""Tests for the eml_cost.recommend_form narrow-scope form recommender.

Ships in 0.8.0. Fires on 4 expression families with E-193 Phase 3
Spearman rho >= 0.77; abstains on everything else.
"""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import (
    FAMILY_RHO,
    RecommendedForm,
    SUPPORTED_FAMILIES,
    recommend_form,
)

x, y, t, k, K, r, t0, omega, amp, A, tau = sp.symbols(
    "x y t k K r t0 omega amp A tau", real=True
)


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------


def test_supported_families_constant() -> None:
    assert set(SUPPORTED_FAMILIES) == {
        "sigmoid",
        "exponential_decay",
        "logistic_growth",
        "cardiac_oscillator",
    }


def test_family_rho_matches_E193_phase3() -> None:
    """Spearman rho values come from phase3_killer.json. Catch silent drift."""
    assert FAMILY_RHO["sigmoid"] == pytest.approx(0.800, abs=0.01)
    assert FAMILY_RHO["exponential_decay"] == pytest.approx(0.775, abs=0.01)
    assert FAMILY_RHO["logistic_growth"] == pytest.approx(0.775, abs=0.01)
    assert FAMILY_RHO["cardiac_oscillator"] == pytest.approx(0.866, abs=0.01)
    # Sanity: all four are above the 0.7 product threshold.
    for f in SUPPORTED_FAMILIES:
        assert FAMILY_RHO[f] > 0.7


def test_recommend_form_accepts_string() -> None:
    r = recommend_form("1/(1 + exp(-x))")
    assert isinstance(r, RecommendedForm)


def test_recommend_form_accepts_sympy_expr() -> None:
    r = recommend_form(1 / (1 + sp.exp(-x)))
    assert isinstance(r, RecommendedForm)


def test_returns_None_when_no_family_matches() -> None:
    """Polynomials and unrelated transcendentals should abstain cleanly."""
    assert recommend_form("x**2 + 3*x + 1") is None
    assert recommend_form("besselj(0, x)") is None
    assert recommend_form("erf(x)") is None
    assert recommend_form("gamma(x)") is None


# ---------------------------------------------------------------------------
# Sigmoid family — 4 alternative forms must all match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_str",
    [
        "1/(1 + exp(-x))",        # canonical
        "exp(x)/(exp(x) + 1)",     # equivalent
        "tanh(x/2)/2 + 1/2",       # equivalent (the form Phase 3 flagged as drifting)
        "1 - 1/(1 + exp(x))",      # equivalent
    ],
)
def test_sigmoid_alternative_forms_all_detected(expr_str: str) -> None:
    r = recommend_form(expr_str)
    assert r is not None, f"Sigmoid form not detected: {expr_str}"
    assert r.family == "sigmoid"


def test_sigmoid_canonical_form_is_one_over_one_plus_exp_neg_x() -> None:
    r = recommend_form("tanh(x/2)/2 + 1/2")
    assert r is not None
    # The canonical form should be structurally `1 / (1 + exp(-<x>))`.
    # We verify by string-form on the SymPy default-typed symbol 'x'
    # that sympify created, so we don't trip on local-vs-parsed symbol
    # identity.
    assert str(r.canonical_form).replace(" ", "") == "1/(1+exp(-x))"


def test_sigmoid_rho_carried_through() -> None:
    r = recommend_form("1/(1 + exp(-x))")
    assert r is not None
    assert r.rho == pytest.approx(0.800, abs=0.01)


# ---------------------------------------------------------------------------
# Exponential decay family
# ---------------------------------------------------------------------------


def test_exp_decay_simple_form_detected() -> None:
    r = recommend_form("exp(-k*t)")
    assert r is not None
    assert r.family == "exponential_decay"


def test_exp_decay_division_form_detected() -> None:
    """The drift form: division in the argument."""
    r = recommend_form("exp(-t/tau)")
    assert r is not None
    assert r.family == "exponential_decay"


def test_exp_decay_canonical_form_collapses_division() -> None:
    """For exp(-t/tau), the canonical form is exp(coeff * t) where the
    coefficient is folded into a single multiplicative term."""
    r = recommend_form("exp(-t/tau)")
    assert r is not None
    cf = r.canonical_form
    assert isinstance(cf, sp.exp)
    inner = cf.args[0]
    # The inner argument should reference both t and tau (the symbols
    # are taken from the parsed expression, so we identify by name).
    sym_names = {s.name for s in inner.free_symbols}
    assert "t" in sym_names
    assert "tau" in sym_names


def test_exp_decay_positive_coefficient_still_detected() -> None:
    """Family detection is structural — exp(c*t) is always exp_decay even
    when c is positive (the user's choice of sign assumption is honest)."""
    r = recommend_form("exp(k*t)")
    assert r is not None
    assert r.family == "exponential_decay"


# ---------------------------------------------------------------------------
# Logistic growth family
# ---------------------------------------------------------------------------


def test_logistic_growth_canonical_detected() -> None:
    r = recommend_form("K/(1 + exp(-r*(t - t0)))")
    assert r is not None
    assert r.family == "logistic_growth"


def test_logistic_growth_alternative_form_detected() -> None:
    """The (-t + z) form is algebraically equivalent."""
    r = recommend_form("K/(exp(r*(-t + t0)) + 1)")
    assert r is not None
    assert r.family == "logistic_growth"


# ---------------------------------------------------------------------------
# Cardiac oscillator family (cosine oscillator)
# ---------------------------------------------------------------------------


def test_cosine_oscillator_simple_form_detected() -> None:
    r = recommend_form("amp * cos(omega * t)")
    assert r is not None
    assert r.family == "cardiac_oscillator"


def test_cosine_oscillator_returns_high_rho() -> None:
    r = recommend_form("amp * cos(omega * t)")
    assert r is not None
    assert r.rho == pytest.approx(0.866, abs=0.01)


# ---------------------------------------------------------------------------
# Abstain behavior on the 6 non-concordant families from E-193 Phase 3
# ---------------------------------------------------------------------------


def test_abstain_on_hill_kernel() -> None:
    """hill_kernel was rho=0.0 in E-193 Phase 3 — must abstain."""
    n = sp.Symbol("n", real=True)
    K_h = sp.Symbol("K_h", real=True)
    expr = x**n / (K_h**n + x**n)
    assert recommend_form(expr) is None


def test_abstain_on_softmax_2key() -> None:
    """softmax_2key was rho=-0.1 in E-193 Phase 3 — must abstain."""
    a, b = sp.symbols("a b", real=True)
    expr = sp.exp(a) / (sp.exp(a) + sp.exp(b))
    assert recommend_form(expr) is None


def test_abstain_on_polynomial() -> None:
    """Polynomials are not in any of the 4 families."""
    assert recommend_form("x**3 + 2*x**2 - 5*x + 1") is None


def test_abstain_on_pne_function() -> None:
    """Pfaffian-not-EML functions (Bessel, etc.) are not in any family."""
    assert recommend_form("besselj(0, x)") is None
    assert recommend_form("airyai(x)") is None
    assert recommend_form("lambertw(x)") is None


# ---------------------------------------------------------------------------
# Honest-framing checks
# ---------------------------------------------------------------------------


def test_honest_note_present() -> None:
    """The honest framing must travel with every recommendation so that
    consumers see the four-family scope warning."""
    r = recommend_form("1/(1 + exp(-x))")
    assert r is not None
    assert "Narrow-scope" in r.honest_note
    assert "abstains" in r.honest_note.lower()
    assert "general-purpose" in r.honest_note.lower()


def test_recommendation_carries_predicted_relerr_for_both_forms() -> None:
    """The user can see what predict_precision_loss says about the input
    AND about the canonical form, so the digits_saved value is auditable."""
    r = recommend_form("tanh(x/2)/2 + 1/2")
    assert r is not None
    assert r.input_predicted_relerr > 0
    assert r.recommended_predicted_relerr > 0


def test_digits_saved_is_a_finite_float() -> None:
    r = recommend_form("tanh(x/2)/2 + 1/2")
    assert r is not None
    import math
    assert math.isfinite(r.digits_saved)


# ---------------------------------------------------------------------------
# Canonical-form invariance: applying recommend_form to the canonical
# form should return the canonical form back (idempotent on the recommendation)
# ---------------------------------------------------------------------------


def test_sigmoid_canonical_is_idempotent() -> None:
    """Applying the recommender to its own output should match the same
    family and produce the same canonical form."""
    r1 = recommend_form("1/(1 + exp(-x))")
    assert r1 is not None
    r2 = recommend_form(r1.canonical_form)
    assert r2 is not None
    assert r2.family == r1.family
    assert sp.simplify(r2.canonical_form - r1.canonical_form) == 0
