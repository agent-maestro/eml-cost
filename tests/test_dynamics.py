"""Tests for eml_cost.analyze_dynamics.

The analyze_dynamics counter implements the slope-2 rule
``r ≈ 2 * n_oscillations + 1 * n_decays`` calibrated on E-196 Phase 4f
(n=175, ρ=0.890). Tests cover:

  - basic mode counting (oscillation / decay / static)
  - the canonical r=0/1/2/3 prototypes
  - confidence and domain_fit semantics
  - PNE primitives produce confidence='low'
"""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import DynamicsProfile, analyze_dynamics


x, t, omega, omega_d, beta, zeta, A, k, T = sp.symbols(
    "x t omega omega_d beta zeta A k T", real=True
)


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------


def test_returns_dynamics_profile() -> None:
    p = analyze_dynamics("A * cos(omega * t)")
    assert isinstance(p, DynamicsProfile)
    assert hasattr(p, "n_oscillations")
    assert hasattr(p, "predicted_r")
    assert hasattr(p, "actual_r")
    assert hasattr(p, "confidence")
    assert hasattr(p, "description")


def test_accepts_string_and_sympy() -> None:
    p1 = analyze_dynamics("A * cos(omega * t)")
    p2 = analyze_dynamics(A * sp.cos(omega * t))
    assert p1.n_oscillations == p2.n_oscillations
    assert p1.predicted_r == p2.predicted_r


# ---------------------------------------------------------------------------
# Mode counting
# ---------------------------------------------------------------------------


def test_polynomial_zero_modes() -> None:
    """T**4 should be 0 oscillations, 0 decays, 1 static."""
    p = analyze_dynamics(T**4)
    assert p.n_oscillations == 0
    assert p.n_decays == 0
    assert p.predicted_r == 0


def test_constant_zero_modes() -> None:
    p = analyze_dynamics(sp.Integer(42))
    assert p.n_oscillations == 0
    assert p.n_decays == 0
    assert p.predicted_r == 0


def test_single_cosine_one_oscillation() -> None:
    """Faraday EMF: B*A*omega*sin(omega*t) - 1 oscillation, predicted r=2."""
    B = sp.Symbol("B", real=True)
    p = analyze_dynamics(B * A * omega * sp.sin(omega * t))
    assert p.n_oscillations == 1
    assert p.n_decays == 0
    assert p.predicted_r == 2


def test_sin_and_cos_same_arg_count_as_one() -> None:
    """sin(omega*t) + cos(omega*t) coalesces to ONE oscillation mode."""
    expr = sp.sin(omega * t) + sp.cos(omega * t)
    p = analyze_dynamics(expr)
    assert p.n_oscillations == 1


def test_single_exp_decay() -> None:
    """exp(-k*t) - 1 decay mode, predicted r=1."""
    p = analyze_dynamics(sp.exp(-k * t))
    assert p.n_oscillations == 0
    assert p.n_decays == 1
    assert p.predicted_r == 1


def test_damped_oscillator_one_osc_one_decay() -> None:
    """A*exp(-zeta*omega*t)*cos(omega_d*t) - 1 osc, 1 decay, predicted r=3."""
    expr = A * sp.exp(-zeta * omega * t) * sp.cos(omega_d * t)
    p = analyze_dynamics(expr)
    assert p.n_oscillations == 1
    assert p.n_decays == 1
    assert p.predicted_r == 3


def test_two_independent_oscillations() -> None:
    """sin(omega1*t) + sin(omega2*t) - 2 distinct oscillation modes."""
    omega1, omega2 = sp.symbols("omega1 omega2", real=True)
    expr = sp.sin(omega1 * t) + sp.sin(omega2 * t)
    p = analyze_dynamics(expr)
    assert p.n_oscillations == 2
    assert p.predicted_r == 4


def test_triple_gaussian_three_decays() -> None:
    """Three parallel Gaussian channels - 3 decay modes."""
    mu1, mu2, mu3, sig1, sig2, sig3, a1, a2, a3 = sp.symbols(
        "mu1 mu2 mu3 sig1 sig2 sig3 a1 a2 a3", real=True
    )
    lam = sp.Symbol("lam", real=True)
    expr = (
        a1 * sp.exp(-((lam - mu1) / sig1) ** 2)
        + a2 * sp.exp(-((lam - mu2) / sig2) ** 2)
        + a3 * sp.exp(-((lam - mu3) / sig3) ** 2)
    )
    p = analyze_dynamics(expr)
    assert p.n_decays == 3
    assert p.n_oscillations == 0
    assert p.predicted_r == 3


# ---------------------------------------------------------------------------
# Confidence + domain_fit semantics
# ---------------------------------------------------------------------------


def test_oscillation_heavy_high_confidence() -> None:
    """Oscillation-heavy expressions get high confidence (rho >0.9 domain)."""
    p = analyze_dynamics(A * sp.cos(omega * t))
    assert p.confidence == "high"


def test_polynomial_high_confidence() -> None:
    """Pure polynomial gets high confidence (unambiguous)."""
    p = analyze_dynamics(T**4)
    assert p.confidence == "high"


def test_pne_primitive_low_confidence() -> None:
    """Bessel J_0 is PNE - counter abstains via low confidence."""
    p = analyze_dynamics(sp.besselj(0, x))
    assert p.confidence == "low"


def test_decay_only_moderate_confidence() -> None:
    """Pure decay (no oscillation) is in the moderate-confidence band."""
    p = analyze_dynamics(sp.exp(-k * t))
    assert p.confidence == "moderate"


def test_domain_fit_strong_for_close_match() -> None:
    """Faraday EMF: predicted_r = actual_r = 2 -> strong fit."""
    B = sp.Symbol("B", real=True)
    p = analyze_dynamics(B * A * omega * sp.sin(omega * t))
    assert p.domain_fit == "strong"


def test_description_includes_mode_counts() -> None:
    """Description should mention the modes found."""
    p = analyze_dynamics(A * sp.cos(omega * t))
    assert "oscillation" in p.description
    p2 = analyze_dynamics(sp.exp(-k * t))
    assert "decay" in p2.description
    p3 = analyze_dynamics(T**4)
    assert "polynomial" in p3.description or "constant" in p3.description


# ---------------------------------------------------------------------------
# Honest framing
# ---------------------------------------------------------------------------


def test_honest_note_present() -> None:
    p = analyze_dynamics(A * sp.cos(omega * t))
    assert "OBSERVATION" in p.honest_note
    assert "0.890" in p.honest_note or "0.89" in p.honest_note
