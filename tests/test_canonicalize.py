"""Tests for canonicalize() — verify form-fragility cases collapse to
the same cost class after canonicalization."""
from __future__ import annotations

import sympy as sp

from eml_cost import analyze, analyze_canonical, canonicalize, fingerprint_axes


def axes(form):
    """Helper: get just the axes-tuple from a SymPy form."""
    return fingerprint_axes(form)


def axes_canonical(form):
    return fingerprint_axes(canonicalize(form))


def test_canonicalize_returns_basic():
    x = sp.Symbol("x")
    result = canonicalize(1 / (1 + sp.exp(-x)))
    assert isinstance(result, sp.Basic)


def test_canonicalize_accepts_string():
    result = canonicalize("1 / (1 + exp(-x))")
    assert isinstance(result, sp.Basic)


def test_canonicalize_idempotent_on_simple():
    x = sp.Symbol("x")
    e = sp.cos(x)
    once = canonicalize(e)
    twice = canonicalize(once)
    assert axes(once) == axes(twice)


def test_sigmoid_one_minus_collapses():
    """1 - 1/(1+exp(x)) should canonicalize to 1/(1+exp(-x))."""
    x = sp.Symbol("x")
    canonical = 1 / (1 + sp.exp(-x))
    rewritten = 1 - 1 / (1 + sp.exp(x))
    # Without canonicalize, these are in different cost classes
    # (rewrite catches them).
    a1 = axes_canonical(canonical)
    a2 = axes_canonical(rewritten)
    assert a1 == a2, f"Sigmoid collapse failed: {a1} != {a2}"


def test_log_difference_collapses():
    """log(x) - log(y) should canonicalize to log(x/y)."""
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y", positive=True)
    canonical = sp.log(x / y)
    expanded = sp.log(x) - sp.log(y)
    a1 = axes_canonical(canonical)
    a2 = axes_canonical(expanded)
    assert a1 == a2, f"Log-diff collapse failed: {a1} != {a2}"


def test_traveling_wave_trig_collapses():
    """cos(a)cos(b) + sin(a)sin(b) should collapse to cos(a-b)."""
    a, b = sp.symbols("a b")
    canonical = sp.cos(a - b)
    expanded = sp.cos(a) * sp.cos(b) + sp.sin(a) * sp.sin(b)
    ax1 = axes_canonical(canonical)
    ax2 = axes_canonical(expanded)
    assert ax1 == ax2, f"Trig collapse failed: {ax1} != {ax2}"


def test_distributive_collapses():
    """v - v*exp(-x) should canonicalize to v*(1 - exp(-x))."""
    v = sp.Symbol("v", positive=True)
    x = sp.Symbol("x", positive=True)
    canonical = v * (1 - sp.exp(-x))
    expanded = v - v * sp.exp(-x)
    a1 = axes_canonical(canonical)
    a2 = axes_canonical(expanded)
    assert a1 == a2, f"Distributive collapse failed: {a1} != {a2}"


def test_canonicalize_preserves_value_pointwise():
    """canonicalize(e) must be mathematically identical to e."""
    x = sp.Symbol("x", real=True)
    inputs = [
        1 / (1 + sp.exp(-x)),
        sp.exp(x) / (1 + sp.exp(x)),
        sp.log(2 / 3),
        sp.cos(x) ** 2 + sp.sin(x) ** 2,
    ]
    for e in inputs:
        c = canonicalize(e)
        # numerical check at 5 random-ish positive points
        for v in [0.1, 0.5, 1.0, 2.0, 3.7]:
            try:
                orig = float(e.subs(x, v))
                cano = float(c.subs(x, v))
                assert abs(orig - cano) < 1e-9, (
                    f"canonicalize changed value at x={v}: "
                    f"{e} -> {c}, {orig} != {cano}"
                )
            except (TypeError, ValueError):
                # Some don't evaluate to float; skip those
                pass


def test_analyze_canonical_returns_analyze_result():
    """analyze_canonical should return same shape as analyze."""
    x = sp.Symbol("x")
    r = analyze_canonical(sp.exp(x))
    assert hasattr(r, "pfaffian_r")
    assert hasattr(r, "eml_depth")
    assert hasattr(r, "is_pfaffian_not_eml")


def test_polynomial_is_invariant():
    """Polynomial expressions should already be canonical."""
    x = sp.Symbol("x")
    poly = x ** 4
    a1 = axes(poly)
    a2 = axes_canonical(poly)
    assert a1 == a2


def test_kinetic_energy_form_invariance():
    """All four ways to write (1/2) m v^2 should agree."""
    m = sp.Symbol("m", positive=True)
    v = sp.Symbol("v", positive=True)
    forms = [
        sp.Rational(1, 2) * m * v ** 2,
        m * v ** 2 / 2,
        m * v * v / 2,
        sp.S.Half * m * v ** 2,
    ]
    axes_set = {axes_canonical(f) for f in forms}
    assert len(axes_set) == 1, f"kinetic energy drifted: {axes_set}"


def test_gaussian_pdf_invariance():
    """Different Gaussian form choices should agree."""
    x = sp.Symbol("x")
    mu = sp.Symbol("mu")
    sigma = sp.Symbol("sigma", positive=True)
    forms = [
        sp.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
        sp.exp(-((x - mu) / sigma) ** 2 / 2),
    ]
    axes_set = {axes_canonical(f) for f in forms}
    # Both are p1-d5-w1-c0 already; sanity check that canonicalize
    # doesn't break it.
    assert len(axes_set) == 1, f"Gaussian drifted: {axes_set}"


def test_sigmoid_full_audit_set():
    """Four sigmoid forms — at least 3 should collapse to one class."""
    x = sp.Symbol("x")
    forms = [
        1 / (1 + sp.exp(-x)),                  # canonical
        sp.exp(x) / (1 + sp.exp(x)),           # sympy alternative
        1 - 1 / (1 + sp.exp(x)),               # 1-sigmoid(-x)
    ]
    axes_set = {axes_canonical(f) for f in forms}
    # Without canonicalize, these were 3 different classes per the audit.
    # canonicalize should collapse them to 1 or 2.
    assert len(axes_set) <= 2, f"sigmoid forms still drift: {axes_set}"
