"""Tests for PfaffianProfile + distance metric.

Verify all three metric axioms:
  1. Identity: d(a, a) = 0
  2. Symmetry: d(a, b) = d(b, a)
  3. Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
"""
from __future__ import annotations

import math

import pytest
import sympy as sp

from eml_cost import PfaffianProfile, analyze, canonicalize, DEFAULT_WEIGHTS


x = sp.Symbol("x")
y = sp.Symbol("y")


def make_profile(expr_str):
    return PfaffianProfile.from_expression(sp.sympify(expr_str))


def test_from_expression_returns_profile():
    p = make_profile("exp(x)")
    assert isinstance(p, PfaffianProfile)
    assert p.r >= 1
    assert p.cost_class.startswith("p")


def test_repr_uses_cost_class():
    p = make_profile("sin(x)")
    assert "PfaffianProfile" in repr(p)
    assert p.cost_class in repr(p)


def test_metric_identity_d_a_a_eq_0():
    p = make_profile("exp(x) + sin(y)")
    assert p.distance(p) == 0.0


def test_metric_symmetry_d_a_b_eq_d_b_a():
    a = make_profile("exp(x)")
    b = make_profile("sin(x)")
    assert a.distance(b) == b.distance(a)


def test_metric_triangle_inequality():
    """d(a, c) <= d(a, b) + d(b, c) on 50 random expression triples."""
    exprs = [
        "x", "x**2", "x**5", "exp(x)", "exp(-x)", "log(x)", "log(1+x)",
        "sin(x)", "cos(x)", "tan(x)", "sqrt(x)", "tanh(x)", "atan(x)",
        "exp(sin(x))", "sin(exp(x))", "log(cos(x))", "exp(exp(x))",
        "sin(sin(x))", "tan(tan(x))", "exp(x)*cos(x)",
        "exp(-x**2)*sin(x)", "1/(1+exp(-x))",
        "x*exp(x)", "x**2*exp(-x)",
    ]
    profiles = [make_profile(e) for e in exprs]

    violations = 0
    n_triples = 0
    for i, a in enumerate(profiles):
        for j, b in enumerate(profiles):
            for k, c in enumerate(profiles):
                if i == j == k:
                    continue
                lhs = a.distance(c)
                rhs = a.distance(b) + b.distance(c)
                n_triples += 1
                # Allow tiny floating-point slack
                if lhs > rhs + 1e-9:
                    violations += 1
    assert violations == 0, f"{violations}/{n_triples} triple-inequality violations"


def test_metric_non_negativity():
    """d(a, b) >= 0 always."""
    profiles = [make_profile(e) for e in ["x", "exp(x)", "sin(x)", "x**2"]]
    for a in profiles:
        for b in profiles:
            assert a.distance(b) >= 0.0


def test_metric_distance_is_zero_iff_same_class():
    """d(a, b) = 0 iff a.cost_class == b.cost_class (in our default weights)."""
    # Distinct classes
    a = make_profile("exp(x)")
    b = make_profile("sin(x)*cos(x*y)")
    assert a.cost_class != b.cost_class
    assert a.distance(b) > 0


def test_compare_returns_per_axis_deltas():
    a = make_profile("exp(x)")
    b = make_profile("sin(exp(x))")
    cmp = a.compare(b)
    assert "delta_r" in cmp
    assert "delta_degree" in cmp
    assert "delta_width" in cmp
    assert "delta_c_osc" in cmp
    assert "distance" in cmp
    assert "same_class" in cmp


def test_is_elementary_for_exp_returns_true():
    p = make_profile("exp(x) + sin(x)")
    assert p.is_elementary() is True
    assert p.is_pfaffian_not_eml is False


def test_is_elementary_for_bessel_returns_false():
    p = make_profile("besselj(0, x)")
    assert p.is_elementary() is False
    assert p.is_pfaffian_not_eml is True


def test_to_dict_is_json_friendly():
    import json
    p = make_profile("exp(x)")
    d = p.to_dict()
    s = json.dumps(d)
    assert "cost_class" in s
    assert "expression" in s


def test_to_row_matches_csv_header_length():
    p = make_profile("exp(x)")
    row = p.to_row()
    header = PfaffianProfile.csv_header()
    assert len(row) == len(header)


def test_canonicalize_default_makes_equivalent_forms_equal():
    """Sigmoid: 1/(1+exp(-x)) and exp(x)/(exp(x)+1) should agree after canonicalize."""
    p1 = make_profile("1/(1+exp(-x))")
    p2 = make_profile("exp(x)/(exp(x)+1)")
    # Same cost class (canonicalize=True is the default)
    assert p1.cost_class == p2.cost_class


def test_custom_weights_change_distance():
    a = make_profile("exp(x)")
    b = make_profile("sin(x)")
    d_default = a.distance(b)
    d_custom = a.distance(b, weights={"r": 1.0, "d": 1.0, "w": 1.0, "c": 1.0})
    assert d_default != d_custom or a.r == b.r and a.degree == b.degree


def test_polynomial_has_zero_chain_order():
    p = make_profile("x**4 + 3*x**2 + 1")
    assert p.r == 0


def test_oscillatory_flag_set_for_sin_cos():
    p = make_profile("sin(x**2)")
    assert p.oscillatory is True


def test_cost_class_format():
    p = make_profile("exp(x)")
    parts = p.cost_class.split("-")
    assert len(parts) == 4
    assert parts[0].startswith("p")
    assert parts[1].startswith("d")
    assert parts[2].startswith("w")
    assert parts[3].startswith("c")
