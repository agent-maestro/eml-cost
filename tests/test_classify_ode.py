"""Tests for eml_cost.classify_ode and the PNE-registry validator.

classify_ode reduces y'' + p y' + q y = 0 to normal form u''=r u and reads the
EML class off the oscillation of r (Sturm = the Kovacic local-exponent -1/4
threshold = the compact differential-Galois torus factor). See
monogate-research exploration/differential_galois_eml_depth_2026_06_24/.
"""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import (
    OdeClass,
    classify_ode,
    validate_pne_registry,
    RegistryValidation,
)

x = sp.Symbol("x", real=True)


# ── the discriminant-flip pairs: same family, opposite EML verdict ──
def test_harmonic_oscillates_eml_infinity():
    res = classify_ode(0, 1)  # y'' + y = 0
    assert res.eml_class == "EML-infinity"
    assert res.oscillatory and res.definitive


def test_hyperbolic_split_eml_finite():
    res = classify_ode(0, -1)  # y'' - y = 0
    assert res.eml_class == "EML-finite"
    assert not res.oscillatory and res.definitive


def test_underdamped_vs_overdamped():
    assert classify_ode(2, 5).eml_class == "EML-infinity"     # complex roots
    assert classify_ode(3, 2).eml_class == "EML-finite"       # real roots


def test_euler_real_vs_complex():
    # x²y'' + xy' - y = 0  (real indicial roots) vs  + y = 0 (complex)
    assert classify_ode(1 / x, -1 / x**2, "pos").eml_class == "EML-finite"
    assert classify_ode(1 / x, 1 / x**2, "pos").eml_class == "EML-infinity"


def test_repeated_root_zero_potential():
    res = classify_ode(-2, 1)  # y'' - 2y' + y = 0 -> r = 0
    assert res.r == 0
    assert res.eml_class == "EML-finite" and res.definitive


# ── the headline non-torus case: Airy ──
def test_airy_case1_impossible_and_oscillatory():
    res = classify_ode(0, -x, "R")  # y'' - x y = 0
    assert res.order_at_infinity == -1
    assert res.case1_possible is False          # Kovacic Case-1 screen rules it out
    assert res.oscillatory                       # r = x -> -oo as x -> -oo
    assert res.eml_class == "EML-infinity"


# ── modified Bessel: non-oscillatory genuine special fn -> non-definitive ──
def test_modified_bessel_is_candidate_not_finite():
    res = classify_ode(1 / x, -(1 + 4 / x**2), "pos")  # I_2 ODE
    assert not res.oscillatory
    assert res.eml_class == "EML-finite?"        # NOT a flat EML-finite
    assert res.definitive is False


# ── string / int coercion ──
def test_accepts_strings():
    assert classify_ode("0", "1").eml_class == "EML-infinity"


def test_result_is_frozen_dataclass():
    res = classify_ode(0, 1)
    assert isinstance(res, OdeClass)
    with pytest.raises(Exception):
        res.eml_class = "x"  # frozen


# ── the registry validator ──
def test_validate_pne_registry_no_contradictions():
    v = validate_pne_registry()
    assert isinstance(v, RegistryValidation)
    assert v.contradictions == ()                # never calls a registry entry EML-finite
    assert len(v.confirmed) >= 10                # Bessel-ordinary + Hankel + spherical + Airy
    assert "airyai" in v.confirmed
    assert "besselj" in v.confirmed
    assert "besseli" in v.consistent             # modified Bessel: non-definitive
    # coverage accounting is exact
    assert v.ode_covered + len(v.out_of_scope) == v.total_registry


def test_validate_confirmed_are_all_eml_infinity():
    from eml_cost import REGISTRY_ODES
    v = validate_pne_registry()
    for name in v.confirmed:
        p, q, dom = REGISTRY_ODES[name]
        assert classify_ode(p, q, dom).eml_class == "EML-infinity"
