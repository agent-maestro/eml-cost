"""Tests for ``eml_cost.fingerprint`` (added in 0.1.1)."""
from __future__ import annotations

import re

import sympy as sp

from eml_cost import analyze, fingerprint


x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)


_FINGERPRINT_RE = re.compile(r"^p(\d+)-d(\d+)-w(\d+)-c(-?\d+)-h([0-9a-f]{6})$")


def test_fingerprint_format_matches_grammar() -> None:
    fp = fingerprint(sp.exp(sp.sin(x)))
    assert _FINGERPRINT_RE.match(fp), f"unexpected format: {fp}"


def test_fingerprint_is_short() -> None:
    """Compact: under 25 characters for any reasonable expression."""
    fp = fingerprint(sp.exp(sp.exp(sp.exp(x))))
    assert len(fp) <= 25, f"too long: {fp!r}"


def test_fingerprint_is_deterministic() -> None:
    expr = sp.cosh(x) ** 2 - sp.sinh(x) ** 2
    assert fingerprint(expr) == fingerprint(expr)


def test_fingerprint_axes_match_analyze_result() -> None:
    """The numeric axes encoded in the fingerprint must agree with
    what analyze() returns for the same expression."""
    expr = sp.exp(sp.sin(x))
    fp = fingerprint(expr)
    a = analyze(expr)
    m = _FINGERPRINT_RE.match(fp)
    assert m is not None
    p, d, w, c = (int(m.group(i)) for i in range(1, 5))
    assert p == a.pfaffian_r
    assert d == a.eml_depth
    assert w == a.max_path_r
    expected_c = (a.corrections.c_osc + a.corrections.c_composite
                  - a.corrections.delta_fused)
    assert c == expected_c


def test_fingerprint_distinguishes_eml_from_extension_class() -> None:
    """Bessel J0 and a structurally similar transcendental in the EML
    class must NOT share a fingerprint, even if their numeric axes
    happen to agree."""
    bessel = sp.besselj(0, x)
    fp1 = fingerprint(bessel)
    fp2 = fingerprint(sp.exp(sp.sin(x)))   # different shape, different name
    assert fp1 != fp2


def test_fingerprint_collides_for_truly_identical_expressions() -> None:
    """Two distinct construction paths that produce the same SymPy
    expression must yield the same fingerprint."""
    a1 = sp.sin(x) ** 2 + sp.cos(x) ** 2
    a2 = sp.cos(x) ** 2 + sp.sin(x) ** 2  # SymPy normalizes Add ordering
    assert a1 == a2  # sanity
    assert fingerprint(a1) == fingerprint(a2)


def test_fingerprint_changes_with_depth() -> None:
    """exp(x) and exp(exp(x)) have different chain orders → different fps."""
    assert fingerprint(sp.exp(x)) != fingerprint(sp.exp(sp.exp(x)))


def test_fingerprint_works_on_string_input() -> None:
    """Inherits analyze()'s string-input convenience."""
    fp = fingerprint("exp(sin(x))")
    assert _FINGERPRINT_RE.match(fp)
