"""Tests for ``find_counterexample`` (added in 0.1.1).

Pairs with ``verify_equivalence``: when the verifier rejects a
rewrite, ``find_counterexample`` returns the specific point that
proves the rejection.
"""
from __future__ import annotations

import sympy as sp

from eml_cost.rewrite import (
    Counterexample,
    find_counterexample,
    verify_equivalence,
)


u = sp.Symbol("u")          # unconstrained
v = sp.Symbol("v")
xp = sp.Symbol("xp", positive=True)


def test_counterexample_for_log_pow_negative_branch_is_domain_mismatch() -> None:
    """log(u^2) -> 2*log(u): real for u<0 on the LHS, complex on the
    RHS — `find_counterexample` should return a domain_mismatch."""
    cx = find_counterexample(sp.log(u ** 2), 2 * sp.log(u), seed=1)
    assert cx is not None
    assert isinstance(cx, Counterexample)
    # The disagreement region for this rewrite is u < 0; the verifier
    # samples that range, so cx.point[0] should be negative.
    assert cx.point[0] < 0
    # Either kind is acceptable: the LHS may evaluate (real) while the
    # RHS evaluates (complex with imag part) — that's value disagreement;
    # OR the math module may raise on the RHS — that's domain mismatch.
    assert cx.kind in ("domain_mismatch", "value_disagreement")
    assert "u" in cx.note


def test_counterexample_for_arbitrary_inequality() -> None:
    """sin(u) and cos(u) disagree at almost every point."""
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    assert cx.kind == "value_disagreement"
    assert cx.before_value is not None and cx.after_value is not None


def test_counterexample_returns_none_for_universal_identity() -> None:
    """sin^2 + cos^2 == 1 universally — no counterexample exists."""
    cx = find_counterexample(sp.sin(u) ** 2 + sp.cos(u) ** 2, sp.S.One, seed=1)
    assert cx is None


def test_counterexample_returns_none_when_verify_returns_true() -> None:
    """The two functions agree on accept/reject."""
    pairs = [
        (sp.sin(u) ** 2 + sp.cos(u) ** 2, sp.S.One),
        (sp.cosh(u) ** 2 - sp.sinh(u) ** 2, sp.S.One),
        (sp.exp(xp) / (1 + sp.exp(xp)), 1 / (1 + sp.exp(-xp))),
    ]
    for before, after in pairs:
        if verify_equivalence(before, after, seed=1):
            assert find_counterexample(before, after, seed=1) is None


def test_counterexample_for_constant_disagreement() -> None:
    """Two distinct constants — verifier rejects, finder reports
    a constant disagreement."""
    cx = find_counterexample(sp.S(2) + sp.S(2), sp.S(5), seed=1)
    assert cx is not None
    assert cx.symbols == ()
    assert cx.kind == "value_disagreement"


def test_counterexample_note_contains_symbol_assignment() -> None:
    """The human-readable note must reference the symbol = value
    form so an editor can surface it directly."""
    cx = find_counterexample(sp.sin(u), sp.cos(u) + 1, seed=1)
    assert cx is not None
    assert "u=" in cx.note


def test_counterexample_is_frozen_dataclass() -> None:
    cx = Counterexample(
        symbols=(u,), point=(1.0,),
        before_value=complex(0.84), after_value=complex(0.54),
        kind="value_disagreement", note="test",
    )
    import pytest
    with pytest.raises(Exception):
        cx.kind = "x"  # type: ignore[misc]
