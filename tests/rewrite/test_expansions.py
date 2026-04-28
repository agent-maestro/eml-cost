"""Tests for ``eml_rewrite.expand`` / ``expand_fully`` (added in 0.3.0)."""
from __future__ import annotations

import sympy as sp

from eml_cost.rewrite import (
    EXPANSION_PATTERNS,
    Suggestion,
    expand,
    expand_fully,
    score,
    suggest,
)


x = sp.Symbol("x", real=True, positive=True)


# ---------------------------------------------------------------------------
# Single-pattern expansions
# ---------------------------------------------------------------------------


def test_expand_tanh_to_sinh_cosh() -> None:
    sugg = expand(sp.tanh(x))
    names = {s.pattern_name for s in sugg}
    assert "tanh_to_sinh_cosh" in names
    target = next(s for s in sugg if s.pattern_name == "tanh_to_sinh_cosh")
    assert target.rewritten == sp.sinh(x) / sp.cosh(x)


def test_expand_cosh_to_exp_form() -> None:
    sugg = expand(sp.cosh(x))
    names = {s.pattern_name for s in sugg}
    assert "cosh_to_exp_form" in names
    target = next(s for s in sugg if s.pattern_name == "cosh_to_exp_form")
    assert sp.simplify(target.rewritten - (sp.exp(x) + sp.exp(-x)) / 2) == 0


def test_expand_sinh_to_exp_form() -> None:
    sugg = expand(sp.sinh(x))
    names = {s.pattern_name for s in sugg}
    assert "sinh_to_exp_form" in names
    target = next(s for s in sugg if s.pattern_name == "sinh_to_exp_form")
    assert sp.simplify(target.rewritten - (sp.exp(x) - sp.exp(-x)) / 2) == 0


def test_expand_sigmoid_canonical_to_textbook() -> None:
    """1/(1+exp(-x)) → exp(x)/(1+exp(x))."""
    sugg = expand(1 / (1 + sp.exp(-x)))
    names = {s.pattern_name for s in sugg}
    assert "sigmoid_to_textbook" in names
    target = next(s for s in sugg if s.pattern_name == "sigmoid_to_textbook")
    assert sp.simplify(target.rewritten - sp.exp(x) / (1 + sp.exp(x))) == 0


# ---------------------------------------------------------------------------
# Subterm expansion
# ---------------------------------------------------------------------------


def test_expand_recognizes_tanh_inside_larger_expression() -> None:
    """tanh in a sum / product should still be expanded via subterm walk."""
    sugg = expand(sp.tanh(x) + sp.S.One)
    names = {s.pattern_name for s in sugg}
    assert "tanh_to_sinh_cosh" in names


# ---------------------------------------------------------------------------
# No reverse for information-destroying simplifications
# ---------------------------------------------------------------------------


def test_expand_returns_empty_for_bare_one() -> None:
    """`1` has no recoverable expansion target — Pythagorean and
    hyperbolic identities are explicitly NOT reversed."""
    assert expand(sp.S.One) == []


def test_expand_returns_empty_for_bare_symbol() -> None:
    """`x` has no recoverable expansion target — exp(log(x)) is
    explicitly NOT reversed (any expression could "expand" to its
    exp/log form, which would be unbounded)."""
    assert expand(x) == []


def test_expand_returns_empty_for_polynomial() -> None:
    assert expand(x ** 3 + 2 * x + 1) == []


# ---------------------------------------------------------------------------
# Cost direction (expansions typically INCREASE cost)
# ---------------------------------------------------------------------------


def test_expand_tanh_increases_cost() -> None:
    sugg = expand(sp.tanh(x))
    target = next(s for s in sugg if s.pattern_name == "tanh_to_sinh_cosh")
    # tanh(x) is a primitive (cost 1); sinh(x)/cosh(x) is structurally
    # heavier (cost 3 in the locked numbers).
    assert target.score_after > target.score_before
    assert target.reduction < 0   # cost INCREASES → reduction is negative


# ---------------------------------------------------------------------------
# Suggestion-shape compatibility (mirrors suggest())
# ---------------------------------------------------------------------------


def test_expand_returns_proper_suggestion_dataclass() -> None:
    sugg = expand(sp.tanh(x))
    assert all(isinstance(s, Suggestion) for s in sugg)
    s = sugg[0]
    assert s.domain_required == ""
    assert s.domain_verified is True
    assert s.numerically_verified is True


# ---------------------------------------------------------------------------
# expand_fully — recursive deep expansion
# ---------------------------------------------------------------------------


def test_expand_fully_walks_tanh_to_pure_exp_form() -> None:
    """tanh(x) → sinh(x)/cosh(x) → ((exp(x)-exp(-x))/2) / ((exp(x)+exp(-x))/2).
    expand_fully should walk both steps."""
    deep = expand_fully(sp.tanh(x), max_depth=4)
    # The deepest form has no remaining sinh/cosh/tanh.
    s = str(deep)
    assert "tanh" not in s
    assert "sinh" not in s
    assert "cosh" not in s
    assert "exp" in s


def test_expand_fully_terminates_at_fixpoint() -> None:
    """For an expression with no expansions, expand_fully returns
    the input unchanged."""
    expr = x + 1
    assert expand_fully(expr) == expr


def test_expand_fully_respects_max_depth() -> None:
    """max_depth=0 returns the input unchanged."""
    assert expand_fully(sp.tanh(x), max_depth=0) == sp.tanh(x)


def test_expand_fully_preserves_equivalence() -> None:
    """The deeply-expanded form must be symbolically equal to the input.

    We normalize both sides via `.rewrite(sp.exp)` then simplify,
    because plain `sp.simplify` doesn't auto-recognize the
    ``tanh(x) ↔ (exp(x)-exp(-x))/(exp(x)+exp(-x))`` identity.
    """
    inputs = [sp.tanh(x), sp.cosh(x), sp.sinh(x), 1 / (1 + sp.exp(-x))]
    for expr in inputs:
        deep = expand_fully(expr)
        diff = sp.simplify(deep.rewrite(sp.exp) - expr.rewrite(sp.exp))
        assert diff == 0, f"expand_fully changed semantics on {expr}: got {deep}"


# ---------------------------------------------------------------------------
# Round-trip: expand then suggest should give the original (or a
# canonical equivalent) back.
# ---------------------------------------------------------------------------


def test_expand_then_suggest_roundtrips_for_tanh() -> None:
    """tanh(x) → sinh(x)/cosh(x) (expand) → tanh(x) (suggest)."""
    expanded = next(s for s in expand(sp.tanh(x))
                    if s.pattern_name == "tanh_to_sinh_cosh").rewritten
    sugg = suggest(expanded, only_improvements=True)
    assert any(s.rewritten == sp.tanh(x) for s in sugg)


def test_expansion_pattern_registry_has_four_entries() -> None:
    """Sanity: the documented four patterns are all registered."""
    names = {name for name, _ in EXPANSION_PATTERNS}
    assert names == {
        "tanh_to_sinh_cosh",
        "cosh_to_exp_form",
        "sinh_to_exp_form",
        "sigmoid_to_textbook",
    }
