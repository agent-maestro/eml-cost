"""Core API and pattern tests."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost.rewrite import (
    DomainRequirement,
    NO_REQUIREMENT,
    Suggestion,
    best,
    score,
    suggest,
    verify_equivalence,
)


x = sp.Symbol("x", real=True, positive=True)
y = sp.Symbol("y", real=True, positive=True)
# Symbols WITHOUT positivity assumption — used to exercise the domain gate.
u = sp.Symbol("u")
v = sp.Symbol("v")


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_returns_int() -> None:
    assert isinstance(score(sp.exp(x)), int)


def test_score_lower_is_simpler() -> None:
    full = sp.sin(x) ** 2 + sp.cos(x) ** 2
    assert score(full) > score(sp.S.One)


# ---------------------------------------------------------------------------
# Pattern recognition (each must produce a Suggestion with reduction > 0)
# ---------------------------------------------------------------------------


def test_sigmoid_pattern_recognized() -> None:
    expr = sp.exp(x) / (1 + sp.exp(x))
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "sigmoid" for s in sugg)


def test_tanh_from_sinh_cosh_recognized() -> None:
    expr = sp.sinh(x) / sp.cosh(x)
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "tanh_from_sinh_cosh" for s in sugg)


def test_pythagorean_recognized() -> None:
    expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "pythagorean" for s in sugg)


def test_hyperbolic_id_recognized() -> None:
    expr = sp.cosh(x) ** 2 - sp.sinh(x) ** 2
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "hyperbolic_id" for s in sugg)


def test_cosh_from_exps_recognized() -> None:
    expr = (sp.exp(x) + sp.exp(-x)) / 2
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "cosh_from_exps" for s in sugg)


def test_sinh_from_exps_recognized() -> None:
    expr = (sp.exp(x) - sp.exp(-x)) / 2
    sugg = suggest(expr, only_improvements=True)
    assert any(s.pattern_name == "sinh_from_exps" for s in sugg)


# ---------------------------------------------------------------------------
# Provable non-regression (THE patent-relevant claim)
# ---------------------------------------------------------------------------


def test_no_regression_across_panel() -> None:
    """Every Suggestion returned with only_improvements=True has score_after < score_before."""
    panel = [
        sp.exp(x) / (1 + sp.exp(x)),
        sp.sinh(x) / sp.cosh(x),
        sp.sin(x) ** 2 + sp.cos(x) ** 2,
        sp.cosh(x) ** 2 - sp.sinh(x) ** 2,
        (sp.exp(x) + sp.exp(-x)) / 2,
        (sp.exp(x) - sp.exp(-x)) / 2,
        sp.exp(sp.log(x)),
        # Cases with no improvement available
        sp.exp(x),
        sp.sin(x),
        sp.tan(x),
        sp.exp(sp.exp(x)),
        sp.log(1 + sp.exp(x)),
        x ** 5,
    ]
    for expr in panel:
        for s in suggest(expr, only_improvements=True):
            assert s.score_after < s.score_before, (
                f"Regression on {expr}: pattern={s.pattern_name} "
                f"before={s.score_before} after={s.score_after}"
            )


def test_best_non_regressing() -> None:
    """best(expr) never returns an expression with higher score."""
    panel = [
        sp.exp(x) / (1 + sp.exp(x)),
        sp.exp(x),
        sp.sin(x) ** 2 + sp.cos(x) ** 2,
        sp.exp(sp.exp(x)),
        x + y,
        sp.tan(x),
    ]
    for expr in panel:
        before = score(expr)
        after = score(best(expr))
        assert after <= before, f"Regression: {expr} -> {best(expr)} ({before} -> {after})"


def test_best_returns_original_when_no_improvement() -> None:
    expr = sp.exp(sp.exp(x))  # nothing in pattern library matches
    assert best(expr) == expr


# ---------------------------------------------------------------------------
# Suggestion dataclass shape
# ---------------------------------------------------------------------------


def test_suggestion_is_frozen() -> None:
    s = Suggestion(
        pattern_name="x", rewritten=sp.S.One,
        score_before=5, score_after=3, reduction=2,
    )
    with pytest.raises(Exception):
        s.score_before = 99  # type: ignore[misc]


def test_suggestion_default_domain_fields() -> None:
    """Unconditional rewrites carry empty domain metadata + verified flags."""
    expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
    sugg = suggest(expr)
    pyth = next(s for s in sugg if s.pattern_name == "pythagorean")
    assert pyth.domain_required == ""
    assert pyth.domain_verified is True
    assert pyth.numerically_verified is True


# ---------------------------------------------------------------------------
# Reductions on hand-checked cases
# ---------------------------------------------------------------------------


def test_pythagorean_reduction_at_least_3() -> None:
    """sin^2 + cos^2 should reduce by at least 3 cost units."""
    sugg = suggest(sp.sin(x) ** 2 + sp.cos(x) ** 2, only_improvements=True)
    pyth = next((s for s in sugg if s.pattern_name == "pythagorean"), None)
    assert pyth is not None
    assert pyth.reduction >= 3


def test_sigmoid_reduces_score() -> None:
    expr = sp.exp(x) / (1 + sp.exp(x))
    rewritten = best(expr)
    assert score(rewritten) < score(expr)


# ---------------------------------------------------------------------------
# Domain gating — the patent-12 correctness fix
# ---------------------------------------------------------------------------


def test_log_pow_rejected_for_unconstrained_symbol() -> None:
    """log(u^2) -> 2*log(u) is silently wrong for u<0; engine must reject."""
    expr = sp.log(u ** 2)
    sugg = suggest(expr, only_improvements=False)
    assert not any(s.pattern_name == "log_pow" for s in sugg)


def test_log_pow_accepted_for_positive_symbol() -> None:
    """log(x^2) -> 2*log(x) is correct when x is provably positive."""
    expr = sp.log(x ** 2)
    sugg = suggest(expr, only_improvements=False)
    assert any(s.pattern_name == "log_pow" for s in sugg)


def test_log_pow_conditional_when_unconstrained() -> None:
    """include_conditional=True surfaces the rewrite with the requirement annotated."""
    expr = sp.log(u ** 2)
    sugg = suggest(expr, only_improvements=False, include_conditional=True)
    log_pow = next((s for s in sugg if s.pattern_name == "log_pow"), None)
    assert log_pow is not None
    assert log_pow.domain_verified is False
    assert "u" in log_pow.domain_required and ">" in log_pow.domain_required


def test_log_split_rejected_for_unconstrained_symbol() -> None:
    """log(u/v) -> log(u) - log(v) requires u,v > 0."""
    expr = sp.log(u / v)
    sugg = suggest(expr, only_improvements=False)
    assert not any(s.pattern_name == "log_split" for s in sugg)


def test_log_split_accepted_for_positive_symbols() -> None:
    expr = sp.log(x / y)
    sugg_all = suggest(expr, only_improvements=False)
    log_splits = [s for s in sugg_all if s.pattern_name == "log_split"]
    assert log_splits, "log_split should be proposed for log(x/y) with x,y positive"
    for s in log_splits:
        assert s.domain_verified is True


def test_exp_log_inverse_rejected_for_unconstrained_symbol() -> None:
    """exp(log(u)) -> u requires u > 0.

    SymPy auto-simplifies ``exp(log(positive_x))`` immediately, so we
    use ``evaluate=False`` to force the unevaluated form the rewriter
    would actually see if a user constructed it explicitly.
    """
    expr = sp.exp(sp.log(u), evaluate=False)
    sugg = suggest(expr, only_improvements=False)
    assert not any(s.pattern_name == "exp_log_inverse" for s in sugg)


def test_log_exp_inverse_requires_real() -> None:
    """log(exp(u)) -> u requires u real (branch correctness)."""
    expr = sp.log(sp.exp(u), evaluate=False)
    sugg = suggest(expr, only_improvements=False)
    assert not any(s.pattern_name == "exp_log_inverse" for s in sugg)


# ---------------------------------------------------------------------------
# DomainRequirement primitives
# ---------------------------------------------------------------------------


def test_no_requirement_is_unconditional() -> None:
    assert NO_REQUIREMENT.is_unconditional() is True
    assert NO_REQUIREMENT.is_satisfied() is True


def test_requirement_unsatisfied_for_unconstrained_symbol() -> None:
    req = DomainRequirement(description="u > 0", positive=(u,))
    assert req.is_unconditional() is False
    assert req.is_satisfied() is False


def test_requirement_satisfied_for_positive_symbol() -> None:
    req = DomainRequirement(description="x > 0", positive=(x,))
    assert req.is_satisfied() is True


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------


def test_verify_equivalence_accepts_true_identity() -> None:
    """sin^2 + cos^2 == 1 holds everywhere; verifier must accept."""
    assert verify_equivalence(sp.sin(u) ** 2 + sp.cos(u) ** 2, sp.S.One, seed=1) is True


def test_verify_equivalence_rejects_log_pow_on_negative_region() -> None:
    """log(u^2) vs 2*log(u) disagree on u<0 — verifier must catch."""
    before = sp.log(u ** 2)
    after = 2 * sp.log(u)
    assert verify_equivalence(before, after, seed=1) is False


def test_verify_equivalence_rejects_arbitrary_inequality() -> None:
    assert verify_equivalence(sp.sin(u), sp.cos(u), seed=1) is False


def test_verify_equivalence_handles_no_free_symbols() -> None:
    assert verify_equivalence(sp.S(2) + sp.S(2), sp.S(4), seed=1) is True
    assert verify_equivalence(sp.S(2) + sp.S(2), sp.S(5), seed=1) is False
