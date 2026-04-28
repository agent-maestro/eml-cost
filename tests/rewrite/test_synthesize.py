"""Tests for render_test (counterexample-driven test synthesizer).

The most important tests here are the **meta-tests**: we render a
test from a real counterexample, exec the source, and confirm the
generated function actually executes and asserts as expected.
"""
from __future__ import annotations

import ast

import pytest
import sympy as sp

from eml_cost.rewrite import (
    Counterexample,
    find_counterexample,
    render_test,
)


u = sp.Symbol("u")


def _exec_to_namespace(source: str) -> dict[str, object]:
    """Compile + execute generated source, return its top-level namespace."""
    ns: dict[str, object] = {}
    exec(compile(source, "<rendered>", "exec"), ns)
    return ns


def test_render_test_returns_string():
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    src = render_test(cx)
    assert isinstance(src, str)
    assert "def test_" in src


def test_rendered_source_parses_as_python():
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_sin_neq_cos")
    # No SyntaxError → parses cleanly.
    ast.parse(src)


def test_rendered_test_uses_requested_name():
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_my_specific_regression")
    ns = _exec_to_namespace(src)
    assert "test_my_specific_regression" in ns
    assert callable(ns["test_my_specific_regression"])


def test_rendered_test_actually_passes_when_run():
    """Meta-test: the generated regression test runs and asserts True
    (the counterexample really IS a counterexample at the recorded point)."""
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_meta_check")
    ns = _exec_to_namespace(src)
    fn = ns["test_meta_check"]
    fn()        # type: ignore[operator]   — should NOT raise


def test_rendered_test_for_constant_disagreement():
    """No-free-symbol case: 2+2 vs 5."""
    cx = find_counterexample(sp.S(2) + sp.S(2), sp.S(5), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_constant_disagree")
    ns = _exec_to_namespace(src)
    ns["test_constant_disagree"]()      # type: ignore[operator]


def test_rendered_test_for_domain_mismatch():
    """log(u^2) vs 2*log(u): domain mismatch on negative u."""
    cx = find_counterexample(sp.log(u ** 2), 2 * sp.log(u), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_log_pow_negative")
    ns = _exec_to_namespace(src)
    ns["test_log_pow_negative"]()       # type: ignore[operator]


def test_render_requires_before_after_expressions():
    """Manually-constructed Counterexample without before_expr/after_expr
    must raise ValueError — render_test depends on those."""
    cx = Counterexample(
        symbols=(u,), point=(1.0,),
        before_value=complex(0.84), after_value=complex(0.54),
        kind="value_disagreement", note="manual",
    )   # before_expr / after_expr default to None
    with pytest.raises(ValueError, match="before_expr"):
        render_test(cx)


def test_rendered_source_includes_kind_in_comment():
    cx = find_counterexample(sp.sin(u), sp.cos(u), seed=1)
    assert cx is not None
    src = render_test(cx)
    assert "kind:" in src
    assert cx.kind in src


def test_assumption_bearing_symbol_substitutes_correctly():
    """Symbols with assumptions (positive=True) must be matched by name
    in the generated test, so subs() picks them up correctly."""
    xp = sp.Symbol("xp", positive=True)
    cx = find_counterexample(sp.sin(xp), sp.cos(xp), seed=1)
    assert cx is not None
    src = render_test(cx, name="test_assumption_symbol")
    ns = _exec_to_namespace(src)
    ns["test_assumption_symbol"]()      # type: ignore[operator]
