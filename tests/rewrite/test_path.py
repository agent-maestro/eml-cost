"""Tests for ``eml_rewrite.path`` (added in 0.2.0)."""
from __future__ import annotations

import sympy as sp

from eml_cost.rewrite import Step, path, score


x = sp.Symbol("x", real=True, positive=True)


# ---------------------------------------------------------------------------
# Trivial paths
# ---------------------------------------------------------------------------


def test_path_zero_step_when_start_equals_end() -> None:
    """If start == end (Python equality), a single seed step is returned."""
    expr = sp.exp(x) + 1
    p = path(expr, expr)
    assert p is not None
    assert len(p) == 1
    assert p[0].pattern_name == "<start>"
    assert p[0].expression == expr


def test_path_returns_none_when_not_equivalent() -> None:
    p = path(sp.sin(x), sp.cos(x))
    assert p is None


# ---------------------------------------------------------------------------
# Single-step paths in the standard rewrite library
# ---------------------------------------------------------------------------


def test_sigmoid_textbook_to_canonical_one_step() -> None:
    start = sp.exp(x) / (1 + sp.exp(x))
    end = 1 / (1 + sp.exp(-x))
    p = path(start, end)
    assert p is not None
    # Steps: <start>, sigmoid (-> end). At least 2 entries.
    assert len(p) >= 2
    assert p[-1].expression == end or sp.simplify(p[-1].expression - end) == 0


def test_pythagorean_collapses_in_one_step() -> None:
    start = sp.sin(x) ** 2 + sp.cos(x) ** 2
    end = sp.S.One
    p = path(start, end)
    assert p is not None
    assert any(s.pattern_name == "pythagorean" for s in p[1:])
    assert p[-1].expression == sp.S.One


def test_tanh_from_sinh_cosh_one_step() -> None:
    start = sp.sinh(x) / sp.cosh(x)
    end = sp.tanh(x)
    p = path(start, end)
    assert p is not None
    assert any(s.pattern_name == "tanh_from_sinh_cosh" for s in p[1:])


def test_hyperbolic_id_collapses() -> None:
    start = sp.cosh(x) ** 2 - sp.sinh(x) ** 2
    end = sp.S.One
    p = path(start, end)
    assert p is not None
    assert p[-1].expression == sp.S.One


# ---------------------------------------------------------------------------
# Cost monotonicity
# ---------------------------------------------------------------------------


def test_path_costs_are_monotone_non_increasing() -> None:
    """Every path returned must have non-increasing cost across
    consecutive steps. This is the *defining* property of the
    cost-anneal walker."""
    pairs = [
        (sp.exp(x) / (1 + sp.exp(x)), 1 / (1 + sp.exp(-x))),
        (sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.S.One),
        (sp.cosh(x) ** 2 - sp.sinh(x) ** 2, sp.S.One),
        (sp.sinh(x) / sp.cosh(x), sp.tanh(x)),
    ]
    for start, end in pairs:
        p = path(start, end)
        assert p is not None, f"no path for {start} -> {end}"
        costs = [s.cost for s in p]
        assert all(b <= a for a, b in zip(costs, costs[1:])), \
            f"non-monotone path for {start} -> {end}: {costs}"


def test_terminal_cost_no_higher_than_start_cost() -> None:
    start = sp.exp(x) / (1 + sp.exp(x))
    end = 1 / (1 + sp.exp(-x))
    p = path(start, end)
    assert p is not None
    assert p[-1].cost <= score(start)


# ---------------------------------------------------------------------------
# Step dataclass shape
# ---------------------------------------------------------------------------


def test_step_is_frozen() -> None:
    s = Step(pattern_name="x", expression=sp.S.One, cost=0)
    import pytest
    with pytest.raises(Exception):
        s.cost = 99  # type: ignore[misc]


def test_step_carries_pattern_name() -> None:
    p = path(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.S.One)
    assert p is not None
    assert p[0].pattern_name == "<start>"
    # At least one non-seed step should carry a real pattern name.
    assert any(s.pattern_name not in ("<start>", "<canonicalize>") for s in p[1:])


# ---------------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------------


def test_path_returns_none_when_budget_too_low() -> None:
    """A reachable target should still return None when max_steps=0
    and the start doesn't already equal the end."""
    start = sp.exp(x) / (1 + sp.exp(x))
    end = 1 / (1 + sp.exp(-x))
    p = path(start, end, max_steps=0)
    assert p is None
