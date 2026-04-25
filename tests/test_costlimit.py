"""Tests for ``@costlimit`` decorator (added in 0.1.2)."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost import CostLimitExceeded, analyze, costlimit


x = sp.Symbol("x", real=True, positive=True)


def test_costlimit_passes_when_return_within_budget() -> None:
    @costlimit(predicted_depth=5)
    def f(x):
        return sp.exp(x) * sp.sin(x)
    result = f(x)
    assert result == sp.exp(x) * sp.sin(x)
    # Sanity: confirm the budget is actually higher than the cost.
    assert analyze(result).predicted_depth <= 5


def test_costlimit_raises_when_predicted_depth_exceeded() -> None:
    @costlimit(predicted_depth=2)
    def f(x):
        return sp.exp(sp.exp(sp.exp(sp.exp(x))))   # depth ≥ 4
    with pytest.raises(CostLimitExceeded) as excinfo:
        f(x)
    assert excinfo.value.axis == "predicted_depth"
    assert excinfo.value.measured > excinfo.value.limit
    assert excinfo.value.limit == 2


def test_costlimit_raises_on_max_path_r_axis() -> None:
    @costlimit(max_path_r=1)
    def f(x):
        return sp.exp(sp.exp(x))   # max_path_r = 2
    with pytest.raises(CostLimitExceeded) as excinfo:
        f(x)
    assert excinfo.value.axis == "max_path_r"


def test_costlimit_raises_on_pfaffian_r_axis() -> None:
    @costlimit(pfaffian_r=2)
    def f(x):
        return sp.exp(sp.exp(sp.exp(x)))   # pfaffian_r = 3
    with pytest.raises(CostLimitExceeded) as excinfo:
        f(x)
    assert excinfo.value.axis == "pfaffian_r"


def test_costlimit_axis_picked_in_declaration_order() -> None:
    """When multiple axes are configured AND multiple are exceeded,
    the decorator reports the FIRST one breached: predicted_depth
    has highest priority, then max_path_r, then pfaffian_r."""
    @costlimit(predicted_depth=1, max_path_r=1, pfaffian_r=1)
    def f(x):
        return sp.exp(sp.sin(x))   # all axes >= 2
    with pytest.raises(CostLimitExceeded) as excinfo:
        f(x)
    assert excinfo.value.axis == "predicted_depth"


def test_costlimit_attaches_offending_expression() -> None:
    @costlimit(predicted_depth=1)
    def f(x):
        return sp.exp(sp.exp(x))
    with pytest.raises(CostLimitExceeded) as excinfo:
        f(x)
    assert excinfo.value.expression == sp.exp(sp.exp(x))


def test_costlimit_passes_through_non_sympy_returns() -> None:
    """If the wrapped function returns a non-Basic value (None, int,
    list, ...), the decorator must NOT crash on the analyze() call."""
    @costlimit(predicted_depth=0)
    def f():
        return 42
    assert f() == 42

    @costlimit(predicted_depth=0)
    def g():
        return None
    assert g() is None


def test_costlimit_requires_at_least_one_axis() -> None:
    with pytest.raises(ValueError, match="at least one"):
        @costlimit()
        def f():
            return sp.S.One


def test_costlimit_preserves_function_name() -> None:
    @costlimit(predicted_depth=10)
    def my_special_function(x):
        return x
    assert my_special_function.__name__ == "my_special_function"


def test_costlimit_preserves_docstring() -> None:
    @costlimit(predicted_depth=10)
    def f(x):
        """Documented function."""
        return x
    assert f.__doc__ == "Documented function."
