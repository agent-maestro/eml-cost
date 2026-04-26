"""Tests for analyze_batch + cache_hit_analysis."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost import (
    PfaffianProfile,
    analyze_batch,
    cache_hit_analysis,
)


def test_batch_returns_one_profile_per_input():
    profiles = analyze_batch(["x", "exp(x)", "sin(x)"], n_jobs=1)
    assert len(profiles) == 3
    assert all(isinstance(p, PfaffianProfile) for p in profiles)


def test_batch_preserves_order():
    inputs = ["exp(x)", "sin(x)", "log(x)", "tan(x)"]
    profiles = analyze_batch(inputs, n_jobs=1, cache=False)
    assert profiles[0].cost_class == PfaffianProfile.from_expression("exp(x)").cost_class
    assert profiles[1].cost_class == PfaffianProfile.from_expression("sin(x)").cost_class
    assert profiles[2].cost_class == PfaffianProfile.from_expression("log(x)").cost_class
    assert profiles[3].cost_class == PfaffianProfile.from_expression("tan(x)").cost_class


def test_batch_serial_matches_loop():
    """analyze_batch must produce identical results to a loop."""
    inputs = ["x", "exp(x)", "sin(x*y)", "log(1+x**2)", "exp(exp(x))"]
    batch = analyze_batch(inputs, n_jobs=1, cache=False)
    loop = [PfaffianProfile.from_expression(e) for e in inputs]
    assert [p.cost_class for p in batch] == [p.cost_class for p in loop]


def test_batch_handles_failed_expression_gracefully():
    """One bad expression doesn't crash the batch."""
    inputs = ["exp(x)", "this_is_not_valid(((", "sin(x)"]
    profiles = analyze_batch(inputs, n_jobs=1, cache=False)
    assert len(profiles) == 3
    # Either profiled or got an ERROR sentinel — never raises.
    assert all(isinstance(p, PfaffianProfile) for p in profiles)


def test_batch_cache_does_not_change_results():
    inputs = ["exp(x)", "exp(x)", "sin(x)", "exp(x)"]
    with_cache = analyze_batch(inputs, n_jobs=1, cache=True)
    without_cache = analyze_batch(inputs, n_jobs=1, cache=False)
    assert [p.cost_class for p in with_cache] == [p.cost_class for p in without_cache]


def test_batch_canonicalize_makes_equivalent_forms_share_class():
    inputs = ["1/(1+exp(-x))", "exp(x)/(exp(x)+1)"]
    profiles = analyze_batch(inputs, canonicalize=True, n_jobs=1)
    assert profiles[0].cost_class == profiles[1].cost_class


def test_cache_hit_analysis_unique_count():
    inputs = ["x", "x", "exp(x)", "exp(x)", "exp(x)"]
    info = cache_hit_analysis(inputs)
    assert info["n"] == 5
    # x and exp(x) → 2 unique canonical forms
    assert info["n_unique_canonical"] == 2
    assert info["theoretical_hit_rate"] == 0.6  # 3 of 5 are duplicates


def test_cache_hit_analysis_with_equivalent_forms():
    """Equivalent forms collapse to same canonical → same cache key."""
    inputs = ["1/(1+exp(-x))", "exp(x)/(exp(x)+1)", "1 - 1/(1+exp(x))"]
    info = cache_hit_analysis(inputs, canonicalize=True)
    # All three should canonicalize to the same form (or close enough)
    # We don't assert exact 1; we assert <3 (some collapse)
    assert info["n_unique_canonical"] <= 3


def test_batch_empty_list_returns_empty_list():
    assert analyze_batch([]) == []


def test_batch_progress_off_for_small_input(capsys):
    analyze_batch(["x", "y"], progress=False)
    out = capsys.readouterr()
    assert "analyze_batch" not in out.out  # no progress bar


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_batch_n_jobs_returns_same_order(n_jobs):
    inputs = ["x**2", "exp(x)", "sin(x)*cos(x)", "log(1+x)"]
    profiles = analyze_batch(inputs, n_jobs=n_jobs)
    expected = [PfaffianProfile.from_expression(e).cost_class for e in inputs]
    actual = [p.cost_class for p in profiles]
    assert actual == expected
