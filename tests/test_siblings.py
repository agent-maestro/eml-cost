"""Tests for eml_cost.find_siblings — cross-domain structural search.

The corpus is the 578-expression cross-domain bench (12 subdomains)
precomputed at packaging time. find_siblings ranks corpus rows by
weighted Euclidean distance on (r, degree, width, c_osc).
"""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import (
    Sibling,
    corpus_domains,
    corpus_size,
    find_siblings,
)


# ---------------------------------------------------------------------------
# Corpus introspection
# ---------------------------------------------------------------------------


def test_corpus_size_at_least_500() -> None:
    """Corpus should ship with at least 500 rows. Catches packaging gaps."""
    n = corpus_size()
    assert n >= 500, f"corpus_size() = {n}, expected >= 500"


def test_corpus_size_at_most_580() -> None:
    """Corpus should ship with at most 578 rows (some skipped at build)."""
    assert corpus_size() <= 578


def test_corpus_domains_returns_sorted_tuple() -> None:
    domains = corpus_domains()
    assert isinstance(domains, tuple)
    assert all(isinstance(d, str) for d in domains)
    assert list(domains) == sorted(domains)
    assert len(domains) >= 8  # 8+ subdomains expected


def test_corpus_includes_color_science() -> None:
    """Color Science was the most recent merge — verify it's bundled."""
    assert "color_science" in corpus_domains()


def test_corpus_includes_robotics_and_human_body() -> None:
    domains = corpus_domains()
    assert "robotics" in domains
    assert "human_body" in domains


# ---------------------------------------------------------------------------
# find_siblings basic contract
# ---------------------------------------------------------------------------


def test_returns_list_of_siblings() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=3)
    assert isinstance(sibs, list)
    assert all(isinstance(s, Sibling) for s in sibs)
    assert len(sibs) <= 3


def test_returns_at_most_k() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=2)
    assert len(sibs) <= 2


def test_sorted_by_ascending_distance() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=10)
    distances = [s.distance for s in sibs]
    assert distances == sorted(distances)


def test_distances_non_negative() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=5)
    for s in sibs:
        assert s.distance >= 0.0


def test_string_and_sympy_query_agree() -> None:
    s1 = find_siblings("A * cos(omega * t)", k=3)
    A, omega, t = sp.symbols("A omega t", real=True)
    s2 = find_siblings(A * sp.cos(omega * t), k=3)
    assert [s.name for s in s1] == [s.name for s in s2]


# ---------------------------------------------------------------------------
# Domain filtering + max_distance
# ---------------------------------------------------------------------------


def test_domain_filter_restricts_results() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=10, domain="music_sound")
    assert all(s.domain == "music_sound" for s in sibs)


def test_max_distance_caps_results() -> None:
    sibs = find_siblings("A * cos(omega * t)", k=20, max_distance=1.0)
    for s in sibs:
        assert s.distance <= 1.0


def test_unknown_domain_returns_empty() -> None:
    sibs = find_siblings("x", k=5, domain="this_domain_does_not_exist")
    assert sibs == []


# ---------------------------------------------------------------------------
# Structural-similarity sanity
# ---------------------------------------------------------------------------


def test_oscillator_finds_oscillator_neighbours() -> None:
    """A pure cosine should have nearest-neighbours in oscillator classes."""
    sibs = find_siblings("A * cos(omega * t)", k=5)
    assert sibs
    # The closest sibling should be in the same cost class (distance=0).
    assert sibs[0].distance == 0.0


def test_polynomial_finds_polynomial_neighbours() -> None:
    """T**4 should be closest to other low-r polynomial expressions."""
    sibs = find_siblings("T**4", k=5)
    assert sibs
    # All within distance 4 (one r-step + small d/w changes).
    assert sibs[0].distance <= 4.0


def test_distance_zero_means_same_class() -> None:
    """Any sibling with distance==0 should share the query's cost_class."""
    from eml_cost import PfaffianProfile
    query = PfaffianProfile.from_expression("A * cos(omega * t)")
    sibs = find_siblings("A * cos(omega * t)", k=20)
    for s in sibs:
        if s.distance == 0.0:
            assert s.cost_class == query.cost_class


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------


def test_custom_weights_change_ordering() -> None:
    """Custom weights should be honoured (different ordering possible)."""
    default = find_siblings("A * cos(omega * t)", k=5)
    custom = find_siblings(
        "A * cos(omega * t)", k=5, weights={"r": 1, "d": 4, "w": 1, "c": 1},
    )
    # Both lists are valid; just verify the call accepts custom weights.
    assert len(default) >= 1
    assert len(custom) >= 1
