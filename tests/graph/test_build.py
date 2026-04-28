"""Tests for eml_graph.build_graph and the EquivalenceGraph data shape."""
from __future__ import annotations

import sympy as sp

from eml_cost.graph import (
    EquivalenceClass,
    EquivalenceGraph,
    GraphNode,
    build_graph,
)


x = sp.Symbol("x")
y = sp.Symbol("y")
z = sp.Symbol("z")


def test_build_graph_on_empty_corpus():
    g = build_graph([])
    assert isinstance(g, EquivalenceGraph)
    assert g.num_nodes() == 0
    assert g.num_classes() == 0


def test_build_graph_on_single_expression():
    g = build_graph([sp.sin(x)])
    assert g.num_nodes() == 1
    assert g.num_classes() == 1
    only = g.nodes[0]
    assert isinstance(only, GraphNode)
    assert only.expression == sp.sin(x)
    assert only.fingerprint.startswith("p")
    assert only.axes.startswith("p")
    assert "-h" not in only.axes
    assert only.label is None


def test_dedup_by_python_equality():
    g = build_graph([sp.sin(x), sp.sin(x), sp.sin(x)])
    assert g.num_nodes() == 1
    assert g.num_classes() == 1


def test_axes_clusters_renamed_expressions():
    """sin(x), sin(y), cos(z) all share axes (same r=2 with c_osc=1)
    — they end up in one cluster despite different symbols and shapes."""
    g = build_graph([sp.sin(x), sp.sin(y), sp.cos(z)])
    assert g.num_nodes() == 3
    assert g.num_classes() == 1
    cls = list(g.classes.values())[0]
    assert isinstance(cls, EquivalenceClass)
    assert len(cls.members) == 3


def test_distinct_axes_separate_clusters():
    """sin (r=2) and exp (r=1) live in different cost classes."""
    g = build_graph([sp.sin(x), sp.exp(x)])
    assert g.num_nodes() == 2
    assert g.num_classes() == 2


def test_string_inputs_are_sympified():
    g = build_graph(["sin(x)", "exp(x)"])
    assert g.num_nodes() == 2
    assert g.num_classes() == 2


def test_unparseable_strings_are_silently_skipped():
    """The corpus loader must be forgiving — junk in, no crash."""
    g = build_graph(["sin(x)", "this is not valid sympy",
                     "exp(", "exp(x)"])
    # Only the two valid expressions parse.
    assert g.num_nodes() == 2


def test_class_sizes_descending():
    """class_sizes returns member counts ordered largest-first."""
    # 3 trig (one class), 1 exp (another class).
    g = build_graph([sp.sin(x), sp.cos(y), sp.sin(z), sp.exp(x)])
    sizes = g.class_sizes()
    assert sizes == [3, 1]


def test_class_of_returns_owning_class():
    g = build_graph([sp.sin(x), sp.exp(x), sp.sin(y)])
    cls = g.class_of(0)
    assert sp.sin(x) in [g.nodes[i].expression for i in cls.members]
    assert sp.sin(y) in [g.nodes[i].expression for i in cls.members]


def test_members_of_returns_correct_nodes():
    g = build_graph([sp.sin(x), sp.exp(x), sp.sin(y)])
    sin_axes = g.nodes[0].axes
    members = g.members_of(sin_axes)
    assert len(members) == 2
    assert all(isinstance(m, GraphNode) for m in members)


def test_members_of_unknown_axes_returns_empty_list():
    g = build_graph([sp.sin(x)])
    assert g.members_of("p99-d99-w99-c99") == []


def test_find_path_between_equivalent_expressions():
    """Two equivalent sigmoid forms (textbook vs canonical) sit in
    DIFFERENT cost classes — the canonical form has lower depth via
    F-family fusion. ``find_path`` walks between them via the
    monotone-decreasing rewrite library."""
    textbook = sp.exp(x) / (1 + sp.exp(x))
    canonical = 1 / (1 + sp.exp(-x))
    g = build_graph([textbook, canonical])
    # Same Pfaffian-r but different correction sums → different clusters.
    assert g.num_classes() == 2
    # Symbolically equivalent → ``find_path`` succeeds across clusters.
    p = g.find_path(0, 1)
    assert p is not None
    assert isinstance(p, list)
    assert len(p) >= 1


def test_label_with_discover_when_available():
    """When eml_discover is installed, exact registry hits get labels."""
    try:
        import eml_discover    # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("eml_discover not installed in this environment")

    canonical_sigmoid = 1 / (1 + sp.exp(-x))
    g = build_graph([canonical_sigmoid], label_with_discover=True)
    assert g.num_nodes() == 1
    label = g.nodes[0].label
    # Either a sigmoid-named match, or None if registry missed.
    if label is not None:
        assert "sigmoid" in label.lower() or "logistic" in label.lower()


def test_label_with_discover_fallback_when_missing():
    """If eml_discover isn't importable, label_with_discover=True
    must NOT crash — labels just stay None."""
    g = build_graph([sp.sin(x)], label_with_discover=True)
    assert g.num_nodes() == 1
    # Either we got a label (eml_discover present) or we didn't —
    # what matters is no exception.
