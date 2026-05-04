"""Tests for eml_graph.render.to_dot."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost.graph import build_graph, to_dot


x = sp.Symbol("x")
y = sp.Symbol("y")


def test_to_dot_returns_string_starting_with_digraph():
    g = build_graph([sp.sin(x), sp.exp(x)])
    dot = to_dot(g)
    assert isinstance(dot, str)
    assert dot.startswith("digraph EquivalenceGraph {")
    assert dot.rstrip().endswith("}")


def test_to_dot_emits_one_subgraph_per_class():
    g = build_graph([sp.sin(x), sp.exp(x), sp.cos(y)])
    dot = to_dot(g)
    cluster_count = dot.count("subgraph cluster_")
    assert cluster_count == g.num_classes()


def test_to_dot_emits_one_node_line_per_node():
    g = build_graph([sp.sin(x), sp.exp(x), sp.cos(y)])
    dot = to_dot(g)
    # Each node gets a "n<idx> [label=..." line.
    node_lines = [ln for ln in dot.split("\n") if ln.strip().startswith("n")]
    # At minimum, one declaration per node.
    declared = sum(
        1 for ln in node_lines
        if "[label=" in ln and not "->" in ln
    )
    assert declared == g.num_nodes()


@pytest.mark.skip(reason="requires eml_rewrite — not yet published")
def test_to_dot_with_edges_includes_edge_lines_for_equivalent_pair():
    """Sigmoid pair: textbook ↔ canonical are equivalent and in the
    same class — to_dot(include_edges=True) must emit at least one
    edge line for them."""
    textbook = sp.exp(x) / (1 + sp.exp(x))
    canonical = 1 / (1 + sp.exp(-x))
    g = build_graph([textbook, canonical])
    dot = to_dot(g, include_edges=True)
    assert " -> " in dot


def test_to_dot_without_edges_has_no_edge_lines():
    g = build_graph([sp.sin(x), sp.cos(y)])
    dot = to_dot(g, include_edges=False)
    assert " -> " not in dot


def test_to_dot_escapes_quotes_in_labels():
    """Expressions whose str representation contains quotes (rare but
    possible via custom Function names) must not break DOT syntax."""
    # Build a corpus where the rendered label contains a quote.
    f = sp.Function('f"name')(x)
    g = build_graph([f])
    dot = to_dot(g)
    # The escaped quote (\\\") must appear inside a label="..." pair.
    assert '\\"' in dot
