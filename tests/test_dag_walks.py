"""The detectors walk an expression as the DAG it is (0.26.0).

A reused subexpression is one node under several parents. Each detector used to recurse once per PATH to a
node, so an expression whose TREE is exponential -- a chain of reuses, as every recursive filter or controller
written with intermediate values builds -- took exponential time. Every value here is computed once per
distinct node (``eml_cost._dag``), and the values are unchanged: on 1,289 expressions (every input Monogate
Forge's profiler hands ``analyze``/``analyze_dynamics`` over the eml-stdlib corpus and Forge's examples, plus
``bench/adversarial.csv``), all eight detectors returned identical results before and after.
"""

from __future__ import annotations

import sympy as sp

from eml_cost import analyze, analyze_dynamics, predict_chain_order_via_additivity
from eml_cost._dag import fold, nodes
from eml_cost.core import eml_depth, is_pfaffian_not_eml, max_path_r, pfaffian_r, structural_overhead

x = sp.Symbol("x")


def _chain(n: int) -> sp.Expr:
    """``t_k = sin(t_k-1) + cos(t_k-1) * t_k-1``: three references per level, a tree of 3^n leaves."""
    t = x
    for _ in range(n):
        t = sp.sin(t) + sp.cos(t) * t
    return t


def test_the_chain_is_a_dag_of_linear_size():
    assert len(list(nodes(_chain(64)))) < 6 * 64


def test_reversal_its_tree_is_exponential():
    assert sum(1 for _ in sp.preorder_traversal(_chain(9))) > 3 ** 9


def test_every_detector_is_linear_in_the_chain():
    """Each level adds the same to each figure, so the values at 64 levels are those at 1 and 2 extended --
    and computing them at all is the test: per path, that is 3^64 visits."""
    def figures(n: int) -> tuple:
        e = _chain(n)
        r = analyze(e)
        d = analyze_dynamics(e)
        return (r.pfaffian_r, r.max_path_r, r.eml_depth, r.structural_overhead, r.predicted_depth,
                d.n_oscillations, d.n_decays, predict_chain_order_via_additivity(e))

    one, two = figures(1), figures(2)
    step = tuple(b - a for a, b in zip(one, two))
    assert figures(64) == tuple(a + 63 * s for a, s in zip(one, step))
    assert figures(5) == tuple(a + 4 * s for a, s in zip(one, step))


def test_depth_is_not_bounded_by_the_recursion_limit():
    """The recursive detectors spent two frames a level (the call and its generator); 3,000 nested exps
    raised RecursionError. ``fold`` runs on an explicit stack."""
    e = x
    for _ in range(3000):
        e = sp.exp(e)
    assert max_path_r(e) == 3000
    assert eml_depth(e) == 3000
    assert pfaffian_r(e) == 3000
    assert structural_overhead(e) == 0
    assert analyze(e).corrections.c_osc == 0


def test_a_shared_argument_is_one_mode():
    s = x + 1
    d = analyze_dynamics(sp.Function("F")(sp.cos(s), sp.sin(s), sp.exp(s), sp.exp(s) * x))
    assert (d.n_oscillations, d.n_decays) == (1, 1)


def test_a_pne_primitive_is_found_under_a_shared_node():
    s = sp.besselj(0, x) + x
    assert is_pfaffian_not_eml(sp.Function("F")(s, s))


def test_fold_evaluates_each_object_once_and_children_first():
    s = x + 1
    expr = sp.Function("F")(s, s)
    order: list = []

    def rule(e):
        def combine(values):
            order.append(e)
            return 1 + sum(values)
        return e.args, combine

    assert fold(expr, rule) == 1 + 2 * (1 + 2)       # F(s, s): s counts twice, computed once
    assert order.count(s) == 1
    assert order.index(s) < order.index(expr)


def test_fold_keeps_a_dependency_the_rule_builds():
    """The sigmoid pattern recurses into ``-g``, an expression the rule constructs: it must stay alive
    while its id keys the values."""
    g = x + 2
    sig = 1 / (1 + sp.exp(-g))
    assert eml_depth(sig) == 1 + eml_depth(g)
