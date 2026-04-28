"""Tests for eml_cost.regression.nodes — EMLNode tree representation."""
from __future__ import annotations

import math
import random

import pytest

np = pytest.importorskip("numpy")

import sympy as sp

from eml_cost.regression.nodes import (
    BINARY_OPS,
    UNARY_OPS,
    EMLNode,
    all_subtree_indices,
    get_subtree,
    random_tree,
    replace_subtree,
)


class TestConstruction:
    def test_var_node(self) -> None:
        n = EMLNode.var("x")
        assert n.kind == "var"
        assert n.name == "x"
        assert n.children == []

    def test_const_node(self) -> None:
        n = EMLNode.const(3.14)
        assert n.kind == "const"
        assert n.value == pytest.approx(3.14)

    def test_unary_validation(self) -> None:
        with pytest.raises(ValueError):
            EMLNode.unary("ack", EMLNode.var("x"))

    def test_binary_validation(self) -> None:
        with pytest.raises(ValueError):
            EMLNode.binary("**", EMLNode.var("x"), EMLNode.var("y"))


class TestEvaluation:
    def test_constant_broadcasts(self) -> None:
        env = {"x": np.linspace(0, 1, 5)}
        out = EMLNode.const(2.5).evaluate(env)
        assert out.shape == (5,)
        assert np.allclose(out, 2.5)

    def test_variable_eval(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        out = EMLNode.var("x").evaluate({"x": x})
        np.testing.assert_array_equal(out, x)

    def test_arithmetic(self) -> None:
        x = np.linspace(-1, 1, 11)
        # (x + 1) * 2
        tree = EMLNode.binary(
            "*",
            EMLNode.binary("+", EMLNode.var("x"), EMLNode.const(1.0)),
            EMLNode.const(2.0),
        )
        out = tree.evaluate({"x": x})
        np.testing.assert_allclose(out, (x + 1) * 2)

    def test_division_by_zero_returns_nan(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        tree = EMLNode.binary("/", EMLNode.const(1.0), EMLNode.var("x"))
        out = tree.evaluate({"x": x})
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(1.0)

    def test_log_of_negative_is_nan(self) -> None:
        x = np.array([-1.0, 1.0])
        tree = EMLNode.unary("log", EMLNode.var("x"))
        out = tree.evaluate({"x": x})
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(0.0, abs=1e-9)

    def test_exp_does_not_overflow(self) -> None:
        x = np.array([1000.0])
        tree = EMLNode.unary("exp", EMLNode.var("x"))
        out = tree.evaluate({"x": x})
        # Clipped → finite.
        assert np.isfinite(out).all()


class TestSympyBridge:
    def test_simple_round_trip(self) -> None:
        tree = EMLNode.binary("+",
            EMLNode.var("x"), EMLNode.const(1.0))
        expr = tree.to_sympy()
        # to_sympy emits Float for general constants; equality
        # against Integer 1 is not symbolic-equal in SymPy.
        assert expr == sp.Symbol("x", real=True) + sp.Float(1.0)

    def test_pi_is_recovered(self) -> None:
        tree = EMLNode.const(math.pi)
        assert tree.to_sympy() == sp.pi

    def test_e_is_recovered(self) -> None:
        tree = EMLNode.const(math.e)
        assert tree.to_sympy() == sp.E

    def test_unary_sin_to_sympy(self) -> None:
        tree = EMLNode.unary("sin", EMLNode.var("x"))
        assert tree.to_sympy() == sp.sin(sp.Symbol("x", real=True))

    def test_neg_to_sympy(self) -> None:
        tree = EMLNode.unary("neg", EMLNode.var("x"))
        assert tree.to_sympy() == -sp.Symbol("x", real=True)


class TestStructure:
    def test_size_counts_all_nodes(self) -> None:
        tree = EMLNode.binary(
            "+",
            EMLNode.unary("sin", EMLNode.var("x")),
            EMLNode.const(1.0),
        )
        assert tree.size() == 4

    def test_depth(self) -> None:
        tree = EMLNode.unary("sin",
            EMLNode.unary("cos", EMLNode.var("x")))
        assert tree.depth() == 3

    def test_copy_is_deep(self) -> None:
        tree = EMLNode.binary("+", EMLNode.var("x"), EMLNode.const(1.0))
        c = tree.copy()
        c.children[1].value = 99.0
        assert tree.children[1].value == 1.0


class TestSubtreeOps:
    def test_all_subtree_indices_includes_root(self) -> None:
        tree = EMLNode.var("x")
        paths = all_subtree_indices(tree)
        assert paths == [[]]

    def test_all_subtree_indices_count(self) -> None:
        tree = EMLNode.binary("+",
            EMLNode.var("x"), EMLNode.const(1.0))
        paths = all_subtree_indices(tree)
        assert len(paths) == 3
        assert [] in paths
        assert [0] in paths
        assert [1] in paths

    def test_get_subtree(self) -> None:
        tree = EMLNode.binary("+",
            EMLNode.var("x"), EMLNode.const(1.0))
        sub = get_subtree(tree, [1])
        assert sub.kind == "const"
        assert sub.value == pytest.approx(1.0)

    def test_replace_subtree_returns_new_tree(self) -> None:
        tree = EMLNode.binary("+",
            EMLNode.var("x"), EMLNode.const(1.0))
        new = replace_subtree(tree, [1], EMLNode.const(99.0))
        assert tree.children[1].value == pytest.approx(1.0)  # unchanged
        assert new.children[1].value == pytest.approx(99.0)

    def test_replace_subtree_root(self) -> None:
        tree = EMLNode.var("x")
        new = replace_subtree(tree, [], EMLNode.const(1.0))
        assert new.kind == "const"


class TestRandomTree:
    def test_random_tree_respects_max_depth(self) -> None:
        rng = random.Random(0)
        for _ in range(20):
            t = random_tree(rng, ["x"], max_depth=3)
            assert t.depth() <= 3

    def test_random_tree_uses_provided_vars(self) -> None:
        rng = random.Random(1)
        t = random_tree(rng, ["alpha", "beta"], max_depth=4)
        # At least one variable in the tree should belong to our set.
        names = []
        stack = [t]
        while stack:
            n = stack.pop()
            if n.kind == "var":
                names.append(n.name)
            stack.extend(n.children)
        if names:
            for nm in names:
                assert nm in {"alpha", "beta"}

    def test_random_tree_reproducible(self) -> None:
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        t1 = random_tree(rng1, ["x"], max_depth=4)
        t2 = random_tree(rng2, ["x"], max_depth=4)
        assert t1._readable() == t2._readable()
