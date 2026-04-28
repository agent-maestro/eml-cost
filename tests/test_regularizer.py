"""Tests for eml_cost.regularizer — chain-order regularizer for
EML-native symbolic regression. Ships in 0.16.0.
"""
from __future__ import annotations

import pytest

from eml_cost import (
    RegularizerConfig,
    RegularizerResult,
    regularize,
)


# ---------------------------------------------------------------------------
# Default config (zero penalty)
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_config_is_zero_penalty(self) -> None:
        r = regularize("sin(x)")
        assert r.total_penalty == 0.0
        assert r.chain_penalty == 0.0
        assert r.node_penalty == 0.0
        assert r.dynamics_penalty == 0.0
        assert r.stability_penalty == 0.0

    def test_default_config_still_reports_structure(self) -> None:
        r = regularize("sin(x)")
        assert r.chain_order >= 1
        assert r.node_count >= 1
        assert r.is_feasible is True

    def test_default_max_chain_order_is_5(self) -> None:
        cfg = RegularizerConfig()
        assert cfg.max_chain_order == 5


# ---------------------------------------------------------------------------
# Chain-order penalty
# ---------------------------------------------------------------------------


class TestChainPenalty:
    def test_polynomial_pays_no_chain_penalty(self) -> None:
        cfg = RegularizerConfig(lambda_chain=1.0, max_chain_order=2)
        r = regularize("x**2 + 2*x + 1", cfg)
        assert r.chain_order == 0
        assert r.chain_penalty == 0.0

    def test_deeply_nested_pays_chain_penalty(self) -> None:
        cfg = RegularizerConfig(lambda_chain=1.0, max_chain_order=2)
        # sin(exp(cos(x))) — chain order >=4 (cos→2, exp→+1, sin→+2)
        r = regularize("sin(exp(cos(x)))", cfg)
        assert r.chain_order > 2
        assert r.chain_penalty > 0
        assert not r.is_feasible

    def test_lambda_chain_scales_linearly(self) -> None:
        cfg1 = RegularizerConfig(lambda_chain=1.0, max_chain_order=0)
        cfg2 = RegularizerConfig(lambda_chain=3.0, max_chain_order=0)
        r1 = regularize("sin(x)", cfg1)
        r2 = regularize("sin(x)", cfg2)
        assert r2.chain_penalty == pytest.approx(3.0 * r1.chain_penalty)

    def test_feasibility_boundary(self) -> None:
        cfg = RegularizerConfig(lambda_chain=1.0, max_chain_order=2)
        r = regularize("sin(x)", cfg)  # chain order 2
        assert r.is_feasible is True
        assert r.chain_penalty == 0.0


# ---------------------------------------------------------------------------
# Node-count penalty
# ---------------------------------------------------------------------------


class TestNodePenalty:
    def test_node_penalty_grows_with_size(self) -> None:
        cfg = RegularizerConfig(lambda_nodes=1.0)
        small = regularize("x", cfg)
        big = regularize("x**2 + 2*x*y + y**2 + sin(x)*cos(y)", cfg)
        assert big.node_penalty > small.node_penalty

    def test_zero_lambda_zero_penalty(self) -> None:
        cfg = RegularizerConfig(lambda_nodes=0.0)
        r = regularize("x**2 + sin(x)*cos(x)", cfg)
        assert r.node_penalty == 0.0


# ---------------------------------------------------------------------------
# Dynamics penalty
# ---------------------------------------------------------------------------


class TestDynamicsPenalty:
    def test_dynamics_penalty_zero_when_no_expectation(self) -> None:
        cfg = RegularizerConfig(lambda_dynamics=10.0)
        r = regularize("sin(x)", cfg)
        assert r.dynamics_penalty == 0.0

    def test_dynamics_penalty_zero_on_match(self) -> None:
        # sin(x) → 1 oscillation, 0 decays
        cfg = RegularizerConfig(
            lambda_dynamics=1.0,
            expected_dynamics=(1, 0),
        )
        r = regularize("sin(x)", cfg)
        assert r.predicted_dynamics == (1, 0)
        assert r.dynamics_penalty == 0.0

    def test_dynamics_penalty_positive_on_mismatch(self) -> None:
        cfg = RegularizerConfig(
            lambda_dynamics=1.0,
            expected_dynamics=(2, 0),
        )
        r = regularize("sin(x)", cfg)  # actually 1 oscillation
        assert r.dynamics_penalty > 0


# ---------------------------------------------------------------------------
# Stability penalty (recommend_form integration)
# ---------------------------------------------------------------------------


class TestStabilityPenalty:
    def test_stability_zero_on_unsupported_family(self) -> None:
        # bessel — recommender abstains, no penalty.
        cfg = RegularizerConfig(lambda_stability=1.0)
        r = regularize("besselj(0, x)", cfg)
        assert r.stability_penalty == 0.0


# ---------------------------------------------------------------------------
# Explanation + result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_is_frozen(self) -> None:
        r = regularize("x")
        with pytest.raises(Exception):
            r.total_penalty = 99.0  # type: ignore[misc]

    def test_explanation_includes_total(self) -> None:
        r = regularize("sin(x)")
        assert "total=" in r.explanation

    def test_explanation_mentions_active_components(self) -> None:
        cfg = RegularizerConfig(lambda_chain=1.0, max_chain_order=0)
        r = regularize("sin(x)", cfg)
        assert "chain=" in r.explanation


# ---------------------------------------------------------------------------
# Search-loop integration scenario
# ---------------------------------------------------------------------------


class TestSearchScenarios:
    def test_simpler_form_wins(self) -> None:
        """Given two semantically related candidates, the regularizer
        should prefer the lower chain-order one."""
        cfg = RegularizerConfig(
            lambda_chain=1.0,
            lambda_nodes=0.05,
            max_chain_order=1,
        )
        # A candidate with simpler structure.
        easy = regularize("x**2 + 1", cfg)
        # A candidate that does the same thing with more transcendentals.
        hard = regularize("exp(2*log(x)) + cos(0)", cfg)
        assert easy.total_penalty <= hard.total_penalty

    def test_combined_penalty_is_sum(self) -> None:
        cfg = RegularizerConfig(
            lambda_chain=2.0,
            lambda_nodes=0.5,
            max_chain_order=0,
        )
        r = regularize("sin(x) + cos(x)", cfg)
        assert r.total_penalty == pytest.approx(
            r.chain_penalty + r.node_penalty
            + r.dynamics_penalty + r.stability_penalty
        )
