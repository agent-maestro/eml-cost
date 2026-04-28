"""Tests for eml_cost.regression.benchmark — Feynman SR-bench harness."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from eml_cost.regression import benchmark


class TestProblemRegistry:
    def test_ten_problems(self) -> None:
        assert len(benchmark.FEYNMAN_PROBLEMS) == 10

    def test_chain_order_coverage(self) -> None:
        orders = {p.true_chain_order for p in benchmark.FEYNMAN_PROBLEMS}
        # Want a mix from chain 0 through chain 3.
        assert 0 in orders
        assert 1 in orders
        assert 2 in orders
        assert 3 in orders

    def test_problem_sample_returns_matching_shapes(self) -> None:
        prob = benchmark.FEYNMAN_PROBLEMS[0]
        x, y = prob.sample(seed=0)
        assert x.shape == y.shape


class TestRunBenchmarkTinyConfig:
    def test_runs_end_to_end(self) -> None:
        # Smoke: 2 problems × 1 seed × tiny GP budget.
        problems = benchmark.FEYNMAN_PROBLEMS[:2]
        table = benchmark.run_benchmark(
            problems=problems,
            seeds=(0,),
            population_size=20,
            n_generations=3,
        )
        assert len(table.rows) == 2
        # 2 problems × 1 seed × 3 conditions = 6 runs.
        assert len(table.runs) == 6

    def test_format_table_returns_markdown(self) -> None:
        problems = benchmark.FEYNMAN_PROBLEMS[:1]
        table = benchmark.run_benchmark(
            problems=problems,
            seeds=(0,),
            population_size=15,
            n_generations=2,
        )
        out = benchmark.format_table(table)
        assert "Problem" in out
        assert "True chain" in out
        assert "Overall chain-order hit rate" in out

    def test_table_overall_rates_in_unit_interval(self) -> None:
        problems = benchmark.FEYNMAN_PROBLEMS[:2]
        table = benchmark.run_benchmark(
            problems=problems,
            seeds=(0,),
            population_size=15,
            n_generations=2,
        )
        for rate in (table.overall_a_chain_rate,
                     table.overall_b_chain_rate,
                     table.overall_c_chain_rate):
            assert 0.0 <= rate <= 1.0
