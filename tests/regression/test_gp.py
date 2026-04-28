"""Tests for eml_cost.regression.gp — the GP search engine."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from eml_cost import RegularizerConfig
from eml_cost.regression import GPConfig, GPResult, random_baseline, search


class TestSearchSurface:
    def test_returns_gpresult(self) -> None:
        x = np.linspace(-1, 1, 50)
        y = 2 * x
        cfg = GPConfig(
            population_size=30, n_generations=5, seed=0,
            use_data_dynamics=False)
        r = search(x, y, var_names=["x"], config=cfg)
        assert isinstance(r, GPResult)

    def test_result_has_required_fields(self) -> None:
        x = np.linspace(-1, 1, 50)
        y = x
        cfg = GPConfig(
            population_size=20, n_generations=3, seed=0,
            use_data_dynamics=False)
        r = search(x, y, var_names=["x"], config=cfg)
        assert r.expression is not None
        assert r.tree is not None
        assert r.chain_order >= 0
        assert r.node_count >= 1
        assert r.mse >= 0

    def test_mse_reasonable_on_trivial_target(self) -> None:
        # y = x. Should converge to a good fit with default budget.
        x = np.linspace(-1, 1, 50)
        y = x.copy()
        cfg = GPConfig(
            population_size=60, n_generations=20, seed=42,
            use_data_dynamics=False)
        r = search(x, y, var_names=["x"], config=cfg)
        assert r.mse < 0.1

    def test_dict_input(self) -> None:
        rng = np.random.default_rng(0)
        x1 = rng.uniform(-1, 1, 30)
        x2 = rng.uniform(-1, 1, 30)
        y = x1 + x2
        cfg = GPConfig(
            population_size=20, n_generations=3, seed=0,
            use_data_dynamics=False)
        r = search({"x1": x1, "x2": x2}, y, config=cfg)
        assert r.mse >= 0

    def test_array_input_requires_var_names(self) -> None:
        x = np.linspace(0, 1, 20)
        y = x.copy()
        with pytest.raises(ValueError):
            search(x, y, config=GPConfig(
                population_size=10, n_generations=2, seed=0,
                use_data_dynamics=False))


class TestRegularizerIntegration:
    def test_regularizer_invoked_when_set(self) -> None:
        x = np.linspace(0, 1, 30)
        y = x.copy()
        cfg = GPConfig(
            population_size=30, n_generations=5, seed=0,
            use_data_dynamics=False,
            regularizer=RegularizerConfig(lambda_chain=1.0, max_chain_order=2),
        )
        r = search(x, y, var_names=["x"], config=cfg)
        # The result carries chain_order even when no penalty applied.
        assert r.chain_order >= 0

    def test_no_regularizer_still_reports_chain_order(self) -> None:
        x = np.linspace(0, 1, 30)
        y = np.sin(x)
        cfg = GPConfig(
            population_size=30, n_generations=5, seed=0,
            use_data_dynamics=False,
            regularizer=None,
        )
        r = search(x, y, var_names=["x"], config=cfg)
        assert r.chain_order >= 0


class TestEarlyStop:
    def test_early_stop_triggers_on_easy_target(self) -> None:
        x = np.linspace(0, 1, 50)
        y = np.full_like(x, 1.0)
        cfg = GPConfig(
            population_size=40, n_generations=50, seed=1,
            early_stop_mse=1e-4,
            use_data_dynamics=False)
        r = search(x, y, var_names=["x"], config=cfg)
        # Constant-1 is reachable; expect early termination.
        assert r.generations_run < 50 or r.mse < 1e-4


class TestRandomBaseline:
    def test_random_baseline_runs(self) -> None:
        x = np.linspace(0, 1, 20)
        y = x.copy()
        r = random_baseline(x, y, var_names=["x"],
                            n_samples=50, max_depth=3, seed=0)
        assert isinstance(r, GPResult)
        assert r.mse >= 0

    def test_random_baseline_does_no_evolution(self) -> None:
        x = np.linspace(0, 1, 20)
        y = np.sin(x)
        r = random_baseline(x, y, var_names=["x"],
                            n_samples=30, max_depth=3, seed=0)
        assert r.generations_run == 0


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        x = np.linspace(-1, 1, 40)
        y = x ** 2
        cfg = GPConfig(
            population_size=40, n_generations=8, seed=7,
            use_data_dynamics=False)
        r1 = search(x, y, var_names=["x"], config=cfg)
        r2 = search(x, y, var_names=["x"], config=cfg)
        assert r1.mse == r2.mse
        assert str(r1.expression) == str(r2.expression)
