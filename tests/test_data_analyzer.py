"""Tests for eml_cost.data_analyzer — FFT + Hilbert dynamics estimator.

Skipped at collection time when numpy/scipy are unavailable so the
core package stays importable on minimal installs. Ships in 0.16.0.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from eml_cost import DataDynamics, estimate_dynamics


# ---------------------------------------------------------------------------
# Static / degenerate inputs
# ---------------------------------------------------------------------------


class TestStaticInputs:
    def test_constant_signal(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.full_like(x, 3.14)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0
        assert d.n_decays == 0
        assert d.n_static == 1

    def test_short_signal_returns_static(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        d = estimate_dynamics(x, y)
        assert d.n_static == 1
        assert d.confidence == 0.0

    def test_nonfinite_input_marks_unbounded(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.zeros_like(x)
        y[100] = np.inf
        d = estimate_dynamics(x, y)
        assert d.is_bounded is False
        assert d.n_static == 1


# ---------------------------------------------------------------------------
# Pure oscillation
# ---------------------------------------------------------------------------


class TestOscillations:
    def test_single_sine_one_mode(self) -> None:
        x = np.linspace(0, 4 * np.pi, 512)
        y = np.sin(2 * x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 1
        assert d.is_periodic is True
        # Detected angular frequency should be near 2.
        assert d.frequencies
        assert abs(d.frequencies[0] - 2.0) < 0.3

    def test_two_distinct_modes(self) -> None:
        x = np.linspace(0, 8 * np.pi, 1024)
        y = np.sin(1.5 * x) + 0.7 * np.cos(5.0 * x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations >= 2

    def test_dc_alone_not_counted_as_oscillation(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.full_like(x, 5.0)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0


# ---------------------------------------------------------------------------
# Pure decay
# ---------------------------------------------------------------------------


class TestDecays:
    def test_single_exponential_decay(self) -> None:
        x = np.linspace(0, 10, 256)
        y = 2.0 * np.exp(-0.5 * x)
        d = estimate_dynamics(x, y)
        assert d.n_decays >= 1
        # Detected rate should be close to 0.5.
        assert d.decay_rates
        assert abs(d.decay_rates[0] - 0.5) < 0.2

    def test_growing_signal_not_decay(self) -> None:
        x = np.linspace(0, 5, 256)
        y = np.exp(0.3 * x)
        d = estimate_dynamics(x, y)
        # A pure exponential rise has its envelope peak at the END of
        # the sample, so the decay detector should reject it.
        assert d.n_decays == 0


# ---------------------------------------------------------------------------
# Mixed signature
# ---------------------------------------------------------------------------


class TestMixedDynamics:
    def test_damped_oscillator(self) -> None:
        x = np.linspace(0, 10, 512)
        y = np.sin(3 * x) * np.exp(-0.2 * x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations >= 1
        assert d.n_decays >= 1
        # Signature suitable as expected_dynamics for the regularizer.
        assert d.estimated_chain_order >= 3


# ---------------------------------------------------------------------------
# Phase 2.5 — monotone / single-peak nonlinear inputs
# ---------------------------------------------------------------------------
#
# These are the v0.17.1 D-condition failures: monotone-nonlinear
# shapes whose FFT (after linear detrending) read curvature as
# low-frequency oscillation, inflating estimated_chain_order.
# 0.18.0 added an extrema-count gate before the FFT path.


class TestMonotoneNonlinear:
    """Inputs whose true chain order is 0 or 1 but which v0.17.1
    over-counted as chain >=2 due to spurious FFT modes."""

    def test_quadratic_is_static(self) -> None:
        x = np.linspace(-2.0, 2.0, 256)
        y = x ** 2
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0
        assert d.estimated_chain_order == 0

    def test_cubic_monotone_is_static(self) -> None:
        x = np.linspace(0.0, 2.0, 256)
        y = x ** 3
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0
        assert d.estimated_chain_order == 0

    def test_exp_decay_no_spurious_oscillation(self) -> None:
        x = np.linspace(0, 10, 256)
        y = 2.0 * np.exp(-0.5 * x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0
        # Decay path should still fire.
        assert d.n_decays >= 1

    def test_log_growth_no_spurious_oscillation(self) -> None:
        x = np.linspace(0.1, 10.0, 256)
        y = np.log(x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0

    def test_pendulum_sqrt_is_monotone(self) -> None:
        x = np.linspace(0.0, 10.0, 256)
        y = 2.0 * np.pi * np.sqrt(x)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0

    def test_gaussian_single_peak_no_oscillation(self) -> None:
        x = np.linspace(-3.0, 3.0, 256)
        y = np.exp(-(x ** 2) / 2.0)
        d = estimate_dynamics(x, y)
        assert d.n_oscillations == 0

    def test_count_extrema_helper(self) -> None:
        from eml_cost.data_analyzer import _count_extrema
        # Strict monotone: 0 extrema.
        assert _count_extrema(np.linspace(0.0, 1.0, 64)) == 0
        # Single-peak: 1 extremum.
        x = np.linspace(-1.0, 1.0, 64)
        assert _count_extrema(np.exp(-(x ** 2))) == 1
        # Sinusoid over multiple periods: many extrema.
        x = np.linspace(0.0, 8.0 * np.pi, 256)
        assert _count_extrema(np.sin(x)) >= 4
        # Constant: 0 extrema.
        assert _count_extrema(np.full(64, 3.0)) == 0


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_is_frozen(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.sin(x)
        d = estimate_dynamics(x, y)
        with pytest.raises(Exception):
            d.n_oscillations = 99  # type: ignore[misc]

    def test_estimated_chain_order_formula(self) -> None:
        x = np.linspace(0, 10, 512)
        y = np.sin(3 * x) * np.exp(-0.2 * x)
        d = estimate_dynamics(x, y)
        assert d.estimated_chain_order == (
            2 * d.n_oscillations + d.n_decays)

    def test_confidence_in_unit_interval(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.sin(x)
        d = estimate_dynamics(x, y)
        assert 0.0 <= d.confidence <= 1.0

    def test_returns_dataclass(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.sin(x)
        d = estimate_dynamics(x, y)
        assert isinstance(d, DataDynamics)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_mismatched_shapes_raise(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.linspace(0, 10, 200)
        with pytest.raises(ValueError):
            estimate_dynamics(x, y)

    def test_2d_input_raises(self) -> None:
        x = np.zeros((10, 10))
        y = np.zeros((10, 10))
        with pytest.raises(ValueError):
            estimate_dynamics(x, y)
