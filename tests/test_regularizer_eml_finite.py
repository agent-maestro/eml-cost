"""Tests for the EML-finiteness penalty (component 5) in the regularizer.

A candidate is EML-infinity (outside the finite EML tree class) when it carries
oscillation modes (sin/cos) and/or Pfaffian-but-not-EML primitives (Bessel,
Airy, ...) — the two obstructions the differential-Galois <-> EML correspondence
identifies. The penalty biases EML symbolic regression toward representable
forms; ``require_eml_finite`` makes it a hard feasibility constraint.
"""
from __future__ import annotations

import pytest

from eml_cost import regularize, RegularizerConfig


def test_oscillation_is_eml_infinity():
    r = regularize("sin(x)", RegularizerConfig(lambda_eml_finite=1.0))
    assert r.is_eml_finite is False
    assert r.eml_finiteness_penalty == pytest.approx(1.0)


def test_decay_is_eml_finite():
    r = regularize("exp(-x)", RegularizerConfig(lambda_eml_finite=1.0))
    assert r.is_eml_finite is True
    assert r.eml_finiteness_penalty == 0.0


def test_polynomial_is_eml_finite():
    r = regularize("x**2 + 3*x", RegularizerConfig(lambda_eml_finite=5.0))
    assert r.is_eml_finite is True
    assert r.eml_finiteness_penalty == 0.0


def test_pfaffian_not_eml_primitive_is_eml_infinity():
    r = regularize("besselj(0, x)", RegularizerConfig(lambda_eml_finite=2.0))
    assert r.is_eml_finite is False
    assert r.eml_finiteness_penalty == pytest.approx(2.0)  # pne contributes 1


def test_penalty_scales_with_weight_and_modes():
    # two distinct oscillation modes -> score 2.
    cfg = RegularizerConfig(lambda_eml_finite=3.0)
    r = regularize("sin(x) + cos(2*x)", cfg)
    assert not r.is_eml_finite
    assert r.eml_finiteness_penalty == pytest.approx(6.0)  # 3.0 * 2 modes


def test_penalty_enters_total():
    cfg = RegularizerConfig(lambda_eml_finite=1.0, lambda_nodes=0.0)
    r = regularize("sin(x)", cfg)
    assert r.total_penalty == pytest.approx(r.eml_finiteness_penalty)


def test_require_eml_finite_hard_filters():
    cfg = RegularizerConfig(require_eml_finite=True)
    assert regularize("sin(x)", cfg).is_feasible is False     # oscillatory -> infeasible
    assert regularize("exp(-x)", cfg).is_feasible is True      # representable
    assert regularize("airyai(x)", cfg).is_feasible is False   # PNE -> infeasible


def test_default_config_off_no_penalty_no_filter():
    # Backwards-compatible: defaults leave the component inert.
    r = regularize("sin(x)")
    assert r.eml_finiteness_penalty == 0.0
    assert r.is_feasible is True          # not hard-filtered by default
    # is_eml_finite is still reported (informational) even at zero weight.
    assert r.is_eml_finite is False


def test_explanation_mentions_component_when_active():
    r = regularize("sin(x)", RegularizerConfig(lambda_eml_finite=1.0))
    assert "eml_finite=" in r.explanation
