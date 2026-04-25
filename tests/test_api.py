"""Public API surface tests."""
from __future__ import annotations

import pytest
import sympy as sp

from eml_cost import (
    AnalyzeResult,
    Corrections,
    analyze,
    eml_depth,
    is_pfaffian_not_eml,
    max_path_r,
    measure,
    pfaffian_r,
    structural_overhead,
)


x = sp.Symbol("x", real=True, positive=True)
y = sp.Symbol("y", real=True)


# ---------------------------------------------------------------------------
# analyze() basic contract
# ---------------------------------------------------------------------------


def test_analyze_returns_analyze_result_for_string() -> None:
    result = analyze("exp(x)")
    assert isinstance(result, AnalyzeResult)
    assert result.pfaffian_r == 1


def test_analyze_returns_analyze_result_for_sympy() -> None:
    result = analyze(sp.exp(x))
    assert isinstance(result, AnalyzeResult)
    assert result.pfaffian_r == 1


def test_analyze_rejects_non_str_non_basic() -> None:
    with pytest.raises(TypeError):
        analyze(42)  # type: ignore[arg-type]


def test_analyze_raises_value_error_on_parse_failure() -> None:
    with pytest.raises(ValueError):
        analyze("@@@invalid syntax@@@")


def test_corrections_is_frozen_dataclass() -> None:
    cor = Corrections(c_osc=1, c_composite=0, delta_fused=0)
    with pytest.raises(Exception):  # frozen dataclass raises on attribute set
        cor.c_osc = 99  # type: ignore[misc]


def test_analyze_result_fields_present() -> None:
    result = analyze("exp(sin(x))")
    assert result.expression is not None
    assert isinstance(result.pfaffian_r, int)
    assert isinstance(result.max_path_r, int)
    assert isinstance(result.eml_depth, int)
    assert isinstance(result.structural_overhead, int)
    assert isinstance(result.corrections, Corrections)
    assert isinstance(result.predicted_depth, int)
    assert isinstance(result.is_pfaffian_not_eml, bool)


def test_predicted_depth_consistency() -> None:
    """predicted_depth == max_path_r + corrections + structural_overhead."""
    result = analyze("exp(sin(exp(x)))")
    expected = (
        result.max_path_r
        + result.corrections.c_osc
        + result.corrections.c_composite
        - result.corrections.delta_fused
        + result.structural_overhead
    )
    assert result.predicted_depth == expected


# ---------------------------------------------------------------------------
# Direct detector entry points
# ---------------------------------------------------------------------------


def test_pfaffian_r_polynomial_is_zero() -> None:
    assert pfaffian_r(x ** 5) == 0
    assert pfaffian_r(x ** 100) == 0
    assert pfaffian_r(sp.Integer(5)) == 0


def test_pfaffian_r_single_exp() -> None:
    assert pfaffian_r(sp.exp(x)) == 1


def test_pfaffian_r_iterated_exp() -> None:
    assert pfaffian_r(sp.exp(sp.exp(sp.exp(x)))) == 3


def test_pfaffian_r_sin_cos_pair() -> None:
    """Khovanskii: sin(g) and cos(g) together count as 2 chain elements."""
    assert pfaffian_r(sp.sin(x)) == 2
    assert pfaffian_r(sp.cos(x)) == 2


def test_max_path_r_independent_product_uses_max() -> None:
    """f(x) * g(y) — max_path_r should be max(r(f), r(g)), not sum."""
    expr = sp.exp(x) * sp.exp(y)
    assert pfaffian_r(expr) == 2  # total counts both
    assert max_path_r(expr) == 1  # path-restricted picks deeper branch


def test_eml_depth_polynomial() -> None:
    assert eml_depth(x ** 10) == 1


def test_eml_depth_sin_via_euler() -> None:
    """sin/cos route through Euler bypass: 3 levels."""
    assert eml_depth(sp.sin(x)) == 3


def test_eml_depth_tan_via_sin_cos() -> None:
    assert eml_depth(sp.tan(x)) == 4


def test_eml_depth_softplus_lead_fusion() -> None:
    """log(1+exp(x)) is LEAd-fused: depth 1, not 2."""
    assert eml_depth(sp.log(1 + sp.exp(x))) == 1


def test_eml_depth_sigmoid_pattern() -> None:
    """1/(1+exp(-x)) is sigmoid-fused: depth 1."""
    assert eml_depth(1 / (1 + sp.exp(-x))) == 1


def test_structural_overhead_pure_transcendental_zero() -> None:
    assert structural_overhead(sp.exp(x)) == 0


def test_structural_overhead_add_increments() -> None:
    so_add = structural_overhead(x + y)
    assert so_add >= 1


def test_is_pfaffian_not_eml_pure_eml() -> None:
    assert is_pfaffian_not_eml(sp.exp(sp.sin(x))) is False


def test_is_pfaffian_not_eml_bessel() -> None:
    assert is_pfaffian_not_eml(sp.besselj(0, x)) is True


def test_is_pfaffian_not_eml_airy() -> None:
    assert is_pfaffian_not_eml(sp.airyai(x)) is True


def test_is_pfaffian_not_eml_lambertw() -> None:
    assert is_pfaffian_not_eml(sp.LambertW(x)) is True


# ---------------------------------------------------------------------------
# measure() drop-in for sp.simplify
# ---------------------------------------------------------------------------


def test_measure_returns_int_for_basic() -> None:
    assert isinstance(measure(sp.exp(x)), int)


def test_measure_lower_is_simpler() -> None:
    """sin^2 + cos^2 should measure higher than 1."""
    full = sp.sin(x) ** 2 + sp.cos(x) ** 2
    assert measure(full) > measure(sp.S.One)


def test_measure_non_basic_returns_sentinel() -> None:
    assert measure("not a sympy thing") == 10**9
    assert measure(None) == 10**9


def test_measure_simplify_picks_shorter_form() -> None:
    """Trig identity should collapse to 1 with measure=measure."""
    expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
    simplified = sp.simplify(expr, measure=measure)
    assert simplified == 1
