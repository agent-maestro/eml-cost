"""Tests for ``eml_rewrite.notebook`` (added in 0.2.1).

The IPython integration imports lazily; these tests exercise the
internal scanner + report formatter directly so they don't require
IPython to be installed."""
from __future__ import annotations

from eml_cost.rewrite.notebook import _format_report, _scan_cell


def test_scan_cell_finds_pythagorean_pattern() -> None:
    src = (
        "from sympy import Symbol, sin, cos\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "sin(x)**2 + cos(x)**2\n"
    )
    found = _scan_cell(src)
    assert len(found) == 1
    line_no, snippet, expr, top = found[0]
    assert line_no == 3
    assert top.pattern_name == "pythagorean"


def test_scan_cell_finds_multiple_patterns() -> None:
    src = (
        "from sympy import Symbol, exp, sinh, cosh\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "exp(x) / (1 + exp(x))\n"
        "sinh(x) / cosh(x)\n"
    )
    found = _scan_cell(src)
    pattern_names = {top.pattern_name for _, _, _, top in found}
    assert "sigmoid" in pattern_names
    assert "tanh_from_sinh_cosh" in pattern_names


def test_scan_cell_returns_empty_when_no_rewrites() -> None:
    src = (
        "from sympy import Symbol\n"
        "x = Symbol('x')\n"
        "x + 1\n"
    )
    found = _scan_cell(src)
    assert found == []


def test_scan_cell_handles_syntax_error_gracefully() -> None:
    src = "this is not valid python ::: !!!"
    assert _scan_cell(src) == []


def test_format_report_for_empty_findings() -> None:
    text = _format_report([])
    assert "no improving rewrites" in text


def test_format_report_includes_pattern_and_cost() -> None:
    src = (
        "from sympy import Symbol, sin, cos\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "sin(x)**2 + cos(x)**2\n"
    )
    found = _scan_cell(src)
    text = _format_report(found)
    assert "pythagorean" in text
    assert "cost units" in text
    assert "line 3" in text


def test_format_report_uses_singular_for_one_rewrite() -> None:
    src = (
        "from sympy import Symbol, sin, cos\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "sin(x)**2 + cos(x)**2\n"
    )
    text = _format_report(_scan_cell(src))
    # "1 improving rewrite found" — no plural s.
    assert "1 improving rewrite found" in text
    assert "1 improving rewrites found" not in text


def test_load_ipython_extension_callable_when_module_imported() -> None:
    """The exported entry point must exist; we don't actually invoke
    it (would need a real IPython instance) but importability is the
    key promise of this module."""
    from eml_cost.rewrite import notebook
    assert callable(notebook.load_ipython_extension)
    assert callable(notebook.unload_ipython_extension)
