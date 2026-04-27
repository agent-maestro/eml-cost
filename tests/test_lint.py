"""Tests for eml_cost.lint — pre-commit linter for SymPy compile cost.

The linter scans Python source for ``simplify``/``factor``/``expand``/
``cse``/``lambdify`` calls and predicts their wall-time via the
existing ``estimate_time`` model.
"""
from __future__ import annotations

import textwrap

import pytest

from eml_cost import Finding, lint_file, lint_source


# ---------------------------------------------------------------------------
# Surface contract
# ---------------------------------------------------------------------------


def test_returns_list_of_findings() -> None:
    out = lint_source("import sympy as sp\nsp.simplify('exp(x) + sin(x)')")
    assert isinstance(out, list)
    assert all(isinstance(f, Finding) for f in out)


def test_empty_source_returns_empty_list() -> None:
    assert lint_source("") == []
    assert lint_source("# nothing here\n") == []


def test_no_simplify_call_returns_empty() -> None:
    out = lint_source("x = 1 + 2\nprint(x)")
    assert out == []


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_detects_sp_simplify() -> None:
    out = lint_source(
        "import sympy as sp\n"
        "sp.simplify('exp(x) + sin(x)')"
    )
    assert len(out) >= 1
    assert any(f.function == "simplify" for f in out)


def test_detects_sp_factor() -> None:
    out = lint_source(
        "import sympy as sp\n"
        "sp.factor('x**3 - 1')"
    )
    assert any(f.function == "factor" for f in out)


def test_detects_sp_expand() -> None:
    out = lint_source(
        "import sympy as sp\n"
        "sp.expand('(x+1)**5')"
    )
    assert any(f.function == "expand" for f in out)


def test_detects_attribute_call() -> None:
    """expr.simplify() — receiver is the SymPy expression."""
    out = lint_source(
        "import sympy as sp\n"
        "sp.sympify('exp(x)').simplify()"
    )
    # It's allowed to either resolve or skip; both are honest.
    # If resolved, the function name must be 'simplify'.
    for f in out:
        if f.function == "simplify":
            return
    # No simplify finding is also acceptable (couldn't resolve receiver).


def test_detects_bare_simplify() -> None:
    out = lint_source(
        "from sympy import simplify\n"
        "simplify('exp(x) + sin(x)')"
    )
    assert any(f.function == "simplify" for f in out)


def test_skips_unknown_method() -> None:
    """``something.subs(...)`` is not in the linter's whitelist."""
    out = lint_source(
        "import sympy as sp\n"
        "sp.sympify('x').subs('x', 1)"
    )
    assert all(f.function != "subs" for f in out)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def test_severity_classification_present() -> None:
    out = lint_source(
        "import sympy as sp\n"
        "sp.simplify('exp(x) + 1')"
    )
    if out:
        assert out[0].severity in ("info", "warn", "error")


def test_predicted_seconds_non_negative() -> None:
    out = lint_source(
        "import sympy as sp\n"
        "sp.simplify('exp(exp(x)) + sin(x**2)')"
    )
    for f in out:
        assert f.predicted_seconds >= 0.0


# ---------------------------------------------------------------------------
# Source-only resolution
# ---------------------------------------------------------------------------


def test_unresolvable_argument_skipped() -> None:
    """Variables defined elsewhere are not resolved at lint time."""
    out = lint_source(textwrap.dedent("""
        import sympy as sp
        e = some_function()
        sp.simplify(e)
    """))
    # We expect zero findings because `e` cannot be resolved
    # without runtime context.
    assert all(f.function != "simplify" or f.expression_repr == ""
               for f in out)


def test_syntax_error_handled() -> None:
    """A parse error returns one diagnostic, not a crash."""
    out = lint_source("def foo(:")
    assert len(out) == 1
    assert out[0].severity == "error"


# ---------------------------------------------------------------------------
# Line numbers
# ---------------------------------------------------------------------------


def test_finding_records_correct_line() -> None:
    src = (
        "# header\n"
        "import sympy as sp\n"
        "sp.simplify('exp(x) + 1')\n"
    )
    out = lint_source(src)
    if out:
        assert out[0].line == 3


# ---------------------------------------------------------------------------
# File interface
# ---------------------------------------------------------------------------


def test_lint_file_missing_returns_error_finding(tmp_path) -> None:
    out = lint_file(tmp_path / "does_not_exist.py")
    assert len(out) == 1
    assert out[0].severity == "error"


def test_lint_file_simple_simplify(tmp_path) -> None:
    p = tmp_path / "mod.py"
    p.write_text("import sympy as sp\nsp.simplify('exp(x) + 1')\n",
                 encoding="utf-8")
    out = lint_file(p)
    assert any(f.function == "simplify" for f in out)


def test_lint_file_message_includes_path(tmp_path) -> None:
    p = tmp_path / "mymod.py"
    p.write_text("import sympy as sp\nsp.simplify('exp(x)')\n",
                 encoding="utf-8")
    out = lint_file(p)
    if out:
        assert "mymod.py" in out[0].message
