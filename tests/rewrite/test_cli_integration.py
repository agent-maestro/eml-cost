"""CLI integration tests added for the 0.1.0 stable release.

Each test exercises a CLI surface end-to-end: argparser construction,
subcommand entry point, output presence on stdout/stderr, and the
new --include-conditional flag.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from eml_cost.rewrite.cli import build_parser, main


# 1. scan on a Python file with a recognizable rewrite pattern.

def test_scan_reports_known_sigmoid_rewrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "model.py"
    # Free-floating expression statement: the scanner picks these up
    # (it walks ast.Expr nodes).
    src.write_text(
        "from sympy import Symbol, exp\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "exp(x) / (1 + exp(x))\n",
        encoding="utf-8",
    )
    rc = main(["scan", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "sigmoid" in out
    assert "1 improving rewrite" in out


# 2. fix subcommand prints the v0.1 dry-run notice and the proposed rewrite.

def test_fix_roundtrip_announces_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "expr.py"
    src.write_text(
        "from sympy import Symbol, sinh, cosh\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "sinh(x) / cosh(x)\n",
        encoding="utf-8",
    )
    rc = main(["fix", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Would rewrite to" in out
    assert "tanh" in out
    assert "v0.1 prints proposed rewrites only" in out


# 3. analyze on a string expression: returns 0, prints the profile and
# any suggestions.

def test_analyze_string_expression_emits_profile_and_suggestions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["analyze", "exp(x) / (1 + exp(x))"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Expression:" in out
    assert "predicted_depth" in out
    # The default scan respects assumptions; symbols here have none, so
    # the sigmoid pattern (which is unconditional) still fires.
    assert "sigmoid" in out


# 4. --include-conditional flag is accepted by both subcommands and
# does not change behaviour for unconditional rewrites.
#
# (Semantic behaviour of the conditional path is covered by
# test_core.py::test_log_pow_conditional_when_unconstrained.)

def test_include_conditional_flag_accepted_by_analyze_and_scan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc1 = main(["analyze", "exp(x) / (1 + exp(x))", "--include-conditional"])
    out1 = capsys.readouterr().out
    assert rc1 == 0
    assert "sigmoid" in out1   # unconditional rewrite still surfaced

    src = tmp_path / "code.py"
    src.write_text(
        "from sympy import Symbol, exp\n"
        "x = Symbol('x', real=True, positive=True)\n"
        "exp(x) / (1 + exp(x))\n",
        encoding="utf-8",
    )
    rc2 = main(["scan", str(src), "--include-conditional"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert "sigmoid" in out2


# 5. --help on the parent parser succeeds without raising and lists
# the three subcommands.

def test_scan_as_patch_emits_unified_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`scan --as-patch FILE` emits a unified diff applying every
    improving rewrite, with no human-readable summary lines that
    would prevent `git apply` from accepting stdout."""
    src = tmp_path / "model.py"
    src.write_text(
        'from sympy import Symbol, exp, sinh, cosh, sin, cos\n'
        'x = Symbol("x", positive=True)\n'
        'exp(x) / (1 + exp(x))\n'
        'sinh(x) / cosh(x)\n'
        'sin(x)**2 + cos(x)**2\n',
        encoding="utf-8",
    )
    rc = main(["scan", "--as-patch", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--- a/" in out
    assert "+++ b/" in out
    assert "@@" in out
    # Each of the three rewrites should appear as a removed line.
    assert "-exp(x) / (1 + exp(x))" in out
    assert "-sinh(x) / cosh(x)" in out
    assert "-sin(x)**2 + cos(x)**2" in out
    # And as added rewritten lines.
    assert "+1/(1 + exp(-x))" in out
    assert "+tanh(x)" in out
    assert "+1" in out
    # Patch mode must NOT emit the human-readable summary.
    assert "improving rewrite(s)" not in out
    assert "Run `eml-rewrite fix" not in out


def test_root_help_lists_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    # argparse exits 0 on --help.
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for sub in ("scan", "fix", "analyze"):
        assert sub in out
