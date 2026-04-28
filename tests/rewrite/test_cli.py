"""CLI smoke tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from eml_cost.rewrite.cli import build_parser, main


def test_parser_constructs() -> None:
    p = build_parser()
    assert p is not None


def test_analyze_subcommand_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["analyze", "exp(x) + sin(x)"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pfaffian_r" in out
    assert "predicted_depth" in out


def test_analyze_subcommand_rejects_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["analyze", "@@@"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Error" in err


def test_scan_with_no_matches_reports_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "clean.py"
    src.write_text("x = 1\ny = x + 2\n", encoding="utf-8")
    rc = main(["scan", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No improving rewrites" in out


def test_fix_subcommand_announces_v01_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "x.py"
    src.write_text("x = 1\n", encoding="utf-8")
    rc = main(["fix", str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "v0.1" in out  # version-gated dry-run notice
