"""Tests for the eml_cost.cli command-line interface (0.7.1+)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from eml_cost.cli import (
    EXIT_OK,
    EXIT_PARSE_ERROR,
    EXIT_THRESHOLD_VIOLATION,
    EXIT_USAGE,
    main,
)


# ---------------------------------------------------------------------------
# `version`
# ---------------------------------------------------------------------------


def test_version_prints_version_and_returns_zero() -> None:
    out = io.StringIO()
    rc = main(["version"], out=out)
    assert rc == EXIT_OK
    txt = out.getvalue()
    assert "eml-cost" in txt
    assert "trilogy" in txt
    assert "estimate_time" in txt
    assert "predict_precision_loss" in txt


# ---------------------------------------------------------------------------
# `report`
# ---------------------------------------------------------------------------


def test_report_pretty_for_one_expression() -> None:
    out = io.StringIO()
    rc = main(["report", "exp(x)"], out=out)
    assert rc == EXIT_OK
    txt = out.getvalue()
    assert "exp(x)" in txt
    assert "estimate_time" in txt
    assert "predict_precision_loss" in txt
    assert "simplify" in txt
    assert "factor" in txt
    assert "cse" in txt
    assert "lambdify" in txt


def test_report_pretty_for_multiple_expressions() -> None:
    out = io.StringIO()
    rc = main(["report", "x", "sin(x)", "exp(exp(x))"], out=out)
    assert rc == EXIT_OK
    txt = out.getvalue()
    # All three expression headers should appear.
    assert "=== x ===" in txt
    assert "=== sin(x) ===" in txt
    assert "=== exp(exp(x)) ===" in txt


def test_report_json_is_valid_json_with_expected_shape() -> None:
    out = io.StringIO()
    rc = main(["report", "--json", "exp(x)", "sin(x**2)"], out=out)
    assert rc == EXIT_OK
    payload = json.loads(out.getvalue())
    assert "version" in payload
    assert "reports" in payload
    assert len(payload["reports"]) == 2
    r0 = payload["reports"][0]
    assert r0["expr"] == "exp(x)"
    assert "pfaffian_r" in r0
    assert "eml_depth" in r0
    assert set(r0["estimate_time_ms"]) == {"simplify", "factor", "cse", "lambdify"}
    assert "predicted_max_relerr" in r0["predict_precision_loss"]
    assert "predicted_digits_lost" in r0["predict_precision_loss"]


def test_report_no_expressions_returns_usage_error() -> None:
    out = io.StringIO()
    rc = main(["report"], out=out)
    assert rc == EXIT_USAGE


def test_report_invalid_expression_pretty_continues_with_parse_error() -> None:
    """Pretty mode reports the parse error and exits EXIT_PARSE_ERROR."""
    out = io.StringIO()
    rc = main(["report", "exp(x)", "not a valid expression!! ??"], out=out)
    assert rc == EXIT_PARSE_ERROR
    txt = out.getvalue()
    # The valid one should still get a full report.
    assert "=== exp(x) ===" in txt
    # The invalid one should be flagged.
    assert "PARSE ERROR" in txt


def test_report_invalid_expression_json_includes_parse_errors_array() -> None:
    """JSON mode collects parse errors instead of erroring out."""
    out = io.StringIO()
    rc = main(["report", "--json", "exp(x)", "not valid !!"], out=out)
    # JSON mode keeps EXIT_OK so that machine consumers get clean JSON
    # plus the parse_errors array; check both.
    assert rc == EXIT_OK
    payload = json.loads(out.getvalue())
    assert len(payload["reports"]) == 1
    assert len(payload["parse_errors"]) == 1
    assert "expr" in payload["parse_errors"][0]
    assert "error" in payload["parse_errors"][0]


def test_report_from_file(tmp_path: Path) -> None:
    f = tmp_path / "exprs.txt"
    f.write_text(
        "# this is a comment\n"
        "exp(x)\n"
        "\n"
        "sin(x**2)\n",
        encoding="utf-8",
    )
    out = io.StringIO()
    rc = main(["report", "--file", str(f)], out=out)
    assert rc == EXIT_OK
    txt = out.getvalue()
    assert "=== exp(x) ===" in txt
    assert "=== sin(x**2) ===" in txt


# ---------------------------------------------------------------------------
# `check`
# ---------------------------------------------------------------------------


def test_check_no_thresholds_passes_everything() -> None:
    out = io.StringIO()
    rc = main(["check", "exp(x)", "sin(x)"], out=out)
    assert rc == EXIT_OK


def test_check_passes_when_under_budget() -> None:
    """Generous budget -> no failures."""
    out = io.StringIO()
    rc = main(
        ["check", "x", "--max-simplify-ms", "100000", "--max-digits-lost", "100"],
        out=out,
    )
    assert rc == EXIT_OK
    assert "OK" in out.getvalue()


def test_check_fails_when_over_simplify_budget(capsys: pytest.CaptureFixture[str]) -> None:
    out = io.StringIO()
    # 1e-9 ms simplify budget will fail virtually any expression.
    rc = main(["check", "exp(exp(x)) + sin(x**2)", "--max-simplify-ms", "1e-9"], out=out)
    assert rc == EXIT_THRESHOLD_VIOLATION
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "simplify predicted" in err


def test_check_fails_when_over_digits_lost_budget(capsys: pytest.CaptureFixture[str]) -> None:
    out = io.StringIO()
    # 0 digits lost budget fails any non-trivial expression.
    rc = main(
        ["check", "sin(sin(sin(sin(x))))", "--max-digits-lost", "0.01"],
        out=out,
    )
    assert rc == EXIT_THRESHOLD_VIOLATION
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "precision loss" in err


def test_check_no_expressions_returns_usage_error() -> None:
    out = io.StringIO()
    rc = main(["check"], out=out)
    assert rc == EXIT_USAGE


def test_check_parse_error_returns_parse_error_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = io.StringIO()
    rc = main(["check", "not valid !!"], out=out)
    assert rc == EXIT_PARSE_ERROR
    err = capsys.readouterr().err
    assert "PARSE ERROR" in err


def test_check_from_file(tmp_path: Path) -> None:
    f = tmp_path / "exprs.txt"
    f.write_text("exp(x)\nsin(x)\n", encoding="utf-8")
    out = io.StringIO()
    rc = main(["check", "--file", str(f)], out=out)
    assert rc == EXIT_OK


# ---------------------------------------------------------------------------
# Console-script entry point smoke
# ---------------------------------------------------------------------------


def test_console_entry_point_is_importable() -> None:
    """The `eml-cost` console-script entry installed by [project.scripts]
    points at eml_cost.cli:_entrypoint. Import must succeed."""
    from eml_cost.cli import _entrypoint  # noqa: F401
    assert callable(_entrypoint)
