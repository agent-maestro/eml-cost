"""Tests for live_profile (eml-cost profile subcommand)."""
from __future__ import annotations

import io
import json

import sympy as sp

from eml_cost.cli import main as cli_main
from eml_cost.live_profile import LiveProfileResult, live_profile


# ──────────────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────────────


def test_live_profile_returns_result_dataclass():
    r = live_profile("exp(x) + log(x)", samples=16, eval_repeats=100)
    assert isinstance(r, LiveProfileResult)
    assert r.expr == "exp(x) + log(x)"
    assert r.free_vars == ["x"]


def test_live_profile_actual_fields_are_positive():
    r = live_profile("x*y + sin(x)", samples=16, eval_repeats=100)
    assert r.actual_lambdify_ms > 0.0
    assert r.actual_eval_ns_per_call > 0.0
    assert r.peak_kib > 0.0


def test_live_profile_relerr_is_finite_for_safe_domain():
    r = live_profile("exp(x) + log(x)", samples=16, eval_repeats=100,
                     sample_lo=0.5, sample_hi=1.5)
    # Float64 evaluating in a safe domain must produce a relerr at
    # most a few ulps even for a chained transcendental expression.
    assert r.relerr_max < 1e-10
    assert r.digits_lost_actual >= 0.0


def test_live_profile_handles_constant_expressions():
    r = live_profile("0", samples=4, eval_repeats=50)
    assert r.free_vars == []
    assert r.relerr_max == 0.0


def test_live_profile_predictions_are_populated():
    r = live_profile("sin(x*y) + cos(x)", samples=16, eval_repeats=100)
    assert r.predicted_lambdify_ms > 0.0
    lo, hi = r.predicted_lambdify_ci95
    assert lo > 0.0 and hi >= lo
    assert r.predicted_relerr > 0.0


def test_live_profile_to_dict_is_json_serializable():
    r = live_profile("x + y", samples=8, eval_repeats=50)
    d = r.to_dict()
    json.dumps(d)  # raises if not serializable
    assert d["expr"] == "x + y"
    assert isinstance(d["predicted_lambdify_ci95"], list)


# ──────────────────────────────────────────────────────────────────
# CLI integration
# ──────────────────────────────────────────────────────────────────


def test_cli_profile_pretty_output():
    buf = io.StringIO()
    rc = cli_main(
        ["profile", "x + sin(x)", "--samples", "8", "--eval-repeats", "100"],
        out=buf,
    )
    assert rc == 0
    text = buf.getvalue()
    assert "x + sin(x)" in text
    assert "WALL-CLOCK" in text
    assert "PRECISION" in text


def test_cli_profile_json_output():
    buf = io.StringIO()
    rc = cli_main(
        ["profile", "x*y", "--samples", "4", "--eval-repeats", "50",
         "--json"],
        out=buf,
    )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "results" in payload
    assert len(payload["results"]) == 1
    assert payload["results"][0]["expr"] == "x*y"


def test_cli_profile_handles_parse_errors():
    buf = io.StringIO()
    rc = cli_main(
        ["profile", "this is not valid sympy", "--samples", "4",
         "--eval-repeats", "20"],
        out=buf,
    )
    # Either a parse-error exit or graceful degradation; we only
    # require the CLI not to crash.
    assert rc in (0, 1, 2)
