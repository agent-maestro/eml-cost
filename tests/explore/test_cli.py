"""Subcommand-level tests for eml-explore.

Each test runs the CLI through main() with capsys-captured stdout
to assert the human-readable text shape AND the --json envelope
shape.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

# Most subcommands depend on `eml_discover` (identify, witness) or
# `eml_rewrite` (corpus walk). Neither is on PyPI yet — skip the
# whole file until they publish. The `analyze` / `class` paths
# don't strictly need them, but isolating those into a separate
# file is out of scope for the CI-green fix.
pytest.importorskip("eml_discover", reason="not yet on PyPI")
pytest.importorskip("eml_rewrite", reason="not yet on PyPI")

from eml_cost.explore import main  # noqa: E402 -- after importorskip


# ---------- analyze ----------


def test_analyze_text_output(capsys):
    rc = main(["analyze", "exp(sin(x))"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "predicted_depth" in out
    assert "fingerprint" in out
    assert "axes" in out


def test_analyze_json_output(capsys):
    rc = main(["--json", "analyze", "exp(sin(x))"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["expr"] == "exp(sin(x))"
    assert "fingerprint" in out
    assert "corrections" in out


# ---------- identify ----------


def test_identify_finds_registry_match(capsys):
    rc = main(["identify", "1/(1+exp(-x))"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sigmoid" in out.lower() or "logistic" in out.lower()


def test_identify_unknown_expression_reports_no_match(capsys):
    rc = main(["identify", "x*y*y*y + 17"])
    assert rc == 0
    out = capsys.readouterr().out
    # Either no match, or a weak axes-only match.
    assert "matches" in out or "no registry match" in out


def test_identify_max_caps_results(capsys):
    rc = main(["--json", "identify", "sin(x)", "--max", "2"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["matches"]) <= 2


# ---------- witness ----------


def test_witness_text_output(capsys):
    rc = main(["witness", "1/(1+exp(-x))"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "witness for" in out
    assert "predicted_depth" in out
    assert "Lean-verified" in out


def test_witness_json_output(capsys):
    rc = main(["--json", "witness", "exp(sin(x))"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["input_expr"] == "exp(sin(x))"
    assert "profile" in out
    # exp(sin(x)) is in the EML class — verified_in_lean True
    # since the universality theorem was user-verified
    # (eml-witness 0.2.0+).
    assert out["verified_in_lean"] is True
    assert out["lean_url"] is not None


def test_witness_no_walk_skips_path(capsys):
    rc = main(["--json", "witness", "exp(x)/(1+exp(x))", "--no-walk"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # canonical_path serialised as a list (JSON has no tuples) but
    # carries no entries when --no-walk is set.
    assert out["canonical_path"] == []


# ---------- class ----------


def test_class_lookup_finds_members(capsys):
    """The cost class p1-d2-w1-c0 has multiple registry members
    (sigmoid, swish, etc.) per the substrate showcase."""
    rc = main(["class", "p1-d2-w1-c0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "p1-d2-w1-c0" in out
    # Either multiple members or "no registry formula" — just don't crash.


def test_class_unknown_axes(capsys):
    rc = main(["class", "p99-d99-w99-c99"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no registry formula" in out


# ---------- corpus ----------


def test_corpus_clusters_a_small_file(capsys):
    src_lines = [
        "sin(x)",
        "cos(y)",        # same axes as sin
        "exp(z)",
        "1 / (1 + exp(-w))",
        "log(t)",
    ]
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(src_lines))
    try:
        rc = main(["corpus", path])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cost classes" in out
        assert "parsed:" in out
    finally:
        os.unlink(path)


def test_corpus_json_envelope(capsys):
    """JSON output must include parsed_count, graph_nodes, top_classes."""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("sin(x)\nexp(y)\n")
        rc = main(["--json", "corpus", path])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["parsed_count"] == 2
        assert out["graph_nodes"] == 2
        assert "top_classes" in out
    finally:
        os.unlink(path)


# ---------- example ----------


def test_example_cross_domain_runs(capsys):
    rc = main(["example", "cross-domain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cross-domain demo" in out
    assert "perpetuity" in out
    assert "Stefan-Boltzmann" in out


def test_example_cross_domain_json(capsys):
    rc = main(["--json", "example", "cross-domain"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["example"] == "cross-domain"
    assert out["distinct_axes"] >= 1
    assert out["collapse_ratio"] >= 1.0


def test_example_witness_walkthrough_runs(capsys):
    rc = main(["example", "witness-walkthrough"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "textbook sigmoid" in out
    assert "Pythagorean" in out


# ---------- error paths ----------


def test_invalid_expression_returns_error_exit(capsys):
    rc = main(["analyze", "this is not valid sympy ((((("])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_unknown_subcommand_fails(capsys):
    with pytest.raises(SystemExit):
        main(["nonexistent-subcommand"])


def test_help_works(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
