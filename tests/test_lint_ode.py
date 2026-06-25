"""Tests for eml_cost.lint_ode — the EML-readiness linter for ODEs."""
from __future__ import annotations

import sympy as sp
import pytest

from eml_cost import OdeFinding, lint_ode, lint_odes, format_ode_findings

x = sp.Symbol("x", real=True)


def test_finite_is_info_clear():
    fs = lint_ode(0, -1)  # y'' - y = 0
    assert [f.code for f in fs] == ["EML-FINITE"]
    assert fs[0].severity == "info"
    assert "safe for EML kernel synthesis" in fs[0].message


def test_oscillatory_is_warn_not_representable():
    fs = lint_ode(0, 1)  # y'' + y = 0
    assert [f.code for f in fs] == ["EML-INF-OSC"]
    assert fs[0].severity == "warn"
    assert "bounded interval" in fs[0].message


def test_airy_flags_oscillatory_and_non_liouvillian():
    fs = lint_ode(0, -x, "R")
    assert sorted(f.code for f in fs) == ["EML-INF-OSC", "NON-LIOUVILLIAN"]
    assert all(f.severity == "warn" for f in fs)
    assert any("SL2" in f.message for f in fs)


def test_modified_bessel_is_candidate():
    fs = lint_ode(1 / x, -(1 + 4 / x**2), "pos")  # I_2
    assert [f.code for f in fs] == ["EML-CANDIDATE"]
    assert fs[0].severity == "warn"


def test_name_is_echoed():
    fs = lint_ode(0, 1, name="harmonic")
    assert fs[0].name == "harmonic"
    assert fs[0].message.startswith("harmonic:")


def test_finding_is_frozen():
    f = lint_ode(0, 1)[0]
    assert isinstance(f, OdeFinding)
    with pytest.raises(Exception):
        f.code = "x"


def test_lint_odes_batch_and_format():
    cat = {
        "harmonic": (0, 1),
        "hyperbolic": (0, -1),
        "Airy": (0, -x, "R"),
        "Euler-real": (1 / x, -1 / x**2, "pos"),
    }
    fs = lint_odes(cat)
    codes = {f.code for f in fs}
    assert {"EML-FINITE", "EML-INF-OSC", "NON-LIOUVILLIAN"} <= codes
    out = format_ode_findings(fs)
    assert "INFO" in out and "WARN" in out
    # every finding renders on its own line
    assert out.count("\n") == len(fs) - 1
